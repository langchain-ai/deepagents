"""Tests for cost estimation and graph-side cumulative cost persistence."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, create_autospec, patch
from uuid import uuid4

import pytest
from deepagents.backends import StateBackend
from deepagents.middleware import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import ApproveDecision
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolRuntime  # noqa: TC002  # Runtime tool injection.
from langgraph.types import Command, Overwrite

from deepagents_code import cost_tracking
from deepagents_code._fake_models import _ToolBindingFakeModel
from deepagents_code.cost_tracking import (
    _CONFIGURED_PROVIDER_METADATA_KEY,
    _RECORDER_VAR,
    MODEL_USAGE_EVENT_TYPE,
    SESSION_COST_EVENT_TYPE,
    CostState,
    CostTrackingMiddleware,
    _ModelCallRecord,
    _SessionCostRecorder,
    _set_configured_model_metadata,
    estimate_cost,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator
    from pathlib import Path

    from genai_prices import UpdatePrices
    from genai_prices.types import Provider
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime

KNOWN_MODEL = "claude-sonnet-4-5"
KNOWN_PROVIDER = "anthropic"
THREAD_ID = "thread-under-test"

_OVERRIDE_MODEL = "dcode-override-test-model"
"""Model only the override fixtures catalog; absent from genai-prices itself."""

_OVERRIDE_USER_MODEL = "dcode-override-user-model"
"""Model only the user-file fixture catalogs."""

_OVERRIDE_CONFLICT_MODEL = "dcode-override-conflict-model"
"""Model both override fixtures catalog, at deliberately different rates."""

_OVERRIDE_BUNDLED_MODEL = "dcode-override-bundled-model"
"""Model only the bundled fixture when a user provider is also present."""


@pytest.fixture(autouse=True)
def recorder() -> Iterator[_SessionCostRecorder]:
    """Give each test its own recorder.

    The production recorder is process-wide, and LangChain's configure hook
    attaches whatever the context variable holds, so setting a fresh instance
    isolates both the collecting and the draining side of a test.
    """
    isolated = _SessionCostRecorder()
    token = _RECORDER_VAR.set(isolated)
    try:
        yield isolated
    finally:
        _RECORDER_VAR.reset(token)


def _usage(
    input_tokens: int = 1_000,
    output_tokens: int = 100,
    *,
    cache_read: int = 0,
    cache_write: int = 0,
) -> dict[str, Any]:
    """Build LangChain usage metadata for a completed request."""
    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cache_read or cache_write:
        usage["input_token_details"] = {
            "cache_read": cache_read,
            "cache_creation": cache_write,
        }
    return usage


def _message(
    usage: dict[str, Any] | None,
    *,
    model: str = KNOWN_MODEL,
    provider: str = KNOWN_PROVIDER,
    message_id: str | None = "response-1",
) -> AIMessage:
    """Build an AI message carrying model and usage metadata."""
    return AIMessage(
        content="response",
        id=message_id,
        usage_metadata=usage,  # ty: ignore[invalid-argument-type]
        response_metadata={"model_name": model, "model_provider": provider},
    )


def _runtime(
    *,
    thread_id: str | None = None,
    checkpoint_ns: str = "",
    events: list[dict[str, Any]] | None = None,
) -> Runtime[Any]:
    """Build the runtime shape required by the middleware hooks.

    Args:
        thread_id: Thread the run belongs to, or `None` for an unthreaded run
            whose recorded calls cannot be drained.
        checkpoint_ns: Namespace of the middleware node being executed.
        events: List that collects emitted custom-stream events, when given.
    """
    return cast(
        "Runtime[Any]",
        SimpleNamespace(
            context=None,
            execution_info=SimpleNamespace(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
            ),
            stream_writer=events.append if events is not None else None,
        ),
    )


def _record(
    *,
    message_id: str | None = "response-1",
    model: str = KNOWN_MODEL,
    provider: str = KNOWN_PROVIDER,
    usage: dict[str, Any] | None = None,
    scope: str = "",
) -> _ModelCallRecord:
    """Build a recorded completed model request."""
    return _ModelCallRecord(
        message_id=message_id,
        usage_metadata=usage if usage is not None else _usage(),
        model_name=model,
        provider=provider,
        scope=scope,
    )


def _collect(
    recorder: _SessionCostRecorder,
    record: _ModelCallRecord,
    *,
    thread_id: str = THREAD_ID,
    configured_provider: str = "",
    checkpoint_ns: str = "",
) -> None:
    """Put one already-built record into a recorder's pending queue."""
    run_id = uuid4()
    metadata = {"thread_id": thread_id}
    if configured_provider:
        metadata[_CONFIGURED_PROVIDER_METADATA_KEY] = configured_provider
    if checkpoint_ns:
        metadata["langgraph_checkpoint_ns"] = checkpoint_ns
    recorder.on_chat_model_start(
        {},
        [],
        run_id=run_id,
        metadata=metadata,
    )
    recorder.on_llm_end(
        LLMResult(
            generations=[
                [
                    ChatGeneration(
                        message=_message(
                            dict(record.usage_metadata),
                            model=record.model_name,
                            provider=record.provider,
                            message_id=record.message_id,
                        )
                    )
                ]
            ]
        ),
        run_id=run_id,
    )


def _subagent_command(result: dict[str, Any], runtime: ToolRuntime) -> Command[Any]:
    """Return a subagent result while preserving its parent cost transfers."""
    return Command(
        update={
            "_session_cost_transfers": result.get("_session_cost_transfers", {}),
            "messages": [
                ToolMessage(
                    result["messages"][-1].text,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


class TestEstimateCost:
    """Tests for the shared `genai-prices` adapter."""


def _override_catalog(
    models: list[dict[str, Any]],
    *,
    provider_id: str = "dcode-test",
    model_match: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a one-provider override catalog in upstream's raw schema.

    Args:
        models: Raw model entries the provider carries.
        provider_id: Provider id, so a fixture can build two providers.
        model_match: Provider-level `model_match` claim. Omitted by default
            because inference would otherwise claim every unpriced model for
            this provider; leaving it out narrows most fixtures to the
            provider-id and full-sweep paths.
    """
    provider: dict[str, Any] = {
        "id": provider_id,
        "name": f"{provider_id} overrides",
        "api_pattern": f"https://api\\.{provider_id}\\.invalid",
        "models": models,
    }
    if model_match is not None:
        provider["model_match"] = model_match
    return [provider]


def _override_model(
    model_id: str,
    prices: dict[str, float],
    *,
    match_prefix: str | None = None,
) -> dict[str, Any]:
    """Build one override model entry, exact- or prefix-matched."""
    match = {"starts_with": match_prefix} if match_prefix else {"equals": model_id}
    return {"id": model_id, "match": match, "prices": prices}


class TestPriceOverrides:
    """Local pricing catalogs consulted after a primary `LookupError`."""

    def _install_raw_built_ins(
        self, monkeypatch: pytest.MonkeyPatch, raw: list[dict[str, Any]]
    ) -> None:
        """Point the built-in resource read at an arbitrary raw catalog.

        The loader reads a real package resource, which cannot be swapped
        cleanly per test, so its small read helper is what gets patched. The
        user-source call delegates to the real helper so tests can combine
        fixture built-ins with an on-disk user file.
        """
        real_read = cost_tracking._read_override_source

        def read(path: Path | None) -> tuple[list[Any] | None, bool]:
            return (raw, False) if path is None else real_read(path)

        monkeypatch.setattr(cost_tracking, "_read_override_source", read)

    def _install_built_ins(
        self,
        monkeypatch: pytest.MonkeyPatch,
        models: list[dict[str, Any]],
        *,
        model_match: dict[str, Any] | None = None,
    ) -> None:
        """Point the built-in resource read at a one-provider fixture catalog."""
        self._install_raw_built_ins(
            monkeypatch, _override_catalog(models, model_match=model_match)
        )

    def _install_user_file(self, tmp_path: Path, models: list[dict[str, Any]]) -> None:
        """Drop a user `prices.json` into the isolated config directory."""
        (tmp_path / "prices.json").write_text(
            json.dumps(_override_catalog(models)), encoding="utf-8"
        )

    def test_an_override_stops_being_consulted_once_upstream_covers_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A built-in entry goes inert on its own once upstream ships the model.

        That is what makes the maintenance policy in `bundled_prices.README.md`
        cheap: removal is bookkeeping, not a migration. The interesting case is
        not a statically covered model but a catalog that gains coverage *after*
        an override already priced the model -- which is what an hourly
        auto-update does.
        """
        import genai_prices.data_snapshot
        from genai_prices.types import _providers_from_raw

        self._install_built_ins(
            monkeypatch,
            [
                _override_model(
                    _OVERRIDE_MODEL, {"input_mtok": 100.0, "output_mtok": 100.0}
                )
            ],
        )

        assert estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test") == pytest.approx(
            (1_000 * 100.0 + 100 * 100.0) / 1_000_000
        )

        # Upstream now prices the model, at rates nothing else in this test uses.
        genai_prices.data_snapshot.set_custom_snapshot(
            genai_prices.data_snapshot.DataSnapshot(
                providers=_providers_from_raw(
                    _override_catalog(
                        [
                            _override_model(
                                _OVERRIDE_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0}
                            )
                        ]
                    )
                ),
                from_auto_update=True,
            )
        )
        try:
            after = estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test")
        finally:
            genai_prices.data_snapshot.set_custom_snapshot(None)

        assert after == pytest.approx((1_000 * 1.0 + 100 * 2.0) / 1_000_000)

    def test_a_vanished_upstream_parser_is_reported_at_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The private-name dependency must not fail silently at DEBUG.

        The pin spans every `0.1.x` patch, so `_providers_from_raw` can move on a
        lockfile refresh alone -- and the visible symptom is that the models this
        catalog exists to price revert to no cost at all, so it has to be louder
        than `_override_price`'s handler.
        """
        import genai_prices.types

        # Deleting the attribute is what a rename looks like to a `from ...
        # import` -- and unlike patching `__import__`, it cannot disturb the
        # primary pricing path's own imports.
        monkeypatch.delattr(genai_prices.types, "_providers_from_raw")

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            assert estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test") is None
            assert estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test") is None

        # Count the message, not the name: `exc_info` repeats the name in the
        # traceback it attaches.
        assert caplog.text.count("genai-prices no longer provides") == 1
        assert "genai_prices.types._providers_from_raw" in caplog.text

    def test_the_override_build_is_serialized_and_runs_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Concurrent unpriced requests must not each parse both catalogs.

        `estimate_cost` runs inline on the event loop and from the executor
        threads that price drained records, so this really is concurrent.
        """
        self._install_built_ins(
            monkeypatch,
            [_override_model(_OVERRIDE_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0})],
        )
        patched_read = cost_tracking._read_override_source
        builds = 0
        lock = threading.Lock()

        def counted(path: Path | None) -> tuple[list[Any] | None, bool]:
            if path is None:
                with lock:
                    nonlocal builds
                    builds += 1
            return patched_read(path)

        monkeypatch.setattr(cost_tracking, "_read_override_source", counted)
        barrier = threading.Barrier(4)

        def price() -> float | None:
            barrier.wait()
            return estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test")

        with ThreadPoolExecutor(max_workers=4) as pool:
            costs = [
                future.result() for future in [pool.submit(price) for _ in range(4)]
            ]

        assert builds == 1
        assert all(cost == pytest.approx(costs[0]) for cost in costs)
        assert costs[0] is not None


class TestPriceUpdater:
    """Process-wide background refresh of the genai-prices catalog."""

    @pytest.fixture(autouse=True)
    def _reset_updater_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Isolate the process-wide updater state between tests.

        `monkeypatch.setattr` restores this module's latches after each test.
        genai-prices' own module globals are rolled back too: a leaked handle
        in `_global_update_prices` would make a later `start()` raise, and a
        leaked `_custom_snapshot` would silently re-price every later test off
        fetched data. Note this does *not* stop a daemon thread that a real
        `start()` already spawned -- it only clears the guard and the catalog.
        Nothing here starts a real one; the conftest opt-out keeps the rest of
        the suite from doing so either.

        `DEEPAGENTS_CODE_OFFLINE` is cleared because the suite-wide
        `_skip_managed_tool_downloads` fixture sets it, and it now suppresses
        this updater too -- leaving it set would make every test here pass
        vacuously. The test that covers the offline gate sets it back.
        """
        from deepagents_code._env_vars import OFFLINE

        monkeypatch.delenv(OFFLINE, raising=False)
        monkeypatch.setattr(cost_tracking, "_PRICE_UPDATER_ATTEMPTED", False)
        monkeypatch.setattr(cost_tracking, "_PRICE_UPDATER", None)
        monkeypatch.setattr(cost_tracking, "_TRUNCATED_CATALOG_REPORTED", False)
        # The opt-out gate reads the user's real `config.toml` unless the
        # read is stubbed, so a local `[update].prices_auto_update = false`
        # would silently disable the updater under test.
        monkeypatch.setattr("deepagents_code.config_manifest.load_config_toml", dict)
        import genai_prices.data_snapshot
        import genai_prices.update_prices

        monkeypatch.setattr(genai_prices.update_prices, "_global_update_prices", None)
        monkeypatch.setattr(genai_prices.data_snapshot, "_custom_snapshot", None)

    def _patch_updater(
        self, monkeypatch: pytest.MonkeyPatch, instance: MagicMock
    ) -> MagicMock:
        """Bind `_build_price_updater` to a factory returning *instance*.

        The factory rather than `genai_prices.UpdatePrices` is patched because
        `_build_price_updater` subclasses whatever it is handed, which a
        `MagicMock` cannot stand in for. Callers pass an autospec instance so
        `start(wait=False)` is checked against the real signature.
        """
        factory = MagicMock(return_value=instance)
        monkeypatch.setattr(cost_tracking, "_build_price_updater", factory)
        return factory

    @staticmethod
    def _enable_auto_update(monkeypatch: pytest.MonkeyPatch) -> None:
        """Clear the conftest opt-out so the updater under test may start."""
        from deepagents_code._env_vars import PRICES_AUTO_UPDATE

        monkeypatch.delenv(PRICES_AUTO_UPDATE, raising=False)

    @staticmethod
    def _autospec_updater() -> MagicMock:
        """An `UpdatePrices` stand-in that enforces the real method signatures.

        Returns:
            An autospec instance whose `start` rejects a call the real class
                would reject.
        """
        from genai_prices import UpdatePrices

        return create_autospec(UpdatePrices, instance=True)

    @staticmethod
    def _stub_calc_price(monkeypatch: pytest.MonkeyPatch) -> None:
        """Price every request at $0.01 without touching the catalog."""
        import genai_prices

        monkeypatch.setattr(
            genai_prices,
            "calc_price",
            lambda *_args, **_kwargs: SimpleNamespace(total_price=0.01),
        )

    def test_env_var_overrides_toml_opt_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A truthy env var wins over a persisted TOML opt-out."""
        from deepagents_code._env_vars import PRICES_AUTO_UPDATE

        monkeypatch.setenv(PRICES_AUTO_UPDATE, "1")
        monkeypatch.setattr(
            "deepagents_code.config_manifest.load_config_toml",
            lambda: {"update": {"prices_auto_update": False}},
        )
        factory = self._patch_updater(monkeypatch, self._autospec_updater())
        self._stub_calc_price(monkeypatch)

        assert estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER) is not None

        factory.assert_called_once()


class TestPriceCatalogGuard:
    """`_build_price_updater` gating on what a fetch is allowed to install."""

    @pytest.fixture(autouse=True)
    def _reset_report_latch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cost_tracking, "_TRUNCATED_CATALOG_REPORTED", False)

    @staticmethod
    def _guarded_updater(
        monkeypatch: pytest.MonkeyPatch, fetched: int | None
    ) -> tuple[UpdatePrices, int]:
        """Build the real guarded updater over a fetch returning *fetched* providers.

        `fetched=None` stands in for the `None` snapshot that
        `UpdatePrices.fetch` is typed to allow.

        Returns:
            The guarded instance and the bundled provider count it is judged
                against.
        """
        from genai_prices import UpdatePrices
        from genai_prices.data import providers as bundled

        snapshot = (
            None if fetched is None else SimpleNamespace(providers=[object()] * fetched)
        )
        monkeypatch.setattr(UpdatePrices, "fetch", lambda _self: snapshot)
        return cost_tracking._build_price_updater(UpdatePrices), len(bundled)


class TestCostTrackingMiddleware:
    """Tests for cumulative cost writes on the model checkpoint path."""

    def test_prepared_operation_cost_can_commit_or_rollback(
        self,
        recorder: _SessionCostRecorder,
    ) -> None:
        """Operation pricing is additive and restores records after failure."""
        _collect(
            recorder,
            _record(message_id="offload-summary"),
            checkpoint_ns="dcode_offload:operation-1",
        )
        state = cast(
            "CostState",
            {
                "messages": [],
                "_model_spec": f"{KNOWN_PROVIDER}:{KNOWN_MODEL}",
            },
        )

        prepared = cost_tracking.prepare_operation_cost(state, THREAD_ID)
        one_call = estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)

        assert one_call is not None
        assert prepared.update == {"_session_cost_usd": pytest.approx(one_call)}
        assert recorder.drain(THREAD_ID) == []

        prepared.rollback()
        retried = cost_tracking.prepare_operation_cost(state, THREAD_ID)
        assert retried.update == {"_session_cost_usd": pytest.approx(one_call)}

    def test_committed_prepare_does_not_restore_records(
        self,
        recorder: _SessionCostRecorder,
    ) -> None:
        """A committed prepare keeps its records drained.

        Restoring them would let the next drain price the same spend a second
        time, so `commit` must settle the instance without touching the
        recorder.
        """
        _collect(
            recorder,
            _record(message_id="offload-summary"),
            checkpoint_ns="dcode_offload:operation-1",
        )
        state = cast(
            "CostState",
            {"messages": [], "_model_spec": f"{KNOWN_PROVIDER}:{KNOWN_MODEL}"},
        )

        prepared = cost_tracking.prepare_operation_cost(state, THREAD_ID)
        prepared.commit()
        prepared.rollback()

        assert cost_tracking.prepare_operation_cost(state, THREAD_ID).update == {}

    def test_abandoned_prepare_warns_that_spend_was_lost(
        self,
        recorder: _SessionCostRecorder,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An unsettled prepare must not disappear silently.

        The drain is destructive, so a prepare that is neither committed nor
        rolled back deletes its spend from the thread's lifetime total. That was
        the one settlement outcome with no observable trace.
        """
        _collect(
            recorder,
            _record(message_id="offload-summary"),
            checkpoint_ns="dcode_offload:operation-1",
        )
        state = cast(
            "CostState",
            {"messages": [], "_model_spec": f"{KNOWN_PROVIDER}:{KNOWN_MODEL}"},
        )

        prepared = cost_tracking.prepare_operation_cost(state, THREAD_ID)
        with caplog.at_level(logging.WARNING):
            del prepared
            gc.collect()

        assert "abandoned without commit or rollback" in caplog.text

    def test_unpriceable_record_leaves_the_message_to_state_pricing(
        self, recorder: _SessionCostRecorder
    ) -> None:
        """A record the recorder could not price must not block the fallback."""
        _collect(
            recorder,
            _record(message_id="response-1", model="", provider=""),
        )
        middleware = CostTrackingMiddleware()
        state: CostState = {
            "messages": [
                AIMessage(
                    content="response",
                    id="response-1",
                    usage_metadata=_usage(),  # ty: ignore[invalid-argument-type]
                )
            ],
            "_model_spec": f"{KNOWN_PROVIDER}:{KNOWN_MODEL}",
        }

        result = middleware.after_model(state, _runtime(thread_id=THREAD_ID))

        assert result is not None
        assert result["_session_cost_usd"] == pytest.approx(
            estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        )

    def test_nested_agent_checkpoints_and_transfers_cost(
        self, recorder: _SessionCostRecorder
    ) -> None:
        """Nested spend is durable locally before the parent receives it."""
        _collect(
            recorder,
            _record(message_id="nested-1"),
            checkpoint_ns="tools:a|1|model:a",
        )
        middleware = CostTrackingMiddleware(nested=True)
        state: CostState = {
            "messages": [_message(_usage(), message_id="nested-1")],
        }
        runtime = _runtime(
            thread_id=THREAD_ID,
            checkpoint_ns="tools:a|1|CostTrackingMiddleware.after_model:a",
        )

        update = middleware.after_model(state, runtime)

        assert update is not None
        nested_cost = update["_session_cost_usd"]
        assert nested_cost > 0
        assert recorder.drain(THREAD_ID) == []

        completed_state = cast(
            "CostState",
            {**state, "_session_cost_usd": nested_cost},
        )
        transfer = middleware.after_agent(completed_state, runtime)
        assert transfer is not None

        parent_update = CostTrackingMiddleware().after_agent(
            cast(
                "CostState",
                {
                    "messages": [],
                    "_session_cost_transfers": transfer[
                        "_session_cost_transfers"
                    ].value,
                },
            ),
            _runtime(
                thread_id=THREAD_ID,
                checkpoint_ns="CostTrackingMiddleware.after_agent:root",
            ),
        )
        assert parent_update is not None
        assert parent_update["_session_cost_usd"] == pytest.approx(nested_cost)

    def test_sibling_nested_agents_claim_only_their_own_records(
        self,
        recorder: _SessionCostRecorder,
    ) -> None:
        """Parallel subagents must not drain and then re-price one sibling."""
        _collect(
            recorder,
            _record(message_id="nested-a"),
            checkpoint_ns="tools:a|model:a",
        )
        _collect(
            recorder,
            _record(message_id="nested-b"),
            checkpoint_ns="tools:b|model:b",
        )
        nested = CostTrackingMiddleware(nested=True)

        first = nested.after_model(
            cast(
                "CostState",
                {"messages": [_message(_usage(), message_id="nested-a")]},
            ),
            _runtime(
                thread_id=THREAD_ID,
                checkpoint_ns="tools:a|CostTrackingMiddleware.after_model:a",
            ),
        )
        second = nested.after_model(
            cast(
                "CostState",
                {"messages": [_message(_usage(), message_id="nested-b")]},
            ),
            _runtime(
                thread_id=THREAD_ID,
                checkpoint_ns="tools:b|CostTrackingMiddleware.after_model:b",
            ),
        )

        one_call = estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        assert one_call is not None
        assert first == {"_session_cost_usd": pytest.approx(one_call)}
        assert second == {"_session_cost_usd": pytest.approx(one_call)}
        assert recorder.drain(THREAD_ID) == []

    def test_nested_agent_claims_only_transfers_owned_by_its_graph(self) -> None:
        """A nested parent checkpoints child costs without stealing a cousin's."""
        state = cast(
            "CostState",
            {
                "messages": [],
                "_session_cost_transfers": {
                    "tools:parent|tools:child": {
                        "owner_scope": "tools:parent",
                        "cost_usd": 0.25,
                    },
                    "tools:other|tools:cousin": {
                        "owner_scope": "tools:other",
                        "cost_usd": 0.75,
                    },
                },
            },
        )

        update = CostTrackingMiddleware(nested=True).after_model(
            state,
            _runtime(
                thread_id=THREAD_ID,
                checkpoint_ns="tools:parent|CostTrackingMiddleware.after_model:a",
            ),
        )

        assert update is not None
        assert update["_session_cost_usd"] == pytest.approx(0.25)
        pending = update["_session_cost_transfers"]
        assert isinstance(pending, Overwrite)
        assert pending.value == {
            "tools:other|tools:cousin": {
                "owner_scope": "tools:other",
                "cost_usd": 0.75,
            }
        }

    def test_uses_persisted_model_spec_when_message_metadata_is_absent(self) -> None:
        middleware = CostTrackingMiddleware()
        message = AIMessage(
            content="response",
            usage_metadata=_usage(),  # ty: ignore[invalid-argument-type]
        )
        state: CostState = {
            "messages": [message],
            "_model_spec": f"{KNOWN_PROVIDER}:{KNOWN_MODEL}",
        }
        result = middleware.after_model(state, _runtime())
        assert result is not None
        assert result["_session_cost_usd"] > 0

    @pytest.mark.parametrize(
        ("configured_provider", "expected_delta"),
        [
            pytest.param("azure_openai", 0.42, id="azure"),
            pytest.param("openai_codex", None, id="codex-subscription"),
        ],
    )
    def test_recorded_openai_response_uses_checkpointed_provider(
        self,
        recorder: _SessionCostRecorder,
        monkeypatch: pytest.MonkeyPatch,
        configured_provider: str,
        expected_delta: float | None,
    ) -> None:
        """Generic callback metadata must not replace the configured provider."""
        from deepagents_code import cost_tracking

        priced_providers: list[str] = []

        def price(
            usage_metadata: object,
            model_name: str,
            provider: str = "",
        ) -> float | None:
            assert usage_metadata
            assert model_name == "gpt-5.4"
            priced_providers.append(provider)
            return None if provider == "openai_codex" else 0.42

        monkeypatch.setattr(cost_tracking, "estimate_cost", price)
        _collect(
            recorder,
            _record(model="gpt-5.4", provider="openai"),
        )
        middleware = CostTrackingMiddleware()
        state: CostState = {
            "messages": [_message(_usage(), model="gpt-5.4", provider="openai")],
            "_model_spec": f"{configured_provider}:gpt-5.4",
        }

        result = middleware.after_model(state, _runtime(thread_id=THREAD_ID))

        assert priced_providers
        assert set(priced_providers) == {configured_provider}
        if expected_delta is None:
            assert result is None
        else:
            assert result is not None
            assert result["_session_cost_usd"] == pytest.approx(expected_delta)

    @pytest.mark.parametrize(
        ("configured_provider", "expected_delta"),
        [
            pytest.param("azure_openai", 0.67, id="azure"),
            pytest.param("openai_codex", 0.25, id="codex-subscription"),
        ],
    )
    def test_checkpointed_provider_does_not_replace_side_request_provider(
        self,
        recorder: _SessionCostRecorder,
        monkeypatch: pytest.MonkeyPatch,
        configured_provider: str,
        expected_delta: float,
    ) -> None:
        """Only the main response inherits its provider from `_model_spec`."""
        from deepagents_code import cost_tracking

        pricing_targets: list[tuple[str, str]] = []

        def price(
            usage_metadata: object,
            model_name: str,
            provider: str = "",
        ) -> float | None:
            assert usage_metadata
            pricing_targets.append((model_name, provider))
            if provider == "anthropic":
                return 0.25
            if provider == "azure_openai":
                return 0.42
            return None

        monkeypatch.setattr(cost_tracking, "estimate_cost", price)
        _collect(
            recorder,
            _record(
                message_id="side-1",
                model=KNOWN_MODEL,
                provider=KNOWN_PROVIDER,
            ),
            configured_provider=KNOWN_PROVIDER,
        )
        _collect(
            recorder,
            _record(
                message_id="response-1",
                model="gpt-5.4",
                provider="openai",
            ),
            configured_provider=configured_provider,
        )
        middleware = CostTrackingMiddleware()
        state: CostState = {
            "messages": [_message(_usage(), model="gpt-5.4", provider="openai")],
            "_model_spec": f"{configured_provider}:gpt-5.4",
        }

        result = middleware.after_model(state, _runtime(thread_id=THREAD_ID))

        assert result is not None
        assert result["_session_cost_usd"] == pytest.approx(expected_delta)
        assert (KNOWN_MODEL, KNOWN_PROVIDER) in pricing_targets
        assert ("gpt-5.4", configured_provider) in pricing_targets


class TestSessionCostRecorder:
    """Tests for the callback handler that collects completed requests."""

    def test_nested_request_emits_provisional_usage(
        self, recorder: _SessionCostRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[dict[str, Any]] = []
        monkeypatch.setattr("langgraph.config.get_stream_writer", lambda: events.append)

        _collect(
            recorder,
            _record(message_id="child-1", usage=_usage(cache_read=200)),
            checkpoint_ns="tools:task|model:call",
        )

        assert events == [
            {
                "type": MODEL_USAGE_EVENT_TYPE,
                "version": 1,
                "request_id": "child-1",
                "usage_metadata": _usage(cache_read=200),
                "model_name": KNOWN_MODEL,
                "provider": KNOWN_PROVIDER,
                "thread_id": THREAD_ID,
                "scope": "tools:task",
            }
        ]
        assert [record.message_id for record in recorder.drain(THREAD_ID)] == [
            "child-1"
        ]

    async def test_nested_event_uses_configured_model_when_response_omits_it(
        self, recorder: _SessionCostRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[dict[str, Any]] = []
        monkeypatch.setattr("langgraph.config.get_stream_writer", lambda: events.append)
        configured_model = "explicit-nested-model"
        model = _fake_model(
            AIMessage(
                content="response",
                id="child-1",
                usage_metadata=_usage(),  # ty: ignore[invalid-argument-type]
            )
        )
        _set_configured_model_metadata(model, configured_model, KNOWN_PROVIDER)

        await model.ainvoke(
            "hello",
            config={
                "metadata": {
                    "thread_id": THREAD_ID,
                    "langgraph_checkpoint_ns": "tools:task|model:call",
                }
            },
        )

        assert events[0]["model_name"] == configured_model
        assert events[0]["provider"] == KNOWN_PROVIDER
        # The durable record carries the same identity, so the graph prices the
        # request as the model that served it rather than falling back to the
        # parent graph's checkpointed spec.
        assert recorder.drain(THREAD_ID)[0].model_name == configured_model

    def test_root_or_unidentified_request_does_not_emit(
        self, recorder: _SessionCostRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[dict[str, Any]] = []
        monkeypatch.setattr("langgraph.config.get_stream_writer", lambda: events.append)

        _collect(recorder, _record(message_id="root"))
        _collect(
            recorder,
            _record(message_id=None),
            checkpoint_ns="tools:task|model:call",
        )

        assert events == []
        assert len(recorder.drain(THREAD_ID)) == 2

    def test_usage_writer_failure_keeps_the_durable_record(
        self, recorder: _SessionCostRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(_: object) -> None:
            msg = "stream closed"
            raise RuntimeError(msg)

        monkeypatch.setattr("langgraph.config.get_stream_writer", lambda: fail)

        _collect(
            recorder,
            _record(message_id="child-1"),
            checkpoint_ns="tools:task|model:call",
        )

        assert [record.message_id for record in recorder.drain(THREAD_ID)] == [
            "child-1"
        ]

    def test_thread_comes_from_the_ambient_config_when_metadata_omits_it(
        self, recorder: _SessionCostRecorder
    ) -> None:
        """A side invoke passing its own metadata replaces the ambient copy.

        The Auto classifier and the summarization model both do this, so the
        thread has to be recoverable from the ambient config instead.
        """
        from langchain_core.runnables.config import set_config_context

        run_id = uuid4()
        with set_config_context({"configurable": {"thread_id": THREAD_ID}}) as ctx:
            ctx.run(
                recorder.on_chat_model_start,
                {},
                [],
                run_id=run_id,
                metadata={"lc_source": "auto_mode_classifier"},
            )
        recorder.on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=_message(_usage()))]]),
            run_id=run_id,
        )

        assert len(recorder.drain(THREAD_ID)) == 1


_CompiledAgent = CompiledStateGraph[Any, Any, Any, Any]
"""The agent object `create_agent` compiles, as these tests use it."""


class _QueuedFakeModel(_ToolBindingFakeModel):
    """Fake chat model returning queued responses with usage metadata."""

    queue: Any = None
    disable_streaming: bool = True

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002  # Chat model interface.
        stop: list[str] | None = None,  # noqa: ARG002  # Chat model interface.
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002  # Chat model interface.
        **kwargs: Any,  # noqa: ARG002  # Chat model interface.
    ) -> ChatResult:
        """Return the next queued response.

        Returns:
            A chat result wrapping the queued message.
        """
        return ChatResult(generations=[ChatGeneration(message=next(self.queue))])


def _fake_model(*messages: AIMessage) -> _QueuedFakeModel:
    """Build a fake model that returns the given responses in order."""
    return _QueuedFakeModel(queue=iter(messages))


def _repeating_fake_model(message_id_prefix: str) -> _QueuedFakeModel:
    """Build a fake model with responses to spare, each priced the same."""
    return _QueuedFakeModel(
        queue=(
            _message(_usage(), message_id=f"{message_id_prefix}-{index}")
            for index in range(100)
        )
    )


class _SideInvokeMiddleware(AgentMiddleware):
    """Invoke a model directly around the agent's own call.

    Reproduces how offload/summarization and the Auto classifier spend money:
    a direct `ainvoke` that never reaches `after_model`, with its own `metadata`
    replacing the ambient copy LangGraph populated.
    """

    def __init__(self, model: BaseChatModel, source: str) -> None:
        super().__init__()
        self._model = model
        self._source = source

    @property
    def name(self) -> str:
        """Name instances apart so several can share one middleware stack.

        Returns:
            A per-source middleware name.
        """
        return f"{type(self).__name__}:{self._source}"

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Spend on a side call, then run the agent's own model call.

        Returns:
            The downstream model response.
        """
        await self._model.ainvoke(
            "side call",
            config={
                "run_name": f"dcode_{self._source}",
                "tags": [f"dcode:{self._source}"],
                "metadata": {"lc_source": self._source},
            },
        )
        return await handler(request)


class _AfterModelBarrier(AgentMiddleware):
    """Hold parallel agents after callbacks fire but before cost is drained."""

    def __init__(self, barrier: asyncio.Barrier) -> None:
        super().__init__()
        self._barrier = barrier

    async def aafter_model(
        self,
        state: object,  # noqa: ARG002  # Middleware interface.
        runtime: Runtime[Any],  # noqa: ARG002  # Middleware interface.
    ) -> None:
        """Release both nested cost hooks only after both records exist."""
        await self._barrier.wait()


class _SignalAfterAgent(AgentMiddleware):
    """Signal only after preceding reverse-order completion hooks finish."""

    def __init__(self, event: asyncio.Event) -> None:
        super().__init__()
        self._event = event

    async def aafter_agent(
        self,
        state: object,  # noqa: ARG002  # Middleware interface.
        runtime: Runtime[Any],  # noqa: ARG002  # Middleware interface.
    ) -> None:
        """Release a sibling after this agent has checkpointed its transfer."""
        self._event.set()


class _WaitBeforeAgent(AgentMiddleware):
    """Hold one agent until its sibling has completed."""

    def __init__(self, event: asyncio.Event) -> None:
        super().__init__()
        self._event = event

    async def abefore_agent(
        self,
        state: object,  # noqa: ARG002  # Middleware interface.
        runtime: Runtime[Any],  # noqa: ARG002  # Middleware interface.
    ) -> None:
        """Wait until the completed sibling has persisted its transfer."""
        await self._event.wait()


class TestGraphCostOwnership:
    """Verify the graph alone produces a complete cumulative thread total.

    Every case here runs a real graph with no client attached, so a passing
    assertion on the checkpoint is also the assertion that correctness does not
    depend on the UI consuming anything.
    """

    @staticmethod
    async def _run(
        agent: _CompiledAgent,
        thread_id: str = THREAD_ID,
        messages: list[BaseMessage] | None = None,
    ) -> tuple[float, list[float]]:
        """Run one turn and return the committed total and streamed totals.

        Args:
            agent: Compiled agent to run.
            thread_id: Thread to run on.
            messages: Conversation to send, defaulting to one user message.

        Returns:
            The checkpointed `_session_cost_usd` and every total streamed for
            the client, in order.
        """
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        totals: list[float] = []
        async for chunk in agent.astream(
            {"messages": messages or [HumanMessage("hello")]},
            stream_mode=["messages", "updates", "custom"],
            subgraphs=True,
            config=config,
        ):
            _namespace, mode, data = chunk
            if (
                mode == "custom"
                and isinstance(data, dict)
                and data.get("type") == SESSION_COST_EVENT_TYPE
            ):
                totals.append(data["total"])
        state = await agent.aget_state(config)
        return state.values.get("_session_cost_usd", 0.0), totals

    @staticmethod
    def _one_call_usd() -> float:
        """Return the estimate for a single fake request.

        Returns:
            The per-request cost every case in this class is a multiple of.
        """
        cost_usd = estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        assert cost_usd is not None
        return cost_usd

    def _agent(
        self,
        *,
        model: BaseChatModel,
        tools: list[BaseTool] | None = None,
        middleware: list[AgentMiddleware[Any, Any, Any]] | None = None,
    ) -> _CompiledAgent:
        """Build a checkpointed agent with cost tracking installed.

        Returns:
            The compiled agent.
        """
        stack: list[AgentMiddleware[Any, Any, Any]] = [
            *(middleware or []),
            CostTrackingMiddleware(),
        ]
        return create_agent(
            model=model,
            tools=tools or [],
            middleware=stack,
            checkpointer=InMemorySaver(),
        )

    async def test_streamed_total_matches_the_checkpoint(self) -> None:
        agent = self._agent(model=_fake_model(_message(_usage(), message_id="a")))

        total_usd, totals = await self._run(agent)

        assert totals == [pytest.approx(total_usd)]

    async def test_concurrent_threads_do_not_borrow_each_other_s_spend(
        self,
    ) -> None:
        """The recorder is process-wide, so keying by thread is load-bearing.

        A server process runs many threads against one recorder. If a request
        were ever attributed to the ambient thread rather than its own, one
        user would be billed for another's spend -- and every single-threaded
        case in this class would still pass.
        """
        busy = self._agent(
            model=_fake_model(_message(_usage(), message_id="a")),
            middleware=[
                _SideInvokeMiddleware(
                    _fake_model(_message(_usage(), message_id="busy-side")),
                    "summarization",
                )
            ],
        )
        quiet = self._agent(model=_fake_model(_message(_usage(), message_id="b")))

        (busy_total, _), (quiet_total, _) = await asyncio.gather(
            self._run(busy, thread_id="thread-busy"),
            self._run(quiet, thread_id="thread-quiet"),
        )

        one_call = self._one_call_usd()
        assert busy_total == pytest.approx(2 * one_call)
        assert quiet_total == pytest.approx(one_call)

    async def test_parallel_subagents_are_each_charged_once(self) -> None:
        """Sibling nested graphs claim records from their own checkpoint scope."""
        barrier = asyncio.Barrier(2)

        def child(message_id: str) -> _CompiledAgent:
            middleware: list[AgentMiddleware[Any, Any, Any]] = [
                CostTrackingMiddleware(nested=True),
                _AfterModelBarrier(barrier),
            ]
            return create_agent(
                model=_fake_model(_message(_usage(), message_id=message_id)),
                tools=[],
                middleware=middleware,
            )

        child_a = child("child-a")
        child_b = child("child-b")

        @tool
        async def task_a(query: str, runtime: ToolRuntime) -> Command[Any]:
            """Run the first nested agent."""
            result = await child_a.ainvoke({"messages": [HumanMessage(query)]})
            return _subagent_command(result, runtime)

        @tool
        async def task_b(query: str, runtime: ToolRuntime) -> Command[Any]:
            """Run the second nested agent."""
            result = await child_b.ainvoke({"messages": [HumanMessage(query)]})
            return _subagent_command(result, runtime)

        agent = self._agent(
            model=_fake_model(
                AIMessage(
                    content="",
                    id="parent-1",
                    usage_metadata=_usage(),  # ty: ignore[invalid-argument-type]
                    response_metadata={
                        "model_name": KNOWN_MODEL,
                        "model_provider": KNOWN_PROVIDER,
                    },
                    tool_calls=[
                        {"name": "task_a", "args": {"query": "a"}, "id": "t1"},
                        {"name": "task_b", "args": {"query": "b"}, "id": "t2"},
                    ],
                ),
                _message(_usage(), message_id="parent-2"),
            ),
            tools=[task_a, task_b],
        )

        total_usd, _totals = await self._run(agent)

        # Two parent calls plus one call from each parallel child.
        assert total_usd == pytest.approx(4 * self._one_call_usd())

    async def test_subagent_spend_survives_restart_during_tool_approval(
        self,
    ) -> None:
        """A nested model checkpoint survives loss of the process recorder."""

        @tool
        def write_file(path: str) -> str:
            """Pretend to write a file."""
            return path

        child_middleware: list[AgentMiddleware[Any, Any, Any]] = [
            HumanInTheLoopMiddleware({"write_file": True}),
            CostTrackingMiddleware(nested=True),
        ]
        child = create_agent(
            model=_fake_model(
                AIMessage(
                    content="",
                    id="child-1",
                    usage_metadata=_usage(),  # ty: ignore[invalid-argument-type]
                    response_metadata={
                        "model_name": KNOWN_MODEL,
                        "model_provider": KNOWN_PROVIDER,
                    },
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {"path": "notes.txt"},
                            "id": "write-1",
                        }
                    ],
                ),
                _message(_usage(), message_id="child-2"),
            ),
            tools=[write_file],
            middleware=child_middleware,
        )

        @tool
        async def task(query: str, runtime: ToolRuntime) -> Command[Any]:
            """Run a nested agent."""
            result = await child.ainvoke({"messages": [HumanMessage(query)]})
            return _subagent_command(result, runtime)

        agent = self._agent(
            model=_fake_model(
                AIMessage(
                    content="",
                    id="parent-1",
                    usage_metadata=_usage(),  # ty: ignore[invalid-argument-type]
                    response_metadata={
                        "model_name": KNOWN_MODEL,
                        "model_provider": KNOWN_PROVIDER,
                    },
                    tool_calls=[{"name": "task", "args": {"query": "go"}, "id": "t1"}],
                ),
                _message(_usage(), message_id="parent-2"),
            ),
            tools=[task],
        )
        config: RunnableConfig = {"configurable": {"thread_id": THREAD_ID}}
        interrupts: list[Any] = []
        async for _namespace, mode, data in agent.astream(
            {"messages": [HumanMessage("hello")]},
            stream_mode=["updates"],
            subgraphs=True,
            config=config,
        ):
            if mode == "updates" and isinstance(data, dict):
                interrupts.extend(data.get("__interrupt__") or [])

        interrupts_by_id = {interrupt.id: interrupt for interrupt in interrupts}
        assert len(interrupts_by_id) == 1
        (pending_interrupt,) = interrupts_by_id.values()

        # Replacing the recorder simulates a server process restart while the
        # checkpointer and its nested graph state remain durable.
        token = _RECORDER_VAR.set(_SessionCostRecorder())
        try:
            async for _chunk in agent.astream(
                Command(
                    resume={
                        pending_interrupt.id: {
                            "decisions": [ApproveDecision(type="approve")]
                        }
                    }
                ),
                stream_mode=["updates"],
                subgraphs=True,
                config=config,
            ):
                pass
        finally:
            _RECORDER_VAR.reset(token)

        state = await agent.aget_state(config)
        assert state.values.get("_session_cost_usd", 0.0) == pytest.approx(
            4 * self._one_call_usd()
        )

    async def test_completed_sibling_spend_survives_restart_during_approval(
        self,
    ) -> None:
        """A completed sibling is checkpointed before another one interrupts."""
        sibling_completed = asyncio.Event()

        completed_middleware: list[AgentMiddleware[Any, Any, Any]] = [
            _SignalAfterAgent(sibling_completed),
            CostTrackingMiddleware(nested=True),
        ]
        completed_child = create_agent(
            model=_fake_model(_message(_usage(), message_id="completed-child")),
            tools=[],
            middleware=completed_middleware,
        )

        @tool
        def write_file(path: str) -> str:
            """Pretend to write a file."""
            return path

        interrupted_middleware: list[AgentMiddleware[Any, Any, Any]] = [
            _WaitBeforeAgent(sibling_completed),
            HumanInTheLoopMiddleware({"write_file": True}),
            CostTrackingMiddleware(nested=True),
        ]
        interrupted_child = create_agent(
            model=_fake_model(
                AIMessage(
                    content="",
                    id="interrupted-child-1",
                    usage_metadata=_usage(),  # ty: ignore[invalid-argument-type]
                    response_metadata={
                        "model_name": KNOWN_MODEL,
                        "model_provider": KNOWN_PROVIDER,
                    },
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {"path": "notes.txt"},
                            "id": "write-1",
                        }
                    ],
                ),
                _message(_usage(), message_id="interrupted-child-2"),
            ),
            tools=[write_file],
            middleware=interrupted_middleware,
        )

        subagents = SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[
                {
                    "name": "completed",
                    "description": "Complete before the other agent interrupts.",
                    "runnable": completed_child,
                },
                {
                    "name": "interrupted",
                    "description": "Request approval after the sibling completes.",
                    "runnable": interrupted_child,
                },
            ],
            private_state_keys=frozenset({"_session_cost_usd"}),
        )

        agent = self._agent(
            model=_fake_model(
                AIMessage(
                    content="",
                    id="parent-1",
                    usage_metadata=_usage(),  # ty: ignore[invalid-argument-type]
                    response_metadata={
                        "model_name": KNOWN_MODEL,
                        "model_provider": KNOWN_PROVIDER,
                    },
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "description": "complete",
                                "subagent_type": "completed",
                            },
                            "id": "t1",
                        },
                        {
                            "name": "task",
                            "args": {
                                "description": "interrupt",
                                "subagent_type": "interrupted",
                            },
                            "id": "t2",
                        },
                    ],
                ),
                _message(_usage(), message_id="parent-2"),
            ),
            middleware=[subagents],
        )
        config: RunnableConfig = {"configurable": {"thread_id": THREAD_ID}}
        interrupts: list[Any] = []
        async for _namespace, mode, data in agent.astream(
            {"messages": [HumanMessage("hello")]},
            stream_mode=["updates"],
            subgraphs=True,
            config=config,
        ):
            if mode == "updates" and isinstance(data, dict):
                interrupts.extend(data.get("__interrupt__") or [])

        interrupts_by_id = {interrupt.id: interrupt for interrupt in interrupts}
        assert len(interrupts_by_id) == 1
        (pending_interrupt,) = interrupts_by_id.values()

        paused = await agent.aget_state(config)
        transfers = paused.values.get("_session_cost_transfers")
        assert isinstance(transfers, dict)
        assert len(transfers) == 1
        assert sum(transfer["cost_usd"] for transfer in transfers.values()) == (
            pytest.approx(self._one_call_usd())
        )

        token = _RECORDER_VAR.set(_SessionCostRecorder())
        try:
            async for _chunk in agent.astream(
                Command(
                    resume={
                        pending_interrupt.id: {
                            "decisions": [ApproveDecision(type="approve")]
                        }
                    }
                ),
                stream_mode=["updates"],
                subgraphs=True,
                config=config,
            ):
                pass
        finally:
            _RECORDER_VAR.reset(token)

        state = await agent.aget_state(config)
        assert state.values.get("_session_cost_usd", 0.0) == pytest.approx(
            5 * self._one_call_usd()
        )

    async def test_every_source_in_one_turn_is_charged_once_and_accumulates(
        self,
    ) -> None:
        """Assistant, subagent, offload, and Auto spend add up across turns.

        Also pins the resume property the status bar depends on: a second turn
        adds to the committed total rather than restarting it.
        """
        child = create_agent(
            model=_fake_model(_message(_usage(), message_id="child")),
            tools=[],
            middleware=[CostTrackingMiddleware(nested=True)],
        )

        @tool
        async def task(query: str, runtime: ToolRuntime) -> Command[Any]:
            """Run a nested agent."""
            result = await child.ainvoke({"messages": [HumanMessage(query)]})
            return _subagent_command(result, runtime)

        agent = self._agent(
            model=_fake_model(
                AIMessage(
                    content="",
                    id="parent-1",
                    usage_metadata=_usage(),  # ty: ignore[invalid-argument-type]
                    response_metadata={
                        "model_name": KNOWN_MODEL,
                        "model_provider": KNOWN_PROVIDER,
                    },
                    tool_calls=[{"name": "task", "args": {"query": "go"}, "id": "t1"}],
                ),
                _message(_usage(), message_id="parent-2"),
                _message(_usage(), message_id="parent-3"),
            ),
            tools=[task],
            middleware=[
                _SideInvokeMiddleware(
                    _repeating_fake_model("summary"), "summarization"
                ),
                _SideInvokeMiddleware(
                    _repeating_fake_model("decision"), "auto_mode_classifier"
                ),
            ],
        )

        total_usd, totals = await self._run(agent)

        # Two model steps, each preceded by a summarization and a classifier
        # call, plus the one nested call: 2 + 4 + 1.
        assert total_usd == pytest.approx(7 * self._one_call_usd())
        assert totals[-1] == pytest.approx(total_usd)

        second_total_usd, _totals = await self._run(agent)

        # One more step with its two side calls, on top of the first turn.
        assert second_total_usd == pytest.approx(10 * self._one_call_usd())
