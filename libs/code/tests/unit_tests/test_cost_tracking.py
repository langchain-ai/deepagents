"""Tests for cost estimation and graph-side cumulative cost persistence."""

from __future__ import annotations

import asyncio
import builtins
import gc
import inspect
import json
import logging
import subprocess
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast, get_type_hints
from unittest.mock import MagicMock, create_autospec, patch
from uuid import uuid4

import pytest
from deepagents.backends import StateBackend
from deepagents.middleware import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import ApproveDecision
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    OmitFromInput,
    PrivateStateAttr,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.tools import BaseTool, tool
from langgraph.channels import BinaryOperatorAggregate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
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
    resolve_message_model,
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

    def test_known_model_returns_positive_cost(self) -> None:
        cost_usd = estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        assert cost_usd is not None
        assert cost_usd > 0

    def test_unknown_model_returns_none(self) -> None:
        assert (
            estimate_cost(
                _usage(),
                "definitely-not-a-real-model",
                "unknown-provider",
            )
            is None
        )

    def test_aggregate_only_usage_returns_none(self) -> None:
        assert (
            estimate_cost(
                {"total_tokens": 1_100},
                KNOWN_MODEL,
                KNOWN_PROVIDER,
            )
            is None
        )

    def test_azure_openai_uses_azure_catalog(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import genai_prices

        provider_ids: list[str | None] = []

        def fake_calc_price(
            usage: object,
            model_ref: str,
            *,
            provider_id: str | None = None,
        ) -> SimpleNamespace:
            assert usage is not None
            assert model_ref == "gpt-5.5"
            provider_ids.append(provider_id)
            return SimpleNamespace(total_price=0.42)

        monkeypatch.setattr(genai_prices, "calc_price", fake_calc_price)

        assert estimate_cost(_usage(), "gpt-5.5", "azure_openai") == pytest.approx(0.42)
        assert provider_ids == ["azure"]

    def test_malformed_price_result_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import genai_prices

        monkeypatch.setattr(
            genai_prices,
            "calc_price",
            lambda *_args, **_kwargs: SimpleNamespace(total_price=object()),
        )

        assert estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER) is None

    @pytest.mark.parametrize("bad_total", [float("inf"), float("nan"), -5.0])
    def test_non_finite_or_negative_price_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, bad_total: float
    ) -> None:
        """A price that is not a usable dollar figure must not reach the total."""
        import genai_prices

        monkeypatch.setattr(
            genai_prices,
            "calc_price",
            lambda *_args, **_kwargs: SimpleNamespace(total_price=bad_total),
        )

        assert estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER) is None

    def test_unavailable_pricing_package_is_reported_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A broken install is a different problem from an unpriced model.

        It makes every request unpriceable, so it is logged at WARNING once per
        process and exposed through `pricing_data_available` -- otherwise the
        user is told their models have no published rates.
        """
        monkeypatch.setattr(cost_tracking, "_PRICING_UNAVAILABLE", False)
        real_import = builtins.__import__

        def fail_genai_prices(name: str, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401  # __import__ passthrough
            if name == "genai_prices":
                msg = "no genai_prices"
                raise ImportError(msg)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_genai_prices)

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            assert estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER) is None
            assert estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER) is None

        assert caplog.text.count("Could not load genai-prices") == 1
        assert not cost_tracking.pricing_data_available()

    def test_azure_fallback_overrides_generic_openai_metadata(self) -> None:
        message = _message(_usage(), model="gpt-5.5", provider="openai")

        _, provider = resolve_message_model(
            message,
            fallback_model="gpt-5.5",
            fallback_provider="azure_openai",
        )

        assert provider == "azure_openai"

    def test_codex_subscription_usage_is_not_priced_as_openai_api(self) -> None:
        assert estimate_cost(_usage(), "gpt-5.4", "openai_codex") is None

    def test_cache_read_is_priced_separately(self) -> None:
        uncached = estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        cached = estimate_cost(
            _usage(cache_read=900),
            KNOWN_MODEL,
            KNOWN_PROVIDER,
        )
        assert uncached is not None
        assert cached is not None
        assert cached < uncached

    def test_cache_write_is_priced_separately(self) -> None:
        uncached = estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        cached = estimate_cost(
            _usage(cache_write=900),
            KNOWN_MODEL,
            KNOWN_PROVIDER,
        )
        assert uncached is not None
        assert cached is not None
        assert cached > uncached

    @pytest.mark.parametrize(
        ("cache_read", "cache_write"),
        [
            (1_500, 0),
            (0, 1_500),
            (600, 900),
        ],
    )
    def test_cache_buckets_over_the_input_total_still_price(
        self,
        cache_read: int,
        cache_write: int,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Self-inconsistent provider counts must not drop the whole request.

        `genai-prices` rejects a negative uncached-input count, and that raise
        is swallowed into `None` -- which would silently remove the request
        from the durable total rather than estimate it. Clamping keeps an
        estimate, and the warning keeps the anomaly diagnosable.
        """
        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            cost = estimate_cost(
                _usage(1_000, 100, cache_read=cache_read, cache_write=cache_write),
                KNOWN_MODEL,
                KNOWN_PROVIDER,
            )

        assert cost is not None
        assert cost > 0
        assert "exceed the inclusive input total" in caplog.text

    def test_in_range_cache_buckets_do_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            assert (
                estimate_cost(
                    _usage(1_000, 100, cache_read=600, cache_write=100),
                    KNOWN_MODEL,
                    KNOWN_PROVIDER,
                )
                is not None
            )

        assert "exceed the inclusive input total" not in caplog.text

    def test_cache_write_alias_is_priced_separately(self) -> None:
        uncached = estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        usage = _usage()
        usage["input_token_details"] = {"cache_write": 900}
        cached = estimate_cost(usage, KNOWN_MODEL, KNOWN_PROVIDER)

        assert uncached is not None
        assert cached is not None
        assert cached > uncached

    def test_only_one_hour_cache_writes_use_a_distinct_rate(self) -> None:
        """One-hour writes cost more; five-minute writes price as generic ones.

        The catalog publishes no five-minute cache-write rate, so that bucket
        falls back to the generic one and only the one-hour split changes the
        price. Forwarding the five-minute count is inert but forward-compatible.
        """
        five_minute_usage = _usage()
        five_minute_usage["input_token_details"] = {
            "cache_creation": 0,
            "ephemeral_5m_input_tokens": 900,
        }
        one_hour_usage = _usage()
        one_hour_usage["input_token_details"] = {
            "cache_creation": 0,
            "ephemeral_1h_input_tokens": 900,
        }

        five_minute = estimate_cost(five_minute_usage, KNOWN_MODEL, KNOWN_PROVIDER)
        one_hour = estimate_cost(one_hour_usage, KNOWN_MODEL, KNOWN_PROVIDER)
        generic = estimate_cost(
            _usage(cache_write=900),
            KNOWN_MODEL,
            KNOWN_PROVIDER,
        )

        assert five_minute is not None
        assert one_hour is not None
        assert generic is not None
        assert five_minute == pytest.approx(generic)
        assert one_hour > five_minute

    def test_detailed_cache_writes_over_the_input_total_still_price(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        usage = _usage()
        usage["input_token_details"] = {
            "cache_read": 600,
            "ephemeral_5m_input_tokens": 500,
            "ephemeral_1h_input_tokens": 500,
        }

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            cost = estimate_cost(usage, KNOWN_MODEL, KNOWN_PROVIDER)

        assert cost is not None
        assert cost > 0
        assert "exceed the inclusive input total" in caplog.text
        # Per bucket and with the clamped values, so the log says which bucket
        # lost tokens -- only the one-hour rate carries a premium.
        assert "5m=500->400" in caplog.text
        assert "1h=500->0" in caplog.text

    def test_clamping_starves_the_one_hour_bucket_first(self) -> None:
        """Pin which bucket survives a clamp, because it moves the price.

        Buckets drain in tuple order, so the pricier one-hour writes are zeroed
        before the five-minute ones and the estimate is biased low. Reversing
        that order would silently raise every clamped estimate.
        """
        over_total = _usage()
        over_total["input_token_details"] = {
            "cache_read": 600,
            "ephemeral_5m_input_tokens": 500,
            "ephemeral_1h_input_tokens": 500,
        }
        surviving_five_minute = _usage()
        surviving_five_minute["input_token_details"] = {
            "cache_read": 600,
            "ephemeral_5m_input_tokens": 400,
        }
        surviving_one_hour = _usage()
        surviving_one_hour["input_token_details"] = {
            "cache_read": 600,
            "ephemeral_1h_input_tokens": 400,
        }

        clamped = estimate_cost(over_total, KNOWN_MODEL, KNOWN_PROVIDER)
        as_five_minute = estimate_cost(
            surviving_five_minute, KNOWN_MODEL, KNOWN_PROVIDER
        )
        as_one_hour = estimate_cost(surviving_one_hour, KNOWN_MODEL, KNOWN_PROVIDER)

        assert clamped is not None
        assert as_five_minute is not None
        assert as_one_hour is not None
        assert clamped == pytest.approx(as_five_minute)
        assert clamped < as_one_hour

    def test_reasoning_and_audio_details_are_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, int | None] = {}

        class CapturingUsage:
            def __init__(self, **values: int | None) -> None:
                captured.update(values)

        def fake_calc_price(
            usage: object,
            model_ref: str,
            *,
            provider_id: str | None = None,
        ) -> SimpleNamespace:
            assert isinstance(usage, CapturingUsage)
            assert model_ref == KNOWN_MODEL
            assert provider_id == KNOWN_PROVIDER
            return SimpleNamespace(total_price=0.42)

        monkeypatch.setattr(
            cost_tracking,
            "_load_pricing",
            lambda: (CapturingUsage, fake_calc_price),
        )
        usage = _usage()
        usage["input_token_details"] = {"audio": 250}
        usage["output_token_details"] = {"audio": 25, "reasoning": 50}

        assert estimate_cost(usage, KNOWN_MODEL, KNOWN_PROVIDER) == pytest.approx(0.42)
        assert captured["input_audio_tokens"] == 250
        assert captured["output_audio_tokens"] == 25
        assert captured["output_tokens"] == 100
        assert captured["output_reasoning_tokens"] == 50

    def test_perplexity_reasoning_is_added_to_output_total(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Perplexity reports reasoning outside its completion-token total."""
        captured: dict[str, int | None] = {}

        class CapturingUsage:
            def __init__(self, **values: int | None) -> None:
                captured.update(values)

        def fake_calc_price(
            usage: object,
            model_ref: str,
            *,
            provider_id: str | None = None,
        ) -> SimpleNamespace:
            assert isinstance(usage, CapturingUsage)
            assert model_ref == "sonar-deep-research"
            assert provider_id == "perplexity"
            return SimpleNamespace(total_price=0.42)

        monkeypatch.setattr(
            cost_tracking,
            "_load_pricing",
            lambda: (CapturingUsage, fake_calc_price),
        )
        usage = _usage(output_tokens=100)
        usage["output_token_details"] = {"reasoning": 50}

        assert estimate_cost(
            usage, "sonar-deep-research", "perplexity"
        ) == pytest.approx(0.42)
        assert captured["output_tokens"] == 150
        assert captured["output_reasoning_tokens"] == 50

    def test_forwarded_usage_keys_are_recognized_by_genai_prices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every key `estimate_cost` forwards must be one `Usage` recognizes.

        `Usage` only warns for an unrecognized key and then drops it, so a rename
        anywhere in the allowed dependency range would silently stop pricing that
        bucket instead of failing. Feeding the real `Usage` exactly what the code
        sent -- rather than a hand-copied list -- keeps this from drifting.
        """
        captured: dict[str, int] = {}

        class CapturingUsage:
            def __init__(self, **values: int | None) -> None:
                # Accumulate only what was actually forwarded: a later request
                # that omits a bucket passes `None` and would otherwise erase
                # the key an earlier one contributed.
                captured.update(
                    {key: value for key, value in values.items() if value is not None}
                )

        monkeypatch.setattr(
            cost_tracking,
            "_load_pricing",
            lambda: (
                CapturingUsage,
                lambda *_args, **_kwargs: SimpleNamespace(total_price=0.0),
            ),
        )
        # Separate calls because the overlap guard drops input audio whenever
        # cached tokens are present, so one request cannot forward every key.
        audio = _usage()
        audio["input_token_details"] = {"audio": 100}
        audio["output_token_details"] = {"audio": 25, "reasoning": 50}
        estimate_cost(audio, KNOWN_MODEL, KNOWN_PROVIDER)
        writes = _usage()
        writes["input_token_details"] = {
            "ephemeral_5m_input_tokens": 200,
            "ephemeral_1h_input_tokens": 200,
        }
        estimate_cost(writes, KNOWN_MODEL, KNOWN_PROVIDER)
        estimate_cost(_usage(cache_read=300), KNOWN_MODEL, KNOWN_PROVIDER)

        forwarded = captured
        assert "cache_read_tokens" in forwarded
        assert "cache_write_5m_tokens" in forwarded
        assert "cache_write_1h_tokens" in forwarded
        assert "input_audio_tokens" in forwarded
        assert "output_audio_tokens" in forwarded
        assert "output_reasoning_tokens" in forwarded

        from genai_prices import Usage

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Usage(**forwarded)

    def test_audio_with_cache_reads_still_prices(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A model pricing the audio/cache intersection must stay priceable.

        `gemini-2.5-flash` prices `cache_audio_read_mtok`, so reporting both audio
        and cache reads makes `genai-prices` demand the intersection LangChain
        never provides. Without the guard the whole request would be dropped from
        the session total instead of priced at the ordinary input rate.
        """
        monkeypatch.setattr(cost_tracking, "_AUDIO_CACHE_OVERLAP_REPORTED", False)
        usage = _usage()
        usage["input_token_details"] = {"audio": 250, "cache_read": 400}

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            cost = estimate_cost(usage, "gemini-2.5-flash", "google")

        assert cost is not None
        assert cost > 0
        assert "audio/cache intersection" in caplog.text

    def test_audio_with_cache_writes_still_prices(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A model pricing audio cache writes must stay priceable.

        `gemini-2.5-flash` prices `cache_audio_write_mtok`, but LangChain does not
        provide that intersection. The audio split must be suppressed so
        `genai-prices` does not reject and omit the entire request.
        """
        monkeypatch.setattr(cost_tracking, "_AUDIO_CACHE_OVERLAP_REPORTED", False)
        audio_and_writes = _usage()
        audio_and_writes["input_token_details"] = {
            "audio": 250,
            "ephemeral_5m_input_tokens": 300,
        }
        writes_only = _usage()
        writes_only["input_token_details"] = {"ephemeral_5m_input_tokens": 300}

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            with_audio = estimate_cost(audio_and_writes, "gemini-2.5-flash", "google")
        without_audio = estimate_cost(writes_only, "gemini-2.5-flash", "google")

        assert with_audio is not None
        assert without_audio is not None
        assert with_audio == pytest.approx(without_audio)
        assert "audio/cache intersection" in caplog.text

    def test_audio_and_reasoning_details_change_the_price(self) -> None:
        """Assert the new detail buckets bill, not merely that they are passed."""
        input_audio = _usage()
        input_audio["input_token_details"] = {"audio": 250}
        output_audio = _usage()
        output_audio["output_token_details"] = {"audio": 50}

        assert (priced := estimate_cost(input_audio, "gpt-4o-transcribe", "openai"))
        assert (plain := estimate_cost(_usage(), "gpt-4o-transcribe", "openai"))
        assert priced > plain

        assert (
            audio_out := estimate_cost(output_audio, "gemini-live-2.5-flash", "google")
        )
        assert (plain_out := estimate_cost(_usage(), "gemini-live-2.5-flash", "google"))
        assert audio_out > plain_out

        # Perplexity reports reasoning outside its completion total, so this is
        # 150 output tokens of which 50 bill at the cheaper reasoning rate --
        # strictly less than 150 tokens all billed as ordinary output.
        split_reasoning = _usage(output_tokens=100)
        split_reasoning["output_token_details"] = {"reasoning": 50}
        assert (
            with_reasoning := estimate_cost(
                split_reasoning, "sonar-deep-research", "perplexity"
            )
        )
        assert (
            all_output := estimate_cost(
                _usage(output_tokens=150), "sonar-deep-research", "perplexity"
            )
        )
        assert with_reasoning < all_output

    def test_a_rejected_usage_schema_reports_broken_pricing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A `Usage` constructor that refuses fixed fields is a broken install.

        Left indistinguishable, a total pricing outage reaches the user as
        "pricing isn't available for the models used", which sends them to change
        models instead of repairing the install.
        """
        rejection = "unexpected keyword argument 'cache_write_5m_tokens'"

        class RejectingUsage:
            def __init__(self, **_values: int | None) -> None:
                raise TypeError(rejection)

        monkeypatch.setattr(cost_tracking, "_PRICING_CONTRACT_BROKEN", False)
        monkeypatch.setattr(
            cost_tracking,
            "_load_pricing",
            lambda: (
                RejectingUsage,
                lambda *_args, **_kwargs: SimpleNamespace(total_price=0.0),
            ),
        )

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            assert estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER) is None

        assert not cost_tracking.pricing_data_available()
        assert "rejected the usage schema" in caplog.text

    def test_a_request_specific_value_error_keeps_pricing_healthy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One invalid decomposition must not look like a broken installation."""

        def rejecting_calc_price(*_args: object, **_kwargs: object) -> object:
            msg = "missing cache_audio_write_tokens overlap"
            raise ValueError(msg)

        monkeypatch.setattr(cost_tracking, "_PRICING_CONTRACT_BROKEN", False)
        monkeypatch.setattr(
            cost_tracking, "_load_pricing", lambda: (dict, rejecting_calc_price)
        )

        assert estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER) is None
        assert cost_tracking.pricing_data_available()

    def test_an_uncovered_model_does_not_report_broken_pricing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model with no published rates must still read as a healthy install."""
        monkeypatch.setattr(cost_tracking, "_PRICING_CONTRACT_BROKEN", False)

        assert estimate_cost(_usage(), "no-such-model-in-the-catalog", "") is None
        assert cost_tracking.pricing_data_available()

    def test_a_successful_price_clears_an_earlier_contract_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One rejected request must not condemn a session that prices fine."""
        monkeypatch.setattr(cost_tracking, "_PRICING_CONTRACT_BROKEN", True)

        assert estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER) is not None
        assert cost_tracking.pricing_data_available()

    def test_detail_over_its_total_is_clamped_and_reported(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A detail bucket larger than its total is clamped, not dropped.

        Anthropic rather than Perplexity: the Perplexity path folds reasoning into
        the output total first, so the clamp could never fire there.
        """
        usage = _usage(output_tokens=100)
        usage["output_token_details"] = {"reasoning": 500}

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            cost = estimate_cost(usage, KNOWN_MODEL, KNOWN_PROVIDER)

        assert cost is not None
        assert "field=output reasoning reported=500 clamped=100" in caplog.text

    def test_cache_tokens_are_not_double_counted(self) -> None:
        uncached = estimate_cost(
            _usage(output_tokens=0),
            KNOWN_MODEL,
            KNOWN_PROVIDER,
        )
        all_cache_read = estimate_cost(
            _usage(output_tokens=0, cache_read=1_000),
            KNOWN_MODEL,
            KNOWN_PROVIDER,
        )
        assert uncached is not None
        assert all_cache_read is not None
        assert all_cache_read < uncached

    def test_module_import_does_not_import_genai_prices(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import deepagents_code.cost_tracking; "
                    "assert 'genai_prices' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


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

    def test_built_in_override_prices_a_model_the_catalog_lacks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The decomposition is genai-prices', not naive input/output math.

        Every published bucket bills at its own rate, so the expected total is
        computed bucket by bucket: 100 uncached input + 800 cache reads + 100
        cache writes + 60 ordinary output + 40 reasoning output.
        """
        self._install_built_ins(
            monkeypatch,
            [
                _override_model(
                    _OVERRIDE_MODEL,
                    {
                        "input_mtok": 1.0,
                        "output_mtok": 2.0,
                        "cache_read_mtok": 0.1,
                        "cache_write_mtok": 1.25,
                        "output_reasoning_mtok": 0.5,
                    },
                )
            ],
        )
        usage = _usage(cache_read=800, cache_write=100)
        usage["output_token_details"] = {"reasoning": 40}

        cost = estimate_cost(usage, _OVERRIDE_MODEL, "dcode-test")

        expected = (
            100 * 1.0 + 800 * 0.1 + 100 * 1.25 + 60 * 2.0 + 40 * 0.5
        ) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_built_in_override_prices_without_a_provider_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty provider searches every override provider's models."""
        self._install_built_ins(
            monkeypatch,
            [_override_model(_OVERRIDE_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0})],
        )

        assert estimate_cost(_usage(), _OVERRIDE_MODEL, "") == pytest.approx(
            (1_000 * 1.0 + 100 * 2.0) / 1_000_000
        )

    def test_an_unclaimed_provider_id_falls_through_to_the_full_sweep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A provider id nothing claims still reaches every override model.

        The fixture provider declares no `provider_match` and no `model_match`,
        so `find_provider_by_id` cannot claim it and inference finds nothing --
        leaving the full sweep, which is the last-resort behavior that keeps
        entries reachable where upstream would raise. Prefix-matched so the
        sweep is resolving through `ModelInfo.match` rather than an exact id.
        """
        self._install_built_ins(
            monkeypatch,
            [
                _override_model(
                    _OVERRIDE_MODEL,
                    {"input_mtok": 1.0, "output_mtok": 2.0},
                    match_prefix="dcode-override-",
                )
            ],
        )

        assert estimate_cost(_usage(), _OVERRIDE_MODEL, "unmatched-provider") == (
            pytest.approx((1_000 * 1.0 + 100 * 2.0) / 1_000_000)
        )

    def test_model_match_inference_wins_over_the_full_sweep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `model_match` claim narrows the search, as upstream's does.

        Two providers carry the model at different rates and neither id matches
        the request. Only the second declares `model_match`, so inference must
        narrow to it; a sweep would reach the first provider and bill its rate
        instead. The rates are what tell the two paths apart.
        """
        prices = {"input_mtok": 1.0, "output_mtok": 2.0}
        claimed = {"input_mtok": 3.0, "output_mtok": 6.0}
        self._install_raw_built_ins(
            monkeypatch,
            [
                *_override_catalog(
                    [_override_model(_OVERRIDE_MODEL, prices)], provider_id="swept"
                ),
                *_override_catalog(
                    [_override_model(_OVERRIDE_MODEL, claimed)],
                    provider_id="claimant",
                    model_match={"starts_with": "dcode-override-"},
                ),
            ],
        )

        assert estimate_cost(_usage(), _OVERRIDE_MODEL, "unmatched-provider") == (
            pytest.approx((1_000 * 3.0 + 100 * 6.0) / 1_000_000)
        )

    def test_a_matched_provider_id_wins_over_a_model_match_claim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The top rung of the ladder: an id claim beats an inferred claim.

        Two providers carry the model. One is named by the request; the other
        claims it via `model_match`. Swapping the two branches in
        `_find_override_model` is otherwise caught by nothing, and the rates are
        what tell them apart.
        """
        self._install_raw_built_ins(
            monkeypatch,
            [
                *_override_catalog(
                    [
                        _override_model(
                            _OVERRIDE_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0}
                        )
                    ],
                    provider_id="named",
                ),
                *_override_catalog(
                    [
                        _override_model(
                            _OVERRIDE_MODEL, {"input_mtok": 3.0, "output_mtok": 6.0}
                        )
                    ],
                    provider_id="claimant",
                    model_match={"starts_with": "dcode-override-"},
                ),
            ],
        )

        assert estimate_cost(_usage(), _OVERRIDE_MODEL, "named") == pytest.approx(
            (1_000 * 1.0 + 100 * 2.0) / 1_000_000
        )

    def test_a_claiming_provider_without_the_model_is_a_miss_not_a_sweep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A claim is final: the search does not fall through past it.

        The request names a provider that exists in the catalog but does not
        carry the model, while another provider does. Reporting nothing is the
        intended outcome -- a file that names a provider is taken at its word
        about which models it serves, and sweeping past it would bill some other
        provider's copy of the model. Pinned because the alternative reads like a
        bug fix, and "fixing" it would silently change which rate bills.
        """
        self._install_raw_built_ins(
            monkeypatch,
            [
                *_override_catalog([], provider_id="claimant"),
                *_override_catalog(
                    [
                        _override_model(
                            _OVERRIDE_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0}
                        )
                    ],
                    provider_id="holder",
                ),
            ],
        )

        assert estimate_cost(_usage(), _OVERRIDE_MODEL, "claimant") is None

    def test_a_sweep_that_prices_under_another_provider_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The one path here that can bill a confidently wrong figure.

        The same model id is commonly published by several providers at very
        different rates, so an entry reached by a sweep may be the wrong
        provider's copy. Unlike every other outcome, this hands the user a dollar
        figure rather than no figure, so it must not stay at DEBUG.
        """
        self._install_built_ins(
            monkeypatch,
            [_override_model(_OVERRIDE_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0})],
        )

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            cost = estimate_cost(_usage(), _OVERRIDE_MODEL, "some-other-provider")
            estimate_cost(_usage(), _OVERRIDE_MODEL, "some-other-provider")

        assert cost == pytest.approx((1_000 * 1.0 + 100 * 2.0) / 1_000_000)
        assert caplog.text.count("the request ran against provider") == 1
        assert "'dcode-test'" in caplog.text
        assert "'some-other-provider'" in caplog.text

    def test_pricing_under_the_named_provider_does_not_warn(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The mismatch warning must not fire on the ordinary case."""
        self._install_built_ins(
            monkeypatch,
            [_override_model(_OVERRIDE_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0})],
        )

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            assert estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test") is not None

        assert "the request ran against provider" not in caplog.text

    def test_user_file_prices_a_model_both_catalogs_lack(self, tmp_path: Path) -> None:
        self._install_user_file(
            tmp_path,
            [
                _override_model(
                    _OVERRIDE_USER_MODEL, {"input_mtok": 4.0, "output_mtok": 8.0}
                )
            ],
        )

        assert estimate_cost(_usage(), _OVERRIDE_USER_MODEL, "dcode-test") == (
            pytest.approx((1_000 * 4.0 + 100 * 8.0) / 1_000_000)
        )

    def test_user_file_beats_a_conflicting_built_in_entry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Same `(provider id, model id)`: the user rate must be the one used."""
        self._install_built_ins(
            monkeypatch,
            [
                _override_model(
                    _OVERRIDE_CONFLICT_MODEL,
                    {"input_mtok": 100.0, "output_mtok": 2.0},
                )
            ],
        )
        self._install_user_file(
            tmp_path,
            [
                _override_model(
                    _OVERRIDE_CONFLICT_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0}
                )
            ],
        )

        assert estimate_cost(_usage(), _OVERRIDE_CONFLICT_MODEL, "dcode-test") == (
            pytest.approx((1_000 * 1.0 + 100 * 2.0) / 1_000_000)
        )

    def test_user_provider_keeps_non_conflicting_bundled_models(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A partial user provider supplements rather than hides built-ins."""
        self._install_built_ins(
            monkeypatch,
            [
                _override_model(
                    _OVERRIDE_BUNDLED_MODEL,
                    {"input_mtok": 3.0, "output_mtok": 6.0},
                )
            ],
        )
        self._install_user_file(
            tmp_path,
            [
                _override_model(
                    _OVERRIDE_USER_MODEL,
                    {"input_mtok": 1.0, "output_mtok": 2.0},
                )
            ],
        )

        assert estimate_cost(_usage(), _OVERRIDE_BUNDLED_MODEL, "dcode-test") == (
            pytest.approx((1_000 * 3.0 + 100 * 6.0) / 1_000_000)
        )

    def test_overrides_are_not_consulted_when_the_primary_catalog_covers_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Upstream wins: a `calc_price` success never reaches the overrides."""
        self._install_built_ins(
            monkeypatch,
            [_override_model(KNOWN_MODEL, {"input_mtok": 999.0, "output_mtok": 999.0})],
        )
        # The built-in fixture has no `model_match` and KNOWN_PROVIDER does not
        # match its id, so were the overrides consulted the request would price
        # at the absurd fixture rate instead of the real catalog one.
        cost = estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        assert cost is not None
        assert cost < 0.01

    def test_malformed_user_file_warns_once_and_built_ins_still_price(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self._install_built_ins(
            monkeypatch,
            [_override_model(_OVERRIDE_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0})],
        )
        (tmp_path / "prices.json").write_text("{not json", encoding="utf-8")

        # Twice, so the count below means "once per process" rather than "once
        # per call" -- the claim the cached one-shot build is what buys.
        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            assert estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test") is not None
            assert estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test") is not None

        # Count the message rather than the filename: the file's path is
        # interpolated into it, so a substring count would also be counting
        # however many times `tmp_path` happens to appear.
        assert caplog.text.count("Could not parse the pricing overrides") == 1
        assert "ignoring that source" in caplog.text
        assert cost_tracking.pricing_data_available()

    def test_schema_invalid_user_file_warns_once_and_built_ins_still_price(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self._install_built_ins(
            monkeypatch,
            [_override_model(_OVERRIDE_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0})],
        )
        (tmp_path / "prices.json").write_text(
            json.dumps([{"id": "dcode-test", "bogus": True}]), encoding="utf-8"
        )

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            assert estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test") is not None
            assert estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test") is not None

        assert caplog.text.count("failed validation") == 1
        assert cost_tracking.pricing_data_available()

    def test_unreadable_user_file_warns_once_and_built_ins_still_price(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A directory where the file should be is the portable EISDIR case.

        Permanent rather than transient: a directory does not become a readable
        file on retry, so the build caches and the second call does not re-read.
        Classifying it transient would cost a re-read and a full re-validation of
        both catalogs on every unpriced request for the life of the process,
        silently -- see `_TRANSIENT_READ_ERRNOS`.
        """
        self._install_built_ins(
            monkeypatch,
            [_override_model(_OVERRIDE_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0})],
        )
        (tmp_path / "prices.json").mkdir()

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            assert estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test") is not None
            assert estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test") is not None

        assert caplog.text.count("Could not read the pricing overrides") == 1
        assert cost_tracking._PRICE_OVERRIDES is not None
        assert cost_tracking.pricing_data_available()

    def test_a_non_utf8_user_file_is_reported_and_not_retried(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A UTF-16 or binary `prices.json` raises `UnicodeDecodeError`.

        That is a `ValueError`, not an `OSError`, so it never reaches the errno
        classification -- and the same bytes cannot decode differently later, so
        it must cache like any other deterministic outcome.
        """
        self._install_built_ins(
            monkeypatch,
            [_override_model(_OVERRIDE_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0})],
        )
        (tmp_path / "prices.json").write_bytes(b"\xff\xfe[\x00]\x00")

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            assert estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test") is not None

        assert "as UTF-8 text" in caplog.text
        assert cost_tracking._PRICE_OVERRIDES is not None
        assert cost_tracking.pricing_data_available()

    def test_a_deterministic_failure_on_one_source_does_not_spam_the_other(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warn-once must not depend on the build having been cached.

        A transient failure on one source suppresses caching by design, so the
        *other* source's deterministic warning would re-fire on every unpriced
        request if it relied on the cache for deduplication rather than on
        `_report_override_once`.
        """
        import errno
        import pathlib

        # Built-ins are structurally invalid, deterministically. The user file
        # read fails transiently every time, so no build ever caches.
        self._install_raw_built_ins(monkeypatch, [{"id": "dcode-test", "bogus": True}])
        real_read_text = pathlib.Path.read_text

        def always_flaky(self: Path, *args: Any, **kwargs: Any) -> str:
            if self.name == "prices.json":
                raise OSError(errno.EMFILE, "too many open files")
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "read_text", always_flaky)
        self._install_user_file(tmp_path, [])

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            for _ in range(3):
                assert estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test") is None

        assert cost_tracking._PRICE_OVERRIDES is None
        assert caplog.text.count("failed validation") == 1
        assert caplog.text.count("Could not read the pricing overrides") == 1

    def test_invalid_overrides_do_not_touch_bundled_pricing_health(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A bad user file must not masquerade as a broken pricing install."""
        (tmp_path / "prices.json").write_text("{not json", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            assert estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER) is not None

        assert cost_tracking.pricing_data_available()

    def test_a_missing_user_file_is_a_silent_no_op(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            assert estimate_cost(_usage(), "no-such-model-in-any-catalog", "") is None

        assert "prices.json" not in caplog.text

    def test_auto_update_snapshot_swap_cannot_clobber_overrides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Composes with #5264: the hourly swap leaves override pricing intact.

        Overrides live outside the snapshot mechanism precisely because
        `UpdatePrices` wholesale-replaces the custom snapshot every refresh.
        Installing an empty snapshot -- the most destructive thing an
        auto-update can do to the primary catalog -- must not change where the
        override rate comes from.
        """
        self._install_built_ins(
            monkeypatch,
            [_override_model(_OVERRIDE_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0})],
        )

        before = estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test")

        import genai_prices.data_snapshot

        genai_prices.data_snapshot.set_custom_snapshot(
            genai_prices.data_snapshot.DataSnapshot(providers=[], from_auto_update=True)
        )
        # Drop the cached build, so the second call rebuilds the catalog *after*
        # the swap. Left cached, `after` would come back from the cache no matter
        # what the swap did to the override path, and this would be a test of the
        # cache rather than of the independence it claims to pin.
        monkeypatch.setattr(cost_tracking, "_PRICE_OVERRIDES", None)
        try:
            after = estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test")
        finally:
            genai_prices.data_snapshot.set_custom_snapshot(None)

        assert before is not None
        assert after == pytest.approx(before)

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

    def test_editing_the_user_file_mid_session_needs_a_restart(
        self, tmp_path: Path
    ) -> None:
        """The merged catalog is built once per process, deliberately.

        Documented here because it is user-visible and otherwise invisible in the
        code: without a test, adding mtime-based reloading looks free, and the
        restart requirement `_PRICE_OVERRIDES` documents would quietly stop being
        true.
        """
        self._install_user_file(
            tmp_path,
            [
                _override_model(
                    _OVERRIDE_USER_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0}
                )
            ],
        )
        first = estimate_cost(_usage(), _OVERRIDE_USER_MODEL, "dcode-test")

        self._install_user_file(
            tmp_path,
            [
                _override_model(
                    _OVERRIDE_USER_MODEL, {"input_mtok": 50.0, "output_mtok": 50.0}
                )
            ],
        )

        assert first == pytest.approx((1_000 * 1.0 + 100 * 2.0) / 1_000_000)
        assert estimate_cost(_usage(), _OVERRIDE_USER_MODEL, "dcode-test") == (
            pytest.approx(first)
        )

    def test_the_bundled_override_resource_ships_and_parses(self) -> None:
        """The wheel `include` for `bundled_prices.json` is load-bearing.

        Every other built-in test patches the read helper away, so without this
        a dropped `pyproject.toml` include or a renamed resource would ship with
        a green suite and a silently dead maintainer catalog.

        The payload is parsed rather than merely shape-checked, because a schema
        typo in the first real stopgap entry has no louder symptom in production:
        `_price_overrides` logs one warning, drops the source, and the model goes
        on showing $0 -- exactly what it showed before the entry was added.
        Deliberately not asserted empty, so adding that first entry does not
        red-fail this test and invite someone to weaken it.
        """
        from genai_prices.types import _providers_from_raw

        payload, transient = cost_tracking._read_override_source(None)

        assert transient is False
        assert isinstance(payload, list)
        assert _providers_from_raw(payload) is not None

    def test_every_bundled_override_entry_is_priced_and_links_upstream(self) -> None:
        """`bundled_prices.README.md`'s policy, enforced instead of just written.

        The policy -- every entry names its upstream genai-prices PR/issue in
        `price_comments`, so nobody has to remember why it is here or when to
        remove it -- otherwise depends on a maintainer noticing a `.README.md`
        next to a data file. An entry with no rates is the other silent defect:
        it validates, matches, and prices at $0.
        """
        from genai_prices.types import _providers_from_raw

        payload, _ = cost_tracking._read_override_source(None)
        assert isinstance(payload, list)

        for provider in _providers_from_raw(payload):
            for model in provider.models:
                where = f"{provider.id}/{model.id}"
                assert model.prices, f"{where} carries no rates"
                comments = model.price_comments or provider.price_comments or ""
                assert "genai-prices" in comments, (
                    f"{where} must link its upstream genai-prices PR/issue in "
                    f"`price_comments` (see bundled_prices.README.md); got "
                    f"{comments!r}"
                )

    def test_a_missing_bundled_resource_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unlike an absent user file, absent package data is always a defect."""
        monkeypatch.setattr(
            cost_tracking, "_BUNDLED_OVERRIDES_RESOURCE", "no-such-prices.json"
        )

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            payload, transient = cost_tracking._read_override_source(None)

        assert payload is None
        assert transient is False
        assert "shipped without its package data" in caplog.text

    def test_a_non_array_user_payload_is_rejected_with_the_schema_named(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """`{"providers": [...]}` is the shape users assume; say what is wanted."""
        self._install_built_ins(
            monkeypatch,
            [_override_model(_OVERRIDE_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0})],
        )
        (tmp_path / "prices.json").write_text(
            json.dumps({"providers": _override_catalog([])}), encoding="utf-8"
        )

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            assert estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test") is not None

        assert "must be an array of providers" in caplog.text
        assert cost_tracking.pricing_data_available()

    def test_an_empty_user_file_is_a_silent_no_op(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """`touch ~/.deepagents/prices.json` must not warn on every session."""
        self._install_built_ins(
            monkeypatch,
            [_override_model(_OVERRIDE_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0})],
        )
        (tmp_path / "prices.json").write_text("   \n", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            assert estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test") is not None

        assert "prices.json" not in caplog.text

    def test_a_transient_read_failure_is_retried_rather_than_cached(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A momentary I/O error must not read as "this user has no overrides".

        Caching the empty build would hide the user's rates for the rest of the
        process -- the mirror of the transient-failure latching
        `_PRICING_UNAVAILABLE` refuses. The warning still fires only once.
        """
        import errno
        import pathlib

        self._install_user_file(
            tmp_path,
            [
                _override_model(
                    _OVERRIDE_USER_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0}
                )
            ],
        )
        real_read_text = pathlib.Path.read_text
        attempts = 0

        def flaky(self: Path, *args: Any, **kwargs: Any) -> str:
            nonlocal attempts
            if self.name == "prices.json":
                attempts += 1
                if attempts == 1:
                    raise OSError(errno.EMFILE, "too many open files")
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "read_text", flaky)

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            first = estimate_cost(_usage(), _OVERRIDE_USER_MODEL, "dcode-test")
            second = estimate_cost(_usage(), _OVERRIDE_USER_MODEL, "dcode-test")

        assert first is None
        assert second == pytest.approx((1_000 * 1.0 + 100 * 2.0) / 1_000_000)
        assert caplog.text.count("Could not read the pricing overrides") == 1

    def test_an_override_rate_that_overflows_a_float_is_not_billed(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Rates are unbounded `Decimal`s, so a mistyped one converts to `inf`.

        `inf > 0`, so nothing downstream filters it out: it would poison the
        session total for good. The primary path returns `None` for the same
        rate, and the override path must not be the looser of the two.
        """
        self._install_built_ins(
            monkeypatch,
            [_override_model(_OVERRIDE_MODEL, {"input_mtok": int("1" + "0" * 400)})],
        )

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            cost = estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test")

        assert cost is None
        assert "not a finite non-negative cost" in caplog.text

    def test_a_negative_override_rate_is_not_billed(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A negative rate parses, so the rejection has to happen at pricing time.

        A negative cost would *subtract* from the running total, which is worse
        than reporting nothing. The installed genai-prices raises out of
        `calc_price` rather than at parse time, so what must hold is the
        observable pair: nothing is billed, and the user hears about the entry
        they hand-wrote instead of it failing at DEBUG.
        """
        self._install_built_ins(
            monkeypatch,
            [
                _override_model(
                    _OVERRIDE_MODEL, {"input_mtok": -5.0, "output_mtok": 2.0}
                )
            ],
        )

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            cost = estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test")
            estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test")

        assert cost is None
        assert caplog.text.count("its rates were rejected") == 1
        assert _OVERRIDE_MODEL in caplog.text

    def test_a_loaded_user_file_that_matches_nothing_is_reported_once(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The likeliest disappointment: ids that do not line up, and no signal.

        Without this the case is indistinguishable from having no override file
        at all, so a user who hand-wrote rates and still sees no cost has
        nothing to go on.
        """
        self._install_user_file(
            tmp_path,
            [
                _override_model(
                    _OVERRIDE_USER_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0}
                )
            ],
        )

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            assert estimate_cost(_usage(), "dcode-unmatched-model", "") is None
            assert estimate_cost(_usage(), "dcode-unmatched-model", "") is None

        assert caplog.text.count("No pricing override matched") == 1
        assert "contributed 1 model entry" in caplog.text
        # The full path, not the bare filename: which `prices.json` is the
        # actionable part, and every other message here names its source in full.
        assert str(tmp_path / "prices.json") in caplog.text

    def test_the_miss_report_is_not_spent_by_an_unrelated_model(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Each model gets its own miss report.

        A miss fires for every model the primary catalog does not cover, side
        models included. Latched once per process, the first of those would spend
        the report and the user's own misconfigured model would then miss in
        silence -- while the warning they did see named a model they never wrote
        an entry for.
        """
        self._install_user_file(
            tmp_path,
            [
                _override_model(
                    _OVERRIDE_USER_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0}
                )
            ],
        )

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            assert estimate_cost(_usage(), "dcode-some-side-model", "") is None
            assert estimate_cost(_usage(), "dcode-the-model-i-configured", "") is None

        assert caplog.text.count("No pricing override matched") == 2
        assert "dcode-the-model-i-configured" in caplog.text

    def test_a_vanished_provider_lookup_degrades_to_inference_and_still_prices(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Losing the id lookup costs narrowing, not the whole catalog.

        A different and weaker consequence than the parser going missing, so it
        needs its own coverage: the request must still be priced, through the
        sweep, with the loss reported.

        Driven through `_override_price` rather than `estimate_cost` because
        `DataSnapshot.find_provider` calls this same module-level name, so
        deleting it takes the *primary* path down too -- with a `NameError` rather
        than the `LookupError` that reaches the fallback. Simulating the rename
        end-to-end is therefore not possible; what is under test is the override
        path's own degradation.
        """
        import genai_prices.data_snapshot
        from genai_prices import Usage

        self._install_built_ins(
            monkeypatch,
            [_override_model(_OVERRIDE_MODEL, {"input_mtok": 1.0, "output_mtok": 2.0})],
        )
        usage = Usage(input_tokens=1_000, output_tokens=100)
        monkeypatch.delattr(genai_prices.data_snapshot, "find_provider_by_id")

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            cost = cost_tracking._override_price(usage, _OVERRIDE_MODEL, "dcode-test")
            cost_tracking._override_price(usage, _OVERRIDE_MODEL, "dcode-test")

        assert cost == pytest.approx((1_000 * 1.0 + 100 * 2.0) / 1_000_000)
        assert caplog.text.count("genai-prices no longer provides") == 1
        assert "find_provider_by_id" in caplog.text

    def test_an_unexpected_load_failure_warns_and_disables_the_catalog(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A load failure must not be downgraded to DEBUG by the pricing net.

        `_override_price`'s handler exists for a bad *rate*; what silently stops
        applying on a bad *load* is the user's own hand-written catalog, so that
        belongs at WARNING. Caching the empty catalog is the other half: letting
        the failure escape would rebuild and re-raise on every unpriced request
        for the life of the process.
        """

        def boom(*_args: Any, **_kwargs: Any) -> tuple[list[Any] | None, bool]:
            msg = "unanticipated"
            raise RuntimeError(msg)

        monkeypatch.setattr(cost_tracking, "_read_override_source", boom)

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            assert estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test") is None
            assert estimate_cost(_usage(), _OVERRIDE_MODEL, "dcode-test") is None

        assert caplog.text.count("Could not load the pricing override catalog") == 1
        assert cost_tracking._PRICE_OVERRIDES == ()
        assert cost_tracking.pricing_data_available()

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

    def test_first_pricing_call_starts_updater_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from genai_prices import UpdatePrices

        self._enable_auto_update(monkeypatch)
        updater = self._autospec_updater()
        factory = self._patch_updater(monkeypatch, updater)
        # Repeated calls keep exercising the real `_load_pricing` import path.
        self._stub_calc_price(monkeypatch)

        for _ in range(3):
            assert estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER) is not None

        factory.assert_called_once_with(UpdatePrices)
        updater.start.assert_called_once_with(wait=False)
        assert cost_tracking._PRICE_UPDATER is updater

    def test_starting_claims_the_genai_prices_logger(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The updater's logger never reaches `logging.lastResort`.

        `config._quiet_sdk_logging` covers the CLI, but the server graph and
        embedded hosts never call it -- and an unhandled ERROR from the hourly
        refresh prints over the TUI. Starting the updater must claim the logger
        on its own.
        """
        self._enable_auto_update(monkeypatch)
        gp_logger = logging.getLogger("genai-prices")
        monkeypatch.setattr(gp_logger, "handlers", [])
        self._patch_updater(monkeypatch, self._autospec_updater())
        self._stub_calc_price(monkeypatch)

        assert estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER) is not None

        assert gp_logger.handlers

    def test_real_updater_accepts_the_start_call_this_module_makes(self) -> None:
        """Pin the genai-prices API the start call depends on.

        Every other test here stands the updater in with an autospec, so this
        is what fails if a `>=0.1.1,<0.2.0` release renames `start`'s keyword.
        Without it, that drift kills the feature in production -- the broad
        `except` downgrades it to a warning -- while the suite stays green.
        """
        from genai_prices import UpdatePrices

        signature = inspect.signature(UpdatePrices.start)
        wait = signature.parameters["wait"]
        assert wait.kind is inspect.Parameter.KEYWORD_ONLY
        assert wait.default is False

    @pytest.mark.parametrize("value", ["false", "0", "off", ""])
    def test_disabled_env_var_never_starts_updater(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        from deepagents_code._env_vars import PRICES_AUTO_UPDATE

        monkeypatch.setenv(PRICES_AUTO_UPDATE, value)
        factory = self._patch_updater(monkeypatch, self._autospec_updater())
        self._stub_calc_price(monkeypatch)

        assert estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER) is not None

        factory.assert_not_called()

    def test_toml_opt_out_never_starts_updater(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The TOML opt-out gates the updater, not just `config get`."""
        monkeypatch.setattr(
            "deepagents_code.config_manifest.load_config_toml",
            lambda: {"update": {"prices_auto_update": False}},
        )
        factory = self._patch_updater(monkeypatch, self._autospec_updater())
        self._stub_calc_price(monkeypatch)

        assert estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER) is not None

        factory.assert_not_called()

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

    def test_offline_mode_never_starts_updater(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`DEEPAGENTS_CODE_OFFLINE` suppresses this fetch like every other."""
        from deepagents_code._env_vars import OFFLINE

        monkeypatch.setenv(OFFLINE, "1")
        factory = self._patch_updater(monkeypatch, self._autospec_updater())
        self._stub_calc_price(monkeypatch)

        assert estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER) is not None

        factory.assert_not_called()

    def test_the_start_attempt_is_serialized_by_a_lock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The latch is read and set inside `_PRICE_UPDATER_LOCK`.

        `estimate_cost` runs on the event loop and on the executor threads that
        price drained records, so an unguarded check-then-set lets two threads
        both reach `start()`. The loser trips genai-prices' process-wide
        singleton guard and its `RuntimeError` gets reported as a failed start
        -- a warning claiming the updater is down while the winner's runs fine.

        Holding the lock here and watching a pricing thread block on it tests
        that directly, rather than trying to lose a race on purpose.
        """
        from genai_prices import UpdatePrices

        self._enable_auto_update(monkeypatch)
        self._stub_calc_price(monkeypatch)
        factory = self._patch_updater(monkeypatch, self._autospec_updater())

        priced = threading.Event()

        def _price() -> None:
            estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
            priced.set()

        thread = threading.Thread(target=_price)
        with cost_tracking._PRICE_UPDATER_LOCK:
            thread.start()
            # Without the lock the thread sails through the start attempt.
            assert not priced.wait(0.25)
            factory.assert_not_called()

        thread.join(timeout=5)
        assert priced.is_set()
        factory.assert_called_once_with(UpdatePrices)

    def test_failed_start_still_prices_and_logs_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A broken updater must not break pricing or spam the log.

        The failure is latched because nothing a later request does can fix an
        incompatible genai-prices release or a claimed singleton -- retrying
        would only repeat the warning on every model call while pricing works.
        """
        self._enable_auto_update(monkeypatch)
        updater = self._autospec_updater()
        updater.start.side_effect = RuntimeError("unexpected genai-prices API")
        self._patch_updater(monkeypatch, updater)
        self._stub_calc_price(monkeypatch)

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            for _ in range(3):
                assert estimate_cost(
                    _usage(), KNOWN_MODEL, KNOWN_PROVIDER
                ) == pytest.approx(0.01)

        updater.start.assert_called_once_with(wait=False)
        warning = "Could not start the genai-prices background updater"
        records = [r for r in caplog.records if warning in r.getMessage()]
        assert len(records) == 1
        # Asserted on the formatted message, not `caplog.text`: `exc_info=True`
        # appends a traceback naming the same exception, so a check against the
        # full text would pass even if the message dropped the type entirely.
        assert "RuntimeError: unexpected genai-prices API" in records[0].getMessage()
        assert cost_tracking.pricing_data_available()
        assert cost_tracking._PRICE_UPDATER is None

    def test_a_start_failure_does_not_price_from_a_half_built_updater(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failure inside the factory itself is caught like any other."""
        self._enable_auto_update(monkeypatch)
        monkeypatch.setattr(
            cost_tracking,
            "_build_price_updater",
            MagicMock(side_effect=AttributeError("no UpdatePrices")),
        )
        self._stub_calc_price(monkeypatch)

        assert estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER) == pytest.approx(
            0.01
        )
        assert cost_tracking._PRICE_UPDATER is None


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

    def test_a_complete_catalog_is_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from genai_prices.data import providers as bundled

        updater, _ = self._guarded_updater(monkeypatch, len(bundled))

        snapshot = updater.fetch()

        assert snapshot is not None
        assert len(snapshot.providers) == len(bundled)

    def test_a_grown_catalog_is_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Upstream adding providers is the normal case, not a truncation."""
        from genai_prices.data import providers as bundled

        updater, _ = self._guarded_updater(monkeypatch, len(bundled) + 5)

        snapshot = updater.fetch()

        assert snapshot is not None
        assert len(snapshot.providers) == len(bundled) + 5

    @pytest.mark.parametrize("fetched", [0, 1, None])
    def test_a_truncated_catalog_is_refused(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        fetched: int | None,
    ) -> None:
        """An empty or half-published data.json must not replace bundled data.

        genai-prices validates only that the payload is an array of well-formed
        providers, so `[]` installs cleanly and makes every later `calc_price`
        raise `LookupError` -- which this module reports as an unpriced model
        rather than a broken catalog, so costs stop with no visible cause.
        """
        updater, bundled_count = self._guarded_updater(monkeypatch, fetched)

        with (
            caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"),
            pytest.raises(ValueError, match="Refused pricing catalog"),
        ):
            updater.fetch()

        assert f"{bundled_count} bundled" in caplog.text

    def test_a_refusal_is_reported_once_not_hourly(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The refusal recurs every hour while upstream stays broken."""
        updater, _ = self._guarded_updater(monkeypatch, 0)

        with caplog.at_level(logging.WARNING, logger="deepagents_code.cost_tracking"):
            for _ in range(3):
                with pytest.raises(ValueError, match="Refused pricing catalog"):
                    updater.fetch()

        assert caplog.text.count("Refusing an upstream pricing catalog") == 1

    def test_a_refusal_leaves_the_installed_catalog_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raising, not returning `None`, is what preserves the last good fetch.

        `UpdatePrices._update_prices` installs whatever `fetch` returns --
        including `None`, which reverts to bundled data. Raising instead makes
        the background loop treat the refresh as failed and retry later.
        """
        import genai_prices.data_snapshot as snapshot_mod

        updater, _ = self._guarded_updater(monkeypatch, 0)
        good = SimpleNamespace(providers=[object()], from_auto_update=True)
        monkeypatch.setattr(snapshot_mod, "_custom_snapshot", good)

        with pytest.raises(ValueError, match="Refused pricing catalog"):
            updater._update_prices()

        assert snapshot_mod._custom_snapshot is good


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

    def test_cost_channel_is_private_and_additive(self) -> None:
        """The channel must compile to a summing reducer, not a `LastValue`.

        LangGraph reads only the *last* entry of the `Annotated` metadata when
        detecting a reducer. Reordering it so `PrivateStateAttr` comes last
        degrades the channel to `LastValue`, at which point every delta
        overwrites the total instead of adding to it and the thread's lifetime
        cost silently collapses to whatever the final step spent. Assert the
        compiled channel and its behavior, not just the annotation.
        """
        hints = get_type_hints(CostState, include_extras=True)
        metadata = tuple(getattr(hints["_session_cost_usd"], "__metadata__", ()))
        assert PrivateStateAttr in metadata
        assert metadata

        channels = StateGraph(cast("Any", CostState)).channels
        cost_channel = channels["_session_cost_usd"]
        assert isinstance(cost_channel, BinaryOperatorAggregate)
        cost_channel.update([1.25, 0.02, 0.005])
        assert cost_channel.get() == pytest.approx(1.275)

        transfer_metadata = tuple(
            getattr(hints["_session_cost_transfers"], "__metadata__", ())
        )
        assert OmitFromInput in transfer_metadata
        assert PrivateStateAttr not in transfer_metadata

        transfer_channel = channels["_session_cost_transfers"]
        assert isinstance(transfer_channel, BinaryOperatorAggregate)
        transfer_channel.update([{"a": {"owner_scope": "", "cost_usd": 1.0}}])
        transfer_channel.update([{"b": {"owner_scope": "", "cost_usd": 2.0}}])
        # Parallel subagents hand off independently, so entries must merge
        # rather than the later write replacing the earlier one.
        assert set(cast("dict[str, Any]", transfer_channel.get())) == {"a", "b"}

    def test_returns_request_cost_as_delta(self) -> None:
        middleware = CostTrackingMiddleware()
        state: CostState = {
            "messages": [HumanMessage("hello"), _message(_usage())],
            "_session_cost_usd": 1.25,
        }
        result = middleware.after_model(state, _runtime())
        assert result is not None
        # Additive channel: only this request's estimate is returned.
        assert 0 < result["_session_cost_usd"] < 1
        assert result["_session_cost_usd"] == estimate_cost(
            _usage(),
            KNOWN_MODEL,
            KNOWN_PROVIDER,
        )

    def test_recorded_request_is_charged_once(
        self, recorder: _SessionCostRecorder
    ) -> None:
        """The recorder and the state fallback must not both charge one call."""
        _collect(recorder, _record(message_id="response-1"))
        middleware = CostTrackingMiddleware()
        state: CostState = {
            "messages": [_message(_usage(), message_id="response-1")],
        }

        result = middleware.after_model(state, _runtime(thread_id=THREAD_ID))

        assert result is not None
        assert result["_session_cost_usd"] == pytest.approx(
            estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        )

    def test_side_call_is_charged_on_top_of_the_agent_response(
        self, recorder: _SessionCostRecorder
    ) -> None:
        """A recorded request with no message in state is extra spend."""
        _collect(recorder, _record(message_id="summary-1"))
        _collect(recorder, _record(message_id="response-1"))
        middleware = CostTrackingMiddleware()
        state: CostState = {
            "messages": [_message(_usage(), message_id="response-1")],
        }

        result = middleware.after_model(state, _runtime(thread_id=THREAD_ID))

        one_call = estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        assert one_call is not None
        assert result is not None
        assert result["_session_cost_usd"] == pytest.approx(2 * one_call)

    def test_unidentified_response_is_not_charged_twice(
        self, recorder: _SessionCostRecorder
    ) -> None:
        """Without a message ID to join on, prefer undercounting to double."""
        _collect(recorder, _record(message_id=None))
        middleware = CostTrackingMiddleware()
        state: CostState = {"messages": [_message(_usage(), message_id=None)]}

        result = middleware.after_model(state, _runtime(thread_id=THREAD_ID))

        assert result is not None
        assert result["_session_cost_usd"] == pytest.approx(
            estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        )

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

    def test_records_are_not_drained_without_a_thread(
        self, recorder: _SessionCostRecorder
    ) -> None:
        """An unthreaded run cannot attribute records, but still prices itself."""
        _collect(recorder, _record(message_id="summary-1"))
        middleware = CostTrackingMiddleware()
        state: CostState = {
            "messages": [_message(_usage(), message_id="response-1")],
        }

        result = middleware.after_model(state, _runtime())

        assert result is not None
        assert result["_session_cost_usd"] == pytest.approx(
            estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        )
        assert recorder.drain(THREAD_ID)

    def test_streams_the_new_absolute_total(self) -> None:
        events: list[dict[str, Any]] = []
        middleware = CostTrackingMiddleware()
        state: CostState = {
            "messages": [_message(_usage())],
            "_session_cost_usd": 1.25,
        }

        result = middleware.after_model(
            state, _runtime(thread_id="thread-1", events=events)
        )

        assert result is not None
        assert events == [
            {
                "type": "session_cost",
                "total": pytest.approx(1.25 + result["_session_cost_usd"]),
                # Tags the total so a client that has since switched threads
                # can discard it instead of showing the old thread's spend.
                "thread_id": "thread-1",
                # Reported from where pricing runs: a remote client cannot see
                # a broken price-data install in the server's process.
                "pricing_ok": True,
            }
        ]

    def test_streamed_event_reports_broken_pricing(self) -> None:
        """The client cannot see a broken install in the server's process.

        Hard-coding this to `True` would leave a remote user reading `/cost`
        and blaming their model choice for a package fault they could fix.
        """
        events: list[dict[str, Any]] = []
        middleware = CostTrackingMiddleware()
        state: Any = {
            "messages": [_message(_usage())],
            "_model_spec": f"{KNOWN_PROVIDER}:{KNOWN_MODEL}",
            "_session_cost_usd": 1.25,
        }

        with (
            patch(
                "deepagents_code.cost_tracking.estimate_cost",
                return_value=None,
            ),
            patch(
                "deepagents_code.cost_tracking.pricing_data_available",
                return_value=False,
            ),
        ):
            result = middleware.after_model(
                state,
                _runtime(thread_id="thread-1", events=events),
            )

        assert result is None
        assert events == [
            {
                "type": "session_cost",
                "total": pytest.approx(1.25),
                "thread_id": "thread-1",
                "pricing_ok": False,
            }
        ]

    def test_failed_pricing_returns_records_to_the_recorder(
        self, recorder: _SessionCostRecorder
    ) -> None:
        """`drain` removes what it returns, so a failure must hand it back.

        Without this the spend is destroyed: there is no second copy anywhere,
        and the next drain finds an empty queue.
        """
        _collect(recorder, _record())
        middleware = CostTrackingMiddleware()
        state: Any = {"messages": []}

        with (
            patch(
                "deepagents_code.cost_tracking.estimate_cost",
                side_effect=RuntimeError("pricing exploded"),
            ),
            pytest.raises(RuntimeError),
        ):
            middleware._charge(
                state,
                _runtime(thread_id=THREAD_ID),
                price_latest_message=False,
            )

        assert len(recorder.drain(THREAD_ID)) == 1

    def test_cancelled_turn_does_not_destroy_drained_records(
        self, recorder: _SessionCostRecorder
    ) -> None:
        """`CancelledError` is a `BaseException`, so `except Exception` misses it."""
        _collect(recorder, _record())
        middleware = CostTrackingMiddleware()
        state: Any = {"messages": []}

        with (
            patch(
                "deepagents_code.cost_tracking.estimate_cost",
                side_effect=asyncio.CancelledError(),
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            middleware._charge(
                state,
                _runtime(thread_id=THREAD_ID),
                price_latest_message=False,
            )

        assert len(recorder.drain(THREAD_ID)) == 1

    def test_after_model_does_not_fail_the_turn_on_a_pricing_error(
        self, recorder: _SessionCostRecorder
    ) -> None:
        """Each hook is its own graph node; raising here would fail the turn."""
        _collect(recorder, _record())
        middleware = CostTrackingMiddleware()
        state: Any = {"messages": []}

        with patch.object(
            middleware, "_charge", side_effect=RuntimeError("boom")
        ) as charge:
            result = middleware.after_model(state, _runtime(thread_id=THREAD_ID))

        assert result is None
        assert charge.called

    def test_after_agent_does_not_fail_the_turn_on_a_pricing_error(self) -> None:
        middleware = CostTrackingMiddleware()
        state: Any = {"messages": []}

        with patch.object(
            middleware, "_after_agent_update", side_effect=RuntimeError("boom")
        ):
            result = middleware.after_agent(state, _runtime(thread_id=THREAD_ID))

        assert result is None

    def test_no_event_is_streamed_without_new_spend(self) -> None:
        events: list[dict[str, Any]] = []
        middleware = CostTrackingMiddleware()
        state = cast("CostState", {"messages": [], "_session_cost_usd": 1.25})

        assert middleware.after_model(state, _runtime(events=events)) is None
        assert events == []

    def test_after_agent_charges_late_records_only(
        self, recorder: _SessionCostRecorder
    ) -> None:
        """Tool-node spend lands in the same turn without re-pricing messages."""
        _collect(recorder, _record(message_id="compaction-summary"))
        middleware = CostTrackingMiddleware()
        state: CostState = {
            "messages": [_message(_usage(), message_id="response-1")],
        }

        result = middleware.after_agent(state, _runtime(thread_id=THREAD_ID))

        assert result is not None
        assert result["_session_cost_usd"] == pytest.approx(
            estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        )

    def test_after_agent_is_a_noop_without_late_records(self) -> None:
        middleware = CostTrackingMiddleware()
        state: CostState = {"messages": [_message(_usage())]}

        assert middleware.after_agent(state, _runtime(thread_id=THREAD_ID)) is None

    def test_nested_agent_resets_cost_on_start(self) -> None:
        middleware = CostTrackingMiddleware(nested=True)
        state = cast("CostState", {"messages": [], "_session_cost_usd": 9.5})
        result = middleware.before_agent(state, _runtime())
        assert result is not None
        assert isinstance(result["_session_cost_usd"], Overwrite)
        assert result["_session_cost_usd"].value == pytest.approx(0.0)

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

    def test_main_agent_does_not_reset_cost_on_start(self) -> None:
        middleware = CostTrackingMiddleware()
        state = cast("CostState", {"messages": [], "_session_cost_usd": 9.5})
        assert middleware.before_agent(state, _runtime()) is None

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

    def test_codex_model_spec_overrides_generic_openai_message_metadata(self) -> None:
        middleware = CostTrackingMiddleware()
        state: CostState = {
            "messages": [_message(_usage(), model="gpt-5.4", provider="openai")],
            "_model_spec": "openai_codex:gpt-5.4",
            "_session_cost_usd": 2.5,
        }

        assert middleware.after_model(state, _runtime()) is None

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
            pytest.param("azure_openai", 0.42, id="azure"),
            pytest.param("openai_codex", None, id="codex-subscription"),
        ],
    )
    def test_hidden_openai_response_uses_its_configured_provider(
        self,
        recorder: _SessionCostRecorder,
        monkeypatch: pytest.MonkeyPatch,
        configured_provider: str,
        expected_delta: float | None,
    ) -> None:
        """A side call retains its provider without sharing the main message ID."""
        from deepagents_code import cost_tracking

        pricing_targets: list[tuple[str, str]] = []

        def price(
            usage_metadata: object,
            model_name: str,
            provider: str = "",
        ) -> float | None:
            assert usage_metadata
            pricing_targets.append((model_name, provider))
            return None if provider == "openai_codex" else 0.42

        monkeypatch.setattr(cost_tracking, "estimate_cost", price)
        _collect(
            recorder,
            _record(message_id="summary-1", model="gpt-5.4", provider="openai"),
            configured_provider=configured_provider,
        )
        middleware = CostTrackingMiddleware()
        state: CostState = {
            "messages": [_message(_usage(), model="gpt-5.4", provider="openai")],
            "_model_spec": f"{configured_provider}:gpt-5.4",
        }

        result = middleware.after_agent(state, _runtime(thread_id=THREAD_ID))

        assert pricing_targets == [("gpt-5.4", configured_provider)]
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

    def test_unpriceable_model_leaves_prior_total_unchanged(self) -> None:
        middleware = CostTrackingMiddleware()
        state: CostState = {
            "messages": [
                _message(
                    _usage(),
                    model="definitely-not-a-real-model",
                    provider="unknown-provider",
                )
            ],
            "_session_cost_usd": 2.5,
        }
        assert middleware.after_model(state, _runtime()) is None

    @pytest.mark.parametrize(
        "messages",
        [[], [HumanMessage("hello")], [AIMessage(content="no usage")]],
    )
    def test_no_priceable_message_is_a_noop(self, messages: list[Any]) -> None:
        middleware = CostTrackingMiddleware()
        state = cast("CostState", {"messages": messages})
        assert middleware.after_model(state, _runtime()) is None


class TestSessionCostRecorder:
    """Tests for the callback handler that collects completed requests."""

    def test_records_are_scoped_to_their_thread(
        self, recorder: _SessionCostRecorder
    ) -> None:
        _collect(recorder, _record(message_id="a"), thread_id="thread-a")
        _collect(recorder, _record(message_id="b"), thread_id="thread-b")

        drained = recorder.drain("thread-a")

        assert [record.message_id for record in drained] == ["a"]
        assert [record.message_id for record in recorder.drain("thread-b")] == ["b"]

    def test_draining_clears_the_queue(self, recorder: _SessionCostRecorder) -> None:
        _collect(recorder, _record())

        assert len(recorder.drain(THREAD_ID)) == 1
        assert recorder.drain(THREAD_ID) == []

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

    async def test_model_provider_metadata_survives_side_invoke_metadata(
        self, recorder: _SessionCostRecorder
    ) -> None:
        """Per-call metadata must not erase the provider attached to the model."""
        model = _fake_model(
            _message(
                _usage(),
                model="gpt-5.4",
                provider="openai",
                message_id="summary-1",
            )
        )
        _set_configured_model_metadata(model, "gpt-5.4", "openai_codex")

        await model.ainvoke(
            "summarize",
            config={
                "metadata": {
                    "thread_id": THREAD_ID,
                    "lc_source": "summarization",
                }
            },
        )

        records = recorder.drain(THREAD_ID)
        assert len(records) == 1
        assert records[0].provider == "openai_codex"

    def test_request_without_a_thread_is_not_recorded(
        self, recorder: _SessionCostRecorder
    ) -> None:
        run_id = uuid4()
        recorder.on_chat_model_start({}, [], run_id=run_id, metadata={})
        assert recorder._run_contexts[run_id].scope == ""
        recorder.on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=_message(_usage()))]]),
            run_id=run_id,
        )

        assert recorder.drain(THREAD_ID) == []

    def test_failed_request_is_forgotten(self, recorder: _SessionCostRecorder) -> None:
        run_id = uuid4()
        recorder.on_chat_model_start(
            {}, [], run_id=run_id, metadata={"thread_id": THREAD_ID}
        )
        recorder.on_llm_error(RuntimeError("provider failed"), run_id=run_id)
        recorder.on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=_message(_usage()))]]),
            run_id=run_id,
        )

        assert recorder.drain(THREAD_ID) == []

    def test_response_without_usage_is_not_recorded(
        self, recorder: _SessionCostRecorder
    ) -> None:
        run_id = uuid4()
        recorder.on_chat_model_start(
            {}, [], run_id=run_id, metadata={"thread_id": THREAD_ID}
        )
        recorder.on_llm_end(
            LLMResult(
                generations=[[ChatGeneration(message=AIMessage(content="no usage"))]]
            ),
            run_id=run_id,
        )

        assert recorder.drain(THREAD_ID) == []

    def test_undrained_threads_are_bounded(
        self, recorder: _SessionCostRecorder
    ) -> None:
        """A process that never drains must not grow for its whole lifetime."""
        from deepagents_code.cost_tracking import _MAX_TRACKED_THREADS

        for index in range(_MAX_TRACKED_THREADS + 5):
            _collect(recorder, _record(), thread_id=f"thread-{index}")

        assert recorder.drain("thread-0") == []
        assert len(recorder.drain(f"thread-{_MAX_TRACKED_THREADS + 4}")) == 1

    def test_undrained_records_for_one_thread_are_bounded(
        self, recorder: _SessionCostRecorder
    ) -> None:
        from deepagents_code.cost_tracking import _MAX_RECORDS_PER_THREAD

        for index in range(_MAX_RECORDS_PER_THREAD + 3):
            _collect(recorder, _record(message_id=f"m-{index}"))

        drained = recorder.drain(THREAD_ID)

        assert len(drained) == _MAX_RECORDS_PER_THREAD
        assert drained[-1].message_id == f"m-{_MAX_RECORDS_PER_THREAD + 2}"

    def test_restored_records_remain_bounded(
        self, recorder: _SessionCostRecorder
    ) -> None:
        """A failed batch cannot bypass either recorder retention limit."""
        from deepagents_code.cost_tracking import (
            _MAX_RECORDS_PER_THREAD,
            _MAX_TRACKED_THREADS,
        )

        for index in range(_MAX_RECORDS_PER_THREAD):
            _collect(recorder, _record(message_id=f"new-{index}"))
        recorder.restore(THREAD_ID, [_record(message_id="old")])

        restored = recorder.drain(THREAD_ID)
        assert len(restored) == _MAX_RECORDS_PER_THREAD
        assert restored[0].message_id == "new-0"

        for index in range(_MAX_TRACKED_THREADS):
            _collect(recorder, _record(), thread_id=f"thread-{index}")
        recorder.restore("restored-thread", [_record(message_id="restored")])

        assert len(recorder._records) == _MAX_TRACKED_THREADS
        assert recorder.drain("thread-0") == []
        assert [record.message_id for record in recorder.drain("restored-thread")] == [
            "restored"
        ]


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

    async def test_main_agent_call_is_charged_once(self) -> None:
        agent = self._agent(model=_fake_model(_message(_usage(), message_id="a")))

        total_usd, _totals = await self._run(agent)

        assert total_usd == pytest.approx(self._one_call_usd())

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

    async def test_offload_summarization_call_is_charged(self) -> None:
        agent = self._agent(
            model=_fake_model(_message(_usage(), message_id="a")),
            middleware=[
                _SideInvokeMiddleware(
                    _fake_model(_message(_usage(), message_id="summary")),
                    "summarization",
                )
            ],
        )

        total_usd, _totals = await self._run(agent)

        assert total_usd == pytest.approx(2 * self._one_call_usd())

    async def test_real_summarization_middleware_call_is_charged(self) -> None:
        """The stock summarization middleware summarizes in its own node.

        Its model call therefore never passes through the agent's model node or
        `after_model` at all, which is the case the recorder exists for.
        """
        summary_model = _fake_model(_message(_usage(), message_id="summary"))
        agent = self._agent(
            model=_fake_model(_message(_usage(), message_id="a")),
            middleware=[
                SummarizationMiddleware(
                    model=summary_model,
                    trigger=("messages", 2),
                    keep=("messages", 1),
                )
            ],
        )

        total_usd, _totals = await self._run(
            agent,
            messages=[
                HumanMessage("first"),
                _message(_usage(), message_id="earlier"),
                HumanMessage("hello"),
            ],
        )

        # The summary call plus this turn's own call. The seeded history is not
        # re-priced: only requests completed in this run are charged.
        assert total_usd == pytest.approx(2 * self._one_call_usd())

    async def test_auto_classifier_call_is_charged(self) -> None:
        agent = self._agent(
            model=_fake_model(_message(_usage(), message_id="a")),
            middleware=[
                _SideInvokeMiddleware(
                    _fake_model(_message(_usage(), message_id="decision")),
                    "auto_mode_classifier",
                )
            ],
        )

        total_usd, _totals = await self._run(agent)

        assert total_usd == pytest.approx(2 * self._one_call_usd())

    async def test_subagent_spend_is_charged_once(self) -> None:
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
            ),
            tools=[task],
        )

        total_usd, _totals = await self._run(agent)

        # Two parent steps and the one nested call, each counted exactly once.
        assert total_usd == pytest.approx(3 * self._one_call_usd())

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

    async def test_unpriceable_model_leaves_the_total_alone(self) -> None:
        agent = self._agent(
            model=_fake_model(
                _message(
                    _usage(),
                    model="definitely-not-a-real-model",
                    provider="unknown-provider",
                    message_id="a",
                )
            )
        )

        total_usd, totals = await self._run(agent)

        assert total_usd == pytest.approx(0.0)
        assert totals == []
