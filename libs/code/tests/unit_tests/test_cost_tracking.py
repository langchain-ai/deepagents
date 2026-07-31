"""Tests for cost estimation and graph-side cumulative cost persistence."""

from __future__ import annotations

import asyncio
import builtins
import logging
import subprocess
import sys
import warnings
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast, get_type_hints
from unittest.mock import patch
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
    SESSION_COST_EVENT_TYPE,
    CostState,
    CostTrackingMiddleware,
    _ModelCallRecord,
    _SessionCostRecorder,
    _set_configured_provider_metadata,
    estimate_cost,
    resolve_message_model,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime

KNOWN_MODEL = "claude-sonnet-4-5"
KNOWN_PROVIDER = "anthropic"
THREAD_ID = "thread-under-test"


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


class TestCostTrackingMiddleware:
    """Tests for cumulative cost writes on the model checkpoint path."""

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
        _set_configured_provider_metadata(model, "openai_codex")

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
