"""Tests for cost estimation and graph-side cumulative cost persistence."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast, get_type_hints
from uuid import uuid4

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Overwrite

from deepagents_code._fake_models import _ToolBindingFakeModel
from deepagents_code.cost_tracking import (
    _RECORDER_VAR,
    SESSION_COST_EVENT_TYPE,
    CostState,
    CostTrackingMiddleware,
    _ModelCallRecord,
    _SessionCostRecorder,
    cache_token_counts,
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
    events: list[dict[str, Any]] | None = None,
) -> Runtime[Any]:
    """Build the runtime shape required by the middleware hooks.

    Args:
        thread_id: Thread the run belongs to, or `None` for an unthreaded run
            whose recorded calls cannot be drained.
        events: List that collects emitted custom-stream events, when given.
    """
    return cast(
        "Runtime[Any]",
        SimpleNamespace(
            context=None,
            execution_info=SimpleNamespace(thread_id=thread_id),
            stream_writer=events.append if events is not None else None,
        ),
    )


def _record(
    *,
    message_id: str | None = "response-1",
    model: str = KNOWN_MODEL,
    provider: str = KNOWN_PROVIDER,
    usage: dict[str, Any] | None = None,
) -> _ModelCallRecord:
    """Build a recorded completed model request."""
    return _ModelCallRecord(
        message_id=message_id,
        usage_metadata=usage if usage is not None else _usage(),
        model_name=model,
        provider=provider,
    )


def _collect(
    recorder: _SessionCostRecorder,
    record: _ModelCallRecord,
    *,
    thread_id: str = THREAD_ID,
) -> None:
    """Put one already-built record into a recorder's pending queue."""
    run_id = uuid4()
    recorder.on_chat_model_start(
        {},
        [],
        run_id=run_id,
        metadata={"thread_id": thread_id},
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


class TestCacheTokenCounts:
    """Tests for cache metadata normalization shared by pricing and the UI."""

    def test_reads_and_writes_are_returned(self) -> None:
        assert cache_token_counts(_usage(cache_read=600, cache_write=300)) == (600, 300)

    def test_counts_are_clamped_to_inclusive_input(self) -> None:
        assert cache_token_counts(_usage(cache_read=900, cache_write=900)) == (900, 100)

    def test_malformed_details_are_ignored(self) -> None:
        usage = _usage()
        usage["input_token_details"] = {"cache_read": True, "cache_creation": -1}
        assert cache_token_counts(usage) == (0, 0)

    def test_detailed_anthropic_writes_are_summed(self) -> None:
        usage = _usage()
        usage["input_token_details"] = {
            "cache_creation": 0,
            "ephemeral_5m_input_tokens": 200,
            "ephemeral_1h_input_tokens": 100,
        }
        assert cache_token_counts(usage) == (0, 300)


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

    def test_cache_write_alias_is_priced_separately(self) -> None:
        uncached = estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        usage = _usage()
        usage["input_token_details"] = {"cache_write": 900}
        cached = estimate_cost(usage, KNOWN_MODEL, KNOWN_PROVIDER)

        assert uncached is not None
        assert cached is not None
        assert cached > uncached

    def test_anthropic_detailed_cache_writes_are_priced_separately(self) -> None:
        """Anthropic zeroes `cache_creation` when TTL breakdown fields are set."""
        uncached = estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        usage = _usage()
        usage["input_token_details"] = {
            "cache_creation": 0,
            "ephemeral_5m_input_tokens": 600,
            "ephemeral_1h_input_tokens": 300,
        }
        cached = estimate_cost(usage, KNOWN_MODEL, KNOWN_PROVIDER)

        assert uncached is not None
        assert cached is not None
        assert cached > uncached
        # Same total as pricing the sum through the generic cache-write field.
        assert cached == estimate_cost(
            _usage(cache_write=900),
            KNOWN_MODEL,
            KNOWN_PROVIDER,
        )

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
        hints = get_type_hints(CostState, include_extras=True)
        metadata = tuple(getattr(hints["_session_cost_usd"], "__metadata__", ()))
        assert PrivateStateAttr in metadata
        assert metadata
        last = metadata[-1]
        assert getattr(last, "__name__", None) == "add"

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

        result = middleware.after_model(state, _runtime(events=events))

        assert result is not None
        assert events == [
            {
                "type": "session_cost",
                "total": pytest.approx(1.25 + result["_session_cost_usd"]),
            }
        ]

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

    def test_nested_agent_records_nothing(self, recorder: _SessionCostRecorder) -> None:
        """Only the thread's main agent writes the channel."""
        _collect(recorder, _record(message_id="nested-1"))
        middleware = CostTrackingMiddleware(nested=True)
        state: CostState = {
            "messages": [_message(_usage(), message_id="nested-1")],
        }
        runtime = _runtime(thread_id=THREAD_ID)

        assert middleware.after_model(state, runtime) is None
        assert middleware.after_agent(state, runtime) is None
        # The records stay for the main agent's next drain.
        assert len(recorder.drain(THREAD_ID)) == 1

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

    def test_request_without_a_thread_is_not_recorded(
        self, recorder: _SessionCostRecorder
    ) -> None:
        run_id = uuid4()
        recorder.on_chat_model_start({}, [], run_id=run_id, metadata={})
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
        async def task(query: str) -> str:
            """Run a nested agent."""
            result = await child.ainvoke({"messages": [HumanMessage(query)]})
            return result["messages"][-1].text

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
        async def task(query: str) -> str:
            """Run a nested agent."""
            result = await child.ainvoke({"messages": [HumanMessage(query)]})
            return result["messages"][-1].text

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
