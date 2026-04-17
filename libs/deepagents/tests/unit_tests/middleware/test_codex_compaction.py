"""Unit tests for `CodexCompactionMiddleware`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from deepagents.middleware.codex_compaction import CodexCompactionMiddleware
from deepagents.middleware.summarization import _DeepAgentsSummarizationMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel
from tests.unit_tests.middleware.test_summarization_middleware import (
    MockBackend,
    make_conversation_messages,
    make_mock_runtime,
    mock_get_config,
)

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState


# -----------------------------------------------------------------------------
# Fixtures and helpers
# -----------------------------------------------------------------------------


class _FakeCodexModel(GenericFakeChatModel):
    """Chat model that looks enough like ChatOpenAI for compaction tests.

    Exposes ``model_name`` so ``_resolve_model_name`` can pick it up, and a
    profile with ``max_input_tokens`` so the inner summarization middleware
    picks fraction-based trigger defaults.
    """

    model_name: str = "gpt-5.3-codex"

    @property
    def profile(self) -> dict[str, Any]:
        return {"max_input_tokens": 1000}


def _make_model() -> _FakeCodexModel:
    """Build a fake Codex model with a spare reply in the iterator.

    The summarization fallback path invokes the model once for its LLM
    summary; supplying a response keeps the test from hanging if fallback
    fires unexpectedly.
    """
    return _FakeCodexModel(messages=iter([AIMessage(content="fallback summary")]))


def _make_state(messages: list[BaseMessage], compaction_item: dict[str, Any] | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {"messages": messages}
    if compaction_item is not None:
        state["codex_compaction_item"] = compaction_item
    return state


def _make_compacted_response(items: list[dict[str, Any]]) -> SimpleNamespace:
    """Build a stand-in for ``CompactedResponse``.

    The middleware only uses ``result.output`` and calls ``model_dump`` on
    each item, so a lightweight namespace with pre-dumped items suffices.
    """
    dumped_items = [SimpleNamespace(model_dump=lambda i=item, **_kw: i) for item in items]
    return SimpleNamespace(output=dumped_items)


async def _run_awrap(
    middleware: CodexCompactionMiddleware,
    state: dict[str, Any],
    runtime: Any,  # noqa: ANN401  # langgraph Runtime is mock-constructed in tests
) -> tuple[Any, ModelRequest | None]:
    """Invoke ``awrap_model_call`` and capture the request handed to the handler."""
    request = ModelRequest(
        model=middleware._model,
        messages=state["messages"],
        system_message=None,
        tools=[],
        runtime=runtime,
        state=cast("AgentState[Any]", state),
    )

    captured: dict[str, ModelRequest] = {}

    async def handler(req: ModelRequest) -> AIMessage:
        captured["req"] = req
        return AIMessage(content="model reply")

    with mock_get_config("test-thread-codex"):
        result = await middleware.awrap_model_call(request, handler)

    return result, captured.get("req")


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------


class TestInit:
    """Construction and composition."""

    def test_composes_inner_summarization(self) -> None:
        middleware = CodexCompactionMiddleware(_make_model(), MockBackend())
        assert isinstance(middleware._inner, _DeepAgentsSummarizationMiddleware)

    def test_resolve_model_name_reads_attribute(self) -> None:
        middleware = CodexCompactionMiddleware(_make_model(), MockBackend())
        assert middleware._resolve_model_name() == "gpt-5.3-codex"


# -----------------------------------------------------------------------------
# Trigger threshold
# -----------------------------------------------------------------------------


class TestTrigger:
    """Below-trigger forwards; above-trigger calls `/compact`."""

    async def test_below_trigger_forwards_untouched(self) -> None:
        messages = make_conversation_messages(num_old=1, num_recent=1)
        middleware = CodexCompactionMiddleware(_make_model(), MockBackend())
        # Force `_should_summarize` to be False so we bypass /compact entirely.
        middleware._inner._should_summarize = MagicMock(return_value=False)

        runtime = make_mock_runtime()
        result, captured = await _run_awrap(middleware, _make_state(messages), runtime)

        assert captured is not None
        assert not isinstance(result, ExtendedModelResponse)
        # No compaction happened, so the message list the handler saw matches the state.
        assert len(captured.messages) == len(messages)

    async def test_above_trigger_calls_compact(self) -> None:
        messages = make_conversation_messages(num_old=6, num_recent=2)
        middleware = CodexCompactionMiddleware(_make_model(), MockBackend())
        middleware._inner._should_summarize = MagicMock(return_value=True)
        middleware._inner._determine_cutoff_index = MagicMock(return_value=4)

        fake_client = MagicMock()
        fake_client.responses.compact = AsyncMock(return_value=_make_compacted_response([{"type": "compaction", "encrypted_content": "opaque-blob"}]))
        middleware._client = fake_client

        runtime = make_mock_runtime()
        result, captured = await _run_awrap(middleware, _make_state(messages), runtime)

        fake_client.responses.compact.assert_awaited_once()
        kwargs = fake_client.responses.compact.await_args.kwargs
        assert kwargs["model"] == "gpt-5.3-codex"
        # Compaction happened — expect ExtendedModelResponse with a state update.
        assert isinstance(result, ExtendedModelResponse)
        event = result.command.update["codex_compaction_item"]
        assert event["output"] == [{"type": "compaction", "encrypted_content": "opaque-blob"}]
        # The handler saw the compact head + preserved messages, not the raw state list.
        assert captured is not None
        assert len(captured.messages) < len(messages)
        head = captured.messages[0]
        assert isinstance(head, AIMessage)
        assert head.content == [{"type": "compaction", "encrypted_content": "opaque-blob"}]


# -----------------------------------------------------------------------------
# Prior item splice
# -----------------------------------------------------------------------------


class TestPriorItemSplice:
    """Stored compaction output is reused on the next turn."""

    async def test_prior_item_prepended_to_effective_messages(self) -> None:
        messages = make_conversation_messages(num_old=2, num_recent=2)
        middleware = CodexCompactionMiddleware(_make_model(), MockBackend())
        middleware._inner._should_summarize = MagicMock(return_value=False)

        prior_output = [{"type": "compaction", "encrypted_content": "earlier-blob"}]
        prior_event = {"cutoff_index": 2, "output": prior_output, "file_path": "/x.md"}
        state = _make_state(messages, compaction_item=prior_event)

        runtime = make_mock_runtime()
        _result, captured = await _run_awrap(middleware, state, runtime)

        assert captured is not None
        # Effective list: [compaction_head, *messages[2:]] where messages has 4 items.
        assert len(captured.messages) == 3
        head = captured.messages[0]
        assert isinstance(head, AIMessage)
        assert head.content == prior_output

    async def test_prior_item_prepended_to_compact_input(self) -> None:
        messages = make_conversation_messages(num_old=4, num_recent=2)
        middleware = CodexCompactionMiddleware(_make_model(), MockBackend())
        middleware._inner._should_summarize = MagicMock(return_value=True)
        middleware._inner._determine_cutoff_index = MagicMock(return_value=3)

        prior_output = [{"type": "compaction", "encrypted_content": "earlier-blob"}]
        prior_event = {"cutoff_index": 1, "output": prior_output, "file_path": "/x.md"}

        fake_client = MagicMock()
        fake_client.responses.compact = AsyncMock(return_value=_make_compacted_response([{"type": "compaction", "encrypted_content": "new-blob"}]))
        middleware._client = fake_client

        runtime = make_mock_runtime()
        state = _make_state(messages, compaction_item=prior_event)
        await _run_awrap(middleware, state, runtime)

        # /compact was called with prior output items prefixed on the input.
        kwargs = fake_client.responses.compact.await_args.kwargs
        assert kwargs["input"][0] == prior_output[0]

    async def test_prior_item_not_duplicated_in_compact_input(self) -> None:
        """Regression guard for duplicate prior compaction items.

        The synthetic head round-trips the prior compaction item through
        ``_construct_responses_api_input``, so we must not also prepend
        ``prior_event["output"]`` on top of it.
        """
        messages = make_conversation_messages(num_old=4, num_recent=2)
        middleware = CodexCompactionMiddleware(_make_model(), MockBackend())
        middleware._inner._should_summarize = MagicMock(return_value=True)
        middleware._inner._determine_cutoff_index = MagicMock(return_value=3)

        prior_output = [
            {
                "type": "compaction",
                "encrypted_content": "earlier-blob",
                "id": "compact_earlier",
            }
        ]
        prior_event = {"cutoff_index": 1, "output": prior_output, "file_path": "/x.md"}

        fake_client = MagicMock()
        fake_client.responses.compact = AsyncMock(return_value=_make_compacted_response([{"type": "compaction", "encrypted_content": "new-blob"}]))
        middleware._client = fake_client

        runtime = make_mock_runtime()
        state = _make_state(messages, compaction_item=prior_event)
        await _run_awrap(middleware, state, runtime)

        input_items = fake_client.responses.compact.await_args.kwargs["input"]
        compaction_items = [item for item in input_items if item.get("type") == "compaction"]
        assert len(compaction_items) == 1, f"prior compaction item sent twice: {compaction_items}"
        assert compaction_items[0]["encrypted_content"] == "earlier-blob"
        assert compaction_items[0]["id"] == "compact_earlier"


# -----------------------------------------------------------------------------
# Phase drift-guard on langchain-openai
# -----------------------------------------------------------------------------


class TestPhaseDriftGuard:
    """`_construct_responses_api_input` must keep the `phase` field on blocks."""

    def test_phase_lifted_to_item(self) -> None:
        # Imported inside the test to keep it isolated from the middleware and
        # act as a pure drift-guard on langchain-openai itself.
        from langchain_openai.chat_models.base import _construct_responses_api_input  # noqa: PLC0415

        msg = AIMessage(
            id="ai-phase-1",
            content=[
                {
                    "type": "output_text",
                    "text": "let me think",
                    "phase": "commentary",
                    "id": "ai-phase-1",
                }
            ],
        )
        items = _construct_responses_api_input([msg])

        assert any(item.get("phase") == "commentary" for item in items)


# -----------------------------------------------------------------------------
# Phase propagation end-to-end
# -----------------------------------------------------------------------------


class TestPhaseEndToEnd:
    """Phase on a content block must survive the trip to `/compact`."""

    async def test_phase_in_compact_input(self) -> None:
        msg_with_phase = AIMessage(
            id="ai-phase-2",
            content=[
                {
                    "type": "output_text",
                    "text": "commentary chunk",
                    "phase": "commentary",
                    "id": "ai-phase-2",
                }
            ],
        )
        messages: list[BaseMessage] = [
            HumanMessage(content="hi", id="u-0"),
            msg_with_phase,
            HumanMessage(content="keep me", id="u-1"),
            HumanMessage(content="keep me too", id="u-2"),
        ]

        middleware = CodexCompactionMiddleware(_make_model(), MockBackend())
        middleware._inner._should_summarize = MagicMock(return_value=True)
        middleware._inner._determine_cutoff_index = MagicMock(return_value=2)

        fake_client = MagicMock()
        fake_client.responses.compact = AsyncMock(return_value=_make_compacted_response([{"type": "compaction", "encrypted_content": "phased-blob"}]))
        middleware._client = fake_client

        runtime = make_mock_runtime()
        await _run_awrap(middleware, _make_state(messages), runtime)

        input_items = fake_client.responses.compact.await_args.kwargs["input"]
        assert any(item.get("phase") == "commentary" for item in input_items), input_items


# -----------------------------------------------------------------------------
# Arg truncation still fires
# -----------------------------------------------------------------------------


class TestArgTruncation:
    """Composition must preserve the inner middleware's arg-truncation pass."""

    async def test_truncate_args_invoked(self) -> None:
        messages = make_conversation_messages(num_old=1, num_recent=1)
        middleware = CodexCompactionMiddleware(_make_model(), MockBackend())
        middleware._inner._should_summarize = MagicMock(return_value=False)
        spy = MagicMock(wraps=middleware._inner._truncate_args)
        middleware._inner._truncate_args = spy

        runtime = make_mock_runtime()
        await _run_awrap(middleware, _make_state(messages), runtime)

        spy.assert_called_once()


# -----------------------------------------------------------------------------
# Fallback on /compact failure
# -----------------------------------------------------------------------------


class TestFallback:
    """Any `/compact` failure falls back to the inner summarization path."""

    async def test_fallback_on_compact_error(self) -> None:
        messages = make_conversation_messages(num_old=4, num_recent=2)
        middleware = CodexCompactionMiddleware(_make_model(), MockBackend())
        middleware._inner._should_summarize = MagicMock(return_value=True)
        middleware._inner._determine_cutoff_index = MagicMock(return_value=3)

        fake_client = MagicMock()
        fake_client.responses.compact = AsyncMock(side_effect=RuntimeError("boom"))
        middleware._client = fake_client

        # Patch the inner middleware's awrap_model_call so we can assert it was
        # called without running the real summarization (which would hit the
        # fake model and potentially run out of responses).
        fallback = AsyncMock(return_value=AIMessage(content="fallback"))
        middleware._inner.awrap_model_call = fallback

        runtime = make_mock_runtime()
        await _run_awrap(middleware, _make_state(messages), runtime)

        fallback.assert_awaited_once()

    async def test_fallback_does_not_double_offload(self) -> None:
        """On `/compact` failure the wrapper must not offload before falling back.

        If the wrapper's own offload ran concurrently with the failed
        ``/compact`` call, it would land in the backend; the summarization
        fallback then performs its own offload, giving us two sections of the
        same pre-compaction window in ``conversation_history/{thread_id}.md``.
        """
        messages = make_conversation_messages(num_old=4, num_recent=2)
        backend = MockBackend()
        middleware = CodexCompactionMiddleware(_make_model(), backend)
        middleware._inner._should_summarize = MagicMock(return_value=True)
        middleware._inner._determine_cutoff_index = MagicMock(return_value=3)

        fake_client = MagicMock()
        fake_client.responses.compact = AsyncMock(side_effect=RuntimeError("boom"))
        middleware._client = fake_client

        # Stub the fallback so we can assert invocation without running the
        # real summarization (the stub replaces the inner middleware's own
        # offload call too, so any write to the backend must have come from
        # the wrapper).
        fallback = AsyncMock(return_value=AIMessage(content="fallback"))
        middleware._inner.awrap_model_call = fallback

        runtime = make_mock_runtime()
        await _run_awrap(middleware, _make_state(messages), runtime)

        fallback.assert_awaited_once()
        assert not backend.write_calls, f"wrapper offloaded before fallback: {backend.write_calls}"
        assert not backend.edit_calls, f"wrapper offloaded before fallback: {backend.edit_calls}"


# -----------------------------------------------------------------------------
# Timeout on /compact
# -----------------------------------------------------------------------------


class TestCompactTimeout:
    """`/compact` calls must have a bounded timeout so hangs do not stall a turn."""

    async def test_compact_called_with_timeout(self) -> None:
        messages = make_conversation_messages(num_old=4, num_recent=2)
        middleware = CodexCompactionMiddleware(_make_model(), MockBackend())
        middleware._inner._should_summarize = MagicMock(return_value=True)
        middleware._inner._determine_cutoff_index = MagicMock(return_value=3)

        fake_client = MagicMock()
        fake_client.responses.compact = AsyncMock(return_value=_make_compacted_response([{"type": "compaction", "encrypted_content": "blob"}]))
        middleware._client = fake_client

        runtime = make_mock_runtime()
        await _run_awrap(middleware, _make_state(messages), runtime)

        kwargs = fake_client.responses.compact.await_args.kwargs
        assert "timeout" in kwargs, "compact call must supply a timeout"
        assert isinstance(kwargs["timeout"], (int, float))
        assert kwargs["timeout"] > 0


# -----------------------------------------------------------------------------
# History offload
# -----------------------------------------------------------------------------


class TestOffload:
    """Pre-compaction messages land in the backend."""

    async def test_offload_writes_to_backend(self) -> None:
        messages = make_conversation_messages(num_old=4, num_recent=2)
        backend = MockBackend()
        middleware = CodexCompactionMiddleware(_make_model(), backend)
        middleware._inner._should_summarize = MagicMock(return_value=True)
        middleware._inner._determine_cutoff_index = MagicMock(return_value=3)

        fake_client = MagicMock()
        fake_client.responses.compact = AsyncMock(return_value=_make_compacted_response([{"type": "compaction", "encrypted_content": "blob"}]))
        middleware._client = fake_client

        runtime = make_mock_runtime()
        await _run_awrap(middleware, _make_state(messages), runtime)

        # MockBackend records writes on its `write_calls` / `edit_calls` lists.
        assert backend.write_calls or backend.edit_calls


# -----------------------------------------------------------------------------
# Subagent independence
# -----------------------------------------------------------------------------


class TestSubagentIndependence:
    """Each middleware instance owns its own client and composes a fresh inner."""

    def test_independent_inner_instances(self) -> None:
        m1 = CodexCompactionMiddleware(_make_model(), MockBackend())
        m2 = CodexCompactionMiddleware(_make_model(), MockBackend())
        assert m1._inner is not m2._inner
        assert m1._model is not m2._model

    async def test_state_is_per_request(self) -> None:
        """Different requests with different states don't bleed compaction items."""
        messages = make_conversation_messages(num_old=1, num_recent=1)
        middleware = CodexCompactionMiddleware(_make_model(), MockBackend())
        middleware._inner._should_summarize = MagicMock(return_value=False)

        state_a = _make_state(messages)
        state_b = _make_state(
            messages,
            compaction_item={
                "cutoff_index": 1,
                "output": [{"type": "compaction", "encrypted_content": "b-blob"}],
                "file_path": None,
            },
        )
        runtime = make_mock_runtime()

        _res_a, cap_a = await _run_awrap(middleware, state_a, runtime)
        _res_b, cap_b = await _run_awrap(middleware, state_b, runtime)

        assert cap_a is not None
        assert cap_b is not None
        # Different heads: A is a normal first message, B is the synthesized compaction message.
        head_b = cap_b.messages[0]
        assert isinstance(head_b, AIMessage)
        assert head_b.content == [{"type": "compaction", "encrypted_content": "b-blob"}]
        assert cap_a.messages[0] is not head_b
