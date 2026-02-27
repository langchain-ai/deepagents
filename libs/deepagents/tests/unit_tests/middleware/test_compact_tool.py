"""Unit tests for the compact_conversation tool via SummarizationToolMiddleware."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, NonCallableMagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from deepagents.middleware.summarization import SummarizationMiddleware, SummarizationToolMiddleware


def _make_mock_backend() -> MagicMock:
    """Create a mock backend with standard response behavior."""
    backend = MagicMock()
    resp = MagicMock()
    resp.content = None
    resp.error = None
    backend.download_files.return_value = [resp]
    backend.adownload_files = MagicMock(return_value=[resp])
    write_result = MagicMock()
    write_result.error = None
    backend.write.return_value = write_result
    backend.awrite = MagicMock(return_value=write_result)
    return backend


_TEST_PROVIDER = "test-provider"


def _make_mock_model() -> MagicMock:
    """Create a mock model that returns a summary response."""
    model = MagicMock()
    model.profile = {"max_input_tokens": 200_000}
    model._get_ls_params.return_value = {"ls_provider": _TEST_PROVIDER}
    response = MagicMock()
    response.text = "Summary of the conversation."
    response.content = "Summary of the conversation."
    model.invoke.return_value = response
    model.ainvoke = MagicMock(return_value=response)
    return model


def _make_messages(n: int, *, total_tokens: int = 120_000) -> list[Any]:
    """Create a list of mock messages with unique IDs.

    The last message is a real AIMessage with usage_metadata so that
    ``_is_eligible_for_compaction`` can evaluate reported token usage.
    """
    messages: list[Any] = []
    for i in range(n - 1):
        msg = MagicMock()
        msg.id = f"msg-{i}"
        msg.content = f"Message {i}"
        msg.additional_kwargs = {}
        messages.append(msg)
    messages.append(
        AIMessage(
            content=f"Message {n - 1}",
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": total_tokens},
            response_metadata={"model_provider": _TEST_PROVIDER},
        )
    )
    return messages


def _make_runtime(
    messages: list[Any],
    *,
    event: dict[str, Any] | None = None,
    thread_id: str = "test-thread",
    tool_call_id: str = "tc-1",
) -> MagicMock:
    """Create a mock ToolRuntime with a real dict for state."""
    runtime = MagicMock()
    state: dict[str, Any] = {"messages": messages}
    if event is not None:
        state["_summarization_event"] = event
    runtime.state = state
    runtime.config = {"configurable": {"thread_id": thread_id}}
    runtime.tool_call_id = tool_call_id
    return runtime


def _make_summarization_middleware(
    model: MagicMock | None = None,
    backend: MagicMock | None = None,
) -> SummarizationMiddleware:
    """Create a bare SummarizationMiddleware (no compact tool)."""
    if model is None:
        model = _make_mock_model()
    if backend is None:
        backend = _make_mock_backend()
    return SummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=("fraction", 0.85),
    )


def _make_middleware(
    model: MagicMock | None = None,
    backend: MagicMock | None = None,
) -> SummarizationToolMiddleware:
    """Create a SummarizationToolMiddleware wrapping a SummarizationMiddleware."""
    summ = _make_summarization_middleware(model, backend)
    return SummarizationToolMiddleware(summ)


class TestToolRegistered:
    """Verify the middleware registers the compact_conversation tool."""

    def test_tool_registered(self) -> None:
        """Middleware should expose exactly one tool named `compact_conversation`."""
        mw = _make_middleware()
        assert len(mw.tools) == 1
        assert mw.tools[0].name == "compact_conversation"

    def test_tool_has_description(self) -> None:
        """Tool should have a non-empty description."""
        mw = _make_middleware()
        assert mw.tools[0].description


class TestApplyEventToMessages:
    """Test the _apply_event_to_messages static method."""

    def test_no_event_returns_all(self) -> None:
        """With no event, should return all messages as-is."""
        messages = _make_messages(5)
        result = SummarizationMiddleware._apply_event_to_messages(messages, None)
        assert len(result) == 5

    def test_with_event_returns_summary_plus_kept(self) -> None:
        """Should return summary message + messages from cutoff onward."""
        messages = _make_messages(10)
        summary_msg = MagicMock()
        event = {
            "cutoff_index": 4,
            "summary_message": summary_msg,
            "file_path": None,
        }
        result = SummarizationMiddleware._apply_event_to_messages(messages, event)
        # summary + messages[4:] -> 1 + 6 = 7
        assert len(result) == 7
        assert result[0] is summary_msg
        assert result[1] is messages[4]


class TestNotEnoughMessages:
    """Test behavior when there are not enough messages to compact."""

    def test_not_enough_messages_sync(self) -> None:
        """Should return Command with a 'nothing to compact' ToolMessage when cutoff is 0."""
        mw = _make_middleware()
        messages = _make_messages(3)
        runtime = _make_runtime(messages)

        with patch.object(mw._summarization, "_determine_cutoff_index", return_value=0):
            result = mw._run_compact(runtime)

        assert isinstance(result, Command)
        assert result.update is not None
        update_messages = result.update["messages"]
        assert len(update_messages) == 1
        assert "Nothing to compact yet" in update_messages[0].content

    @pytest.mark.asyncio
    async def test_not_enough_messages_async(self) -> None:
        """Async: return Command with a 'nothing to compact' ToolMessage when cutoff is 0."""
        mw = _make_middleware()
        messages = _make_messages(3)
        runtime = _make_runtime(messages)

        with patch.object(mw._summarization, "_determine_cutoff_index", return_value=0):
            result = await mw._arun_compact(runtime)

        assert isinstance(result, Command)
        assert result.update is not None
        update_messages = result.update["messages"]
        assert "Nothing to compact yet" in update_messages[0].content


class TestCompactSuccess:
    """Test successful compaction flow."""

    def test_compact_success_no_prior_event(self) -> None:
        """Should return Command with _summarization_event and success ToolMessage."""
        mw = _make_middleware()
        messages = _make_messages(10)
        runtime = _make_runtime(messages)

        with (
            patch.object(mw._summarization, "_determine_cutoff_index", return_value=4),
            patch.object(
                mw._summarization,
                "_partition_messages",
                side_effect=lambda msgs, idx: (msgs[:idx], msgs[idx:]),
            ),
            patch.object(mw._summarization, "_offload_to_backend", return_value="/conversation_history/test-thread.md"),
            patch.object(mw._summarization, "_create_summary", return_value="Summary of the conversation."),
        ):
            result = mw._run_compact(runtime)

        assert isinstance(result, Command)
        assert result.update is not None
        event = result.update["_summarization_event"]
        assert event["cutoff_index"] == 4
        assert event["summary_message"] is not None

        update_messages = result.update["messages"]
        assert len(update_messages) == 1
        assert "Summarized 4 messages" in update_messages[0].content

    def test_compact_success_with_prior_event(self) -> None:
        """State cutoff should be old_cutoff + new_cutoff - 1 with a prior event."""
        mw = _make_middleware()
        messages = _make_messages(20)

        prior_summary = MagicMock()
        prior_event = {
            "cutoff_index": 5,
            "summary_message": prior_summary,
            "file_path": None,
        }
        runtime = _make_runtime(messages, event=prior_event)

        with (
            patch.object(mw._summarization, "_determine_cutoff_index", return_value=3),
            patch.object(
                mw._summarization,
                "_partition_messages",
                side_effect=lambda msgs, idx: (msgs[:idx], msgs[idx:]),
            ),
            patch.object(mw._summarization, "_offload_to_backend", return_value=None),
            patch.object(mw._summarization, "_create_summary", return_value="Summary."),
        ):
            result = mw._run_compact(runtime)

        assert isinstance(result, Command)
        assert result.update is not None
        event = result.update["_summarization_event"]
        # old_cutoff(5) + new_cutoff(3) - 1 = 7
        assert event["cutoff_index"] == 7

    @pytest.mark.asyncio
    async def test_compact_success_async(self) -> None:
        """Async path should produce the same Command structure."""
        mw = _make_middleware()
        messages = _make_messages(10)
        runtime = _make_runtime(messages)

        with (
            patch.object(mw._summarization, "_determine_cutoff_index", return_value=4),
            patch.object(
                mw._summarization,
                "_partition_messages",
                side_effect=lambda msgs, idx: (msgs[:idx], msgs[idx:]),
            ),
            patch.object(mw._summarization, "_aoffload_to_backend", return_value="/conversation_history/test-thread.md"),
            patch.object(mw._summarization, "_acreate_summary", return_value="Summary of the conversation."),
        ):
            result = await mw._arun_compact(runtime)

        assert isinstance(result, Command)
        assert result.update is not None
        event = result.update["_summarization_event"]
        assert event["cutoff_index"] == 4

    def test_summary_message_has_file_path(self) -> None:
        """Summary message should reference the file path when offload succeeds."""
        mw = _make_middleware()
        messages = _make_messages(10)
        runtime = _make_runtime(messages)

        with (
            patch.object(mw._summarization, "_determine_cutoff_index", return_value=4),
            patch.object(
                mw._summarization,
                "_partition_messages",
                side_effect=lambda msgs, idx: (msgs[:idx], msgs[idx:]),
            ),
            patch.object(mw._summarization, "_offload_to_backend", return_value="/conversation_history/t.md"),
            patch.object(mw._summarization, "_create_summary", return_value="Summary."),
        ):
            result = mw._run_compact(runtime)

        assert result.update is not None
        event = result.update["_summarization_event"]
        assert "/conversation_history/t.md" in event["summary_message"].content

    def test_summary_message_without_file_path(self) -> None:
        """Summary message should use plain format when offload returns None."""
        mw = _make_middleware()
        messages = _make_messages(10)
        runtime = _make_runtime(messages)

        with (
            patch.object(mw._summarization, "_determine_cutoff_index", return_value=4),
            patch.object(
                mw._summarization,
                "_partition_messages",
                side_effect=lambda msgs, idx: (msgs[:idx], msgs[idx:]),
            ),
            patch.object(mw._summarization, "_offload_to_backend", return_value=None),
            patch.object(mw._summarization, "_create_summary", return_value="Summary."),
        ):
            result = mw._run_compact(runtime)

        assert result.update is not None
        event = result.update["_summarization_event"]
        assert "saved to" not in event["summary_message"].content


class TestOffloadFailure:
    """Test that compaction still succeeds even if backend offload fails."""

    def test_offload_failure_still_succeeds(self) -> None:
        """Tool should still return a successful Command when offload fails."""
        mw = _make_middleware()
        messages = _make_messages(10)
        runtime = _make_runtime(messages)

        with (
            patch.object(mw._summarization, "_determine_cutoff_index", return_value=4),
            patch.object(
                mw._summarization,
                "_partition_messages",
                side_effect=lambda msgs, idx: (msgs[:idx], msgs[idx:]),
            ),
            patch.object(mw._summarization, "_offload_to_backend", return_value=None),
            patch.object(mw._summarization, "_create_summary", return_value="Summary."),
        ):
            result = mw._run_compact(runtime)

        assert isinstance(result, Command)
        assert result.update is not None
        event = result.update["_summarization_event"]
        assert event["file_path"] is None
        assert "Summarized 4 messages" in result.update["messages"][0].content


class TestCompactErrorHandling:
    """Test that compact gracefully handles errors during summarization."""

    def test_sync_summary_failure_returns_error_tool_message(self) -> None:
        """Sync: LLM failure should return an error ToolMessage, not raise."""
        mw = _make_middleware()
        messages = _make_messages(10)
        runtime = _make_runtime(messages)

        with (
            patch.object(mw._summarization, "_determine_cutoff_index", return_value=4),
            patch.object(
                mw._summarization,
                "_partition_messages",
                side_effect=lambda msgs, idx: (msgs[:idx], msgs[idx:]),
            ),
            patch.object(mw._summarization, "_offload_to_backend", return_value=None),
            patch.object(mw._summarization, "_create_summary", side_effect=RuntimeError("model unavailable")),
        ):
            result = mw._run_compact(runtime)

        assert isinstance(result, Command)
        assert result.update is not None
        msg = result.update["messages"][0]
        assert "Compaction failed" in msg.content
        assert "model unavailable" in msg.content
        # Should NOT have a _summarization_event (state not modified)
        assert "_summarization_event" not in result.update

    @pytest.mark.asyncio
    async def test_async_summary_failure_returns_error_tool_message(self) -> None:
        """Async: LLM failure should return an error ToolMessage, not raise."""
        mw = _make_middleware()
        messages = _make_messages(10)
        runtime = _make_runtime(messages)

        with (
            patch.object(mw._summarization, "_determine_cutoff_index", return_value=4),
            patch.object(
                mw._summarization,
                "_partition_messages",
                side_effect=lambda msgs, idx: (msgs[:idx], msgs[idx:]),
            ),
            patch.object(mw._summarization, "_aoffload_to_backend", return_value=None),
            patch.object(
                mw._summarization,
                "_acreate_summary",
                side_effect=RuntimeError("model unavailable"),
            ),
        ):
            result = await mw._arun_compact(runtime)

        assert isinstance(result, Command)
        assert result.update is not None
        msg = result.update["messages"][0]
        assert "Compaction failed" in msg.content
        assert "_summarization_event" not in result.update

    def test_backend_resolve_failure_returns_error_tool_message(self) -> None:
        """Backend factory failure should return an error ToolMessage."""
        mw = _make_middleware()
        messages = _make_messages(10)
        runtime = _make_runtime(messages)

        with (
            patch.object(mw._summarization, "_determine_cutoff_index", return_value=4),
            patch.object(
                mw._summarization,
                "_partition_messages",
                side_effect=lambda msgs, idx: (msgs[:idx], msgs[idx:]),
            ),
            patch.object(mw._summarization, "_create_summary", return_value="Summary."),
            patch.object(
                mw,
                "_resolve_backend",
                side_effect=ConnectionError("sandbox unreachable"),
            ),
        ):
            result = mw._run_compact(runtime)

        assert isinstance(result, Command)
        assert result.update is not None
        msg = result.update["messages"][0]
        assert "Compaction failed" in msg.content
        assert "_summarization_event" not in result.update


class TestMalformedEvent:
    """Test handling of corrupted _summarization_event state."""

    def test_malformed_event_falls_back_to_raw_messages(self) -> None:
        """Should fall back to raw messages when event keys are missing."""
        messages = _make_messages(5)
        # Event missing required keys
        bad_event: dict[str, Any] = {"unexpected_key": 42}
        result = SummarizationMiddleware._apply_event_to_messages(messages, bad_event)
        assert result == messages

    def test_none_event_returns_message_copy(self) -> None:
        """Should return a copy of messages when event is None."""
        messages = _make_messages(5)
        result = SummarizationMiddleware._apply_event_to_messages(messages, None)
        assert result == messages
        assert result is not messages

    def test_cutoff_exceeds_message_count(self) -> None:
        """Should return only summary when cutoff_index > len(messages)."""
        messages = _make_messages(3)
        summary_msg = MagicMock()
        event = {
            "cutoff_index": 10,
            "summary_message": summary_msg,
            "file_path": None,
        }
        result = SummarizationMiddleware._apply_event_to_messages(messages, event)
        assert len(result) == 1
        assert result[0] is summary_msg


class TestResolveBackend:
    """Test backend resolution for tool context."""

    def test_static_backend(self) -> None:
        """Should return the backend directly when it's not callable."""
        backend = NonCallableMagicMock()
        summ = SummarizationMiddleware(
            model=_make_mock_model(),
            backend=backend,
        )
        mw = SummarizationToolMiddleware(summ)
        runtime = _make_runtime(_make_messages(1))
        assert mw._resolve_backend(runtime) is backend

    def test_callable_backend(self) -> None:
        """Should call the factory with the ToolRuntime."""
        resolved = _make_mock_backend()
        factory = MagicMock(return_value=resolved)
        summ = SummarizationMiddleware(
            model=_make_mock_model(),
            backend=factory,
        )
        mw = SummarizationToolMiddleware(summ)
        runtime = _make_runtime(_make_messages(1))
        result = mw._resolve_backend(runtime)
        assert result is resolved
        factory.assert_called_once_with(runtime)


class TestComputeStateCutoff:
    """Tests for _compute_state_cutoff arithmetic."""

    def test_no_event_returns_effective_cutoff(self) -> None:
        """With no prior event, should return effective_cutoff as-is."""
        assert SummarizationMiddleware._compute_state_cutoff(None, 0) == 0
        assert SummarizationMiddleware._compute_state_cutoff(None, 5) == 5

    def test_with_event_applies_offset(self) -> None:
        """Should return old_cutoff + effective_cutoff - 1."""
        event: dict[str, Any] = {
            "cutoff_index": 10,
            "summary_message": MagicMock(),
            "file_path": None,
        }
        # old(10) + new(1) - 1 = 10
        assert SummarizationMiddleware._compute_state_cutoff(event, 1) == 10

    def test_with_zero_old_cutoff(self) -> None:
        """Edge case: old cutoff of 0."""
        event: dict[str, Any] = {
            "cutoff_index": 0,
            "summary_message": MagicMock(),
            "file_path": None,
        }
        # old(0) + new(3) - 1 = 2
        assert SummarizationMiddleware._compute_state_cutoff(event, 3) == 2

    def test_malformed_event_missing_cutoff(self) -> None:
        """Should fall back to effective_cutoff when cutoff_index is missing."""
        bad_event: dict[str, Any] = {"summary_message": MagicMock()}
        assert SummarizationMiddleware._compute_state_cutoff(bad_event, 4) == 4

    def test_malformed_event_non_int_cutoff(self) -> None:
        """Should fall back to effective_cutoff when cutoff_index is not an int."""
        bad_event: dict[str, Any] = {
            "cutoff_index": "five",
            "summary_message": MagicMock(),
        }
        assert SummarizationMiddleware._compute_state_cutoff(bad_event, 4) == 4


def _ai_message_with_usage(total_tokens: int, provider: str = "test-provider") -> AIMessage:
    """Create an AIMessage with usage_metadata and response_metadata."""
    return AIMessage(
        content="response",
        usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": total_tokens},
        response_metadata={"model_provider": provider},
    )


def _make_middleware_with_trigger(
    trigger: Any,  # noqa: ANN401
    provider: str = "test-provider",
) -> SummarizationToolMiddleware:
    model = _make_mock_model()
    model._get_ls_params.return_value = {"ls_provider": provider}
    summ = SummarizationMiddleware(
        model=model,
        backend=_make_mock_backend(),
        trigger=trigger,
    )
    return SummarizationToolMiddleware(summ)


class TestIsEligibleForCompaction:
    """Test the _is_eligible_for_compaction early-exit in compact."""

    def test_under_50pct_tokens_trigger_returns_nothing(self) -> None:
        """Under 50% of a tokens trigger → not eligible → nothing to compact."""
        mw = _make_middleware_with_trigger(("tokens", 100_000))
        messages = [HumanMessage(content="hi"), _ai_message_with_usage(40_000)]
        runtime = _make_runtime(messages)
        result = mw._run_compact(runtime)
        assert "Nothing to compact" in result.update["messages"][0].content

    def test_over_50pct_tokens_trigger_proceeds(self) -> None:
        """Over 50% of a tokens trigger → eligible → compaction proceeds."""
        mw = _make_middleware_with_trigger(("tokens", 100_000))
        messages = [HumanMessage(content="hi"), _ai_message_with_usage(60_000)]
        runtime = _make_runtime(messages)
        with (
            patch.object(mw._summarization, "_determine_cutoff_index", return_value=1),
            patch.object(mw._summarization, "_partition_messages", side_effect=lambda m, i: (m[:i], m[i:])),
            patch.object(mw._summarization, "_create_summary", return_value="Summary."),
            patch.object(mw._summarization, "_offload_to_backend", return_value=None),
        ):
            result = mw._run_compact(runtime)
        assert "_summarization_event" in result.update

    def test_under_50pct_fraction_trigger_returns_nothing(self) -> None:
        """Under 50% of a fraction trigger → not eligible."""
        mw = _make_middleware_with_trigger(("fraction", 0.8))
        messages = [HumanMessage(content="hi"), _ai_message_with_usage(50_000)]
        runtime = _make_runtime(messages)
        result = mw._run_compact(runtime)
        assert "Nothing to compact" in result.update["messages"][0].content

    def test_over_50pct_fraction_trigger_proceeds(self) -> None:
        """Over 50% of a fraction trigger → eligible."""
        mw = _make_middleware_with_trigger(("fraction", 0.8))
        messages = [HumanMessage(content="hi"), _ai_message_with_usage(100_000)]
        runtime = _make_runtime(messages)
        with (
            patch.object(mw._summarization, "_determine_cutoff_index", return_value=1),
            patch.object(mw._summarization, "_partition_messages", side_effect=lambda m, i: (m[:i], m[i:])),
            patch.object(mw._summarization, "_create_summary", return_value="Summary."),
            patch.object(mw._summarization, "_offload_to_backend", return_value=None),
        ):
            result = mw._run_compact(runtime)
        assert "_summarization_event" in result.update

    def test_no_usage_metadata_falls_through(self) -> None:
        """No usage metadata → not eligible → falls through to cutoff check."""
        mw = _make_middleware_with_trigger(("tokens", 100_000))
        messages = _make_messages(30)
        runtime = _make_runtime(messages)
        with patch.object(mw._summarization, "_determine_cutoff_index", return_value=0):
            result = mw._run_compact(runtime)
        assert "Nothing to compact" in result.update["messages"][0].content

    @pytest.mark.asyncio
    async def test_under_50pct_async(self) -> None:
        """Async path: under 50% → nothing to compact."""
        mw = _make_middleware_with_trigger(("tokens", 100_000))
        messages = [HumanMessage(content="hi"), _ai_message_with_usage(40_000)]
        runtime = _make_runtime(messages)
        result = await mw._arun_compact(runtime)
        assert "Nothing to compact" in result.update["messages"][0].content
