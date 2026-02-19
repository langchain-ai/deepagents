"""Unit tests for the compact_conversation tool middleware."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from langgraph.types import Command

from deepagents_cli.compact_tool import (
    CompactToolMiddleware,
    _apply_event,
    _arun_compact,
    _build_summary_message,
    _run_compact,
)
from deepagents_cli.tool_display import format_tool_display

if TYPE_CHECKING:
    from collections.abc import Generator

# Patch targets — these are deferred imports inside the tool functions,
# so we patch at the source module where the names are looked up.
_CREATE_MODEL_PATH = "deepagents_cli.config.create_model"
_COMPUTE_DEFAULTS_PATH = (
    "deepagents.middleware.summarization._compute_summarization_defaults"
)
_LC_MIDDLEWARE_PATH = (
    "langchain.agents.middleware.summarization.SummarizationMiddleware"
)

# Patch targets for offload and summary helpers inside compact_tool
_OFFLOAD_PATH = "deepagents_cli.compact_tool._offload"
_AOFFLOAD_PATH = "deepagents_cli.compact_tool._aoffload"
_GENERATE_SUMMARY_PATH = "deepagents_cli.compact_tool._generate_summary"
_AGENERATE_SUMMARY_PATH = "deepagents_cli.compact_tool._agenerate_summary"


def _make_mock_backend() -> MagicMock:
    """Create a mock backend with standard response behavior."""
    backend = MagicMock()
    # download_files returns empty (no existing file)
    resp = MagicMock()
    resp.content = None
    resp.error = None
    backend.download_files.return_value = [resp]
    backend.adownload_files = MagicMock(return_value=[resp])
    # write succeeds
    write_result = MagicMock()
    write_result.error = None
    backend.write.return_value = write_result
    backend.awrite = MagicMock(return_value=write_result)
    return backend


def _make_messages(n: int) -> list[MagicMock]:
    """Create a list of mock messages with unique IDs."""
    messages = []
    for i in range(n):
        msg = MagicMock()
        msg.id = f"msg-{i}"
        msg.content = f"Message {i}"
        msg.additional_kwargs = {}
        messages.append(msg)
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
    # Use a real dict so .get() works naturally
    runtime.state = state
    runtime.config = {"configurable": {"thread_id": thread_id}}
    runtime.tool_call_id = tool_call_id
    return runtime


@contextmanager
def _mock_compact_deps(
    *,
    cutoff: int,
    summary: str = "Summary of the conversation.",
    file_path: str | None = "/conversation_history/test-thread.md",
) -> Generator[MagicMock, None, None]:
    """Patch create_model, defaults, LCSummarizationMiddleware, offload, and summary.

    Args:
        cutoff: Value returned by `_determine_cutoff_index`.
        summary: Text returned by the summary generator.
        file_path: Path returned by the offload helper.

    Yields:
        The mock LCSummarizationMiddleware instance.
    """
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.model = mock_model

    mock_mw = MagicMock()
    mock_mw._determine_cutoff_index.return_value = cutoff
    mock_mw._partition_messages.side_effect = lambda msgs, idx: (
        msgs[:idx],
        msgs[idx:],
    )

    with (
        patch(_CREATE_MODEL_PATH, return_value=mock_result),
        patch(
            _COMPUTE_DEFAULTS_PATH,
            return_value={"keep": ("fraction", 0.10)},
        ),
        patch(_LC_MIDDLEWARE_PATH, return_value=mock_mw),
        patch(_OFFLOAD_PATH, return_value=file_path),
        patch(_AOFFLOAD_PATH, return_value=file_path),
        patch(_GENERATE_SUMMARY_PATH, return_value=summary),
        patch(_AGENERATE_SUMMARY_PATH, return_value=summary),
    ):
        yield mock_mw


class TestToolRegistered:
    """Verify the middleware registers the compact_conversation tool."""

    def test_tool_registered(self) -> None:
        """Middleware should expose exactly one tool named `compact_conversation`."""
        backend = _make_mock_backend()
        middleware = CompactToolMiddleware(backend=backend)
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "compact_conversation"

    def test_tool_has_description(self) -> None:
        """Tool should have a non-empty description."""
        backend = _make_mock_backend()
        middleware = CompactToolMiddleware(backend=backend)
        assert middleware.tools[0].description


class TestNotEnoughMessages:
    """Test behavior when there are not enough messages to compact."""

    def test_not_enough_messages_sync(self) -> None:
        """Should return Command with 'not enough' ToolMessage when cutoff is 0."""
        backend = _make_mock_backend()
        messages = _make_messages(3)
        runtime = _make_runtime(messages)

        with _mock_compact_deps(cutoff=0):
            result = _run_compact(runtime, backend, "/conversation_history")

        assert isinstance(result, Command)
        assert result.update is not None
        update_messages = result.update["messages"]
        assert len(update_messages) == 1
        assert "Nothing to compact yet" in update_messages[0].content

    @pytest.mark.asyncio
    async def test_not_enough_messages_async(self) -> None:
        """Async: return Command with 'not enough' ToolMessage when cutoff is 0."""
        backend = _make_mock_backend()
        messages = _make_messages(3)
        runtime = _make_runtime(messages)

        with _mock_compact_deps(cutoff=0):
            result = await _arun_compact(runtime, backend, "/conversation_history")

        assert isinstance(result, Command)
        assert result.update is not None
        update_messages = result.update["messages"]
        assert "Nothing to compact yet" in update_messages[0].content


class TestCompactSuccess:
    """Test successful compaction flow."""

    def test_compact_success_no_prior_event(self) -> None:
        """Should return Command with _summarization_event and success ToolMessage."""
        backend = _make_mock_backend()
        messages = _make_messages(10)
        runtime = _make_runtime(messages)

        with _mock_compact_deps(cutoff=4):
            result = _run_compact(runtime, backend, "/conversation_history")

        assert isinstance(result, Command)
        assert result.update is not None

        # Check _summarization_event is present
        event = result.update["_summarization_event"]
        assert event["cutoff_index"] == 4
        assert event["summary_message"] is not None

        # Check ToolMessage
        update_messages = result.update["messages"]
        assert len(update_messages) == 1
        assert "Summarized 4 messages" in update_messages[0].content

    def test_compact_success_with_prior_event(self) -> None:
        """State cutoff should be old_cutoff + new_cutoff - 1 with a prior event."""
        backend = _make_mock_backend()
        messages = _make_messages(20)

        # Simulate a prior event at cutoff 5
        prior_summary = MagicMock()
        prior_event = {
            "cutoff_index": 5,
            "summary_message": prior_summary,
            "file_path": None,
        }
        runtime = _make_runtime(messages, event=prior_event)

        with _mock_compact_deps(cutoff=3):
            result = _run_compact(runtime, backend, "/conversation_history")

        assert isinstance(result, Command)
        assert result.update is not None
        event = result.update["_summarization_event"]
        # old_cutoff(5) + new_cutoff(3) - 1 = 7
        assert event["cutoff_index"] == 7

    @pytest.mark.asyncio
    async def test_compact_success_async(self) -> None:
        """Async path should produce the same Command structure."""
        backend = _make_mock_backend()
        messages = _make_messages(10)
        runtime = _make_runtime(messages)

        with _mock_compact_deps(cutoff=4):
            result = await _arun_compact(runtime, backend, "/conversation_history")

        assert isinstance(result, Command)
        assert result.update is not None
        event = result.update["_summarization_event"]
        assert event["cutoff_index"] == 4

    def test_summary_message_has_file_path(self) -> None:
        """Summary message should reference the file path when offload succeeds."""
        backend = _make_mock_backend()
        messages = _make_messages(10)
        runtime = _make_runtime(messages)

        with _mock_compact_deps(cutoff=4, file_path="/conversation_history/t.md"):
            result = _run_compact(runtime, backend, "/conversation_history")

        assert result.update is not None
        event = result.update["_summarization_event"]
        assert "/conversation_history/t.md" in event["summary_message"].content

    def test_summary_message_without_file_path(self) -> None:
        """Summary message should use plain format when offload returns None."""
        backend = _make_mock_backend()
        messages = _make_messages(10)
        runtime = _make_runtime(messages)

        with _mock_compact_deps(cutoff=4, file_path=None):
            result = _run_compact(runtime, backend, "/conversation_history")

        assert result.update is not None
        event = result.update["_summarization_event"]
        assert "saved to" not in event["summary_message"].content


class TestOffloadFailure:
    """Test that compaction still succeeds even if backend offload fails."""

    def test_offload_failure_still_succeeds(self) -> None:
        """Tool should still return a successful Command when offload fails."""
        backend = _make_mock_backend()
        messages = _make_messages(10)
        runtime = _make_runtime(messages)

        with _mock_compact_deps(cutoff=4, file_path=None):
            result = _run_compact(runtime, backend, "/conversation_history")

        assert isinstance(result, Command)
        assert result.update is not None
        event = result.update["_summarization_event"]
        assert event["file_path"] is None
        assert "Summarized 4 messages" in result.update["messages"][0].content


class TestApplyEvent:
    """Test the _apply_event helper function."""

    def test_no_event_returns_all_messages(self) -> None:
        """With no event, should return all messages as-is."""
        messages = _make_messages(5)
        result = _apply_event(messages, None)  # type: ignore[arg-type]
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
        result = _apply_event(messages, event)  # type: ignore[arg-type]
        # summary + messages[4:]  -> 1 + 6 = 7
        assert len(result) == 7
        assert result[0] is summary_msg
        assert result[1] is messages[4]


class TestBuildSummaryMessage:
    """Test the _build_summary_message helper."""

    def test_with_file_path(self) -> None:
        """Should include file path reference when provided."""
        msg = _build_summary_message("Test summary", "/history/thread.md")
        assert "/history/thread.md" in msg.content
        assert "<summary>" in msg.content
        assert "Test summary" in msg.content
        assert msg.additional_kwargs["lc_source"] == "summarization"

    def test_without_file_path(self) -> None:
        """Should use plain format when file_path is None."""
        msg = _build_summary_message("Test summary", None)
        assert "saved to" not in msg.content
        assert "Test summary" in msg.content
        assert msg.additional_kwargs["lc_source"] == "summarization"


class TestHITLGating:
    """Test that compact_conversation HITL gating respects the constant."""

    def test_hitl_gating_default_off(self) -> None:
        """compact_conversation should NOT be in interrupt_on by default."""
        from deepagents_cli.agent import _add_interrupt_on

        result = _add_interrupt_on()
        assert "compact_conversation" not in result

    def test_hitl_gating_when_enabled(self) -> None:
        """With REQUIRE_COMPACT_TOOL_APPROVAL=True, tool should be gated."""
        with patch("deepagents_cli.agent.REQUIRE_COMPACT_TOOL_APPROVAL", True):
            from deepagents_cli.agent import _add_interrupt_on

            result = _add_interrupt_on()
            assert "compact_conversation" in result


class TestDisplayFormatting:
    """Test tool display formatting for compact_conversation."""

    def test_display_formatting(self) -> None:
        """format_tool_display should return the expected string."""
        result = format_tool_display("compact_conversation", {})
        assert "compact_conversation()" in result
