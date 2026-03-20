"""Tests for the CompletionNotifierMiddleware."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from deepagents.middleware.completion_notifier import (
    CompletionNotifierMiddleware,
    CompletionNotifierState,
    _extract_last_message,
    _notify_parent,
)


def _make_state(
    *,
    parent_thread_id: str | None = None,
    parent_assistant_id: str | None = None,
    task_id: str | None = None,
    messages: list | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    if messages is not None:
        state["messages"] = messages
    if parent_thread_id is not None:
        state["parent_thread_id"] = parent_thread_id
    if parent_assistant_id is not None:
        state["parent_assistant_id"] = parent_assistant_id
    if task_id is not None:
        state["task_id"] = task_id
    return state


class TestExtractLastMessage:
    def test_no_messages_returns_no_output(self):
        assert _extract_last_message({}) == "(no output)"
        assert _extract_last_message({"messages": []}) == "(no output)"

    def test_dict_message_extracts_content(self):
        state = {"messages": [{"content": "hello world"}]}
        assert _extract_last_message(state) == "hello world"

    def test_object_message_extracts_content(self):
        msg = AIMessage(content="test result")
        state = {"messages": [msg]}
        assert _extract_last_message(state) == "test result"

    def test_long_content_truncated_to_500_chars(self):
        long_content = "x" * 1000
        state = {"messages": [{"content": long_content}]}
        result = _extract_last_message(state)
        assert len(result) == 500

    def test_non_string_content_converted_to_string(self):
        msg = MagicMock()
        msg.content = ["block1", "block2"]
        state = {"messages": [msg]}
        result = _extract_last_message(state)
        assert "block1" in result

    def test_plain_value_message_converted_to_string(self):
        state = {"messages": [42]}
        assert _extract_last_message(state) == "42"


class TestNotifyParent:
    @patch("langgraph_sdk.get_client")
    async def test_sends_run_to_parent_thread(self, mock_get_client):
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        await _notify_parent(
            parent_thread_id="thread-123",
            parent_assistant_id="asst-456",
            notification="Job completed",
            subagent_name="researcher",
        )

        mock_client.runs.create.assert_awaited_once_with(
            thread_id="thread-123",
            assistant_id="asst-456",
            input={
                "messages": [{"role": "user", "content": "Job completed"}],
            },
        )

    @patch("langgraph_sdk.get_client")
    async def test_swallows_exceptions(self, mock_get_client):
        mock_client = AsyncMock()
        mock_client.runs.create.side_effect = Exception("network error")
        mock_get_client.return_value = mock_client

        # Should not raise
        await _notify_parent(
            parent_thread_id="thread-123",
            parent_assistant_id="asst-456",
            notification="Job completed",
            subagent_name="researcher",
        )


class TestCompletionNotifierMiddleware:
    def test_default_subagent_name(self):
        mw = CompletionNotifierMiddleware()
        assert mw.subagent_name == "subagent"

    def test_custom_subagent_name(self):
        mw = CompletionNotifierMiddleware(subagent_name="researcher")
        assert mw.subagent_name == "researcher"

    def test_should_notify_false_when_no_parent_ids(self):
        mw = CompletionNotifierMiddleware()
        assert not mw._should_notify({})

    def test_should_notify_false_when_only_thread_id(self):
        mw = CompletionNotifierMiddleware()
        state = _make_state(parent_thread_id="thread-123")
        assert not mw._should_notify(state)

    def test_should_notify_false_when_only_assistant_id(self):
        mw = CompletionNotifierMiddleware()
        state = _make_state(parent_assistant_id="asst-456")
        assert not mw._should_notify(state)

    def test_should_notify_true_when_both_present(self):
        mw = CompletionNotifierMiddleware()
        state = _make_state(
            parent_thread_id="thread-123",
            parent_assistant_id="asst-456",
        )
        assert mw._should_notify(state)

    def test_should_notify_false_after_already_notified(self):
        mw = CompletionNotifierMiddleware()
        mw._notified = True
        state = _make_state(
            parent_thread_id="thread-123",
            parent_assistant_id="asst-456",
        )
        assert not mw._should_notify(state)


class TestAfterAgent:
    @patch("deepagents.middleware.completion_notifier._notify_parent", new_callable=AsyncMock)
    async def test_sends_completion_notification_with_task_id(self, mock_notify):
        mw = CompletionNotifierMiddleware(subagent_name="coder")
        state = _make_state(
            parent_thread_id="thread-123",
            parent_assistant_id="asst-456",
            task_id="task-789",
            messages=[AIMessage(content="Here is the result")],
        )
        runtime = MagicMock()

        result = await mw.aafter_agent(state, runtime)

        assert result is None
        mock_notify.assert_awaited_once()
        notification = mock_notify.call_args[0][2]
        assert "[task_id=task-789]" in notification
        assert "[subagent=coder]" in notification
        assert "Here is the result" in notification

    @patch("deepagents.middleware.completion_notifier._notify_parent", new_callable=AsyncMock)
    async def test_notification_without_task_id_omits_prefix(self, mock_notify):
        mw = CompletionNotifierMiddleware(subagent_name="coder")
        state = _make_state(
            parent_thread_id="thread-123",
            parent_assistant_id="asst-456",
            messages=[AIMessage(content="result")],
        )
        runtime = MagicMock()

        await mw.aafter_agent(state, runtime)

        notification = mock_notify.call_args[0][2]
        assert "[task_id=" not in notification
        assert "[subagent=coder]" in notification

    @patch("deepagents.middleware.completion_notifier._notify_parent", new_callable=AsyncMock)
    async def test_no_notification_without_parent_ids(self, mock_notify):
        mw = CompletionNotifierMiddleware(subagent_name="coder")
        state = _make_state(messages=[AIMessage(content="result")])
        runtime = MagicMock()

        await mw.aafter_agent(state, runtime)

        mock_notify.assert_not_awaited()

    @patch("deepagents.middleware.completion_notifier._notify_parent", new_callable=AsyncMock)
    async def test_notifies_only_once(self, mock_notify):
        mw = CompletionNotifierMiddleware(subagent_name="coder")
        state = _make_state(
            parent_thread_id="thread-123",
            parent_assistant_id="asst-456",
            messages=[AIMessage(content="result")],
        )
        runtime = MagicMock()

        await mw.aafter_agent(state, runtime)
        await mw.aafter_agent(state, runtime)

        assert mock_notify.await_count == 1


class TestWrapModelCall:
    @patch("deepagents.middleware.completion_notifier._notify_parent", new_callable=AsyncMock)
    async def test_passes_through_on_success(self, mock_notify):
        mw = CompletionNotifierMiddleware(subagent_name="coder")
        mock_response = MagicMock()
        handler = AsyncMock(return_value=mock_response)

        request = MagicMock()
        request.state = _make_state(
            parent_thread_id="thread-123",
            parent_assistant_id="asst-456",
        )

        result = await mw.awrap_model_call(request, handler)

        assert result is mock_response
        handler.assert_awaited_once_with(request)
        mock_notify.assert_not_awaited()

    @patch("deepagents.middleware.completion_notifier._notify_parent", new_callable=AsyncMock)
    async def test_sends_error_notification_on_exception(self, mock_notify):
        mw = CompletionNotifierMiddleware(subagent_name="coder")
        handler = AsyncMock(side_effect=ValueError("model crashed"))

        request = MagicMock()
        request.state = _make_state(
            parent_thread_id="thread-123",
            parent_assistant_id="asst-456",
            task_id="task-789",
        )

        with pytest.raises(ValueError, match="model crashed"):
            await mw.awrap_model_call(request, handler)

        mock_notify.assert_awaited_once()
        notification = mock_notify.call_args[0][2]
        assert "[task_id=task-789]" in notification
        assert "[subagent=coder]" in notification
        assert "model crashed" in notification

    @patch("deepagents.middleware.completion_notifier._notify_parent", new_callable=AsyncMock)
    async def test_no_error_notification_without_parent_ids(self, mock_notify):
        mw = CompletionNotifierMiddleware(subagent_name="coder")
        handler = AsyncMock(side_effect=ValueError("model crashed"))

        request = MagicMock()
        request.state = _make_state()

        with pytest.raises(ValueError, match="model crashed"):
            await mw.awrap_model_call(request, handler)

        mock_notify.assert_not_awaited()

    @patch("deepagents.middleware.completion_notifier._notify_parent", new_callable=AsyncMock)
    async def test_error_notification_only_once(self, mock_notify):
        """If wrap_model_call is retried externally, error notification fires only once."""
        mw = CompletionNotifierMiddleware(subagent_name="coder")
        handler = AsyncMock(side_effect=ValueError("fail"))

        request = MagicMock()
        request.state = _make_state(
            parent_thread_id="thread-123",
            parent_assistant_id="asst-456",
        )

        with pytest.raises(ValueError, match="fail"):
            await mw.awrap_model_call(request, handler)

        with pytest.raises(ValueError, match="fail"):
            await mw.awrap_model_call(request, handler)

        # Only one notification even though two errors
        assert mock_notify.await_count == 1


class TestStateSchema:
    def test_middleware_has_state_schema(self):
        mw = CompletionNotifierMiddleware()
        assert hasattr(mw, "state_schema")

    def test_state_schema_includes_parent_fields(self):
        annotations = CompletionNotifierState.__annotations__
        assert "parent_thread_id" in annotations
        assert "parent_assistant_id" in annotations
        assert "task_id" in annotations
