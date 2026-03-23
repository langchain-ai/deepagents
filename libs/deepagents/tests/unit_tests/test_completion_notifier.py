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
    _resolve_headers,
)


def _make_state(
    *,
    callback_thread_id: str | None = None,
    messages: list | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    if messages is not None:
        state["messages"] = messages
    if callback_thread_id is not None:
        state["callback_thread_id"] = callback_thread_id
    return state


def _make_middleware(**kwargs: Any) -> CompletionNotifierMiddleware:
    """Create a middleware with sensible defaults."""
    kwargs.setdefault("callback_graph_id", "parent-agent")
    return CompletionNotifierMiddleware(**kwargs)


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


class TestResolveHeaders:
    def test_adds_auth_scheme_by_default(self):
        assert _resolve_headers(None) == {"x-auth-scheme": "langsmith"}

    def test_preserves_custom_headers(self):
        result = _resolve_headers({"x-custom": "value"})
        assert result == {"x-custom": "value", "x-auth-scheme": "langsmith"}

    def test_does_not_override_explicit_auth_scheme(self):
        result = _resolve_headers({"x-auth-scheme": "custom"})
        assert result == {"x-auth-scheme": "custom"}


class TestNotifyParent:
    @patch("langgraph_sdk.get_client")
    async def test_sends_run_to_parent_thread(self, mock_get_client):
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        await _notify_parent(
            callback_graph_id="parent-agent",
            callback_thread_id="thread-123",
            notification="Job completed",
        )

        mock_get_client.assert_called_once_with(url=None, headers={"x-auth-scheme": "langsmith"})
        mock_client.runs.create.assert_awaited_once_with(
            thread_id="thread-123",
            assistant_id="parent-agent",
            input={
                "messages": [{"role": "user", "content": "Job completed"}],
            },
        )

    @patch("langgraph_sdk.get_client")
    async def test_passes_url_and_headers(self, mock_get_client):
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        await _notify_parent(
            callback_graph_id="parent-agent",
            callback_thread_id="thread-123",
            notification="done",
            url="https://parent.langsmith.dev",
            headers={"x-custom": "val"},
        )

        mock_get_client.assert_called_once_with(
            url="https://parent.langsmith.dev",
            headers={"x-custom": "val", "x-auth-scheme": "langsmith"},
        )

    @patch("langgraph_sdk.get_client")
    async def test_swallows_exceptions(self, mock_get_client):
        mock_client = AsyncMock()
        mock_client.runs.create.side_effect = Exception("network error")
        mock_get_client.return_value = mock_client

        # Should not raise
        await _notify_parent(
            callback_graph_id="parent-agent",
            callback_thread_id="thread-123",
            notification="Job completed",
        )


class TestCompletionNotifierMiddleware:
    def test_callback_graph_id_is_required(self):
        with pytest.raises(TypeError):
            CompletionNotifierMiddleware()  # type: ignore[call-arg]

    def test_stores_callback_graph_id(self):
        mw = _make_middleware(callback_graph_id="my-parent")
        assert mw.callback_graph_id == "my-parent"

    def test_url_defaults_to_none(self):
        mw = _make_middleware()
        assert mw.url is None

    def test_stores_url(self):
        mw = _make_middleware(url="https://parent.langsmith.dev")
        assert mw.url == "https://parent.langsmith.dev"

    def test_headers_defaults_to_none(self):
        mw = _make_middleware()
        assert mw.headers is None

    def test_stores_headers(self):
        mw = _make_middleware(headers={"x-custom": "val"})
        assert mw.headers == {"x-custom": "val"}

    def test_should_notify_false_when_no_callback_thread_id(self):
        mw = _make_middleware()
        assert not mw._should_notify({})

    def test_should_notify_true_when_callback_thread_id_present(self):
        mw = _make_middleware()
        state = _make_state(callback_thread_id="thread-123")
        assert mw._should_notify(state)

    def test_should_notify_false_after_already_notified(self):
        mw = _make_middleware()
        mw._notified = True
        state = _make_state(callback_thread_id="thread-123")
        assert not mw._should_notify(state)


class TestAfterAgent:
    @patch("deepagents.middleware.completion_notifier._notify_parent", new_callable=AsyncMock)
    async def test_sends_completion_notification(self, mock_notify):
        mw = _make_middleware(callback_graph_id="parent-agent")
        state = _make_state(
            callback_thread_id="thread-123",
            messages=[AIMessage(content="Here is the result")],
        )
        runtime = MagicMock()

        result = await mw.aafter_agent(state, runtime)

        assert result is None
        mock_notify.assert_awaited_once()
        assert mock_notify.call_args[0][0] == "parent-agent"
        assert mock_notify.call_args[0][1] == "thread-123"
        notification = mock_notify.call_args[0][2]
        assert "Here is the result" in notification

    @patch("deepagents.middleware.completion_notifier._notify_parent", new_callable=AsyncMock)
    async def test_notification_includes_task_id_from_config(self, mock_notify):
        mw = _make_middleware()
        state = _make_state(
            callback_thread_id="thread-123",
            messages=[AIMessage(content="result")],
        )
        runtime = MagicMock()

        with patch(
            "deepagents.middleware.completion_notifier.CompletionNotifierMiddleware._get_task_id",
            return_value="task-789",
        ):
            await mw.aafter_agent(state, runtime)

        notification = mock_notify.call_args[0][2]
        assert "[task_id=task-789]" in notification

    @patch("deepagents.middleware.completion_notifier._notify_parent", new_callable=AsyncMock)
    async def test_notification_without_task_id_omits_prefix(self, mock_notify):
        mw = _make_middleware()
        state = _make_state(
            callback_thread_id="thread-123",
            messages=[AIMessage(content="result")],
        )
        runtime = MagicMock()

        with patch(
            "deepagents.middleware.completion_notifier.CompletionNotifierMiddleware._get_task_id",
            return_value=None,
        ):
            await mw.aafter_agent(state, runtime)

        notification = mock_notify.call_args[0][2]
        assert "[task_id=" not in notification

    @patch("deepagents.middleware.completion_notifier._notify_parent", new_callable=AsyncMock)
    async def test_no_notification_without_callback_thread_id(self, mock_notify):
        mw = _make_middleware()
        state = _make_state(messages=[AIMessage(content="result")])
        runtime = MagicMock()

        await mw.aafter_agent(state, runtime)

        mock_notify.assert_not_awaited()

    @patch("deepagents.middleware.completion_notifier._notify_parent", new_callable=AsyncMock)
    async def test_notifies_only_once(self, mock_notify):
        mw = _make_middleware()
        state = _make_state(
            callback_thread_id="thread-123",
            messages=[AIMessage(content="result")],
        )
        runtime = MagicMock()

        await mw.aafter_agent(state, runtime)
        await mw.aafter_agent(state, runtime)

        assert mock_notify.await_count == 1


class TestWrapModelCall:
    @patch("deepagents.middleware.completion_notifier._notify_parent", new_callable=AsyncMock)
    async def test_passes_through_on_success(self, mock_notify):
        mw = _make_middleware()
        mock_response = MagicMock()
        handler = AsyncMock(return_value=mock_response)

        request = MagicMock()
        request.state = _make_state(callback_thread_id="thread-123")

        result = await mw.awrap_model_call(request, handler)

        assert result is mock_response
        handler.assert_awaited_once_with(request)
        mock_notify.assert_not_awaited()

    @patch("deepagents.middleware.completion_notifier._notify_parent", new_callable=AsyncMock)
    async def test_sends_error_notification_on_exception(self, mock_notify):
        mw = _make_middleware()
        handler = AsyncMock(side_effect=ValueError("model crashed"))

        request = MagicMock()
        request.state = _make_state(callback_thread_id="thread-123")

        with pytest.raises(ValueError, match="model crashed"):
            await mw.awrap_model_call(request, handler)

        mock_notify.assert_awaited_once()
        notification = mock_notify.call_args[0][2]
        assert "model crashed" in notification

    @patch("deepagents.middleware.completion_notifier._notify_parent", new_callable=AsyncMock)
    async def test_no_error_notification_without_callback_thread_id(self, mock_notify):
        mw = _make_middleware()
        handler = AsyncMock(side_effect=ValueError("model crashed"))

        request = MagicMock()
        request.state = _make_state()

        with pytest.raises(ValueError, match="model crashed"):
            await mw.awrap_model_call(request, handler)

        mock_notify.assert_not_awaited()

    @patch("deepagents.middleware.completion_notifier._notify_parent", new_callable=AsyncMock)
    async def test_error_notification_only_once(self, mock_notify):
        """If wrap_model_call is retried externally, error notification fires only once."""
        mw = _make_middleware()
        handler = AsyncMock(side_effect=ValueError("fail"))

        request = MagicMock()
        request.state = _make_state(callback_thread_id="thread-123")

        with pytest.raises(ValueError, match="fail"):
            await mw.awrap_model_call(request, handler)

        with pytest.raises(ValueError, match="fail"):
            await mw.awrap_model_call(request, handler)

        assert mock_notify.await_count == 1


class TestGetTaskId:
    def test_returns_thread_id_from_config(self):
        mw = _make_middleware()
        with patch(
            "langgraph.config.get_config",
            return_value={"configurable": {"thread_id": "thread-abc"}},
        ):
            assert mw._get_task_id() == "thread-abc"

    def test_returns_none_outside_runnable_context(self):
        mw = _make_middleware()
        # get_config raises RuntimeError outside a runnable context
        assert mw._get_task_id() is None

    def test_returns_none_when_no_thread_id(self):
        mw = _make_middleware()
        with patch(
            "langgraph.config.get_config",
            return_value={"configurable": {}},
        ):
            assert mw._get_task_id() is None


class TestStateSchema:
    def test_middleware_has_state_schema(self):
        mw = _make_middleware()
        assert hasattr(mw, "state_schema")

    def test_state_schema_includes_callback_thread_id(self):
        annotations = CompletionNotifierState.__annotations__
        assert "callback_thread_id" in annotations

    def test_state_schema_does_not_include_removed_fields(self):
        annotations = CompletionNotifierState.__annotations__
        assert "parent_assistant_id" not in annotations
        assert "task_id" not in annotations
