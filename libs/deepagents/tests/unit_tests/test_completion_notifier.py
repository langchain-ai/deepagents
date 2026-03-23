"""Tests for the CompletionCallbackMiddleware."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from deepagents.middleware.completion_callback import (
    CompletionCallbackMiddleware,
    CompletionCallbackState,
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


def _make_middleware(**kwargs: Any) -> CompletionCallbackMiddleware:
    """Create a middleware with sensible defaults."""
    kwargs.setdefault("callback_graph_id", "parent-agent")
    return CompletionCallbackMiddleware(**kwargs)


class TestExtractLastMessage:
    def test_no_messages_raises_assertion_error(self) -> None:
        with pytest.raises(AssertionError, match="Expected at least one message"):
            _extract_last_message({})

        with pytest.raises(AssertionError, match="Expected at least one message"):
            _extract_last_message({"messages": []})

    def test_dict_message_raises_type_error(self) -> None:
        state = {"messages": [{"content": "hello world"}]}
        with pytest.raises(TypeError, match="Expected an AIMessage"):
            _extract_last_message(state)

    def test_object_message_extracts_content(self) -> None:
        msg = AIMessage(content="test result")
        state = {"messages": [msg]}
        assert _extract_last_message(state) == "test result"

    def test_long_content_truncated_to_ellipsis_suffix(self) -> None:
        long_content = "x" * 1000
        state = {"messages": [AIMessage(content=long_content)]}
        result = _extract_last_message(state)
        assert result == ("x" * 500) + "... [full result truncated]"

    def test_non_ai_message_raises_type_error(self) -> None:
        msg = MagicMock()
        msg.content = ["block1", "block2"]
        state = {"messages": [msg]}
        with pytest.raises(TypeError, match="Expected an AIMessage"):
            _extract_last_message(state)

    def test_plain_value_message_raises_type_error(self) -> None:
        state = {"messages": [42]}
        with pytest.raises(TypeError, match="Expected an AIMessage"):
            _extract_last_message(state)


class TestResolveHeaders:
    def test_adds_auth_scheme_by_default(self) -> None:
        assert _resolve_headers(None) == {"x-auth-scheme": "langsmith"}

    def test_preserves_custom_headers(self) -> None:
        result = _resolve_headers({"x-custom": "value"})
        assert result == {"x-custom": "value", "x-auth-scheme": "langsmith"}

    def test_does_not_override_explicit_auth_scheme(self) -> None:
        result = _resolve_headers({"x-auth-scheme": "custom"})
        assert result == {"x-auth-scheme": "custom"}


class TestNotifyParent:
    @patch("langgraph_sdk.get_client")
    async def test_sends_run_to_parent_thread(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        await _notify_parent(
            callback_graph_id="parent-agent",
            callback_thread_id="thread-123",
            message="Job completed",
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
    async def test_passes_url_and_headers(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        await _notify_parent(
            callback_graph_id="parent-agent",
            callback_thread_id="thread-123",
            message="done",
            url="https://parent.langsmith.dev",
            headers={"x-custom": "val"},
        )

        mock_get_client.assert_called_once_with(
            url="https://parent.langsmith.dev",
            headers={"x-custom": "val", "x-auth-scheme": "langsmith"},
        )

    @patch("langgraph_sdk.get_client")
    async def test_swallows_exceptions(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.runs.create.side_effect = Exception("network error")
        mock_get_client.return_value = mock_client

        await _notify_parent(
            callback_graph_id="parent-agent",
            callback_thread_id="thread-123",
            message="Job completed",
        )


class TestCompletionCallbackMiddleware:
    def test_callback_graph_id_is_required(self) -> None:
        with pytest.raises(TypeError):
            CompletionCallbackMiddleware()  # type: ignore[call-arg]

    def test_stores_callback_graph_id(self) -> None:
        mw = _make_middleware(callback_graph_id="my-parent")
        assert mw.callback_graph_id == "my-parent"

    def test_url_defaults_to_none(self) -> None:
        mw = _make_middleware()
        assert mw.url is None

    def test_stores_url(self) -> None:
        mw = _make_middleware(url="https://parent.langsmith.dev")
        assert mw.url == "https://parent.langsmith.dev"

    def test_headers_defaults_to_none(self) -> None:
        mw = _make_middleware()
        assert mw.headers is None

    def test_stores_headers(self) -> None:
        mw = _make_middleware(headers={"x-custom": "val"})
        assert mw.headers == {"x-custom": "val"}


class TestAfterAgent:
    @patch("deepagents.middleware.completion_callback.get_config")
    @patch("deepagents.middleware.completion_callback._notify_parent", new_callable=AsyncMock)
    async def test_sends_completion_notification(self, mock_notify: AsyncMock, mock_get_config: MagicMock) -> None:
        mw = _make_middleware(callback_graph_id="parent-agent")
        state = _make_state(
            callback_thread_id="thread-123",
            messages=[AIMessage(content="Here is the result")],
        )
        runtime = MagicMock()
        mock_get_config.return_value = {"configurable": {"thread_id": "task-789"}}

        result = await mw.aafter_agent(state, runtime)

        assert result is None
        mock_notify.assert_awaited_once()
        assert mock_notify.call_args[0][0] == "parent-agent"
        assert mock_notify.call_args[0][1] == "thread-123"
        notification = mock_notify.call_args[0][2]
        assert notification == "[task_id=task-789]Completed. Result: Here is the result"

    @patch("deepagents.middleware.completion_callback.get_config")
    @patch("deepagents.middleware.completion_callback._notify_parent", new_callable=AsyncMock)
    async def test_notification_includes_task_id_from_config(self, mock_notify: AsyncMock, mock_get_config: MagicMock) -> None:
        mw = _make_middleware()
        state = _make_state(
            callback_thread_id="thread-123",
            messages=[AIMessage(content="result")],
        )
        runtime = MagicMock()
        mock_get_config.return_value = {"configurable": {"thread_id": "task-789"}}

        await mw.aafter_agent(state, runtime)

        notification = mock_notify.call_args[0][2]
        assert "[task_id=task-789]" in notification

    @patch("deepagents.middleware.completion_callback.get_config")
    @patch("deepagents.middleware.completion_callback._notify_parent", new_callable=AsyncMock)
    async def test_aafter_agent_raises_outside_runnable_context(self, mock_notify: AsyncMock, mock_get_config: MagicMock) -> None:
        mw = _make_middleware()
        state = _make_state(
            callback_thread_id="thread-123",
            messages=[AIMessage(content="result")],
        )
        runtime = MagicMock()
        mock_get_config.side_effect = RuntimeError("Called get_config outside of a runnable context")

        with pytest.raises(RuntimeError, match="Called get_config outside of a runnable context"):
            await mw.aafter_agent(state, runtime)

        mock_notify.assert_not_awaited()

    @patch("deepagents.middleware.completion_callback._notify_parent", new_callable=AsyncMock)
    async def test_aafter_agent_raises_without_callback_thread_id(self, mock_notify: AsyncMock) -> None:
        mw = _make_middleware()
        state = _make_state(messages=[AIMessage(content="result")])
        runtime = MagicMock()

        with pytest.raises(KeyError, match="callback_thread_id"):
            await mw.aafter_agent(state, runtime)

        mock_notify.assert_not_awaited()


class TestWrapModelCall:
    @patch("deepagents.middleware.completion_callback._notify_parent", new_callable=AsyncMock)
    async def test_passes_through_on_success(self, mock_notify: AsyncMock) -> None:
        mw = _make_middleware()
        mock_response = MagicMock()
        handler = AsyncMock(return_value=mock_response)

        request = MagicMock()
        request.state = _make_state(callback_thread_id="thread-123")

        result = await mw.awrap_model_call(request, handler)

        assert result is mock_response
        handler.assert_awaited_once_with(request)
        mock_notify.assert_not_awaited()

    @patch("deepagents.middleware.completion_callback.get_config")
    @patch("deepagents.middleware.completion_callback._notify_parent", new_callable=AsyncMock)
    async def test_sends_generic_error_notification_on_exception(self, mock_notify: AsyncMock, mock_get_config: MagicMock) -> None:
        mw = _make_middleware()
        handler = AsyncMock(side_effect=ValueError("model crashed"))

        request = MagicMock()
        request.state = _make_state(callback_thread_id="thread-123")
        mock_get_config.return_value = {"configurable": {"thread_id": "task-789"}}

        with pytest.raises(ValueError, match="model crashed"):
            await mw.awrap_model_call(request, handler)

        mock_notify.assert_awaited_once()
        notification = mock_notify.call_args[0][2]
        assert "The agent encountered an error while calling the model." in notification
        assert "model crashed" not in notification

    @patch("deepagents.middleware.completion_callback._notify_parent", new_callable=AsyncMock)
    async def test_no_error_notification_without_callback_thread_id(self, mock_notify: AsyncMock) -> None:
        mw = _make_middleware()
        handler = AsyncMock(side_effect=ValueError("model crashed"))

        request = MagicMock()
        request.state = _make_state()

        with pytest.raises(ValueError, match="model crashed"):
            await mw.awrap_model_call(request, handler)

        mock_notify.assert_not_awaited()

    @patch("deepagents.middleware.completion_callback.get_config")
    @patch("deepagents.middleware.completion_callback._notify_parent", new_callable=AsyncMock)
    async def test_error_notification_on_each_exception(self, mock_notify: AsyncMock, mock_get_config: MagicMock) -> None:
        mw = _make_middleware()
        handler = AsyncMock(side_effect=ValueError("fail"))

        request = MagicMock()
        request.state = _make_state(callback_thread_id="thread-123")
        mock_get_config.return_value = {"configurable": {"thread_id": "task-789"}}

        with pytest.raises(ValueError, match="fail"):
            await mw.awrap_model_call(request, handler)

        with pytest.raises(ValueError, match="fail"):
            await mw.awrap_model_call(request, handler)

        assert mock_notify.await_count == 2


class TestStateSchema:
    def test_middleware_has_state_schema(self) -> None:
        mw = _make_middleware()
        assert hasattr(mw, "state_schema")

    def test_state_schema_includes_callback_thread_id(self) -> None:
        annotations = CompletionCallbackState.__annotations__
        assert "callback_thread_id" in annotations

    def test_state_schema_does_not_include_removed_fields(self) -> None:
        annotations = CompletionCallbackState.__annotations__
        assert "parent_assistant_id" not in annotations
        assert "task_id" not in annotations
