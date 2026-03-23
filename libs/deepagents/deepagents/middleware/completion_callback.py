"""Callback middleware for async subagents.

!!! warning "Experimental"
    This middleware is experimental and may change in future releases.

This middleware sends a notification to a callback thread when a subagent
completes successfully or raises an error. The callback agent can then
process that notification instead of relying only on polling via
`check_async_task`.

## Architecture

A parent agent launches a subagent with `start_async_task` and can later
inspect task state with `check_async_task`. This middleware adds an optional
completion signal by creating a run on the callback thread when the subagent
finishes.

```
Parent                        Subagent
    |                            |
    |--- start_async_task -----> |
    |<-- task_id (immediately) - |
    |                            |  (working...)
    |                            |  (done!)
    |                            |
    |<-- runs.create(            |
    |      callback_thread,      |
    |      "completed: ...")     |
    |                            |
    |  (processes result)        |
```

The middleware calls `runs.create()` on the callback thread. From the
callback agent's perspective, this appears as a new user message containing
structured output from the subagent.

## Callback context

- `callback_graph_id` identifies the callback graph or assistant. It is
  provided when the middleware is constructed.
- `url` and `headers` optionally configure a remote callback destination.
  Omit `url` for same-deployment ASGI transport.
- `callback_thread_id` is stored in the subagent state by the parent's
  `start_async_task` tool. Because it is stored in state rather than config,
  it survives thread updates and interrupts.
- If `callback_thread_id` is not present in state, the middleware does
  nothing.

## Usage

Add this middleware to the subagent's middleware stack:

```python
from deepagents.middleware.completion_callback import CompletionCallbackMiddleware

# Same deployment (ASGI transport -- callback agent and subagent share a server):
notifier = CompletionCallbackMiddleware(callback_graph_id="supervisor")

# Remote deployment (callback destination on a different server):
notifier = CompletionCallbackMiddleware(
    callback_graph_id="supervisor",
    url="url to your langsmith deployment",
)

graph = create_agent(
    model=model,
    tools=[...],
    middleware=[notifier],
)
```

The middleware reads `callback_thread_id` from the agent state at the end of
execution. This value is injected by the parent's `start_async_task` tool when
it creates the run.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NotRequired, cast

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ResponseT, StateT
from langchain.messages import AIMessage
from langgraph.config import get_config

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse, Runtime

logger = logging.getLogger(__name__)


_CALLBACK_THREAD_ID_KEY = "callback_thread_id"


class CompletionCallbackState(AgentState):
    """State extension for subagents that use completion callbacks.

    !!! warning "Experimental"
        This state schema is experimental and may change in future releases.

    `callback_thread_id` is written by the parent's `start_async_task` tool
    and read by `CompletionCallbackMiddleware` when sending callback
    notifications.
    """

    callback_thread_id: NotRequired[str | None]
    """The callback thread ID. Used to address the notification."""


def _resolve_headers(headers: dict[str, str] | None) -> dict[str, str]:
    """Build headers for the callback LangGraph server.

    Ensures `x-auth-scheme: langsmith` is present unless explicitly overridden.
    """
    resolved: dict[str, str] = dict(headers or {})
    if "x-auth-scheme" not in resolved:
        resolved["x-auth-scheme"] = "langsmith"
    return resolved


async def _notify_parent(
    callback_graph_id: str,
    callback_thread_id: str,
    message: str,
    *,
    url: str | None = None,
    headers: dict[str, str] | None = None,
) -> None:
    """Send a notification run to the callback thread.

    Args:
        callback_graph_id: The callback graph ID used as `assistant_id`
            in the `runs.create` call.
        callback_thread_id: The callback thread ID.
        message: The message content to send.
        url: URL of the callback LangGraph server. Omit for ASGI
            transport (same deployment).
        headers: Additional headers for the request.
    """
    from langgraph_sdk import get_client  # noqa: PLC0415  # deferred to avoid import cost at module level

    try:
        client = get_client(url=url, headers=_resolve_headers(headers))
        await client.runs.create(
            thread_id=callback_thread_id,
            assistant_id=callback_graph_id,
            input={
                "messages": [{"role": "user", "content": message}],
            },
        )
        logger.info(
            "Notified callback thread %s via graph '%s'",
            callback_thread_id,
            callback_graph_id,
        )
    except Exception:  # noqa: BLE001  # LangGraph SDK raises untyped errors
        logger.warning(
            "Failed to notify callback thread %s",
            callback_thread_id,
            exc_info=True,
        )


_MAX_MESSAGE_LENGTH = 500
_TRUNCATION_SUFFIX = "... [full result truncated]"


def _extract_last_message(state: dict[str, Any], *, task_id: str | None = None) -> str:
    """Extract a summary from the subagent's final message.

    Returns at most 500 characters from the last message's content.
    """
    messages = state.get("messages", [])
    if not messages:
        msg = f"Expected at least one message in state {state}"
        raise AssertionError(msg)
    last = messages[-1]

    if not isinstance(last, AIMessage):
        msg = f"Expected an AIMessage, got {type(last)} instead"
        raise TypeError(msg)

    text_content = last.text
    if len(text_content) > _MAX_MESSAGE_LENGTH:
        text_content = text_content[:_MAX_MESSAGE_LENGTH] + _TRUNCATION_SUFFIX
        if task_id:
            text_content += f" Result truncated. Use `check_async_task(task_id={task_id!r})` to retrieve the full result if needed."
    return text_content


class CompletionCallbackMiddleware(AgentMiddleware[CompletionCallbackState, ContextT, ResponseT]):
    """Send callback notifications when a subagent finishes.

    !!! warning "Experimental"
        This middleware is experimental and may change in future releases.

    This middleware is added to a subagent's middleware stack. On success or
    model-call error, it sends a notification to the configured callback
    thread by calling `runs.create()`.

    The callback destination is configured with `callback_graph_id` and
    optional `url` and `headers`. The target thread is read from
    `callback_thread_id` in the subagent state.

    If `callback_thread_id` is not present in state, the middleware does
    nothing.

    Args:
        callback_graph_id: Callback graph or assistant identifier used as the
            `assistant_id` argument in `runs.create()`.
        url: URL of the callback LangGraph server. Omit to use same-deployment
            ASGI transport.
        headers: Additional headers to include in requests to the callback
            server.

    Example:
        ```python
        from deepagents.middleware.completion_callback import CompletionCallbackMiddleware

        notifier = CompletionCallbackMiddleware(callback_graph_id="supervisor")

        graph = create_agent(
            model=model,
            tools=[...],
            middleware=[notifier],
        )
        ```
    """

    state_schema = CompletionCallbackState

    def __init__(
        self,
        callback_graph_id: str,
        *,
        url: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the `CompletionCallbackMiddleware`."""
        super().__init__()
        self.callback_graph_id = callback_graph_id
        self.url = url
        self.headers = headers

    async def _send_notification(self, callback_thread_id: str, message: str) -> None:
        """Send a notification to the callback destination if conditions are met."""
        await _notify_parent(
            self.callback_graph_id,
            callback_thread_id,
            message,
            url=self.url,
            headers=self.headers,
        )

    def _format_notification(self, body: str) -> str:
        """Build a notification string with task_id and subagent name."""
        config = get_config()
        thread_id = (config.get("configurable") or {}).get("thread_id")
        if not isinstance(thread_id, str):
            msg = f"Expected `thread_id` to be str, got {type(thread_id)}"
            raise TypeError(msg)
        prefix = f"[task_id={thread_id}]" if thread_id else ""
        return f"{prefix}{body}"

    async def aafter_agent(
        self,
        state: StateT,
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """After-agent hook for successful subagent completion."""
        state_dict = cast("dict[str, Any]", state)
        callback_thread_id = state_dict[_CALLBACK_THREAD_ID_KEY]
        config = get_config()
        task_id = (config.get("configurable") or {}).get("thread_id")
        summary = _extract_last_message(state_dict, task_id=task_id if isinstance(task_id, str) else None)
        notification = self._format_notification(f"Completed. Result: {summary}")
        await self._send_notification(callback_thread_id, notification)
        return None

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Wrap model calls to send callback notifications on model-call errors."""
        try:
            return await handler(request)
        except Exception:
            callback_thread_id = request.state.get(_CALLBACK_THREAD_ID_KEY)
            if isinstance(callback_thread_id, str):
                notification = self._format_notification("The agent encountered an error while calling the model.")
                await self._send_notification(callback_thread_id, notification)
            raise
