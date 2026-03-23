"""Completion notifier middleware for async subagents.

!!! warning "Experimental"
    This middleware is experimental and may change in future releases.

When an async subagent finishes (success or error), this middleware sends a
message back to the callback thread so the callback agent wakes up and can
proactively relay results to the user -- without the user having to poll via
`check_async_task`.

## Architecture

The async subagent protocol is inherently fire-and-forget: the parent agent
launches a job via `start_async_task` and only learns about completion
when someone calls `check_async_task`. This middleware closes that gap.

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
    |  (wakes up, sees result)   |
```

The notifier calls `runs.create()` on the callback thread, which
queues a new run. From the callback agent's perspective, it looks like a new
user message arrived -- except the content is a structured notification
from the subagent.

## How callback context is propagated

- `callback_graph_id` is passed as a **constructor argument** to the middleware.
  This is the parent graph ID (or assistant ID), which the subagent
  developer knows at configuration time.
- `url` and `headers` are optional constructor arguments for reaching the
  callback destination on a remote deployment. Omit `url` for same-deployment
  ASGI transport.
- `callback_thread_id` is injected into the subagent's input state by the
  parent's `start_async_task` tool. It survives thread interrupts and
  updates because it lives in state, not config.
- If `callback_thread_id` is not present in state, the notifier silently no-ops.

## Usage

Add this middleware to the subagent's middleware stack:

```python
from deepagents.middleware.completion_notifier import CompletionNotifierMiddleware

# Same deployment (ASGI transport -- callback agent and subagent share a server):
notifier = CompletionNotifierMiddleware(callback_graph_id="supervisor")

# Remote deployment (callback destination on a different server):
notifier = CompletionNotifierMiddleware(
    callback_graph_id="supervisor",
    url="url to your langsmith deployment",
)

graph = create_agent(
    model=model,
    tools=[...],
    middleware=[notifier],
)
```

The middleware will read `callback_thread_id` from the agent's state at the
end of execution. This is injected automatically by the parent's
`start_async_task` tool when it creates the run.
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


# State key where the launch tool stores callback context.
_CALLBACK_THREAD_ID_KEY = "callback_thread_id"


class CompletionNotifierState(AgentState):
    """State extension for subagents that use the completion notifier.

    !!! warning "Experimental"
        This state schema is experimental and may change in future releases.

    These fields are injected by the parent's `start_async_task`
    tool and read by `CompletionNotifierMiddleware` to send notifications
    back to the parent's thread.
    """

    callback_thread_id: NotRequired[str | None]
    """The callback thread ID. Used to address the notification."""


def _resolve_headers(headers: dict[str, str] | None) -> dict[str, str]:
    """Build headers for the parent's LangGraph server.

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


_MAX_MESSAGE_LENGTH = 500  # max characters to include in notification summary


def _extract_last_message(state: dict[str, Any]) -> str:
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
        text_content = text_content[:_MAX_MESSAGE_LENGTH] + "... [full result truncated]"
    return text_content


class CompletionNotifierMiddleware(AgentMiddleware[CompletionNotifierState, ContextT, ResponseT]):
    """Notifies another agent when work is complete.

    !!! warning "Experimental"
        This middleware is experimental and may change in future releases.

    This middleware is added to a subagent's middleware stack. When the
    subagent finishes (success or error), it sends a message to the callback
    thread via `runs.create()`, waking the callback agent so it can
    proactively relay results.

    The `callback_thread_id` is read from the subagent's own state
    (injected by the callback agent's `start_async_task` tool at launch time).
    The `callback_graph_id` is provided as a constructor argument since it's
    static configuration known at deployment time.

    If `callback_thread_id` is not present in state, the middleware silently
    does nothing.

    Args:
        callback_graph_id: The callback graph ID (or assistant ID). Used
            as the `assistant_id` parameter when calling `runs.create()` to
            send notifications to the callback destination.
        url: URL of the callback LangGraph server (e.g.,
            `"https://my-deployment.langsmith.dev"`). Omit to use ASGI
            transport for same-deployment communication.
        headers: Additional headers to include in requests to the
            callback server.

    Example:
        ```python
        from deepagents.middleware.completion_notifier import CompletionNotifierMiddleware

        # Same deployment (ASGI transport):
        notifier = CompletionNotifierMiddleware(callback_graph_id="supervisor")

        # Remote deployment:
        notifier = CompletionNotifierMiddleware(
            callback_graph_id="supervisor",
            url="https://supervisor.langsmith.dev",
        )

        graph = create_agent(
            model=model,
            tools=[...],
            middleware=[notifier],
        )
        ```
    """

    state_schema = CompletionNotifierState

    def __init__(
        self,
        callback_graph_id: str,
        *,
        url: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the `CompletionNotifierMiddleware`."""
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
        """After-agent hook: fires when the subagent completes successfully.

        Extracts the last message as a summary and sends it to the callback destination.
        """
        state_dict = cast("dict[str, Any]", state)
        callback_thread_id = state_dict[_CALLBACK_THREAD_ID_KEY]
        summary = _extract_last_message(state_dict)
        notification = self._format_notification(f"Completed. Result: {summary}")
        await self._send_notification(callback_thread_id, notification)
        return None

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Wrap model calls to catch errors and notify the callback destination.

        If a model call raises an exception, the error is reported before
        re-raising so the callback agent can inform the user.
        """
        try:
            return await handler(request)
        except Exception:
            callback_thread_id = request.state.get(_CALLBACK_THREAD_ID_KEY)
            if isinstance(callback_thread_id, str):
                notification = self._format_notification("The agent encountered an error while calling the model.")
                await self._send_notification(callback_thread_id, notification)
            raise
