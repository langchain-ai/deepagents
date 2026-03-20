"""Completion notifier middleware for async subagents.

!!! warning "Experimental"
    This middleware is experimental and may change in future releases.

When an async subagent finishes (success or error), this middleware sends a
message back to the **supervisor's** thread so the supervisor wakes up and can
proactively relay results to the user -- without the user having to poll via
`check_async_task`.

## Architecture

The async subagent protocol is inherently fire-and-forget: the supervisor
launches a job via `start_async_task` and only learns about completion
when someone calls `check_async_task`. This middleware closes that gap.

```
Supervisor                    Subagent
    |                            |
    |--- start_async_task -----> |
    |<-- task_id (immediately) - |
    |                            |  (working...)
    |                            |  (done!)
    |                            |
    |<-- runs.create(            |
    |      supervisor_thread,    |
    |      "completed: ...")     |
    |                            |
    |  (wakes up, sees result)   |
```

The notifier calls `runs.create()` on the supervisor's thread, which
queues a new run. From the supervisor's perspective, it looks like a new
user message arrived -- except the content is a structured notification
from the subagent.

## How parent context is propagated

- `parent_graph_id` is passed as a **constructor argument** to the middleware.
  This is the supervisor's graph ID (or assistant ID), which the subagent
  developer knows at configuration time.
- `parent_thread_id` is injected into the subagent's input state by the
  supervisor's `start_async_task` tool. It survives thread interrupts and
  updates because it lives in state, not config.
- If `parent_thread_id` is not present in state, the notifier silently no-ops.

## Usage

Add this middleware to the subagent's middleware stack:

```python
from deepagents.middleware.completion_notifier import CompletionNotifierMiddleware

notifier = CompletionNotifierMiddleware(
    parent_graph_id="supervisor",
    subagent_name="researcher",
)

graph = create_agent(
    model=model,
    tools=[...],
    middleware=[notifier],
)
```

The middleware will read `parent_thread_id` from the agent's state at the
end of execution. This is injected automatically by the supervisor's
`start_async_task` tool when it creates the run.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NotRequired

from langchain.agents.middleware.types import AgentMiddleware, AgentState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ContextT, ModelRequest, ModelResponse, ResponseT, Runtime


# State keys where the supervisor's launch tool stores parent context.
_PARENT_THREAD_ID_KEY = "parent_thread_id"


class CompletionNotifierState(AgentState):
    """State extension for subagents that use the completion notifier.

    !!! warning "Experimental"
        This state schema is experimental and may change in future releases.

    These fields are injected by the supervisor's `start_async_task`
    tool and read by `CompletionNotifierMiddleware` to send notifications
    back to the supervisor's thread.
    """

    parent_thread_id: NotRequired[str | None]
    """The supervisor's thread ID. Used to address the notification."""


async def _notify_parent(
    parent_thread_id: str,
    parent_graph_id: str,
    notification: str,
    subagent_name: str,
) -> None:
    """Send a notification run to the parent supervisor's thread.

    Uses `get_client()` with no URL, which resolves to ASGI transport when
    running in the same LangGraph deployment. For split deployments, the
    subagent needs network access back to the supervisor.

    Args:
        parent_thread_id: The supervisor's thread ID.
        parent_graph_id: The supervisor's graph ID (used as `assistant_id`
            in the `runs.create` call).
        notification: The message content to send.
        subagent_name: Human-readable name for logging.
    """
    from langgraph_sdk import get_client  # noqa: PLC0415  # deferred to avoid import cost at module level

    try:
        client = get_client()
        await client.runs.create(
            thread_id=parent_thread_id,
            assistant_id=parent_graph_id,
            input={
                "messages": [{"role": "user", "content": notification}],
            },
        )
        logger.info(
            "Notified parent thread %s that subagent '%s' finished",
            parent_thread_id,
            subagent_name,
        )
    except Exception:  # noqa: BLE001  # LangGraph SDK raises untyped errors
        logger.warning(
            "Failed to notify parent thread %s",
            parent_thread_id,
            exc_info=True,
        )


def _extract_last_message(state: dict[str, Any]) -> str:
    """Extract a summary from the subagent's final message.

    Returns at most 500 characters from the last message's content.
    """
    messages = state.get("messages", [])
    if not messages:
        return "(no output)"
    last = messages[-1]
    if hasattr(last, "content"):
        content = last.content
        return content[:500] if isinstance(content, str) else str(content)[:500]
    if isinstance(last, dict):
        return str(last.get("content", ""))[:500]
    return str(last)[:500]


class CompletionNotifierMiddleware(AgentMiddleware):
    """Notifies the supervisor when an async subagent completes or errors.

    !!! warning "Experimental"
        This middleware is experimental and may change in future releases.

    This middleware is added to the **subagent's** middleware stack (not the
    supervisor's). When the subagent finishes, it sends a message to the
    supervisor's thread via `runs.create()`, waking the supervisor so it can
    proactively relay results.

    The supervisor's `parent_thread_id` is read from the subagent's own state
    (injected by the supervisor's `start_async_task` tool at launch time).
    The `parent_graph_id` is provided as a constructor argument since it's
    static configuration known at deployment time.

    If `parent_thread_id` is not present in state (e.g., the subagent was
    launched manually without a supervisor), the middleware silently does
    nothing.

    Args:
        parent_graph_id: The supervisor's graph ID (or assistant ID). Used
            as the `assistant_id` parameter when calling `runs.create()` to
            send notifications back to the supervisor.
        subagent_name: Human-readable name used in notification messages
            and logs. Helps the supervisor identify which subagent sent
            the notification.

    Example:
        ```python
        from deepagents.middleware.completion_notifier import (
            CompletionNotifierMiddleware,
        )

        notifier = CompletionNotifierMiddleware(
            parent_graph_id="supervisor",
            subagent_name="researcher",
        )

        graph = create_agent(
            model=model,
            tools=[...],
            middleware=[notifier],
        )
        ```
    """

    state_schema = CompletionNotifierState

    def __init__(self, parent_graph_id: str, subagent_name: str = "subagent") -> None:
        """Initialize the `CompletionNotifierMiddleware`."""
        super().__init__()
        self.parent_graph_id = parent_graph_id
        self.subagent_name = subagent_name
        self._notified = False

    def _should_notify(self, state: dict[str, Any]) -> bool:
        """Check whether we should send a notification."""
        if self._notified:
            return False
        return bool(state.get(_PARENT_THREAD_ID_KEY))

    async def _send_notification(self, state: dict[str, Any], message: str) -> None:
        """Send a notification to the parent if conditions are met."""
        if not self._should_notify(state):
            return
        self._notified = True
        await _notify_parent(
            state[_PARENT_THREAD_ID_KEY],
            self.parent_graph_id,
            message,
            self.subagent_name,
        )

    @staticmethod
    def _get_task_id() -> str | None:
        """Read the subagent's own thread_id from config.

        The subagent's `thread_id` is the same as the `task_id` from the
        supervisor's perspective. Available via `get_config()` at runtime.
        """
        try:
            from langgraph.config import get_config  # noqa: PLC0415

            config = get_config()
            return (config.get("configurable") or {}).get("thread_id")
        except RuntimeError:
            return None

    def _format_notification(self, body: str) -> str:
        """Build a notification string with task_id and subagent name."""
        task_id = self._get_task_id()
        prefix = f"[task_id={task_id}]" if task_id else ""
        return f"{prefix}[subagent={self.subagent_name}] {body}"

    async def aafter_agent(
        self,
        state: dict[str, Any],
        runtime: Runtime,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """After-agent hook: fires when the subagent completes successfully.

        Extracts the last message as a summary and sends it to the supervisor.
        """
        summary = _extract_last_message(state)
        notification = self._format_notification(f"Completed. Result: {summary}")
        await self._send_notification(state, notification)
        return None

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Wrap model calls to catch errors and notify the supervisor.

        If a model call raises an exception, the error is reported to the
        supervisor before re-raising so the supervisor can inform the user.
        """
        try:
            return await handler(request)
        except Exception as e:
            notification = self._format_notification(f"Error: {e!s}")
            await self._send_notification(request.state, notification)
            raise
