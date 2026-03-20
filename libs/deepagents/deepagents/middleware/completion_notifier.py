"""Completion notifier middleware for async subagents.

When an async subagent finishes (success or error), this middleware sends a
message back to the **supervisor's** thread so the supervisor wakes up and can
proactively relay results to the user -- without the user having to poll via
`check_async_task`.

## Architecture

The async subagent protocol is inherently fire-and-forget: the supervisor
launches a job via `start_async_task` and only learns about completion
when someone calls `check_async_task`.

The notifier calls `runs.create()` on the supervisor's thread and assistant
ID, which queues a new run. From the supervisor's perspective, it looks like
a new user message arrived -- except the content is a structured notification
from the subagent.

## How parent context is propagated

The supervisor's `start_async_task` tool includes the parent's
`thread_id` and `assistant_id` in the subagent's input state. The notifier
reads these from the subagent's own state, which means:

- The IDs survive thread interrupts and updates (they're in state, not config)
- If the IDs are not present, the notifier silently no-ops

## Usage

Add this middleware to the subagent's middleware stack:

```python
import contextlib

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig

from deepagents.middleware.completion_notifier import CompletionNotifierMiddleware


@contextlib.asynccontextmanager
async def graph(config: RunnableConfig):
    yield create_agent(
        model=model,
        tools=[...],
        middleware=[CompletionNotifierMiddleware(subagent_name="researcher")],
    )
```

The middleware will read `parent_thread_id` and `parent_assistant_id` from
the agent's state at the end of execution. These are injected automatically
by the supervisor's `start_async_task` tool when it creates the run.
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
_PARENT_ASSISTANT_ID_KEY = "parent_assistant_id"
_TASK_ID_KEY = "task_id"


class CompletionNotifierState(AgentState):
    """State extension for subagents that use the completion notifier.

    These fields are injected by the supervisor's `launch_async_subagent`
    tool and read by `CompletionNotifierMiddleware` to send notifications
    back to the supervisor's thread.
    """

    parent_thread_id: NotRequired[str | None]
    """The supervisor's thread ID. Used to address the notification."""

    parent_assistant_id: NotRequired[str | None]
    """The supervisor's assistant ID. Used to create a run on the supervisor's thread."""

    task_id: NotRequired[str | None]
    """The task ID assigned by the supervisor.

    This is the subagent's thread ID, which the supervisor uses to track
    the job. Included in notifications so the supervisor can correlate
    completions back to specific jobs when multiple tasks of the same
    subagent type are running concurrently.
    """


async def _notify_parent(
    parent_thread_id: str,
    parent_assistant_id: str,
    notification: str,
    subagent_name: str,
) -> None:
    """Send a notification run to the parent supervisor's thread.

    Uses `get_client()` with no URL, which resolves to ASGI transport when
    running in the same LangGraph deployment. For split deployments, the
    subagent needs network access back to the supervisor.

    Args:
        parent_thread_id: The supervisor's thread ID.
        parent_assistant_id: The supervisor's assistant ID.
        notification: The message content to send.
        subagent_name: Human-readable name for logging.
    """
    from langgraph_sdk import get_client  # noqa: PLC0415  # deferred to avoid import cost at module level

    try:
        client = get_client()
        await client.runs.create(
            thread_id=parent_thread_id,
            assistant_id=parent_assistant_id,
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

    This middleware is added to the **subagent's** middleware stack (not the
    supervisor's). When the subagent finishes, it sends a message to the
    supervisor's thread via `runs.create()`, waking the supervisor so it can
    proactively relay results.

    The supervisor's thread ID and assistant ID are read from the subagent's
    own state (keys `parent_thread_id` and `parent_assistant_id`). These are
    injected by the supervisor's `launch_async_subagent` tool at launch time.

    If the parent context is not present in state (e.g., the subagent was
    launched manually without a supervisor), the middleware silently does
    nothing.

    Args:
        subagent_name: Human-readable name used in notification messages
            and logs. Helps the supervisor identify which subagent sent
            the notification.

    Example:
        ```python
        from deepagents.middleware.completion_notifier import (
            CompletionNotifierMiddleware,
        )

        notifier = CompletionNotifierMiddleware(subagent_name="researcher")

        graph = create_agent(
            model=model,
            tools=[...],
            middleware=[notifier],
        )
        ```
    """

    state_schema = CompletionNotifierState

    def __init__(self, subagent_name: str = "subagent") -> None:
        """Initialize the `CompletionNotifierMiddleware`."""
        super().__init__()
        self.subagent_name = subagent_name
        self._notified = False

    def _get_parent_ids(self, state: dict[str, Any]) -> tuple[str | None, str | None]:
        """Extract parent thread and assistant IDs from the subagent's state."""
        return (
            state.get(_PARENT_THREAD_ID_KEY),
            state.get(_PARENT_ASSISTANT_ID_KEY),
        )

    def _should_notify(self, state: dict[str, Any]) -> bool:
        """Check whether we should send a notification."""
        if self._notified:
            return False
        parent_thread_id, parent_assistant_id = self._get_parent_ids(state)
        return bool(parent_thread_id) and bool(parent_assistant_id)

    async def _send_notification(self, state: dict[str, Any], message: str) -> None:
        """Send a notification to the parent if conditions are met."""
        if not self._should_notify(state):
            return
        self._notified = True
        parent_thread_id, parent_assistant_id = self._get_parent_ids(state)
        await _notify_parent(
            parent_thread_id,  # type: ignore[arg-type]
            parent_assistant_id,  # type: ignore[arg-type]
            message,
            self.subagent_name,
        )

    def _format_notification(self, state: dict[str, Any], body: str) -> str:
        """Build a notification string with task_id and subagent name."""
        task_id = state.get(_TASK_ID_KEY)
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
        notification = self._format_notification(state, f"Completed. Result: {summary}")
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
            notification = self._format_notification(request.state, f"Error: {e!s}")
            await self._send_notification(request.state, notification)
            raise
