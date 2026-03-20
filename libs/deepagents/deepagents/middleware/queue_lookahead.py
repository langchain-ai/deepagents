"""Middleware that consumes pending queued messages before each model call.

When deployed on LangGraph Platform, users can send messages while the agent is
already running (double-texting). By default these are enqueued as separate
pending runs. This middleware runs as a `before_model` hook -- before each LLM
call it checks for pending runs on the current thread, extracts their input
messages, cancels the pending runs, and returns them as a state update so they
are persisted (checkpointed) before the model sees them.

When running inside the LangGraph Platform server, the SDK client automatically
uses in-process ASGI transport (`httpx.ASGITransport`) to talk to the server's
ASGI app directly -- no HTTP round-trips leave the process.

Usage::

    from deepagents.middleware.queue_lookahead import QueueLookaheadMiddleware

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-20250514",
        middleware=[QueueLookaheadMiddleware()],
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ResponseT
from langchain_core.messages import HumanMessage
from langgraph.config import get_config

if TYPE_CHECKING:
    from langgraph.runtime import Runtime
    from langgraph_sdk.client import LangGraphClient
    from langgraph_sdk.schema import CancelAction

logger = logging.getLogger(__name__)


def _extract_messages_from_run(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract input messages from a pending run's kwargs.

    Args:
        run: A run dict returned by the LangGraph SDK.

    Returns:
        List of message dicts from the run's input, or empty list if none found.
    """
    kwargs = run.get("kwargs", {})
    run_input = kwargs.get("input", {})
    if isinstance(run_input, dict):
        return run_input.get("messages", [])
    return []


def _convert_to_human_messages(
    raw_messages: list[dict[str, Any]],
) -> list[HumanMessage]:
    """Convert raw message dicts from pending runs into HumanMessage objects.

    Only includes messages with role "user" or "human". Other message types
    from pending runs are dropped since injecting AI or system messages
    mid-conversation would be confusing.

    Args:
        raw_messages: List of message dicts (e.g., `{"role": "user", "content": "..."}`).

    Returns:
        List of HumanMessage objects.
    """
    result: list[HumanMessage] = []
    for msg in raw_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "human") and content:
            result.append(HumanMessage(content=content))
    return result


def _get_thread_id() -> str | None:
    """Extract thread_id from the current LangGraph config context.

    Returns:
        The thread_id string, or None if not available.
    """
    try:
        config = get_config()
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id is not None:
            return str(thread_id)
    except RuntimeError:
        pass
    return None


def _get_default_client() -> LangGraphClient:
    """Create a LangGraph SDK client using in-process ASGI transport.

    When running inside LangGraph Platform, `get_client()` with no URL
    uses `httpx.ASGITransport` to call the server's ASGI app directly,
    avoiding any network round-trips. Outside the platform, it falls back
    to deferred transport registration.

    Returns:
        An async LangGraph SDK client (`LangGraphClient`).
    """
    from langgraph_sdk import get_client  # noqa: PLC0415  # deferred to avoid import before server is ready

    return get_client()


class QueueLookaheadMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Consume pending queued messages and persist them before each model call.

    When the agent is deployed on LangGraph Platform with `multitask_strategy="enqueue"`,
    user messages sent during an active run are queued as pending runs on the
    thread. This middleware uses `abefore_model` to check for those pending runs
    before each LLM invocation, extract the user messages, cancel the pending
    runs, and return them as a state update. Because `before_model` writes to
    state, the injected messages are checkpointed and survive crashes.

    When running inside the same deployment, the SDK client uses in-process
    ASGI transport -- no HTTP overhead.
    """

    def __init__(
        self,
        *,
        client: LangGraphClient | None = None,
        cancel_action: CancelAction = "interrupt",
    ) -> None:
        """Initialize the queue lookahead middleware.

        Args:
            client: An async LangGraph SDK client (`LangGraphClient`). If not
                provided, one is created via `get_client()` which auto-detects
                in-process ASGI transport when running on LangGraph Platform.
            cancel_action: How to cancel consumed pending runs.
        """
        self._client = client
        self._cancel_action = cancel_action

    @property
    def _resolved_client(self) -> LangGraphClient:
        """Lazily resolve the SDK client.

        Deferred so the import and ASGI transport binding happen at runtime
        (when the server is ready) rather than at middleware construction time.

        Returns:
            The async LangGraph SDK client.
        """
        if self._client is None:
            self._client = _get_default_client()
        return self._client

    async def abefore_model(
        self,
        state: AgentState,  # noqa: ARG002
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Drain pending queued messages and persist them to state.

        Runs before each model call in the agent loop. Checks the LangGraph
        server for pending runs on the current thread, extracts user messages,
        cancels the pending runs, and returns a state update that appends the
        messages via the `add_messages` reducer. This ensures the messages are
        checkpointed before the model call.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            State update with pending messages appended, or None if no
            pending messages found.
        """
        thread_id = _get_thread_id()
        if thread_id is None:
            logger.debug("No thread_id found, skipping queue lookahead")
            return None

        pending_messages = await self._drain_pending(thread_id)
        if not pending_messages:
            return None

        logger.info(
            "Injecting %d pending message(s) into state",
            len(pending_messages),
        )
        return {"messages": pending_messages}

    async def _drain_pending(self, thread_id: str) -> list[HumanMessage]:
        """Fetch and cancel all pending runs, returning their user messages.

        Args:
            thread_id: The current thread ID.

        Returns:
            List of HumanMessage objects extracted from pending runs.
        """
        client = self._resolved_client
        try:
            pending_runs = await client.runs.list(
                thread_id=thread_id,
                status="pending",
            )
        except Exception:  # noqa: BLE001  # SDK client can raise varied HTTP/connection errors
            logger.warning("Failed to list pending runs", exc_info=True)
            return []

        if not pending_runs:
            return []

        messages: list[HumanMessage] = []
        for run in pending_runs:
            raw = _extract_messages_from_run(run)
            messages.extend(_convert_to_human_messages(raw))

            try:
                await client.runs.cancel(
                    thread_id=thread_id,
                    run_id=run["run_id"],
                    action=self._cancel_action,
                )
            except Exception:  # noqa: BLE001  # SDK client can raise varied HTTP/connection errors
                logger.warning(
                    "Failed to cancel pending run %s",
                    run.get("run_id"),
                    exc_info=True,
                )

        return messages
