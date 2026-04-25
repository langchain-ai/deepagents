"""Middleware to patch dangling tool calls in the messages history."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.config import get_config
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command, Overwrite

logger = logging.getLogger(__name__)


class PatchToolCallsMiddleware(AgentMiddleware):
    """Middleware to patch dangling tool calls in the messages history."""

    def __init__(self) -> None:
        """Initialize the middleware."""
        super().__init__()
        self._in_flight: set[tuple[str | None, str]] = set()

    def _get_thread_id(self) -> str | None:
        """Get the current thread ID from langgraph config.

        Returns:
            The thread ID if in a runnable context, else None.
        """
        try:
            return get_config().get("configurable", {}).get("thread_id")
        except RuntimeError:
            return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Wrap the tool call to track its in-flight execution."""
        tool_call_id = request.tool_call.get("id")
        thread_id = self._get_thread_id()
        if tool_call_id:
            self._in_flight.add((thread_id, tool_call_id))

        try:
            return handler(request)
        finally:
            if tool_call_id:
                self._in_flight.discard((thread_id, tool_call_id))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Wrap the tool call to track its in-flight execution."""
        tool_call_id = request.tool_call.get("id")
        thread_id = self._get_thread_id()
        if tool_call_id:
            self._in_flight.add((thread_id, tool_call_id))

        try:
            return await handler(request)
        finally:
            if tool_call_id:
                self._in_flight.discard((thread_id, tool_call_id))

    def before_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Before the agent runs, handle dangling tool calls from any AIMessage."""
        return self._process_state(state)

    async def abefore_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Async hook before agent runs. Waits for in-flight tools to complete cancellation."""
        thread_id = self._get_thread_id()
        in_flight_for_thread = {tid_tcid[1] for tid_tcid in self._in_flight if tid_tcid[0] == thread_id}

        if in_flight_for_thread:
            logger.info(
                "PatchToolCallsMiddleware: Holding before_agent to allow %d in-flight tool calls (thread=%s) to complete or cancel.",
                len(in_flight_for_thread),
                thread_id,
            )
            # Wait up to a short period for tasks to exit their finally blocks
            for _ in range(20):
                in_flight_for_thread = {tid_tcid[1] for tid_tcid in self._in_flight if tid_tcid[0] == thread_id}
                if not in_flight_for_thread:
                    break
                await asyncio.sleep(0.05)
            if in_flight_for_thread:
                logger.warning(
                    "PatchToolCallsMiddleware: Some tool calls (thread=%s) remain in-flight after waiting: %s",
                    thread_id,
                    in_flight_for_thread,
                )

        return self._process_state(state, thread_id)

    def _process_state(self, state: AgentState, thread_id: str | None = None) -> dict[str, Any] | None:
        """Core logic to handle dangling tool calls.

        Args:
            state: The current agent state containing messages.
            thread_id: Optional thread ID to isolate tool tracking.

        Returns:
            A state update dictionary with patched messages, or None if no patching needed.
        """
        messages = state["messages"]
        if not messages:
            return None

        in_flight_for_thread = {tid_tcid[1] for tid_tcid in self._in_flight if tid_tcid[0] == thread_id}
        answered_ids = {msg.tool_call_id for msg in messages if isinstance(msg, ToolMessage)}

        dangling_found = any(
            tool_call["id"] not in answered_ids and tool_call["id"] not in in_flight_for_thread
            for msg in messages
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None)
            for tool_call in msg.tool_calls
        )

        if not dangling_found:
            return None

        logger.info("PatchToolCallsMiddleware: Found dangling tool calls for thread %s. Injecting cancellation ToolMessage.", thread_id)

        patched_messages = []
        for msg in messages:
            patched_messages.append(msg)
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tool_call in msg.tool_calls:
                    if tool_call["id"] not in answered_ids and tool_call["id"] not in in_flight_for_thread:
                        logger.info(
                            "PatchToolCallsMiddleware: Cancelling tool call %s (%s) for thread %s",
                            tool_call["name"],
                            tool_call["id"],
                            thread_id,
                        )
                        patched_messages.append(
                            ToolMessage(
                                content=(
                                    f"Tool call {tool_call['name']} with id {tool_call['id']} was "
                                    "cancelled - another message came in before it could be completed."
                                ),
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )

        return {"messages": Overwrite(patched_messages)}
