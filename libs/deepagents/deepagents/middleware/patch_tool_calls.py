"""Middleware to patch dangling tool calls in the messages history.

This middleware serves two purposes:

1. **Dangling call patching**: When a tool call is in the message history with no
   corresponding ``ToolMessage`` (because the agent loop was interrupted mid-flight),
   it injects a synthetic cancellation ``ToolMessage`` so the LLM receives a coherent
   conversation history on the next turn.

2. **MCP transport protection**: As noted by m13v on issue #2471, slow tools like
   ``playwright_browser_navigate`` (cold Chromium start, 5-15 s) are vulnerable to
   ``ClosedResourceError`` when the MCP stdio transport is torn down before the
   subprocess finishes its response. Two mitigations are applied here:

   - **Sequential execution** (via a per-instance semaphore): prevents concurrent MCP
     calls over the same stdio pipe from multiplexing badly, which is the root cause of
     transport teardowns under load.

   - **Configurable tool timeout** (via ``asyncio.wait_for``): gives the agent-level
     tool executor a wall-clock timeout that is independent of ``PLAYWRIGHT_TIMEOUT``
     (which only covers the Playwright side, not the MCP client read timeout). When the
     timeout fires the ``asyncio.CancelledError`` is re-raised so the ``finally`` block
     cleans up ``_in_flight`` correctly before the next ``before_agent`` hook runs.
"""

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

_DEFAULT_TOOL_TIMEOUT: float = 120.0
"""Default agent-level tool execution timeout in seconds.

Set to 120 s (2 min) to accommodate slow tools like ``playwright_browser_navigate``
(cold Chromium spawn + network) which can take 5-15 s on first call, leaving plenty
of headroom while still preventing indefinite hangs.

Override via ``PatchToolCallsMiddleware(tool_timeout=...)``.
"""


class PatchToolCallsMiddleware(AgentMiddleware):
    """Middleware to patch dangling tool calls and protect MCP stdio transports.

    Args:
        sequential: If ``True`` (default), a per-instance semaphore serialises async
            tool calls so that concurrent MCP calls over the same stdio pipe do not
            multiplex. Pass ``False`` only if your tools are known to be safe to run
            concurrently (e.g. pure-HTTP tools with no shared transport).
        tool_timeout: Wall-clock timeout in seconds for each individual async tool
            call. Defaults to ``120.0``. Set to ``None`` to disable. This timeout
            covers the full MCP round-trip, unlike ``PLAYWRIGHT_TIMEOUT`` which only
            covers the Playwright side.
    """

    def __init__(
        self,
        *,
        sequential: bool = True,
        tool_timeout: float | None = _DEFAULT_TOOL_TIMEOUT,
    ) -> None:
        """Initialize the middleware.

        Args:
            sequential: Serialise async tool calls via a semaphore to prevent
                concurrent MCP stdio multiplexing issues.
            tool_timeout: Per-tool wall-clock timeout (seconds). ``None`` disables.
        """
        super().__init__()
        self._in_flight: set[tuple[str | None, str]] = set()
        self._sequential = sequential
        self._tool_timeout = tool_timeout
        # Semaphore(1) gives us a mutex for sequential execution.
        self._semaphore: asyncio.Semaphore | None = asyncio.Semaphore(1) if sequential else None

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
        """Wrap the tool call to track its in-flight execution (sync path).

        Args:
            request: The tool call request to process.
            handler: The next handler in the middleware chain.

        Returns:
            The ``ToolMessage`` or ``Command`` produced by the handler.
        """
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
        """Wrap the tool call to track in-flight execution (async path).

        Applies two MCP transport protections when enabled:

        - **Sequential execution**: acquires a semaphore before calling the handler so
          that concurrent calls over the same stdio pipe are serialised.
        - **Tool timeout**: wraps the handler in ``asyncio.wait_for`` so a wall-clock
          deadline is enforced independently of ``PLAYWRIGHT_TIMEOUT``.

        Args:
            request: The tool call request to process.
            handler: The async next handler in the middleware chain.

        Returns:
            The ``ToolMessage`` or ``Command`` produced by the handler.

        Raises:
            asyncio.CancelledError: If the tool is cancelled (e.g. timeout or upstream
                task cancellation). The ``_in_flight`` entry is always cleaned up in the
                ``finally`` block before re-raising so ``abefore_agent`` sees a
                consistent state.
        """
        tool_call_id = request.tool_call.get("id")
        thread_id = self._get_thread_id()
        tool_name = request.tool_call.get("name", "<unknown>")
        if tool_call_id:
            self._in_flight.add((thread_id, tool_call_id))

        try:
            if self._semaphore is not None:
                async with self._semaphore:
                    logger.debug(
                        "PatchToolCallsMiddleware: acquired semaphore for tool %s (thread=%s)",
                        tool_name,
                        thread_id,
                    )
                    return await self._invoke_with_timeout(handler, request, tool_name)
            return await self._invoke_with_timeout(handler, request, tool_name)
        finally:
            if tool_call_id:
                self._in_flight.discard((thread_id, tool_call_id))

    async def _invoke_with_timeout(
        self,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
        request: ToolCallRequest,
        tool_name: str,
    ) -> ToolMessage | Command[Any]:
        """Invoke the handler with an optional wall-clock timeout.

        Args:
            handler: The async handler to call.
            request: The tool call request.
            tool_name: Name of the tool (for logging only).

        Returns:
            The ``ToolMessage`` or ``Command`` from the handler.

        Raises:
            asyncio.CancelledError: If the timeout fires or the task is cancelled.
        """
        if self._tool_timeout is None:
            return await handler(request)

        try:
            return await asyncio.wait_for(handler(request), timeout=self._tool_timeout)
        except TimeoutError:
            msg = f"tool {tool_name!r} exceeded timeout of {self._tool_timeout:.1f} s"
            logger.warning(
                "PatchToolCallsMiddleware: tool %s timed out after %.1f s — cancelling",
                tool_name,
                self._tool_timeout,
            )
            raise asyncio.CancelledError(msg) from None

    def before_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Before the agent runs, handle dangling tool calls from any AIMessage.

        Args:
            state: The current agent state.
            runtime: The agent runtime.

        Returns:
            A state update dictionary with patched messages, or None if no patching needed.
        """
        return self._process_state(state)

    async def abefore_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Async hook before agent runs. Waits for in-flight tools to complete cancellation.

        Args:
            state: The current agent state.
            runtime: The agent runtime.

        Returns:
            A state update dictionary with patched messages, or None if no patching needed.
        """
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

        logger.info(
            "PatchToolCallsMiddleware: Found dangling tool calls for thread %s. "
            "Injecting cancellation ToolMessage.",
            thread_id,
        )

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
