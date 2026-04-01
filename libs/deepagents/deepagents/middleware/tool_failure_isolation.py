"""Middleware to isolate tool failures during parallel execution.

When LangGraph's ``ToolNode`` runs multiple tool calls via ``asyncio.gather``,
a single unhandled exception cancels **all** sibling tool calls. This
middleware catches tool exceptions and converts them into
``ToolMessage(status="error")`` responses so sibling tools can complete
normally.

Example::

    from deepagents.middleware.tool_failure_isolation import ToolFailureIsolationMiddleware

    agent = create_deep_agent(
        middleware=[ToolFailureIsolationMiddleware()],
    )

See `GitHub issue #694 <https://github.com/langchain-ai/deepagents/issues/694>`_
for the motivation behind this middleware.
"""

import logging
import traceback
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)


class ToolFailureIsolationMiddleware(AgentMiddleware):
    """Catch tool execution errors and convert them to error ``ToolMessage`` responses.

    Without this middleware, a single tool failure during parallel execution
    (via ``asyncio.gather`` in ``ToolNode``) propagates as an unhandled
    exception, cancelling all sibling tool calls. This middleware wraps each
    tool invocation so that failures are reported back to the LLM as
    ``ToolMessage(status="error")`` instead of raising, allowing other tools
    to finish.

    Args:
        include_traceback: Whether to include a Python traceback in the error
            message returned to the LLM.

    Example::

        from deepagents.middleware.tool_failure_isolation import (
            ToolFailureIsolationMiddleware,
        )

        # Basic usage — isolate all tool failures
        agent = create_deep_agent(
            middleware=[ToolFailureIsolationMiddleware()],
        )

        # Include tracebacks for debugging
        agent = create_deep_agent(
            middleware=[ToolFailureIsolationMiddleware(include_traceback=True)],
        )
    """

    def __init__(self, *, include_traceback: bool = False) -> None:
        """Initialize the middleware.

        Args:
            include_traceback: Whether to include a Python traceback in the
                error message returned to the LLM.
        """
        super().__init__()
        self._include_traceback = include_traceback

    def _format_error(self, exc: Exception) -> str:
        """Format an exception into an error message for the LLM.

        Args:
            exc: The exception that occurred during tool execution.

        Returns:
            A formatted error string.
        """
        parts = [f"Tool execution failed: {type(exc).__name__}: {exc}"]
        if self._include_traceback:
            tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
            parts.append("".join(tb))
        return "\n".join(parts)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Wrap a tool call to catch and isolate failures.

        Args:
            request: The tool call request being processed.
            handler: The next handler in the middleware chain.

        Returns:
            The original tool result on success, or a ``ToolMessage`` with
            ``status="error"`` on failure.
        """
        try:
            return handler(request)
        except Exception as exc:  # noqa: BLE001  # intentional catch-all to isolate tool failures
            tool_name = request.tool_call["name"]
            tool_call_id = request.tool_call["id"]
            logger.warning(
                "Tool '%s' (call_id=%s) failed: %s",
                tool_name,
                tool_call_id,
                exc,
            )
            return ToolMessage(
                content=self._format_error(exc),
                name=tool_name,
                tool_call_id=tool_call_id,
                status="error",
            )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async version of `wrap_tool_call`.

        Args:
            request: The tool call request being processed.
            handler: The next async handler in the middleware chain.

        Returns:
            The original tool result on success, or a ``ToolMessage`` with
            ``status="error"`` on failure.
        """
        try:
            return await handler(request)
        except Exception as exc:  # noqa: BLE001  # intentional catch-all to isolate tool failures
            tool_name = request.tool_call["name"]
            tool_call_id = request.tool_call["id"]
            logger.warning(
                "Tool '%s' (call_id=%s) failed: %s",
                tool_name,
                tool_call_id,
                exc,
            )
            return ToolMessage(
                content=self._format_error(exc),
                name=tool_name,
                tool_call_id=tool_call_id,
                status="error",
            )
