"""Middleware that catches tool execution errors and surfaces them as ``ToolMessage``s.

By default, langchain's ``ToolNode`` only converts ``ToolInvocationError`` (bad
arguments from the model) into a ``ToolMessage``. Any other exception raised
from within a tool — including inside a subagent — propagates out and crashes
the agent run. This middleware wraps every tool call (via ``wrap_tool_call`` /
``awrap_tool_call``) in a ``try``/``except`` so that recoverable failures are
returned to the model as an error ``ToolMessage`` instead.

It is applied by ``create_deep_agent`` to both the supervisor (main agent) and
every synchronous subagent (including the auto-added general-purpose
subagent), so a failing tool inside a subagent is converted to an error
``ToolMessage`` at the subagent level, and the ``task`` tool's own invocation
of the subagent is likewise protected at the parent level.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, ContextT, ResponseT
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage

if TYPE_CHECKING:
    from langgraph.types import Command

ToolErrorFormatter = Callable[[BaseException, ToolCallRequest], str]
"""Signature for a custom tool-error formatter."""


def _default_format_error(error: BaseException, request: ToolCallRequest) -> str:
    """Format a tool exception as a message for the model."""
    tool_name = request.tool_call.get("name", "<unknown>")
    return f"Error executing tool `{tool_name}`: {type(error).__name__}: {error}"


class ToolErrorHandlingMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Catch tool execution errors and return them as error ``ToolMessage``s.

    By default catches ``Exception`` (not ``BaseException``) so that
    ``KeyboardInterrupt`` and ``SystemExit`` propagate. Callers may restrict
    or broaden the caught types and/or customize the error message via
    ``format_error``.

    Args:
        catch: Exception type(s) to catch. Defaults to ``Exception``.
        format_error: Optional callable mapping ``(error, request)`` to the
            content string returned to the model. Defaults to
            ``"Error executing tool `<name>`: <ErrType>: <message>"``.
    """

    def __init__(
        self,
        *,
        catch: type[BaseException] | tuple[type[BaseException], ...] = Exception,
        format_error: ToolErrorFormatter | None = None,
    ) -> None:
        """Initialize the middleware."""
        self._catch = catch
        self._format_error = format_error or _default_format_error

    def _to_error_message(self, error: BaseException, request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(
            content=self._format_error(error, request),
            name=request.tool_call.get("name"),
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Run the tool, converting caught exceptions into error ``ToolMessage``s."""
        try:
            return handler(request)
        except self._catch as error:
            return self._to_error_message(error, request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """(async) Run the tool, converting caught exceptions into error ``ToolMessage``s."""
        try:
            return await handler(request)
        except self._catch as error:
            return self._to_error_message(error, request)
