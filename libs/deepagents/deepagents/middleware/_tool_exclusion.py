"""Middleware for filtering excluded tools from model requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import (
        ExtendedModelResponse,
        ModelRequest,
        ModelResponse,
        ResponseT,
    )
    from langchain_core.messages import AIMessage
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.types import Command


def _tool_name(tool: BaseTool | dict[str, str]) -> str | None:
    """Extract tool name from a `BaseTool` or dict tool."""
    if isinstance(tool, dict):
        name = tool.get("name")
        return name if isinstance(name, str) else None
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


class _ToolExclusionMiddleware(AgentMiddleware[Any, Any, Any]):
    """Middleware that filters excluded tools from the model request.

    Should be placed late in the middleware stack (after all
    tool-injecting middleware) so it can strip middleware-injected tools
    (filesystem, subagent, etc.) that the harness profile marks as excluded.

    Excluded names are also rejected at the tool-call boundary, since the
    executor still has them registered and dispatches on the name the model
    emits. That keeps execution consistent with what was advertised; it is not a
    security surface. Exclusions resolve per model.

    Args:
        excluded: Tool names to remove before the model sees them.
    """

    def __init__(self, *, excluded: frozenset[str]) -> None:
        self._excluded = excluded

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Filter excluded tools before they reach the model."""
        if self._excluded:
            filtered = [t for t in request.tools if _tool_name(t) not in self._excluded]
            request = request.override(tools=filtered)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        """Async variant of `wrap_model_call`."""
        if self._excluded:
            filtered = [t for t in request.tools if _tool_name(t) not in self._excluded]
            request = request.override(tools=filtered)
        return await handler(request)

    def _rejection(self, request: ToolCallRequest) -> ToolMessage | None:
        """Build the error result for a call naming an excluded tool."""
        name = request.tool_call["name"]
        if name not in self._excluded:
            return None
        return ToolMessage(
            content=f"Error: {name} is not available.",
            tool_call_id=request.tool_call["id"] or "",
            name=name,
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Reject calls naming an excluded tool instead of executing them."""
        return self._rejection(request) or handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Async variant of `wrap_tool_call`."""
        return self._rejection(request) or await handler(request)
