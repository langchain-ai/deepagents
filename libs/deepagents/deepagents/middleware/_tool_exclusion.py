"""Middleware for filtering excluded tools from model requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware

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


def _tool_name(tool: BaseTool | dict[str, str]) -> str | None:
    """Extract tool name from a `BaseTool` or dict tool."""
    if isinstance(tool, dict):
        return tool.get("name")
    return getattr(tool, "name", None)


class _ToolExclusionMiddleware(AgentMiddleware[Any, Any, Any]):
    """Middleware that filters excluded tools from the model request.

    Should be placed late in the middleware stack (after all
    tool-injecting middleware) so it can strip middleware-injected tools
    (filesystem, subagent, etc.) that the harness profile marks as excluded.

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
