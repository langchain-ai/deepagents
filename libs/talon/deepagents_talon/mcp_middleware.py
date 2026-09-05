"""Talon middleware for MCP tool invocation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.agents.middleware import wrap_tool_call

from deepagents_talon.mcp import _normalize_mcp_arguments, _run_authorized

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
    from langchain_core.messages import ToolMessage

MCP_TOOL_METADATA_KEY = "_deepagents_talon_mcp"
MCP_SERVER_METADATA_KEY = "_deepagents_talon_mcp_server"


def talon_mcp_middleware() -> AgentMiddleware:
    """Bind Talon authorization state to metadata-marked MCP tool calls."""

    @wrap_tool_call(name="TalonMCPMiddleware")
    async def _wrap(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        tool = request.tool
        metadata = getattr(tool, "metadata", None) or {}
        if tool is None or not metadata.get(MCP_TOOL_METADATA_KEY):
            return await handler(request)

        arguments = _normalize_mcp_arguments(
            request.tool_call.get("args") or {},
            getattr(tool, "args_schema", None),
        )
        request = request.override(tool_call={**request.tool_call, "args": arguments})
        return await _run_authorized(
            request.tool_call["id"],
            lambda: handler(request),
        )

    return _wrap


__all__ = ["MCP_SERVER_METADATA_KEY", "MCP_TOOL_METADATA_KEY", "talon_mcp_middleware"]
