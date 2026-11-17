"""Talon middleware for MCP model binding and tool invocation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai.chat_models.base import BaseChatOpenAI
from mcp.shared.exceptions import MCPError

from deepagents_talon.mcp import _normalize_mcp_arguments, _run_authorized

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ToolCallRequest

MCP_TOOL_METADATA_KEY = "_deepagents_talon_mcp"
MCP_SKILL_ONLY_METADATA_KEY = "_deepagents_talon_skill_only"


def _preserve_mcp_schemas(request: ModelRequest) -> ModelRequest:
    if not isinstance(request.model, BaseChatOpenAI) or not request.model.use_responses_api:
        return request
    tools = [
        convert_to_openai_tool(tool, strict=False)
        if isinstance(tool, BaseTool) and (tool.metadata or {}).get(MCP_TOOL_METADATA_KEY)
        else tool
        for tool in request.tools
    ]
    return request.override(tools=tools)


class _TalonMCPMiddleware(AgentMiddleware):
    name = "TalonMCPMiddleware"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(_preserve_mcp_schemas(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(_preserve_mcp_schemas(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        tool = request.tool
        if tool is None or not (tool.metadata or {}).get(MCP_TOOL_METADATA_KEY):
            return await handler(request)

        arguments = _normalize_mcp_arguments(
            request.tool_call.get("args") or {},
            tool.args_schema,
        )
        request = request.override(tool_call={**request.tool_call, "args": arguments})
        try:
            return await _run_authorized(
                request.tool_call["id"],
                lambda: handler(request),
            )
        except MCPError as exc:
            return ToolMessage(
                content=f"MCP protocol error {exc.code}: {exc.message}",
                name=request.tool_call["name"],
                tool_call_id=request.tool_call["id"],
                status="error",
            )


def talon_mcp_middleware() -> AgentMiddleware:
    """Preserve MCP schemas and bind authorization state to MCP tool calls."""
    return _TalonMCPMiddleware()


__all__ = ["MCP_TOOL_METADATA_KEY", "talon_mcp_middleware"]
