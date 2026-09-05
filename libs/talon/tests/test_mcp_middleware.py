from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool

from deepagents_talon.authorization import current_authorization_invocation
from deepagents_talon.mcp_middleware import MCP_TOOL_METADATA_KEY, talon_mcp_middleware


class Request:
    def __init__(self, tool: StructuredTool, tool_call: dict[str, object]) -> None:
        self.tool = tool
        self.tool_call = tool_call

    def override(self, *, tool_call: dict[str, object]) -> Request:
        return Request(self.tool, tool_call)


async def test_middleware_normalizes_and_binds_tool_call_id() -> None:
    async def invoke(**_kwargs: object) -> str:
        return "unused"

    tool = StructuredTool.from_function(
        coroutine=invoke,
        name="remote_search",
        description="Search",
        args_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "optional": {"type": "string"}},
            "required": ["query"],
        },
        metadata={MCP_TOOL_METADATA_KEY: True},
    )
    seen: list[tuple[str | None, dict[str, object]]] = []

    async def handler(request: Request) -> ToolMessage:
        seen.append((current_authorization_invocation(), request.tool_call["args"]))
        return ToolMessage(content="ok", tool_call_id=request.tool_call["id"])

    request = Request(
        tool,
        {"id": "call-42", "name": tool.name, "args": {"query": "", "optional": ""}},
    )
    result = await talon_mcp_middleware().awrap_tool_call(request, handler)

    assert result.content == "ok"
    assert seen == [("call-42", {"query": ""})]
    assert current_authorization_invocation() is None


async def test_middleware_ignores_unmarked_tools() -> None:
    tool = SimpleNamespace(metadata=None)
    request = Request(tool, {"id": "call-7", "name": "local", "args": {"value": ""}})

    async def handler(passed: Request) -> ToolMessage:
        assert current_authorization_invocation() is None
        assert passed is request
        return ToolMessage(content="ok", tool_call_id="call-7")

    await talon_mcp_middleware().awrap_tool_call(request, handler)
