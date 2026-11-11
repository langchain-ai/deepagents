from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from mcp.shared.exceptions import MCPError

from deepagents_talon.authorization import current_authorization_invocation
from deepagents_talon.mcp import _device_authorization_completed
from deepagents_talon.mcp_auth import DeviceAuthorizationCompletedError, format_login_error
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


async def test_protocol_error_reaches_model_without_error_data() -> None:
    tool = StructuredTool(
        name="remote_search",
        description="Search",
        args_schema={},
        metadata={MCP_TOOL_METADATA_KEY: True},
    )
    request = Request(tool, {"id": "call-8", "name": "remote_search", "args": {}})

    async def handler(_request: Request) -> ToolMessage:
        raise MCPError(-32602, "Invalid query", data={"private": "response-body"})

    message = await talon_mcp_middleware().awrap_tool_call(request, handler)

    assert message.status == "error"
    assert message.tool_call_id == "call-8"
    assert (
        message.content
        == f"MCP tool remote_search failed: {format_login_error(MCPError(-32602, 'Invalid query'))}"
    )
    assert message.name == "remote_search"
    assert message.artifact is None
    assert current_authorization_invocation() is None


async def test_non_protocol_error_reaches_model_without_retry() -> None:
    tool = StructuredTool(
        name="remote_search",
        description="Search",
        args_schema={},
        metadata={MCP_TOOL_METADATA_KEY: True},
    )
    request = Request(tool, {"id": "call-8", "name": "remote_search", "args": {}})

    calls = 0

    async def handler(_request: Request) -> ToolMessage:
        nonlocal calls
        calls += 1
        msg = "transport failed"
        raise RuntimeError(msg)

    message = await talon_mcp_middleware().awrap_tool_call(request, handler)

    assert message.status == "error"
    assert message.name == "remote_search"
    assert message.tool_call_id == "call-8"
    assert message.content == (
        f"MCP tool remote_search failed: {format_login_error(RuntimeError('transport failed'))}"
    )
    assert calls == 1
    assert current_authorization_invocation() is None


@pytest.mark.parametrize("outcome", ["success", "failure", "completed", "cancelled"])
async def test_completed_login_retries_once(outcome: str) -> None:
    tool = StructuredTool(
        name="remote_search",
        description="Search",
        args_schema={},
        metadata={MCP_TOOL_METADATA_KEY: True},
    )
    request = Request(tool, {"id": "call-9", "name": "remote_search", "args": {}})
    calls = 0

    async def handler(passed: Request) -> ToolMessage:
        nonlocal calls
        calls += 1
        assert current_authorization_invocation() == "call-9"
        if calls == 1 or outcome == "completed":
            msg = "Client failed to connect: "
            raise RuntimeError(msg) from DeviceAuthorizationCompletedError()
        if outcome == "failure":
            msg = "transport failed"
            raise RuntimeError(msg)
        if outcome == "cancelled":
            raise asyncio.CancelledError
        return ToolMessage(content="search results", tool_call_id=passed.tool_call["id"])

    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await talon_mcp_middleware().awrap_tool_call(request, handler)
    else:
        message = await talon_mcp_middleware().awrap_tool_call(request, handler)
        assert message.tool_call_id == "call-9"
        assert message.status == ("success" if outcome == "success" else "error")
        if outcome == "success":
            assert message.content == "search results"
        else:
            assert message.name == "remote_search"
            assert str(message.content).startswith("MCP tool remote_search failed: ")
    assert calls == 2
    assert current_authorization_invocation() is None


async def test_tool_cancellation_propagates_without_retry() -> None:
    tool = SimpleNamespace(metadata={MCP_TOOL_METADATA_KEY: True}, args_schema={})
    request = Request(tool, {"id": "call-10", "name": "remote_search", "args": {}})
    calls = 0

    async def handler(_request: Request) -> ToolMessage:
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await talon_mcp_middleware().awrap_tool_call(request, handler)
    assert calls == 1
    assert current_authorization_invocation() is None


@pytest.mark.parametrize("chain", ["cause", "context", "group", "group_cause", "cycle"])
def test_completed_login_detection(chain: str) -> None:
    completed = DeviceAuthorizationCompletedError()
    wrapper = RuntimeError("Client failed to connect: ")
    if chain == "cause":
        wrapper.__cause__ = completed
    elif chain == "context":
        wrapper.__context__ = completed
    elif chain == "group":
        wrapper = ExceptionGroup("connection", [ExceptionGroup("login", [completed])])
    elif chain == "group_cause":
        wrapper.__cause__ = ExceptionGroup("login", [completed])
        wrapper = ExceptionGroup("connection", [wrapper])
    else:
        wrapper.__context__ = wrapper
    assert _device_authorization_completed(wrapper) is (chain != "cycle")
