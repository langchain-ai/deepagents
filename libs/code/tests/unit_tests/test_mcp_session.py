"""Malformed server results stay failed tool messages through the MCP router."""

import asyncio
import json
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP
from fastmcp.client.transports import FastMCPTransport
from langchain_core.messages import ToolMessage
from mcp import ClientSession
from mcp.types import CallToolRequestParams, CallToolResult, TextContent

from deepagents_code._mcp_session import _MCPClientSession
from deepagents_code.mcp_tools import (
    _build_mcp_tool,
    _mount_backends,
    _warm_mcp_adapter_imports,
)

_SCHEMA = {
    "type": "object",
    "properties": {"count": {"type": "integer"}},
    "required": ["count"],
}
_ERROR = '{"error":"invalid_argument","summary":"filter expects an integer"}'


@pytest.mark.parametrize(
    ("structured", "is_error", "text", "status"),
    [
        (None, False, _ERROR, "error"),
        ({"count": "bad"}, False, "invalid count", "error"),
        ({"count": 1}, False, "valid result", "success"),
        (None, True, _ERROR, "error"),
    ],
)
async def test_router_preserves_result_and_status(
    structured: dict[str, object] | None, is_error: bool, text: str, status: str
) -> None:
    await asyncio.to_thread(_warm_mcp_adapter_imports)
    server = FastMCP("test")
    calls = 0

    @server.tool(output_schema=_SCHEMA)
    def report() -> dict[str, int]:
        return {"count": 1}

    async def respond(  # noqa: RUF029  # MCP request handlers must be async.
        _context: object, _params: CallToolRequestParams
    ) -> CallToolResult:
        nonlocal calls
        calls += 1
        return CallToolResult(
            content=[TextContent(type="text", text=text)],
            structured_content=structured,
            is_error=is_error,
        )

    # Emit the server's wire response directly, including deliberately invalid ones.
    server._mcp_server.add_request_handler("tools/call", CallToolRequestParams, respond)
    client, stack, discovered, failures = await _mount_backends(
        {"test": FastMCPTransport(server)}, redact={"test": False}
    )
    try:
        assert not failures
        tool = await _build_mcp_tool(
            mcp_tool=discovered["test"][0], server_name="test", client=client
        )
        result = await tool.ainvoke(
            {
                "name": tool.name,
                "args": {},
                "id": "call-1",
                "type": "tool_call",
            }
        )
        assert isinstance(result, ToolMessage)
        assert result.status == status
        assert result.tool_call_id == "call-1"
        assert text in [block["text"] for block in result.content]
        assert calls == 1
        if status == "error":
            assert result.artifact is None
        if not is_error and status == "error":
            assert "output-schema validation" in json.dumps(result.content)
        else:
            assert "output-schema validation" not in json.dumps(result.content)
    finally:
        await stack.aclose()


async def test_unrelated_runtime_error_propagates() -> None:
    session = _MCPClientSession(MagicMock(), MagicMock())
    result = CallToolResult(content=[])
    with (
        patch.object(
            ClientSession,
            "validate_tool_result",
            AsyncMock(side_effect=RuntimeError("discovery failed")),
        ),
        pytest.raises(RuntimeError, match="discovery failed"),
    ):
        await session.validate_tool_result("report", result)
    assert not result.is_error


async def test_invalid_schema_is_failed_output() -> None:
    session = _MCPClientSession(MagicMock(), MagicMock())
    session._tool_output_schemas["report"] = {"type": "not-a-json-schema-type"}
    result = CallToolResult(content=[], structured_content={})
    await session.validate_tool_result("report", result)
    assert result.is_error
    assert result.structured_content is None


async def test_unconstrained_error_field_is_not_reinterpreted() -> None:
    session = _MCPClientSession(MagicMock(), MagicMock())
    session._tool_output_schemas["report"] = None
    result = CallToolResult(content=[TextContent(type="text", text=_ERROR)])
    await session.validate_tool_result("report", result)
    assert not result.is_error


def test_cold_validation_does_not_block_event_loop() -> None:
    code = """
import asyncio
import sys
from unittest.mock import MagicMock
from blockbuster import blockbuster_ctx
from mcp.types import CallToolResult
from deepagents_code._mcp_session import _MCPClientSession
from deepagents_code.mcp_tools import _warm_mcp_adapter_imports

async def main():
    assert 'jsonschema' not in sys.modules
    session = _MCPClientSession(MagicMock(), MagicMock())
    session._tool_output_schemas['report'] = {'type': 'object'}
    with blockbuster_ctx():
        await asyncio.to_thread(_warm_mcp_adapter_imports)
        result = CallToolResult(content=[], structured_content={})
        await session.validate_tool_result('report', result)
        assert not result.is_error
        result = CallToolResult(content=[])
        await session.validate_tool_result('report', result)
        assert result.is_error

asyncio.run(main())
"""
    process = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
