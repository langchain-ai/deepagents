"""Tests for MCP tool-call middleware behavior."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from langgraph.prebuilt.tool_node import ToolCallRequest

from deepagents_code.mcp_middleware import mcp_tool_middleware


class _ReauthMessage:
    def __str__(self) -> str:
        return "run /mcp login for linear"


def _request() -> ToolCallRequest:
    tool = SimpleNamespace(
        metadata={
            "_deepagents_code_mcp": True,
            "_deepagents_code_mcp_server": "linear",
        },
        args_schema={},
        name="linear_get_issue",
    )
    return ToolCallRequest(
        runtime=cast("Any", None),
        tool_call={"id": "call-1", "name": "linear_get_issue", "args": {}},
        state={},
        tool=cast("Any", tool),
    )


async def _slow_handler(_request: ToolCallRequest) -> ToolMessage:
    await asyncio.sleep(10)
    return ToolMessage(content="ok", tool_call_id="call-1")


@pytest.mark.asyncio
async def test_timeout_returns_server_and_tool_error() -> None:
    result = await mcp_tool_middleware(0.001).awrap_tool_call(_request(), _slow_handler)

    assert result.status == "error"
    assert "linear" in result.content
    assert "linear_get_issue" in result.content
    assert "timed out after 0.001 seconds" in result.content
    assert "may still be running server-side" in result.content
    assert "retrying may duplicate work" in result.content


@pytest.mark.asyncio
async def test_fast_call_passes_through() -> None:
    async def handler(_request: ToolCallRequest) -> ToolMessage:
        await asyncio.sleep(0)
        return ToolMessage(content="ok", tool_call_id="call-1", status="success")

    result = await mcp_tool_middleware(1).awrap_tool_call(_request(), handler)

    assert result.content == "ok"
    assert result.status == "success"


@pytest.mark.asyncio
async def test_tool_exception_is_preserved() -> None:
    async def handler(_request: ToolCallRequest) -> ToolMessage:
        await asyncio.sleep(0)
        message = "server rejected the request"
        raise ToolException(message)

    with pytest.raises(ToolException, match="server rejected"):
        await mcp_tool_middleware(1).awrap_tool_call(_request(), handler)


@pytest.mark.asyncio
async def test_reauth_error_is_translated() -> None:
    async def handler(_request: ToolCallRequest) -> ToolMessage:
        await asyncio.sleep(0)
        message = "expired"
        raise RuntimeError(message)

    with patch(
        "deepagents_code.mcp_auth.find_reauth_required",
        return_value=_ReauthMessage(),
    ):
        result = await mcp_tool_middleware(1).awrap_tool_call(_request(), handler)

    assert result.status == "error"
    assert result.content == "run /mcp login for linear"


@pytest.mark.asyncio
async def test_cancellation_propagates() -> None:
    async def handler(_request: ToolCallRequest) -> ToolMessage:
        await asyncio.sleep(0)
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await mcp_tool_middleware(1).awrap_tool_call(_request(), handler)
