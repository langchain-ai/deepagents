import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage

from deepagents.middleware.filesystem import FilesystemMiddleware


def test_rate_limiting_sync():
    # 1 token per minute, burst of 1
    middleware = FilesystemMiddleware(rate_limit_per_minute=1, burst_limit=1)

    request = ToolCallRequest(
        tool_call={"name": "ls_info", "args": {"path": "/"}, "id": "1"}, runtime=MagicMock(), tool=MagicMock(), state=MagicMock()
    )
    handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="1", name="ls_info"))

    # First call - allowed
    result = middleware.wrap_tool_call(request, handler)
    assert result.content == "ok"

    # Second call - rejected
    result = middleware.wrap_tool_call(request, handler)
    assert "Rate limit exceeded" in result.content


@pytest.mark.asyncio
async def test_rate_limiting_async():
    # 60 tokens per minute (1 per sec), burst of 1
    middleware = FilesystemMiddleware(rate_limit_per_minute=60, burst_limit=1)

    request = ToolCallRequest(
        tool_call={"name": "grep_raw", "args": {"pattern": "foo"}, "id": "2"}, runtime=MagicMock(), tool=MagicMock(), state=MagicMock()
    )
    handler = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="2", name="grep_raw"))

    # First call - allowed
    result = await middleware.awrap_tool_call(request, handler)
    assert result.content == "ok"

    # Second call - rejected (immediate)
    result = await middleware.awrap_tool_call(request, handler)
    assert "Rate limit exceeded" in result.content

    # Wait for token
    await asyncio.sleep(1.1)
    result = await middleware.awrap_tool_call(request, handler)
    assert result.content == "ok"


def test_non_expensive_tools_not_limited():
    middleware = FilesystemMiddleware(rate_limit_per_minute=0, burst_limit=0)

    request = ToolCallRequest(
        tool_call={"name": "read_file", "args": {"file_path": "f.txt"}, "id": "3"}, runtime=MagicMock(), tool=MagicMock(), state=MagicMock()
    )
    handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="3", name="read_file"))

    # read_file should NOT be limited
    result = middleware.wrap_tool_call(request, handler)
    assert result.content == "ok"
