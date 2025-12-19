"""Tests for RetryAndSanitizeInterceptor."""

import pytest
from unittest.mock import Mock, AsyncMock
import asyncio

import httpx
from langchain_core.tools import ToolException
from langchain_mcp_adapters.tools import MCPToolCallRequest
from mcp.types import CallToolResult, TextContent

from chatlas_agents.tools.interceptors import RetryAndSanitizeInterceptor


@pytest.mark.asyncio
async def test_interceptor_sanitizes_none_arguments():
    """Test that the interceptor removes None-valued arguments."""
    interceptor = RetryAndSanitizeInterceptor(max_attempts=1)
    
    # Create a mock handler
    handler = AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="success")]))
    
    # Create a request with None arguments
    request = MCPToolCallRequest(
        name="test_tool",
        args={"param1": "value1", "param2": None, "param3": "value3"},
        server_name="test_server",
        headers=None,
        runtime=None,
    )
    
    # Execute the interceptor
    result = await interceptor(request, handler)
    
    # Verify the handler was called with sanitized arguments
    handler.assert_called_once()
    call_args = handler.call_args[0][0]  # Get the MCPToolCallRequest passed to handler
    assert call_args.args == {"param1": "value1", "param3": "value3"}
    assert "param2" not in call_args.args


@pytest.mark.asyncio
async def test_interceptor_does_not_retry_tool_exception():
    """Test that ToolException is not retried."""
    interceptor = RetryAndSanitizeInterceptor(max_attempts=3)
    
    # Create a mock handler that raises ToolException
    handler = AsyncMock(side_effect=ToolException("Invalid parameter"))
    
    # Create a request
    request = MCPToolCallRequest(
        name="test_tool",
        args={"param": "value"},
        server_name="test_server",
        headers=None,
        runtime=None,
    )
    
    # Execute the interceptor - should raise ToolException immediately
    with pytest.raises(ToolException, match="Invalid parameter"):
        await interceptor(request, handler)
    
    # Verify handler was called only once (no retry)
    assert handler.call_count == 1


@pytest.mark.asyncio
async def test_interceptor_retries_http_5xx_errors():
    """Test that HTTP 5xx errors are retried."""
    interceptor = RetryAndSanitizeInterceptor(max_attempts=3, backoff_base=0.01)
    
    # Create a mock handler that fails twice then succeeds
    call_count = 0
    async def mock_handler(request):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            response = Mock()
            response.status_code = 503
            raise httpx.HTTPStatusError("Service Unavailable", request=Mock(), response=response)
        return CallToolResult(content=[TextContent(type="text", text="success")])
    
    # Create a request
    request = MCPToolCallRequest(
        name="test_tool",
        args={"param": "value"},
        server_name="test_server",
        headers=None,
        runtime=None,
    )
    
    # Execute the interceptor - should succeed after retries
    result = await interceptor(request, mock_handler)
    
    # Verify handler was called 3 times (2 failures + 1 success)
    assert call_count == 3
    assert result.content[0].text == "success"


@pytest.mark.asyncio
async def test_interceptor_retries_timeout_errors():
    """Test that timeout errors are retried."""
    interceptor = RetryAndSanitizeInterceptor(max_attempts=3, backoff_base=0.01)
    
    # Create a mock handler that times out twice then succeeds
    call_count = 0
    async def mock_handler(request):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise httpx.TimeoutException("Request timeout")
        return CallToolResult(content=[TextContent(type="text", text="success")])
    
    # Create a request
    request = MCPToolCallRequest(
        name="test_tool",
        args={"param": "value"},
        server_name="test_server",
        headers=None,
        runtime=None,
    )
    
    # Execute the interceptor - should succeed after retries
    result = await interceptor(request, mock_handler)
    
    # Verify handler was called 3 times (2 timeouts + 1 success)
    assert call_count == 3


@pytest.mark.asyncio
async def test_interceptor_exhausts_retries_on_persistent_errors():
    """Test that persistent errors exhaust retries and raise."""
    interceptor = RetryAndSanitizeInterceptor(max_attempts=3, backoff_base=0.01)
    
    # Create a mock handler that always fails
    handler = AsyncMock(side_effect=httpx.TimeoutException("Persistent timeout"))
    
    # Create a request
    request = MCPToolCallRequest(
        name="test_tool",
        args={"param": "value"},
        server_name="test_server",
        headers=None,
        runtime=None,
    )
    
    # Execute the interceptor - should raise after exhausting retries
    with pytest.raises(httpx.TimeoutException):
        await interceptor(request, handler)
    
    # Verify handler was called max_attempts times
    assert handler.call_count == 3
