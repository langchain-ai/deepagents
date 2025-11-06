"""Tests for ToolExceptionHandlerMiddleware."""

import pytest
from langchain_core.tools import BaseTool, ToolException
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from deepagents.middleware.tool_exception_handler import ToolExceptionHandlerMiddleware


class FailingToolInput(BaseModel):
    """Input schema for the failing tool."""

    message: str = Field(description="The error message to raise")


class SuccessfulToolInput(BaseModel):
    """Input schema for the successful tool."""

    value: str = Field(description="The value to return")


class FailingTool(BaseTool):
    """A tool that always raises ToolException."""

    name: str = "failing_tool"
    description: str = "A tool that always fails with ToolException"
    args_schema: type[BaseModel] = FailingToolInput

    def _run(self, message: str) -> str:
        """Execute the tool."""
        raise ToolException(message)


class UnexpectedErrorTool(BaseTool):
    """A tool that raises an unexpected exception."""

    name: str = "unexpected_error_tool"
    description: str = "A tool that raises an unexpected error"
    args_schema: type[BaseModel] = FailingToolInput

    def _run(self, message: str) -> str:
        """Execute the tool."""
        raise RuntimeError(message)


class SuccessfulTool(BaseTool):
    """A tool that always succeeds."""

    name: str = "successful_tool"
    description: str = "A tool that always succeeds"
    args_schema: type[BaseModel] = SuccessfulToolInput

    def _run(self, value: str) -> str:
        """Execute the tool."""
        return f"Success: {value}"


def test_tool_exception_is_caught():
    """Test that ToolException is caught and converted to error message."""
    middleware = ToolExceptionHandlerMiddleware()
    tool = FailingTool()
    error_msg = "This is a test error"

    # Create a mock request
    class MockRequest:
        tool = tool
        tool_call = {"id": "test-123", "args": {"message": error_msg}}

    # Create a mock handler that will raise ToolException
    def handler(request):
        return tool.invoke(request.tool_call["args"])

    result = middleware.wrap_tool_call(MockRequest(), handler)

    # Verify the result is a ToolMessage with error status
    assert isinstance(result, ToolMessage)
    assert "Tool execution failed:" in result.content
    assert error_msg in result.content
    assert result.name == "failing_tool"
    assert result.tool_call_id == "test-123"
    assert result.status == "error"


def test_unexpected_exception_is_caught():
    """Test that unexpected exceptions are caught and converted to error messages."""
    middleware = ToolExceptionHandlerMiddleware()
    tool = UnexpectedErrorTool()
    error_msg = "Unexpected runtime error"

    # Create a mock request
    class MockRequest:
        tool = tool
        tool_call = {"id": "test-456", "args": {"message": error_msg}}

    # Create a mock handler that will raise RuntimeError
    def handler(request):
        return tool.invoke(request.tool_call["args"])

    result = middleware.wrap_tool_call(MockRequest(), handler)

    # Verify the result is a ToolMessage with error status
    assert isinstance(result, ToolMessage)
    assert "Tool execution encountered an unexpected error:" in result.content
    assert error_msg in result.content
    assert result.name == "unexpected_error_tool"
    assert result.tool_call_id == "test-456"
    assert result.status == "error"
    assert "Traceback" in result.content


def test_successful_tool_not_affected():
    """Test that successful tool calls are not affected by the middleware."""
    middleware = ToolExceptionHandlerMiddleware()
    tool = SuccessfulTool()
    test_value = "test value"

    # Create a mock request
    class MockRequest:
        tool = tool
        tool_call = {"id": "test-789", "args": {"value": test_value}}

    # Create a mock handler that will succeed
    def handler(request):
        result = tool.invoke(request.tool_call["args"])
        return ToolMessage(
            content=result,
            name=tool.name,
            tool_call_id=request.tool_call["id"],
        )

    result = middleware.wrap_tool_call(MockRequest(), handler)

    # Verify the result is a successful ToolMessage
    assert isinstance(result, ToolMessage)
    assert f"Success: {test_value}" in result.content
    assert result.name == "successful_tool"
    assert result.tool_call_id == "test-789"
    # Successful calls don't have status='error'
    assert result.status != "error"


@pytest.mark.asyncio
async def test_async_tool_exception_is_caught():
    """Test that async ToolException is caught and converted to error message."""
    middleware = ToolExceptionHandlerMiddleware()
    tool = FailingTool()
    error_msg = "Async test error"

    # Create a mock request
    class MockRequest:
        tool = tool
        tool_call = {"id": "test-async-123", "args": {"message": error_msg}}

    # Create a mock async handler that will raise ToolException
    async def handler(request):
        return tool.invoke(request.tool_call["args"])

    result = await middleware.awrap_tool_call(MockRequest(), handler)

    # Verify the result is a ToolMessage with error status
    assert isinstance(result, ToolMessage)
    assert "Tool execution failed:" in result.content
    assert error_msg in result.content
    assert result.name == "failing_tool"
    assert result.tool_call_id == "test-async-123"
    assert result.status == "error"

