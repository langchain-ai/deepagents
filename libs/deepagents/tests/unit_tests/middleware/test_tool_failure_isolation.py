"""Unit tests for ToolFailureIsolationMiddleware."""

import asyncio
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deepagents.middleware.tool_failure_isolation import ToolFailureIsolationMiddleware


def _make_request(
    tool_name: str = "test_tool",
    tool_call_id: str = "call_123",
) -> ToolCallRequest:
    """Create a minimal ToolCallRequest for testing."""
    return ToolCallRequest(
        tool_call={"name": tool_name, "args": {}, "id": tool_call_id, "type": "tool_call"},
        tool=None,
        state={"messages": []},
        runtime=None,  # type: ignore[arg-type]
    )


def _success_handler(request: ToolCallRequest) -> ToolMessage:
    return ToolMessage(
        content="success",
        name=request.tool_call["name"],
        tool_call_id=request.tool_call["id"],
    )


def _failing_handler(_request: ToolCallRequest) -> ToolMessage:
    msg = "something went wrong"
    raise ValueError(msg)


async def _async_failing_handler(_request: ToolCallRequest) -> ToolMessage:
    msg = "async failure"
    raise ValueError(msg)


class TestToolFailureIsolationMiddleware:
    """Tests for ToolFailureIsolationMiddleware."""

    def test_successful_tool_call_passes_through(self) -> None:
        """Successful tool calls are returned unchanged."""
        middleware = ToolFailureIsolationMiddleware()
        result = middleware.wrap_tool_call(_make_request(), _success_handler)

        assert isinstance(result, ToolMessage)
        assert result.content == "success"

    def test_failed_tool_call_returns_error_message(self) -> None:
        """Failed tool calls are converted to error ToolMessages with correct metadata."""
        middleware = ToolFailureIsolationMiddleware()
        result = middleware.wrap_tool_call(
            _make_request(tool_name="read_file", tool_call_id="call_abc"),
            _failing_handler,
        )

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.name == "read_file"
        assert result.tool_call_id == "call_abc"
        assert result.content == "Tool execution failed: ValueError: something went wrong"

    def test_include_traceback(self) -> None:
        """Error messages include traceback when configured."""
        middleware = ToolFailureIsolationMiddleware(include_traceback=True)
        result = middleware.wrap_tool_call(_make_request(), _failing_handler)

        assert isinstance(result, ToolMessage)
        assert "Traceback" in result.content

    def test_command_result_passes_through(self) -> None:
        """Command results from successful tools are returned unchanged."""
        middleware = ToolFailureIsolationMiddleware()

        def handler(_request: ToolCallRequest) -> Command[Any]:
            return Command(update={"messages": []})

        result = middleware.wrap_tool_call(_make_request(), handler)
        assert isinstance(result, Command)

    async def test_async_failed_tool_call(self) -> None:
        """Async failed tool calls are converted to error ToolMessages."""
        middleware = ToolFailureIsolationMiddleware()
        result = await middleware.awrap_tool_call(_make_request(), _async_failing_handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "ValueError" in result.content

    async def test_parallel_calls_one_fails_others_succeed(self) -> None:
        """Issue #694: when parallel tools run and 1 fails, others should complete."""
        middleware = ToolFailureIsolationMiddleware()

        async def run_tool(name: str, call_id: str, *, should_fail: bool) -> ToolMessage:
            request = _make_request(name, call_id)

            async def handler(req: ToolCallRequest) -> ToolMessage:
                await asyncio.sleep(0.01)
                if should_fail:
                    msg = "context limit exceeded"
                    raise ValueError(msg)
                return ToolMessage(
                    content=f"{name} completed",
                    name=req.tool_call["name"],
                    tool_call_id=req.tool_call["id"],
                )

            return await middleware.awrap_tool_call(request, handler)

        tool_a, tool_b, tool_c = await asyncio.gather(
            run_tool("search_a", "c1", should_fail=False),
            run_tool("search_b", "c2", should_fail=True),
            run_tool("search_c", "c3", should_fail=False),
        )

        # Successful tools completed normally
        assert tool_a.content == "search_a completed"
        assert tool_c.content == "search_c completed"

        # Failed tool got an error ToolMessage instead of raising
        assert tool_b.status == "error"
        assert "context limit exceeded" in tool_b.content
