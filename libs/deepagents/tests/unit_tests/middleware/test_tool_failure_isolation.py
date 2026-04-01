"""Unit tests for ToolFailureIsolationMiddleware."""

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deepagents.middleware.tool_failure_isolation import ToolFailureIsolationMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


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
    """Tests for ToolFailureIsolationMiddleware wrap_tool_call behavior."""

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


class TestToolFailureIsolationInGraph:
    """Test that the middleware works within a real LangGraph agent execution.

    These tests use `create_agent` with `GenericFakeChatModel` to verify
    that the middleware correctly intercepts tool errors through the full
    ToolNode → middleware wrapper chain, not just direct method calls.
    """

    def test_failing_tool_in_agent_graph(self) -> None:
        """A tool that raises is caught by the middleware and returned as error ToolMessage.

        The agent should see the error and continue (not crash).
        """

        @tool
        def unstable_tool(query: str) -> str:  # noqa: ARG001
            """A tool that always fails."""
            msg = "service unavailable"
            raise RuntimeError(msg)

        model = GenericFakeChatModel(
            messages=iter([
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "unstable_tool", "args": {"query": "test"}, "id": "call_1", "type": "tool_call"},
                    ],
                ),
                AIMessage(content="The tool failed, but I can handle it."),
            ]),
        )

        agent = create_agent(
            model=model,
            tools=[unstable_tool],
            middleware=[ToolFailureIsolationMiddleware()],
            checkpointer=InMemorySaver(),
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content="test")]},
            config={"configurable": {"thread_id": "test_failing"}},
        )

        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].status == "error"
        assert "RuntimeError" in tool_messages[0].content
        assert "service unavailable" in tool_messages[0].content

        # Agent continued past the error and produced a final response
        assert result["messages"][-1].content == "The tool failed, but I can handle it."

    def test_parallel_tools_one_fails_in_agent_graph(self) -> None:
        """Issue #694: parallel tool calls where one fails, others complete.

        The model issues two tool calls in a single AIMessage. One tool
        raises; the middleware converts it to an error ToolMessage. The other
        tool succeeds. The agent receives both results and continues.
        """

        @tool
        def reliable_tool(query: str) -> str:
            """A tool that always succeeds."""
            return f"result for {query}"

        @tool
        def flaky_tool(query: str) -> str:  # noqa: ARG001
            """A tool that always fails."""
            msg = "context limit exceeded"
            raise ValueError(msg)

        model = GenericFakeChatModel(
            messages=iter([
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "reliable_tool", "args": {"query": "a"}, "id": "call_ok", "type": "tool_call"},
                        {"name": "flaky_tool", "args": {"query": "b"}, "id": "call_fail", "type": "tool_call"},
                    ],
                ),
                AIMessage(content="One tool succeeded, one failed."),
            ]),
        )

        agent = create_agent(
            model=model,
            tools=[reliable_tool, flaky_tool],
            middleware=[ToolFailureIsolationMiddleware()],
            checkpointer=InMemorySaver(),
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content="test parallel")]},
            config={"configurable": {"thread_id": "test_parallel"}},
        )

        tool_messages = {m.tool_call_id: m for m in result["messages"] if isinstance(m, ToolMessage)}

        # reliable_tool succeeded
        assert tool_messages["call_ok"].status != "error"
        assert "result for a" in tool_messages["call_ok"].content

        # flaky_tool was caught by middleware
        assert tool_messages["call_fail"].status == "error"
        assert "context limit exceeded" in tool_messages["call_fail"].content

        # Agent continued and produced final response
        assert result["messages"][-1].content == "One tool succeeded, one failed."
