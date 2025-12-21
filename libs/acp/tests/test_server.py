from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

from acp.schema import NewSessionRequest, PromptRequest
from acp.schema import TextContentBlock
from dirty_equals import IsUUID
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool

from deepagents_acp.server import DeepagentsACP, get_weather
from tests.chat_model import GenericFakeChatModel


class FakeAgentSideConnection:
    """Simple fake implementation of AgentSideConnection for testing."""

    def __init__(self) -> None:
        """Initialize the fake connection with an empty calls list."""
        self.calls: list[dict[str, Any]] = []

    async def sessionUpdate(self, notification: Any) -> None:
        """Track sessionUpdate calls."""
        self.calls.append(notification)


class FixedGenericFakeChatModel(GenericFakeChatModel):
    """Fixed version of GenericFakeChatModel that properly handles bind_tools."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Override bind_tools to return self."""
        return self


@tool(description="Get the current weather for a location")
def get_weather_tool(location: str) -> str:
    """Get the current weather for a location.

    Args:
        location: The city and state, e.g. "San Francisco, CA"

    Returns:
        A string describing the current weather
    """
    # Return fake weather data for testing
    return f"The weather in {location} is sunny and 72°F"


@asynccontextmanager
async def deepagents_acp_test_context(
    messages: list[BaseMessage],
    prompt_request: PromptRequest,
    tools: list[Any] | None = None,
    stream_delimiter: str | None = r"(\s)",
):
    """Context manager for testing DeepagentsACP.

    Args:
        messages: List of messages for the fake model to return
        prompt_request: The prompt request to send to the agent
        tools: List of tools to provide to the agent (defaults to [])
        stream_delimiter: How to chunk content when streaming (default: r"(\\s)" for whitespace)

    Yields:
        FakeAgentSideConnection: The connection object that tracks sessionUpdate calls
    """
    connection = FakeAgentSideConnection()
    model = FixedGenericFakeChatModel(
        messages=iter(messages),
        stream_delimiter=stream_delimiter,
    )
    tools = tools if tools is not None else []

    deepagents_acp = DeepagentsACP(
        connection=connection,
        model=model,
        tools=tools,
    )

    # Create a new session
    session_response = await deepagents_acp.newSession(
        NewSessionRequest(cwd="/tmp", mcpServers=[])
    )
    session_id = session_response.sessionId

    # Update the prompt request with the session ID
    prompt_request.sessionId = session_id

    # Call prompt
    await deepagents_acp.prompt(prompt_request)

    try:
        yield connection
    finally:
        pass


class TestDeepAgentsACP:
    """Test suite for DeepagentsACP initialization."""

    async def test_initialization(self) -> None:
        """Test that DeepagentsACP can be initialized without errors."""
        prompt_request = PromptRequest(
            sessionId="",  # Will be set by context manager
            prompt=[TextContentBlock(text="Hi!", type="text")],
        )

        async with deepagents_acp_test_context(
            messages=[AIMessage(content="Hello!")],
            prompt_request=prompt_request,
            tools=[get_weather],
        ) as connection:
            assert len(connection.calls) == 1
            first_call = connection.calls[0].model_dump()
            assert first_call == {
                "field_meta": None,
                "sessionId": IsUUID,
                "update": {
                    "content": {
                        "annotations": None,
                        "field_meta": None,
                        "text": "Hello!",
                        "type": "text",
                    },
                    "field_meta": None,
                    "sessionUpdate": "agent_message_chunk",
                },
            }

    async def test_tool_call_and_response(self) -> None:
        """Test that DeepagentsACP handles tool calls correctly.

        This test verifies that when an AI message contains tool_calls, the agent:
        1. Detects and executes the tool call
        2. Sends tool call progress notifications (pending and completed)
        3. Streams the AI response content as chunks after tool execution

        Note: The FakeChat model streams messages but the agent graph must actually
        execute the tools for the flow to complete.
        """
        prompt_request = PromptRequest(
            sessionId="",  # Will be set by context manager
            prompt=[TextContentBlock(text="What's the weather in Paris?", type="text")],
        )

        # The fake model will be called multiple times by the agent graph:
        # 1. First call: AI decides to use the tool (with tool_calls)
        # 2. After tool execution: AI responds with the result
        async with deepagents_acp_test_context(
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_weather_tool",
                            "args": {"location": "Paris, France"},
                            "id": "call_123",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="The weather in Paris is sunny and 72°F today!"),
            ],
            prompt_request=prompt_request,
            tools=[get_weather_tool],
        ) as connection:
            # Expected call sequence:
            # Call 0: Tool call progress (status="pending")
            # Call 1: Tool call progress (status="completed")
            # Calls 2+: Message chunks for "The weather in Paris is sunny and 72°F today!"

            # Find tool call indices
            tool_call_indices = []
            for i, call in enumerate(connection.calls):
                call_dict = call.model_dump()
                if call_dict["update"]["sessionUpdate"] == "tool_call_update":
                    tool_call_indices.append(i)

            # Verify we have exactly 2 tool call updates
            assert len(tool_call_indices) == 2

            # Verify tool call pending
            pending_idx = tool_call_indices[0]
            pending_call = connection.calls[pending_idx].model_dump()
            assert pending_call["update"]["sessionUpdate"] == "tool_call_update"
            assert pending_call["update"]["status"] == "pending"
            assert pending_call["update"]["toolCallId"] == "call_123"
            assert pending_call["update"]["title"] == "get_weather_tool"
            assert pending_call["update"]["rawInput"] == {"location": "Paris, France"}

            # Verify tool call completed
            completed_idx = tool_call_indices[1]
            completed_call = connection.calls[completed_idx].model_dump()
            assert completed_call["update"]["sessionUpdate"] == "tool_call_update"
            assert completed_call["update"]["status"] == "completed"
            assert completed_call["update"]["toolCallId"] == "call_123"
            assert completed_call["update"]["title"] == "get_weather_tool"
            assert completed_call["update"]["rawOutput"] == "The weather in Paris, France is sunny and 72°F"

            # Verify all non-tool-call updates are message chunks
            for i, call in enumerate(connection.calls):
                if i not in tool_call_indices:
                    call_dict = call.model_dump()
                    assert call_dict["update"]["sessionUpdate"] == "agent_message_chunk"
                    assert call_dict["update"]["content"]["type"] == "text"


async def test_fake_chat_model_streaming() -> None:
    """Test to verify GenericFakeChatModel stream_delimiter API.

    This test demonstrates the different streaming modes available via stream_delimiter.
    """
    # Test 1: No streaming (stream_delimiter=None) - single chunk
    model_no_stream = FixedGenericFakeChatModel(
        messages=iter([AIMessage(content="Hello world")]),
        stream_delimiter=None,
    )
    chunks = []
    async for chunk in model_no_stream.astream("test"):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0].content == "Hello world"

    # Test 2: Stream on whitespace using regex (default behavior)
    model_whitespace = FixedGenericFakeChatModel(
        messages=iter([AIMessage(content="Hello world")]),
        stream_delimiter=r"(\s)",
    )
    chunks = []
    async for chunk in model_whitespace.astream("test"):
        chunks.append(chunk)
    # Should split into: "Hello", " ", "world"
    assert len(chunks) == 3
    assert chunks[0].content == "Hello"
    assert chunks[1].content == " "
    assert chunks[2].content == "world"

    # Test 3: Stream with tool_calls
    model_with_tools = FixedGenericFakeChatModel(
        messages=iter([
            AIMessage(
                content="Checking weather",
                tool_calls=[
                    {
                        "name": "get_weather_tool",
                        "args": {"location": "paris, france"},
                        "id": "call_123",
                        "type": "tool_call",
                    }
                ],
            ),
        ]),
        stream_delimiter=r"(\s)",
    )
    chunks = []
    async for chunk in model_with_tools.astream("test"):
        chunks.append(chunk)
    # Tool calls should only be in the last chunk
    assert len(chunks) > 0
    assert chunks[-1].tool_calls == [
        {
            "name": "get_weather_tool",
            "args": {"location": "paris, france"},
            "id": "call_123",
            "type": "tool_call",
        }
    ]
    # Earlier chunks should not have tool_calls
    for chunk in chunks[:-1]:
        assert chunk.tool_calls == []


