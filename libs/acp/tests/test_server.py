from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

from acp.schema import NewSessionRequest, PromptRequest
from acp.schema import TextContentBlock
from dirty_equals import IsUUID
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool

from deepagents_acp.server import DeepagentsACP, get_weather


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
):
    """Context manager for testing DeepagentsACP.

    Args:
        messages: List of messages for the fake model to return
        prompt_request: The prompt request to send to the agent
        tools: List of tools to provide to the agent (defaults to [])

    Yields:
        FakeAgentSideConnection: The connection object that tracks sessionUpdate calls
    """
    connection = FakeAgentSideConnection()
    model = FixedGenericFakeChatModel(messages=iter(messages))
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
            # Expected calls:
            # 1. Tool call progress (status="pending") when tool_call is detected
            # 2. Tool call progress (status="completed") when tool execution finishes
            # 3-N. Message chunks for "The weather in Paris is sunny and 72°F today!"

            # Verify we have at least 3 calls (2 tool updates + at least 1 message chunk)
            assert len(connection.calls) >= 3

            # Verify first call is tool call pending
            first_call = connection.calls[0].model_dump()
            assert first_call == {
                "field_meta": None,
                "sessionId": IsUUID,
                "update": {
                    "field_meta": None,
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "call_123",
                    "title": "get_weather_tool",
                    "rawInput": {"location": "Paris, France"},
                    "content": None,
                    "rawOutput": None,
                    "status": "pending",
                },
            }

            # Verify second call is tool call completed
            second_call = connection.calls[1].model_dump()
            assert second_call == {
                "field_meta": None,
                "sessionId": IsUUID,
                "update": {
                    "field_meta": None,
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "call_123",
                    "title": "get_weather_tool",
                    "rawInput": None,
                    "content": [
                        {
                            "type": "content",
                            "content": {
                                "annotations": None,
                                "field_meta": None,
                                "text": "The weather in Paris, France is sunny and 72°F",
                                "type": "text",
                            },
                        }
                    ],
                    "rawOutput": "The weather in Paris, France is sunny and 72°F",
                    "status": "completed",
                },
            }

            # Verify remaining calls are message chunks
            for call in connection.calls[2:]:
                call_dict = call.model_dump()
                assert call_dict["update"]["sessionUpdate"] == "agent_message_chunk"
                assert call_dict["update"]["content"]["type"] == "text"


async def test_fake_chat_model_streaming() -> None:
    """Test to verify what GenericFakeChatModel streams directly.

    This test documents the behavior of GenericFakeChatModel.astream() when given
    AIMessages with empty content + tool_calls, followed by regular messages.
    """
    # Create fake model with messages
    # Note: GenericFakeChatModel streams by splitting on whitespace
    model = FixedGenericFakeChatModel(
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
            AIMessage(content="the weather in paris is sunny and 72°f today!"),
        ])
    )

    # Stream the first message
    print("\n=== First message (with tool_calls) ===")
    chunks = []
    async for chunk in model.astream("What's the weather?"):
        chunks.append(chunk)
        print(f"Chunk {len(chunks)}: {chunk}")
        print(f"  content: '{chunk.content}'")
        print(f"  tool_calls: {chunk.tool_calls}")

    print(f"\nTotal chunks from first message: {len(chunks)}")

    # Stream the second message
    print("\n=== Second message (regular text) ===")
    chunks2 = []
    async for chunk in model.astream("Thanks!"):
        chunks2.append(chunk)
        print(f"Chunk {len(chunks2)}: {chunk}")
        print(f"  content: '{chunk.content}'")

    print(f"\nTotal chunks from second message: {len(chunks2)}")

    # Verify we got chunks
    assert len(chunks) > 0
    assert len(chunks2) > 0


