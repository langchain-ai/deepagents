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
        1. Streams the AI message content as chunks
        2. Detects and executes the tool call
        3. Sends tool call progress notifications
        4. Continues with a response based on the tool result

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
                    content="I'll check the weather for you.",
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
            # The fake chat model streams on whitespace tokens, creating multiple chunks
            # Filter to find different types of updates
            message_chunks = [c for c in connection.calls if c.update.sessionUpdate == "agent_message_chunk"]
            tool_call_updates = [c for c in connection.calls if c.update.sessionUpdate == "tool_call_update"]

            # Verify we got message chunks (from streaming)
            assert len(message_chunks) > 0, "Should have message chunks from streaming"

            # Reconstruct messages from chunks
            all_text = "".join(c.update.content.text for c in message_chunks)

            # Should contain text from the first AI message
            assert "I'll check the weather for you." in all_text

            # If tool calls were executed, verify the details
            first_tool_update = tool_call_updates[0].model_dump()
            assert first_tool_update == {
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
                    "status": "running",
                },
            }

            # And the second message should also appear
            assert "The weather in Paris is sunny and 72°F today!" in all_text
