from contextlib import asynccontextmanager
from typing import Any

from acp.schema import NewSessionRequest, PromptRequest
from acp.schema import (
    TextContentBlock,
    RequestPermissionRequest,
    RequestPermissionResponse,
    AllowedOutcome,
)
from dirty_equals import IsUUID
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from deepagents_acp.server import DeepagentsACP
from tests.chat_model import GenericFakeChatModel


class FakeAgentSideConnection:
    """Simple fake implementation of AgentSideConnection for testing."""

    def __init__(self) -> None:
        """Initialize the fake connection with an empty calls list."""
        self.calls: list[dict[str, Any]] = []
        self.permission_requests: list[RequestPermissionRequest] = []
        self.permission_response: RequestPermissionResponse | None = None

    async def sessionUpdate(self, notification: Any) -> None:
        """Track sessionUpdate calls."""
        self.calls.append(notification)

    async def requestPermission(
        self, request: RequestPermissionRequest
    ) -> RequestPermissionResponse:
        """Track permission requests and return a mocked response."""
        self.permission_requests.append(request)
        if self.permission_response:
            return self.permission_response
        # Default: approve the action
        return RequestPermissionResponse(
            outcome=AllowedOutcome(
                outcome="selected",
                optionId="allow-once",
            )
        )


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
    middleware: list[Any] | None = None,
):
    """Context manager for testing DeepagentsACP.

    Args:
        messages: List of messages for the fake model to return
        prompt_request: The prompt request to send to the agent
        tools: List of tools to provide to the agent (defaults to [])
        stream_delimiter: How to chunk content when streaming (default: r"(\\s)" for whitespace)
        middleware: Optional middleware to add to the agent graph

    Yields:
        FakeAgentSideConnection: The connection object that tracks sessionUpdate calls
    """
    from deepagents.graph import create_deep_agent

    connection = FakeAgentSideConnection()
    model = GenericFakeChatModel(
        messages=iter(messages),
        stream_delimiter=stream_delimiter,
    )
    tools = tools if tools is not None else []

    # Create the agent graph
    agent_graph = create_deep_agent(
        model=model,
        tools=tools,
        checkpointer=InMemorySaver(),
        middleware=middleware or [],
    )

    deepagents_acp = DeepagentsACP(
        connection=connection,
        agent_graph=agent_graph,
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
            tools=[get_weather_tool],
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

            tool_call_updates = [
                call.model_dump()
                for call in connection.calls
                if call.model_dump()["update"]["sessionUpdate"] == "tool_call_update"
            ]

            # Verify we have exactly 2 tool call updates
            assert len(tool_call_updates) == 2

            # Verify tool call pending with full structure
            assert tool_call_updates[0]["update"] == {
                "sessionUpdate": "tool_call_update",
                "status": "pending",
                "toolCallId": "call_123",
                "title": "get_weather_tool",
                "rawInput": {"location": "Paris, France"},
                "content": None,
                "rawOutput": None,
                "kind": None,
                "locations": None,
                "field_meta": None,
            }

            # Verify tool call completed with full structure
            assert tool_call_updates[1]["update"] == {
                "sessionUpdate": "tool_call_update",
                "status": "completed",
                "toolCallId": "call_123",
                "title": "get_weather_tool",
                "rawInput": None,  # rawInput not included in completed status
                "content": [
                    {
                        "type": "content",
                        "content": {
                            "type": "text",
                            "text": "The weather in Paris, France is sunny and 72°F",
                            "annotations": None,
                            "field_meta": None,
                        },
                    }
                ],
                "rawOutput": "The weather in Paris, France is sunny and 72°F",
                "kind": None,
                "locations": None,
                "field_meta": None,
            }

            # Verify all non-tool-call updates are message chunks
            message_chunks = [
                call.model_dump()
                for call in connection.calls
                if call.model_dump()["update"]["sessionUpdate"] == "agent_message_chunk"
            ]
            assert len(message_chunks) > 0
            for chunk in message_chunks:
                assert chunk["update"]["sessionUpdate"] == "agent_message_chunk"
                assert chunk["update"]["content"]["type"] == "text"


async def test_todo_list_handling() -> None:
    """Test that DeepagentsACP handles todo list updates correctly."""
    from deepagents.graph import create_deep_agent

    prompt_request = PromptRequest(
        sessionId="",  # Will be set by context manager
        prompt=[TextContentBlock(text="Create a shopping list", type="text")],
    )

    # Create a mock connection to track calls
    connection = FakeAgentSideConnection()
    model = GenericFakeChatModel(
        messages=iter([AIMessage(content="I'll create that shopping list for you.")]),
        stream_delimiter=r"(\s)",
    )

    # Create agent graph
    agent_graph = create_deep_agent(
        model=model,
        tools=[get_weather_tool],
        checkpointer=InMemorySaver(),
    )

    deepagents_acp = DeepagentsACP(
        connection=connection,
        agent_graph=agent_graph,
    )

    # Create a new session
    session_response = await deepagents_acp.newSession(
        NewSessionRequest(cwd="/tmp", mcpServers=[])
    )
    session_id = session_response.sessionId
    prompt_request.sessionId = session_id

    # Manually inject a tools update with todos into the agent stream
    # Simulate the graph's behavior by patching the astream method
    agent = deepagents_acp._sessions[session_id]["agent"]
    original_astream = agent.astream

    async def mock_astream(*args, **kwargs):
        # First yield the normal message chunks
        async for item in original_astream(*args, **kwargs):
            yield item

        # Then inject a tools update with todos
        yield (
            "updates",
            {
                "tools": {
                    "todos": [
                        {"content": "Buy fresh bananas", "status": "pending"},
                        {"content": "Buy whole grain bread", "status": "in_progress"},
                        {"content": "Buy organic eggs", "status": "completed"},
                    ],
                    "messages": [],
                }
            },
        )

    agent.astream = mock_astream

    # Call prompt
    await deepagents_acp.prompt(prompt_request)

    # Find the plan update in the calls
    plan_updates = [
        call.model_dump()
        for call in connection.calls
        if call.model_dump()["update"]["sessionUpdate"] == "plan"
    ]

    # Verify we got exactly one plan update with correct structure
    assert len(plan_updates) == 1
    assert plan_updates[0]["update"] == {
        "sessionUpdate": "plan",
        "entries": [
            {
                "content": "Buy fresh bananas",
                "status": "pending",
                "priority": "medium",
                "field_meta": None,
            },
            {
                "content": "Buy whole grain bread",
                "status": "in_progress",
                "priority": "medium",
                "field_meta": None,
            },
            {
                "content": "Buy organic eggs",
                "status": "completed",
                "priority": "medium",
                "field_meta": None,
            },
        ],
        "field_meta": None,
    }


async def test_fake_chat_model_streaming() -> None:
    """Test to verify GenericFakeChatModel stream_delimiter API.

    This test demonstrates the different streaming modes available via stream_delimiter.
    """
    # Test 1: No streaming (stream_delimiter=None) - single chunk
    model_no_stream = GenericFakeChatModel(
        messages=iter([AIMessage(content="Hello world")]),
        stream_delimiter=None,
    )
    chunks = []
    async for chunk in model_no_stream.astream("test"):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0].content == "Hello world"

    # Test 2: Stream on whitespace using regex (default behavior)
    model_whitespace = GenericFakeChatModel(
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
    model_with_tools = GenericFakeChatModel(
        messages=iter(
            [
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
            ]
        ),
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


async def test_human_in_the_loop_approval() -> None:
    """Test that DeepagentsACP handles HITL interrupts and permission requests correctly."""
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from deepagents.graph import create_deep_agent

    prompt_request = PromptRequest(
        sessionId="",  # Will be set below
        prompt=[TextContentBlock(text="What's the weather in Tokyo?", type="text")],
    )

    # Create connection with permission response configured
    connection = FakeAgentSideConnection()
    # Set up the connection to approve the tool call
    connection.permission_response = RequestPermissionResponse(
        outcome=AllowedOutcome(
            outcome="selected",
            optionId="allow-once",
        )
    )

    model = GenericFakeChatModel(
        messages=iter(
            [
                # First message: AI decides to call the tool
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_weather_tool",
                            "args": {"location": "Tokyo, Japan"},
                            "id": "call_tokyo_123",
                            "type": "tool_call",
                        }
                    ],
                ),
                # Second message: AI responds with the weather result after tool execution
                AIMessage(content="The weather in Tokyo is sunny and 72°F!"),
            ]
        ),
        stream_delimiter=r"(\s)",
    )

    # Create agent graph with HITL middleware
    agent_graph = create_deep_agent(
        model=model,
        tools=[get_weather_tool],
        checkpointer=InMemorySaver(),
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"get_weather_tool": True})],
    )

    deepagents_acp = DeepagentsACP(
        connection=connection,
        agent_graph=agent_graph,
    )

    # Create a new session
    session_response = await deepagents_acp.newSession(
        NewSessionRequest(cwd="/tmp", mcpServers=[])
    )
    session_id = session_response.sessionId
    prompt_request.sessionId = session_id

    # Call prompt - this should trigger HITL
    await deepagents_acp.prompt(prompt_request)

    # Verify that a permission request was made with correct structure
    assert len(connection.permission_requests) == 1
    perm_request = connection.permission_requests[0]

    assert {
        "sessionId": perm_request.sessionId,
        "toolCall": {
            "title": perm_request.toolCall.title,
            "rawInput": perm_request.toolCall.rawInput,
            "status": perm_request.toolCall.status,
        },
        "option_ids": [opt.optionId for opt in perm_request.options],
    } == {
        "sessionId": session_id,
        "toolCall": {
            "title": "get_weather_tool",
            "rawInput": {"location": "Tokyo, Japan"},
            "status": "pending",
        },
        "option_ids": ["allow-once", "reject-once"],
    }

    # Verify that tool execution happened after approval
    tool_call_updates = [
        call.model_dump()
        for call in connection.calls
        if call.model_dump()["update"]["sessionUpdate"] == "tool_call_update"
    ]

    assert len(tool_call_updates) == 2
    assert tool_call_updates[0]["update"] == {
        "sessionUpdate": "tool_call_update",
        "status": "pending",
        "title": "get_weather_tool",
        "toolCallId": "call_tokyo_123",
        "rawInput": {"location": "Tokyo, Japan"},
        "content": None,
        "rawOutput": None,
        "kind": None,
        "locations": None,
        "field_meta": None,
    }

    # Check completed status
    completed_update = tool_call_updates[1]["update"]
    assert completed_update["sessionUpdate"] == "tool_call_update"
    assert completed_update["status"] == "completed"
    assert completed_update["title"] == "get_weather_tool"
    assert "Tokyo, Japan" in completed_update["rawOutput"]

    # Verify final AI message was streamed
    message_chunks = [
        call
        for call in connection.calls
        if call.model_dump()["update"]["sessionUpdate"] == "agent_message_chunk"
    ]
    assert len(message_chunks) > 0
