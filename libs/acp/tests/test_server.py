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


@tool(description="Sample tool")
def sample_tool(sample_input: str) -> str:
    """A sample tool that returns the input string."""
    return sample_input


#
# class TestDeepAgentEndToEnd:
#     """Test suite for end-to-end deepagent functionality with fake LLM."""
#
#     def test_deep_agent_with_fake_llm_basic(self) -> None:
#         """Test basic deepagent functionality with a fake LLM model.
#
#         This test verifies that a deepagent can be created and invoked with
#         a fake LLM model that returns predefined responses.
#         """
#         # Create a fake model that returns predefined messages
#         model = FixedGenericFakeChatModel(
#             messages=iter(
#                 [
#                     AIMessage(
#                         content="I'll use the sample_tool to process your request.",
#                         tool_calls=[
#                             {
#                                 "name": "write_todos",
#                                 "args": {"todos": []},
#                                 "id": "call_1",
#                                 "type": "tool_call",
#                             }
#                         ],
#                     ),
#                     AIMessage(
#                         content="Task completed successfully!",
#                     ),
#                 ]
#             )
#         )
#
#         # Create a deep agent with the fake model
#         agent = create_deep_agent(model=model)
#
#         # Invoke the agent with a simple message
#         result = agent.invoke({"messages": [HumanMessage(content="Hello, agent!")]})
#
#         # Verify the agent executed correctly
#         assert "messages" in result
#         assert len(result["messages"]) > 0
#
#         # Verify we got AI responses
#         ai_messages = [msg for msg in result["messages"] if msg.type == "ai"]
#         assert len(ai_messages) > 0
#
#         # Verify the final AI message contains our expected content
#         final_ai_message = ai_messages[-1]
#         assert "Task completed successfully!" in final_ai_message.content
#
#     def test_deep_agent_with_fake_llm_with_tools(self) -> None:
#         """Test deepagent with tools using a fake LLM model.
#
#         This test verifies that a deepagent can handle tool calls correctly
#         when using a fake LLM model.
#         """
#         # Create a fake model that calls sample_tool
#         model = FixedGenericFakeChatModel(
#             messages=iter(
#                 [
#                     AIMessage(
#                         content="",
#                         tool_calls=[
#                             {
#                                 "name": "sample_tool",
#                                 "args": {"sample_input": "test input"},
#                                 "id": "call_1",
#                                 "type": "tool_call",
#                             }
#                         ],
#                     ),
#                     AIMessage(
#                         content="I called the sample_tool with 'test input'.",
#                     ),
#                 ]
#             )
#         )
#
#         # Create a deep agent with the fake model and sample_tool
#         agent = create_deep_agent(model=model, tools=[sample_tool])
#
#         # Invoke the agent
#         result = agent.invoke(
#             {"messages": [HumanMessage(content="Use the sample tool")]}
#         )
#
#         # Verify the agent executed correctly
#         assert "messages" in result
#
#         # Verify tool was called
#         tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
#         assert len(tool_messages) > 0
#
#         # Verify the tool message contains our expected input
#         assert any("test input" in msg.content for msg in tool_messages)
#
#     def test_deep_agent_with_fake_llm_filesystem_tool(self) -> None:
#         """Test deepagent with filesystem tools using a fake LLM model.
#
#         This test verifies that a deepagent can use the built-in filesystem
#         tools (ls, read_file, etc.) with a fake LLM model.
#         """
#         # Create a fake model that uses filesystem tools
#         model = FixedGenericFakeChatModel(
#             messages=iter(
#                 [
#                     AIMessage(
#                         content="",
#                         tool_calls=[
#                             {
#                                 "name": "ls",
#                                 "args": {"path": "."},
#                                 "id": "call_1",
#                                 "type": "tool_call",
#                             }
#                         ],
#                     ),
#                     AIMessage(
#                         content="I've listed the files in the current directory.",
#                     ),
#                 ]
#             )
#         )
#
#         # Create a deep agent with the fake model
#         agent = create_deep_agent(model=model)
#
#         # Invoke the agent
#         result = agent.invoke({"messages": [HumanMessage(content="List files")]})
#
#         # Verify the agent executed correctly
#         assert "messages" in result
#
#         # Verify ls tool was called
#         tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
#         assert len(tool_messages) > 0
#
#     def test_deep_agent_with_fake_llm_multiple_tool_calls(self) -> None:
#         """Test deepagent with multiple tool calls using a fake LLM model.
#
#         This test verifies that a deepagent can handle multiple sequential
#         tool calls with a fake LLM model.
#         """
#         # Create a fake model that makes multiple tool calls
#         model = FixedGenericFakeChatModel(
#             messages=iter(
#                 [
#                     AIMessage(
#                         content="",
#                         tool_calls=[
#                             {
#                                 "name": "sample_tool",
#                                 "args": {"sample_input": "first call"},
#                                 "id": "call_1",
#                                 "type": "tool_call",
#                             }
#                         ],
#                     ),
#                     AIMessage(
#                         content="",
#                         tool_calls=[
#                             {
#                                 "name": "sample_tool",
#                                 "args": {"sample_input": "second call"},
#                                 "id": "call_2",
#                                 "type": "tool_call",
#                             }
#                         ],
#                     ),
#                     AIMessage(
#                         content="I completed both tool calls successfully.",
#                     ),
#                 ]
#             )
#         )
#
#         # Create a deep agent with the fake model and sample_tool
#         agent = create_deep_agent(model=model, tools=[sample_tool])
#
#         # Invoke the agent
#         result = agent.invoke(
#             {"messages": [HumanMessage(content="Use sample tool twice")]}
#         )
#
#         # Verify the agent executed correctly
#         assert "messages" in result
#
#         # Verify multiple tool calls occurred
#         tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
#         assert len(tool_messages) >= 2
#
#         # Verify both inputs were used
#         tool_contents = [msg.content for msg in tool_messages]
#         assert any("first call" in content for content in tool_contents)
#         assert any("second call" in content for content in tool_contents)
