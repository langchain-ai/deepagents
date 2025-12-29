"""End-to-end unit tests for deepagents with fake LLM models."""

from collections.abc import Callable, Sequence
from typing import Any
from unittest.mock import patch

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool

from deepagents.graph import create_deep_agent


@tool(description="Sample tool")
def sample_tool(sample_input: str) -> str:
    """A sample tool that returns the input string."""
    return sample_input


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


class TestDeepAgentEndToEnd:
    """Test suite for end-to-end deepagent functionality with fake LLM."""

    def test_deep_agent_with_fake_llm_basic(self) -> None:
        """Test basic deepagent functionality with a fake LLM model.

        This test verifies that a deepagent can be created and invoked with
        a fake LLM model that returns predefined responses.
        """
        # Create a fake model that returns predefined messages
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="I'll use the sample_tool to process your request.",
                        tool_calls=[
                            {
                                "name": "write_todos",
                                "args": {"todos": []},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="Task completed successfully!",
                    ),
                ]
            )
        )

        # Create a deep agent with the fake model
        agent = create_deep_agent(model=model)

        # Invoke the agent with a simple message
        result = agent.invoke({"messages": [HumanMessage(content="Hello, agent!")]})

        # Verify the agent executed correctly
        assert "messages" in result
        assert len(result["messages"]) > 0

        # Verify we got AI responses
        ai_messages = [msg for msg in result["messages"] if msg.type == "ai"]
        assert len(ai_messages) > 0

        # Verify the final AI message contains our expected content
        final_ai_message = ai_messages[-1]
        assert "Task completed successfully!" in final_ai_message.content

    def test_deep_agent_with_fake_llm_with_tools(self) -> None:
        """Test deepagent with tools using a fake LLM model.

        This test verifies that a deepagent can handle tool calls correctly
        when using a fake LLM model.
        """
        # Create a fake model that calls sample_tool
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "sample_tool",
                                "args": {"sample_input": "test input"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="I called the sample_tool with 'test input'.",
                    ),
                ]
            )
        )

        # Create a deep agent with the fake model and sample_tool
        agent = create_deep_agent(model=model, tools=[sample_tool])

        # Invoke the agent
        result = agent.invoke({"messages": [HumanMessage(content="Use the sample tool")]})

        # Verify the agent executed correctly
        assert "messages" in result

        # Verify tool was called
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) > 0

        # Verify the tool message contains our expected input
        assert any("test input" in msg.content for msg in tool_messages)

    def test_deep_agent_with_fake_llm_filesystem_tool(self) -> None:
        """Test deepagent with filesystem tools using a fake LLM model.

        This test verifies that a deepagent can use the built-in filesystem
        tools (ls, read_file, etc.) with a fake LLM model.
        """
        # Create a fake model that uses filesystem tools
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ls",
                                "args": {"path": "."},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="I've listed the files in the current directory.",
                    ),
                ]
            )
        )

        # Create a deep agent with the fake model
        agent = create_deep_agent(model=model)

        # Invoke the agent
        result = agent.invoke({"messages": [HumanMessage(content="List files")]})

        # Verify the agent executed correctly
        assert "messages" in result

        # Verify ls tool was called
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) > 0

    def test_deep_agent_with_fake_llm_multiple_tool_calls(self) -> None:
        """Test deepagent with multiple tool calls using a fake LLM model.

        This test verifies that a deepagent can handle multiple sequential
        tool calls with a fake LLM model.
        """
        # Create a fake model that makes multiple tool calls
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "sample_tool",
                                "args": {"sample_input": "first call"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "sample_tool",
                                "args": {"sample_input": "second call"},
                                "id": "call_2",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="I completed both tool calls successfully.",
                    ),
                ]
            )
        )

        # Create a deep agent with the fake model and sample_tool
        agent = create_deep_agent(model=model, tools=[sample_tool])

        # Invoke the agent
        result = agent.invoke({"messages": [HumanMessage(content="Use sample tool twice")]})

        # Verify the agent executed correctly
        assert "messages" in result

        # Verify multiple tool calls occurred
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) >= 2

        # Verify both inputs were used
        tool_contents = [msg.content for msg in tool_messages]
        assert any("first call" in content for content in tool_contents)
        assert any("second call" in content for content in tool_contents)

    def test_deep_agent_with_string_model_name(self) -> None:
        """Test that create_deep_agent handles string model names correctly.

        This test verifies that when a model name is passed as a string,
        it is properly initialized using init_chat_model instead of
        causing an AttributeError when accessing the profile attribute.
        """
        # Mock init_chat_model to return a fake model
        fake_model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="Response from string-initialized model.",
                    )
                ]
            )
        )

        with patch("deepagents.graph.init_chat_model", return_value=fake_model):
            # This should not raise AttributeError: 'str' object has no attribute 'profile'
            agent = create_deep_agent(model="claude-sonnet-4-5-20250929", tools=[sample_tool])

            # Verify agent was created successfully
            assert agent is not None

            # Invoke the agent to ensure it works
            result = agent.invoke({"messages": [HumanMessage(content="Test message")]})

            # Verify the agent executed correctly
            assert "messages" in result
            assert len(result["messages"]) > 0
