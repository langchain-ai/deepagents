from typing import ClassVar

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from deepagents.backends.state import StateBackend
from deepagents.graph import create_agent
from deepagents.middleware.subagents import (
    GENERAL_PURPOSE_SUBAGENT,
    SubAgentMiddleware,
)


@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."


class WeatherMiddleware(AgentMiddleware):
    tools: ClassVar = [get_weather]


def assert_expected_subgraph_actions(expected_tool_calls, agent, inputs):
    current_idx = 0
    for update in agent.stream(
        inputs,
        subgraphs=True,
        stream_mode="updates",
    ):
        if "model" in update[1]:
            ai_message = update[1]["model"]["messages"][-1]
            tool_calls = ai_message.tool_calls
            for tool_call in tool_calls:
                if tool_call["name"] == expected_tool_calls[current_idx]["name"]:
                    if "model" in expected_tool_calls[current_idx]:
                        assert ai_message.response_metadata["model_name"] == expected_tool_calls[current_idx]["model"]
                    for arg in expected_tool_calls[current_idx]["args"]:
                        assert arg in tool_call["args"]
                        assert tool_call["args"][arg] == expected_tool_calls[current_idx]["args"][arg]
                    current_idx += 1
    assert current_idx == len(expected_tool_calls)


@pytest.mark.requires("langchain_anthropic", "langchain_openai")
class TestSubagentMiddleware:
    """Integration tests for the SubagentMiddleware class."""

    def test_general_purpose_subagent(self):
        agent = create_agent(
            model="claude-sonnet-4-20250514",
            system_prompt="Use the general-purpose subagent to get the weather in a city.",
            middleware=[
                SubAgentMiddleware(
                    backend=StateBackend,
                    subagents=[
                        {
                            **GENERAL_PURPOSE_SUBAGENT,
                            "model": "claude-sonnet-4-20250514",
                            "tools": [get_weather],
                        }
                    ],
                )
            ],
        )
        assert "task" in agent.nodes["tools"].bound._tools_by_name
        response = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
        assert response["messages"][1].tool_calls[0]["name"] == "task"
        assert response["messages"][1].tool_calls[0]["args"]["subagent_type"] == "general-purpose"

    def test_defined_subagent_tool_calls(self):
        agent = create_agent(
            model="claude-sonnet-4-20250514",
            system_prompt="Use the task tool to call a subagent.",
            middleware=[
                SubAgentMiddleware(
                    backend=StateBackend,
                    subagents=[
                        {
                            "name": "weather",
                            "description": "This subagent can get weather in cities.",
                            "system_prompt": "Use the get_weather tool to get the weather in a city.",
                            "model": "claude-sonnet-4-20250514",
                            "tools": [get_weather],
                        }
                    ],
                )
            ],
        )
        expected_tool_calls = [
            {"name": "task", "args": {"subagent_type": "weather"}},
            {"name": "get_weather", "args": {}},
        ]
        assert_expected_subgraph_actions(
            expected_tool_calls,
            agent,
            {"messages": [HumanMessage(content="What is the weather in Tokyo?")]},
        )

    def test_defined_subagent_custom_model(self):
        agent = create_agent(
            model="claude-sonnet-4-20250514",
            system_prompt="Use the task tool to call a subagent.",
            middleware=[
                SubAgentMiddleware(
                    backend=StateBackend,
                    subagents=[
                        {
                            "name": "weather",
                            "description": "This subagent can get weather in cities.",
                            "system_prompt": "Use the get_weather tool to get the weather in a city.",
                            "tools": [get_weather],
                            "model": "gpt-4.1",
                        }
                    ],
                )
            ],
        )
        expected_tool_calls = [
            {
                "name": "task",
                "args": {"subagent_type": "weather"},
                "model": "claude-sonnet-4-20250514",
            },
            {"name": "get_weather", "args": {}, "model": "gpt-4.1-2025-04-14"},
        ]
        assert_expected_subgraph_actions(
            expected_tool_calls,
            agent,
            {"messages": [HumanMessage(content="What is the weather in Tokyo?")]},
        )

    def test_defined_subagent_custom_middleware(self):
        agent = create_agent(
            model="claude-sonnet-4-20250514",
            system_prompt="Use the task tool to call a subagent.",
            middleware=[
                SubAgentMiddleware(
                    backend=StateBackend,
                    subagents=[
                        {
                            "name": "weather",
                            "description": "This subagent can get weather in cities.",
                            "system_prompt": "Use the get_weather tool to get the weather in a city.",
                            "tools": [],  # No tools, only in middleware
                            "model": "gpt-4.1",
                            "middleware": [WeatherMiddleware()],
                        }
                    ],
                )
            ],
        )
        expected_tool_calls = [
            {
                "name": "task",
                "args": {"subagent_type": "weather"},
                "model": "claude-sonnet-4-20250514",
            },
            {"name": "get_weather", "args": {}, "model": "gpt-4.1-2025-04-14"},
        ]
        assert_expected_subgraph_actions(
            expected_tool_calls,
            agent,
            {"messages": [HumanMessage(content="What is the weather in Tokyo?")]},
        )

    def test_defined_subagent_custom_runnable(self):
        custom_subagent = create_agent(
            model="gpt-4.1-2025-04-14",
            system_prompt="Use the get_weather tool to get the weather in a city.",
            tools=[get_weather],
        )
        agent = create_agent(
            model="claude-sonnet-4-20250514",
            system_prompt="Use the task tool to call a subagent.",
            middleware=[
                SubAgentMiddleware(
                    backend=StateBackend,
                    subagents=[
                        {
                            "name": "weather",
                            "description": "This subagent can get weather in cities.",
                            "runnable": custom_subagent,
                        }
                    ],
                )
            ],
        )
        expected_tool_calls = [
            {
                "name": "task",
                "args": {"subagent_type": "weather"},
                "model": "claude-sonnet-4-20250514",
            },
            {"name": "get_weather", "args": {}, "model": "gpt-4.1-2025-04-14"},
        ]
        assert_expected_subgraph_actions(
            expected_tool_calls,
            agent,
            {"messages": [HumanMessage(content="What is the weather in Tokyo?")]},
        )

    def test_deprecated_api_subagents_inherit_model(self):
        """Test that subagents inherit default_model when not specified."""
        with pytest.warns(DeprecationWarning, match="default_model"):
            agent = create_agent(
                model="claude-sonnet-4-20250514",
                system_prompt="Use the task tool to call a subagent.",
                middleware=[
                    SubAgentMiddleware(
                        default_model="gpt-4.1",  # Custom subagent should inherit this
                        default_tools=[get_weather],
                        subagents=[
                            {
                                "name": "custom",
                                "description": "Custom subagent that gets weather.",
                                "system_prompt": "Use the get_weather tool.",
                                # No model specified - should inherit from default_model
                            }
                        ],
                    )
                ],
            )
        # Verify the custom subagent uses the inherited model
        expected_tool_calls = [
            {"name": "task", "args": {"subagent_type": "custom"}, "model": "claude-sonnet-4-20250514"},
            {"name": "get_weather", "args": {}, "model": "gpt-4.1-2025-04-14"},  # Inherited model
        ]
        assert_expected_subgraph_actions(
            expected_tool_calls,
            agent,
            {"messages": [HumanMessage(content="What is the weather in Tokyo?")]},
        )

    def test_deprecated_api_subagents_inherit_tools(self):
        """Test that subagents inherit default_tools when not specified."""
        with pytest.warns(DeprecationWarning, match="default_model"):
            agent = create_agent(
                model="claude-sonnet-4-20250514",
                system_prompt="Use the task tool to call a subagent.",
                middleware=[
                    SubAgentMiddleware(
                        default_model="claude-sonnet-4-20250514",
                        default_tools=[get_weather],  # Custom subagent should inherit this
                        subagents=[
                            {
                                "name": "custom",
                                "description": "Custom subagent that gets weather.",
                                "system_prompt": "Use the get_weather tool to get weather.",
                                # No tools specified - should inherit from default_tools
                            }
                        ],
                    )
                ],
            )
        # Verify the custom subagent can use the inherited tools
        expected_tool_calls = [
            {"name": "task", "args": {"subagent_type": "custom"}},
            {"name": "get_weather", "args": {}},  # Inherited tool
        ]
        assert_expected_subgraph_actions(
            expected_tool_calls,
            agent,
            {"messages": [HumanMessage(content="What is the weather in Tokyo?")]},
        )
