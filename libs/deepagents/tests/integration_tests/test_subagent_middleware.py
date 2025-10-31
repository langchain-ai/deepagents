import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from deepagents.graph import create_agent
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import (
    DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    TASK_SYSTEM_PROMPT,
    TASK_TOOL_DESCRIPTION,
    SubAgentMiddleware,
)


@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."


class WeatherMiddleware(AgentMiddleware):
    tools = [get_weather]


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
                    default_model="claude-sonnet-4-20250514",
                    default_tools=[get_weather],
                )
            ],
        )
        assert "task" in agent.nodes["tools"].bound._tools_by_name.keys()
        response = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
        assert response["messages"][1].tool_calls[0]["name"] == "task"
        assert response["messages"][1].tool_calls[0]["args"]["subagent_type"] == "general-purpose"

    def test_defined_subagent(self):
        agent = create_agent(
            model="claude-sonnet-4-20250514",
            system_prompt="Use the task tool to call a subagent.",
            middleware=[
                SubAgentMiddleware(
                    default_model="claude-sonnet-4-20250514",
                    default_tools=[],
                    subagents=[
                        {
                            "name": "weather",
                            "description": "This subagent can get weather in cities.",
                            "system_prompt": "Use the get_weather tool to get the weather in a city.",
                            "tools": [get_weather],
                        }
                    ],
                )
            ],
        )
        assert "task" in agent.nodes["tools"].bound._tools_by_name.keys()
        response = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
        assert response["messages"][1].tool_calls[0]["name"] == "task"
        assert response["messages"][1].tool_calls[0]["args"]["subagent_type"] == "weather"

    def test_defined_subagent_tool_calls(self):
        agent = create_agent(
            model="claude-sonnet-4-20250514",
            system_prompt="Use the task tool to call a subagent.",
            middleware=[
                SubAgentMiddleware(
                    default_model="claude-sonnet-4-20250514",
                    default_tools=[],
                    subagents=[
                        {
                            "name": "weather",
                            "description": "This subagent can get weather in cities.",
                            "system_prompt": "Use the get_weather tool to get the weather in a city.",
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
                    default_model="claude-sonnet-4-20250514",
                    default_tools=[],
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
                    default_model="claude-sonnet-4-20250514",
                    default_tools=[],
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
                    default_model="claude-sonnet-4-20250514",
                    default_tools=[],
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

    def test_multiple_subagents_with_interrupt_on_no_middleware_accumulation(self):
        agent = create_agent(
            model="claude-sonnet-4-20250514",
            system_prompt="Use the task tool to call subagents.",
            middleware=[
                SubAgentMiddleware(
                    default_model="claude-sonnet-4-20250514",
                    default_tools=[],
                    default_middleware=[PatchToolCallsMiddleware()],
                    subagents=[
                        {
                            "name": "subagent1",
                            "description": "First subagent.",
                            "system_prompt": "You are subagent 1.",
                            "tools": [get_weather],
                            "interrupt_on": {"get_weather": True},
                        },
                        {
                            "name": "subagent2",
                            "description": "Second subagent.",
                            "system_prompt": "You are subagent 2.",
                            "tools": [get_weather],
                            "interrupt_on": {"get_weather": True},
                        },
                    ],
                )
            ],
        )
        # This would error if the default middleware was accumulated
        assert True

    def test_subagent_middleware_init(self):
        middleware = SubAgentMiddleware(
            default_model="gpt-4o-mini",
        )
        assert middleware is not None
        assert middleware.system_prompt is TASK_SYSTEM_PROMPT
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "task"
        expected_desc = TASK_TOOL_DESCRIPTION.format(available_agents=f"- general-purpose: {DEFAULT_GENERAL_PURPOSE_DESCRIPTION}")
        assert middleware.tools[0].description == expected_desc

    def test_default_subagent_with_tools(self):
        middleware = SubAgentMiddleware(
            default_model="gpt-4o-mini",
            default_tools=[],
        )
        assert middleware is not None
        assert middleware.system_prompt == TASK_SYSTEM_PROMPT

    def test_default_subagent_custom_system_prompt(self):
        middleware = SubAgentMiddleware(
            default_model="gpt-4o-mini",
            default_tools=[],
            system_prompt="Use the task tool to call a subagent.",
        )
        assert middleware is not None
        assert middleware.system_prompt == "Use the task tool to call a subagent."
