import warnings

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from deepagents.backends.state import StateBackend
from deepagents.graph import create_agent
from deepagents.middleware.subagents import (
    GENERAL_PURPOSE_SUBAGENT,
    TASK_SYSTEM_PROMPT,
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
        assert "task" in agent.nodes["tools"].bound._tools_by_name.keys()
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

    def test_multiple_subagents_with_interrupt_on(self):
        agent = create_agent(
            model="claude-sonnet-4-20250514",
            system_prompt="Use the task tool to call subagents.",
            middleware=[
                SubAgentMiddleware(
                    backend=StateBackend,
                    subagents=[
                        {
                            "name": "subagent1",
                            "description": "First subagent.",
                            "system_prompt": "You are subagent 1.",
                            "model": "claude-sonnet-4-20250514",
                            "tools": [get_weather],
                            "interrupt_on": {"get_weather": True},
                        },
                        {
                            "name": "subagent2",
                            "description": "Second subagent.",
                            "system_prompt": "You are subagent 2.",
                            "model": "claude-sonnet-4-20250514",
                            "tools": [get_weather],
                            "interrupt_on": {"get_weather": True},
                        },
                    ],
                )
            ],
        )
        # This would error if the middleware was accumulated incorrectly
        assert agent is not None

    def test_subagent_middleware_init(self):
        middleware = SubAgentMiddleware(
            backend=StateBackend,
            subagents=[
                {
                    **GENERAL_PURPOSE_SUBAGENT,
                    "model": "gpt-4o-mini",
                    "tools": [],
                }
            ],
        )
        assert middleware is not None
        # System prompt includes TASK_SYSTEM_PROMPT plus available subagent types
        assert middleware.system_prompt.startswith(TASK_SYSTEM_PROMPT)
        assert "Available subagent types:" in middleware.system_prompt
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "task"

    def test_subagent_middleware_with_custom_subagent(self):
        middleware = SubAgentMiddleware(
            backend=StateBackend,
            subagents=[
                {
                    "name": "weather",
                    "description": "Weather subagent",
                    "system_prompt": "Get weather.",
                    "model": "gpt-4o-mini",
                    "tools": [get_weather],
                }
            ],
        )
        assert middleware is not None
        # System prompt includes TASK_SYSTEM_PROMPT plus available subagent types
        assert middleware.system_prompt.startswith(TASK_SYSTEM_PROMPT)
        assert "weather" in middleware.system_prompt

    def test_subagent_middleware_custom_system_prompt(self):
        middleware = SubAgentMiddleware(
            backend=StateBackend,
            subagents=[
                {
                    "name": "weather",
                    "description": "Weather subagent",
                    "system_prompt": "Get weather.",
                    "model": "gpt-4o-mini",
                    "tools": [],
                }
            ],
            system_prompt="Use the task tool to call a subagent.",
        )
        assert middleware is not None
        # Custom system prompt plus available subagent types
        assert middleware.system_prompt.startswith("Use the task tool to call a subagent.")

    # ========== Tests for new API ==========

    def test_new_api_requires_backend(self):
        """Test that the new API requires backend parameter."""
        with pytest.raises(ValueError, match="requires either"):
            SubAgentMiddleware(
                subagents=[
                    {
                        "name": "test",
                        "description": "Test",
                        "system_prompt": "Test.",
                        "model": "gpt-4o-mini",
                        "tools": [],
                    }
                ],
            )

    def test_new_api_requires_subagents(self):
        """Test that the new API requires at least one subagent."""
        with pytest.raises(ValueError, match="At least one subagent"):
            SubAgentMiddleware(
                backend=StateBackend,
                subagents=[],
            )

    def test_new_api_no_deprecation_warning(self):
        """Test that using only new API args does not emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            middleware = SubAgentMiddleware(
                backend=StateBackend,
                subagents=[
                    {
                        "name": "test",
                        "description": "Test subagent",
                        "system_prompt": "Test.",
                        "model": "gpt-4o-mini",
                        "tools": [],
                    }
                ],
            )
            # Filter for DeprecationWarnings only
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0, f"Unexpected deprecation warnings: {deprecation_warnings}"
        assert middleware is not None

    def test_new_api_subagent_requires_model(self):
        """Test that subagents must specify model when using new API."""
        with pytest.raises(ValueError, match="must specify 'model'"):
            SubAgentMiddleware(
                backend=StateBackend,
                subagents=[
                    {
                        "name": "test",
                        "description": "Test",
                        "system_prompt": "Test.",
                        "tools": [],
                        # Missing "model"
                    }
                ],
            )

    def test_new_api_subagent_requires_tools(self):
        """Test that subagents must specify tools when using new API."""
        with pytest.raises(ValueError, match="must specify 'tools'"):
            SubAgentMiddleware(
                backend=StateBackend,
                subagents=[
                    {
                        "name": "test",
                        "description": "Test",
                        "system_prompt": "Test.",
                        "model": "gpt-4o-mini",
                        # Missing "tools"
                    }
                ],
            )

    # ========== Tests for deprecated API ==========

    def test_deprecated_api_still_works(self):
        """Test that the deprecated API still works for backward compatibility."""
        with pytest.warns(DeprecationWarning, match="default_model"):
            middleware = SubAgentMiddleware(
                default_model="gpt-4o-mini",
                default_tools=[get_weather],
                subagents=[
                    {
                        "name": "custom",
                        "description": "Custom subagent",
                        "system_prompt": "You are custom.",
                        "tools": [get_weather],
                    }
                ],
            )
        assert middleware is not None
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "task"
        assert "general-purpose" in middleware.system_prompt
        assert "custom" in middleware.system_prompt

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

    def test_deprecated_api_general_purpose_agent_disabled(self):
        """Test deprecated API with general_purpose_agent=False."""
        with pytest.warns(DeprecationWarning, match="default_model"):
            middleware = SubAgentMiddleware(
                default_model="gpt-4o-mini",
                general_purpose_agent=False,
                subagents=[
                    {
                        "name": "only_agent",
                        "description": "The only agent",
                        "system_prompt": "You are the only one.",
                        "tools": [],
                    }
                ],
            )
        assert middleware is not None
        assert "only_agent" in middleware.system_prompt
        assert "general-purpose" not in middleware.system_prompt

    # ========== Tests for mixing old and new args ==========

    def test_mixed_args_prefers_new_api(self):
        """Test that when both backend and deprecated args are provided, new API is used with warning."""
        with pytest.warns(DeprecationWarning, match="default_model"):
            middleware = SubAgentMiddleware(
                backend=StateBackend,
                subagents=[
                    {
                        "name": "test",
                        "description": "Test subagent",
                        "system_prompt": "Test.",
                        "model": "gpt-4o-mini",
                        "tools": [],
                    }
                ],
                default_model="gpt-4o-mini",  # This is deprecated but still triggers warning
            )
        assert middleware is not None
