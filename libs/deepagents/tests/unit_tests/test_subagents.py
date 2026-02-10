"""Tests for sub-agent middleware functionality.

This module contains tests for the subagent system, focusing on how subagents
are invoked, how they return results, and how state is managed between parent
and child agents.
"""

import warnings
from pathlib import Path
from typing import Any, TypedDict

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.graph import create_deep_agent
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


def _make_skill_content(name: str, description: str) -> str:
    """Create SKILL.md content with YAML frontmatter."""
    return f"""---
name: {name}
description: {description}
---

# {name.title()} Skill

Instructions go here.
"""


class TestSubAgents:
    """Tests for sub-agent middleware functionality."""

    def test_subagent_returns_final_message_as_tool_result(self) -> None:
        """Test that a subagent's final message is returned as a ToolMessage.

        This test verifies the core subagent functionality:
        1. Parent agent invokes the 'task' tool to launch a subagent
        2. Subagent executes and returns a result
        3. The subagent's final message is extracted and returned to the parent
           as a ToolMessage in the parent's message list
        4. Only the final message content is included (not the full conversation)

        The response flow is:
        - Parent receives ToolMessage with content from subagent's last AIMessage
        - State updates (excluding messages/todos/structured_response) are merged
        - Parent can then process the subagent's response and continue
        """
        # Create the parent agent's chat model that will call the subagent
        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    # First response: invoke the task tool to launch subagent
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Calculate the sum of 2 and 3",
                                    "subagent_type": "general-purpose",
                                },
                                "id": "call_calculate_sum",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    # Second response: acknowledge the subagent's result
                    AIMessage(content="The calculation has been completed."),
                ]
            )
        )

        # Create the subagent's chat model that will handle the calculation
        subagent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="The sum of 2 and 3 is 5."),
                ]
            )
        )

        # Create the compiled subagent
        compiled_subagent = create_agent(model=subagent_chat_model)

        # Create the parent agent with subagent support
        parent_agent = create_deep_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            subagents=[
                CompiledSubAgent(
                    name="general-purpose",
                    description="A general-purpose agent for various tasks.",
                    runnable=compiled_subagent,
                )
            ],
        )

        # Invoke the parent agent with an initial message
        result = parent_agent.invoke(
            {"messages": [HumanMessage(content="What is 2 + 3?")]},
            config={"configurable": {"thread_id": "test_thread_calculation"}},
        )

        # Verify the result contains messages
        assert "messages" in result, "Result should contain messages key"
        assert len(result["messages"]) > 0, "Result should have at least one message"

        # Find the ToolMessage that contains the subagent's response
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) > 0, "Should have at least one ToolMessage from subagent"

        # Verify the ToolMessage contains the subagent's final response
        subagent_tool_message = tool_messages[0]
        assert "The sum of 2 and 3 is 5." in subagent_tool_message.content, "ToolMessage should contain subagent's final message content"

    def test_multiple_subagents_invoked_in_parallel(self) -> None:
        """Test that multiple different subagents can be launched in parallel.

        This test verifies parallel execution with distinct subagent types:
        1. Parent agent makes a single AIMessage with multiple tool_calls
        2. Two different subagents are invoked concurrently (math-adder and math-multiplier)
        3. Each specialized subagent completes its task independently
        4. Both subagent results are returned as separate ToolMessages
        5. Parent agent receives both results and can synthesize them

        The parallel execution pattern is important for:
        - Reducing latency when tasks are independent
        - Efficient resource utilization
        - Handling multi-part user requests with specialized agents
        """
        # Create the parent agent's chat model that will call both subagents in parallel
        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    # First response: invoke TWO different task tools in parallel
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Calculate the sum of 5 and 7",
                                    "subagent_type": "math-adder",
                                },
                                "id": "call_addition",
                                "type": "tool_call",
                            },
                            {
                                "name": "task",
                                "args": {
                                    "description": "Calculate the product of 4 and 6",
                                    "subagent_type": "math-multiplier",
                                },
                                "id": "call_multiplication",
                                "type": "tool_call",
                            },
                        ],
                    ),
                    # Second response: acknowledge both results
                    AIMessage(content="Both calculations completed successfully."),
                ]
            )
        )

        # Create specialized subagent models - each handles a specific math operation
        addition_subagent_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="The sum of 5 and 7 is 12."),
                ]
            )
        )

        multiplication_subagent_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="The product of 4 and 6 is 24."),
                ]
            )
        )

        # Compile the two different specialized subagents
        addition_subagent = create_agent(model=addition_subagent_model)
        multiplication_subagent = create_agent(model=multiplication_subagent_model)

        # Create the parent agent with BOTH specialized subagents
        parent_agent = create_deep_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            subagents=[
                CompiledSubAgent(
                    name="math-adder",
                    description="Specialized agent for addition operations.",
                    runnable=addition_subagent,
                ),
                CompiledSubAgent(
                    name="math-multiplier",
                    description="Specialized agent for multiplication operations.",
                    runnable=multiplication_subagent,
                ),
            ],
        )

        # Invoke the parent agent with a request that triggers parallel subagent calls
        result = parent_agent.invoke(
            {"messages": [HumanMessage(content="What is 5+7 and what is 4*6?")]},
            config={"configurable": {"thread_id": "test_thread_parallel"}},
        )

        # Verify the result contains messages
        assert "messages" in result, "Result should contain messages key"

        # Find all ToolMessages - should have one for each subagent invocation
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 2, f"Should have exactly 2 ToolMessages (one per subagent), but got {len(tool_messages)}"

        # Create a lookup map from tool_call_id to ToolMessage for precise verification
        tool_messages_by_id = {msg.tool_call_id: msg for msg in tool_messages}

        # Verify we have both expected tool call IDs
        assert "call_addition" in tool_messages_by_id, "Should have response from addition subagent"
        assert "call_multiplication" in tool_messages_by_id, "Should have response from multiplication subagent"

        # Verify the exact content of each response by looking up the specific tool message
        addition_tool_message = tool_messages_by_id["call_addition"]
        assert addition_tool_message.content == "The sum of 5 and 7 is 12.", (
            f"Addition subagent should return exact message, got: {addition_tool_message.content}"
        )

        multiplication_tool_message = tool_messages_by_id["call_multiplication"]
        assert multiplication_tool_message.content == "The product of 4 and 6 is 24.", (
            f"Multiplication subagent should return exact message, got: {multiplication_tool_message.content}"
        )

    def test_agent_with_structured_output_tool_strategy(self) -> None:
        """Test that an agent with ToolStrategy properly generates structured output.

        This test verifies the structured output setup:
        1. Define a Pydantic model as the response schema
        2. Configure agent with ToolStrategy for structured output
        3. Fake model calls the structured output tool
        4. Agent validates and returns the structured response
        5. The structured_response key contains the validated Pydantic instance

        This validates our understanding of how to set up structured output
        correctly using the fake model for testing.
        """

        # Define the Pydantic model for structured output
        class WeatherReport(BaseModel):
            """Structured weather information."""

            location: str = Field(description="The city or location for the weather report")
            temperature: float = Field(description="Temperature in Celsius")
            condition: str = Field(description="Weather condition (e.g., sunny, rainy)")

        # Create a fake model that calls the structured output tool
        # The tool name will be the schema class name: "WeatherReport"
        fake_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "WeatherReport",
                                "args": {
                                    "location": "San Francisco",
                                    "temperature": 18.5,
                                    "condition": "sunny",
                                },
                                "id": "call_weather_report",
                                "type": "tool_call",
                            }
                        ],
                    ),
                ]
            )
        )

        # Create agent with ToolStrategy for structured output
        agent = create_agent(
            model=fake_model,
            response_format=ToolStrategy(schema=WeatherReport),
        )

        # Invoke the agent
        result = agent.invoke({"messages": [HumanMessage(content="What's the weather in San Francisco?")]})

        # Verify the structured_response key exists in the result
        assert "structured_response" in result, "Result should contain structured_response key"

        # Verify the structured response is the correct type
        structured_response = result["structured_response"]
        assert isinstance(structured_response, WeatherReport), f"Expected WeatherReport instance, got {type(structured_response)}"

        # Verify the structured response has the correct values
        expected_response = WeatherReport(location="San Francisco", temperature=18.5, condition="sunny")
        assert structured_response == expected_response, f"Expected {expected_response}, got {structured_response}"

    def test_parallel_subagents_with_todo_lists(self) -> None:
        """Test that multiple subagents can manage their own isolated todo lists.

        This test verifies that:
        1. Multiple subagents can be invoked in parallel
        2. Each subagent can use write_todos to manage its own todo list
        3. Todo lists are properly isolated to each subagent (not merged into parent)
        4. Parent receives clean ToolMessages from each subagent
        5. The 'todos' key is excluded from parent state per _EXCLUDED_STATE_KEYS

        This validates that todo list state isolation works correctly in parallel execution.
        """
        # Create parent agent's chat model that calls two subagents in parallel
        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    # First response: invoke TWO subagents in parallel
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Research the history of Python programming language",
                                    "subagent_type": "python-researcher",
                                },
                                "id": "call_research_python",
                                "type": "tool_call",
                            },
                            {
                                "name": "task",
                                "args": {
                                    "description": "Research the history of JavaScript programming language",
                                    "subagent_type": "javascript-researcher",
                                },
                                "id": "call_research_javascript",
                                "type": "tool_call",
                            },
                        ],
                    ),
                    # Second response: acknowledge both results
                    AIMessage(content="Both research tasks completed successfully."),
                ]
            )
        )

        # Create first subagent that uses write_todos and returns a result
        python_subagent_model = GenericFakeChatModel(
            messages=iter(
                [
                    # First: write some todos
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_todos",
                                "args": {
                                    "todos": [
                                        {
                                            "content": "Search for Python history",
                                            "status": "in_progress",
                                            "activeForm": "Searching for Python history",
                                        },
                                        {"content": "Summarize findings", "status": "pending", "activeForm": "Summarizing findings"},
                                    ]
                                },
                                "id": "call_write_todos_python_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    # Second: update todos and return final message
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_todos",
                                "args": {
                                    "todos": [
                                        {"content": "Search for Python history", "status": "completed", "activeForm": "Searching for Python history"},
                                        {"content": "Summarize findings", "status": "completed", "activeForm": "Summarizing findings"},
                                    ]
                                },
                                "id": "call_write_todos_python_2",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    # Final result message
                    AIMessage(content="Python was created by Guido van Rossum and released in 1991."),
                ]
            )
        )

        # Create second subagent that uses write_todos and returns a result
        javascript_subagent_model = GenericFakeChatModel(
            messages=iter(
                [
                    # First: write some todos
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_todos",
                                "args": {
                                    "todos": [
                                        {
                                            "content": "Search for JavaScript history",
                                            "status": "in_progress",
                                            "activeForm": "Searching for JavaScript history",
                                        },
                                        {"content": "Compile summary", "status": "pending", "activeForm": "Compiling summary"},
                                    ]
                                },
                                "id": "call_write_todos_js_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    # Second: update todos and return final message
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_todos",
                                "args": {
                                    "todos": [
                                        {
                                            "content": "Search for JavaScript history",
                                            "status": "completed",
                                            "activeForm": "Searching for JavaScript history",
                                        },
                                        {"content": "Compile summary", "status": "completed", "activeForm": "Compiling summary"},
                                    ]
                                },
                                "id": "call_write_todos_js_2",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    # Final result message
                    AIMessage(content="JavaScript was created by Brendan Eich at Netscape in 1995."),
                ]
            )
        )

        python_research_agent = create_agent(
            model=python_subagent_model,
            middleware=[TodoListMiddleware()],
        )

        javascript_research_agent = create_agent(
            model=javascript_subagent_model,
            middleware=[TodoListMiddleware()],
        )

        # Create parent agent with both specialized subagents
        parent_agent = create_deep_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            subagents=[
                CompiledSubAgent(
                    name="python-researcher",
                    description="Agent specialized in Python research.",
                    runnable=python_research_agent,
                ),
                CompiledSubAgent(
                    name="javascript-researcher",
                    description="Agent specialized in JavaScript research.",
                    runnable=javascript_research_agent,
                ),
            ],
        )

        # Invoke the parent agent
        result = parent_agent.invoke(
            {"messages": [HumanMessage(content="Research Python and JavaScript history")]},
            config={"configurable": {"thread_id": "test_thread_todos"}},
        )

        # Verify the result contains messages
        assert "messages" in result, "Result should contain messages key"

        # Find all ToolMessages from the subagents
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 2, f"Should have exactly 2 ToolMessages, got {len(tool_messages)}"

        # Create lookup map by tool_call_id
        tool_messages_by_id = {msg.tool_call_id: msg for msg in tool_messages}

        # Verify both expected tool call IDs are present
        assert "call_research_python" in tool_messages_by_id, "Should have response from Python researcher"
        assert "call_research_javascript" in tool_messages_by_id, "Should have response from JavaScript researcher"

        # Verify that todos are NOT in the parent agent's final state
        # (they should be excluded per _EXCLUDED_STATE_KEYS)
        assert "todos" not in result, "Parent agent state should not contain todos key (it should be excluded per _EXCLUDED_STATE_KEYS)"

        # Verify the final messages contain the research results
        python_tool_message = tool_messages_by_id["call_research_python"]
        assert "Python was created by Guido van Rossum" in python_tool_message.content, (
            f"Expected Python research result in message, got: {python_tool_message.content}"
        )

        javascript_tool_message = tool_messages_by_id["call_research_javascript"]
        assert "JavaScript was created by Brendan Eich" in javascript_tool_message.content, (
            f"Expected JavaScript research result in message, got: {javascript_tool_message.content}"
        )

    def test_parallel_subagents_with_different_structured_outputs(self) -> None:
        """Test that multiple subagents with different structured outputs work correctly.

        This test verifies that:
        1. Two different subagents can be invoked in parallel
        2. Each subagent has its own structured output schema
        3. Structured responses are properly excluded from parent state (per _EXCLUDED_STATE_KEYS)
        4. Parent receives clean ToolMessages from each subagent
        5. Each subagent's structured_response stays isolated to that subagent

        This validates that structured_response exclusion prevents schema conflicts
        between parent and subagent agents.
        """

        # Define structured output schemas for the two specialized subagents
        class CityWeather(BaseModel):
            """Weather information for a city."""

            city: str = Field(description="Name of the city")
            temperature_celsius: float = Field(description="Temperature in Celsius")
            humidity_percent: int = Field(description="Humidity percentage")

        class CityPopulation(BaseModel):
            """Population statistics for a city."""

            city: str = Field(description="Name of the city")
            population: int = Field(description="Total population")
            metro_area_population: int = Field(description="Metropolitan area population")

        # Create parent agent's chat model that calls both subagents in parallel
        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    # First response: invoke TWO different subagents in parallel
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Get weather information for Tokyo",
                                    "subagent_type": "weather-analyzer",
                                },
                                "id": "call_weather",
                                "type": "tool_call",
                            },
                            {
                                "name": "task",
                                "args": {
                                    "description": "Get population statistics for Tokyo",
                                    "subagent_type": "population-analyzer",
                                },
                                "id": "call_population",
                                "type": "tool_call",
                            },
                        ],
                    ),
                    # Second response: acknowledge both results
                    AIMessage(content="I've gathered weather and population data for Tokyo."),
                ]
            )
        )

        # Create weather subagent with structured output
        weather_subagent_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "CityWeather",
                                "args": {
                                    "city": "Tokyo",
                                    "temperature_celsius": 22.5,
                                    "humidity_percent": 65,
                                },
                                "id": "call_weather_struct",
                                "type": "tool_call",
                            }
                        ],
                    ),
                ]
            )
        )

        weather_subagent = create_agent(
            model=weather_subagent_model,
            response_format=ToolStrategy(schema=CityWeather),
        )

        # Create population subagent with structured output
        population_subagent_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "CityPopulation",
                                "args": {
                                    "city": "Tokyo",
                                    "population": 14000000,
                                    "metro_area_population": 37400000,
                                },
                                "id": "call_population_struct",
                                "type": "tool_call",
                            }
                        ],
                    ),
                ]
            )
        )

        population_subagent = create_agent(
            model=population_subagent_model,
            response_format=ToolStrategy(schema=CityPopulation),
        )

        # Create parent agent with both specialized subagents
        parent_agent = create_deep_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            subagents=[
                CompiledSubAgent(
                    name="weather-analyzer",
                    description="Specialized agent for weather analysis.",
                    runnable=weather_subagent,
                ),
                CompiledSubAgent(
                    name="population-analyzer",
                    description="Specialized agent for population analysis.",
                    runnable=population_subagent,
                ),
            ],
        )

        # Invoke the parent agent
        result = parent_agent.invoke(
            {"messages": [HumanMessage(content="Tell me about Tokyo's weather and population")]},
            config={"configurable": {"thread_id": "test_thread_structured"}},
        )

        # Verify the result contains messages
        assert "messages" in result, "Result should contain messages key"

        # Find all ToolMessages from the subagents
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 2, f"Should have exactly 2 ToolMessages, got {len(tool_messages)}"

        # Create lookup map by tool_call_id
        tool_messages_by_id = {msg.tool_call_id: msg for msg in tool_messages}

        # Verify both expected tool call IDs are present
        assert "call_weather" in tool_messages_by_id, "Should have response from weather subagent"
        assert "call_population" in tool_messages_by_id, "Should have response from population subagent"

        # Verify that structured_response is NOT in the parent agent's final state
        # (it should be excluded per _EXCLUDED_STATE_KEYS)
        assert "structured_response" not in result, (
            "Parent agent state should not contain structured_response key (it should be excluded per _EXCLUDED_STATE_KEYS)"
        )

        # Verify the exact content of the ToolMessages
        # When a subagent uses ToolStrategy for structured output, the default tool message
        # content shows the structured response using the Pydantic model's string representation
        weather_tool_message = tool_messages_by_id["call_weather"]
        expected_weather_content = "Returning structured response: city='Tokyo' temperature_celsius=22.5 humidity_percent=65"
        assert weather_tool_message.content == expected_weather_content, (
            f"Expected weather ToolMessage content:\n{expected_weather_content}\nGot:\n{weather_tool_message.content}"
        )

        population_tool_message = tool_messages_by_id["call_population"]
        expected_population_content = "Returning structured response: city='Tokyo' population=14000000 metro_area_population=37400000"
        assert population_tool_message.content == expected_population_content, (
            f"Expected population ToolMessage content:\n{expected_population_content}\nGot:\n{population_tool_message.content}"
        )

    def test_lc_agent_name_and_tags_in_streaming_metadata(self) -> None:
        """Test that lc_agent_name and tags are correctly set in streaming metadata.

        Verifies:
        1. Parent content chunks have lc_agent_name='supervisor'
        2. Subagent content chunks have lc_agent_name='worker'
        3. Tags from parent config appear in subagent streaming chunks
        """
        parent_content = "PARENT_RESPONSE"
        subagent_content = "SUBAGENT_RESPONSE"
        test_tags = ["test-tag", "session-123"]

        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {"description": "Do task", "subagent_type": "worker"},
                                "id": "call_worker",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content=parent_content),
                ]
            )
        )
        subagent_chat_model = GenericFakeChatModel(messages=iter([AIMessage(content=subagent_content)]))

        compiled_subagent = create_agent(model=subagent_chat_model, name="worker")
        parent_agent = create_deep_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            name="supervisor",
            subagents=[CompiledSubAgent(name="worker", description="Does work.", runnable=compiled_subagent)],
        )

        saw_parent_content = saw_subagent_content = False
        for _ns, (chunk, metadata) in parent_agent.stream(
            {"messages": [HumanMessage(content="Do something")]},
            stream_mode="messages",
            subgraphs=True,
            config={"configurable": {"thread_id": "test_thread"}, "tags": test_tags},
        ):
            agent_name = metadata.get("lc_agent_name")
            tags = metadata.get("tags", [])

            # Check parent content has correct agent name
            if parent_content in chunk.content and not saw_parent_content:
                assert agent_name == "supervisor", f"Parent content should have agent_name='supervisor', got '{agent_name}'"
                saw_parent_content = True

            # Check subagent content has correct agent name and tags
            if subagent_content in chunk.content and agent_name == "worker" and not saw_subagent_content:
                assert all(t in tags for t in test_tags), f"Subagent chunk missing tags. Expected {test_tags}, got {tags}"
                saw_subagent_content = True

        assert saw_parent_content, "Should have seen parent content with supervisor agent name"
        assert saw_subagent_content, "Should have seen subagent content with worker agent name and tags"

    def test_config_passed_to_runnable_lambda_subagent(self) -> None:
        """Test that config (including tags) is passed to a RunnableLambda subagent.

        RunnableLambda doesn't have a 'config' attribute, so this tests the safe getattr fallback.
        """
        received_configs: list[RunnableConfig] = []

        def lambda_subagent(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:  # noqa: ARG001
            received_configs.append(config)
            return {"messages": [AIMessage(content="Lambda response")]}

        runnable_lambda = RunnableLambda(lambda_subagent)
        assert not hasattr(runnable_lambda, "config")

        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {"description": "Do something", "subagent_type": "lambda-agent"},
                                "id": "call_lambda",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        parent_agent = create_deep_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            name="parent",
            subagents=[CompiledSubAgent(name="lambda-agent", description="Lambda subagent.", runnable=runnable_lambda)],
        )

        test_tags = ["lambda-tag", "config-test"]
        parent_agent.invoke(
            {"messages": [HumanMessage(content="Do something")]},
            config={"configurable": {"thread_id": "test_lambda"}, "tags": test_tags},
        )

        assert len(received_configs) > 0, "Lambda should have been invoked"
        assert all(t in received_configs[0].get("tags", []) for t in test_tags), f"Missing tags in config: {received_configs[0].get('tags')}"

    def test_context_passed_to_subagent_tool_runtime(self) -> None:
        """Test that context passed to main agent is available in subagent's ToolRuntime.context."""
        received_contexts: list[Any] = []

        @tool
        def capture_context(query: str, runtime: ToolRuntime) -> str:
            """Captures runtime context."""
            received_contexts.append(runtime.context)
            return f"Processed: {query}"

        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {"description": "Use capture_context", "subagent_type": "ctx-agent"},
                                "id": "call_ctx",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )
        subagent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "capture_context",
                                "args": {"query": "test"},
                                "id": "call_tool",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Captured."),
                ]
            )
        )

        compiled_subagent = create_agent(model=subagent_chat_model, tools=[capture_context], name="ctx-agent")
        parent_agent = create_deep_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            name="orchestrator",
            subagents=[CompiledSubAgent(name="ctx-agent", description="Context-aware subagent.", runnable=compiled_subagent)],
        )

        test_context = {"user_id": "user-123", "session_id": "session-456"}
        parent_agent.invoke(
            {"messages": [HumanMessage(content="Process")]},
            config={"configurable": {"thread_id": "test_context"}},
            context=test_context,
        )

        assert len(received_contexts) > 0, "Subagent tool should have been invoked"
        assert received_contexts[0] == test_context, f"Expected {test_context}, got {received_contexts[0]}"

    def test_compiled_subagent_without_messages_raises_error(self) -> None:
        """Test that a CompiledSubAgent without 'messages' in state raises a clear error.

        This test verifies that when a custom StateGraph is used with CompiledSubAgent
        and doesn't include a 'messages' key in its state, a helpful ValueError is raised
        explaining the requirement.
        """

        # Define a custom state without 'messages' key
        class CustomState(TypedDict):
            custom_field: str

        def custom_node(_state: CustomState) -> CustomState:
            return {"custom_field": "processed"}

        # Build a custom graph that doesn't use messages
        graph_builder = StateGraph(CustomState)
        graph_builder.add_node("process", custom_node)
        graph_builder.add_edge(START, "process")
        graph_builder.add_edge("process", END)
        custom_graph = graph_builder.compile()

        # Create parent agent with this custom subagent
        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Process something",
                                    "subagent_type": "custom-processor",
                                },
                                "id": "call_custom",
                            }
                        ],
                    ),
                ]
            )
        )

        parent_agent = create_deep_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            subagents=[
                CompiledSubAgent(
                    name="custom-processor",
                    description="A custom processor",
                    runnable=custom_graph,
                )
            ],
        )

        # Attempting to invoke should raise a clear error about missing 'messages' key
        with pytest.raises(
            ValueError,
            match="CompiledSubAgent must return a state containing a 'messages' key",
        ):
            parent_agent.invoke(
                {"messages": [HumanMessage(content="Process this")]},
                config={"configurable": {"thread_id": "test_thread_no_messages"}},
            )

    def test_custom_subagent_does_not_inherit_skills(self, tmp_path: Path) -> None:
        """Test that custom subagents do NOT inherit skills middleware from create_deep_agent.

        This test verifies that:
        1. When create_deep_agent is called with skills, only the general-purpose subagent gets SkillsMiddleware
        2. Custom subagents (defined via SubAgent spec) do NOT get SkillsMiddleware
        3. This prevents skills_metadata from being added to custom subagent state
        """
        # Set up filesystem backend with a skill
        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        skills_dir = tmp_path / "skills" / "user"
        skill_path = str(skills_dir / "test-skill" / "SKILL.md")
        skill_content = _make_skill_content("test-skill", "A test skill")

        responses = backend.upload_files([(skill_path, skill_content.encode("utf-8"))])
        assert responses[0].error is None

        # Track the runtime state seen by the custom subagent's tool
        captured_subagent_states: list[dict[str, Any]] = []

        @tool
        def capture_subagent_state(query: str, runtime: ToolRuntime) -> str:
            """Captures runtime state from the subagent."""
            captured_subagent_states.append(dict(runtime.state))
            return f"Processed: {query}"

        # Create custom subagent model that calls the capture tool
        custom_subagent_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "capture_subagent_state",
                                "args": {"query": "check state"},
                                "id": "call_capture",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Custom subagent response."),
                ]
            )
        )

        # Create leader that calls the custom subagent
        leader_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Do custom work",
                                    "subagent_type": "custom-worker",
                                },
                                "id": "call_custom",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        leader = create_deep_agent(
            model=leader_model,
            checkpointer=InMemorySaver(),
            backend=backend,
            skills=[str(skills_dir)],  # Leader has skills
            subagents=[
                SubAgent(
                    name="custom-worker",
                    description="A custom worker agent",
                    system_prompt="You are a custom worker.",
                    model=custom_subagent_model,
                    tools=[capture_subagent_state],
                )
            ],
        )

        leader.invoke(
            {"messages": [HumanMessage(content="Go")]},
            config={"configurable": {"thread_id": "test_custom_no_skills"}},
        )

        # Verify the custom subagent tool was called
        assert len(captured_subagent_states) > 0, "Custom subagent tool should have been invoked"

        # Verify the custom subagent's runtime.state does NOT contain skills_metadata
        for state in captured_subagent_states:
            assert "skills_metadata" not in state, (
                "Custom subagent should NOT have skills_metadata in runtime.state - skills middleware should only apply to general-purpose subagent"
            )

    def test_skills_metadata_not_bubbled_to_parent(self, tmp_path: Path) -> None:
        """Test that skills_metadata from subagent middleware doesn't bubble up to parent.

        This test verifies that:
        1. A subagent with SkillsMiddleware loads skills and populates skills_metadata in its state
        2. When the subagent completes, skills_metadata is NOT included in the parent's state
        3. The PrivateStateAttr annotation correctly filters the field from invoke() output

        This works because PrivateStateAttr (OmitFromSchema with output=True) tells LangGraph
        to exclude the field from the output schema, which filters it from invoke() results.
        """
        # Set up filesystem backend with a skill
        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        skills_dir = tmp_path / "skills" / "user"
        skill_path = str(skills_dir / "test-skill" / "SKILL.md")
        skill_content = _make_skill_content("test-skill", "A test skill for subagent")

        responses = backend.upload_files([(skill_path, skill_content.encode("utf-8"))])
        assert responses[0].error is None

        # Create parent agent's chat model
        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Process this request",
                                    "subagent_type": "skills-agent",
                                },
                                "id": "call_skills_agent",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Task completed."),
                ]
            )
        )

        # Create subagent with SkillsMiddleware
        subagent_chat_model = GenericFakeChatModel(messages=iter([AIMessage(content="Subagent processed request using skills.")]))

        skills_middleware = SkillsMiddleware(
            backend=backend,
            sources=[str(skills_dir)],
        )

        subagent = create_agent(
            model=subagent_chat_model,
            middleware=[skills_middleware],
        )

        # Create parent agent with the subagent
        parent_agent = create_deep_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            subagents=[
                CompiledSubAgent(
                    name="skills-agent",
                    description="Agent with skills middleware.",
                    runnable=subagent,
                )
            ],
        )

        # Invoke parent agent
        result = parent_agent.invoke(
            {"messages": [HumanMessage(content="Hello")]},
            config={"configurable": {"thread_id": "test_skills_isolation"}},
        )

        # Verify skills_metadata is NOT in the parent agent's final state
        assert "skills_metadata" not in result, (
            "Parent agent state should not contain skills_metadata key (PrivateStateAttr should filter it from subagent output)"
        )

        # Verify the subagent did return a response
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        assert "Subagent processed request" in tool_messages[0].content

    def test_general_purpose_subagent_inherits_skills_from_main_agent(self, tmp_path: Path) -> None:
        """Test that the general-purpose subagent DOES inherit skills from main agent.

        This test verifies that:
        1. When create_deep_agent is called with skills, the general-purpose subagent gets SkillsMiddleware
        2. The skills_metadata is present in the general-purpose subagent's runtime.state
        3. This is the intended behavior - only general-purpose subagents should have skills

        This complements test_custom_subagent_does_not_inherit_skills which verifies
        that custom subagents do NOT get skills.
        """
        # Set up filesystem backend with a skill
        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        skills_dir = tmp_path / "skills" / "user"
        skill_path = str(skills_dir / "gp-test-skill" / "SKILL.md")
        skill_content = _make_skill_content("gp-test-skill", "A skill for general purpose agent")

        responses = backend.upload_files([(skill_path, skill_content.encode("utf-8"))])
        assert responses[0].error is None

        # Track runtime states from both leader and general-purpose subagent
        captured_leader_states: list[dict[str, Any]] = []
        captured_gp_states: list[dict[str, Any]] = []

        @tool
        def capture_leader_state(query: str, runtime: ToolRuntime) -> str:
            """Captures runtime state from the leader agent."""
            captured_leader_states.append(dict(runtime.state))
            return f"Leader processed: {query}"

        @tool
        def capture_gp_state(query: str, runtime: ToolRuntime) -> str:
            """Captures runtime state from the general-purpose subagent."""
            captured_gp_states.append(dict(runtime.state))
            return f"GP processed: {query}"

        # The general-purpose subagent inherits tools from the leader and uses the same model.
        # We provide enough responses for both the leader (3 calls) and subagent (2 calls).
        shared_model = GenericFakeChatModel(
            messages=iter(
                [
                    # Leader first captures its own state
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "capture_leader_state",
                                "args": {"query": "check leader state"},
                                "id": "call_leader_capture",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    # Leader then invokes the task tool
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Call capture_gp_state tool to check state",
                                    "subagent_type": "general-purpose",
                                },
                                "id": "call_gp",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    # General-purpose subagent captures its state (inherits tools from leader)
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "capture_gp_state",
                                "args": {"query": "check gp state"},
                                "id": "call_gp_capture",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    # General-purpose subagent's final response
                    AIMessage(content="General purpose subagent response."),
                    # Leader's final response
                    AIMessage(content="Done."),
                ]
            )
        )

        leader = create_deep_agent(
            model=shared_model,
            checkpointer=InMemorySaver(),
            backend=backend,
            skills=[str(skills_dir)],  # Leader has skills
            tools=[capture_leader_state, capture_gp_state],
        )

        leader.invoke(
            {"messages": [HumanMessage(content="Go")]},
            config={"configurable": {"thread_id": "test_gp_with_skills"}},
        )

        # Verify the leader tool was called and has skills_metadata
        assert len(captured_leader_states) > 0, "Leader tool should have been invoked"
        assert "skills_metadata" in captured_leader_states[0], "Leader should have skills_metadata in runtime.state"

        # Verify the general-purpose subagent tool was called and has skills_metadata
        assert len(captured_gp_states) > 0, "General-purpose subagent tool should have been invoked"
        assert "skills_metadata" in captured_gp_states[0], (
            "General-purpose subagent SHOULD have skills_metadata in runtime.state - skills middleware should be applied to general-purpose subagent"
        )

        # Verify the skill name is in the skills_metadata
        # skills_metadata is a list[SkillMetadata] where each item is a TypedDict with a 'name' key
        gp_skills_metadata = captured_gp_states[0]["skills_metadata"]
        skill_names = [s["name"] for s in gp_skills_metadata]
        assert "gp-test-skill" in skill_names, f"General-purpose subagent should have 'gp-test-skill' in skills_metadata. Found skills: {skill_names}"

    def test_custom_subagent_with_skills_parameter(self, tmp_path: Path) -> None:
        """Test that a custom SubAgent with skills parameter loads skills correctly.

        This test verifies that:
        1. When a SubAgent spec includes a `skills` parameter, the subagent gets SkillsMiddleware
        2. The skills_metadata is present in the subagent's runtime.state
        3. The skills are correctly loaded from the specified source paths
        """
        # Set up filesystem backend with a skill
        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        skills_dir = tmp_path / "skills" / "custom"
        skill_path = str(skills_dir / "custom-skill" / "SKILL.md")
        skill_content = _make_skill_content("custom-skill", "A skill for custom subagent")

        responses = backend.upload_files([(skill_path, skill_content.encode("utf-8"))])
        assert responses[0].error is None

        # Track the runtime state seen by the custom subagent's tool
        captured_subagent_states: list[dict[str, Any]] = []

        @tool
        def capture_subagent_state(query: str, runtime: ToolRuntime) -> str:
            """Captures runtime state from the subagent."""
            captured_subagent_states.append(dict(runtime.state))
            return f"Processed: {query}"

        # Create custom subagent model that calls the capture tool
        custom_subagent_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "capture_subagent_state",
                                "args": {"query": "check state"},
                                "id": "call_capture",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Custom subagent response with skills."),
                ]
            )
        )

        # Create leader that calls the custom subagent
        leader_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Do custom work with skills",
                                    "subagent_type": "skilled-worker",
                                },
                                "id": "call_custom",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        leader = create_deep_agent(
            model=leader_model,
            checkpointer=InMemorySaver(),
            backend=backend,
            subagents=[
                SubAgent(
                    name="skilled-worker",
                    description="A custom worker agent with skills",
                    system_prompt="You are a custom worker with skills.",
                    model=custom_subagent_model,
                    tools=[capture_subagent_state],
                    skills=[str(skills_dir)],  # Custom subagent with skills
                )
            ],
        )

        leader.invoke(
            {"messages": [HumanMessage(content="Go")]},
            config={"configurable": {"thread_id": "test_custom_with_skills"}},
        )

        # Verify the custom subagent tool was called
        assert len(captured_subagent_states) > 0, "Custom subagent tool should have been invoked"

        # Verify the custom subagent's runtime.state DOES contain skills_metadata
        subagent_state = captured_subagent_states[0]
        assert "skills_metadata" in subagent_state, "Custom subagent with skills parameter SHOULD have skills_metadata in runtime.state"

        # Verify the skill name is in the skills_metadata
        skills_metadata = subagent_state["skills_metadata"]
        skill_names = [s["name"] for s in skills_metadata]
        assert "custom-skill" in skill_names, f"Custom subagent should have 'custom-skill' in skills_metadata. Found skills: {skill_names}"

    def test_custom_subagent_with_skills_multiple_sources(self, tmp_path: Path) -> None:
        """Test that a custom SubAgent with multiple skill sources loads skills with proper override.

        This test verifies that:
        1. Skills from multiple sources are merged
        2. Later sources override earlier ones (last-wins semantics)
        """
        # Set up filesystem backend with skills in two directories
        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

        base_dir = tmp_path / "skills" / "base"
        user_dir = tmp_path / "skills" / "user"

        # Create same-named skill in both directories (user should win)
        base_skill_path = str(base_dir / "shared-skill" / "SKILL.md")
        user_skill_path = str(user_dir / "shared-skill" / "SKILL.md")
        unique_skill_path = str(base_dir / "base-only-skill" / "SKILL.md")

        base_content = _make_skill_content("shared-skill", "Base version - should be overridden")
        user_content = _make_skill_content("shared-skill", "User version - should win")
        unique_content = _make_skill_content("base-only-skill", "Only in base")

        responses = backend.upload_files(
            [
                (base_skill_path, base_content.encode("utf-8")),
                (user_skill_path, user_content.encode("utf-8")),
                (unique_skill_path, unique_content.encode("utf-8")),
            ]
        )
        assert all(r.error is None for r in responses)

        # Track the runtime state
        captured_subagent_states: list[dict[str, Any]] = []

        @tool
        def capture_state(query: str, runtime: ToolRuntime) -> str:
            """Captures runtime state from the subagent."""
            captured_subagent_states.append(dict(runtime.state))
            return f"Processed: {query}"

        # Create custom subagent model
        custom_subagent_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "capture_state",
                                "args": {"query": "check"},
                                "id": "call_capture",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        # Create leader
        leader_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Do work",
                                    "subagent_type": "multi-skills-worker",
                                },
                                "id": "call_task",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        leader = create_deep_agent(
            model=leader_model,
            checkpointer=InMemorySaver(),
            backend=backend,
            subagents=[
                SubAgent(
                    name="multi-skills-worker",
                    description="Worker with multiple skill sources",
                    system_prompt="You are a worker.",
                    model=custom_subagent_model,
                    tools=[capture_state],
                    skills=[str(base_dir), str(user_dir)],  # Multiple sources, user wins
                )
            ],
        )

        leader.invoke(
            {"messages": [HumanMessage(content="Go")]},
            config={"configurable": {"thread_id": "test_multi_skills"}},
        )

        # Verify tool was called
        assert len(captured_subagent_states) > 0

        # Verify skills_metadata contains both skills with correct override
        skills_metadata = captured_subagent_states[0]["skills_metadata"]
        skills_by_name = {s["name"]: s for s in skills_metadata}

        # Should have both skills
        assert "shared-skill" in skills_by_name, "Should have shared-skill"
        assert "base-only-skill" in skills_by_name, "Should have base-only-skill"

        # shared-skill should have user version (last wins)
        assert skills_by_name["shared-skill"]["description"] == "User version - should win", "shared-skill should have user version description"

    def test_custom_subagent_without_skills_has_no_skills_metadata(self, tmp_path: Path) -> None:
        """Test that a custom SubAgent WITHOUT skills parameter has no skills_metadata.

        This confirms that the skills parameter is optional and only adds SkillsMiddleware
        when explicitly specified.
        """
        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

        # Track the runtime state
        captured_subagent_states: list[dict[str, Any]] = []

        @tool
        def capture_state(query: str, runtime: ToolRuntime) -> str:
            """Captures runtime state from the subagent."""
            captured_subagent_states.append(dict(runtime.state))
            return f"Processed: {query}"

        # Create custom subagent model
        custom_subagent_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "capture_state",
                                "args": {"query": "check"},
                                "id": "call_capture",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        # Create leader
        leader_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Do work",
                                    "subagent_type": "no-skills-worker",
                                },
                                "id": "call_task",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        leader = create_deep_agent(
            model=leader_model,
            checkpointer=InMemorySaver(),
            backend=backend,
            subagents=[
                SubAgent(
                    name="no-skills-worker",
                    description="Worker without skills",
                    system_prompt="You are a worker.",
                    model=custom_subagent_model,
                    tools=[capture_state],
                    # No skills parameter
                )
            ],
        )

        leader.invoke(
            {"messages": [HumanMessage(content="Go")]},
            config={"configurable": {"thread_id": "test_no_skills"}},
        )

        # Verify tool was called
        assert len(captured_subagent_states) > 0

        # Verify skills_metadata is NOT in the subagent state
        subagent_state = captured_subagent_states[0]
        assert "skills_metadata" not in subagent_state, "Subagent without skills parameter should NOT have skills_metadata"


class TestSubAgentMiddlewareValidation:
    """Tests for SubAgentMiddleware initialization validation."""

    def test_unknown_kwargs_raises_type_error(self) -> None:
        """Test that passing unknown kwargs to SubAgentMiddleware raises TypeError.

        This validates that deprecated_kwargs are properly validated and unknown
        kwargs like 'fooofoobar' are caught and reported.
        """
        with pytest.raises(TypeError, match=r"unexpected keyword argument.*fooofoobar"):
            SubAgentMiddleware(
                default_model="openai:gpt-4o",  # type: ignore[call-arg]
                fooofoobar=2,  # type: ignore[call-arg]
            )

    def test_multiple_unknown_kwargs_reported(self) -> None:
        """Test that multiple unknown kwargs are all reported in the error message."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            SubAgentMiddleware(
                default_model="openai:gpt-4o",  # type: ignore[call-arg]
                unknown_arg_1=1,  # type: ignore[call-arg]
                unknown_arg_2=2,  # type: ignore[call-arg]
            )

    def test_valid_deprecated_kwargs_accepted(self) -> None:
        """Test that valid deprecated kwargs don't raise TypeError."""
        fake_model = GenericFakeChatModel(messages=iter([]))

        # This should not raise TypeError, only emit a deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SubAgentMiddleware(
                default_model=fake_model,  # type: ignore[call-arg]
                default_tools=[],  # type: ignore[call-arg]
            )

        # Should have received deprecation warning but no TypeError
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()
