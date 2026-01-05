"""Tests for sub-agent middleware functionality.

This module contains tests for the subagent system, focusing on how subagents
are invoked, how they return results, and how state is managed between parent
and child agents.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from deepagents.graph import create_deep_agent
from deepagents.middleware.subagents import CompiledSubAgent
from tests.unit_tests.chat_model import GenericFakeChatModel


class TestSubAgentInvocation:
    """Tests for basic subagent invocation and response handling."""

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


class TestStructuredOutput:
    """Tests for agents with structured output using ToolStrategy."""

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


class TestSubAgentTodoList:
    """Tests for subagents that manage their own todo lists."""

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


class TestSubAgentsWithStructuredOutput:
    """Tests for subagents that return structured responses."""

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
