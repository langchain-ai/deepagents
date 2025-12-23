"""Tests for sub-agent middleware functionality.

This module contains tests for the subagent system, focusing on how subagents
are invoked, how they return results, and how state is managed between parent
and child agents.
"""

from langchain.agents import create_agent
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
