"""Tests for sub-agent middleware functionality.

This module contains tests for the subagent system, focusing on how subagents
are invoked, how they return results, and how state is managed between parent
and child agents.
"""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

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
        """Test that multiple subagents can be launched in parallel.

        This test verifies parallel subagent execution:
        1. Parent agent makes a single AIMessage with multiple tool_calls
        2. Both subagents are invoked concurrently (one for addition, one for multiplication)
        3. Each subagent completes its task independently
        4. Both subagent results are returned as separate ToolMessages
        5. Parent agent receives both results and can synthesize them

        The parallel execution pattern is important for:
        - Reducing latency when tasks are independent
        - Efficient resource utilization
        - Handling multi-part user requests
        """
        # Create the parent agent's chat model that will call both subagents in parallel
        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    # First response: invoke TWO task tools in parallel
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Calculate the sum of 5 and 7",
                                    "subagent_type": "general-purpose",
                                },
                                "id": "call_addition",
                                "type": "tool_call",
                            },
                            {
                                "name": "task",
                                "args": {
                                    "description": "Calculate the product of 4 and 6",
                                    "subagent_type": "general-purpose",
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

        # Create subagent models - each will handle one calculation
        # Note: In a real scenario, the same subagent would be reused,
        # but for testing we simulate responses with separate model instances
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

        # Create two separate compiled subagents for this test
        # In practice, the same subagent definition handles multiple invocations
        addition_subagent = create_agent(model=addition_subagent_model)
        multiplication_subagent = create_agent(model=multiplication_subagent_model)

        # For this test, we'll use a single general-purpose subagent
        # The key is that it gets invoked twice in parallel
        # We'll use the addition model for the shared subagent
        # (In reality, the subagent would handle both, but for testing we simplify)
        general_purpose_subagent = create_agent(
            model=GenericFakeChatModel(
                messages=iter(
                    [
                        AIMessage(content="The sum of 5 and 7 is 12."),
                        AIMessage(content="The product of 4 and 6 is 24."),
                    ]
                )
            )
        )

        # Create the parent agent with subagent support
        parent_agent = create_deep_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            subagents=[
                CompiledSubAgent(
                    name="general-purpose",
                    description="A general-purpose agent for various tasks.",
                    runnable=general_purpose_subagent,
                )
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

        # Verify each ToolMessage has a unique tool_call_id corresponding to the parallel calls
        tool_call_ids = {msg.tool_call_id for msg in tool_messages}
        assert len(tool_call_ids) == 2, "Each ToolMessage should have a unique tool_call_id"
        assert "call_addition" in tool_call_ids, "Should have response from addition subagent"
        assert "call_multiplication" in tool_call_ids, "Should have response from multiplication subagent"

        # Verify the content of both responses
        tool_message_contents = [msg.content for msg in tool_messages]
        addition_result_found = any("12" in content for content in tool_message_contents)
        multiplication_result_found = any("24" in content for content in tool_message_contents)

        assert addition_result_found, "Should have addition result (12) in one of the ToolMessages"
        assert multiplication_result_found, "Should have multiplication result (24) in one of the ToolMessages"
