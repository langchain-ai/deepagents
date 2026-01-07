"""Tests for TodoListMiddleware functionality.

This module contains tests for the todo list middleware, focusing on how it handles
write_todos tool calls, state management, and edge cases.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage, HumanMessage

from tests.unit_tests.chat_model import GenericFakeChatModel


class TestTodoMiddleware:
    """Tests for TodoListMiddleware behavior."""

    def test_todo_middleware_rejects_multiple_write_todos_in_same_message(self) -> None:
        """Test that todo middleware rejects multiple write_todos calls in one AIMessage.

        This test verifies that:
        1. When an agent calls write_todos multiple times in the same AIMessage
        2. The middleware detects this and returns error messages for both calls
        3. The errors inform that write_todos should not be called in parallel
        4. The agent receives the error messages and can recover

        This validates that the todo middleware properly enforces the constraint that
        write_todos should not be called multiple times in parallel, as stated in the
        system prompt.
        """
        # Create a fake model that calls write_todos twice in the same AIMessage
        fake_model = GenericFakeChatModel(
            messages=iter(
                [
                    # First response: call write_todos TWICE in the same message
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_todos",
                                "args": {
                                    "todos": [
                                        {
                                            "content": "First task",
                                            "status": "in_progress",
                                            "activeForm": "Working on first task",
                                        },
                                    ]
                                },
                                "id": "call_write_todos_1",
                                "type": "tool_call",
                            },
                            {
                                "name": "write_todos",
                                "args": {
                                    "todos": [
                                        {
                                            "content": "First task",
                                            "status": "completed",
                                            "activeForm": "Working on first task",
                                        },
                                        {
                                            "content": "Second task",
                                            "status": "pending",
                                            "activeForm": "Working on second task",
                                        },
                                    ]
                                },
                                "id": "call_write_todos_2",
                                "type": "tool_call",
                            },
                        ],
                    ),
                    # Second response: final message
                    AIMessage(content="Both tasks have been planned successfully."),
                ]
            )
        )

        # Create an agent with TodoListMiddleware
        agent = create_agent(
            model=fake_model,
            middleware=[TodoListMiddleware()],
        )

        # Invoke the agent
        result = agent.invoke({"messages": [HumanMessage(content="Plan the work")]})

        # Verify the result contains messages
        assert "messages" in result, "Result should contain messages key"

        # Verify the agent has the todos stream channel
        assert "todos" in agent.stream_channels, "Agent should have 'todos' stream channel"

        # Verify there are exactly 2 ToolMessages for both write_todos calls
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 2, f"Should have exactly 2 ToolMessages (one per write_todos call), got {len(tool_messages)}"

        # Verify both ToolMessages contain error messages about parallel calls
        for tool_msg in tool_messages:
            assert "Error" in tool_msg.content, f"Expected error message, got: {tool_msg.content}"
            assert "write_todos" in tool_msg.content, f"Error should mention write_todos, got: {tool_msg.content}"
            assert "parallel" in tool_msg.content, f"Error should mention parallel calls, got: {tool_msg.content}"

        # Verify the chat model received the error messages in the second call
        assert len(fake_model.call_history) == 2, f"Expected 2 calls to chat model, got {len(fake_model.call_history)}"

        second_call_messages = fake_model.call_history[1]["messages"]
        error_tool_messages = [msg for msg in second_call_messages if msg.type == "tool"]
        assert len(error_tool_messages) == 2, "Second call should have 2 tool error messages"

        # Verify todos state is empty (no todos were actually written due to errors)
        # The middleware should prevent the parallel writes from succeeding
        if "todos" in result:
            todos = result["todos"]
            # Should be empty since both writes were rejected
            assert len(todos) == 0, f"Expected 0 todos due to rejected parallel writes, got {len(todos)}"

        # Verify the final AI message is present
        ai_messages = [msg for msg in result["messages"] if msg.type == "ai"]
        final_ai_message = ai_messages[-1]
        assert final_ai_message.content == "Both tasks have been planned successfully.", (
            f"Expected final message content, got: {final_ai_message.content}"
        )
