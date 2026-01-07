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

        # The middleware should return error messages for both parallel write_todos calls
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 2, f"Expected 2 error messages, got {len(tool_messages)}"

        # Verify exact error content and status for both tool messages
        expected_error = "Error: The `write_todos` tool should never be called multiple times in parallel. Please call it only once per model invocation to update the todo list."
        for tool_msg in tool_messages:
            assert tool_msg.content == expected_error, f"Expected exact error message, got: {tool_msg.content}"
            assert tool_msg.status == "error", f"Tool message status should be 'error', got: {tool_msg.status}"

        # No todos should be written since both calls were rejected
        assert result.get("todos", []) == [], "Todos should be empty when parallel writes are rejected"
