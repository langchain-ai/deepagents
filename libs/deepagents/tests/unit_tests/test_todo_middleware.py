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

    def test_todo_middleware_handles_multiple_write_todos_in_same_message(self) -> None:
        """Test that todo middleware properly handles multiple write_todos calls in one AIMessage.

        This test verifies that:
        1. An agent can call write_todos multiple times in the same AIMessage
        2. The middleware processes both tool calls correctly
        3. The final todo state reflects both updates (last one wins)
        4. The agent continues execution normally after handling multiple todo writes

        This validates that the todo middleware can handle edge cases where an agent
        attempts to write todos multiple times in a single response.
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
