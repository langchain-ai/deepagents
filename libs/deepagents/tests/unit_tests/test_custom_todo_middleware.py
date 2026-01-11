"""Tests for custom TodoListMiddleware with system_prompt.

This module tests the fix for the issue where passing a custom TodoListMiddleware
with a system_prompt parameter would cause an AssertionError about duplicate
middleware instances.
"""

from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import HumanMessage

from deepagents import create_deep_agent
from tests.unit_tests.chat_model import GenericFakeChatModel
from langchain_core.messages import AIMessage


class TestCustomTodoListMiddleware:
    """Tests for custom TodoListMiddleware with system_prompt."""

    def test_create_deep_agent_with_custom_todo_middleware(self) -> None:
        """Test that create_deep_agent accepts custom TodoListMiddleware with system_prompt.

        This test verifies that:
        1. Users can pass their own TodoListMiddleware instance with custom system_prompt
        2. No AssertionError about duplicate middleware is raised
        3. The agent is created successfully
        """
        custom_todo_prompt = """You are an advanced task planning assistant.
        Use the write_todos tool to maintain a structured task list.
        Always break down complex tasks into smaller, manageable subtasks."""

        # Create a fake model for testing
        fake_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="Ready to plan tasks."),
                ]
            )
        )

        # This should NOT raise AssertionError about duplicate middleware
        agent = create_deep_agent(
            model=fake_model,
            tools=[],
            system_prompt="You are a helpful assistant.",
            middleware=[TodoListMiddleware(system_prompt=custom_todo_prompt)],
            name="test-custom-todo",
        )

        # Verify agent was created successfully
        assert agent is not None

        # Invoke the agent to ensure it works
        result = agent.invoke({"messages": [HumanMessage(content="Plan my tasks")]})
        assert "messages" in result
        assert len(result["messages"]) > 0

    def test_create_deep_agent_without_custom_todo_middleware(self) -> None:
        """Test that create_deep_agent still works with default TodoListMiddleware.

        This ensures the fix doesn't break the default behavior.
        """
        # Create a fake model for testing
        fake_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="Ready to assist."),
                ]
            )
        )

        # Create agent with default TodoListMiddleware
        agent = create_deep_agent(
            model=fake_model,
            tools=[],
            system_prompt="You are a helpful assistant.",
            name="test-default-todo",
        )

        # Verify agent was created successfully
        assert agent is not None

        # Invoke the agent to ensure it works
        result = agent.invoke({"messages": [HumanMessage(content="Help me")]})
        assert "messages" in result
        assert len(result["messages"]) > 0
