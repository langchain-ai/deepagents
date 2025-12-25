"""Unit tests for create_deep_agent function."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deepagents.graph import BASE_AGENT_PROMPT, create_deep_agent

from .chat_model import GenericFakeChatModel


class TestCreateDeepAgentSystemPrompt:
    """Test suite for create_deep_agent system_prompt parameter."""

    def test_create_deep_agent_with_string_system_prompt(self) -> None:
        """Test create_deep_agent with string system_prompt.

        This test verifies that the system_prompt parameter accepts a string
        and properly combines it with BASE_AGENT_PROMPT.
        """
        custom_prompt = "You are a helpful assistant."

        # Create a fake model
        model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="Hello!"),
                ]
            )
        )

        # Create deep agent with string system_prompt
        agent = create_deep_agent(model=model, system_prompt=custom_prompt)

        # Verify the agent was created successfully
        assert agent is not None

        # Invoke the agent to ensure it works
        result = agent.invoke({"messages": [HumanMessage(content="Hi")]})
        assert "messages" in result

    def test_create_deep_agent_with_system_message_prompt(self) -> None:
        """Test create_deep_agent with SystemMessage system_prompt.

        This test verifies that the system_prompt parameter accepts a SystemMessage
        object and properly extracts its content to combine with BASE_AGENT_PROMPT.
        """
        custom_prompt = SystemMessage(content="You are a helpful assistant.")

        # Create a fake model
        model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="Hello!"),
                ]
            )
        )

        # Create deep agent with SystemMessage system_prompt
        agent = create_deep_agent(model=model, system_prompt=custom_prompt)

        # Verify the agent was created successfully
        assert agent is not None

        # Invoke the agent to ensure it works
        result = agent.invoke({"messages": [HumanMessage(content="Hi")]})
        assert "messages" in result

    def test_create_deep_agent_with_none_system_prompt(self) -> None:
        """Test create_deep_agent with None system_prompt.

        This test verifies that when no system_prompt is provided,
        only BASE_AGENT_PROMPT is used.
        """
        # Create a fake model
        model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="Hello!"),
                ]
            )
        )

        # Create deep agent without system_prompt
        agent = create_deep_agent(model=model, system_prompt=None)

        # Verify the agent was created successfully
        assert agent is not None

        # Invoke the agent to ensure it works
        result = agent.invoke({"messages": [HumanMessage(content="Hi")]})
        assert "messages" in result

    def test_create_deep_agent_default_system_prompt(self) -> None:
        """Test create_deep_agent with default system_prompt (not provided).

        This test verifies that when system_prompt parameter is not provided at all,
        the agent still works correctly with just BASE_AGENT_PROMPT.
        """
        # Create a fake model
        model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="Hello!"),
                ]
            )
        )

        # Create deep agent without specifying system_prompt parameter
        agent = create_deep_agent(model=model)

        # Verify the agent was created successfully
        assert agent is not None

        # Invoke the agent to ensure it works
        result = agent.invoke({"messages": [HumanMessage(content="Hi")]})
        assert "messages" in result

    def test_create_deep_agent_system_message_with_complex_content(self) -> None:
        """Test create_deep_agent with SystemMessage containing complex content.

        This test verifies that SystemMessage objects with longer, more complex
        content are handled correctly.
        """
        complex_prompt = SystemMessage(
            content="You are a specialized AI assistant with the following capabilities:\n"
            "1. Code analysis and review\n"
            "2. Bug detection and fixing\n"
            "3. Performance optimization suggestions\n"
            "Please be thorough and precise in your responses."
        )

        # Create a fake model
        model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="I understand my capabilities."),
                ]
            )
        )

        # Create deep agent with complex SystemMessage
        agent = create_deep_agent(model=model, system_prompt=complex_prompt)

        # Verify the agent was created successfully
        assert agent is not None

        # Invoke the agent to ensure it works
        result = agent.invoke({"messages": [HumanMessage(content="Review this code")]})
        assert "messages" in result

    def test_create_deep_agent_empty_string_system_prompt(self) -> None:
        """Test create_deep_agent with empty string system_prompt.

        This test verifies that an empty string system_prompt is handled correctly
        and doesn't cause issues.
        """
        # Create a fake model
        model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="Hello!"),
                ]
            )
        )

        # Create deep agent with empty string system_prompt
        agent = create_deep_agent(model=model, system_prompt="")

        # Verify the agent was created successfully
        assert agent is not None

        # Invoke the agent to ensure it works
        result = agent.invoke({"messages": [HumanMessage(content="Hi")]})
        assert "messages" in result

    def test_create_deep_agent_system_message_empty_content(self) -> None:
        """Test create_deep_agent with SystemMessage containing empty content.

        This test verifies that a SystemMessage with empty content is handled correctly.
        """
        empty_system_message = SystemMessage(content="")

        # Create a fake model
        model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="Hello!"),
                ]
            )
        )

        # Create deep agent with empty SystemMessage
        agent = create_deep_agent(model=model, system_prompt=empty_system_message)

        # Verify the agent was created successfully
        assert agent is not None

        # Invoke the agent to ensure it works
        result = agent.invoke({"messages": [HumanMessage(content="Hi")]})
        assert "messages" in result
