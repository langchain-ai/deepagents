import pytest
from unittest.mock import patch, MagicMock
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph

from deepagents import create_deep_agent, DeepAgentState, SubAgent


class TestCreateDeepAgent:
    """Test suite for create_deep_agent function."""

    @patch("deepagents.graph.get_default_model")
    def test_create_deep_agent_happy_path(self, mock_get_default_model):
        """Test creating a deep agent with basic configuration."""
        mock_model = MagicMock()
        mock_get_default_model.return_value = mock_model

        @tool
        def sample_tool(query: str) -> str:
            """A sample tool for testing."""
            return f"Result for: {query}"

        instructions = "You are a helpful AI assistant for testing purposes."

        agent = create_deep_agent(tools=[sample_tool], instructions=instructions)

        assert agent is not None
        assert isinstance(agent, CompiledStateGraph)
        mock_get_default_model.assert_called_once()

    @patch("deepagents.graph.get_default_model")
    def test_create_deep_agent_with_subagents(self, mock_get_default_model):
        """Test creating a deep agent with subagents."""
        mock_model = MagicMock()
        mock_get_default_model.return_value = mock_model

        research_agent = SubAgent(
            name="research",
            description="Research information on the web",
            prompt="You are a research assistant. Focus on finding accurate information.",
            tools=[],
        )

        agent = create_deep_agent(
            tools=[],
            instructions="Main agent with subagents",
            subagents=[research_agent],
        )

        assert agent is not None
        assert isinstance(agent, CompiledStateGraph)
        mock_get_default_model.assert_called_once()

    @patch("deepagents.graph.get_default_model")
    def test_create_deep_agent_with_custom_model(self, mock_get_default_model):
        """Test creating a deep agent with a custom model."""
        custom_model = MagicMock()

        agent = create_deep_agent(
            tools=[], instructions="Agent with custom model", model=custom_model
        )

        assert agent is not None
        assert isinstance(agent, CompiledStateGraph)
        mock_get_default_model.assert_not_called()

    @patch("deepagents.graph.get_default_model")
    def test_create_deep_agent_with_custom_state_schema(self, mock_get_default_model):
        """Test creating a deep agent with custom state schema."""
        mock_model = MagicMock()
        mock_get_default_model.return_value = mock_model

        class CustomAgentState(DeepAgentState):
            custom_field: str = ""

        agent = create_deep_agent(
            tools=[],
            instructions="Agent with custom state",
            state_schema=CustomAgentState,
        )

        assert agent is not None
        assert isinstance(agent, CompiledStateGraph)
        mock_get_default_model.assert_called_once()

    @patch("deepagents.graph.get_default_model")
    def test_create_deep_agent_with_multiple_tools(self, mock_get_default_model):
        """Test creating a deep agent with multiple custom tools."""
        mock_model = MagicMock()
        mock_get_default_model.return_value = mock_model

        @tool
        def calculator(expression: str) -> str:
            """Calculate mathematical expressions."""
            return f"Calculated: {expression}"

        @tool
        def translator(text: str, target_language: str) -> str:
            """Translate text to target language."""
            return f"Translated '{text}' to {target_language}"

        agent = create_deep_agent(
            tools=[calculator, translator], instructions="Multi-tool agent for testing"
        )

        assert agent is not None
        assert isinstance(agent, CompiledStateGraph)
        mock_get_default_model.assert_called_once()

    @patch("deepagents.graph.get_default_model")
    def test_create_deep_agent_empty_tools(self, mock_get_default_model):
        """Test creating a deep agent with no additional tools."""
        mock_model = MagicMock()
        mock_get_default_model.return_value = mock_model

        agent = create_deep_agent(
            tools=[], instructions="Basic agent with only built-in tools"
        )

        assert agent is not None
        assert isinstance(agent, CompiledStateGraph)
        mock_get_default_model.assert_called_once()

    @patch("deepagents.graph.get_default_model")
    def test_create_deep_agent_complex_scenario(self, mock_get_default_model):
        """Test creating a deep agent with all features combined."""
        mock_model = MagicMock()
        mock_get_default_model.return_value = mock_model

        @tool
        def web_search(query: str) -> str:
            """Search the web for information."""
            return f"Search results for: {query}"

        coding_agent = SubAgent(
            name="coder",
            description="Write and review code",
            prompt="You are an expert programmer.",
            tools=[],
        )

        writer_agent = SubAgent(
            name="writer",
            description="Write and edit text content",
            prompt="You are a professional writer.",
            tools=[],
        )

        class ProjectState(DeepAgentState):
            project_name: str = "test_project"

        agent = create_deep_agent(
            tools=[web_search],
            instructions="You are a project manager coordinating various tasks.",
            subagents=[coding_agent, writer_agent],
            state_schema=ProjectState,
        )

        assert agent is not None
        assert isinstance(agent, CompiledStateGraph)
        mock_get_default_model.assert_called_once()
