"""Unit tests for SubAgentMiddleware initialization and configuration."""

import pytest
from langchain.agents import create_agent
from langchain_core.tools import tool

from deepagents.backends.state import StateBackend
from deepagents.middleware.subagents import (
    _TASK_ONLY_SYSTEM_PROMPT,
    GENERAL_PURPOSE_SUBAGENT,
    SubAgentMiddleware,
)


@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."


class TestSubagentMiddlewareInit:
    """Tests for SubAgentMiddleware initialization that don't require LLM invocation."""

    @pytest.fixture(autouse=True)
    def set_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set dummy API key for model initialization."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def test_subagent_middleware_init(self) -> None:
        """Test basic SubAgentMiddleware initialization with general-purpose subagent."""
        middleware = SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[
                {
                    **GENERAL_PURPOSE_SUBAGENT,
                    "model": "gpt-4o-mini",
                    "tools": [],
                }
            ],
        )
        assert middleware is not None
        assert "Available subagent types:" in middleware.system_prompt
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "task"

    def test_subagent_middleware_with_custom_subagent(self) -> None:
        """Test SubAgentMiddleware initialization with a custom subagent."""
        middleware = SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[
                {
                    "name": "weather",
                    "description": "Weather subagent",
                    "system_prompt": "Get weather.",
                    "model": "gpt-4o-mini",
                    "tools": [get_weather],
                }
            ],
        )
        assert middleware is not None
        # System prompt includes TASK_SYSTEM_PROMPT plus available subagent types
        assert middleware.system_prompt.startswith(_TASK_ONLY_SYSTEM_PROMPT)
        assert "weather" in middleware.system_prompt

    def test_subagent_middleware_custom_system_prompt(self) -> None:
        """Test SubAgentMiddleware with a custom system prompt."""
        middleware = SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[
                {
                    "name": "weather",
                    "description": "Weather subagent",
                    "system_prompt": "Get weather.",
                    "model": "gpt-4o-mini",
                    "tools": [],
                }
            ],
            system_prompt="Use the task tool to call a subagent.",
        )
        assert middleware is not None
        # Custom system prompt plus available subagent types
        assert middleware.system_prompt.startswith("Use the task tool to call a subagent.")

    def test_requires_subagents(self) -> None:
        """Test that at least one subagent is required."""
        with pytest.raises(ValueError, match="At least one subagent"):
            SubAgentMiddleware(
                backend=StateBackend(),
                subagents=[],
            )

    def test_subagent_requires_model(self) -> None:
        """Test that subagents must specify model."""
        with pytest.raises(ValueError, match="must specify 'model'"):
            SubAgentMiddleware(
                backend=StateBackend(),
                subagents=[
                    {
                        "name": "test",
                        "description": "Test",
                        "system_prompt": "Test.",
                        "tools": [],
                        # Missing "model"
                    }
                ],
            )

    def test_subagent_requires_tools(self) -> None:
        """Test that subagents must specify tools."""
        with pytest.raises(ValueError, match="must specify 'tools'"):
            SubAgentMiddleware(
                backend=StateBackend(),
                subagents=[
                    {
                        "name": "test",
                        "description": "Test",
                        "system_prompt": "Test.",
                        "model": "gpt-4o-mini",
                        # Missing "tools"
                    }
                ],
            )

    def test_multiple_subagents_with_interrupt_on(self) -> None:
        """Test creating agent with multiple subagents that have interrupt_on configured."""
        agent = create_agent(
            model="claude-sonnet-4-6",
            system_prompt="Use the task tool to call subagents.",
            middleware=[
                SubAgentMiddleware(
                    backend=StateBackend(),
                    subagents=[
                        {
                            "name": "subagent1",
                            "description": "First subagent.",
                            "system_prompt": "You are subagent 1.",
                            "model": "claude-sonnet-4-6",
                            "tools": [get_weather],
                            "interrupt_on": {"get_weather": True},
                        },
                        {
                            "name": "subagent2",
                            "description": "Second subagent.",
                            "system_prompt": "You are subagent 2.",
                            "model": "claude-sonnet-4-6",
                            "tools": [get_weather],
                            "interrupt_on": {"get_weather": True},
                        },
                    ],
                )
            ],
        )
        # This would error if the middleware was accumulated incorrectly
        assert agent is not None
