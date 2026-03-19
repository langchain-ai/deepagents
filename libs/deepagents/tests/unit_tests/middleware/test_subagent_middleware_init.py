"""Unit tests for SubAgentMiddleware initialization and configuration."""

import warnings

import pytest
from langchain.agents import create_agent
from langchain_core.tools import tool

from deepagents.backends.state import StateBackend
from deepagents.middleware.subagents import (
    GENERAL_PURPOSE_SUBAGENT,
    TASK_SYSTEM_PROMPT,
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
            backend=StateBackend,
            subagents=[
                {
                    **GENERAL_PURPOSE_SUBAGENT,
                    "model": "gpt-4o-mini",
                    "tools": [],
                }
            ],
        )
        assert middleware is not None
        # System prompt includes TASK_SYSTEM_PROMPT plus available subagent types
        assert middleware.system_prompt.startswith(TASK_SYSTEM_PROMPT)
        assert "Available subagent types:" in middleware.system_prompt
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "task"

    def test_subagent_middleware_with_custom_subagent(self) -> None:
        """Test SubAgentMiddleware initialization with a custom subagent."""
        middleware = SubAgentMiddleware(
            backend=StateBackend,
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
        assert middleware.system_prompt.startswith(TASK_SYSTEM_PROMPT)
        assert "weather" in middleware.system_prompt

    def test_subagent_middleware_custom_system_prompt(self) -> None:
        """Test SubAgentMiddleware with a custom system prompt."""
        middleware = SubAgentMiddleware(
            backend=StateBackend,
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

    # ========== Tests for new API ==========

    def test_new_api_requires_backend(self) -> None:
        """Test that the new API requires backend parameter."""
        with pytest.raises(ValueError, match="requires either"):
            SubAgentMiddleware(
                subagents=[
                    {
                        "name": "test",
                        "description": "Test",
                        "system_prompt": "Test.",
                        "model": "gpt-4o-mini",
                        "tools": [],
                    }
                ],
            )

    def test_new_api_requires_subagents(self) -> None:
        """Test that the new API requires at least one subagent."""
        with pytest.raises(ValueError, match="At least one subagent"):
            SubAgentMiddleware(
                backend=StateBackend,
                subagents=[],
            )

    def test_new_api_no_deprecation_warning(self) -> None:
        """Test that using only new API args does not emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            middleware = SubAgentMiddleware(
                backend=StateBackend,
                subagents=[
                    {
                        "name": "test",
                        "description": "Test subagent",
                        "system_prompt": "Test.",
                        "model": "gpt-4o-mini",
                        "tools": [],
                    }
                ],
            )
            # Filter for DeprecationWarnings only
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0, f"Unexpected deprecation warnings: {deprecation_warnings}"
        assert middleware is not None

    def test_new_api_subagent_requires_model(self) -> None:
        """Test that subagents must specify model when using new API."""
        with pytest.raises(ValueError, match="must specify 'model'"):
            SubAgentMiddleware(
                backend=StateBackend,
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

    def test_new_api_subagent_requires_tools(self) -> None:
        """Test that subagents must specify tools when using new API."""
        with pytest.raises(ValueError, match="must specify 'tools'"):
            SubAgentMiddleware(
                backend=StateBackend,
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

    # ========== Tests for deprecated API ==========

    def test_deprecated_api_still_works(self) -> None:
        """Test that the deprecated API still works for backward compatibility."""
        with pytest.warns(DeprecationWarning, match="default_model"):
            middleware = SubAgentMiddleware(
                default_model="gpt-4o-mini",
                default_tools=[get_weather],
                subagents=[
                    {
                        "name": "custom",
                        "description": "Custom subagent",
                        "system_prompt": "You are custom.",
                        "tools": [get_weather],
                    }
                ],
            )
        assert middleware is not None
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "task"
        assert "general-purpose" in middleware.system_prompt
        assert "custom" in middleware.system_prompt

    def test_deprecated_api_general_purpose_agent_disabled(self) -> None:
        """Test deprecated API with general_purpose_agent=False."""
        with pytest.warns(DeprecationWarning, match="default_model"):
            middleware = SubAgentMiddleware(
                default_model="gpt-4o-mini",
                general_purpose_agent=False,
                subagents=[
                    {
                        "name": "only_agent",
                        "description": "The only agent",
                        "system_prompt": "You are the only one.",
                        "tools": [],
                    }
                ],
            )
        assert middleware is not None
        assert "only_agent" in middleware.system_prompt
        assert "general-purpose" not in middleware.system_prompt

    # ========== Tests for mixing old and new args ==========

    def test_mixed_args_prefers_new_api(self) -> None:
        """Test that when both backend and deprecated args are provided, new API is used with warning."""
        with pytest.warns(DeprecationWarning, match="default_model"):
            middleware = SubAgentMiddleware(
                backend=StateBackend,
                subagents=[
                    {
                        "name": "test",
                        "description": "Test subagent",
                        "system_prompt": "Test.",
                        "model": "gpt-4o-mini",
                        "tools": [],
                    }
                ],
                default_model="gpt-4o-mini",  # This is deprecated but still triggers warning
            )
        assert middleware is not None

    def test_multiple_subagents_with_interrupt_on(self) -> None:
        """Test creating agent with multiple subagents that have interrupt_on configured."""
        agent = create_agent(
            model="claude-sonnet-4-20250514",
            system_prompt="Use the task tool to call subagents.",
            middleware=[
                SubAgentMiddleware(
                    backend=StateBackend,
                    subagents=[
                        {
                            "name": "subagent1",
                            "description": "First subagent.",
                            "system_prompt": "You are subagent 1.",
                            "model": "claude-sonnet-4-20250514",
                            "tools": [get_weather],
                            "interrupt_on": {"get_weather": True},
                        },
                        {
                            "name": "subagent2",
                            "description": "Second subagent.",
                            "system_prompt": "You are subagent 2.",
                            "model": "claude-sonnet-4-20250514",
                            "tools": [get_weather],
                            "interrupt_on": {"get_weather": True},
                        },
                    ],
                )
            ],
        )
        # This would error if the middleware was accumulated incorrectly
        assert agent is not None
