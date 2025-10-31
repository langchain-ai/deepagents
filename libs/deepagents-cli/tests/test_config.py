"""Tests for configuration and utilities."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from deepagents_cli.config import (
    COLORS,
    COMMANDS,
    COMMON_BASH_COMMANDS,
    MAX_ARG_LENGTH,
    SessionState,
    create_model,
    get_default_coding_instructions,
)


def test_session_state_initialization():
    """Test SessionState initializes with correct defaults."""
    state = SessionState()
    assert state.auto_approve is False

    state_with_approve = SessionState(auto_approve=True)
    assert state_with_approve.auto_approve is True


def test_session_state_toggle_auto_approve():
    """Test SessionState toggle functionality."""
    state = SessionState(auto_approve=False)

    # Toggle from False to True
    result = state.toggle_auto_approve()
    assert result is True
    assert state.auto_approve is True

    # Toggle from True to False
    result = state.toggle_auto_approve()
    assert result is False
    assert state.auto_approve is False


def test_get_default_coding_instructions_returns_content():
    """Test that get_default_coding_instructions reads and returns content."""
    instructions = get_default_coding_instructions()

    # Should return non-empty string
    assert isinstance(instructions, str)
    assert len(instructions) > 0


def test_get_default_coding_instructions_file_exists():
    """Test that the default agent prompt file exists."""
    from deepagents_cli import config

    prompt_path = Path(config.__file__).parent / "default_agent_prompt.md"
    assert prompt_path.exists()
    assert prompt_path.is_file()


def test_constants_are_defined():
    """Test that important constants are defined correctly."""
    # Color constants
    assert isinstance(COLORS, dict)
    assert "primary" in COLORS
    assert "agent" in COLORS
    assert "user" in COLORS

    # Command constants
    assert isinstance(COMMANDS, dict)
    assert "clear" in COMMANDS
    assert "help" in COMMANDS
    assert "quit" in COMMANDS

    # Bash commands
    assert isinstance(COMMON_BASH_COMMANDS, dict)
    assert "ls" in COMMON_BASH_COMMANDS
    assert "cd" in COMMON_BASH_COMMANDS

    # Max arg length
    assert isinstance(MAX_ARG_LENGTH, int)
    assert MAX_ARG_LENGTH > 0


def test_create_model_no_api_keys():
    """Test that create_model exits when no API keys are configured."""
    with patch.dict(os.environ, {}, clear=True):
        # Should exit with code 1
        with pytest.raises(SystemExit) as exc_info:
            create_model()
        assert exc_info.value.code == 1


def test_create_model_with_openai_key():
    """Test that create_model returns OpenAI model when key is set."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        model = create_model()
        # Check that we got a ChatOpenAI instance
        assert model.__class__.__name__ == "ChatOpenAI"


def test_create_model_with_anthropic_key():
    """Test that create_model returns Anthropic model when key is set."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
        model = create_model()
        # Check that we got a ChatAnthropic instance
        assert model.__class__.__name__ == "ChatAnthropic"


def test_create_model_prefers_openai():
    """Test that create_model prefers OpenAI when both keys are set."""
    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "openai-key", "ANTHROPIC_API_KEY": "anthropic-key"},
        clear=True,
    ):
        model = create_model()
        # Should prefer OpenAI
        assert model.__class__.__name__ == "ChatOpenAI"


def test_create_model_custom_model_name_openai():
    """Test that create_model respects custom OpenAI model name."""
    with patch.dict(
        os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o"}, clear=True
    ):
        model = create_model()
        assert model.model_name == "gpt-4o"


def test_create_model_custom_model_name_anthropic():
    """Test that create_model respects custom Anthropic model name."""
    with patch.dict(
        os.environ,
        {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_MODEL": "claude-3-opus-20240229"},
        clear=True,
    ):
        model = create_model()
        # ChatAnthropic uses 'model' attribute, not 'model_name'
        assert model.model == "claude-3-opus-20240229"
