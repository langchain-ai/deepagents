"""Tests for config.py model creation."""

import os
from unittest.mock import patch

import pytest


def test_create_model_with_ollama():
    """Test that create_model returns ChatOllama when OLLAMA_BASE_URL is set."""
    from deepagents_cli.config import create_model

    with (
        patch.dict(
            os.environ,
            {
                "OLLAMA_BASE_URL": "http://localhost:11434",
                "OLLAMA_MODEL": "qwen2.5-coder:14b",
            },
            clear=True,
        ),
        patch("deepagents_cli.config.console.print") as mock_print,
    ):
        model = create_model()

        # Verify we got a ChatOllama instance
        assert model.__class__.__name__ == "ChatOllama"
        assert model.model == "qwen2.5-coder:14b"
        assert model.base_url == "http://localhost:11434"

        # Verify console output
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "Ollama" in call_args
        assert "qwen2.5-coder:14b" in call_args


def test_create_model_with_ollama_default_model():
    """Test that create_model uses default llama2 model when OLLAMA_MODEL is not set."""
    from deepagents_cli.config import create_model

    with (
        patch.dict(
            os.environ,
            {"OLLAMA_BASE_URL": "http://localhost:11434"},
            clear=True,
        ),
        patch("deepagents_cli.config.console.print"),
    ):
        model = create_model()

        assert model.__class__.__name__ == "ChatOllama"
        assert model.model == "llama2"  # Default model


def test_create_model_with_openai():
    """Test that create_model returns ChatOpenAI when OPENAI_API_KEY is set."""
    from deepagents_cli.config import create_model

    with (
        patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key"},
            clear=True,
        ),
        patch("deepagents_cli.config.console.print"),
    ):
        model = create_model()

        assert model.__class__.__name__ == "ChatOpenAI"


def test_create_model_with_anthropic():
    """Test that create_model returns ChatAnthropic when ANTHROPIC_API_KEY is set."""
    from deepagents_cli.config import create_model

    with (
        patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "test-key"},
            clear=True,
        ),
        patch("deepagents_cli.config.console.print"),
    ):
        model = create_model()

        assert model.__class__.__name__ == "ChatAnthropic"


def test_create_model_priority_openai_over_anthropic():
    """Test that OpenAI is prioritized when both API keys are set."""
    from deepagents_cli.config import create_model

    with (
        patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-key",
                "ANTHROPIC_API_KEY": "anthropic-key",
            },
            clear=True,
        ),
        patch("deepagents_cli.config.console.print"),
    ):
        model = create_model()

        assert model.__class__.__name__ == "ChatOpenAI"


def test_create_model_priority_openai_over_ollama():
    """Test that OpenAI is prioritized when both OpenAI and Ollama are configured."""
    from deepagents_cli.config import create_model

    with (
        patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-key",
                "OLLAMA_BASE_URL": "http://localhost:11434",
            },
            clear=True,
        ),
        patch("deepagents_cli.config.console.print"),
    ):
        model = create_model()

        assert model.__class__.__name__ == "ChatOpenAI"


def test_create_model_priority_anthropic_over_ollama():
    """Test that Anthropic is prioritized when both Anthropic and Ollama are configured."""
    from deepagents_cli.config import create_model

    with (
        patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "anthropic-key",
                "OLLAMA_BASE_URL": "http://localhost:11434",
            },
            clear=True,
        ),
        patch("deepagents_cli.config.console.print"),
    ):
        model = create_model()

        assert model.__class__.__name__ == "ChatAnthropic"


def test_create_model_exits_with_no_config():
    """Test that create_model exits when no configuration is provided."""
    from deepagents_cli.config import create_model

    with (
        patch.dict(os.environ, {}, clear=True),
        patch("deepagents_cli.config.console.print"),
        pytest.raises(SystemExit) as exc_info,
    ):
        create_model()

    assert exc_info.value.code == 1


def test_ollama_model_temperature():
    """Test that Ollama model is created with correct temperature."""
    from deepagents_cli.config import create_model

    with (
        patch.dict(
            os.environ,
            {
                "OLLAMA_BASE_URL": "http://localhost:11434",
                "OLLAMA_MODEL": "qwen2.5-coder:14b",
            },
            clear=True,
        ),
        patch("deepagents_cli.config.console.print"),
    ):
        model = create_model()

        assert hasattr(model, "temperature")
        assert model.temperature == 0.7
