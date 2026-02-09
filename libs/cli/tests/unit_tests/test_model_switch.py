"""Tests for model switching functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_cli.app import DeepAgentsApp
from deepagents_cli.config import settings
from deepagents_cli.model_config import ModelConfigError
from deepagents_cli.widgets.messages import AppMessage, ErrorMessage


class TestModelSwitchNoOp:
    """Tests for no-op when switching to the same model."""

    @pytest.mark.asyncio
    async def test_no_message_when_switching_to_same_model(self) -> None:
        """Switching to the already-active model should not print 'Switched to'.

        This is a regression test for the bug where selecting the same model
        from the model selector would print "Switched to X" even though no
        actual switch occurred.
        """
        app = DeepAgentsApp()
        # Replace method with mock to track calls (hence ignore)
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._checkpointer = MagicMock()  # Enable hot-swap path

        # Set current model
        settings.model_name = "claude-opus-4-5"
        settings.model_provider = "anthropic"

        captured_messages: list[str] = []
        original_init = AppMessage.__init__

        def capture_init(self: AppMessage, message: str, **kwargs: object) -> None:
            captured_messages.append(message)
            original_init(self, message, **kwargs)

        with (
            patch("deepagents_cli.app.has_provider_credentials", return_value=True),
            patch.object(AppMessage, "__init__", capture_init),
        ):
            # Attempt to switch to the same model
            await app._switch_model("anthropic:claude-opus-4-5")

        # Should show "Already using" message, not "Switched to"
        # Type checker doesn't track that _mount_message was replaced with mock
        app._mount_message.assert_called_once()  # type: ignore[union-attr]
        assert len(captured_messages) == 1
        assert "Already using" in captured_messages[0]
        assert "Switched to" not in captured_messages[0]


class TestModelSwitchErrorHandling:
    """Tests for error handling in _switch_model."""

    @pytest.mark.asyncio
    async def test_missing_credentials_shows_error(self) -> None:
        """_switch_model shows error when provider credentials are missing."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._checkpointer = MagicMock()

        # Set a different current model
        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"

        captured_errors: list[str] = []
        original_init = ErrorMessage.__init__

        def capture_init(self: ErrorMessage, message: str, **kwargs: object) -> None:
            captured_errors.append(message)
            original_init(self, message, **kwargs)

        env_map = {"anthropic": "ANTHROPIC_API_KEY"}
        with (
            patch("deepagents_cli.app.has_provider_credentials", return_value=False),
            patch("deepagents_cli.app.PROVIDER_API_KEY_ENV", env_map),
            patch.object(ErrorMessage, "__init__", capture_init),
        ):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        app._mount_message.assert_called_once()  # type: ignore[union-attr]
        assert len(captured_errors) == 1
        assert "Missing credentials" in captured_errors[0]
        assert "ANTHROPIC_API_KEY" in captured_errors[0]

    @pytest.mark.asyncio
    async def test_create_model_config_error_shows_error(self) -> None:
        """_switch_model shows error when create_model raises ModelConfigError."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._checkpointer = MagicMock()

        # Set a different current model
        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"

        captured_errors: list[str] = []
        original_init = ErrorMessage.__init__

        def capture_init(self: ErrorMessage, message: str, **kwargs: object) -> None:
            captured_errors.append(message)
            original_init(self, message, **kwargs)

        error = ModelConfigError("Missing package for provider 'anthropic'")
        with (
            patch("deepagents_cli.app.has_provider_credentials", return_value=True),
            patch("deepagents_cli.app.create_model", side_effect=error),
            patch.object(ErrorMessage, "__init__", capture_init),
        ):
            await app._switch_model("anthropic:invalid-model")

        app._mount_message.assert_called_once()  # type: ignore[union-attr]
        assert len(captured_errors) == 1
        assert "Missing package" in captured_errors[0]

    @pytest.mark.asyncio
    async def test_create_model_exception_shows_error(self) -> None:
        """_switch_model shows error when create_model raises an exception."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._checkpointer = MagicMock()

        # Set a different current model
        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"

        captured_errors: list[str] = []
        original_init = ErrorMessage.__init__

        def capture_init(self: ErrorMessage, message: str, **kwargs: object) -> None:
            captured_errors.append(message)
            original_init(self, message, **kwargs)

        model_error = ValueError("Invalid model")
        with (
            patch("deepagents_cli.app.has_provider_credentials", return_value=True),
            patch("deepagents_cli.app.create_model", side_effect=model_error),
            patch.object(ErrorMessage, "__init__", capture_init),
        ):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        app._mount_message.assert_called_once()  # type: ignore[union-attr]
        assert len(captured_errors) == 1
        assert "Failed to create model" in captured_errors[0]
        assert "Invalid model" in captured_errors[0]

    @pytest.mark.asyncio
    async def test_agent_recreation_failure_shows_error(self) -> None:
        """_switch_model shows error when create_cli_agent fails."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._checkpointer = MagicMock()

        # Set a different current model
        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"

        captured_errors: list[str] = []
        original_init = ErrorMessage.__init__

        def capture_init(self: ErrorMessage, message: str, **kwargs: object) -> None:
            captured_errors.append(message)
            original_init(self, message, **kwargs)

        mock_model = MagicMock()

        agent_error = RuntimeError("Agent creation failed")
        with (
            patch("deepagents_cli.app.has_provider_credentials", return_value=True),
            patch("deepagents_cli.app.create_model", return_value=mock_model),
            patch("deepagents_cli.app.create_cli_agent", side_effect=agent_error),
            patch.object(ErrorMessage, "__init__", capture_init),
        ):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        app._mount_message.assert_called_once()  # type: ignore[union-attr]
        assert len(captured_errors) == 1
        assert "Model switch failed" in captured_errors[0]
        assert "Agent creation failed" in captured_errors[0]

    @pytest.mark.asyncio
    async def test_no_checkpointer_saves_preference(self) -> None:
        """_switch_model without checkpointer saves preference but doesn't hot-swap."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._checkpointer = None  # No checkpointer

        # Set a different current model
        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"

        captured_messages: list[str] = []
        original_init = AppMessage.__init__

        def capture_init(self: AppMessage, message: str, **kwargs: object) -> None:
            captured_messages.append(message)
            original_init(self, message, **kwargs)

        with (
            patch("deepagents_cli.app.has_provider_credentials", return_value=True),
            patch("deepagents_cli.app.save_default_model", return_value=True),
            patch.object(AppMessage, "__init__", capture_init),
        ):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        app._mount_message.assert_called_once()  # type: ignore[union-attr]
        assert len(captured_messages) == 1
        assert "Default model set" in captured_messages[0]
        assert "Restart" in captured_messages[0]
