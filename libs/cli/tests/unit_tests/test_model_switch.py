"""Tests for model switching functionality."""

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_cli import model_config
from deepagents_cli.app import DeepAgentsApp
from deepagents_cli.config import ModelResult, settings
from deepagents_cli.model_config import ModelConfigError, clear_caches
from deepagents_cli.widgets.messages import AppMessage, ErrorMessage


@pytest.fixture(autouse=True)
def _restore_settings() -> Iterator[None]:
    """Save and restore global settings mutated by tests."""
    original_name = settings.model_name
    original_provider = settings.model_provider
    original_context_limit = settings.model_context_limit
    yield
    settings.model_name = original_name
    settings.model_provider = original_provider
    settings.model_context_limit = original_context_limit


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
            patch(
                "deepagents_cli.model_config.has_provider_credentials",
                return_value=True,
            ),
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

        with (
            patch(
                "deepagents_cli.model_config.has_provider_credentials",
                return_value=False,
            ),
            patch(
                "deepagents_cli.model_config.get_credential_env_var",
                return_value="ANTHROPIC_API_KEY",
            ),
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
            patch(
                "deepagents_cli.model_config.has_provider_credentials",
                return_value=True,
            ),
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
            patch(
                "deepagents_cli.model_config.has_provider_credentials",
                return_value=True,
            ),
            patch("deepagents_cli.app.create_model", side_effect=model_error),
            patch.object(ErrorMessage, "__init__", capture_init),
        ):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        app._mount_message.assert_called_once()  # type: ignore[union-attr]
        assert len(captured_errors) == 1
        assert "Failed to create model" in captured_errors[0]
        assert "Invalid model" in captured_errors[0]

    @pytest.mark.asyncio
    async def test_agent_recreation_failure_shows_error_and_preserves_settings(
        self,
    ) -> None:
        """_switch_model shows error and preserves settings on agent failure."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._checkpointer = MagicMock()

        # Set a different current model
        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"
        settings.model_context_limit = 128_000

        captured_errors: list[str] = []
        original_init = ErrorMessage.__init__

        def capture_init(self: ErrorMessage, message: str, **kwargs: object) -> None:
            captured_errors.append(message)
            original_init(self, message, **kwargs)

        mock_model = MagicMock()
        mock_result = ModelResult(
            model=mock_model,
            model_name="claude-sonnet-4-5",
            provider="anthropic",
            context_limit=200_000,
        )

        agent_error = RuntimeError("Agent creation failed")
        with (
            patch(
                "deepagents_cli.model_config.has_provider_credentials",
                return_value=True,
            ),
            patch("deepagents_cli.app.create_model", return_value=mock_result),
            patch("deepagents_cli.agent.create_cli_agent", side_effect=agent_error),
            patch.object(ErrorMessage, "__init__", capture_init),
        ):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        app._mount_message.assert_called_once()  # type: ignore[union-attr]
        assert len(captured_errors) == 1
        assert "Model switch failed" in captured_errors[0]
        assert "Agent creation failed" in captured_errors[0]

        # Settings are rolled back to previous values on agent creation failure
        assert settings.model_name == "gpt-4o"
        assert settings.model_provider == "openai"
        assert settings.model_context_limit == 128_000

    @pytest.mark.asyncio
    async def test_context_limit_cleared_when_new_model_has_none(self) -> None:
        """Switching to a model without a context limit clears the old value."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._checkpointer = MagicMock()

        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"
        settings.model_context_limit = 128_000

        mock_result = ModelResult(
            model=MagicMock(),
            model_name="custom-model",
            provider="ollama",
            context_limit=None,
        )
        mock_agent = MagicMock()
        mock_backend = MagicMock()

        with (
            patch(
                "deepagents_cli.model_config.has_provider_credentials",
                return_value=True,
            ),
            patch("deepagents_cli.app.create_model", return_value=mock_result),
            patch(
                "deepagents_cli.agent.create_cli_agent",
                return_value=(mock_agent, mock_backend),
            ),
            patch("deepagents_cli.app.save_recent_model", return_value=True),
        ):
            await app._switch_model("ollama:custom-model")

        assert settings.model_context_limit is None

    @pytest.mark.asyncio
    async def test_agent_failure_rollback_with_none_context_limit(self) -> None:
        """Rollback restores previous context limit when new model has None."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._checkpointer = MagicMock()

        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"
        settings.model_context_limit = 128_000

        mock_result = ModelResult(
            model=MagicMock(),
            model_name="custom-model",
            provider="custom",
            context_limit=None,
        )

        with (
            patch(
                "deepagents_cli.model_config.has_provider_credentials",
                return_value=True,
            ),
            patch("deepagents_cli.app.create_model", return_value=mock_result),
            patch(
                "deepagents_cli.agent.create_cli_agent",
                side_effect=RuntimeError("fail"),
            ),
        ):
            await app._switch_model("custom:custom-model")

        assert settings.model_context_limit == 128_000

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
            patch(
                "deepagents_cli.model_config.has_provider_credentials",
                return_value=True,
            ),
            patch("deepagents_cli.app.save_recent_model", return_value=True),
            patch.object(AppMessage, "__init__", capture_init),
        ):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        app._mount_message.assert_called_once()  # type: ignore[union-attr]
        assert len(captured_messages) == 1
        assert "Model preference set" in captured_messages[0]
        assert "Restart" in captured_messages[0]

    @pytest.mark.asyncio
    async def test_no_checkpointer_save_failure_shows_error(self) -> None:
        """_switch_model without checkpointer shows error when save fails."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._checkpointer = None

        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"

        captured_errors: list[str] = []
        original_init = ErrorMessage.__init__

        def capture_init(self: ErrorMessage, message: str, **kwargs: object) -> None:
            captured_errors.append(message)
            original_init(self, message, **kwargs)

        with (
            patch(
                "deepagents_cli.model_config.has_provider_credentials",
                return_value=True,
            ),
            patch("deepagents_cli.app.save_recent_model", return_value=False),
            patch.object(ErrorMessage, "__init__", capture_init),
        ):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        app._mount_message.assert_called_once()  # type: ignore[union-attr]
        assert len(captured_errors) == 1
        assert "Could not save model preference" in captured_errors[0]

    @pytest.mark.asyncio
    async def test_hot_swap_save_failure_warns_in_message(self) -> None:
        """Successful hot-swap warns when save_recent_model fails."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._checkpointer = MagicMock()

        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"

        captured_messages: list[str] = []
        original_init = AppMessage.__init__

        def capture_init(self: AppMessage, message: str, **kwargs: object) -> None:
            captured_messages.append(message)
            original_init(self, message, **kwargs)

        mock_model = MagicMock()
        mock_result = ModelResult(
            model=mock_model,
            model_name="claude-sonnet-4-5",
            provider="anthropic",
        )
        mock_agent = MagicMock()
        mock_backend = MagicMock()

        with (
            patch(
                "deepagents_cli.model_config.has_provider_credentials",
                return_value=True,
            ),
            patch("deepagents_cli.app.create_model", return_value=mock_result),
            patch(
                "deepagents_cli.agent.create_cli_agent",
                return_value=(mock_agent, mock_backend),
            ),
            patch("deepagents_cli.app.save_recent_model", return_value=False),
            patch.object(AppMessage, "__init__", capture_init),
        ):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        app._mount_message.assert_called_once()  # type: ignore[union-attr]
        assert len(captured_messages) == 1
        assert "Switched to" in captured_messages[0]
        assert "preference not saved" in captured_messages[0]


class TestModelSwitchConfigProvider:
    """Tests for switching to config-file-defined providers."""

    def setup_method(self) -> None:
        """Clear model config cache before each test."""
        clear_caches()

    @pytest.mark.asyncio
    async def test_switch_to_config_provider_no_whitelist_error(self, tmp_path) -> None:
        """Switching to a provider not in PROVIDER_API_KEY_ENV succeeds.

        Previously this would error with "Unknown provider". Now it falls
        through to credential check and create_model().
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.fireworks]
models = ["llama-v3p1-70b"]
api_key_env = "FIREWORKS_API_KEY"
""")
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._checkpointer = MagicMock()

        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"

        captured_messages: list[str] = []
        original_app_init = AppMessage.__init__

        def capture_app(self: AppMessage, message: str, **kwargs: object) -> None:
            captured_messages.append(message)
            original_app_init(self, message, **kwargs)

        mock_model = MagicMock()
        mock_result = ModelResult(
            model=mock_model,
            model_name="llama-v3p1-70b",
            provider="fireworks",
        )
        mock_agent = MagicMock()
        mock_backend = MagicMock()

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"FIREWORKS_API_KEY": "test-key"}),
            patch("deepagents_cli.app.create_model", return_value=mock_result),
            patch(
                "deepagents_cli.agent.create_cli_agent",
                return_value=(mock_agent, mock_backend),
            ),
            patch("deepagents_cli.app.save_recent_model", return_value=True),
            patch.object(AppMessage, "__init__", capture_app),
        ):
            await app._switch_model("fireworks:llama-v3p1-70b")

        # Should succeed, not show "Unknown provider"
        assert any("Switched to" in m for m in captured_messages)
        assert not any("Unknown provider" in m for m in captured_messages)

    @pytest.mark.asyncio
    async def test_switch_config_provider_missing_credentials(self, tmp_path) -> None:
        """Config provider with missing credentials shows appropriate error."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.fireworks]
models = ["llama-v3p1-70b"]
api_key_env = "FIREWORKS_API_KEY"
""")
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._checkpointer = MagicMock()

        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"

        captured_errors: list[str] = []
        original_err_init = ErrorMessage.__init__

        def capture_err(self: ErrorMessage, message: str, **kwargs: object) -> None:
            captured_errors.append(message)
            original_err_init(self, message, **kwargs)

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {}, clear=True),
            patch.object(ErrorMessage, "__init__", capture_err),
        ):
            await app._switch_model("fireworks:llama-v3p1-70b")

        app._mount_message.assert_called_once()  # type: ignore[union-attr]
        assert len(captured_errors) == 1
        assert "Missing credentials" in captured_errors[0]
        assert "FIREWORKS_API_KEY" in captured_errors[0]

    @pytest.mark.asyncio
    async def test_switch_to_ollama_no_key_required(self, tmp_path) -> None:
        """Ollama (no api_key_env) passes credential check."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["llama3"]
""")
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._checkpointer = MagicMock()

        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"

        captured_messages: list[str] = []
        original_app_init = AppMessage.__init__

        def capture_app(self: AppMessage, message: str, **kwargs: object) -> None:
            captured_messages.append(message)
            original_app_init(self, message, **kwargs)

        mock_model = MagicMock()
        mock_result = ModelResult(
            model=mock_model,
            model_name="llama3",
            provider="ollama",
        )
        mock_agent = MagicMock()
        mock_backend = MagicMock()

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch("deepagents_cli.app.create_model", return_value=mock_result),
            patch(
                "deepagents_cli.agent.create_cli_agent",
                return_value=(mock_agent, mock_backend),
            ),
            patch("deepagents_cli.app.save_recent_model", return_value=True),
            patch.object(AppMessage, "__init__", capture_app),
        ):
            await app._switch_model("ollama:llama3")

        assert any("Switched to" in m for m in captured_messages)


class TestModelSwitchBareModelName:
    """Tests for _switch_model with bare model names (no provider prefix)."""

    @pytest.mark.asyncio
    async def test_bare_model_name_auto_detects_provider(self) -> None:
        """Bare model name like 'gpt-4o' auto-detects provider and succeeds."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._checkpointer = MagicMock()

        settings.model_name = "claude-sonnet-4-5"
        settings.model_provider = "anthropic"

        captured_messages: list[str] = []
        original_init = AppMessage.__init__

        def capture_init(self: AppMessage, message: str, **kwargs: object) -> None:
            captured_messages.append(message)
            original_init(self, message, **kwargs)

        mock_model = MagicMock()
        mock_result = ModelResult(
            model=mock_model,
            model_name="gpt-4o",
            provider="openai",
        )
        mock_agent = MagicMock()
        mock_backend = MagicMock()

        with (
            patch("deepagents_cli.app.detect_provider", return_value="openai"),
            patch(
                "deepagents_cli.model_config.has_provider_credentials",
                return_value=True,
            ),
            patch("deepagents_cli.app.create_model", return_value=mock_result),
            patch(
                "deepagents_cli.agent.create_cli_agent",
                return_value=(mock_agent, mock_backend),
            ),
            patch("deepagents_cli.app.save_recent_model", return_value=True),
            patch.object(AppMessage, "__init__", capture_init),
        ):
            await app._switch_model("gpt-4o")

        assert any("Switched to" in m for m in captured_messages)

    @pytest.mark.asyncio
    async def test_bare_model_name_missing_credentials(self) -> None:
        """Bare model name shows credential error when provider creds are missing."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._checkpointer = MagicMock()

        settings.model_name = "claude-sonnet-4-5"
        settings.model_provider = "anthropic"

        captured_errors: list[str] = []
        original_init = ErrorMessage.__init__

        def capture_init(self: ErrorMessage, message: str, **kwargs: object) -> None:
            captured_errors.append(message)
            original_init(self, message, **kwargs)

        with (
            patch("deepagents_cli.app.detect_provider", return_value="openai"),
            patch(
                "deepagents_cli.model_config.has_provider_credentials",
                return_value=False,
            ),
            patch(
                "deepagents_cli.model_config.get_credential_env_var",
                return_value="OPENAI_API_KEY",
            ),
            patch.object(ErrorMessage, "__init__", capture_init),
        ):
            await app._switch_model("gpt-4o")

        app._mount_message.assert_called_once()  # type: ignore[union-attr]
        assert len(captured_errors) == 1
        assert "Missing credentials" in captured_errors[0]
        assert "OPENAI_API_KEY" in captured_errors[0]

    @pytest.mark.asyncio
    async def test_bare_model_name_already_using(self) -> None:
        """Bare model name matching current model shows 'Already using'."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._checkpointer = MagicMock()

        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"

        captured_messages: list[str] = []
        original_init = AppMessage.__init__

        def capture_init(self: AppMessage, message: str, **kwargs: object) -> None:
            captured_messages.append(message)
            original_init(self, message, **kwargs)

        with (
            patch("deepagents_cli.app.detect_provider", return_value="openai"),
            patch(
                "deepagents_cli.model_config.has_provider_credentials",
                return_value=True,
            ),
            patch.object(AppMessage, "__init__", capture_init),
        ):
            await app._switch_model("gpt-4o")

        app._mount_message.assert_called_once()  # type: ignore[union-attr]
        assert len(captured_messages) == 1
        assert "Already using" in captured_messages[0]
