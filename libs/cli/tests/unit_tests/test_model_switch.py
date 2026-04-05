"""Tests for model switching functionality."""

from collections.abc import Iterator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from deepagents_cli import model_config
from deepagents_cli.app import DeepAgentsApp, _extract_model_params_flag
from deepagents_cli.config import settings
from deepagents_cli.model_config import ModelSpec, clear_caches
from deepagents_cli.remote_client import RemoteAgent
from deepagents_cli.widgets.messages import AppMessage, ErrorMessage


def _make_remote_agent() -> RemoteAgent:
    """Create a RemoteAgent pointing at a dummy URL for test scaffolding."""
    return RemoteAgent("http://test:0")


class _FakeModelResult:
    """Minimal model result for `_switch_model` tests."""

    def __init__(
        self,
        *,
        model_name: str,
        provider: str,
        context_limit: int,
        unsupported_modalities: frozenset[str] = frozenset(),
    ) -> None:
        self.model_name = model_name
        self.provider = provider
        self.context_limit = context_limit
        self.unsupported_modalities = unsupported_modalities

    def apply_to_settings(self) -> None:
        """Mirror `ModelResult.apply_to_settings()` for test isolation."""
        settings.model_name = self.model_name
        settings.model_provider = self.provider
        settings.model_context_limit = self.context_limit
        settings.model_unsupported_modalities = self.unsupported_modalities


@pytest.fixture(autouse=True)
def _restore_settings() -> Iterator[None]:
    """Save and restore global settings mutated by tests."""
    original_name = settings.model_name
    original_provider = settings.model_provider
    original_context_limit = settings.model_context_limit
    original_modalities = settings.model_unsupported_modalities
    yield
    settings.model_name = original_name
    settings.model_provider = original_provider
    settings.model_context_limit = original_context_limit
    settings.model_unsupported_modalities = original_modalities


@pytest.fixture(autouse=True)
def mock_create_model() -> Iterator[Mock]:
    """Avoid provider package imports while preserving metadata updates."""
    context_limits = {
        "anthropic:claude-opus-4-5": 200_000,
        "anthropic:claude-sonnet-4-5": 200_000,
        "fireworks:llama-v3p1-70b": 131_072,
        "ollama:llama3": 8_192,
        "openai:gpt-4o": 128_000,
    }

    def fake_create_model(
        model_spec: str,
        *,
        extra_kwargs: dict[str, object] | None = None,
        profile_overrides: dict[str, object] | None = None,
    ) -> _FakeModelResult:
        del extra_kwargs, profile_overrides
        parsed = ModelSpec.try_parse(model_spec)
        if parsed is None:
            provider = "openai"
            model_name = model_spec
        else:
            provider = parsed.provider
            model_name = parsed.model

        context_limit = context_limits.get(f"{provider}:{model_name}", 65_536)
        return _FakeModelResult(
            model_name=model_name,
            provider=provider,
            context_limit=context_limit,
        )

    with patch(
        "deepagents_cli.config.create_model",
        side_effect=fake_create_model,
    ) as mock:
        yield mock


class TestModelSwitchNoOp:
    """Tests for no-op when switching to the same model."""

    async def test_no_message_when_switching_to_same_model(self) -> None:
        """Switching to the already-active model should not print 'Switched to'.

        This is a regression test for the bug where selecting the same model
        from the model selector would print "Switched to X" even though no
        actual switch occurred.
        """
        app = DeepAgentsApp()
        # Replace method with mock to track calls (hence ignore)
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._agent = _make_remote_agent()

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
        assert app._model_switching is False


class TestModelSwitchErrorHandling:
    """Tests for error handling in _switch_model."""

    async def test_missing_credentials_shows_error(self) -> None:
        """_switch_model shows error when provider credentials are missing."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._agent = _make_remote_agent()

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
        assert app._model_switching is False

    async def test_codex_missing_session_shows_codex_specific_error(self) -> None:
        """Codex provider shows a targeted login-session error."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._agent = _make_remote_agent()

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
            patch.object(ErrorMessage, "__init__", capture_init),
        ):
            await app._switch_model("codex:o4-mini")

        app._mount_message.assert_called_once()  # type: ignore[union-attr]
        assert len(captured_errors) == 1
        assert "Codex session not found" in captured_errors[0]
        assert "codex login" in captured_errors[0]
        assert app._model_switching is False

    async def test_save_recent_model_failure_shows_warning(self) -> None:
        """Permission error saving recent model shows error, no success message."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._agent = _make_remote_agent()

        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"

        captured_errors: list[str] = []
        original_err_init = ErrorMessage.__init__

        def capture_err(self: ErrorMessage, message: str, **kwargs: object) -> None:
            captured_errors.append(message)
            original_err_init(self, message, **kwargs)

        captured_messages: list[str] = []
        original_app_init = AppMessage.__init__

        def capture_app(self: AppMessage, message: str, **kwargs: object) -> None:
            captured_messages.append(message)
            original_app_init(self, message, **kwargs)

        with (
            patch(
                "deepagents_cli.model_config.has_provider_credentials",
                return_value=True,
            ),
            patch("deepagents_cli.model_config.save_recent_model", return_value=False),
            patch.object(ErrorMessage, "__init__", capture_err),
            patch.object(AppMessage, "__init__", capture_app),
        ):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        # Should warn about save failure
        assert len(captured_errors) == 1
        assert "could not save" in captured_errors[0].lower()
        assert "~/.deepagents/" in captured_errors[0]

        # Should NOT show success message when save fails
        assert not any("Switched to" in m for m in captured_messages)
        assert app._model_override == "anthropic:claude-sonnet-4-5"

    async def test_remote_agent_sets_model_override(self) -> None:
        """With remote agent, sets model override for ConfigurableModelMiddleware."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._agent = _make_remote_agent()

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
            patch(
                "deepagents_cli.model_config.save_recent_model", return_value=True
            ) as mock_save,
            patch.object(AppMessage, "__init__", capture_init),
        ):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        assert app._model_override == "anthropic:claude-sonnet-4-5"
        assert app._model_params_override is None
        mock_save.assert_called_once()
        assert settings.model_name == "claude-sonnet-4-5"
        assert settings.model_provider == "anthropic"
        assert any("Switched to" in m for m in captured_messages)

    async def test_remote_agent_refreshes_model_metadata(
        self, mock_create_model: Mock
    ) -> None:
        """Switching models should refresh derived settings like context size."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._agent = _make_remote_agent()
        app._profile_override = {"max_input_tokens": 180_000}

        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"
        settings.model_context_limit = 128_000

        with (
            patch(
                "deepagents_cli.model_config.has_provider_credentials",
                return_value=True,
            ),
            patch("deepagents_cli.model_config.save_recent_model", return_value=True),
        ):
            await app._switch_model(
                "anthropic:claude-sonnet-4-5",
                extra_kwargs={"temperature": 0.7},
            )

        assert settings.model_name == "claude-sonnet-4-5"
        assert settings.model_provider == "anthropic"
        assert settings.model_context_limit == 200_000
        mock_create_model.assert_called_once_with(
            "anthropic:claude-sonnet-4-5",
            extra_kwargs={"temperature": 0.7},
            profile_overrides={"max_input_tokens": 180_000},
        )

    async def test_remote_agent_sets_model_params_override(self) -> None:
        """With remote agent, extra_kwargs are stored as _model_params_override."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._agent = _make_remote_agent()

        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"

        with (
            patch(
                "deepagents_cli.model_config.has_provider_credentials",
                return_value=True,
            ),
            patch("deepagents_cli.model_config.save_recent_model", return_value=True),
        ):
            await app._switch_model(
                "anthropic:claude-sonnet-4-5",
                extra_kwargs={"temperature": 0.7, "max_tokens": 1024},
            )

        assert app._model_override == "anthropic:claude-sonnet-4-5"
        assert app._model_params_override == {
            "temperature": 0.7,
            "max_tokens": 1024,
        }


class TestModelSwitchConcurrencyGuard:
    """Tests for _model_switching concurrency guard."""

    async def test_concurrent_model_switch_blocked(self) -> None:
        """Second _switch_model call is rejected while first is in-flight."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._model_switching = True

        captured_messages: list[str] = []
        original_init = AppMessage.__init__

        def capture_init(self: AppMessage, message: str, **kwargs: object) -> None:
            captured_messages.append(message)
            original_init(self, message, **kwargs)

        with patch.object(AppMessage, "__init__", capture_init):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        app._mount_message.assert_called_once()  # type: ignore[union-attr]
        assert len(captured_messages) == 1
        assert "already in progress" in captured_messages[0]

    async def test_model_switching_flag_reset_on_success(self) -> None:
        """_model_switching resets to False after a successful switch."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._agent = _make_remote_agent()

        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"

        with (
            patch(
                "deepagents_cli.model_config.has_provider_credentials",
                return_value=True,
            ),
            patch("deepagents_cli.model_config.save_recent_model", return_value=True),
        ):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        assert app._model_switching is False


class TestModelSwitchConfigProvider:
    """Tests for switching to config-file-defined providers."""

    def setup_method(self) -> None:
        """Clear model config cache before each test."""
        clear_caches()

    async def test_switch_to_config_provider_no_whitelist_error(self, tmp_path) -> None:
        """Switching to a provider not in PROVIDER_API_KEY_ENV succeeds.

        Previously this would error with "Unknown provider". Now it switches
        immediately in the server-backed session.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.fireworks]
models = ["llama-v3p1-70b"]
api_key_env = "FIREWORKS_API_KEY"
""")
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._agent = _make_remote_agent()

        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"

        captured_messages: list[str] = []
        original_app_init = AppMessage.__init__

        def capture_app(self: AppMessage, message: str, **kwargs: object) -> None:
            captured_messages.append(message)
            original_app_init(self, message, **kwargs)

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"FIREWORKS_API_KEY": "test-key"}),
            patch(
                "deepagents_cli.model_config.save_recent_model", return_value=True
            ) as mock_save,
            patch.object(AppMessage, "__init__", capture_app),
        ):
            await app._switch_model("fireworks:llama-v3p1-70b")

        mock_save.assert_called_once_with("fireworks:llama-v3p1-70b")
        assert app._model_override == "fireworks:llama-v3p1-70b"
        assert settings.model_name == "llama-v3p1-70b"
        assert settings.model_provider == "fireworks"
        # Should succeed, not show "Unknown provider"
        assert any(
            "Switched to fireworks:llama-v3p1-70b" in m for m in captured_messages
        )
        assert not any("Unknown provider" in m for m in captured_messages)

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
        app._agent = _make_remote_agent()

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

    async def test_switch_to_ollama_no_key_required(self, tmp_path) -> None:
        """Ollama (no api_key_env) passes credential check and switches."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["llama3"]
""")
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._agent = _make_remote_agent()

        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"

        captured_messages: list[str] = []
        original_app_init = AppMessage.__init__

        def capture_app(self: AppMessage, message: str, **kwargs: object) -> None:
            captured_messages.append(message)
            original_app_init(self, message, **kwargs)

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch(
                "deepagents_cli.model_config.save_recent_model", return_value=True
            ) as mock_save,
            patch.object(AppMessage, "__init__", capture_app),
        ):
            await app._switch_model("ollama:llama3")

        mock_save.assert_called_once_with("ollama:llama3")
        assert app._model_override == "ollama:llama3"
        assert settings.model_name == "llama3"
        assert settings.model_provider == "ollama"
        assert any("Switched to ollama:llama3" in m for m in captured_messages)


class TestModelSwitchBareModelName:
    """Tests for _switch_model with bare model names (no provider prefix)."""

    async def test_bare_model_name_auto_detects_provider(self) -> None:
        """Bare model name like 'gpt-4o' auto-detects provider and switches."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._agent = _make_remote_agent()

        settings.model_name = "claude-sonnet-4-5"
        settings.model_provider = "anthropic"

        captured_messages: list[str] = []
        original_init = AppMessage.__init__

        def capture_init(self: AppMessage, message: str, **kwargs: object) -> None:
            captured_messages.append(message)
            original_init(self, message, **kwargs)

        with (
            patch("deepagents_cli.config.detect_provider", return_value="openai"),
            patch(
                "deepagents_cli.model_config.has_provider_credentials",
                return_value=True,
            ),
            patch(
                "deepagents_cli.model_config.save_recent_model", return_value=True
            ) as mock_save,
            patch.object(AppMessage, "__init__", capture_init),
        ):
            await app._switch_model("gpt-4o")

        mock_save.assert_called_once_with("openai:gpt-4o")
        assert app._model_override == "openai:gpt-4o"
        assert settings.model_name == "gpt-4o"
        assert settings.model_provider == "openai"
        assert any("Switched to openai:gpt-4o" in m for m in captured_messages)

    async def test_bare_model_name_missing_credentials(self) -> None:
        """Bare model name shows credential error when provider creds are missing."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._agent = _make_remote_agent()

        settings.model_name = "claude-sonnet-4-5"
        settings.model_provider = "anthropic"

        captured_errors: list[str] = []
        original_init = ErrorMessage.__init__

        def capture_init(self: ErrorMessage, message: str, **kwargs: object) -> None:
            captured_errors.append(message)
            original_init(self, message, **kwargs)

        with (
            patch("deepagents_cli.config.detect_provider", return_value="openai"),
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

    async def test_bare_model_name_already_using(self) -> None:
        """Bare model name matching current model shows 'Already using'."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]
        app._agent = _make_remote_agent()

        settings.model_name = "gpt-4o"
        settings.model_provider = "openai"

        captured_messages: list[str] = []
        original_init = AppMessage.__init__

        def capture_init(self: AppMessage, message: str, **kwargs: object) -> None:
            captured_messages.append(message)
            original_init(self, message, **kwargs)

        with (
            patch("deepagents_cli.config.detect_provider", return_value="openai"),
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


class TestExtractModelParamsFlag:
    """Tests for _extract_model_params_flag helper."""

    def test_no_flag(self) -> None:
        """Returns original string and None when flag absent."""
        remaining, params = _extract_model_params_flag("anthropic:claude-sonnet-4-5")
        assert remaining == "anthropic:claude-sonnet-4-5"
        assert params is None

    def test_single_quoted_json(self) -> None:
        """Extracts JSON from single-quoted value."""
        raw = """--model-params '{"temperature": 0.7}' anthropic:claude-sonnet-4-5"""
        remaining, params = _extract_model_params_flag(raw)
        assert remaining == "anthropic:claude-sonnet-4-5"
        assert params == {"temperature": 0.7}

    def test_double_quoted_json_with_escaped_quotes(self) -> None:
        """Extracts JSON from double-quoted value with escaped inner quotes."""
        raw = '--model-params "{\\"temperature\\": 0.7}" anthropic:claude-sonnet-4-5'
        remaining, params = _extract_model_params_flag(raw)
        assert remaining == "anthropic:claude-sonnet-4-5"
        assert params == {"temperature": 0.7}

    def test_bare_braces(self) -> None:
        """Extracts JSON from unquoted braces with balanced matching."""
        raw = '--model-params {"temperature": 0.7, "max_tokens": 100}'
        remaining, params = _extract_model_params_flag(raw)
        assert remaining == ""
        assert params == {"temperature": 0.7, "max_tokens": 100}

    def test_bare_braces_with_model_after(self) -> None:
        """Model arg after bare-brace JSON is preserved."""
        raw = '--model-params {"temperature":0.7} anthropic:claude-sonnet-4-5'
        remaining, params = _extract_model_params_flag(raw)
        assert remaining == "anthropic:claude-sonnet-4-5"
        assert params == {"temperature": 0.7}

    def test_model_before_flag(self) -> None:
        """Model arg before --model-params is preserved."""
        raw = "anthropic:claude-sonnet-4-5 --model-params '{\"temperature\": 0.7}'"
        remaining, params = _extract_model_params_flag(raw)
        assert remaining == "anthropic:claude-sonnet-4-5"
        assert params == {"temperature": 0.7}

    def test_missing_value_raises(self) -> None:
        """Raises ValueError when --model-params has no value."""
        with pytest.raises(ValueError, match="requires a JSON object"):
            _extract_model_params_flag("--model-params")

    def test_invalid_json_raises(self) -> None:
        """Raises ValueError with hint for malformed JSON."""
        with pytest.raises(ValueError, match=r"Invalid JSON.*Expected format"):
            _extract_model_params_flag("--model-params '{not json}'")

    def test_non_dict_json_raises(self) -> None:
        """Raises TypeError when JSON is not an object."""
        with pytest.raises(TypeError, match="must be a JSON object"):
            _extract_model_params_flag("--model-params '[1, 2, 3]'")

    def test_unclosed_quote_raises(self) -> None:
        """Raises ValueError for unclosed quote."""
        with pytest.raises(ValueError, match="Unclosed"):
            _extract_model_params_flag("""--model-params '{"temperature": 0.7}""")

    def test_unbalanced_braces_raises(self) -> None:
        """Raises ValueError for unbalanced braces."""
        with pytest.raises(ValueError, match="Unbalanced"):
            _extract_model_params_flag('--model-params {"temperature": 0.7')

    def test_with_default_flag(self) -> None:
        """Works alongside --default flag."""
        raw = (
            """--model-params '{"temperature": 0.7}' """
            "--default anthropic:claude-sonnet-4-5"
        )
        remaining, params = _extract_model_params_flag(raw)
        assert remaining == "--default anthropic:claude-sonnet-4-5"
        assert params == {"temperature": 0.7}

    def test_empty_object(self) -> None:
        """Empty JSON object is valid."""
        remaining, params = _extract_model_params_flag("--model-params '{}'")
        assert remaining == ""
        assert params == {}


class TestModelCommandIntegration:
    """Tests for /model command handler integration."""

    async def test_invalid_model_params_shows_error(self) -> None:
        """/model with invalid --model-params JSON shows error."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]

        captured_errors: list[str] = []
        original_init = ErrorMessage.__init__

        def capture_init(self: ErrorMessage, message: str, **kwargs: object) -> None:
            captured_errors.append(message)
            original_init(self, message, **kwargs)

        with patch.object(ErrorMessage, "__init__", capture_init):
            await app._handle_command("/model --model-params '{bad}'")

        assert len(captured_errors) == 1
        assert "Invalid JSON" in captured_errors[0]
        assert "Expected format" in captured_errors[0]

    async def test_model_params_with_default_rejected(self) -> None:
        """/model --model-params with --default shows error."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # type: ignore[method-assign]

        captured_errors: list[str] = []
        original_init = ErrorMessage.__init__

        def capture_init(self: ErrorMessage, message: str, **kwargs: object) -> None:
            captured_errors.append(message)
            original_init(self, message, **kwargs)

        cmd = (
            """/model --model-params '{"temperature": 0.7}' """
            "--default anthropic:claude-sonnet-4-5"
        )
        with patch.object(ErrorMessage, "__init__", capture_init):
            await app._handle_command(cmd)

        assert len(captured_errors) == 1
        assert "cannot be used with --default" in captured_errors[0]
