"""Tests for runtime config reload behavior."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deepagents_cli.config import Settings
from deepagents_cli.widgets.autocomplete import SLASH_COMMANDS

_RELOAD_ENV_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "NVIDIA_API_KEY",
    "TAVILY_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "DEEPAGENTS_LANGSMITH_PROJECT",
    "DEEPAGENTS_SHELL_ALLOW_LIST",
)


class TestReloadFromEnvironment:
    """Tests for `Settings.reload_from_environment`."""

    @pytest.fixture(autouse=True)
    def _clear_reload_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clear env vars used by reload tests."""
        for key in _RELOAD_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)

    @pytest.fixture(autouse=True)
    def _stub_dotenv_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Disable real `.env` loading for deterministic tests."""

        def _fake_load_dotenv(*_args: object, **_kwargs: object) -> bool:
            return False

        monkeypatch.setattr(
            "deepagents_cli.config.dotenv.load_dotenv",
            _fake_load_dotenv,
        )

    def test_picks_up_new_api_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reload should read API keys added after initialization."""
        settings = Settings.from_environment(start_path=tmp_path)
        assert settings.openai_api_key is None

        monkeypatch.setenv("OPENAI_API_KEY", "sk-new-key")
        changes = settings.reload_from_environment(start_path=tmp_path)

        assert settings.openai_api_key == "sk-new-key"
        assert "openai_api_key: unset -> set" in changes

    def test_preserves_model_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reload should preserve runtime model fields."""
        settings = Settings.from_environment(start_path=tmp_path)
        settings.model_name = "gpt-5"
        settings.model_provider = "openai"
        settings.model_context_limit = 200_000

        monkeypatch.setenv("OPENAI_API_KEY", "sk-reloaded")
        settings.reload_from_environment(start_path=tmp_path)

        assert settings.model_name == "gpt-5"
        assert settings.model_provider == "openai"
        assert settings.model_context_limit == 200_000

    def test_no_changes_returns_empty(self, tmp_path: Path) -> None:
        """Reload should report no changes when environment is unchanged."""
        settings = Settings.from_environment(start_path=tmp_path)
        changes = settings.reload_from_environment(start_path=tmp_path)

        assert changes == []

    def test_masks_api_keys_in_report(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Change reports should mask API key values."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-old-secret")
        settings = Settings.from_environment(start_path=tmp_path)

        monkeypatch.setenv("OPENAI_API_KEY", "sk-new-secret")
        changes = settings.reload_from_environment(start_path=tmp_path)
        key_changes = [
            change for change in changes if change.startswith("openai_api_key:")
        ]

        assert key_changes == ["openai_api_key: set -> set"]
        assert "sk-old-secret" not in key_changes[0]
        assert "sk-new-secret" not in key_changes[0]

    def test_updates_shell_allow_list(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reload should update parsed shell allow-list values."""
        monkeypatch.setenv("DEEPAGENTS_SHELL_ALLOW_LIST", "ls,cat")
        settings = Settings.from_environment(start_path=tmp_path)
        assert settings.shell_allow_list == ["ls", "cat"]

        monkeypatch.setenv("DEEPAGENTS_SHELL_ALLOW_LIST", "ls,grep")
        changes = settings.reload_from_environment(start_path=tmp_path)

        assert settings.shell_allow_list == ["ls", "grep"]
        assert any(change.startswith("shell_allow_list:") for change in changes)

    def test_calls_dotenv_load(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reload should call dotenv with override enabled."""
        settings = Settings.from_environment(start_path=tmp_path)
        mock_load = MagicMock(return_value=False)
        monkeypatch.setattr("deepagents_cli.config.dotenv.load_dotenv", mock_load)

        settings.reload_from_environment(start_path=tmp_path)

        mock_load.assert_called_once_with(override=True)


class TestReloadInAutocomplete:
    """Tests for autocomplete slash command registration."""

    def test_reload_in_slash_commands(self) -> None:
        """`/reload` should be registered in slash command completions."""
        assert any(command == "/reload" for command, _, _ in SLASH_COMMANDS)
