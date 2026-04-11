"""Tests for runtime config reload behavior."""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, call

import dotenv as _dotenv_module
import pytest

from deepagents_cli.command_registry import SLASH_COMMANDS
from deepagents_cli.config import Settings

# Capture before any monkeypatching replaces it on the module.
_real_load_dotenv = _dotenv_module.load_dotenv

_RELOAD_ENV_KEYS = (
    "OPENAI_API_KEY",
    "DEEPAGENTS_CLI_OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEEPAGENTS_CLI_ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "DEEPAGENTS_CLI_GOOGLE_API_KEY",
    "NVIDIA_API_KEY",
    "DEEPAGENTS_CLI_NVIDIA_API_KEY",
    "TAVILY_API_KEY",
    "DEEPAGENTS_CLI_TAVILY_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "DEEPAGENTS_CLI_GOOGLE_CLOUD_PROJECT",
    "DEEPAGENTS_CLI_LANGSMITH_PROJECT",
    "DEEPAGENTS_CLI_SHELL_ALLOW_LIST",
)


class TestReloadFromEnvironment:
    """Tests for `Settings.reload_from_environment`."""

    @pytest.fixture(autouse=True)
    def _clear_reload_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clear env vars used by reload tests."""
        for key in _RELOAD_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)

    @pytest.fixture(autouse=True)
    def _stub_dotenv_load(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Disable real `.env` loading for deterministic tests."""

        def _fake_load_dotenv(*_args: object, **_kwargs: object) -> bool:
            return False

        monkeypatch.setattr(
            "dotenv.load_dotenv",
            _fake_load_dotenv,
        )
        # Point global dotenv to a nonexistent path so it's never loaded
        monkeypatch.setattr(
            "deepagents_cli.config._GLOBAL_DOTENV_PATH",
            tmp_path / "nonexistent" / ".env",
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
        """Reload should preserve runtime model fields and user project."""
        settings = Settings.from_environment(start_path=tmp_path)
        settings.model_name = "gpt-5"
        settings.model_provider = "openai"
        settings.model_context_limit = 200_000
        settings.user_langchain_project = "my-project"

        monkeypatch.setenv("OPENAI_API_KEY", "sk-reloaded")
        settings.reload_from_environment(start_path=tmp_path)

        assert settings.model_name == "gpt-5"
        assert settings.model_provider == "openai"
        assert settings.model_context_limit == 200_000
        assert settings.user_langchain_project == "my-project"

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

    def test_api_key_removal_shows_unset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Removing an API key should report `set -> unset`."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
        settings = Settings.from_environment(start_path=tmp_path)

        monkeypatch.delenv("ANTHROPIC_API_KEY")
        changes = settings.reload_from_environment(start_path=tmp_path)

        assert settings.anthropic_api_key is None
        assert "anthropic_api_key: set -> unset" in changes

    def test_empty_api_key_treated_as_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Empty-string API key should be normalized to `None`."""
        monkeypatch.setenv("OPENAI_API_KEY", "")
        settings = Settings.from_environment(start_path=tmp_path)
        changes = settings.reload_from_environment(start_path=tmp_path)

        assert settings.openai_api_key is None
        assert changes == []

    def test_updates_shell_allow_list(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reload should update parsed shell allow-list values."""
        monkeypatch.setenv("DEEPAGENTS_CLI_SHELL_ALLOW_LIST", "ls,cat")
        settings = Settings.from_environment(start_path=tmp_path)
        assert settings.shell_allow_list == ["ls", "cat"]

        monkeypatch.setenv("DEEPAGENTS_CLI_SHELL_ALLOW_LIST", "ls,grep")
        changes = settings.reload_from_environment(start_path=tmp_path)

        assert settings.shell_allow_list == ["ls", "grep"]
        assert any(change.startswith("shell_allow_list:") for change in changes)

    def test_calls_dotenv_load(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reload should anchor dotenv loading to the explicit start path."""
        settings = Settings.from_environment(start_path=tmp_path)
        mock_load = MagicMock(return_value=False)
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-test\n")
        monkeypatch.setattr("dotenv.load_dotenv", mock_load)

        settings.reload_from_environment(start_path=tmp_path)

        # Project .env loads first (before global) with override=False.
        mock_load.assert_any_call(dotenv_path=env_file, override=False)

    def test_loads_global_dotenv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reload should load project dotenv first, then global."""
        settings = Settings.from_environment(start_path=tmp_path)

        global_env = tmp_path / "global" / ".env"
        global_env.parent.mkdir()
        global_env.write_text("OPENAI_API_KEY=sk-global\n")
        monkeypatch.setattr("deepagents_cli.config._GLOBAL_DOTENV_PATH", global_env)

        project_env = tmp_path / ".env"
        project_env.write_text("ANTHROPIC_API_KEY=sk-project\n")

        mock_load = MagicMock(return_value=True)
        monkeypatch.setattr("dotenv.load_dotenv", mock_load)

        settings.reload_from_environment(start_path=tmp_path)

        assert mock_load.call_count == 2
        mock_load.assert_has_calls(
            [
                call(dotenv_path=project_env, override=False),
                call(dotenv_path=global_env, override=False),
            ]
        )

    def test_global_dotenv_oserror_does_not_crash(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """OSError reading global `.env` should log a warning and continue."""
        settings = Settings.from_environment(start_path=tmp_path)

        broken = MagicMock()
        msg = "permission denied"
        broken.is_file.side_effect = OSError(msg)
        monkeypatch.setattr("deepagents_cli.config._GLOBAL_DOTENV_PATH", broken)

        # Should not raise — project .env still loads
        project_env = tmp_path / ".env"
        project_env.write_text("OPENAI_API_KEY=sk-fallback\n")

        mock_load = MagicMock(return_value=True)
        monkeypatch.setattr("dotenv.load_dotenv", mock_load)

        with caplog.at_level(logging.WARNING, logger="deepagents_cli.config"):
            settings.reload_from_environment(start_path=tmp_path)

        assert any("Could not read global dotenv" in r.message for r in caplog.records)
        # Project .env loads first; global failed via is_file OSError
        mock_load.assert_called_once_with(dotenv_path=project_env, override=False)

    def test_project_dotenv_beats_global(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Project `.env` should always beat global `.env`."""
        from deepagents_cli.config import _load_dotenv

        global_dir = tmp_path / "global"
        global_dir.mkdir()
        global_env = global_dir / ".env"
        global_env.write_text("TEST_PRECEDENCE_KEY=global-value\n")
        monkeypatch.setattr("deepagents_cli.config._GLOBAL_DOTENV_PATH", global_env)

        project_env = tmp_path / ".env"
        project_env.write_text("TEST_PRECEDENCE_KEY=project-value\n")

        # Use real dotenv (not the stub) to test actual precedence
        monkeypatch.setattr(
            "dotenv.load_dotenv",
            _real_load_dotenv,
        )
        monkeypatch.delenv("TEST_PRECEDENCE_KEY", raising=False)

        _load_dotenv(start_path=tmp_path)

        assert os.environ.get("TEST_PRECEDENCE_KEY") == "project-value"
        monkeypatch.delenv("TEST_PRECEDENCE_KEY", raising=False)

    def test_shell_env_beats_project_dotenv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Shell-exported vars should beat project `.env`."""
        from deepagents_cli.config import _load_dotenv

        # No global dotenv
        monkeypatch.setattr(
            "deepagents_cli.config._GLOBAL_DOTENV_PATH",
            tmp_path / "nonexistent" / ".env",
        )

        project_env = tmp_path / ".env"
        project_env.write_text("TEST_SHELL_PROJECT_KEY=project-value\n")

        monkeypatch.setenv("TEST_SHELL_PROJECT_KEY", "shell-value")

        monkeypatch.setattr(
            "dotenv.load_dotenv",
            _real_load_dotenv,
        )

        _load_dotenv(start_path=tmp_path)

        assert os.environ.get("TEST_SHELL_PROJECT_KEY") == "shell-value"
        monkeypatch.delenv("TEST_SHELL_PROJECT_KEY", raising=False)

    def test_shell_env_beats_global_dotenv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Shell-exported vars should beat global `~/.deepagents/.env`."""
        from deepagents_cli.config import _load_dotenv

        global_dir = tmp_path / "global"
        global_dir.mkdir()
        global_env = global_dir / ".env"
        global_env.write_text("TEST_BOOT_KEY=global-value\n")
        monkeypatch.setattr("deepagents_cli.config._GLOBAL_DOTENV_PATH", global_env)

        # Simulate a shell-exported variable (e.g., from $ZDOTDIR/.env)
        monkeypatch.setenv("TEST_BOOT_KEY", "shell-value")

        monkeypatch.setattr(
            "dotenv.load_dotenv",
            _real_load_dotenv,
        )
        # No project .env
        monkeypatch.setattr(
            "deepagents_cli.config._find_dotenv_from_start_path",
            lambda _: None,
        )

        _load_dotenv(start_path=tmp_path)

        assert os.environ.get("TEST_BOOT_KEY") == "shell-value"
        monkeypatch.delenv("TEST_BOOT_KEY", raising=False)

    def test_global_only_no_project_dotenv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Global `.env` values should apply when no project `.env` exists."""
        from deepagents_cli.config import _load_dotenv

        global_dir = tmp_path / "global"
        global_dir.mkdir()
        global_env = global_dir / ".env"
        global_env.write_text("TEST_GLOBAL_ONLY=global-value\n")
        monkeypatch.setattr("deepagents_cli.config._GLOBAL_DOTENV_PATH", global_env)

        monkeypatch.setattr(
            "dotenv.load_dotenv",
            _real_load_dotenv,
        )
        monkeypatch.delenv("TEST_GLOBAL_ONLY", raising=False)

        # No .env in isolated dir; global is the only source
        monkeypatch.setattr(
            "deepagents_cli.config._find_dotenv_from_start_path",
            lambda _: None,
        )
        isolated = tmp_path / "no_project_env"
        isolated.mkdir()
        result = _load_dotenv(start_path=isolated)

        assert result is True
        assert os.environ.get("TEST_GLOBAL_ONLY") == "global-value"
        monkeypatch.delenv("TEST_GLOBAL_ONLY", raising=False)

    def test_global_load_dotenv_raises_oserror(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """OSError from `dotenv.load_dotenv` itself (not `is_file`) is caught."""
        settings = Settings.from_environment(start_path=tmp_path)

        global_env = tmp_path / "global" / ".env"
        global_env.parent.mkdir()
        global_env.write_text("KEY=val\n")
        monkeypatch.setattr("deepagents_cli.config._GLOBAL_DOTENV_PATH", global_env)

        project_env = tmp_path / ".env"
        project_env.write_text("OPENAI_API_KEY=sk-ok\n")

        call_count = 0

        def _fail_on_global(*_args: object, **_kwargs: object) -> bool:
            nonlocal call_count
            call_count += 1
            # Project loads first; global is the second call
            if call_count == 2:
                msg = "read error"
                raise OSError(msg)
            return True

        monkeypatch.setattr("dotenv.load_dotenv", _fail_on_global)

        with caplog.at_level(logging.WARNING, logger="deepagents_cli.config"):
            settings.reload_from_environment(start_path=tmp_path)

        assert call_count == 2
        assert any("Could not read global dotenv" in r.message for r in caplog.records)

    def test_multiple_simultaneous_changes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reload should accumulate changes across multiple fields."""
        settings = Settings.from_environment(start_path=tmp_path)

        monkeypatch.setenv("OPENAI_API_KEY", "sk-new")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        monkeypatch.setenv("DEEPAGENTS_CLI_SHELL_ALLOW_LIST", "ls")
        changes = settings.reload_from_environment(start_path=tmp_path)

        assert len(changes) == 3
        fields = {c.split(":")[0] for c in changes}
        assert fields == {"openai_api_key", "anthropic_api_key", "shell_allow_list"}

    def test_prefixed_env_var_beats_canonical(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """DEEPAGENTS_CLI_ prefixed var should override canonical on reload."""
        settings = Settings.from_environment(start_path=tmp_path)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-canonical")
        monkeypatch.setenv("DEEPAGENTS_CLI_ANTHROPIC_API_KEY", "sk-override")
        settings.reload_from_environment(start_path=tmp_path)

        assert settings.anthropic_api_key == "sk-override"

    def test_from_environment_uses_prefixed_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Settings.from_environment should honour the DEEPAGENTS_CLI_ prefix."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-canonical")
        monkeypatch.setenv("DEEPAGENTS_CLI_OPENAI_API_KEY", "sk-override")

        settings = Settings.from_environment(start_path=tmp_path)

        assert settings.openai_api_key == "sk-override"


class TestReloadErrorPaths:
    """Tests for error handling during reload."""

    @pytest.fixture(autouse=True)
    def _clear_reload_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clear env vars used by reload tests."""
        for key in _RELOAD_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)

    @pytest.fixture(autouse=True)
    def _stub_dotenv_load(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Disable real `.env` loading for deterministic tests."""

        def _fake_load_dotenv(*_args: object, **_kwargs: object) -> bool:
            return False

        monkeypatch.setattr(
            "dotenv.load_dotenv",
            _fake_load_dotenv,
        )
        monkeypatch.setattr(
            "deepagents_cli.config._GLOBAL_DOTENV_PATH",
            tmp_path / "nonexistent" / ".env",
        )

    def test_invalid_shell_allow_list_keeps_previous(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Malformed shell allow-list should fall back to previous value."""
        monkeypatch.setenv("DEEPAGENTS_CLI_SHELL_ALLOW_LIST", "ls,cat")
        settings = Settings.from_environment(start_path=tmp_path)
        assert settings.shell_allow_list == ["ls", "cat"]

        monkeypatch.setenv("DEEPAGENTS_CLI_SHELL_ALLOW_LIST", "all,ls")
        changes = settings.reload_from_environment(start_path=tmp_path)

        assert settings.shell_allow_list == ["ls", "cat"]
        assert not any(change.startswith("shell_allow_list:") for change in changes)

    def test_deleted_cwd_keeps_previous_project_root(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Unreachable cwd should fall back to previous project root."""
        settings = Settings.from_environment(start_path=tmp_path)
        original_root = settings.project_root

        def _raise_oserror(_start: Path | None = None) -> None:
            msg = "No such file or directory"
            raise FileNotFoundError(msg)

        monkeypatch.setattr(
            "deepagents_cli.project_utils.find_project_root", _raise_oserror
        )
        changes = settings.reload_from_environment(start_path=tmp_path)

        assert settings.project_root == original_root
        assert not any(change.startswith("project_root:") for change in changes)

    def test_settings_consistent_after_partial_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Settings should remain consistent when one field fails to reload."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-original")
        monkeypatch.setenv("DEEPAGENTS_CLI_SHELL_ALLOW_LIST", "ls")
        settings = Settings.from_environment(start_path=tmp_path)

        # Change API key (succeeds) + break shell allow-list (falls back)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-updated")
        monkeypatch.setenv("DEEPAGENTS_CLI_SHELL_ALLOW_LIST", "all,ls")
        changes = settings.reload_from_environment(start_path=tmp_path)

        assert settings.openai_api_key == "sk-updated"
        assert settings.shell_allow_list == ["ls"]
        assert any(c.startswith("openai_api_key:") for c in changes)


class TestReloadInAutocomplete:
    """Tests for autocomplete slash command registration."""

    def test_reload_in_slash_commands(self) -> None:
        """`/reload` should be registered in slash command completions."""
        assert any(command == "/reload" for command, _, _, _ in SLASH_COMMANDS)
