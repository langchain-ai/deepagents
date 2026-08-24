"""Tests for runtime config reload behavior."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import dotenv as _dotenv_module
import pytest

from deepagents_code import _env_vars
from deepagents_code.command_registry import get_slash_commands
from deepagents_code.config import Settings
from deepagents_code.skills.load import ExtendedSkillMetadata

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from pathlib import Path

    from deepagents_code.app import _PluginFingerprint
    from deepagents_code.configuration.types import TomlSnapshot
    from deepagents_code.plugins.models import PluginInstance
    from deepagents_code.tui.modals.plugin_manager import PluginManagerScreen
    from deepagents_code.tui.modals.plugin_manager.models import PluginManagerResult

# Capture before any monkeypatching replaces it on the module.
_real_load_dotenv = _dotenv_module.load_dotenv


def _test_plugin_fingerprint(version: str) -> _PluginFingerprint:
    from deepagents_code.app import _PluginFingerprint

    return _PluginFingerprint(version=version, manifest=None, components=())


async def _check_plugin_reload(screen: PluginManagerScreen) -> bool | None:
    check_reload_required = screen._check_reload_required
    assert check_reload_required is not None
    return await check_reload_required()


_RELOAD_ENV_KEYS = (
    "OPENAI_API_KEY",
    "DEEPAGENTS_CODE_OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEEPAGENTS_CODE_ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "DEEPAGENTS_CODE_GOOGLE_API_KEY",
    "NVIDIA_API_KEY",
    "DEEPAGENTS_CODE_NVIDIA_API_KEY",
    "TAVILY_API_KEY",
    "DEEPAGENTS_CODE_TAVILY_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "DEEPAGENTS_CODE_GOOGLE_CLOUD_PROJECT",
    "DEEPAGENTS_CODE_LANGSMITH_PROJECT",
    "DEEPAGENTS_CODE_SHELL_ALLOW_LIST",
    "DEEPAGENTS_CODE_EXTRA_SKILLS_DIRS",
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
            "deepagents_code.config._GLOBAL_DOTENV_PATH",
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

    def test_preview_reload_reports_changes_without_mutating(
        self, tmp_path: Path
    ) -> None:
        """Previewing reload changes should not update settings or `os.environ`."""
        current = tmp_path / "current"
        target = tmp_path / "target"
        current.mkdir()
        target.mkdir()
        (target / ".env").write_text("DEEPAGENTS_CODE_SHELL_ALLOW_LIST=ls\n")
        settings = Settings.from_environment(start_path=current)

        changes = settings.preview_reload_from_environment(start_path=target)

        assert any(change.startswith("shell_allow_list:") for change in changes)
        assert settings.shell_allow_list is None
        assert "DEEPAGENTS_CODE_SHELL_ALLOW_LIST" not in os.environ

    def test_preview_reload_sees_shell_allow_list_toml_edit(
        self, tmp_path: Path
    ) -> None:
        """A preview reports a `[shell].allow_list` edit made since startup.

        The shared resolver's user snapshot is cached at startup; a preview
        that read it would report no change while the accepted reload applies
        the edit, breaking the preview/apply contract. The preview resolves
        the user tier from a fresh file read instead.
        """
        from deepagents_code import model_config

        config_path = model_config.DEFAULT_CONFIG_PATH
        settings = Settings.from_environment(start_path=tmp_path)
        assert settings.shell_allow_list is None

        config_path.write_text('[shell]\nallow_list = ["ls"]\n', encoding="utf-8")
        changes = settings.preview_reload_from_environment(start_path=tmp_path)

        assert any(change.startswith("shell_allow_list:") for change in changes)
        assert settings.shell_allow_list is None

    def test_preview_reload_sees_extra_skill_roots_toml_edit(
        self, tmp_path: Path
    ) -> None:
        """A preview and accepted reload use the same fresh skill roots."""
        from deepagents_code import model_config

        skills_dir = tmp_path / "external-skills"
        skills_dir.mkdir()
        config_path = model_config.DEFAULT_CONFIG_PATH
        settings = Settings.from_environment(start_path=tmp_path)
        assert settings.extra_skills_dirs is None

        config_path.write_text(
            f'[skills]\nextra_allowed_dirs = ["{skills_dir}"]\n',
            encoding="utf-8",
        )
        preview = settings.preview_reload_from_environment(start_path=tmp_path)
        applied = settings.reload_from_environment(start_path=tmp_path)

        assert any(change.startswith("extra_skills_dirs:") for change in preview)
        assert preview == applied
        assert settings.extra_skills_dirs == [skills_dir]

    def test_preview_reload_retains_shell_allow_list_on_corrupt_toml(
        self, tmp_path: Path
    ) -> None:
        """Preview and apply retain the last readable user snapshot."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[shell]\nallow_list = ["ls"]\n', encoding="utf-8")
        settings = Settings.from_environment(start_path=tmp_path)
        assert settings.shell_allow_list == ["ls"]

        config_path.write_text("[shell\n", encoding="utf-8")

        preview = settings.preview_reload_from_environment(start_path=tmp_path)
        applied = settings.reload_from_environment(start_path=tmp_path)

        assert not any(change.startswith("shell_allow_list:") for change in preview)
        assert not any(change.startswith("shell_allow_list:") for change in applied)
        assert settings.shell_allow_list == ["ls"]

    def test_reload_reports_an_unparseable_user_config(self, tmp_path: Path) -> None:
        """A corrupt `config.toml` must not be reported as a clean reload.

        Retaining the previous generation is the right runtime behavior, but
        the retention is otherwise silent: the only signal is a warning in the
        debug buffer, while the report the user reads says "Configuration
        reloaded. No changes detected." They edited the file a moment ago and
        would have no way to tell the edit was rejected.
        """
        from deepagents_code import model_config

        config_path = model_config.DEFAULT_CONFIG_PATH
        config_path.write_text('[shell]\nallow_list = ["ls"]\n', encoding="utf-8")
        settings = Settings.from_environment(start_path=tmp_path)
        assert settings.shell_allow_list == ["ls"]

        config_path.write_text("[shell\n", encoding="utf-8")
        changes = settings.reload_from_environment(start_path=tmp_path)

        assert changes
        assert changes[0].startswith("Kept previous config.toml:")
        # The retained value is still in force -- the notice reports the
        # rejection, it does not describe a rollback.
        assert settings.shell_allow_list == ["ls"]

    def test_preview_reports_an_unparseable_user_config(self, tmp_path: Path) -> None:
        """The preview reports its own read, so accept/decline agree with it."""
        from deepagents_code import model_config

        config_path = model_config.DEFAULT_CONFIG_PATH
        config_path.write_text('[shell]\nallow_list = ["ls"]\n', encoding="utf-8")
        settings = Settings.from_environment(start_path=tmp_path)

        config_path.write_text("[shell\n", encoding="utf-8")
        preview = settings.preview_reload_from_environment(start_path=tmp_path)

        assert preview
        assert preview[0].startswith("Kept previous config.toml:")
        assert settings.shell_allow_list == ["ls"]

    def test_reload_reports_no_notice_for_a_readable_config(
        self, tmp_path: Path
    ) -> None:
        """A healthy file reports changes only -- no spurious notice."""
        from deepagents_code import model_config

        config_path = model_config.DEFAULT_CONFIG_PATH
        config_path.write_text('[shell]\nallow_list = ["ls"]\n', encoding="utf-8")
        settings = Settings.from_environment(start_path=tmp_path)

        config_path.write_text(
            '[shell]\nallow_list = ["ls", "cat"]\n', encoding="utf-8"
        )
        changes = settings.reload_from_environment(start_path=tmp_path)

        assert not any(
            change.startswith("Kept previous config.toml:") for change in changes
        )
        assert settings.shell_allow_list == ["ls", "cat"]

    def test_reload_retains_extra_skill_roots_on_corrupt_toml(
        self, tmp_path: Path
    ) -> None:
        """A failed user-config reload cannot drop active skill containment roots."""
        skills_dir = tmp_path / "external-skills"
        skills_dir.mkdir()
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            f'[skills]\nextra_allowed_dirs = ["{skills_dir}"]\n',
            encoding="utf-8",
        )
        settings = Settings.from_environment(start_path=tmp_path)
        assert settings.extra_skills_dirs == [skills_dir]

        config_path.write_text("[skills\n", encoding="utf-8")

        changes = settings.reload_from_environment(start_path=tmp_path)

        assert not any(change.startswith("extra_skills_dirs:") for change in changes)
        assert settings.extra_skills_dirs == [skills_dir]

    def test_reload_reads_managed_policy_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """One reload must use one managed-policy file generation."""
        from deepagents_code.configuration import service

        settings = Settings.from_environment(start_path=tmp_path)
        original_load = service._load_managed
        loads = 0

        def counted_load(path: Path | None = None) -> TomlSnapshot:
            nonlocal loads
            loads += 1
            return original_load(path)

        monkeypatch.setattr(service, "_load_managed", counted_load)

        settings.reload_from_environment(start_path=tmp_path)

        assert loads == 1

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
        monkeypatch.setenv("DEEPAGENTS_CODE_SHELL_ALLOW_LIST", "ls,cat")
        settings = Settings.from_environment(start_path=tmp_path)
        assert settings.shell_allow_list == ["ls", "cat"]

        monkeypatch.setenv("DEEPAGENTS_CODE_SHELL_ALLOW_LIST", "ls,grep")
        changes = settings.reload_from_environment(start_path=tmp_path)

        assert settings.shell_allow_list == ["ls", "grep"]
        assert any(change.startswith("shell_allow_list:") for change in changes)

    def test_loads_project_dotenv_from_explicit_start_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reload should anchor dotenv loading to the explicit start path."""
        settings = Settings.from_environment(start_path=tmp_path)
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-test\n")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        settings.reload_from_environment(start_path=tmp_path)

        assert os.environ["OPENAI_API_KEY"] == "sk-test"

    def test_loads_global_dotenv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reload should load project dotenv first, then global."""
        settings = Settings.from_environment(start_path=tmp_path)

        global_env = tmp_path / "global" / ".env"
        global_env.parent.mkdir()
        global_env.write_text("OPENAI_API_KEY=sk-global\n")
        monkeypatch.setattr("deepagents_code.config._GLOBAL_DOTENV_PATH", global_env)

        project_env = tmp_path / ".env"
        project_env.write_text("ANTHROPIC_API_KEY=sk-project\n")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        settings.reload_from_environment(start_path=tmp_path)

        assert os.environ["ANTHROPIC_API_KEY"] == "sk-project"
        assert os.environ["OPENAI_API_KEY"] == "sk-global"

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
        monkeypatch.setattr("deepagents_code.config._GLOBAL_DOTENV_PATH", broken)

        # Should not raise — project .env still loads
        project_env = tmp_path / ".env"
        project_env.write_text("OPENAI_API_KEY=sk-fallback\n")

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with caplog.at_level(logging.WARNING, logger="deepagents_code.config"):
            settings.reload_from_environment(start_path=tmp_path)

        assert any("Could not read global dotenv" in r.message for r in caplog.records)
        assert os.environ["OPENAI_API_KEY"] == "sk-fallback"

    def test_project_dotenv_beats_global(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Project `.env` should always beat global `.env`."""
        from deepagents_code.config import _load_dotenv

        global_dir = tmp_path / "global"
        global_dir.mkdir()
        global_env = global_dir / ".env"
        global_env.write_text("TEST_PRECEDENCE_KEY=global-value\n")
        monkeypatch.setattr("deepagents_code.config._GLOBAL_DOTENV_PATH", global_env)

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
        from deepagents_code.config import _load_dotenv

        # No global dotenv
        monkeypatch.setattr(
            "deepagents_code.config._GLOBAL_DOTENV_PATH",
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
        from deepagents_code.config import _load_dotenv

        global_dir = tmp_path / "global"
        global_dir.mkdir()
        global_env = global_dir / ".env"
        global_env.write_text("TEST_BOOT_KEY=global-value\n")
        monkeypatch.setattr("deepagents_code.config._GLOBAL_DOTENV_PATH", global_env)

        # Simulate a shell-exported variable (e.g., from $ZDOTDIR/.env)
        monkeypatch.setenv("TEST_BOOT_KEY", "shell-value")

        monkeypatch.setattr(
            "dotenv.load_dotenv",
            _real_load_dotenv,
        )
        # No project .env
        monkeypatch.setattr(
            "deepagents_code.config._find_dotenv_from_start_path",
            lambda _: None,
        )

        _load_dotenv(start_path=tmp_path)

        assert os.environ.get("TEST_BOOT_KEY") == "shell-value"
        monkeypatch.delenv("TEST_BOOT_KEY", raising=False)

    def test_global_only_no_project_dotenv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Global `.env` values should apply when no project `.env` exists."""
        from deepagents_code.config import _load_dotenv

        global_dir = tmp_path / "global"
        global_dir.mkdir()
        global_env = global_dir / ".env"
        global_env.write_text("TEST_GLOBAL_ONLY=global-value\n")
        monkeypatch.setattr("deepagents_code.config._GLOBAL_DOTENV_PATH", global_env)

        monkeypatch.setattr(
            "dotenv.load_dotenv",
            _real_load_dotenv,
        )
        monkeypatch.delenv("TEST_GLOBAL_ONLY", raising=False)

        # No .env in isolated dir; global is the only source
        monkeypatch.setattr(
            "deepagents_code.config._find_dotenv_from_start_path",
            lambda _: None,
        )
        isolated = tmp_path / "no_project_env"
        isolated.mkdir()
        result = _load_dotenv(start_path=isolated)

        assert result is True
        assert os.environ.get("TEST_GLOBAL_ONLY") == "global-value"
        monkeypatch.delenv("TEST_GLOBAL_ONLY", raising=False)

    def test_global_dotenv_values_raises_oserror(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """OSError from `dotenv.dotenv_values` itself is caught."""
        settings = Settings.from_environment(start_path=tmp_path)

        global_env = tmp_path / "global" / ".env"
        global_env.parent.mkdir()
        global_env.write_text("KEY=val\n")
        monkeypatch.setattr("deepagents_code.config._GLOBAL_DOTENV_PATH", global_env)

        project_env = tmp_path / ".env"
        project_env.write_text("OPENAI_API_KEY=sk-ok\n")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        original_dotenv_values = _dotenv_module.dotenv_values
        global_calls = 0

        def _fail_on_global(*, dotenv_path: Path) -> dict[str, str | None]:
            nonlocal global_calls
            if dotenv_path == global_env:
                global_calls += 1
                msg = "read error"
                raise OSError(msg)
            return dict(original_dotenv_values(dotenv_path=dotenv_path))

        monkeypatch.setattr("dotenv.dotenv_values", _fail_on_global)

        with caplog.at_level(logging.WARNING, logger="deepagents_code.config"):
            settings.reload_from_environment(start_path=tmp_path)

        # The global file is read once for the trusted `read_project_dotenv`
        # pre-check and once for its remaining values; both hit the failure.
        assert global_calls == 2
        assert os.environ["OPENAI_API_KEY"] == "sk-ok"
        assert any("Could not read global dotenv" in r.message for r in caplog.records)

    def test_project_dotenv_denies_environment_hijack_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Project `.env` must not inject keys that alter subprocess startup."""
        from deepagents_code.config import _load_dotenv

        project_env = tmp_path / ".env"
        project_env.write_text(
            "BASH_ENV=/tmp/evil.sh\n"
            "BASHOPTS=expand_aliases\n"
            "CDPATH=/tmp\n"
            "COMSPEC=C:\\repo\\cmd.exe\n"
            "ENV=/tmp/evil.sh\n"
            "GIT_CONFIG_COUNT=1\n"
            "GIT_CONFIG_KEY_0=core.fsmonitor\n"
            "GIT_CONFIG_VALUE_0=/tmp/evil.sh\n"
            "GIT_CONFIG_PARAMETERS='core.pager=/tmp/evil.sh'\n"
            "GIT_CONFIG_GLOBAL=/tmp/evil.gitconfig\n"
            "GIT_CONFIG_SYSTEM=/tmp/evil.gitconfig\n"
            "GIT_DIR=/tmp/evil.git\n"
            "GIT_EDITOR=/tmp/evil.sh\n"
            "GIT_SSH_COMMAND=/tmp/evil.sh\n"
            "GIT_WORK_TREE=/tmp/evil\n"
            "GLOBIGNORE=*\n"
            "LD_PRELOAD=/tmp/evil.so\n"
            "PYTHONPATH=/tmp/evil\n"
            "PATH=/tmp/evil\n"
            "NODE_OPTIONS=--require /tmp/evil.js\n"
            "SHELLOPTS=xtrace\n"
            "SYSTEMROOT=C:\\repo\\windows\n"
            "WINDIR=C:\\repo\\windows\n"
            "DEEPAGENTS_INHERITED_PYTHONPATH=/tmp/evil\n"
            "OPENAI_API_KEY=sk-ok\n"
        )
        for key in (
            "BASH_ENV",
            "BASHOPTS",
            "CDPATH",
            "COMSPEC",
            "ENV",
            "GIT_CONFIG_COUNT",
            "GIT_CONFIG_KEY_0",
            "GIT_CONFIG_VALUE_0",
            "GIT_CONFIG_PARAMETERS",
            "GIT_CONFIG_GLOBAL",
            "GIT_CONFIG_SYSTEM",
            "GIT_DIR",
            "GIT_EDITOR",
            "GIT_SSH_COMMAND",
            "GIT_WORK_TREE",
            "GLOBIGNORE",
            "LD_PRELOAD",
            "PYTHONPATH",
            "NODE_OPTIONS",
            "SHELLOPTS",
            "SYSTEMROOT",
            "WINDIR",
            "DEEPAGENTS_INHERITED_PYTHONPATH",
            "OPENAI_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)

        _load_dotenv(start_path=tmp_path)

        assert "BASH_ENV" not in os.environ
        assert "BASHOPTS" not in os.environ
        assert "CDPATH" not in os.environ
        assert "COMSPEC" not in os.environ
        assert "ENV" not in os.environ
        assert "GIT_CONFIG_COUNT" not in os.environ
        assert "GIT_CONFIG_KEY_0" not in os.environ
        assert "GIT_CONFIG_VALUE_0" not in os.environ
        assert "GIT_CONFIG_PARAMETERS" not in os.environ
        assert "GIT_CONFIG_GLOBAL" not in os.environ
        assert "GIT_CONFIG_SYSTEM" not in os.environ
        assert "GIT_DIR" not in os.environ
        assert "GIT_EDITOR" not in os.environ
        assert "GIT_SSH_COMMAND" not in os.environ
        assert "GIT_WORK_TREE" not in os.environ
        assert "GLOBIGNORE" not in os.environ
        assert "LD_PRELOAD" not in os.environ
        assert "PYTHONPATH" not in os.environ
        assert "NODE_OPTIONS" not in os.environ
        assert "SHELLOPTS" not in os.environ
        assert "SYSTEMROOT" not in os.environ
        assert "WINDIR" not in os.environ
        # The carrier var must not be injectable from `.env`, or a project could
        # smuggle a PYTHONPATH into agent `execute` commands through it.
        assert "DEEPAGENTS_INHERITED_PYTHONPATH" not in os.environ
        assert os.environ["OPENAI_API_KEY"] == "sk-ok"

    def test_project_dotenv_denies_lowercase_git_config_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Denied keys are matched case-insensitively for Windows env semantics.

        On Windows `os.environ` keys are case-insensitive and Python normalizes
        assigned keys to uppercase, so a lowercase `git_config_key_0` in a
        committed `.env` would otherwise pass a case-sensitive check and become
        an active `GIT_CONFIG_KEY_0` for the `git` commands dcode runs during
        startup detection. On POSIX the lowercase spelling is inert (git reads
        only the canonical case), so denying it there is harmless.
        """
        from deepagents_code.config import _load_dotenv

        project_env = tmp_path / ".env"
        project_env.write_text(
            "git_config_count=1\n"
            "git_config_key_0=core.fsmonitor\n"
            "Git_Config_Value_0=/tmp/evil.sh\n"
            "git_dir=/tmp/evil.git\n"
            "OPENAI_API_KEY=sk-ok\n"
        )
        # On POSIX the lowercase names are distinct env vars; ensure neither the
        # lowercase spelling nor its uppercase normalization is already set.
        for key in (
            "git_config_count",
            "git_config_key_0",
            "Git_Config_Value_0",
            "git_dir",
            "GIT_CONFIG_COUNT",
            "GIT_CONFIG_KEY_0",
            "GIT_CONFIG_VALUE_0",
            "GIT_DIR",
            "OPENAI_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)

        _load_dotenv(start_path=tmp_path)

        for key in (
            "git_config_count",
            "git_config_key_0",
            "Git_Config_Value_0",
            "git_dir",
            "GIT_CONFIG_COUNT",
            "GIT_CONFIG_KEY_0",
            "GIT_CONFIG_VALUE_0",
            "GIT_DIR",
        ):
            assert key not in os.environ
        assert os.environ["OPENAI_API_KEY"] == "sk-ok"

    def test_project_dotenv_skipped_when_read_project_dotenv_false(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """`startup.read_project_dotenv = false` skips the project `.env`.

        The project file must not apply its values, while the global
        `~/.deepagents/.env` still loads — disabling is scoped to the untrusted,
        repo-traveling file, not the user's own global defaults.
        """
        from deepagents_code.config import _load_dotenv

        project_env = tmp_path / ".env"
        project_env.write_text("GIT_CONFIG_COUNT=1\nPROJECT_ONLY_KEY=project-value\n")
        global_dir = tmp_path / "global"
        global_dir.mkdir()
        global_env = global_dir / ".env"
        global_env.write_text("GLOBAL_ONLY_KEY=global-value\n")
        monkeypatch.setattr("deepagents_code.config._GLOBAL_DOTENV_PATH", global_env)
        for key in ("GIT_CONFIG_COUNT", "PROJECT_ONLY_KEY", "GLOBAL_ONLY_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("DEEPAGENTS_CODE_READ_PROJECT_DOTENV", raising=False)

        monkeypatch.setattr(
            "deepagents_code.config_manifest.resolve_read_project_dotenv",
            lambda **_kw: False,
        )

        _load_dotenv(start_path=tmp_path)

        assert "GIT_CONFIG_COUNT" not in os.environ
        assert "PROJECT_ONLY_KEY" not in os.environ
        assert os.environ["GLOBAL_ONLY_KEY"] == "global-value"

    def test_project_dotenv_loads_when_read_project_dotenv_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Default (`startup.read_project_dotenv` true) still loads the project file."""
        from deepagents_code.config import _load_dotenv

        project_env = tmp_path / ".env"
        project_env.write_text("PROJECT_ONLY_KEY=project-value\n")
        monkeypatch.setattr(
            "deepagents_code.config._GLOBAL_DOTENV_PATH",
            tmp_path / "nonexistent" / ".env",
        )
        monkeypatch.delenv("PROJECT_ONLY_KEY", raising=False)

        monkeypatch.setattr(
            "deepagents_code.config_manifest.resolve_read_project_dotenv",
            lambda **_kw: True,
        )

        _load_dotenv(start_path=tmp_path)

        assert os.environ["PROJECT_ONLY_KEY"] == "project-value"

    def test_project_dotenv_cannot_set_read_project_dotenv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A project `.env` cannot inject the toggle that skips it.

        `DEEPAGENTS_CODE_READ_PROJECT_DOTENV` is denied from every `.env` (the
        `_DOTENV_DENIED_ENV_KEYS` set), so a hostile project file cannot pin it
        true and block the trusted global file from opting out via
        first-write-wins.
        """
        from deepagents_code.config import _load_dotenv

        project_env = tmp_path / ".env"
        project_env.write_text(
            "DEEPAGENTS_CODE_READ_PROJECT_DOTENV=1\nPROJECT_ONLY_KEY=project-value\n"
        )
        monkeypatch.setattr(
            "deepagents_code.config._GLOBAL_DOTENV_PATH",
            tmp_path / "nonexistent" / ".env",
        )
        monkeypatch.delenv("DEEPAGENTS_CODE_READ_PROJECT_DOTENV", raising=False)
        monkeypatch.delenv("PROJECT_ONLY_KEY", raising=False)

        _load_dotenv(start_path=tmp_path)

        assert "DEEPAGENTS_CODE_READ_PROJECT_DOTENV" not in os.environ
        assert os.environ["PROJECT_ONLY_KEY"] == "project-value"

    def test_global_dotenv_read_project_dotenv_false_protects_startup(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The trusted global `.env` opt-out is honored for the current startup.

        The toggle is read from the global file *before* the project file is
        touched, so `DEEPAGENTS_CODE_READ_PROJECT_DOTENV=false` in
        `~/.deepagents/.env` skips the untrusted project `.env` even though the
        global file is otherwise loaded after it.
        """
        from deepagents_code.config import _load_dotenv

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / ".env").write_text("PROJECT_ONLY_KEY=project-value\n")
        global_dir = tmp_path / "global"
        global_dir.mkdir()
        (global_dir / ".env").write_text(
            "DEEPAGENTS_CODE_READ_PROJECT_DOTENV=false\nGLOBAL_ONLY_KEY=global-value\n"
        )
        monkeypatch.setattr(
            "deepagents_code.config._GLOBAL_DOTENV_PATH", global_dir / ".env"
        )
        for key in (
            "DEEPAGENTS_CODE_READ_PROJECT_DOTENV",
            "PROJECT_ONLY_KEY",
            "GLOBAL_ONLY_KEY",
        ):
            monkeypatch.delenv(key, raising=False)

        _load_dotenv(start_path=project_dir)

        assert "PROJECT_ONLY_KEY" not in os.environ
        # The toggle's env var is denied from every `.env`, so the global file's
        # own copy is consumed for the decision but not injected into os.environ.
        assert "DEEPAGENTS_CODE_READ_PROJECT_DOTENV" not in os.environ
        # The global file's other values still load.
        assert os.environ["GLOBAL_ONLY_KEY"] == "global-value"

    def test_preview_dotenv_skipped_when_read_project_dotenv_false(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Preview mirrors the loader: a disabled project `.env` is not reported.

        The preview drives the user-facing cwd-switch prompt
        (`_preview_project_settings_change`); if it still read the project file
        while the runtime loader skipped it, the app would warn about settings
        changes that a real reload would never apply.
        """
        from deepagents_code.config import _preview_dotenv_environ

        (tmp_path / ".env").write_text("PROJECT_ONLY_KEY=project-value\n")
        monkeypatch.setattr(
            "deepagents_code.config._GLOBAL_DOTENV_PATH",
            tmp_path / "nonexistent" / ".env",
        )
        monkeypatch.delenv("PROJECT_ONLY_KEY", raising=False)
        monkeypatch.setattr(
            "deepagents_code.config_manifest.resolve_read_project_dotenv",
            lambda **_kw: False,
        )

        env = _preview_dotenv_environ(start_path=tmp_path)

        assert "PROJECT_ONLY_KEY" not in env

    def test_project_dotenv_cannot_set_mcp_trust_lists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A committed project `.env` must not self-approve project MCP servers.

        The MCP trust-list env vars are a user-level decision; honoring them from
        a repo-committed `.env` would let an attacker pair a malicious `.mcp.json`
        with a `.env` and pre-approve their own servers, defeating the whole
        point of the trust gate. Ordinary project vars are still loaded.
        """
        from deepagents_code.config import _load_dotenv

        project_env = tmp_path / ".env"
        project_env.write_text(
            f"{_env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS}=exfil\n"
            "DEEPAGENTS_CODE_DISABLED_PROJECT_MCP_SERVERS=\n"
            "OPENAI_API_KEY=sk-ok\n"
        )
        for key in (
            _env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS,
            "DEEPAGENTS_CODE_DISABLED_PROJECT_MCP_SERVERS",
            "OPENAI_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)

        _load_dotenv(start_path=tmp_path)

        assert _env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS not in os.environ
        assert "DEEPAGENTS_CODE_DISABLED_PROJECT_MCP_SERVERS" not in os.environ
        # A normal project var is unaffected — only the trust-list keys are gated.
        assert os.environ["OPENAI_API_KEY"] == "sk-ok"

    def test_global_dotenv_can_set_mcp_trust_lists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The global `~/.deepagents/.env` (is_project=False) MAY set trust lists.

        Positive counterpart to `test_project_dotenv_cannot_set_mcp_trust_lists`:
        the deny is scoped to the *project* `.env`. This pins the allow half so a
        regression that gates these keys unconditionally (e.g. dropping the
        `is_project` qualifier, or moving them into `_DOTENV_DENIED_ENV_KEYS`)
        would fail here rather than silently breaking the user's own global
        pre-approval path.
        """
        from deepagents_code.config import _load_dotenv

        global_dir = tmp_path / "global"
        global_dir.mkdir()
        global_env = global_dir / ".env"
        global_env.write_text(
            f"{_env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS}=docs\n"
            "DEEPAGENTS_CODE_DISABLED_PROJECT_MCP_SERVERS=blocked\n"
        )
        monkeypatch.setattr("deepagents_code.config._GLOBAL_DOTENV_PATH", global_env)
        # No project `.env`, so the global file is the only source.
        monkeypatch.setattr(
            "deepagents_code.config._find_dotenv_from_start_path",
            lambda _: None,
        )
        for key in (
            _env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS,
            "DEEPAGENTS_CODE_DISABLED_PROJECT_MCP_SERVERS",
        ):
            monkeypatch.delenv(key, raising=False)

        isolated = tmp_path / "no_project_env"
        isolated.mkdir()
        _load_dotenv(start_path=isolated)

        assert (
            os.environ.get(_env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS) == "docs"
        )
        assert (
            os.environ.get("DEEPAGENTS_CODE_DISABLED_PROJECT_MCP_SERVERS") == "blocked"
        )

    def test_project_dotenv_cannot_set_auto_classifier_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A committed project `.env` must not choose the Auto classifier model.

        The classifier authorizes gated tool calls in Auto mode, so a cloned repo
        that could set this would silently downgrade the review — including its
        resistance to prompt injection in the untrusted text it reads. Picking
        the classifier stays a user-level decision (THREAT_MODEL T14).
        """
        from deepagents_code.config import _load_dotenv

        project_env = tmp_path / ".env"
        project_env.write_text(
            f"{_env_vars.AUTO_CLASSIFIER_MODEL}=openai:weak-model\n"
            "OPENAI_API_KEY=sk-ok\n"
        )
        for key in (_env_vars.AUTO_CLASSIFIER_MODEL, "OPENAI_API_KEY"):
            monkeypatch.delenv(key, raising=False)

        _load_dotenv(start_path=tmp_path)

        assert _env_vars.AUTO_CLASSIFIER_MODEL not in os.environ
        # A normal project var is unaffected — only the classifier key is gated.
        assert os.environ["OPENAI_API_KEY"] == "sk-ok"

    def test_global_dotenv_can_set_auto_classifier_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The global `~/.deepagents/.env` MAY choose the Auto classifier model.

        Positive counterpart to
        `test_project_dotenv_cannot_set_auto_classifier_model`: the deny is
        scoped to the *project* `.env`, so a regression that gates the key
        unconditionally would break the user's own configuration path.
        """
        from deepagents_code.config import _load_dotenv

        global_dir = tmp_path / "global"
        global_dir.mkdir()
        global_env = global_dir / ".env"
        global_env.write_text(
            f"{_env_vars.AUTO_CLASSIFIER_MODEL}=anthropic:claude-haiku-4-5\n"
        )
        monkeypatch.setattr("deepagents_code.config._GLOBAL_DOTENV_PATH", global_env)
        monkeypatch.setattr(
            "deepagents_code.config._find_dotenv_from_start_path",
            lambda _: None,
        )
        monkeypatch.delenv(_env_vars.AUTO_CLASSIFIER_MODEL, raising=False)

        isolated = tmp_path / "no_project_env"
        isolated.mkdir()
        _load_dotenv(start_path=isolated)

        assert (
            os.environ.get(_env_vars.AUTO_CLASSIFIER_MODEL)
            == "anthropic:claude-haiku-4-5"
        )

    def test_preview_project_dotenv_cannot_set_mcp_trust_lists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Preview mirrors `_load_dotenv`: a project `.env` can't set trust lists.

        The same `is_project` guard was added to both `_load_dotenv` and
        `_preview_dotenv_environ`; keep their coverage parallel so the two copies
        cannot drift.
        """
        from deepagents_code.config import _preview_dotenv_environ

        project_env = tmp_path / ".env"
        project_env.write_text(
            f"{_env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS}=exfil\n"
            "OPENAI_API_KEY=sk-ok\n"
        )
        monkeypatch.delenv(
            _env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS,
            raising=False,
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        env = _preview_dotenv_environ(start_path=tmp_path)

        assert _env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS not in env
        assert env["OPENAI_API_KEY"] == "sk-ok"

    def test_preview_global_dotenv_can_set_mcp_trust_lists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Preview allows the global `.env` (is_project=False) to set trust lists."""
        from deepagents_code.config import _preview_dotenv_environ

        global_env = tmp_path / "global" / ".env"
        global_env.parent.mkdir()
        global_env.write_text("DEEPAGENTS_CODE_DISABLED_PROJECT_MCP_SERVERS=blocked\n")
        monkeypatch.setattr("deepagents_code.config._GLOBAL_DOTENV_PATH", global_env)
        # No project `.env` to find, so only the global file contributes.
        monkeypatch.setattr(
            "deepagents_code.config._find_dotenv_from_start_path",
            lambda _: None,
        )
        monkeypatch.delenv(
            "DEEPAGENTS_CODE_DISABLED_PROJECT_MCP_SERVERS", raising=False
        )

        env = _preview_dotenv_environ(start_path=tmp_path)

        assert env["DEEPAGENTS_CODE_DISABLED_PROJECT_MCP_SERVERS"] == "blocked"

    def test_multiple_simultaneous_changes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reload should accumulate changes across multiple fields."""
        settings = Settings.from_environment(start_path=tmp_path)

        monkeypatch.setenv("OPENAI_API_KEY", "sk-new")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        monkeypatch.setenv("DEEPAGENTS_CODE_SHELL_ALLOW_LIST", "ls")
        changes = settings.reload_from_environment(start_path=tmp_path)

        assert len(changes) == 3
        fields = {c.split(":")[0] for c in changes}
        assert fields == {"openai_api_key", "anthropic_api_key", "shell_allow_list"}

    def test_prefixed_env_var_beats_canonical(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """DEEPAGENTS_CODE_ prefixed var should override canonical on reload."""
        settings = Settings.from_environment(start_path=tmp_path)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-canonical")
        monkeypatch.setenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", "sk-override")
        settings.reload_from_environment(start_path=tmp_path)

        assert settings.anthropic_api_key == "sk-override"

    def test_from_environment_uses_prefixed_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Settings.from_environment should honour the DEEPAGENTS_CODE_ prefix."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-canonical")
        monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "sk-override")

        settings = Settings.from_environment(start_path=tmp_path)

        assert settings.openai_api_key == "sk-override"

    def test_google_cloud_location_uses_prefixed_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Google Cloud location follows the standard prefixed-env precedence."""
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        monkeypatch.setenv("DEEPAGENTS_CODE_GOOGLE_CLOUD_LOCATION", "us-east5")

        settings = Settings.from_environment(start_path=tmp_path)

        assert settings.google_cloud_location == "us-east5"

    def test_preview_dotenv_shell_beats_project(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Preview env mirrors `_load_dotenv`: a shell var beats a project `.env`."""
        from deepagents_code.config import _preview_dotenv_environ

        monkeypatch.setattr(
            "deepagents_code.config._GLOBAL_DOTENV_PATH",
            tmp_path / "nonexistent" / ".env",
        )
        (tmp_path / ".env").write_text("TEST_PREVIEW_KEY=project-value\n")
        monkeypatch.setenv("TEST_PREVIEW_KEY", "shell-value")

        env = _preview_dotenv_environ(start_path=tmp_path)

        assert env["TEST_PREVIEW_KEY"] == "shell-value"

    def test_preview_dotenv_project_beats_global(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Preview env mirrors `_load_dotenv`: project `.env` beats global `.env`."""
        from deepagents_code.config import _preview_dotenv_environ

        global_dir = tmp_path / "global"
        global_dir.mkdir()
        global_env = global_dir / ".env"
        global_env.write_text("TEST_PREVIEW_KEY2=global-value\n")
        monkeypatch.setattr("deepagents_code.config._GLOBAL_DOTENV_PATH", global_env)
        (tmp_path / ".env").write_text("TEST_PREVIEW_KEY2=project-value\n")
        monkeypatch.delenv("TEST_PREVIEW_KEY2", raising=False)

        env = _preview_dotenv_environ(start_path=tmp_path)

        assert env["TEST_PREVIEW_KEY2"] == "project-value"

    def test_preview_dotenv_denies_environment_hijack_keys(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Preview env mirrors `_load_dotenv`: denied keys are omitted.

        Exercises the full shell-startup-hook set so the preview path stays
        visibly parallel to the mutating path, and asserts the debug breadcrumb
        names the denied key (and only the key, never its value).
        """
        from deepagents_code.config import _preview_dotenv_environ

        denied_keys = (
            "BASH_ENV",
            "BASHOPTS",
            "CDPATH",
            "COMSPEC",
            "ENV",
            "GIT_CONFIG_COUNT",
            "GIT_CONFIG_KEY_0",
            "GIT_CONFIG_VALUE_0",
            "GIT_CONFIG_PARAMETERS",
            "GIT_DIR",
            "GIT_EDITOR",
            "GIT_SSH_COMMAND",
            "GLOBIGNORE",
            "SHELLOPTS",
        )
        evil_value = "/tmp/evil.sh"  # test fixture value, never read

        monkeypatch.setattr(
            "deepagents_code.config._GLOBAL_DOTENV_PATH",
            tmp_path / "nonexistent" / ".env",
        )
        dotenv_lines = [f"{key}={evil_value}\n" for key in denied_keys]
        dotenv_lines.append("OPENAI_API_KEY=sk-ok\n")
        (tmp_path / ".env").write_text("".join(dotenv_lines))
        for key in (*denied_keys, "OPENAI_API_KEY"):
            monkeypatch.delenv(key, raising=False)

        with caplog.at_level(logging.DEBUG, logger="deepagents_code.config"):
            env = _preview_dotenv_environ(start_path=tmp_path)

        for key in denied_keys:
            assert key not in env
        assert env["OPENAI_API_KEY"] == "sk-ok"

        # The breadcrumb names each denied key but never leaks the value.
        for key in denied_keys:
            assert any(key in record.getMessage() for record in caplog.records)
        assert evil_value not in caplog.text

    def test_preview_reports_api_key_masked_without_mutating(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Previewing an API-key change reports it masked and mutates nothing."""
        settings = Settings.from_environment(start_path=tmp_path)
        assert settings.openai_api_key is None

        monkeypatch.setenv("OPENAI_API_KEY", "sk-preview-secret")
        changes = settings.preview_reload_from_environment(start_path=tmp_path)

        assert "openai_api_key: unset -> set" in changes
        assert "sk-preview-secret" not in "\n".join(changes)
        assert settings.openai_api_key is None


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
            "deepagents_code.config._GLOBAL_DOTENV_PATH",
            tmp_path / "nonexistent" / ".env",
        )

    def test_invalid_shell_allow_list_keeps_previous(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Malformed shell allow-list should fall back to previous value."""
        monkeypatch.setenv("DEEPAGENTS_CODE_SHELL_ALLOW_LIST", "ls,cat")
        settings = Settings.from_environment(start_path=tmp_path)
        assert settings.shell_allow_list == ["ls", "cat"]

        monkeypatch.setenv("DEEPAGENTS_CODE_SHELL_ALLOW_LIST", "all,ls")
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
            "deepagents_code.project_utils.find_project_root", _raise_oserror
        )
        changes = settings.reload_from_environment(start_path=tmp_path)

        assert settings.project_root == original_root
        assert not any(change.startswith("project_root:") for change in changes)

    def test_settings_consistent_after_partial_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Settings should remain consistent when one field fails to reload."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-original")
        monkeypatch.setenv("DEEPAGENTS_CODE_SHELL_ALLOW_LIST", "ls")
        settings = Settings.from_environment(start_path=tmp_path)

        # Change API key (succeeds) + break shell allow-list (falls back)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-updated")
        monkeypatch.setenv("DEEPAGENTS_CODE_SHELL_ALLOW_LIST", "all,ls")
        changes = settings.reload_from_environment(start_path=tmp_path)

        assert settings.openai_api_key == "sk-updated"
        assert settings.shell_allow_list == ["ls"]
        assert any(c.startswith("openai_api_key:") for c in changes)

    def test_invalid_extra_skills_dirs_keeps_previous(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A failure resolving extra skills dirs falls back to the previous value.

        Guards the cwd-switch path: `reload_from_environment` runs after
        `os.chdir`, so an unhandled resolution error would strand the process in
        a half-applied cwd.
        """
        import deepagents_code.config as config_mod

        settings = Settings.from_environment(start_path=tmp_path)
        sentinel = [tmp_path / "skills"]
        settings.extra_skills_dirs = sentinel
        monkeypatch.setenv("DEEPAGENTS_CODE_EXTRA_SKILLS_DIRS", str(sentinel[0]))

        def boom(*_args: object, **_kwargs: object) -> list[Path] | None:
            msg = "broken symlink loop"
            raise OSError(msg)

        monkeypatch.setattr(config_mod, "_parse_extra_skills_dirs", boom)
        changes = settings.reload_from_environment(start_path=tmp_path)

        assert settings.extra_skills_dirs == sentinel
        assert not any(change.startswith("extra_skills_dirs:") for change in changes)


class TestReloadableFieldConstants:
    """Guards for the derived reloadable-field constants."""

    def test_api_key_fields_derived_from_reloadable(self) -> None:
        """`_API_KEY_FIELDS` is the `*_api_key` subset of `_RELOADABLE_FIELDS`."""
        from deepagents_code.config import _API_KEY_FIELDS, _RELOADABLE_FIELDS

        assert {
            "openai_api_key",
            "anthropic_api_key",
            "google_api_key",
            "nvidia_api_key",
            "tavily_api_key",
        } == _API_KEY_FIELDS
        assert set(_RELOADABLE_FIELDS) >= _API_KEY_FIELDS


class TestReloadInAutocomplete:
    """Tests for autocomplete slash command registration."""

    def test_reload_in_slash_commands(self) -> None:
        """`/reload` should be registered in slash command completions."""
        assert any(entry.name == "/reload" for entry in get_slash_commands())


class TestReloadInputResponsiveness:
    """`/reload` should not block the Textual message pump."""

    @pytest.mark.timeout(15)
    async def test_keeps_chat_input_responsive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Typing should render while the detached reload task is still running."""
        from deepagents_code.app import DeepAgentsApp

        app = DeepAgentsApp(agent=MagicMock())
        async with app.run_test() as pilot:
            await pilot.pause()
            chat_input = app._chat_input
            assert chat_input is not None
            chat_input.focus_input()
            await pilot.pause()

            started = asyncio.Event()
            release = asyncio.Event()

            async def _blocked_reload() -> None:
                started.set()
                await release.wait()

            monkeypatch.setattr(app, "_run_reload", _blocked_reload)

            chat_input.mode = "command"
            chat_input._submit_value("/reload")
            await started.wait()

            await pilot.press("h", "i")
            await pilot.pause()
            typed = chat_input.value

            release.set()
            await pilot.pause()

            assert typed == "hi"

    @pytest.mark.timeout(15)
    async def test_queues_prompt_for_entire_reload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A prompt submitted before restart must wait for reload completion."""
        from deepagents_code.app import DeepAgentsApp

        app = DeepAgentsApp(agent=MagicMock())
        async with app.run_test() as pilot:
            await pilot.pause()
            started = asyncio.Event()
            release = asyncio.Event()

            async def _blocked_reload() -> None:
                started.set()
                await release.wait()

            monkeypatch.setattr(app, "_run_reload", _blocked_reload)

            task = app._schedule_reload()
            await started.wait()
            await app._submit_input("do not interrupt", "normal")

            assert app._reloading is True
            assert [message.text for message in app._pending_messages] == [
                "do not interrupt"
            ]

            release.set()
            await task

    @pytest.mark.timeout(15)
    async def test_coalesces_overlapping_reloads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second `/reload` shares the first task rather than racing it."""
        from deepagents_code.app import DeepAgentsApp

        app = DeepAgentsApp(agent=MagicMock())
        async with app.run_test() as pilot:
            await pilot.pause()
            started = asyncio.Event()
            release = asyncio.Event()
            runs = 0

            async def _blocked_reload() -> None:
                nonlocal runs
                runs += 1
                started.set()
                await release.wait()

            monkeypatch.setattr(app, "_run_reload", _blocked_reload)

            first = app._schedule_reload()
            await started.wait()
            second = app._schedule_reload()

            assert second is first
            assert runs == 1

            release.set()
            await first
            assert app._reloading is False

    @pytest.mark.parametrize("restarted", [True, False])
    @pytest.mark.timeout(15)
    async def test_runs_requested_restart_when_reload_skips_respawn(
        self, monkeypatch: pytest.MonkeyPatch, *, restarted: bool
    ) -> None:
        """A skipped reload respawn preserves `/restart` and queued prompts."""
        from deepagents_code.app import AppMessage, DeepAgentsApp
        from deepagents_code.config import settings

        app = DeepAgentsApp(agent=MagicMock())
        async with app.run_test() as pilot:
            await pilot.pause()
            app._server_proc = MagicMock()
            app._server_kwargs = {}
            started = asyncio.Event()
            release = asyncio.Event()

            async def _blocked_reload() -> None:
                started.set()
                await release.wait()

            restart = AsyncMock(return_value=restarted)
            monkeypatch.setattr(app, "_run_reload", _blocked_reload)
            monkeypatch.setattr(app, "_restart_server_manual", restart)
            monkeypatch.setattr(settings, "reload_from_environment", list)
            monkeypatch.setattr(
                "deepagents_code.model_config.clear_caches", lambda: None
            )

            task = app._schedule_reload()
            await started.wait()
            await app._handle_command("/restart")
            await app._submit_input("keep this prompt", "normal")

            restart.assert_not_awaited()
            assert any(
                "Reload already in progress" in str(message._content)
                for message in app.query(AppMessage)
            )

            release.set()
            await task
            assert app._restart_respawn_task is not None
            await app._restart_respawn_task

            restart.assert_awaited_once()
            assert [message.text for message in app._pending_messages] == [
                "keep this prompt"
            ]

    @pytest.mark.parametrize("restart_raises", [False, True])
    @pytest.mark.timeout(15)
    async def test_reload_respawn_consumes_requested_restart(
        self, monkeypatch: pytest.MonkeyPatch, *, restart_raises: bool
    ) -> None:
        """A `/restart` requested during reload's respawn does not run twice."""
        from deepagents_code.app import DeepAgentsApp, UserMessage, _ServerRespawnResult
        from deepagents_code.config import settings
        from deepagents_code.plugins.models import PluginDiscoveryResult

        app = DeepAgentsApp(agent=MagicMock())
        async with app.run_test() as pilot:
            await pilot.pause()
            app._server_proc = MagicMock()
            app._server_kwargs = {}
            started = asyncio.Event()
            release = asyncio.Event()
            echo_started = asyncio.Event()
            release_echo = asyncio.Event()

            async def _fake_discover() -> bool:  # noqa: RUF029
                return True

            async def _blocked_restart() -> _ServerRespawnResult:
                started.set()
                await release.wait()
                if restart_raises:
                    msg = "respawn exploded"
                    raise RuntimeError(msg)
                return _ServerRespawnResult(restarted=True)

            mount_message = app._mount_message

            async def _blocked_command_echo(widget: UserMessage) -> bool:
                if isinstance(widget, UserMessage):
                    echo_started.set()
                    await release_echo.wait()
                return await mount_message(widget)

            restart = AsyncMock(side_effect=_blocked_restart)
            monkeypatch.setattr(app, "_mount_message", _blocked_command_echo)
            monkeypatch.setattr(app, "_discover_skills", _fake_discover)
            monkeypatch.setattr(app, "_reload_hooks", AsyncMock())
            monkeypatch.setattr(app, "_restart_server_manual_result", restart)
            monkeypatch.setattr(settings, "reload_from_environment", list)
            monkeypatch.setattr(
                "deepagents_code.model_config.clear_caches", lambda: None
            )
            monkeypatch.setattr(
                "deepagents_code.plugins.discover_plugins",
                lambda: PluginDiscoveryResult(plugins=()),
            )
            monkeypatch.setattr(
                "deepagents_code.plugins.adapters.mcp.plugin_mcp_configs",
                lambda _plugins: (),
            )

            task = app._schedule_reload()
            await started.wait()
            command_task = asyncio.create_task(app._handle_command("/restart"))
            await echo_started.wait()

            assert restart.await_count == 1
            release.set()
            await task
            release_echo.set()
            await command_task

            restart.assert_awaited_once()
            assert app._restart_respawn_task is None


class TestReloadModelProfileHints:
    """`/reload` should refresh profile-derived command hints."""

    async def test_refreshes_status_without_owned_server(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Client-only sessions should resync after profile caches are cleared."""
        from deepagents_code import model_config
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.config import settings
        from deepagents_code.plugins.models import PluginDiscoveryResult

        config_path = tmp_path / "config.toml"

        def write_config(level: str) -> None:
            config_path.write_text(f"""
[models.providers.acme]
models = ["foo"]
[models.providers.acme.profile]
reasoning_output = true
reasoning_effort_levels = ["{level}"]
""")

        write_config("old")
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)
        monkeypatch.setattr(model_config, "_get_provider_profile_modules", list)
        monkeypatch.setattr(settings, "model_provider", "acme")
        monkeypatch.setattr(settings, "model_name", "foo")
        model_config.clear_caches()

        app = DeepAgentsApp()
        try:
            async with app.run_test() as pilot:
                await pilot.pause()

                async def _fake_discover() -> bool:  # noqa: RUF029
                    return True

                monkeypatch.setattr(app, "_discover_skills", _fake_discover)
                monkeypatch.setattr(
                    "deepagents_code.plugins.discover_plugins",
                    lambda: PluginDiscoveryResult(plugins=()),
                )
                monkeypatch.setattr(
                    "deepagents_code.plugins.adapters.mcp.plugin_mcp_configs",
                    lambda _plugins: (),
                )
                assert app._server_proc is None
                assert app._chat_input is not None
                assert app._chat_input._argument_hint_overrides["effort"] == (
                    "[old|clear]"
                )

                write_config("new")
                await app._handle_command("/reload")
                if app._reload_task is not None:
                    await app._reload_task

                assert app._chat_input._argument_hint_overrides["effort"] == (
                    "[new|clear]"
                )
        finally:
            model_config.clear_caches()


class TestReloadSkillReport:
    """`/reload` should surface skill add/remove diff in its report."""

    @staticmethod
    def _fake_skill(name: str) -> ExtendedSkillMetadata:
        return ExtendedSkillMetadata(
            name=name,
            description=f"{name} desc",
            path=f"/skills/{name}/SKILL.md",
            license=None,
            compatibility=None,
            metadata={},
            allowed_tools=[],
            source="user",
        )

    async def _run_reload(
        self,
        monkeypatch: pytest.MonkeyPatch,
        before: list[str],
        after: list[str] | None,
        *,
        discovery_ok: bool = True,
    ) -> str:
        """Drive `/reload` once and return the mounted `AppMessage` text.

        Args:
            monkeypatch: pytest fixture for restorable patching.
            before: skill names cached before reload.
            after: skill names produced by discovery, or ignored when
                `discovery_ok=False`.
            discovery_ok: when `False`, simulate discovery failure
                (preserves cache and returns `False`).
        """
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.tui.widgets.messages import AppMessage

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            app._discovered_skills = [self._fake_skill(n) for n in before]

            async def _fake_discover() -> bool:  # noqa: RUF029  # awaited as coroutine by `_handle_command`
                if not discovery_ok:
                    return False
                assert after is not None
                app._discovered_skills = [self._fake_skill(n) for n in after]
                return True

            monkeypatch.setattr(app, "_discover_skills", _fake_discover)

            await app._handle_command("/reload")
            if app._reload_task is not None:
                await app._reload_task
            await pilot.pause()

            return "\n".join(str(w._content) for w in app.query(AppMessage))

    async def test_reports_added_skills(self, monkeypatch: pytest.MonkeyPatch) -> None:
        text = await self._run_reload(
            monkeypatch, before=["alpha"], after=["alpha", "beta"]
        )
        assert "Skills updated" in text
        assert "  - Added: beta" in text
        assert "Removed:" not in text

    async def test_reports_removed_skills(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text = await self._run_reload(
            monkeypatch, before=["alpha", "beta"], after=["alpha"]
        )
        assert "Skills updated" in text
        assert "  - Removed: beta" in text
        assert "Added:" not in text

    async def test_reports_added_and_removed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text = await self._run_reload(
            monkeypatch, before=["alpha", "beta"], after=["alpha", "gamma"]
        )
        assert "Skills updated" in text
        assert "  - Added: gamma" in text
        assert "  - Removed: beta" in text

    async def test_reports_no_changes_stays_silent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the skill set is unchanged, the report should not mention skills."""
        text = await self._run_reload(monkeypatch, before=["alpha"], after=["alpha"])
        assert "Skills updated" not in text
        assert "Added:" not in text
        assert "Removed:" not in text
        assert "Skill re-discovery failed" not in text

    async def test_first_skill_added_from_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """User installs first skill, then `/reload` — empty -> non-empty."""
        text = await self._run_reload(monkeypatch, before=[], after=["alpha"])
        assert "  - Added: alpha" in text
        assert "Removed:" not in text

    async def test_all_skills_removed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All known skills removed — non-empty -> empty."""
        text = await self._run_reload(monkeypatch, before=["alpha", "beta"], after=[])
        assert "  - Removed: alpha, beta" in text
        assert "Added:" not in text

    async def test_added_skills_sorted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Added skill names should be sorted (deterministic output)."""
        text = await self._run_reload(
            monkeypatch, before=["alpha"], after=["alpha", "zeta", "beta"]
        )
        assert "  - Added: beta, zeta" in text

    async def test_discovery_failure_preserves_cache_and_warns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Discovery failure must not produce a misleading 'Removed: X' diff."""
        text = await self._run_reload(
            monkeypatch,
            before=["alpha", "beta"],
            after=None,
            discovery_ok=False,
        )
        assert "Skill re-discovery failed" in text
        # Critical: must not claim every prior skill was removed.
        assert "Removed:" not in text
        assert "Skills updated" not in text


class TestReloadThemeReapply:
    """`/reload` should re-apply the resolved theme preference.

    Guards the cross-session behavior: saving a per-terminal (or global)
    default theme in one window should be picked up by an already-running
    session's `/reload`, matching startup resolution.
    """

    async def _run_reload_theme(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        initial_theme: str,
        resolved_theme: str,
    ) -> tuple[str, str]:
        """Drive `/reload` once with a stubbed preference resolver.

        Args:
            monkeypatch: pytest fixture for restorable patching.
            initial_theme: theme active before reload.
            resolved_theme: value `_load_theme_preference` returns on reload.

        Returns:
            The active theme after reload and the mounted `AppMessage` text.
        """
        from deepagents_code import app as app_module
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.tui.widgets.messages import AppMessage

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.theme = initial_theme

            async def _fake_discover() -> bool:  # noqa: RUF029  # awaited by handler
                return True

            monkeypatch.setattr(app, "_discover_skills", _fake_discover)
            monkeypatch.setattr(
                app_module, "_load_theme_preference", lambda: resolved_theme
            )

            await app._handle_command("/reload")
            if app._reload_task is not None:
                await app._reload_task
            await pilot.pause()

            text = "\n".join(str(w._content) for w in app.query(AppMessage))
            return app.theme, text

    async def test_switches_to_new_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A newly resolved preference should become the active theme."""
        active, text = await self._run_reload_theme(
            monkeypatch,
            initial_theme="langchain",
            resolved_theme="langchain-light",
        )
        assert active == "langchain-light"
        assert "Switched theme to" in text

    async def test_no_switch_when_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the resolved preference matches the active theme, no switch."""
        active, text = await self._run_reload_theme(
            monkeypatch,
            initial_theme="langchain",
            resolved_theme="langchain",
        )
        assert active == "langchain"
        assert "Switched theme to" not in text

    async def test_unregistered_preference_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A resolved name that isn't registered must not change the theme."""
        active, text = await self._run_reload_theme(
            monkeypatch,
            initial_theme="langchain",
            resolved_theme="not-a-real-theme",
        )
        assert active == "langchain"
        assert "Switched theme to" not in text


class TestReloadPluginsViaReload:
    """Plugins should reload through `/reload`."""

    def test_fingerprint_detects_nested_skill_edits(self, tmp_path: Path) -> None:
        """Editing `SKILL.md` under a skills directory must change the fingerprint."""
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import ComponentInventory, PluginInstance

        skills_root = tmp_path / "skills"
        skill_dir = skills_root / "demo"
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("---\nname: demo\n---\noriginal\n", encoding="utf-8")

        plugin = PluginInstance(
            plugin_id="demo@tools",
            name="demo",
            marketplace="tools",
            version="1.0",
            root=tmp_path,
            data_dir=tmp_path / "data",
            manifest=None,
            inventory=ComponentInventory(skills=(skills_root,)),
        )

        before = DeepAgentsApp._fingerprint_plugins((plugin,))
        skill_md.write_text("---\nname: demo\n---\nedited\n", encoding="utf-8")
        after = DeepAgentsApp._fingerprint_plugins((plugin,))

        assert before != after

    def test_fingerprint_detects_added_and_removed_nested_files(
        self, tmp_path: Path
    ) -> None:
        """Adding then removing a nested file must each change the fingerprint.

        Directory stat alone can stay put across a child add/remove on some
        filesystems, so the recursive walk is what has to notice.
        """
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import ComponentInventory, PluginInstance

        skills_root = tmp_path / "skills" / "demo"
        skills_root.mkdir(parents=True)
        (skills_root / "SKILL.md").write_text("---\nname: demo\n---\n", "utf-8")

        plugin = PluginInstance(
            plugin_id="demo@tools",
            name="demo",
            marketplace="tools",
            version="1.0",
            root=tmp_path,
            data_dir=tmp_path / "data",
            manifest=None,
            inventory=ComponentInventory(skills=(tmp_path / "skills",)),
        )

        base = DeepAgentsApp._fingerprint_plugins((plugin,))
        extra = skills_root / "helper.py"
        extra.write_text("x = 1\n", encoding="utf-8")
        with_extra = DeepAgentsApp._fingerprint_plugins((plugin,))
        extra.unlink()
        removed = DeepAgentsApp._fingerprint_plugins((plugin,))

        assert base != with_extra
        assert with_extra != removed
        assert base == removed

    def test_fingerprint_records_sentinel_for_unreadable_subdir(
        self, tmp_path: Path
    ) -> None:
        """An unreadable subdirectory must perturb the fingerprint, not vanish.

        Recursive scanners can skip subtrees they cannot descend into, which
        would otherwise let an edit hidden beneath one read as "no change".
        """
        import sys

        if sys.platform == "win32" or os.geteuid() == 0:
            pytest.skip("chmod-based unreadability is not enforced here")

        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import ComponentInventory, PluginInstance

        skills_root = tmp_path / "skills"
        locked = skills_root / "locked"
        locked.mkdir(parents=True)
        (locked / "SKILL.md").write_text("---\nname: demo\n---\n", "utf-8")
        plugin = PluginInstance(
            plugin_id="demo@tools",
            name="demo",
            marketplace="tools",
            version="1.0",
            root=tmp_path,
            data_dir=tmp_path / "data",
            manifest=None,
            inventory=ComponentInventory(skills=(skills_root,)),
        )

        readable = DeepAgentsApp._fingerprint_plugins((plugin,))
        locked.chmod(0o000)
        try:
            unreadable = DeepAgentsApp._fingerprint_plugins((plugin,))
        finally:
            locked.chmod(0o755)

        # The locked directory is recorded as a -1/-1 sentinel rather than
        # dropped, so the change is visible instead of silently swallowed.
        assert readable != unreadable
        fingerprint = unreadable["demo@tools"]
        assert any(
            entry.path == str(locked) and entry.mtime_ns == -1
            for entry in fingerprint.components
        )
        assert DeepAgentsApp._plugin_fingerprint_changed(fingerprint, fingerprint)

    def test_fingerprint_caps_recursive_scan(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Oversized component trees stop at the hard entry limit."""
        from deepagents_code import app as app_module
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import ComponentInventory, PluginInstance

        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        for index in range(5):
            (skills_root / f"skill-{index}.md").write_text(str(index), encoding="utf-8")
        monkeypatch.setattr(app_module, "_MAX_PLUGIN_FINGERPRINT_ENTRIES", 3)
        plugin = PluginInstance(
            plugin_id="demo@tools",
            name="demo",
            marketplace="tools",
            version="1.0",
            root=tmp_path,
            data_dir=tmp_path / "data",
            manifest=None,
            inventory=ComponentInventory(skills=(skills_root,)),
        )

        first = DeepAgentsApp._fingerprint_plugins((plugin,))[plugin.plugin_id]
        second = DeepAgentsApp._fingerprint_plugins((plugin,))[plugin.plugin_id]

        assert any(
            entry.mtime_ns == app_module._TRUNCATED_PLUGIN_FINGERPRINT_STAT
            for entry in first.components
        )
        assert DeepAgentsApp._plugin_fingerprint_changed(first, second)

    def test_fingerprint_does_not_follow_symlinks(self, tmp_path: Path) -> None:
        """A nested directory symlink is recorded without scanning its target."""
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import ComponentInventory, PluginInstance

        plugin_root = tmp_path / "plugin"
        skills_root = plugin_root / "skills"
        outside = tmp_path / "outside"
        skills_root.mkdir(parents=True)
        outside.mkdir()
        outside_file = outside / "external.md"
        outside_file.write_text("external", encoding="utf-8")
        link = skills_root / "linked"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (NotImplementedError, OSError):
            pytest.skip("directory symlinks are unavailable")
        plugin = PluginInstance(
            plugin_id="demo@tools",
            name="demo",
            marketplace="tools",
            version="1.0",
            root=plugin_root,
            data_dir=tmp_path / "data",
            manifest=None,
            inventory=ComponentInventory(skills=(skills_root,)),
        )

        fingerprint = DeepAgentsApp._fingerprint_plugins((plugin,))[plugin.plugin_id]
        paths = {entry.path for entry in fingerprint.components}

        assert str(link) in paths
        assert str(outside_file) not in paths

    def test_fingerprint_rejects_component_outside_plugin_root(
        self, tmp_path: Path
    ) -> None:
        """A malformed inventory cannot make fingerprinting inspect another tree."""
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import ComponentInventory, PluginInstance

        plugin_root = tmp_path / "plugin"
        plugin_root.mkdir()
        outside = tmp_path / "outside.json"
        outside.write_text('{"mcpServers": {}}', encoding="utf-8")
        plugin = PluginInstance(
            plugin_id="demo@tools",
            name="demo",
            marketplace="tools",
            version="1.0",
            root=plugin_root,
            data_dir=tmp_path / "data",
            manifest=None,
            inventory=ComponentInventory(mcp_files=(outside,)),
        )

        fingerprint = DeepAgentsApp._fingerprint_plugins((plugin,))[plugin.plugin_id]

        assert fingerprint.components == ((str(outside), -1, -1),)

    def test_fingerprint_detects_version_change(self, tmp_path: Path) -> None:
        """A version bump must change the fingerprint even with identical files."""
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import ComponentInventory, PluginInstance

        def _plugin(version: str) -> PluginInstance:
            return PluginInstance(
                plugin_id="demo@tools",
                name="demo",
                marketplace="tools",
                version=version,
                root=tmp_path,
                data_dir=tmp_path / "data",
                manifest=None,
                inventory=ComponentInventory(),
            )

        before = DeepAgentsApp._fingerprint_plugins((_plugin("1.0"),))
        after = DeepAgentsApp._fingerprint_plugins((_plugin("2.0"),))

        assert before != after

    def test_fingerprint_detects_manifest_change(self, tmp_path: Path) -> None:
        """A manifest change must flip the fingerprint even when files match."""
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import (
            ComponentInventory,
            PluginInstance,
            PluginManifest,
        )

        def _plugin(manifest_version: str) -> PluginInstance:
            manifest = PluginManifest(
                name="demo",
                version=manifest_version,
                component_paths={},
                inline_mcp={},
            )
            return PluginInstance(
                plugin_id="demo@tools",
                name="demo",
                marketplace="tools",
                # Hold the instance version fixed to isolate the manifest dimension.
                version="1.0",
                root=tmp_path,
                data_dir=tmp_path / "data",
                manifest=manifest,
                inventory=ComponentInventory(),
            )

        before = DeepAgentsApp._fingerprint_plugins((_plugin("1.0"),))
        after = DeepAgentsApp._fingerprint_plugins((_plugin("2.0"),))

        assert before != after

    def test_fingerprint_detects_mcp_file_edits(self, tmp_path: Path) -> None:
        """Editing an `mcp_files` entry (a file path) must change the fingerprint."""
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import ComponentInventory, PluginInstance

        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text('{"mcpServers": {}}', encoding="utf-8")

        plugin = PluginInstance(
            plugin_id="demo@tools",
            name="demo",
            marketplace="tools",
            version="1.0",
            root=tmp_path,
            data_dir=tmp_path / "data",
            manifest=None,
            inventory=ComponentInventory(mcp_files=(mcp_file,)),
        )

        before = DeepAgentsApp._fingerprint_plugins((plugin,))
        mcp_file.write_text('{"mcpServers": {"x": {}}}', encoding="utf-8")
        after = DeepAgentsApp._fingerprint_plugins((plugin,))

        assert before != after

    def test_plugin_login_labels_filters_dedupes_and_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Only added, login-needing plugins are listed, deduped, with name fallback."""
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import (
            ComponentInventory,
            PluginInstance,
            PluginManifest,
        )

        def _plugin(name: str, *, display_name: str | None) -> PluginInstance:
            manifest = (
                None
                if display_name is None
                else PluginManifest(
                    name=name,
                    display_name=display_name,
                    version="1.0",
                    component_paths={},
                    inline_mcp={},
                )
            )
            return PluginInstance(
                plugin_id=f"{name}@tools",
                name=name,
                marketplace="tools",
                version="1.0",
                root=tmp_path,
                data_dir=tmp_path / "data",
                manifest=manifest,
                inventory=ComponentInventory(),
            )

        # login_needed twice -> should dedupe to one "Login" label; no_login has
        # no login-requiring server; nameless falls back to plugin.name;
        # not_added is excluded because it is not in the added set.
        entries_by_id = {
            "login_needed@tools": (("s", "scoped", True),),
            "no_login@tools": (("s", "scoped", False),),
            "nameless@tools": (("s", "scoped", True),),
            "not_added@tools": (("s", "scoped", True),),
        }
        monkeypatch.setattr(
            "deepagents_code.plugins.adapters.mcp.plugin_mcp_server_entries",
            lambda plugin: entries_by_id[plugin.plugin_id],
        )

        plugins = (
            _plugin("login_needed", display_name="Login"),
            _plugin("dupe", display_name="Login"),
            _plugin("no_login", display_name="Silent"),
            _plugin("nameless", display_name=None),
            _plugin("not_added", display_name="Skipped"),
        )
        # "dupe" shares the "Login" display name to exercise deduplication.
        entries_by_id["dupe@tools"] = (("s", "scoped", True),)
        added = {
            "login_needed@tools",
            "dupe@tools",
            "no_login@tools",
            "nameless@tools",
        }

        labels = DeepAgentsApp._plugin_login_labels(plugins, added)

        assert labels == ("Login", "nameless")

    @pytest.mark.parametrize(
        "change", ["none", "fingerprint", "fingerprint_key_added", "enabled"]
    )
    async def test_plugin_manager_reminder_compares_actual_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        change: str,
    ) -> None:
        """Closing compares persisted state even when the modal reports no result."""
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import PluginDiscoveryResult

        if change == "fingerprint":
            # Same plugin id, different fingerprint: exercises the intersection
            # branch of _plugin_fingerprints_changed.
            before = {"demo@tools": _test_plugin_fingerprint("before")}
            after = {"demo@tools": _test_plugin_fingerprint("after")}
        elif change == "fingerprint_key_added":
            # A plugin appears (e.g. a marketplace added while the manager was
            # open) with enabled ids unchanged: exercises the keys-differ branch,
            # which is this PR's headline async-marketplace-add scenario.
            before = {}
            after = {"demo@tools": _test_plugin_fingerprint("after")}
        else:
            before = {}
            after = {}
        fingerprints = iter((before, after))
        enabled_before = frozenset[str]()
        enabled_after = (
            frozenset({"demo@tools"}) if change == "enabled" else enabled_before
        )
        enabled_ids = iter((enabled_before, enabled_after))
        app = DeepAgentsApp()
        pushed: list[PluginManagerScreen] = []
        ui_thread = threading.get_ident()
        fingerprint_threads: list[int] = []

        def fingerprint_plugins(_plugins: object) -> dict[str, _PluginFingerprint]:
            fingerprint_threads.append(threading.get_ident())
            return next(fingerprints)

        monkeypatch.setattr(
            "deepagents_code.plugins.discover_plugins",
            lambda: PluginDiscoveryResult(plugins=()),
        )
        monkeypatch.setattr(
            "deepagents_code.plugins.store.load_enabled_plugin_ids",
            lambda: next(enabled_ids),
        )
        monkeypatch.setattr(
            app,
            "_fingerprint_plugins",
            fingerprint_plugins,
        )
        monkeypatch.setattr(
            app,
            "push_screen",
            lambda screen, _callback: pushed.append(screen),
        )

        await app._show_plugin_manager()
        screen = pushed[0]
        reload_required = await _check_plugin_reload(screen)

        assert app._plugin_fingerprints == before
        assert len(fingerprint_threads) == 2
        assert all(thread != ui_thread for thread in fingerprint_threads)
        assert reload_required is (change != "none")

    async def test_plugin_manager_state_error_schedules_reminder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A state read failure should not escape the modal dismiss callback."""
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import PluginDiscoveryResult

        discovery = MagicMock(
            side_effect=[
                PluginDiscoveryResult(plugins=()),
                PermissionError("plugin directory is unreadable"),
            ]
        )
        app = DeepAgentsApp()
        mount = AsyncMock()
        pushed: list[PluginManagerScreen] = []
        callbacks: list[Callable[[PluginManagerResult], None]] = []
        scheduled: list[Coroutine[object, object, None]] = []

        monkeypatch.setattr("deepagents_code.plugins.discover_plugins", discovery)
        monkeypatch.setattr(
            "deepagents_code.plugins.store.load_enabled_plugin_ids",
            lambda: frozenset[str](),
        )
        monkeypatch.setattr(
            app,
            "push_screen",
            lambda screen, callback: (
                pushed.append(screen),
                callbacks.append(callback),
            ),
        )
        monkeypatch.setattr(app, "call_after_refresh", lambda callback: callback())
        monkeypatch.setattr(
            app,
            "run_worker",
            lambda coroutine, **_kwargs: scheduled.append(coroutine),
        )
        monkeypatch.setattr(app, "_mount_message", mount)

        await app._show_plugin_manager()
        reload_required = await _check_plugin_reload(pushed[0])
        assert reload_required is None
        callbacks[0]("check_failed")
        await scheduled[0]

        mount.assert_awaited_once()
        mount_call = mount.await_args
        assert mount_call is not None
        message = mount_call.args[0]
        assert "Couldn't check plugin state" in str(message._content)
        assert "/reload" in str(message._content)

    async def test_plugin_manager_reopen_preserves_deferred_reload_baseline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reopening after deferral must retain the state from before changes."""
        from deepagents_code.app import DeepAgentsApp

        before: dict[str, _PluginFingerprint] = {}
        after = {"linear@tools": _test_plugin_fingerprint("v1")}
        snapshots = iter(
            (
                (frozenset[str](), before),
                (frozenset({"linear@tools"}), after),
                (frozenset({"linear@tools"}), after),
                (frozenset({"linear@tools"}), after),
            )
        )
        app = DeepAgentsApp()
        pushed: list[PluginManagerScreen] = []

        monkeypatch.setattr(app, "_snapshot_plugin_state", lambda: next(snapshots))
        monkeypatch.setattr(
            app,
            "push_screen",
            lambda screen, _callback: pushed.append(screen),
        )

        await app._show_plugin_manager()
        first_reload_required = await _check_plugin_reload(pushed[0])
        await app._show_plugin_manager()
        second_reload_required = await _check_plugin_reload(pushed[1])

        assert app._plugin_fingerprints == before
        assert first_reload_required is True
        assert second_reload_required is False

    async def test_plugin_manager_warns_when_snapshot_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pre-open snapshot failure still opens the manager and warns on close."""
        from deepagents_code.app import DeepAgentsApp

        app = DeepAgentsApp()
        pushed: list[PluginManagerScreen] = []
        callbacks: list[Callable[[PluginManagerResult], None]] = []
        scheduled: list[Coroutine[object, object, None]] = []
        mount = AsyncMock()

        monkeypatch.setattr(
            app,
            "_snapshot_plugin_state",
            MagicMock(side_effect=PermissionError("plugin directory is unreadable")),
        )
        monkeypatch.setattr(
            app,
            "push_screen",
            lambda screen, callback: (
                pushed.append(screen),
                callbacks.append(callback),
            ),
        )
        monkeypatch.setattr(app, "call_after_refresh", lambda callback: callback())
        monkeypatch.setattr(
            app,
            "run_worker",
            lambda coroutine, **_kwargs: scheduled.append(coroutine),
        )
        monkeypatch.setattr(app, "_mount_message", mount)

        await app._show_plugin_manager()
        reload_required = await _check_plugin_reload(pushed[0])
        assert reload_required is None
        callbacks[0]("check_failed")
        await scheduled[0]

        assert len(pushed) == 1
        assert len(scheduled) == 1
        mount.assert_awaited_once()
        mount_call = mount.await_args
        assert mount_call is not None
        assert "Couldn't check plugin state" in str(mount_call.args[0]._content)
        assert app._plugin_fingerprints is None

    @pytest.mark.parametrize(
        "result",
        ["reload", "later", "check_failed", None],
    )
    async def test_plugin_manager_close_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        result: PluginManagerResult,
    ) -> None:
        """Manager outcomes reload, defer, warn, or close without feedback."""
        from deepagents_code.app import DeepAgentsApp

        app = DeepAgentsApp()
        callbacks: list[Callable[[PluginManagerResult], None]] = []
        scheduled: list[Coroutine[object, object, None]] = []
        submit = AsyncMock()
        mount = AsyncMock()
        monkeypatch.setattr(
            app,
            "_snapshot_plugin_state",
            lambda: (frozenset[str](), {}),
        )
        monkeypatch.setattr(
            app,
            "push_screen",
            lambda _screen, callback: callbacks.append(callback),
        )
        monkeypatch.setattr(app, "call_after_refresh", lambda callback: callback())
        monkeypatch.setattr(
            app,
            "run_worker",
            lambda coroutine, **_kwargs: scheduled.append(coroutine),
        )
        monkeypatch.setattr(app, "_submit_input", submit)
        monkeypatch.setattr(app, "_mount_message", mount)

        await app._show_plugin_manager()
        callbacks[0](result)
        if result is not None:
            await scheduled[0]

        if result == "reload":
            submit.assert_awaited_once_with("/reload", "command")
            mount.assert_not_awaited()
        elif result is None:
            submit.assert_not_awaited()
            mount.assert_not_awaited()
            assert scheduled == []
        else:
            submit.assert_not_awaited()
            mount.assert_awaited_once()
            mount_call = mount.await_args
            assert mount_call is not None
            message = str(mount_call.args[0]._content)
            assert "/reload" in message
            if result == "check_failed":
                assert "Couldn't check plugin state" in message
            else:
                assert "Plugin changes are pending" in message

    async def test_reports_plugin_summary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`/reload` includes a plugin summary."""
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import PluginDiscoveryResult
        from deepagents_code.tui.widgets.messages import AppMessage

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            async def _fake_discover() -> bool:  # noqa: RUF029
                return True

            monkeypatch.setattr(app, "_discover_skills", _fake_discover)
            monkeypatch.setattr(
                "deepagents_code.plugins.discover_plugins",
                lambda: PluginDiscoveryResult(plugins=()),
            )
            monkeypatch.setattr(
                "deepagents_code.plugins.adapters.mcp.plugin_mcp_configs",
                lambda _plugins: (),
            )

            await app._handle_command("/reload")
            if app._reload_task is not None:
                await app._reload_task
            await pilot.pause()

            text = "\n".join(str(w._content) for w in app.query(AppMessage))
            assert "Plugins: 0 plugins · 0 skills · 0 plugin MCP servers" in text

    async def _reload_transcript_with_fingerprints(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        old: dict[str, _PluginFingerprint] | None,
        new: dict[str, _PluginFingerprint],
        plugins: tuple[PluginInstance, ...] = (),
        fingerprint_threads: list[int] | None = None,
    ) -> str:
        """Drive `/reload` with seeded before/after fingerprints.

        Returns:
            The joined transcript text of all rendered app messages.
        """
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import PluginDiscoveryResult
        from deepagents_code.tui.widgets.messages import AppMessage

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            async def _fake_discover() -> bool:  # noqa: RUF029
                return True

            def fingerprint_plugins(
                _plugins: object,
            ) -> dict[str, _PluginFingerprint]:
                if fingerprint_threads is not None:
                    fingerprint_threads.append(threading.get_ident())
                return new

            monkeypatch.setattr(app, "_discover_skills", _fake_discover)
            monkeypatch.setattr(
                "deepagents_code.plugins.discover_plugins",
                lambda: PluginDiscoveryResult(plugins=plugins),
            )
            monkeypatch.setattr(
                "deepagents_code.plugins.adapters.mcp.plugin_mcp_configs",
                lambda _plugins: (),
            )
            monkeypatch.setattr(app, "_fingerprint_plugins", fingerprint_plugins)
            app._plugin_fingerprints = old

            await app._handle_command("/reload")
            if app._reload_task is not None:
                await app._reload_task
            await pilot.pause()

            return "\n".join(str(w._content) for w in app.query(AppMessage))

    @pytest.mark.parametrize(
        ("old_versions", "new_versions", "expected"),
        [
            pytest.param(
                {"demo@tools": "v1"},
                {"demo@tools": "v1"},
                "Plugin changes: no changes detected.",
                id="no-changes",
            ),
            pytest.param(
                {},
                {"demo@tools": "v1"},
                "Plugin changes: 1 plugin added.",
                id="added-singular",
            ),
            pytest.param(
                {},
                {"demo@tools": "v1", "extra@tools": "v1"},
                "Plugin changes: 2 plugins added.",
                id="added-plural",
            ),
            pytest.param(
                {"demo@tools": "v1"},
                {},
                "Plugin changes: 1 plugin removed.",
                id="removed",
            ),
            pytest.param(
                {"demo@tools": "v1"},
                {"demo@tools": "v2"},
                "Plugin changes: 1 plugin changed.",
                id="changed",
            ),
            pytest.param(
                {"demo@tools": "v1", "gone@tools": "v1"},
                {"demo@tools": "v2", "new@tools": "v1"},
                "Plugin changes: 1 plugin added, 1 plugin removed, 1 plugin changed.",
                id="added-removed-changed",
            ),
        ],
    )
    async def test_reload_report_summarizes_plugin_changes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        old_versions: dict[str, str],
        new_versions: dict[str, str],
        expected: str,
    ) -> None:
        """`/reload` summarizes added/removed/changed plugins against the baseline."""
        old = {
            plugin_id: _test_plugin_fingerprint(version)
            for plugin_id, version in old_versions.items()
        }
        new = {
            plugin_id: _test_plugin_fingerprint(version)
            for plugin_id, version in new_versions.items()
        }
        text = await self._reload_transcript_with_fingerprints(
            monkeypatch, old=old, new=new
        )

        assert expected in text

    async def test_reload_report_omits_changes_on_first_reload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The first `/reload` has no baseline, so it omits the changes line."""
        text = await self._reload_transcript_with_fingerprints(
            monkeypatch,
            old=None,
            new={"demo@tools": _test_plugin_fingerprint("v1")},
        )

        assert "Plugin changes:" not in text

    async def test_reload_fingerprints_plugins_off_ui_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`/reload` must not recursively scan plugin files on the UI thread."""
        ui_thread = threading.get_ident()
        fingerprint_threads: list[int] = []

        await self._reload_transcript_with_fingerprints(
            monkeypatch,
            old={},
            new={},
            fingerprint_threads=fingerprint_threads,
        )

        assert len(fingerprint_threads) == 1
        assert fingerprint_threads[0] != ui_thread

    async def test_reload_reports_mcp_login_for_new_plugin(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """New HTTP MCP plugins retain their post-reload sign-in guidance."""
        from deepagents_code.plugins.models import (
            ComponentInventory,
            PluginInstance,
            PluginManifest,
        )

        plugin = PluginInstance(
            plugin_id="linear@tools",
            name="linear",
            marketplace="tools",
            version="1.0",
            root=tmp_path,
            data_dir=tmp_path / "data",
            manifest=PluginManifest(
                name="linear",
                display_name="Linear",
                version="1.0",
                component_paths={},
                inline_mcp={
                    "mcpServers": {
                        "linear": {
                            "type": "http",
                            "url": "https://mcp.example.com",
                        }
                    }
                },
            ),
            inventory=ComponentInventory(),
        )

        text = await self._reload_transcript_with_fingerprints(
            monkeypatch,
            old={},
            new={plugin.plugin_id: _test_plugin_fingerprint("v1")},
            plugins=(plugin,),
        )

        assert "Sign in to Linear via `/mcp`." in text

    async def test_reload_omits_sign_in_on_first_reload(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """First `/reload` has no baseline, so sign-in guidance is suppressed too."""
        from deepagents_code.plugins.models import (
            ComponentInventory,
            PluginInstance,
            PluginManifest,
        )

        plugin = PluginInstance(
            plugin_id="linear@tools",
            name="linear",
            marketplace="tools",
            version="1.0",
            root=tmp_path,
            data_dir=tmp_path / "data",
            manifest=PluginManifest(
                name="linear",
                display_name="Linear",
                version="1.0",
                component_paths={},
                inline_mcp={
                    "mcpServers": {
                        "linear": {"type": "http", "url": "https://mcp.example.com"}
                    }
                },
            ),
            inventory=ComponentInventory(),
        )

        text = await self._reload_transcript_with_fingerprints(
            monkeypatch,
            old=None,
            new={plugin.plugin_id: _test_plugin_fingerprint("v1")},
            plugins=(plugin,),
        )

        assert "Sign in to" not in text

    async def test_reload_reports_plugin_discovery_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A discovery failure degrades to a manual-retry note, not a crash."""
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.tui.widgets.messages import AppMessage

        app = DeepAgentsApp()
        reload_hooks = AsyncMock()
        monkeypatch.setattr(app, "_reload_hooks", reload_hooks)
        async with app.run_test() as pilot:
            await pilot.pause()

            async def _fake_discover() -> bool:  # noqa: RUF029
                return True

            monkeypatch.setattr(app, "_discover_skills", _fake_discover)
            monkeypatch.setattr(
                "deepagents_code.plugins.discover_plugins",
                MagicMock(side_effect=PermissionError("unreadable plugin dir")),
            )

            await app._handle_command("/reload")
            if app._reload_task is not None:
                await app._reload_task
            await pilot.pause()

            text = "\n".join(str(w._content) for w in app.query(AppMessage))
            assert "Couldn't read plugin state; run /reload to be safe." in text
            # The reload still completed: the config-reload header is present and
            # no plugin summary line was emitted.
            assert "Configuration reloaded." in text
            assert "Plugins:" not in text
            reload_hooks.assert_awaited_once_with(plugins=())

    @pytest.mark.parametrize(
        ("restarted", "expected_ids"),
        [
            (False, frozenset({"old@tools"})),
            (True, frozenset({"new@tools"})),
        ],
    )
    async def test_updates_loaded_ids_only_after_successful_restart(
        self,
        monkeypatch: pytest.MonkeyPatch,
        restarted: bool,
        expected_ids: frozenset[str],
    ) -> None:
        """A failed restart leaves the prior server's plugin status intact."""
        from deepagents_code.app import DeepAgentsApp, _ServerRespawnResult
        from deepagents_code.plugins.models import PluginDiscoveryResult
        from deepagents_code.tui.widgets.messages import AppMessage

        plugin = MagicMock(plugin_id="new@tools")
        app = DeepAgentsApp()
        order: list[str] = []
        reload_hooks = AsyncMock(side_effect=lambda **_kwargs: order.append("hooks"))
        async with app.run_test() as pilot:
            await pilot.pause()
            app._session_plugin_ids = frozenset({"old@tools"})
            app._server_proc = MagicMock()
            app._server_kwargs = {}

            async def _fake_discover() -> bool:  # noqa: RUF029
                return True

            async def _fake_restart() -> _ServerRespawnResult:  # noqa: RUF029
                order.append("restart")
                return _ServerRespawnResult(restarted=restarted)

            monkeypatch.setattr(app, "_discover_skills", _fake_discover)
            monkeypatch.setattr(app, "_reload_hooks", reload_hooks)
            monkeypatch.setattr(app, "_restart_server_manual_result", _fake_restart)
            monkeypatch.setattr(app, "_discard_queue", lambda: None)
            monkeypatch.setattr(
                "deepagents_code.plugins.discover_plugins",
                lambda: PluginDiscoveryResult(plugins=(plugin,)),
            )
            monkeypatch.setattr(
                "deepagents_code.plugins.adapters.mcp.plugin_mcp_configs",
                lambda _plugins: (),
            )

            await app._handle_command("/reload")
            if app._reload_task is not None:
                await app._reload_task
            await pilot.pause()

            assert app._session_plugin_ids == expected_ids
            assert order == ["hooks", "restart"]

            # A report that says the server never restarted must not also
            # claim anything about MCP — the two lines would contradict.
            text = "\n".join(str(w._content) for w in app.query(AppMessage))
            assert ("MCP server changes" in text) is restarted

    @pytest.mark.parametrize(
        ("mcp_status", "expected", "forbidden"),
        [
            # `--no-mcp`: nothing to refresh, so an empty diff is the truth —
            # stated as the reason rather than as a bare "no changes", which
            # would read as a load next to the plugin MCP count printed above.
            (
                "disabled",
                "MCP is disabled for this session (--no-mcp); no servers were loaded.",
                "couldn't be determined",
            ),
            # MCP is on but the refresh failed: saying "no changes" here would
            # affirmatively misreport tool availability at the moment the app
            # knows least. These two cases must never collapse into one.
            (
                "unavailable",
                "couldn't be determined",
                "no changes detected",
            ),
        ],
    )
    async def test_reload_distinguishes_disabled_mcp_from_failed_refresh(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mcp_status: str,
        expected: str,
        forbidden: str,
    ) -> None:
        """`mcp_server_info=None` means opposite things in these two sessions."""
        from deepagents_code.app import DeepAgentsApp, _ServerRespawnResult
        from deepagents_code.mcp_tools import MCPServerInfo
        from deepagents_code.plugins.models import PluginDiscoveryResult
        from deepagents_code.tui.widgets.messages import AppMessage

        plugin = MagicMock(plugin_id="new@tools")
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._session_plugin_ids = frozenset({"old@tools"})
            app._server_proc = MagicMock()
            app._server_kwargs = {}
            if mcp_status == "unavailable":
                # An MCP-enabled session with a usable pre-reload baseline.
                app._mcp_preload_kwargs = {}
                app._mcp_server_info = [MCPServerInfo(name="notion", transport="stdio")]

            async def _fake_discover() -> bool:  # noqa: RUF029
                return True

            async def _fake_restart() -> _ServerRespawnResult:  # noqa: RUF029
                return _ServerRespawnResult(
                    restarted=True,
                    mcp_server_info=None,
                    mcp_status=mcp_status,  # ty: ignore[invalid-argument-type]
                )

            monkeypatch.setattr(app, "_discover_skills", _fake_discover)
            monkeypatch.setattr(app, "_reload_hooks", AsyncMock())
            monkeypatch.setattr(app, "_restart_server_manual_result", _fake_restart)
            monkeypatch.setattr(app, "_discard_queue", lambda: None)
            monkeypatch.setattr(
                "deepagents_code.plugins.discover_plugins",
                lambda: PluginDiscoveryResult(plugins=(plugin,)),
            )
            monkeypatch.setattr(
                "deepagents_code.plugins.adapters.mcp.plugin_mcp_configs",
                lambda _plugins: (),
            )

            await app._handle_command("/reload")
            if app._reload_task is not None:
                await app._reload_task
            await pilot.pause()

            text = "\n".join(str(w._content) for w in app.query(AppMessage))
            assert expected in text
            assert forbidden not in text

    async def test_reload_names_the_mcp_refresh_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refresh error identifies what broke; the log is not enough."""
        from deepagents_code.app import DeepAgentsApp, _ServerRespawnResult
        from deepagents_code.plugins.models import PluginDiscoveryResult
        from deepagents_code.tui.widgets.messages import AppMessage

        plugin = MagicMock(plugin_id="new@tools")
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._session_plugin_ids = frozenset({"old@tools"})
            app._server_proc = MagicMock()
            app._server_kwargs = {}
            app._mcp_preload_kwargs = {}

            async def _fake_discover() -> bool:  # noqa: RUF029
                return True

            async def _fake_restart() -> _ServerRespawnResult:  # noqa: RUF029
                return _ServerRespawnResult(
                    restarted=True,
                    mcp_status="unavailable",
                    mcp_error="RuntimeError: Invalid MCP server config: notion",
                )

            monkeypatch.setattr(app, "_discover_skills", _fake_discover)
            monkeypatch.setattr(app, "_reload_hooks", AsyncMock())
            monkeypatch.setattr(app, "_restart_server_manual_result", _fake_restart)
            monkeypatch.setattr(app, "_discard_queue", lambda: None)
            monkeypatch.setattr(
                "deepagents_code.plugins.discover_plugins",
                lambda: PluginDiscoveryResult(plugins=(plugin,)),
            )
            monkeypatch.setattr(
                "deepagents_code.plugins.adapters.mcp.plugin_mcp_configs",
                lambda _plugins: (),
            )

            await app._handle_command("/reload")
            if app._reload_task is not None:
                await app._reload_task
            await pilot.pause()

            text = "\n".join(str(w._content) for w in app.query(AppMessage))
            assert "Invalid MCP server config: notion" in text

    async def test_reload_says_remote_sessions_cannot_reload_mcp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without an owned server the MCP config was never re-read.

        The plugin line still counts plugin MCP servers, so saying nothing
        implies the reload picked them up.
        """
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import PluginDiscoveryResult
        from deepagents_code.tui.widgets.messages import AppMessage

        plugin = MagicMock(plugin_id="new@tools")
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._session_plugin_ids = frozenset({"old@tools"})
            # Attached to an externally managed server.
            app._server_proc = None
            app._server_kwargs = None

            async def _fake_discover() -> bool:  # noqa: RUF029
                return True

            monkeypatch.setattr(app, "_discover_skills", _fake_discover)
            monkeypatch.setattr(app, "_reload_hooks", AsyncMock())
            monkeypatch.setattr(app, "_discard_queue", lambda: None)
            monkeypatch.setattr(
                "deepagents_code.plugins.discover_plugins",
                lambda: PluginDiscoveryResult(plugins=(plugin,)),
            )
            monkeypatch.setattr(
                "deepagents_code.plugins.adapters.mcp.plugin_mcp_configs",
                lambda _plugins: (),
            )

            await app._handle_command("/reload")
            if app._reload_task is not None:
                await app._reload_task
            await pilot.pause()

            text = "\n".join(str(w._content) for w in app.query(AppMessage))
            assert "cannot reload MCP config" in text

    @pytest.mark.parametrize("no_mcp", [False, True])
    async def test_reload_deferred_local_startup_is_not_labeled_remote(
        self, monkeypatch: pytest.MonkeyPatch, no_mcp: bool
    ) -> None:
        """A pending local server start is not a remote session.

        Deferred startup leaves `_server_kwargs` populated with `_server_proc`
        still `None`; calling that "remote" would mislabel the session mode and
        contradict the deferred start that `/reload` itself may trigger.
        """
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import PluginDiscoveryResult
        from deepagents_code.tui.widgets.messages import AppMessage

        plugin = MagicMock(plugin_id="new@tools")
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._session_plugin_ids = frozenset({"old@tools"})
            # Deferred local startup: owned kwargs cached, no process yet.
            app._server_proc = None
            app._server_kwargs = {"no_mcp": no_mcp}
            app._server_startup_deferred = True

            async def _fake_discover() -> bool:  # noqa: RUF029
                return True

            monkeypatch.setattr(app, "_discover_skills", _fake_discover)
            monkeypatch.setattr(app, "_reload_hooks", AsyncMock())
            monkeypatch.setattr(app, "_discard_queue", lambda: None)
            # `_server_kwargs` is a stub, so the deferred start the reload
            # triggers must not actually boot a server.
            real_run_worker = app.run_worker

            def _run_worker_skip_server_start(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                if kwargs.get("group") == "server-startup":
                    return None
                return real_run_worker(*args, **kwargs)

            monkeypatch.setattr(app, "run_worker", _run_worker_skip_server_start)
            monkeypatch.setattr(
                "deepagents_code.plugins.discover_plugins",
                lambda: PluginDiscoveryResult(plugins=(plugin,)),
            )
            monkeypatch.setattr(
                "deepagents_code.plugins.adapters.mcp.plugin_mcp_configs",
                lambda _plugins: (),
            )

            await app._handle_command("/reload")
            if app._reload_task is not None:
                await app._reload_task
            await pilot.pause()

            text = "\n".join(str(w._content) for w in app.query(AppMessage))
            assert "remote" not in text
            assert "cannot reload MCP config" not in text
            if no_mcp:
                assert "MCP is disabled for this session (--no-mcp)" in text
            else:
                assert "hasn't started yet" in text

    async def test_preserves_messages_queued_before_cancelling_for_restart(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reload retains prompts queued while its active turn is cancelled."""
        from deepagents_code.app import DeepAgentsApp, QueuedMessage
        from deepagents_code.plugins.models import PluginDiscoveryResult

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._server_proc = MagicMock()
            app._server_kwargs = {}
            app._agent_worker = MagicMock()
            app._set_agent_running(True)
            app._pending_messages.append(QueuedMessage("follow up", "normal"))

            async def _fake_discover() -> bool:  # noqa: RUF029
                return True

            async def _fake_restart() -> bool:  # noqa: RUF029
                return True

            monkeypatch.setattr(app, "_discover_skills", _fake_discover)
            monkeypatch.setattr(app, "_reload_hooks", AsyncMock())
            monkeypatch.setattr(app, "_restart_server_manual", _fake_restart)
            monkeypatch.setattr(app, "_cancel_worker", app._discard_queue)
            monkeypatch.setattr(
                "deepagents_code.plugins.discover_plugins",
                lambda: PluginDiscoveryResult(plugins=()),
            )
            monkeypatch.setattr(
                "deepagents_code.plugins.adapters.mcp.plugin_mcp_configs",
                lambda _plugins: (),
            )

            await app._run_reload()

            assert [message.text for message in app._pending_messages] == ["follow up"]


class TestConfigGenerationAdvancesOnlyOnReload:
    """Config files are read once into one generation; `/reload` advances it.

    The whole point of a single process-wide generation is that no two readers
    disagree, which means a hand edit is deliberately inert until an explicit
    reload. Both halves need pinning: without the first assertion a future
    change could reintroduce per-call file reads and nothing would notice;
    without the second, `/reload` could stop refreshing the shared resolver and
    every reader would be frozen for the life of the process.
    """

    def test_hand_edit_is_inert_until_reload(self, tmp_path: Path) -> None:
        """A settings edit takes effect on `/reload`, not before."""
        from deepagents_code.config import is_memory_auto_save_enabled

        # `_isolate_state_dir` redirects `DEFAULT_CONFIG_PATH` here, and the
        # path is stable across this test — so the resolver cache key does not
        # change and staleness is genuinely observable.
        config_path = tmp_path / "config.toml"
        config_path.write_text("[memory]\nauto_save = true\n", encoding="utf-8")

        settings = Settings.from_environment()
        assert is_memory_auto_save_enabled() is True

        config_path.write_text("[memory]\nauto_save = false\n", encoding="utf-8")
        assert is_memory_auto_save_enabled() is True, (
            "a hand edit must not change a value mid-session"
        )

        settings.reload_from_environment()
        assert is_memory_auto_save_enabled() is False, (
            "`/reload` must advance the shared generation"
        )

    def test_preview_does_not_advance_the_generation(self, tmp_path: Path) -> None:
        """A dry run must not swap the config every other reader observes."""
        from deepagents_code.config import is_memory_auto_save_enabled

        config_path = tmp_path / "config.toml"
        config_path.write_text("[memory]\nauto_save = true\n", encoding="utf-8")

        settings = Settings.from_environment()
        assert is_memory_auto_save_enabled() is True

        config_path.write_text("[memory]\nauto_save = false\n", encoding="utf-8")
        settings.preview_reload_from_environment()

        assert is_memory_auto_save_enabled() is True, (
            "a preview must leave the in-force generation untouched"
        )


class TestClearCachesDropsEveryConfigView:
    """`clear_caches` must drop every module cache holding `config.toml`.

    `[threads]` is cached separately from the `[models]` snapshot, and its read
    path deliberately has no invalidator -- so `clear_caches` is the only thing
    standing between a `/reload` and a value frozen for the process lifetime.
    The four in-app thread writers invalidate it themselves, which is why
    dropping it here regressed only hand edits and left the suite green.
    """

    def test_reload_picks_up_a_hand_edited_thread_setting(
        self,
        tmp_path: Path,
    ) -> None:
        """A `[threads]` hand edit takes effect once caches are cleared."""
        from deepagents_code.model_config import clear_caches, load_thread_config

        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '[threads]\nsort_order = "created_at"\n', encoding="utf-8"
        )
        # Populates `_thread_config_cache`; only `load_thread_config` reads it.
        assert load_thread_config().sort_order == "created_at"

        config_path.write_text(
            '[threads]\nsort_order = "updated_at"\n', encoding="utf-8"
        )
        assert load_thread_config().sort_order == "created_at", (
            "a hand edit must not change a value mid-session"
        )

        clear_caches()

        assert load_thread_config().sort_order == "updated_at", (
            "`clear_caches` must invalidate the thread config cache"
        )


class TestDiagnosticDedupIsPerGeneration:
    """A repeated `/reload` of a still-broken file must keep reporting it.

    The dedup exists so one `dcode config` sweep over the whole manifest
    reports a bad file once instead of once per option. Scoped to the process
    it also silenced the second `/reload` -- the reason string is identical --
    which is the edit-and-retry loop where the user most needs the message.
    """

    def test_reload_lets_the_same_rejection_be_reported_again(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The same corrupt file warns once per generation, not once ever."""
        import logging

        from deepagents_code import model_config
        from deepagents_code.config_manifest import (
            _emit_ranked_diagnostics,
            get_option,
        )
        from deepagents_code.configuration.resolver import get_config_resolver

        config_path = model_config.DEFAULT_CONFIG_PATH
        config_path.write_text('[shell]\nallow_list = ["ls"]\n', encoding="utf-8")
        option = get_option("shell.allow_list")
        assert option is not None
        resolver = get_config_resolver()
        resolver.get(option)

        config_path.write_text("[shell\n", encoding="utf-8")

        def reload_and_count() -> int:
            resolver.reload()
            with caplog.at_level(
                logging.WARNING, logger="deepagents_code.config_manifest"
            ):
                caplog.clear()
                _emit_ranked_diagnostics(option, resolver.get(option))
            return len(caplog.records)

        assert reload_and_count() == 1
        # Second attempt, same failure, same reason string.
        assert reload_and_count() == 1

    def test_one_generation_still_reports_a_bad_file_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Within a generation the sweep stays deduplicated."""
        import logging

        from deepagents_code import model_config
        from deepagents_code.config_manifest import (
            _emit_ranked_diagnostics,
            get_option,
        )
        from deepagents_code.configuration.resolver import get_config_resolver

        config_path = model_config.DEFAULT_CONFIG_PATH
        config_path.write_text('[shell]\nallow_list = ["ls"]\n', encoding="utf-8")
        option = get_option("shell.allow_list")
        assert option is not None
        resolver = get_config_resolver()
        resolver.get(option)

        config_path.write_text("[shell\n", encoding="utf-8")
        resolver.reload()

        with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
            caplog.clear()
            for _ in range(5):
                _emit_ranked_diagnostics(option, resolver.get(option))

        assert len(caplog.records) == 1

    def test_rebuilding_the_resolver_re_arms_the_dedup(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A rebuilt resolver is a new generation and must report afresh.

        The set was cleared on reload but not on the cache miss that builds a
        fresh resolver, and the cache key includes the managed path -- so
        installing or removing policy advanced the generation while the dedup
        stayed alive from the one before it.

        Asserted on the set rather than on log output: the rejection reasons
        differ across a rebuild (the source paths change), so counting log
        records passes either way and pins nothing. The key is moved by
        repointing the user path, because the managed route to a rebuild runs
        through `invalidate_config_sources`, which clears the set itself and
        would mask what this test is for.
        """
        from deepagents_code import config_manifest, model_config
        from deepagents_code.configuration.resolver import get_config_resolver

        get_config_resolver()
        config_manifest._warned_non_table_paths.add(("ranked provider", "stale"))

        moved = tmp_path / "moved" / "config.toml"
        moved.parent.mkdir(parents=True, exist_ok=True)
        moved.write_text("", encoding="utf-8")
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", moved)

        get_config_resolver()

        assert config_manifest._warned_non_table_paths == set(), (
            "a rebuilt generation must re-arm the source diagnostics"
        )
