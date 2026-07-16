"""Tests for runtime config reload behavior."""

from __future__ import annotations

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
    from collections.abc import Coroutine
    from pathlib import Path

    from deepagents_code.plugins.models import PluginInstance

# Capture before any monkeypatching replaces it on the module.
_real_load_dotenv = _dotenv_module.load_dotenv

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
        call_count = 0

        def _fail_on_global(*, dotenv_path: Path) -> dict[str, str | None]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                msg = "read error"
                raise OSError(msg)
            return dict(original_dotenv_values(dotenv_path=dotenv_path))

        monkeypatch.setattr("dotenv.dotenv_values", _fail_on_global)

        with caplog.at_level(logging.WARNING, logger="deepagents_code.config"):
            settings.reload_from_environment(start_path=tmp_path)

        assert call_count == 2
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
            "ENV=/tmp/evil.sh\n"
            "GLOBIGNORE=*\n"
            "LD_PRELOAD=/tmp/evil.so\n"
            "PYTHONPATH=/tmp/evil\n"
            "PATH=/tmp/evil\n"
            "NODE_OPTIONS=--require /tmp/evil.js\n"
            "SHELLOPTS=xtrace\n"
            "DEEPAGENTS_INHERITED_PYTHONPATH=/tmp/evil\n"
            "OPENAI_API_KEY=sk-ok\n"
        )
        for key in (
            "BASH_ENV",
            "BASHOPTS",
            "CDPATH",
            "ENV",
            "GLOBIGNORE",
            "LD_PRELOAD",
            "PYTHONPATH",
            "NODE_OPTIONS",
            "SHELLOPTS",
            "DEEPAGENTS_INHERITED_PYTHONPATH",
            "OPENAI_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)

        _load_dotenv(start_path=tmp_path)

        assert "BASH_ENV" not in os.environ
        assert "BASHOPTS" not in os.environ
        assert "CDPATH" not in os.environ
        assert "ENV" not in os.environ
        assert "GLOBIGNORE" not in os.environ
        assert "LD_PRELOAD" not in os.environ
        assert "PYTHONPATH" not in os.environ
        assert "NODE_OPTIONS" not in os.environ
        assert "SHELLOPTS" not in os.environ
        # The carrier var must not be injectable from `.env`, or a project could
        # smuggle a PYTHONPATH into agent `execute` commands through it.
        assert "DEEPAGENTS_INHERITED_PYTHONPATH" not in os.environ
        assert os.environ["OPENAI_API_KEY"] == "sk-ok"

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
            "ENV",
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
    """Experimental plugins should reload through `/reload`."""

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

    @pytest.mark.parametrize("change", ["none", "fingerprint", "enabled"])
    async def test_plugin_manager_reminder_compares_actual_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        change: str,
    ) -> None:
        """Closing compares persisted state even when the modal reports no result."""
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import PluginDiscoveryResult

        before = {"demo@tools": ("before",)} if change == "fingerprint" else {}
        after = {"demo@tools": ("after",)} if change == "fingerprint" else {}
        fingerprints = iter((before, after))
        enabled_before = frozenset[str]()
        enabled_after = (
            frozenset({"demo@tools"}) if change == "enabled" else enabled_before
        )
        enabled_ids = iter((enabled_before, enabled_after))
        app = DeepAgentsApp()
        offer_reload = AsyncMock()
        scheduled: list[Coroutine[object, object, None]] = []
        ui_thread = threading.get_ident()
        fingerprint_threads: list[int] = []

        def fingerprint_plugins(_plugins: object) -> dict[str, tuple[str, ...]]:
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
            lambda _screen, callback: callback(None),
        )
        monkeypatch.setattr(app, "call_after_refresh", lambda callback: callback())
        monkeypatch.setattr(
            app,
            "run_worker",
            lambda coroutine, **_kwargs: scheduled.append(coroutine),
        )
        monkeypatch.setattr(app, "_offer_plugin_reload", offer_reload)

        await app._show_plugin_manager()
        await scheduled[0]

        assert app._plugin_fingerprints == before
        assert len(fingerprint_threads) == 2
        assert all(thread != ui_thread for thread in fingerprint_threads)
        if change != "none":
            offer_reload.assert_awaited_once()
        else:
            offer_reload.assert_not_awaited()

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
        scheduled: list[Coroutine[object, object, None]] = []

        monkeypatch.setattr("deepagents_code.plugins.discover_plugins", discovery)
        monkeypatch.setattr(
            "deepagents_code.plugins.store.load_enabled_plugin_ids",
            lambda: frozenset[str](),
        )
        monkeypatch.setattr(
            app,
            "push_screen",
            lambda _screen, callback: callback(None),
        )
        monkeypatch.setattr(app, "call_after_refresh", lambda callback: callback())
        monkeypatch.setattr(
            app,
            "run_worker",
            lambda coroutine, **_kwargs: scheduled.append(coroutine),
        )
        monkeypatch.setattr(app, "_mount_message", mount)

        await app._show_plugin_manager()
        await scheduled[0]

        mount.assert_awaited_once()
        mount_call = mount.await_args
        assert mount_call is not None
        message = mount_call.args[0]
        assert "Couldn't check plugin state" in str(message._content)
        assert "/reload" in str(message._content)

    @pytest.mark.parametrize("choice", ["reload", "later", None])
    async def test_plugin_reload_prompt_choice(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        choice: str | None,
    ) -> None:
        """Reload runs the command; deferral leaves one transcript reminder."""
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.tui.widgets.messages import AppMessage

        app = DeepAgentsApp()
        submit = AsyncMock()
        mount = AsyncMock()
        monkeypatch.setattr(
            app,
            "_push_screen_wait",
            AsyncMock(return_value=choice),
        )
        monkeypatch.setattr(app, "_submit_input", submit)
        monkeypatch.setattr(app, "_mount_message", mount)

        await app._offer_plugin_reload()

        if choice == "reload":
            submit.assert_awaited_once_with("/reload", "command")
            mount.assert_not_awaited()
        else:
            submit.assert_not_awaited()
            mount.assert_awaited_once()
            mount_call = mount.await_args
            assert mount_call is not None
            message = mount_call.args[0]
            assert isinstance(message, AppMessage)
            assert "/reload" in str(message._content)

    @pytest.mark.parametrize(
        "error",
        [TimeoutError(), RuntimeError("prompt mount failed")],
        ids=["timeout", "unexpected-error"],
    )
    async def test_plugin_reload_prompt_error_leaves_reminder(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        error: Exception,
    ) -> None:
        """Prompt failures should retain a manual `/reload` recovery path."""
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.tui.widgets.messages import AppMessage

        app = DeepAgentsApp()
        submit = AsyncMock()
        mount = AsyncMock()
        monkeypatch.setattr(
            app,
            "_push_screen_wait",
            AsyncMock(side_effect=error),
        )
        monkeypatch.setattr(app, "_submit_input", submit)
        monkeypatch.setattr(app, "_mount_message", mount)

        await app._offer_plugin_reload()

        submit.assert_not_awaited()
        mount.assert_awaited_once()
        mount_call = mount.await_args
        assert mount_call is not None
        message = mount_call.args[0]
        assert isinstance(message, AppMessage)
        assert "/reload" in str(message._content)

    async def test_reports_plugin_summary_when_experimental(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`/reload` includes a plugin summary when experimental mode is on."""
        from deepagents_code._env_vars import EXPERIMENTAL
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import PluginDiscoveryResult
        from deepagents_code.tui.widgets.messages import AppMessage

        monkeypatch.setenv(EXPERIMENTAL, "1")

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
            await pilot.pause()

            text = "\n".join(str(w._content) for w in app.query(AppMessage))
            assert "Plugins: 0 plugins · 0 skills · 0 plugin MCP servers" in text

    async def _reload_transcript_with_fingerprints(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        old: dict[str, tuple[object, ...]] | None,
        new: dict[str, tuple[object, ...]],
        plugins: tuple[PluginInstance, ...] = (),
        fingerprint_threads: list[int] | None = None,
    ) -> str:
        """Drive `/reload` with seeded before/after fingerprints.

        Returns:
            The joined transcript text of all rendered app messages.
        """
        from deepagents_code._env_vars import EXPERIMENTAL
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import PluginDiscoveryResult
        from deepagents_code.tui.widgets.messages import AppMessage

        monkeypatch.setenv(EXPERIMENTAL, "1")

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            async def _fake_discover() -> bool:  # noqa: RUF029
                return True

            def fingerprint_plugins(
                _plugins: object,
            ) -> dict[str, tuple[object, ...]]:
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
            await pilot.pause()

            return "\n".join(str(w._content) for w in app.query(AppMessage))

    @pytest.mark.parametrize(
        ("old", "new", "expected"),
        [
            pytest.param(
                {"demo@tools": ("v1",)},
                {"demo@tools": ("v1",)},
                "Plugin changes: no changes detected.",
                id="no-changes",
            ),
            pytest.param(
                {},
                {"demo@tools": ("v1",)},
                "Plugin changes: 1 plugin added.",
                id="added-singular",
            ),
            pytest.param(
                {},
                {"demo@tools": ("v1",), "extra@tools": ("v1",)},
                "Plugin changes: 2 plugins added.",
                id="added-plural",
            ),
            pytest.param(
                {"demo@tools": ("v1",)},
                {},
                "Plugin changes: 1 plugin removed.",
                id="removed",
            ),
            pytest.param(
                {"demo@tools": ("v1",)},
                {"demo@tools": ("v2",)},
                "Plugin changes: 1 plugin changed.",
                id="changed",
            ),
            pytest.param(
                {"demo@tools": ("v1",), "gone@tools": ("v1",)},
                {"demo@tools": ("v2",), "new@tools": ("v1",)},
                "Plugin changes: 1 plugin added, 1 plugin removed, 1 plugin changed.",
                id="added-removed-changed",
            ),
        ],
    )
    async def test_reload_report_summarizes_plugin_changes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        old: dict[str, tuple[object, ...]],
        new: dict[str, tuple[object, ...]],
        expected: str,
    ) -> None:
        """`/reload` summarizes added/removed/changed plugins against the baseline."""
        text = await self._reload_transcript_with_fingerprints(
            monkeypatch, old=old, new=new
        )

        assert expected in text

    async def test_reload_report_omits_changes_on_first_reload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The first `/reload` has no baseline, so it omits the changes line."""
        text = await self._reload_transcript_with_fingerprints(
            monkeypatch, old=None, new={"demo@tools": ("v1",)}
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
            new={plugin.plugin_id: ("v1",)},
            plugins=(plugin,),
        )

        assert "Sign in to Linear via `/mcp`." in text

    async def test_skips_plugin_summary_when_experimental_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`/reload` omits plugin summary when experimental mode is off."""
        from deepagents_code._env_vars import EXPERIMENTAL
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.tui.widgets.messages import AppMessage

        monkeypatch.setenv(EXPERIMENTAL, "0")

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            async def _fake_discover() -> bool:  # noqa: RUF029
                return True

            monkeypatch.setattr(app, "_discover_skills", _fake_discover)

            await app._handle_command("/reload")
            await pilot.pause()

            text = "\n".join(str(w._content) for w in app.query(AppMessage))
            assert "Plugins:" not in text

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
        from deepagents_code._env_vars import EXPERIMENTAL
        from deepagents_code.app import DeepAgentsApp
        from deepagents_code.plugins.models import PluginDiscoveryResult

        monkeypatch.setenv(EXPERIMENTAL, "1")
        plugin = MagicMock(plugin_id="new@tools")
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._session_plugin_ids = frozenset({"old@tools"})
            app._server_proc = MagicMock()
            app._server_kwargs = {}

            async def _fake_discover() -> bool:  # noqa: RUF029
                return True

            async def _fake_restart() -> bool:  # noqa: RUF029
                return restarted

            monkeypatch.setattr(app, "_discover_skills", _fake_discover)
            monkeypatch.setattr(app, "_restart_server_manual", _fake_restart)
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
            await pilot.pause()

            assert app._session_plugin_ids == expected_ids
