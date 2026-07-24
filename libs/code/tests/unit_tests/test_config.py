"""Tests for config module including project discovery utilities."""

import logging
import time
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import Mock, patch

import pytest

from deepagents_code import _git as git_module, model_config
from deepagents_code._env_vars import SERVER_ENV_PREFIX
from deepagents_code._version import __version__
from deepagents_code.config import (
    _QUIET_SDK_LOGGER_NAMES,
    CLI_MAX_RETRIES_KEY,
    LANGSMITH_EU_ENDPOINT,
    LANGSMITH_US_ENDPOINT,
    RECOMMENDED_SAFE_SHELL_COMMANDS,
    SHELL_ALLOW_ALL,
    LangSmithApiError,
    LangSmithProjectNotFoundError,
    LangsmithShadowResult,
    ModelResult,
    Settings,
    _apply_default_langsmith_project,
    _apply_stored_langsmith_tracing,
    _create_model_from_class,
    _create_model_via_init,
    _disable_orphaned_tracing,
    _get_provider_kwargs,
    _quiet_sdk_logging,
    _read_config_toml_retries,
    _resolve_retry_kwargs,
    _resolve_retry_param_name,
    apply_stored_langsmith_auth,
    build_langsmith_thread_url,
    configure_langsmith_secret_redaction,
    consume_orphaned_tracing_disabled_notice,
    create_model,
    detect_mode_prefix,
    detect_provider,
    fetch_langsmith_project_url,
    fetch_langsmith_project_url_or_raise,
    get_cached_langsmith_thread_url,
    get_langsmith_project_name,
    is_http_url,
    is_langsmith_redaction_enabled,
    langsmith_key_shadowed_by_empty_override,
    newline_shortcut,
    normalize_langsmith_endpoint,
    parse_shell_allow_list,
    reset_langsmith_url_cache,
    settings,
    validate_model_capabilities,
)
from deepagents_code.model_config import ModelConfigError, clear_caches
from deepagents_code.project_utils import (
    ProjectContext,
    find_project_agent_md as _find_project_agent_md,
    find_project_root as _find_project_root,
    get_server_project_context,
)


class TestRuntimeDotenvReload:
    """Tests for project-scoped dotenv refresh behavior."""

    def test_reload_from_environment_refreshes_loaded_project_dotenv_values(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Runtime reload replaces managed `.env` values after a cwd switch."""
        import os

        import deepagents_code.config as config_mod

        current = tmp_path / "current"
        target = tmp_path / "target"
        current.mkdir()
        target.mkdir()
        (current / ".env").write_text(
            "DEEPAGENTS_CODE_OPENAI_API_KEY=sk-current\n",
        )
        (target / ".env").write_text(
            "DEEPAGENTS_CODE_OPENAI_API_KEY=sk-target\n"
            "DEEPAGENTS_CODE_ANTHROPIC_API_KEY=sk-target-anthropic\n",
        )

        monkeypatch.delenv("DEEPAGENTS_CODE_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", "sk-shell")
        monkeypatch.setattr(
            config_mod,
            "_GLOBAL_DOTENV_PATH",
            tmp_path / "missing-global.env",
        )
        config_mod._dotenv_loaded_values.clear()

        try:
            config_mod._load_dotenv(start_path=current)
            runtime = Settings.from_environment(start_path=current)
            assert runtime.openai_api_key == "sk-current"

            changes = runtime.reload_from_environment(start_path=target)

            assert runtime.openai_api_key == "sk-target"
            assert os.environ["DEEPAGENTS_CODE_OPENAI_API_KEY"] == "sk-target"
            assert runtime.anthropic_api_key == "sk-shell"
            assert os.environ["DEEPAGENTS_CODE_ANTHROPIC_API_KEY"] == "sk-shell"
            assert "openai_api_key: set -> set" in changes
        finally:
            config_mod._dotenv_loaded_values.clear()

    def test_reload_resets_prefixed_resolution_logging(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Runtime reload starts a new generation of resolution diagnostics."""
        import deepagents_code.config as config_mod
        from deepagents_code.model_config import (
            reset_env_resolution_log,
            resolve_env_var,
        )

        monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "sk-prefixed")
        monkeypatch.setattr(
            config_mod,
            "_GLOBAL_DOTENV_PATH",
            tmp_path / "missing-global.env",
        )
        caplog.set_level(logging.DEBUG, logger="deepagents_code.model_config")
        reset_env_resolution_log()
        try:
            runtime = Settings.from_environment(start_path=tmp_path)
            assert resolve_env_var("OPENAI_API_KEY") == "sk-prefixed"
            runtime.reload_from_environment(start_path=tmp_path)
            assert resolve_env_var("OPENAI_API_KEY") == "sk-prefixed"
            assert (
                caplog.messages.count(
                    "Resolved OPENAI_API_KEY from DEEPAGENTS_CODE_OPENAI_API_KEY"
                )
                == 2
            )
        finally:
            reset_env_resolution_log()

    def test_reload_redefaults_project_when_override_cleared_and_tracing_on(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Clearing the agent project on reload re-applies the default.

        Regression: when a cwd switch unsets `DEEPAGENTS_CODE_LANGSMITH_PROJECT`
        and the user has no original `LANGSMITH_PROJECT`, the reload must fall
        back to `deepagents-code` (not leave the var unset) so trace ingestion
        keeps matching the name `get_langsmith_project_name` displays.
        """
        import os

        import deepagents_code.config as config_mod
        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        current = tmp_path / "current"
        target = tmp_path / "target"
        current.mkdir()
        target.mkdir()

        monkeypatch.setattr(
            config_mod,
            "_GLOBAL_DOTENV_PATH",
            tmp_path / "missing-global.env",
        )
        config_mod._dotenv_loaded_values.clear()
        original_ls = config_mod._bootstrap_state.original_langsmith_project

        try:
            # User never set LANGSMITH_PROJECT; tracing is active with a key.
            config_mod._bootstrap_state.original_langsmith_project = None
            monkeypatch.setenv("LANGSMITH_TRACING", "true")
            monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
            monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
            # Agent-project override is active before the reload, cleared after.
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", "agent-project")

            runtime = Settings.from_environment(start_path=current)
            assert runtime.deepagents_langchain_project == "agent-project"

            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)
            runtime.reload_from_environment(start_path=target)

            assert os.environ["LANGSMITH_PROJECT"] == LANGSMITH_PROJECT_DEFAULT
        finally:
            config_mod._bootstrap_state.original_langsmith_project = original_ls
            config_mod._dotenv_loaded_values.clear()


class TestProjectRootDetection:
    """Test project root detection via .git directory."""

    def test_find_project_root_with_git(self, tmp_path: Path) -> None:
        """Test that project root is found when .git directory exists."""
        # Create a mock project structure
        project_root = tmp_path / "my-project"
        project_root.mkdir()
        git_dir = project_root / ".git"
        git_dir.mkdir()

        # Create a subdirectory to search from
        subdir = project_root / "src" / "components"
        subdir.mkdir(parents=True)

        # Should find project root from subdirectory
        result = _find_project_root(subdir)
        assert result == project_root

    def test_find_project_root_no_git(self, tmp_path: Path) -> None:
        """Test that None is returned when no .git directory exists."""
        # Create directory without .git
        no_git_dir = tmp_path / "no-git"
        no_git_dir.mkdir()

        result = _find_project_root(no_git_dir)
        assert result is None

    def test_find_project_root_nested_git(self, tmp_path: Path) -> None:
        """Test that nearest .git directory is found (not parent repos)."""
        # Create nested git repos
        outer_repo = tmp_path / "outer"
        outer_repo.mkdir()
        (outer_repo / ".git").mkdir()

        inner_repo = outer_repo / "inner"
        inner_repo.mkdir()
        (inner_repo / ".git").mkdir()

        # Should find inner repo, not outer
        result = _find_project_root(inner_repo)
        assert result == inner_repo

    def test_find_project_root_with_gitdir_file(self, tmp_path: Path) -> None:
        """Test that worktree-style `.git` files resolve to the worktree root."""
        repo = tmp_path / "repo"
        worktree = tmp_path / "worktree"
        nested = worktree / "src"
        git_dir = repo / ".git" / "worktrees" / "feature"

        nested.mkdir(parents=True)
        git_dir.mkdir(parents=True)
        (worktree / ".git").write_text(
            "gitdir: ../repo/.git/worktrees/feature\n",
            encoding="utf-8",
        )

        result = _find_project_root(nested)
        assert result == worktree


class TestGitMetadataLookup:
    """Tests for shared git metadata helpers."""

    def setup_method(self) -> None:
        """Clear git metadata caches between tests."""
        git_module._git_dir_cache.clear()

    def test_find_git_dir_reuses_cached_resolution(self, tmp_path: Path) -> None:
        """Repeated git-dir lookups should reuse the cached result."""
        repo = tmp_path / "repo"
        repo.mkdir()
        expected = repo / ".git"

        with patch(
            "deepagents_code._git._find_git_dir_uncached",
            return_value=expected,
        ) as mock_find:
            assert git_module.find_git_dir(repo) == expected
            assert git_module.find_git_dir(repo) == expected

        mock_find.assert_called_once()


class TestProjectContext:
    """Tests for explicit project context handling."""

    def test_from_user_cwd_uses_explicit_path_not_process_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Project context should resolve from the provided user cwd."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        user_cwd = project_root / "src"
        user_cwd.mkdir()

        other_cwd = tmp_path / "elsewhere"
        other_cwd.mkdir()
        monkeypatch.chdir(other_cwd)

        context = ProjectContext.from_user_cwd(user_cwd)

        assert context.user_cwd == user_cwd.resolve()
        assert context.project_root == project_root

    def test_get_server_project_context_from_env_mapping(self, tmp_path: Path) -> None:
        """Server context should reconstruct explicit cwd and project root."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        user_cwd = project_root / "src"
        user_cwd.mkdir()

        env = {
            f"{SERVER_ENV_PREFIX}CWD": str(user_cwd),
            f"{SERVER_ENV_PREFIX}PROJECT_ROOT": str(project_root),
        }
        context = get_server_project_context(env)

        assert context is not None
        assert context.user_cwd == user_cwd.resolve()
        assert context.project_root == project_root.resolve()


class TestProjectAgentMdFinding:
    """Test finding project-specific AGENTS.md files."""

    def test_find_agent_md_in_deepagents_dir(self, tmp_path: Path) -> None:
        """Test finding AGENTS.md in .deepagents/ directory."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        # Create .deepagents/AGENTS.md
        deepagents_dir = project_root / ".deepagents"
        deepagents_dir.mkdir()
        agent_md = deepagents_dir / "AGENTS.md"
        agent_md.write_text("Project instructions")

        result = _find_project_agent_md(project_root)
        assert len(result) == 1
        assert result[0] == agent_md

    def test_find_agent_md_in_root(self, tmp_path: Path) -> None:
        """Test finding AGENTS.md in project root (fallback)."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        # Create root-level AGENTS.md (no .deepagents/)
        agent_md = project_root / "AGENTS.md"
        agent_md.write_text("Project instructions")

        result = _find_project_agent_md(project_root)
        assert len(result) == 1
        assert result[0] == agent_md

    def test_both_agent_md_files_combined(self, tmp_path: Path) -> None:
        """Test that both AGENTS.md files are returned when both exist."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        # Create both locations
        deepagents_dir = project_root / ".deepagents"
        deepagents_dir.mkdir()
        deepagents_md = deepagents_dir / "AGENTS.md"
        deepagents_md.write_text("In .deepagents/")

        root_md = project_root / "AGENTS.md"
        root_md.write_text("In root")

        # Should return both, with .deepagents/ first
        result = _find_project_agent_md(project_root)
        assert len(result) == 2
        assert result[0] == deepagents_md
        assert result[1] == root_md

    def test_find_agent_md_not_found(self, tmp_path: Path) -> None:
        """Test that empty list is returned when no AGENTS.md exists."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        result = _find_project_agent_md(project_root)
        assert result == []

    def test_skips_paths_with_permission_errors(self, tmp_path: Path) -> None:
        """`OSError` from `Path.resolve()` is caught and the candidate is skipped."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        real_md = project_root / "AGENTS.md"
        real_md.write_text("root instructions")

        original_resolve = Path.resolve

        def patched_resolve(self: Path, *args: object, **kwargs: object) -> Path:
            if self.name == "AGENTS.md" and ".deepagents" in str(self):
                msg = "Permission denied"
                raise PermissionError(msg)
            return original_resolve(self, *args, **kwargs)  # ty: ignore

        with patch.object(Path, "resolve", patched_resolve):
            result = _find_project_agent_md(project_root)

        assert len(result) == 1
        assert result[0].samefile(real_md)
        assert not result[0].is_symlink()

    def test_in_tree_symlink_resolves_to_target(self, tmp_path: Path) -> None:
        """`AGENTS.md -> CLAUDE.md` returns a non-symlink path same-file as target."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        target = project_root / "CLAUDE.md"
        target.write_text("real instructions")

        link = project_root / "AGENTS.md"
        link.symlink_to(target)

        result = _find_project_agent_md(project_root)

        assert len(result) == 1
        # Returned path must be the resolved target — not the symlink — so
        # `FilesystemBackend.download_files` opens the regular file rather
        # than tripping `O_NOFOLLOW` on the link itself.
        assert not result[0].is_symlink()
        assert result[0].samefile(target)
        assert result[0].is_relative_to(project_root.resolve())

    def test_out_of_tree_symlink_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Symlink pointing outside project root is skipped with a warning."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        outside_target = tmp_path / "outside.md"
        outside_target.write_text("attacker-controlled")

        link = project_root / "AGENTS.md"
        link.symlink_to(outside_target)

        with caplog.at_level(logging.WARNING, logger="deepagents_code.project_utils"):
            result = _find_project_agent_md(project_root)

        assert result == []
        assert any("outside the project root" in r.getMessage() for r in caplog.records)

    def test_out_of_tree_parent_symlink_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Parent directory symlink cannot bypass project-root containment."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        outside_agent_md = outside / "AGENTS.md"
        outside_agent_md.write_text("attacker-controlled")

        (project_root / ".deepagents").symlink_to(outside, target_is_directory=True)

        with caplog.at_level(logging.WARNING, logger="deepagents_code.project_utils"):
            result = _find_project_agent_md(project_root)

        assert result == []
        assert any("outside the project root" in r.getMessage() for r in caplog.records)

    def test_broken_symlink_skipped(self, tmp_path: Path) -> None:
        """Symlink whose target does not exist is skipped without crashing."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        link = project_root / "AGENTS.md"
        link.symlink_to(project_root / "missing.md")

        result = _find_project_agent_md(project_root)

        # `Path.exists()` returns False for broken symlinks, so the candidate
        # is silently skipped — matches pre-existing behavior for absent files.
        assert result == []

    def test_symlink_loop_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Symlink loop is skipped with a warning instead of crashing the agent."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        a = project_root / "AGENTS.md"
        b = project_root / "loop.md"
        a.symlink_to(b)
        b.symlink_to(a)

        with caplog.at_level(logging.WARNING, logger="deepagents_code.project_utils"):
            result = _find_project_agent_md(project_root)

        assert result == []
        assert any(
            "Skipping AGENTS.md candidate" in r.getMessage() for r in caplog.records
        )

    def test_regular_file_unchanged_by_resolution(self, tmp_path: Path) -> None:
        """Regular (non-symlink) AGENTS.md returns a non-symlink, in-tree path."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        agent_md = project_root / "AGENTS.md"
        agent_md.write_text("plain file")

        result = _find_project_agent_md(project_root)

        assert len(result) == 1
        assert not result[0].is_symlink()
        assert result[0].samefile(agent_md)
        assert result[0].is_relative_to(project_root.resolve())

    def test_non_canonical_project_root_handled(self, tmp_path: Path) -> None:
        """Non-canonical `project_root` (symlinked ancestor) still locates AGENTS.md.

        Regression test: a symlinked `project_root` previously caused the
        regular-file candidate to fail the absolute-vs-resolved equality check
        and be returned as the canonical target rather than reported as missing.
        Pin behavior so that callers passing an uncanonicalized root (common
        when `Settings.project_root` originates from an unresolved cwd) still
        find a regular AGENTS.md.
        """
        real_root = tmp_path / "real"
        real_root.mkdir()
        agent_md = real_root / "AGENTS.md"
        agent_md.write_text("instructions")

        link_root = tmp_path / "link"
        link_root.symlink_to(real_root, target_is_directory=True)

        result = _find_project_agent_md(link_root)

        assert len(result) == 1
        assert not result[0].is_symlink()
        assert result[0].samefile(agent_md)
        assert result[0].is_relative_to(link_root.resolve())


class TestSettingsGetProjectAgentMdPath:
    """Test Settings.get_project_agent_md_path() integration."""

    def test_returns_empty_list_when_no_project_root(self) -> None:
        """Should return [] when project_root is None."""
        s = Settings.__new__(Settings)
        s.project_root = None
        assert s.get_project_agent_md_path() == []

    def test_returns_existing_paths(self, tmp_path: Path) -> None:
        """Should return existing AGENTS.md paths from project root."""
        deepagents_dir = tmp_path / ".deepagents"
        deepagents_dir.mkdir()
        deepagents_md = deepagents_dir / "AGENTS.md"
        deepagents_md.write_text("inner")

        root_md = tmp_path / "AGENTS.md"
        root_md.write_text("root")

        s = Settings.__new__(Settings)
        s.project_root = tmp_path

        result = s.get_project_agent_md_path()
        assert result == [deepagents_md, root_md]

    def test_returns_empty_when_no_agents_md_files(self, tmp_path: Path) -> None:
        """Should return [] when project exists but has no AGENTS.md."""
        s = Settings.__new__(Settings)
        s.project_root = tmp_path
        assert s.get_project_agent_md_path() == []


class TestNewlineShortcut:
    """Tests for newline shortcut labels.

    The label depends on both the platform and whether the attached
    terminal advertises kitty-keyboard-protocol support. Each test
    patches the cached capability probe so the platform-fallback logic
    is exercised in isolation.
    """

    def test_returns_shift_enter_when_kitty_supported(self) -> None:
        """Should show Shift+Enter on any platform when kitty kbd is negotiated."""
        with patch(
            "deepagents_code.terminal_capabilities.supports_kitty_keyboard_protocol",
            return_value=True,
        ):
            assert newline_shortcut() == "Shift+Enter"

    def test_returns_option_enter_on_macos(self) -> None:
        """Should show Option+Enter on darwin when kitty kbd is unavailable."""
        with (
            patch(
                "deepagents_code.terminal_capabilities.supports_kitty_keyboard_protocol",
                return_value=False,
            ),
            patch("deepagents_code.config.sys.platform", "darwin"),
        ):
            assert newline_shortcut() == "Option+Enter"

    def test_returns_ctrl_j_on_non_macos(self) -> None:
        """Should show Ctrl+J on non-darwin platforms when kitty kbd is unavailable."""
        with (
            patch(
                "deepagents_code.terminal_capabilities.supports_kitty_keyboard_protocol",
                return_value=False,
            ),
            patch("deepagents_code.config.sys.platform", "linux"),
        ):
            assert newline_shortcut() == "Ctrl+J"

    def test_returns_ctrl_j_on_win32(self) -> None:
        """Windows falls into the non-darwin branch and must show Ctrl+J."""
        with (
            patch(
                "deepagents_code.terminal_capabilities.supports_kitty_keyboard_protocol",
                return_value=False,
            ),
            patch("deepagents_code.config.sys.platform", "win32"),
        ):
            assert newline_shortcut() == "Ctrl+J"


class TestValidateModelCapabilities:
    """Tests for model capability validation."""

    @patch("deepagents_code.config.console")
    def test_model_without_profile_attribute_warns(self, mock_console: Mock) -> None:
        """Test that models without profile attribute trigger a warning."""
        model = Mock(spec=[])  # No profile attribute
        validate_model_capabilities(model, "test-model")

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "No capability profile" in call_args
        assert "test-model" in call_args

    @patch("deepagents_code.config.console")
    def test_model_with_none_profile_warns(self, mock_console: Mock) -> None:
        """Test that models with `profile=None` trigger a warning."""
        model = Mock()
        model.profile = None

        validate_model_capabilities(model, "test-model")

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "No capability profile" in call_args

    @patch("deepagents_code.config.console")
    def test_model_with_tool_calling_false_exits(self, mock_console: Mock) -> None:
        """Test that models with `tool_calling=False` cause `sys.exit(1)`."""
        model = Mock()
        model.profile = {"tool_calling": False}

        with pytest.raises(SystemExit) as exc_info:
            validate_model_capabilities(model, "no-tools-model")

        assert exc_info.value.code == 1
        # Verify error messages were printed
        assert mock_console.print.call_count == 3
        error_call = mock_console.print.call_args_list[0][0][0]
        assert "does not support tool calling" in error_call
        assert "no-tools-model" in error_call

    @patch("deepagents_code.config.console")
    def test_model_with_tool_calling_true_passes(self, mock_console: Mock) -> None:
        """Test that models with `tool_calling=True` pass without messages."""
        model = Mock()
        model.profile = {"tool_calling": True}

        validate_model_capabilities(model, "tools-model")

        mock_console.print.assert_not_called()

    @patch("deepagents_code.config.console")
    def test_model_with_tool_calling_none_passes(self, mock_console: Mock) -> None:
        """Test that models with `tool_calling=None` (missing) pass."""
        model = Mock()
        model.profile = {"other_capability": True}

        validate_model_capabilities(model, "model-without-tool-key")

        mock_console.print.assert_not_called()

    @patch("deepagents_code.config.console")
    def test_model_with_limited_context_warns(self, mock_console: Mock) -> None:
        """Test that models with <8000 token context trigger a warning."""
        model = Mock()
        model.profile = {"tool_calling": True, "max_input_tokens": 4096}

        validate_model_capabilities(model, "small-context-model")

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "limited context" in call_args
        assert "4,096" in call_args
        assert "small-context-model" in call_args

    @patch("deepagents_code.config.console")
    def test_model_with_adequate_context_passes(self, mock_console: Mock) -> None:
        """Confirm that models with >=8000 token context pass silently."""
        model = Mock()
        model.profile = {"tool_calling": True, "max_input_tokens": 128000}

        validate_model_capabilities(model, "large-context-model")

        mock_console.print.assert_not_called()

    @patch("deepagents_code.config.console")
    def test_model_without_max_input_tokens_passes(self, mock_console: Mock) -> None:
        """Test that models without `max_input_tokens` key pass silently."""
        model = Mock()
        model.profile = {"tool_calling": True}

        validate_model_capabilities(model, "no-context-info-model")

        mock_console.print.assert_not_called()

    @patch("deepagents_code.config.console")
    def test_model_with_zero_max_input_tokens_passes(self, mock_console: Mock) -> None:
        """Test that models with `max_input_tokens=0` pass (falsy value check)."""
        model = Mock()
        model.profile = {"tool_calling": True, "max_input_tokens": 0}

        validate_model_capabilities(model, "zero-context-model")

        # Should pass because 0 is falsy, so the condition `if max_input_tokens` fails
        mock_console.print.assert_not_called()

    @patch("deepagents_code.config.console")
    def test_model_with_empty_profile_passes(self, mock_console: Mock) -> None:
        """Test that models with empty profile dict pass silently."""
        model = Mock()
        model.profile = {}

        validate_model_capabilities(model, "empty-profile-model")

        mock_console.print.assert_not_called()


class TestAgentsAliasDirectories:
    """Tests for .agents directory alias methods."""

    def test_user_agents_dir(self) -> None:
        """Test user_agents_dir returns ~/.agents."""
        settings = Settings.from_environment()
        expected = Path.home() / ".agents"
        assert settings.user_agents_dir == expected

    def test_get_user_agent_skills_dir(self) -> None:
        """Test get_user_agent_skills_dir returns ~/.agents/skills."""
        settings = Settings.from_environment()
        expected = Path.home() / ".agents" / "skills"
        assert settings.get_user_agent_skills_dir() == expected

    def test_get_project_agent_skills_dir_with_project(self, tmp_path: Path) -> None:
        """Test get_project_agent_skills_dir returns .agents/skills in project."""
        # Create a mock project with .git
        project_root = tmp_path / "my-project"
        project_root.mkdir()
        (project_root / ".git").mkdir()

        settings = Settings.from_environment(start_path=project_root)
        expected = project_root / ".agents" / "skills"
        assert settings.get_project_agent_skills_dir() == expected

    def test_get_project_agent_skills_dir_without_project(self, tmp_path: Path) -> None:
        """Test get_project_agent_skills_dir returns None when not in a project."""
        # Create a directory without .git
        no_project = tmp_path / "no-project"
        no_project.mkdir()

        settings = Settings.from_environment(start_path=no_project)
        assert settings.get_project_agent_skills_dir() is None


class TestClaudeSkillsDirs:
    """Tests for .claude/skills/ directory methods."""

    def test_get_user_claude_skills_dir(self) -> None:
        """Test get_user_claude_skills_dir returns ~/.claude/skills."""
        expected = Path.home() / ".claude" / "skills"
        assert Settings.get_user_claude_skills_dir() == expected

    def test_get_project_claude_skills_dir_with_project(self, tmp_path: Path) -> None:
        """Test get_project_claude_skills_dir returns .claude/skills in project."""
        project_root = tmp_path / "my-project"
        project_root.mkdir()
        (project_root / ".git").mkdir()

        settings = Settings.from_environment(start_path=project_root)
        expected = project_root / ".claude" / "skills"
        assert settings.get_project_claude_skills_dir() == expected

    def test_project_claude_skills_dir_without_project(self, tmp_path: Path) -> None:
        """Test get_project_claude_skills_dir returns None outside a project."""
        no_project = tmp_path / "no-project"
        no_project.mkdir()

        settings = Settings.from_environment(start_path=no_project)
        assert settings.get_project_claude_skills_dir() is None


class TestCreateModelProfileExtraction:
    """Tests for profile extraction in create_model.

    These tests verify that create_model correctly extracts the context_limit
    from the model's profile attribute. We mock init_chat_model since create_model
    now uses it internally.
    """

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_extracts_context_limit_from_profile(
        self, mock_init_chat_model: Mock
    ) -> None:
        """Test that context_limit is extracted from model profile."""
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 200000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model

        result = create_model("anthropic:claude-sonnet-4-5")
        assert result.context_limit == 200000

    @patch("langchain.chat_models.init_chat_model")
    def test_handles_missing_profile_gracefully(
        self, mock_init_chat_model: Mock
    ) -> None:
        """Test that missing profile attribute leaves context_limit as None."""
        mock_model = Mock(spec=["invoke"])  # No profile attribute
        mock_init_chat_model.return_value = mock_model

        result = create_model("anthropic:claude-sonnet-4-5")
        assert result.context_limit is None

    @patch("langchain.chat_models.init_chat_model")
    def test_handles_none_profile(self, mock_init_chat_model: Mock) -> None:
        """Test that profile=None leaves context_limit as None."""
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        result = create_model("anthropic:claude-sonnet-4-5")
        assert result.context_limit is None

    @patch("langchain.chat_models.init_chat_model")
    def test_handles_non_dict_profile(self, mock_init_chat_model: Mock) -> None:
        """Test that non-dict profile is handled safely."""
        mock_model = Mock()
        mock_model.profile = "not a dict"
        mock_init_chat_model.return_value = mock_model

        result = create_model("anthropic:claude-sonnet-4-5")
        assert result.context_limit is None

    @patch("langchain.chat_models.init_chat_model")
    def test_handles_non_int_max_input_tokens(self, mock_init_chat_model: Mock) -> None:
        """Test that string max_input_tokens is ignored."""
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": "200000"}  # String, not int
        mock_init_chat_model.return_value = mock_model

        result = create_model("anthropic:claude-sonnet-4-5")
        assert result.context_limit is None

    @patch("langchain.chat_models.init_chat_model")
    def test_handles_missing_max_input_tokens_key(
        self, mock_init_chat_model: Mock
    ) -> None:
        """Test that profile without max_input_tokens key is handled."""
        mock_model = Mock()
        mock_model.profile = {"tool_calling": True}  # No max_input_tokens
        mock_init_chat_model.return_value = mock_model

        result = create_model("anthropic:claude-sonnet-4-5")
        assert result.context_limit is None

    @patch("langchain.chat_models.init_chat_model")
    def test_extracts_unsupported_modalities(self, mock_init_chat_model: Mock) -> None:
        """Test that explicitly False modality flags are extracted."""
        mock_model = Mock()
        mock_model.profile = {
            "max_input_tokens": 64000,
            "tool_calling": True,
            "image_inputs": False,
            "audio_inputs": False,
            "video_inputs": False,
            "pdf_inputs": False,
        }
        mock_init_chat_model.return_value = mock_model

        result = create_model("deepseek:deepseek-r1")
        assert result.unsupported_modalities == frozenset(
            {"image", "audio", "video", "pdf"}
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_supported_modalities_not_flagged(self, mock_init_chat_model: Mock) -> None:
        """Test that True modality flags produce empty unsupported set."""
        mock_model = Mock()
        mock_model.profile = {
            "max_input_tokens": 200000,
            "tool_calling": True,
            "image_inputs": True,
            "audio_inputs": True,
            "video_inputs": True,
            "pdf_inputs": True,
        }
        mock_init_chat_model.return_value = mock_model

        result = create_model("anthropic:claude-sonnet-4-5")
        assert result.unsupported_modalities == frozenset()

    @patch("langchain.chat_models.init_chat_model")
    def test_missing_modality_keys_not_flagged(
        self, mock_init_chat_model: Mock
    ) -> None:
        """Test that absent modality keys are not treated as unsupported."""
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 128000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model

        result = create_model("openai:gpt-5.5")
        assert result.unsupported_modalities == frozenset()

    @patch("langchain.chat_models.init_chat_model")
    def test_mixed_modality_flags(self, mock_init_chat_model: Mock) -> None:
        """Test partial modality support extraction."""
        mock_model = Mock()
        mock_model.profile = {
            "tool_calling": True,
            "image_inputs": True,
            "audio_inputs": False,
            "video_inputs": True,
            "pdf_inputs": False,
        }
        mock_init_chat_model.return_value = mock_model

        result = create_model("anthropic:claude-sonnet-4-5")
        assert result.unsupported_modalities == frozenset({"audio", "pdf"})

    @patch("langchain.chat_models.init_chat_model")
    def test_no_profile_leaves_modalities_empty(
        self, mock_init_chat_model: Mock
    ) -> None:
        """Test that missing profile produces empty unsupported set."""
        mock_model = Mock(spec=["invoke"])
        mock_init_chat_model.return_value = mock_model

        result = create_model("anthropic:claude-sonnet-4-5")
        assert result.unsupported_modalities == frozenset()


class TestCreateModelSplitCredentialWiring:
    """`create_model` wires the split-credential diagnostic in correctly."""

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )

    @pytest.fixture(autouse=True)
    def _isolate_openai_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "OPENAI_API_KEY",
            "DEEPAGENTS_CODE_OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
            "DEEPAGENTS_CODE_OPENAI_BASE_URL",
            "DEEPAGENTS_CODE_OPENAI_API_BASE",
        ):
            monkeypatch.delenv(var, raising=False)

    @patch("langchain.chat_models.init_chat_model")
    def test_create_model_emits_split_credential_warning(
        self,
        mock_init_chat_model: Mock,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A prefixed key + plain base URL surfaces the DEBUG diagnostic.

        Guards the call site itself: `TestSplitCredentialSource` only exercises
        the helper in isolation, so without this a dropped call would go unnoticed.
        """
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 128000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model

        monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "sk-secret-value")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")

        with caplog.at_level(logging.DEBUG, logger="deepagents_code.model_config"):
            create_model("openai:gpt-5.5")

        messages = [r.getMessage() for r in caplog.records]
        assert any(
            "DEEPAGENTS_CODE_OPENAI_API_KEY" in m and "OPENAI_BASE_URL" in m
            for m in messages
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_diagnostic_runs_before_apply_stored_credentials(
        self,
        mock_init_chat_model: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The diagnostic must observe raw env intent, i.e. run before the bridge.

        `apply_stored_credentials` rewrites the unprefixed base-URL env vars, so
        the ordering claimed by the call-site comment is load-bearing. Pin it by
        asserting the relative call order, which a reorder/removal would break.
        """
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 128000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model

        manager = Mock()
        monkeypatch.setattr(
            "deepagents_code.model_config.warn_on_split_credential_source",
            manager.warn,
        )
        monkeypatch.setattr(
            "deepagents_code.model_config.apply_stored_credentials",
            manager.apply,
        )

        create_model("openai:gpt-5.5")

        ordered = [name for name, _args, _kwargs in manager.mock_calls]
        assert ordered == ["warn", "apply"]


class TestModelResultApplyToSettings:
    """Tests for ModelResult.apply_to_settings propagation."""

    def test_propagates_unsupported_modalities(self) -> None:
        """Test that apply_to_settings writes unsupported_modalities to settings."""
        model_result = ModelResult(
            model=Mock(),
            model_name="deepseek-r1",
            provider="deepseek",
            context_limit=64000,
            unsupported_modalities=frozenset({"image", "audio"}),
        )
        original = settings.model_unsupported_modalities
        try:
            model_result.apply_to_settings()
            expected = frozenset({"image", "audio"})
            assert settings.model_unsupported_modalities == expected
        finally:
            settings.model_unsupported_modalities = original


class TestRetriesConfig:
    """Tests for `[retries]` config.toml support."""

    def test_read_retries_returns_none_when_section_absent(
        self, tmp_path: Path
    ) -> None:
        """Missing `[retries]` returns `None`."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[models]\ndefault = 'openai:gpt-5.5'\n")

        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            assert _read_config_toml_retries() is None

    def test_read_retries_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        """Missing config file returns `None`."""
        config_path = tmp_path / "config.toml"

        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            assert _read_config_toml_retries() is None

    def test_read_retries_returns_none_when_unreadable(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unreadable config returns `None` with a warning."""
        config_path = tmp_path / "config.toml"

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.object(Path, "open", side_effect=PermissionError("denied")),
            caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
        ):
            assert _read_config_toml_retries() is None

        assert "Could not read retries config" in caplog.text

    def test_read_retries_allows_unknown_provider_with_param(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unknown providers can opt into retries with an explicit param."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[retries.custom_provider]\nparam = 'max_retries'\n")

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
        ):
            assert _read_config_toml_retries() == {
                "custom_provider": {"param": "max_retries"}
            }

        assert "is not a known provider" not in caplog.text

    def test_read_retries_warns_unknown_provider_without_param(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unknown retry tables still warn when they cannot provide a kwarg."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[retries.custom_provider]\nmax_retries = 2\n")

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
        ):
            assert _read_config_toml_retries() == {
                "custom_provider": {"max_retries": 2}
            }

        assert "is not a known provider" in caplog.text

    def test_resolve_retry_kwargs_global(self) -> None:
        """Global retry config applies to supported providers."""
        assert _resolve_retry_kwargs({"max_retries": 2}, "fireworks") == {
            "max_retries": 2
        }

    @pytest.mark.parametrize(
        ("provider", "retry_param"),
        [
            ("anthropic", "max_retries"),
            ("azure_openai", "max_retries"),
            ("baseten", "max_retries"),
            ("bedrock", "max_retries"),
            ("deepseek", "max_retries"),
            ("fireworks", "max_retries"),
            ("google_genai", "max_retries"),
            ("google_vertexai", "max_retries"),
            ("groq", "max_retries"),
            ("litellm", "max_retries"),
            ("mistralai", "max_retries"),
            ("openai", "max_retries"),
            ("openrouter", "max_retries"),
            ("perplexity", "max_retries"),
            ("together", "max_retries"),
            ("xai", "max_retries"),
        ],
    )
    def test_resolve_retry_kwargs_registered_providers(
        self, provider: str, retry_param: str
    ) -> None:
        """Registered providers receive the retry kwarg their constructor expects."""
        assert _resolve_retry_kwargs({"max_retries": 2}, provider) == {retry_param: 2}

    def test_resolve_retry_kwargs_provider_override_wins(self) -> None:
        """Provider retry config beats the global value."""
        section = {"max_retries": 2, "fireworks": {"max_retries": 3}}
        assert _resolve_retry_kwargs(section, "fireworks") == {"max_retries": 3}

    def test_resolve_retry_kwargs_provider_only(self) -> None:
        """Provider retry config works without a global value."""
        assert _resolve_retry_kwargs(
            {"fireworks": {"max_retries": 3}}, "fireworks"
        ) == {"max_retries": 3}

    def test_resolve_retry_kwargs_custom_provider_param(self) -> None:
        """Custom providers can name the retry constructor kwarg."""
        section = {
            "max_retries": 2,
            "custom_provider": {"param": "retries", "max_retries": 4},
        }
        assert _resolve_retry_kwargs(section, "custom_provider") == {"retries": 4}

    def test_resolve_retry_kwargs_custom_provider_param_uses_global(self) -> None:
        """Custom provider param can use the global retry count."""
        section = {"max_retries": 2, "custom_provider": {"param": "retries"}}
        assert _resolve_retry_kwargs(section, "custom_provider") == {"retries": 2}

    def test_resolve_retry_kwargs_param_overrides_registry(self) -> None:
        """Known providers can override the registered retry kwarg name."""
        section = {"max_retries": 2, "fireworks": {"param": "retries"}}
        assert _resolve_retry_kwargs(section, "fireworks") == {"retries": 2}

    @pytest.mark.parametrize("value", [-1, 1.5, True, False, "3"])
    def test_resolve_retry_kwargs_invalid_values_warn(
        self, value: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Invalid retry values are ignored with a warning."""
        with caplog.at_level(logging.WARNING, logger="deepagents_code.config"):
            assert _resolve_retry_kwargs({"max_retries": value}, "fireworks") == {}

        assert "Ignoring [retries].max_retries" in caplog.text

    def test_resolve_retry_kwargs_unknown_provider_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unsupported providers do not receive retry kwargs."""
        with caplog.at_level(logging.WARNING, logger="deepagents_code.config"):
            assert _resolve_retry_kwargs({"max_retries": 2}, "custom_provider") == {}

        assert (
            "does not support a registered or configured retry parameter" in caplog.text
        )

    @pytest.mark.parametrize("value", ["max-retries", "class", "", 2, True])
    def test_resolve_retry_kwargs_invalid_param_warns(
        self, value: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Invalid retry kwarg names are ignored with a warning."""
        section = {"max_retries": 2, "custom_provider": {"param": value}}
        with caplog.at_level(logging.WARNING, logger="deepagents_code.config"):
            assert _resolve_retry_kwargs(section, "custom_provider") == {}

        assert "Ignoring [retries.custom_provider].param" in caplog.text
        assert (
            "does not support a registered or configured retry parameter" in caplog.text
        )

    def test_resolve_retry_kwargs_unknown_keys_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unknown retry keys are ignored with warnings."""
        section = {"max_retries": 2, "fireworks": {"other": 4}, "other": 5}
        with caplog.at_level(logging.WARNING, logger="deepagents_code.config"):
            assert _resolve_retry_kwargs(section, "fireworks") == {"max_retries": 2}

        assert "Ignoring [retries].other" in caplog.text
        assert "Ignoring [retries.fireworks].other" in caplog.text

    def test_get_provider_kwargs_includes_retries(self, tmp_path: Path) -> None:
        """Provider kwargs include retries from `[retries]`."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[retries.fireworks]\nmax_retries = 3\n")

        clear_caches()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            assert _get_provider_kwargs("fireworks")["max_retries"] == 3

    def test_get_provider_kwargs_params_beat_retries(self, tmp_path: Path) -> None:
        """Provider params keep precedence over `[retries]`."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            """
[models.providers.fireworks.params]
max_retries = 5

[retries.fireworks]
max_retries = 3
"""
        )

        clear_caches()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            assert _get_provider_kwargs("fireworks")["max_retries"] == 5

    def test_get_provider_kwargs_includes_global_retries(self, tmp_path: Path) -> None:
        """A global `[retries]` default reaches provider kwargs via setdefault."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[retries]\nmax_retries = 2\n")

        clear_caches()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            assert _get_provider_kwargs("fireworks")["max_retries"] == 2

    def test_resolve_retry_kwargs_zero_is_valid(self) -> None:
        """`max_retries = 0` is a valid count (disables retries)."""
        assert _resolve_retry_kwargs({"max_retries": 0}, "fireworks") == {
            "max_retries": 0
        }

    def test_resolve_retry_kwargs_provider_scalar_falls_back_to_global(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A non-table provider value warns and falls back to the global count."""
        section = {"max_retries": 2, "fireworks": 5}
        with caplog.at_level(logging.WARNING, logger="deepagents_code.config"):
            assert _resolve_retry_kwargs(section, "fireworks") == {"max_retries": 2}

        assert "expected table" in caplog.text

    def test_read_retries_returns_none_when_not_table(self, tmp_path: Path) -> None:
        """A scalar `retries` value (not a table) yields `None`."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("retries = 5\n")

        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            assert _read_config_toml_retries() is None

    def test_read_retries_returns_none_on_malformed_toml(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Malformed TOML returns `None` with a warning."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[retries\nmax_retries = 1\n")

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
        ):
            assert _read_config_toml_retries() is None

        assert "Could not read retries config" in caplog.text

    def test_read_retries_warns_unknown_provider_table(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A mistyped provider sub-table warns; a valid one does not."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            "[retries.fireworks]\nmax_retries = 3\n\n"
            "[retries.fireorks]\nmax_retries = 2\n"
        )

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
        ):
            section = _read_config_toml_retries()

        assert section is not None
        assert "'fireorks' is not a known provider" in caplog.text
        # The correctly spelled provider table must not be flagged.
        assert "[retries.fireworks]" not in caplog.text

    def test_read_retries_allows_bedrock_table(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Retry-capable providers without API-key env entries are still known."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[retries.bedrock]\nmax_retries = 3\n")

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
        ):
            section = _read_config_toml_retries()

        assert section == {"bedrock": {"max_retries": 3}}
        assert "is not a known provider" not in caplog.text


class TestResolveRetryParamName:
    """`_resolve_retry_param_name` picks the constructor kwarg for a provider."""

    def test_registered_provider_returns_mapped_name(self) -> None:
        """A registered provider resolves to its mapped kwarg."""
        assert _resolve_retry_param_name("openai") == "max_retries"

    def test_unknown_provider_defaults_to_max_retries(self) -> None:
        """An unregistered provider falls back to the universal `max_retries`."""
        assert _resolve_retry_param_name("some_unregistered_provider") == "max_retries"

    def test_config_param_override_wins_for_custom_provider(
        self, tmp_path: Path
    ) -> None:
        """`[retries.<provider>].param` names the kwarg for a custom provider."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[retries.custom]\nparam = "num_retries"\n')
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            assert _resolve_retry_param_name("custom") == "num_retries"

    def test_config_param_override_wins_for_registered_provider(
        self, tmp_path: Path
    ) -> None:
        """A configured `param` overrides even a registered provider's mapping."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[retries.openai]\nparam = "request_retries"\n')
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            assert _resolve_retry_param_name("openai") == "request_retries"

    def test_invalid_config_param_falls_back_to_registry(self, tmp_path: Path) -> None:
        """An invalid `param` value is ignored, falling back to the registry."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[retries.openai]\nparam = "not an identifier"\n')
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            assert _resolve_retry_param_name("openai") == "max_retries"


class TestCreateModelMaxRetries:
    """`create_model` folds the `--max-retries` sentinel into the constructor.

    The flag value rides `extra_kwargs` under `CLI_MAX_RETRIES_KEY`; these tests
    mock `init_chat_model` and assert on the kwargs forwarded to it.
    """

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_sentinel_folds_to_resolved_param(self, mock_init: Mock) -> None:
        """A registered provider receives the value under `max_retries`."""
        mock_init.return_value = Mock()
        create_model(
            "anthropic:claude-sonnet-4-5", extra_kwargs={CLI_MAX_RETRIES_KEY: 4}
        )
        kwargs = mock_init.call_args.kwargs
        assert kwargs["max_retries"] == 4
        # The internal carrier must never reach the constructor.
        assert CLI_MAX_RETRIES_KEY not in kwargs

    @patch("langchain.chat_models.init_chat_model")
    def test_sentinel_beats_model_params_max_retries(self, mock_init: Mock) -> None:
        """The CLI flag outranks a `max_retries` supplied via `--model-params`."""
        mock_init.return_value = Mock()
        create_model(
            "anthropic:claude-sonnet-4-5",
            extra_kwargs={CLI_MAX_RETRIES_KEY: 4, "max_retries": 1},
        )
        assert mock_init.call_args.kwargs["max_retries"] == 4

    @patch("langchain.chat_models.init_chat_model")
    def test_sentinel_folds_to_configured_param(
        self, mock_init: Mock, tmp_path: Path
    ) -> None:
        """A `[retries.<provider>].param` override redirects the folded kwarg."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[retries.anthropic]\nparam = "request_retries"\n')
        mock_init.return_value = Mock()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            create_model(
                "anthropic:claude-sonnet-4-5",
                extra_kwargs={CLI_MAX_RETRIES_KEY: 4},
            )
        kwargs = mock_init.call_args.kwargs
        assert kwargs["request_retries"] == 4
        assert CLI_MAX_RETRIES_KEY not in kwargs

    @patch("langchain.chat_models.init_chat_model")
    def test_zero_is_forwarded(self, mock_init: Mock) -> None:
        """`--max-retries 0` (disable retries) folds through, not dropped."""
        mock_init.return_value = Mock()
        create_model(
            "anthropic:claude-sonnet-4-5", extra_kwargs={CLI_MAX_RETRIES_KEY: 0}
        )
        assert mock_init.call_args.kwargs["max_retries"] == 0

    @patch("langchain.chat_models.init_chat_model")
    def test_caller_extra_kwargs_not_mutated(self, mock_init: Mock) -> None:
        """The caller's dict keeps the sentinel for reuse (runtime model switch)."""
        mock_init.return_value = Mock()
        extra = {CLI_MAX_RETRIES_KEY: 4}
        create_model("anthropic:claude-sonnet-4-5", extra_kwargs=extra)
        assert extra == {CLI_MAX_RETRIES_KEY: 4}


class TestCreateModelProfileOverrides:
    """Tests for profile overrides from config.toml in create_model."""

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_profile_override_sets_context_limit(
        self, mock_init_chat_model: Mock, tmp_path: Path
    ) -> None:
        """Profile override for max_input_tokens flows to context_limit."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic.profile]
max_input_tokens = 4096
""")
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 200000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model

        clear_caches()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            result = create_model("anthropic:claude-sonnet-4-5")

        assert result.context_limit == 4096

    @patch("langchain.chat_models.init_chat_model")
    def test_per_model_profile_override_takes_precedence(
        self, mock_init_chat_model: Mock, tmp_path: Path
    ) -> None:
        """Per-model profile override wins over provider-wide default."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic.profile]
max_input_tokens = 4096

[models.providers.anthropic.profile."claude-sonnet-4-5"]
max_input_tokens = 8192
""")
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 200000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model

        clear_caches()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            result = create_model("anthropic:claude-sonnet-4-5")

        assert result.context_limit == 8192

    @patch("langchain.chat_models.init_chat_model")
    def test_no_profile_override_preserves_original(
        self, mock_init_chat_model: Mock, tmp_path: Path
    ) -> None:
        """Without config overrides, original profile value is used."""
        config_path = tmp_path / "config.toml"  # Does not exist — empty config
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 200000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model

        clear_caches()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            result = create_model("anthropic:claude-sonnet-4-5")
        assert result.context_limit == 200000

    @patch("langchain.chat_models.init_chat_model")
    def test_profile_override_on_model_without_profile(
        self, mock_init_chat_model: Mock, tmp_path: Path
    ) -> None:
        """Profile override is applied even when model has no profile attr."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic.profile]
max_input_tokens = 4096
""")
        mock_model = Mock(spec=["invoke"])  # No profile attribute
        mock_init_chat_model.return_value = mock_model

        clear_caches()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            result = create_model("anthropic:claude-sonnet-4-5")

        assert result.context_limit == 4096

    @patch("langchain.chat_models.init_chat_model")
    def test_profile_override_preserves_non_overridden_keys(
        self, mock_init_chat_model: Mock, tmp_path: Path
    ) -> None:
        """Override merges into existing profile without dropping other keys."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic.profile]
max_input_tokens = 4096
""")
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 200000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model

        clear_caches()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            create_model("anthropic:claude-sonnet-4-5")

        assert mock_model.profile == {"max_input_tokens": 4096, "tool_calling": True}

    @patch("langchain.chat_models.init_chat_model")
    def test_profile_override_when_profile_is_none(
        self, mock_init_chat_model: Mock, tmp_path: Path
    ) -> None:
        """Override is applied when model.profile is explicitly None."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic.profile]
max_input_tokens = 4096
""")
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        clear_caches()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            result = create_model("anthropic:claude-sonnet-4-5")

        assert result.context_limit == 4096

    @patch("langchain.chat_models.init_chat_model")
    def test_profile_override_logs_warning_on_frozen_model(
        self,
        mock_init_chat_model: Mock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Graceful warning when model rejects attribute assignment."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic.profile]
max_input_tokens = 4096
""")
        mock_model = Mock()
        # Make .profile read return a dict but assignment raises
        type(mock_model).profile = property(
            fget=lambda _: {"max_input_tokens": 200000},
            fset=lambda _, __: (_ for _ in ()).throw(AttributeError("frozen")),
        )
        mock_init_chat_model.return_value = mock_model

        clear_caches()
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
        ):
            result = create_model("anthropic:claude-sonnet-4-5")

        assert any(
            "Could not apply" in r.message and "profile overrides" in r.message
            for r in caplog.records
        )
        # Falls back to original profile extraction
        assert result.context_limit == 200000


class TestCreateModelCLIProfileOverrides:
    """Tests for CLI --profile-override in create_model."""

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_cli_profile_override_sets_context_limit(
        self, mock_init_chat_model: Mock, tmp_path: Path
    ) -> None:
        """CLI profile override for max_input_tokens flows to context_limit."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("")  # empty config
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 200000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model

        clear_caches()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            result = create_model(
                "anthropic:claude-sonnet-4-5",
                profile_overrides={"max_input_tokens": 4096},
            )

        assert result.context_limit == 4096

    @patch("langchain.chat_models.init_chat_model")
    def test_cli_profile_override_beats_config_toml(
        self, mock_init_chat_model: Mock, tmp_path: Path
    ) -> None:
        """CLI --profile-override wins over config.toml profile."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic.profile]
max_input_tokens = 8192
""")
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 200000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model

        clear_caches()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            result = create_model(
                "anthropic:claude-sonnet-4-5",
                profile_overrides={"max_input_tokens": 4096},
            )

        # CLI (4096) beats config.toml (8192)
        assert result.context_limit == 4096

    @patch("langchain.chat_models.init_chat_model")
    def test_cli_profile_override_preserves_other_keys(
        self, mock_init_chat_model: Mock, tmp_path: Path
    ) -> None:
        """CLI override merges into profile without dropping other keys."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("")
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 200000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model

        clear_caches()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            create_model(
                "anthropic:claude-sonnet-4-5",
                profile_overrides={"max_input_tokens": 4096},
            )

        assert mock_model.profile == {"max_input_tokens": 4096, "tool_calling": True}

    @patch("langchain.chat_models.init_chat_model")
    def test_cli_profile_override_on_model_without_profile(
        self, mock_init_chat_model: Mock, tmp_path: Path
    ) -> None:
        """CLI override applied even when model has no profile attr."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("")
        mock_model = Mock(spec=["invoke"])
        mock_init_chat_model.return_value = mock_model

        clear_caches()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            result = create_model(
                "anthropic:claude-sonnet-4-5",
                profile_overrides={"max_input_tokens": 4096},
            )

        assert result.context_limit == 4096

    @patch("langchain.chat_models.init_chat_model")
    def test_cli_profile_override_raises_on_frozen_model(
        self,
        mock_init_chat_model: Mock,
        tmp_path: Path,
    ) -> None:
        """CLI --profile-override raises when model rejects assignment."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("")
        mock_model = Mock()
        type(mock_model).profile = property(
            fget=lambda _: {"max_input_tokens": 200000},
            fset=lambda _, __: (_ for _ in ()).throw(AttributeError("frozen")),
        )
        mock_init_chat_model.return_value = mock_model

        clear_caches()
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            pytest.raises(ModelConfigError, match="Could not apply CLI"),
        ):
            create_model(
                "anthropic:claude-sonnet-4-5",
                profile_overrides={"max_input_tokens": 4096},
            )


class TestParseShellAllowList:
    """Test parsing shell allow-list strings."""

    def test_none_input_returns_none(self) -> None:
        """Test that None input returns None."""
        result = parse_shell_allow_list(None)
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        """Test that empty string returns None."""
        result = parse_shell_allow_list("")
        assert result is None

    def test_recommended_only(self) -> None:
        """Test that 'recommended' returns the full recommended list."""
        result = parse_shell_allow_list("recommended")
        assert result == list(RECOMMENDED_SAFE_SHELL_COMMANDS)

    def test_recommended_case_insensitive(self) -> None:
        """Test that 'RECOMMENDED', 'Recommended', etc. all work."""
        for variant in ["RECOMMENDED", "Recommended", "ReCoMmEnDeD", "  recommended  "]:
            result = parse_shell_allow_list(variant)
            assert result == list(RECOMMENDED_SAFE_SHELL_COMMANDS)

    def test_custom_commands_only(self) -> None:
        """Test parsing custom commands without 'recommended'."""
        result = parse_shell_allow_list("ls,cat,grep")
        assert result == ["ls", "cat", "grep"]

    def test_custom_commands_with_whitespace(self) -> None:
        """Test parsing custom commands with whitespace."""
        result = parse_shell_allow_list("ls , cat , grep")
        assert result == ["ls", "cat", "grep"]

    def test_recommended_merged_with_custom_commands(self) -> None:
        """Test that 'recommended' in list merges with custom commands."""
        result = parse_shell_allow_list("recommended,mycmd,myothercmd")
        expected = [*list(RECOMMENDED_SAFE_SHELL_COMMANDS), "mycmd", "myothercmd"]
        assert result == expected

    def test_custom_commands_before_recommended(self) -> None:
        """Test custom commands before 'recommended' keyword."""
        result = parse_shell_allow_list("mycmd,recommended,myothercmd")
        # mycmd first, then all recommended, then myothercmd
        expected = ["mycmd", *list(RECOMMENDED_SAFE_SHELL_COMMANDS), "myothercmd"]
        assert result == expected

    def test_duplicate_removal(self) -> None:
        """Test that duplicates are removed while preserving order."""
        result = parse_shell_allow_list("ls,cat,ls,grep,cat")
        assert result == ["ls", "cat", "grep"]

    def test_duplicate_removal_with_recommended(self) -> None:
        """Test that duplicates from recommended are removed."""
        # 'ls' is in RECOMMENDED_SAFE_SHELL_COMMANDS
        result = parse_shell_allow_list("ls,recommended,mycmd")
        # Should have ls once (first occurrence), then all recommended commands
        # except ls (since it's already in), then mycmd
        assert result is not None
        assert result[0] == "ls"
        # ls should not appear again
        assert result.count("ls") == 1
        # mycmd should appear once at the end
        assert result[-1] == "mycmd"
        # Total should be: 1 (ls) + len(recommended) - 1 (duplicate ls) + 1 (mycmd)
        # Which simplifies to: len(recommended) + 1
        assert len(result) == len(RECOMMENDED_SAFE_SHELL_COMMANDS) + 1

    def test_all_returns_sentinel(self) -> None:
        """Test that 'all' returns SHELL_ALLOW_ALL sentinel."""
        result = parse_shell_allow_list("all")
        assert result is SHELL_ALLOW_ALL

    def test_all_case_insensitive(self) -> None:
        """Test that 'ALL', 'All', etc. all return sentinel."""
        for variant in ["ALL", "All", "aLl", "  all  "]:
            result = parse_shell_allow_list(variant)
            assert result is SHELL_ALLOW_ALL

    def test_all_mixed_with_commands_raises(self) -> None:
        """Combining 'all' with other commands should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot combine 'all'"):
            parse_shell_allow_list("all,ls")

    def test_all_mixed_case_insensitive_raises(self) -> None:
        """Combining 'ALL' with other commands should also raise."""
        with pytest.raises(ValueError, match="Cannot combine 'all'"):
            parse_shell_allow_list("ls,ALL,cat")

    def test_empty_commands_ignored(self) -> None:
        """Test that empty strings from split are ignored."""
        result = parse_shell_allow_list("ls,,cat,,,grep,")
        assert result == ["ls", "cat", "grep"]


class TestGetLangsmithProjectName:
    """Tests for get_langsmith_project_name()."""

    def test_returns_none_without_api_key(self) -> None:
        """Should return None when no LangSmith API key is set."""
        env = {
            "LANGSMITH_API_KEY": "",
            "LANGCHAIN_API_KEY": "",
            "DEEPAGENTS_CODE_LANGSMITH_API_KEY": "",
            "DEEPAGENTS_CODE_LANGCHAIN_API_KEY": "",
            "LANGSMITH_TRACING": "true",
        }
        with patch.dict("os.environ", env, clear=False):
            assert get_langsmith_project_name() is None

    def test_returns_none_without_tracing(self) -> None:
        """Should return None when tracing is not enabled."""
        env = {
            "LANGSMITH_API_KEY": "lsv2_test",
            "LANGSMITH_TRACING": "",
            "LANGCHAIN_TRACING_V2": "",
            "DEEPAGENTS_CODE_LANGSMITH_TRACING": "",
            "DEEPAGENTS_CODE_LANGCHAIN_TRACING_V2": "",
        }
        with patch.dict("os.environ", env, clear=False):
            assert get_langsmith_project_name() is None

    def test_returns_project_from_settings(self) -> None:
        """Should prefer settings.deepagents_langchain_project."""
        env = {
            "LANGSMITH_API_KEY": "lsv2_test",
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_PROJECT": "env-project",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch("deepagents_code.config.settings") as mock_settings,
        ):
            mock_settings.deepagents_langchain_project = "settings-project"
            assert get_langsmith_project_name() == "settings-project"

    def test_falls_back_to_env_project(self) -> None:
        """Should fall back to LANGSMITH_PROJECT env var."""
        env = {
            "LANGSMITH_API_KEY": "lsv2_test",
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_PROJECT": "env-project",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch("deepagents_code.config.settings") as mock_settings,
        ):
            mock_settings.deepagents_langchain_project = None
            assert get_langsmith_project_name() == "env-project"

    def test_falls_back_to_default(self) -> None:
        """Should fall back to the default project when none is configured."""
        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        env = {
            "LANGSMITH_API_KEY": "lsv2_test",
            "LANGSMITH_TRACING": "true",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch("deepagents_code.config.settings") as mock_settings,
        ):
            mock_settings.deepagents_langchain_project = None
            assert get_langsmith_project_name() == LANGSMITH_PROJECT_DEFAULT

    def test_accepts_langchain_api_key(self) -> None:
        """Should accept LANGCHAIN_API_KEY as alternative to LANGSMITH_API_KEY."""
        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        env = {
            "LANGSMITH_API_KEY": "",
            "LANGCHAIN_API_KEY": "lsv2_test",
            "LANGSMITH_TRACING": "true",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch("deepagents_code.config.settings") as mock_settings,
        ):
            mock_settings.deepagents_langchain_project = None
            assert get_langsmith_project_name() == LANGSMITH_PROJECT_DEFAULT

    def test_agrees_with_config_manifest_resolution(self) -> None:
        """`get_langsmith_project_name` and `resolve_scalar` agree on the project.

        The `fallback_env_vars` mechanism exists so `config`/`config get` report
        the project agent traces actually route to. This pins that parity for
        the bare-env and unset cases, catching future drift between the two
        resolution paths.
        """
        from deepagents_code.config_manifest import (
            LANGSMITH_PROJECT_DEFAULT,
            get_option,
            resolve_scalar,
        )

        opt = get_option("tracing.langsmith_project")
        assert opt is not None

        # Bare `LANGSMITH_PROJECT` set, no prefixed override, no settings value.
        bare_env = {
            "LANGSMITH_API_KEY": "lsv2_test",
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_PROJECT": "parity-bare",
            "DEEPAGENTS_CODE_LANGSMITH_PROJECT": "",
        }
        with (
            patch.dict("os.environ", bare_env, clear=False),
            patch("deepagents_code.config.settings") as mock_settings,
        ):
            mock_settings.deepagents_langchain_project = None
            manifest_value, _ = resolve_scalar(opt, toml_data={})
            assert get_langsmith_project_name() == manifest_value == "parity-bare"

        # Nothing configured: both fall back to the shared default.
        default_env = {
            "LANGSMITH_API_KEY": "lsv2_test",
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_PROJECT": "",
            "DEEPAGENTS_CODE_LANGSMITH_PROJECT": "",
        }
        with (
            patch.dict("os.environ", default_env, clear=False),
            patch("deepagents_code.config.settings") as mock_settings,
        ):
            mock_settings.deepagents_langchain_project = None
            manifest_value, _ = resolve_scalar(opt, toml_data={})
            assert (
                get_langsmith_project_name()
                == manifest_value
                == LANGSMITH_PROJECT_DEFAULT
            )


class TestLangsmithKeyShadowedByEmptyOverride:
    """Tests for langsmith_key_shadowed_by_empty_override()."""

    @pytest.fixture(autouse=True)
    def _clear_tracing_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Start each test from a clean slate for the four tracing key vars.

        Every case below sets only the vars it cares about; clearing the rest
        up front keeps results independent of whatever the test runner happens
        to have exported (e.g. a developer's own empty override).
        """
        for var in (
            "LANGSMITH_API_KEY",
            "LANGCHAIN_API_KEY",
            "DEEPAGENTS_CODE_LANGSMITH_API_KEY",
            "DEEPAGENTS_CODE_LANGCHAIN_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)

    @pytest.fixture
    def fake_state_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Redirect the credential store into a temp directory."""
        state = tmp_path / ".state"
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_STATE_DIR", state)
        return state

    def test_returns_none_without_empty_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No empty prefixed override means nothing is being shadowed."""
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult()

    def test_returns_none_when_override_shadows_nothing(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty override with no underlying key is not a shadow."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult()

    def test_detects_shadowed_canonical_env_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty override shadowing a canonical env key is reported."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "")
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult(
            shadowing_var="DEEPAGENTS_CODE_LANGSMITH_API_KEY"
        )

    def test_detects_shadowed_stored_key(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty override shadowing a `/auth`-stored key is reported."""
        from deepagents_code import auth_store

        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "")
        auth_store.set_stored_key("langsmith", "lsv2_test")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult(
            shadowing_var="DEEPAGENTS_CODE_LANGSMITH_API_KEY"
        )

    def test_langchain_override_does_not_consult_the_stored_key(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The stored-key bridge is `LANGSMITH`-only, so `LANGCHAIN` ignores it.

        `/auth` only bridges its stored key onto `LANGSMITH_API_KEY`, never
        `LANGCHAIN_API_KEY`. An empty `LANGCHAIN` override with no canonical
        `LANGCHAIN_API_KEY` therefore shadows nothing even when a key is stored,
        so no shadow is reported. Pins the asymmetry against a future change that
        wrongly makes the `LANGCHAIN` path consult the store.
        """
        from deepagents_code import auth_store

        monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "")
        auth_store.set_stored_key("langsmith", "lsv2_test")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult()

    def test_returns_none_when_override_carries_a_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-empty prefixed override resolves normally, so no shadow."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_override")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult()

    def test_detects_shadowed_langchain_canonical_env_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The legacy `LANGCHAIN_API_KEY` override path is reported too."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult(
            shadowing_var="DEEPAGENTS_CODE_LANGCHAIN_API_KEY"
        )

    def test_reports_the_override_that_actually_shadows_the_key(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With both overrides empty, name the one hiding the only real key.

        The `LANGSMITH` override is empty but shadows nothing (no canonical
        value, no stored key); only `LANGCHAIN_API_KEY` carries a key, so
        unsetting the `LANGCHAIN` override -- not the `LANGSMITH` one -- is what
        restores tracing. The hint must name the `LANGCHAIN` override.
        """
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult(
            shadowing_var="DEEPAGENTS_CODE_LANGCHAIN_API_KEY"
        )

    def test_prefers_langsmith_when_both_overrides_shadow_a_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When both overrides genuinely shadow a key, `LANGSMITH` wins."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "")
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult(
            shadowing_var="DEEPAGENTS_CODE_LANGSMITH_API_KEY"
        )

    def test_ignores_empty_override_when_a_key_already_resolves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty override is not reported when a key resolves anyway.

        `LANGSMITH_API_KEY` resolves fine; the empty `LANGCHAIN` override hides
        no key (no canonical `LANGCHAIN_API_KEY`). Tracing may still be off for
        an unrelated reason (e.g. a missing tracing flag), but this override is
        not the cause, so the generic hint -- not a false shadow claim -- is
        correct.
        """
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "")
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult()

    def test_ignores_lower_precedence_override_when_langsmith_key_resolves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A resolving `LANGSMITH` key silences a genuine `LANGCHAIN` shadow.

        `LANGSMITH_API_KEY` resolves and wins under precedence, so the effective
        key is present and unsetting the empty `LANGCHAIN` override (which does
        shadow `LANGCHAIN_API_KEY`) would change nothing. Reporting it would send
        the user to unset the wrong variable, so nothing is reported.
        """
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_langsmith")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_langchain")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult()

    def test_reports_store_unreadable_when_no_other_shadow_found(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A corrupt store with no other shadow surfaces `store_unreadable`.

        The warning must carry the underlying exception text (not a static
        guess) so logs point at the real fault.
        """
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "")
        with (
            patch(
                "deepagents_code.auth_store.get_stored_key",
                side_effect=RuntimeError("bad json at line 3"),
            ),
            caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
        ):
            assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult(
                store_unreadable=True
            )
        messages = [r.getMessage() for r in caplog.records]
        assert any("empty-override shadow" in m for m in messages)
        assert any("bad json at line 3" in m for m in messages)

    def test_unreadable_store_does_not_abort_scan_of_later_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A store error on `LANGSMITH` must not stop the `LANGCHAIN` check.

        `LANGSMITH`'s stored-key read raises, but an empty `LANGCHAIN` override
        genuinely shadows a canonical `LANGCHAIN_API_KEY`. That concrete shadow
        is the actionable answer and must win over the store uncertainty, which
        proves the loop continued past the exception rather than bailing.
        """
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test")
        with patch(
            "deepagents_code.auth_store.get_stored_key",
            side_effect=RuntimeError("corrupt"),
        ):
            assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult(
                shadowing_var="DEEPAGENTS_CODE_LANGCHAIN_API_KEY"
            )


class TestLangsmithSecretRedaction:
    """Tests for LangSmith trace secret redaction configuration."""

    @pytest.fixture(autouse=True)
    def _enable_redaction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_REDACT", "true")

    def test_redaction_disabled_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """LangSmith trace redaction defaults to disabled."""
        monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_REDACT")
        with patch("deepagents_code.config_manifest.load_config_toml", return_value={}):
            assert is_langsmith_redaction_enabled() is False

    def test_redaction_can_be_enabled_by_env(self) -> None:
        """The redaction env var can opt in to secret redaction."""
        with patch("deepagents_code.config_manifest.load_config_toml", return_value={}):
            assert is_langsmith_redaction_enabled() is True

    def test_configures_langsmith_client_with_secret_anonymizer(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Active tracing installs a client whose anonymizer scrubs secrets."""
        client = object()
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_test")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")

        # Exercise the real `create_secret_anonymizer` (network-free) so the test
        # fails if the installed anonymizer does not actually redact secrets.
        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch("langsmith.Client", return_value=client) as client_cls,
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is True

        configure.assert_called_once_with(client=client)
        _, kwargs = client_cls.call_args
        assert kwargs["api_key"] == "lsv2_test"
        assert "api_url" not in kwargs
        secret = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijkl"
        redacted = str(kwargs["anonymizer"]([{"text": f"key={secret}"}]))
        assert secret not in redacted
        assert "[SECRET_DETECTED]" in redacted

    def test_skips_client_configuration_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tracing leaves the LangSmith client untouched until redaction is enabled."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_test")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")
        monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_REDACT")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch("langsmith.Client") as client_cls,
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is False

        client_cls.assert_not_called()
        configure.assert_not_called()

    def test_skips_when_tracing_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Redaction is skipped when tracing is inactive, even if credentialed."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_test")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch("langsmith.Client") as client_cls,
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is False

        client_cls.assert_not_called()
        configure.assert_not_called()

    def test_skips_when_no_credentials(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Redaction is skipped when tracing is active but uncredentialed."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            # No env key is set; ignore any LangSmith profile on the dev machine
            # so the no-credentials branch is exercised hermetically.
            patch(
                "deepagents_code.config._has_langsmith_profile_credentials",
                return_value=False,
            ),
            patch("langsmith.Client") as client_cls,
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is False

        client_cls.assert_not_called()
        configure.assert_not_called()

    def test_falls_back_to_langchain_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The legacy LANGCHAIN_API_KEY is used when LANGSMITH_API_KEY is absent."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "lsv2_legacy")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch("langsmith.Client", return_value=object()) as client_cls,
            patch("langsmith.configure"),
        ):
            assert configure_langsmith_secret_redaction() is True

        _, kwargs = client_cls.call_args
        assert kwargs["api_key"] == "lsv2_legacy"

    def test_forwards_custom_endpoint_as_api_url(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A custom tracing endpoint is forwarded to the client as `api_url`."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_test")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://eu.smith.example.com")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch("langsmith.Client", return_value=object()) as client_cls,
            patch("langsmith.configure"),
        ):
            assert configure_langsmith_secret_redaction() is True

        _, kwargs = client_cls.call_args
        assert kwargs["api_url"] == "https://eu.smith.example.com"

    def test_configures_client_for_keyless_custom_endpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Keyless custom endpoints still get the secret anonymizer installed."""
        client = object()
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "http://localhost:1984")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch(
                "deepagents_code.config._has_langsmith_profile_credentials",
                return_value=False,
            ),
            patch("langsmith.Client", return_value=client) as client_cls,
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is True

        configure.assert_called_once_with(client=client)
        _, kwargs = client_cls.call_args
        assert "api_key" not in kwargs
        assert kwargs["api_url"] == "http://localhost:1984"
        assert "anonymizer" in kwargs

    def test_configures_client_for_runs_endpoints(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Replica trace endpoints still get the secret anonymizer installed."""
        client = object()
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")
        monkeypatch.setenv(
            "LANGSMITH_RUNS_ENDPOINTS",
            '[{"api_url":"https://replica.example.com","api_key":"lsv2_replica"}]',
        )

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch(
                "deepagents_code.config._has_langsmith_profile_credentials",
                return_value=False,
            ),
            patch("langsmith.Client", return_value=client) as client_cls,
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is True

        configure.assert_called_once_with(client=client)
        _, kwargs = client_cls.call_args
        assert "api_key" not in kwargs
        assert "api_url" not in kwargs
        assert "anonymizer" in kwargs

    @pytest.mark.parametrize(
        "value",
        ["[]", '[{"api_url":"https://replica.example.com"}]', "not json"],
    )
    def test_skips_invalid_runs_endpoints(
        self,
        value: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Only valid replica endpoint configs count as upload targets."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_RUNS_ENDPOINTS", value)

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch(
                "deepagents_code.config._has_langsmith_profile_credentials",
                return_value=False,
            ),
            patch("langsmith.Client") as client_cls,
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is False

        client_cls.assert_not_called()
        configure.assert_not_called()

    def test_fails_closed_by_disabling_tracing_on_setup_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A redaction setup failure disables tracing to avoid leaking secrets."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_test")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch("langsmith.Client", side_effect=RuntimeError("boom")),
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is False

        configure.assert_called_once_with(enabled=False)

    def test_configures_client_with_profile_only_credentials(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Profile-only credentials (no env key) still install the anonymizer."""
        client = object()
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            # Credentials come only from an active LangSmith profile, not the env.
            # No endpoint either, so the profile credentials are the sole reason
            # the upload gate passes. The SDK client self-resolves that profile
            # auth when no key is forwarded.
            patch(
                "deepagents_code.config._has_langsmith_profile_credentials",
                return_value=True,
            ),
            patch("deepagents_code.config._tracing_endpoint_from", return_value=None),
            patch("langsmith.Client", return_value=client) as client_cls,
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is True

        configure.assert_called_once_with(client=client)
        _, kwargs = client_cls.call_args
        assert "api_key" not in kwargs
        assert "anonymizer" in kwargs

    def test_redaction_can_be_enabled_by_toml(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A `[tracing] langsmith_redact = true` in config.toml opts in."""
        monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_REDACT")
        with patch(
            "deepagents_code.config_manifest.load_config_toml",
            return_value={"tracing": {"langsmith_redact": True}},
        ):
            assert is_langsmith_redaction_enabled() is True

    def test_env_redaction_toggle_overrides_toml(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The redaction env var takes precedence over a conflicting config.toml."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_REDACT", "false")
        with patch(
            "deepagents_code.config_manifest.load_config_toml",
            return_value={"tracing": {"langsmith_redact": True}},
        ):
            assert is_langsmith_redaction_enabled() is False

    def test_fail_closed_clears_env_when_sdk_disable_also_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the SDK cannot disable tracing, all enable env vars are cleared.

        This is the worst-case fail-closed path: redaction setup raised and the
        SDK's `configure(enabled=False)` raised too, so the only remaining
        barrier is removing every env var the LangChain tracer falls back to —
        both the canonical names and their `DEEPAGENTS_CODE_`-prefixed forms.
        """
        import os

        from deepagents_code.config import _TRACING_ENABLE_ENV_VARS
        from deepagents_code.model_config import _ENV_PREFIX

        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
        enable_vars = [
            *_TRACING_ENABLE_ENV_VARS,
            *(f"{_ENV_PREFIX}{var}" for var in _TRACING_ENABLE_ENV_VARS),
        ]
        for var in enable_vars:
            monkeypatch.setenv(var, "true")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch("langsmith.Client", side_effect=RuntimeError("boom")),
            patch("langsmith.configure", side_effect=RuntimeError("nope")),
        ):
            assert configure_langsmith_secret_redaction() is False

        for var in enable_vars:
            assert var not in os.environ, f"{var} should have been cleared"

    def test_fails_closed_when_redaction_toggle_lookup_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unexpected error from the redaction-toggle lookup fails closed.

        `is_langsmith_redaction_enabled()` runs inside the fail-closed boundary,
        so even an unexpected exception there disables tracing rather than
        escaping the function and leaving tracing live but unredacted.
        """
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_test")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")

        with (
            patch(
                "deepagents_code.config.is_langsmith_redaction_enabled",
                side_effect=RuntimeError("boom"),
            ),
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is False

        configure.assert_called_once_with(enabled=False)

    def test_reconfigures_on_each_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each call reinstalls the redacting client (no fail-open caching)."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_test")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch("langsmith.Client", return_value=object()),
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is True
            assert configure_langsmith_secret_redaction() is True

        assert configure.call_count == 2


class TestDisableOrphanedTracing:
    """Tests for _disable_orphaned_tracing()."""

    _ALL_TRACING_VARS = (
        "LANGSMITH_TRACING_V2",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_ENDPOINT",
        "LANGCHAIN_ENDPOINT",
        "LANGSMITH_RUNS_ENDPOINTS",
        "LANGCHAIN_RUNS_ENDPOINTS",
        "LANGSMITH_CONFIG_FILE",
        "LANGSMITH_PROFILE",
    )

    def _clean_env(self) -> dict[str, str]:
        consume_orphaned_tracing_disabled_notice()
        env = dict.fromkeys(self._ALL_TRACING_VARS, "")
        env["LANGSMITH_CONFIG_FILE"] = "/__deepagents_missing_langsmith_config__.json"
        return env

    def test_disables_tracing_when_no_key(self) -> None:
        """Tracing flag on with empty key should be turned off."""
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
            assert consume_orphaned_tracing_disabled_notice() is not None
            # One-shot: the notice clears on read, so a second read is empty.
            assert consume_orphaned_tracing_disabled_notice() is None

    def test_notice_mentions_langsmith_auth_login_when_cli_available(self) -> None:
        """The startup notice gives the CLI login command only when available."""
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        with (
            patch.dict("os.environ", env, clear=False),
            patch("deepagents_code.config.shutil.which", return_value="/bin/langsmith"),
        ):
            _disable_orphaned_tracing()

        notice = consume_orphaned_tracing_disabled_notice()
        assert notice is not None
        assert "langsmith auth login" in notice

    def test_notice_omits_langsmith_auth_login_when_cli_unavailable(self) -> None:
        """The startup notice avoids unavailable CLI commands."""
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        with (
            patch.dict("os.environ", env, clear=False),
            patch("deepagents_code.config.shutil.which", return_value=None),
        ):
            _disable_orphaned_tracing()

        notice = consume_orphaned_tracing_disabled_notice()
        assert notice is not None
        assert "langsmith auth login" not in notice
        assert "LANGSMITH_API_KEY" in notice

    def test_preserves_tracing_when_custom_endpoint_set(self) -> None:
        """A custom endpoint (self-hosted/proxied) is trusted even without a key.

        Keyless ingestion is valid against a self-hosted LangSmith, so an
        explicitly configured endpoint must not trip the orphaned-tracing guard.
        """
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        env["LANGSMITH_ENDPOINT"] = "http://localhost:1984"
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
            # Nothing was disabled, so no startup notice should be staged.
            assert consume_orphaned_tracing_disabled_notice() is None

    def test_preserves_tracing_when_runs_endpoints_set(self) -> None:
        """Replica endpoints are trusted upload targets even without a top-level key."""
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        env["LANGSMITH_RUNS_ENDPOINTS"] = (
            '[{"api_url":"https://replica.example.com","api_key":"lsv2_replica"}]'
        )
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
            assert consume_orphaned_tracing_disabled_notice() is None

    def test_preserves_tracing_when_profile_custom_endpoint_set(
        self, tmp_path: Path
    ) -> None:
        """Profile api_url is a custom endpoint and is trusted without a key."""
        config = tmp_path / "config.json"
        config.write_text(
            "{"
            '"current_profile":"default",'
            '"profiles":{"default":{"api_url":"http://localhost:1984"}}'
            "}",
            encoding="utf-8",
        )
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        env["LANGSMITH_CONFIG_FILE"] = str(config)
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
            # Nothing was disabled, so no startup notice should be staged.
            assert consume_orphaned_tracing_disabled_notice() is None

    def test_preserves_tracing_when_key_present(self) -> None:
        """Tracing stays enabled when a usable API key is set."""
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        env["LANGSMITH_API_KEY"] = "lsv2_test"
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
            # No tracing was disabled, so no startup notice should be staged.
            assert consume_orphaned_tracing_disabled_notice() is None

    def test_accepts_langchain_api_key(self) -> None:
        """LANGCHAIN_API_KEY also counts as a usable key."""
        env = self._clean_env()
        env["LANGSMITH_TRACING"] = "true"
        env["LANGCHAIN_API_KEY"] = "lsv2_test"
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGSMITH_TRACING"] == "true"

    def test_preserves_tracing_when_profile_api_key_present(
        self, tmp_path: Path
    ) -> None:
        """LangSmith profile API keys count as usable credentials."""
        config = tmp_path / "config.json"
        config.write_text(
            '{"current_profile":"default","profiles":{"default":{"api_key":"lsv2_profile"}}}',
            encoding="utf-8",
        )
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        env["LANGSMITH_CONFIG_FILE"] = str(config)
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "true"

    def test_preserves_tracing_when_profile_oauth_present(self, tmp_path: Path) -> None:
        """LangSmith profile OAuth credentials count as usable credentials."""
        config = tmp_path / "config.json"
        config.write_text(
            "{"
            '"current_profile":"default",'
            '"profiles":{"default":{"oauth":{"refresh_token":"refresh"}}}'
            "}",
            encoding="utf-8",
        )
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        env["LANGSMITH_CONFIG_FILE"] = str(config)
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "true"

    def test_noop_when_tracing_disabled(self) -> None:
        """Does nothing when no tracing flag is enabled."""
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "false"
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "false"

    def test_disables_all_set_tracing_flags(self) -> None:
        """Every set tracing flag is turned off, not just one."""
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        env["LANGSMITH_TRACING"] = "1"
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
            assert os.environ["LANGSMITH_TRACING"] == "false"


class TestApplyStoredLangSmithTracing:
    """Tests for _apply_stored_langsmith_tracing()."""

    @pytest.fixture
    def fake_state_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Redirect the credential store into a temp directory."""
        state = tmp_path / ".state"
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_STATE_DIR", state)
        return state

    def test_noop_without_stored_key(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No stored key leaves the environment untouched (no auto-enable)."""
        import os

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        _apply_stored_langsmith_tracing()
        assert "LANGSMITH_TRACING" not in os.environ

    def test_enables_tracing_when_key_stored(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stored LangSmith key turns tracing on by default."""
        import os

        from deepagents_code import auth_store

        for var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
            monkeypatch.delenv(var, raising=False)
        auth_store.set_stored_key("langsmith", "lsv2_test")
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGSMITH_TRACING"] == "true"

    def test_respects_explicit_opt_out(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An explicit falsy tracing flag is honored as a temporary opt-out."""
        import os

        from deepagents_code import auth_store

        monkeypatch.setenv("LANGSMITH_TRACING", "false")
        auth_store.set_stored_key("langsmith", "lsv2_test")
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGSMITH_TRACING"] == "false"

    def test_scoped_opt_out_disables_sibling_tracing_flags(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The documented scoped opt-out wins over other truthy tracing flags."""
        import os

        from deepagents_code import auth_store

        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "false")
        monkeypatch.setenv("LANGSMITH_TRACING", "false")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        auth_store.set_stored_key("langsmith", "lsv2_test")
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGSMITH_TRACING"] == "false"
        assert os.environ["LANGCHAIN_TRACING_V2"] == "false"

    def test_leaves_explicit_enable_untouched(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An already-truthy tracing flag is left as-is."""
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        auth_store.set_stored_key("langsmith", "lsv2_test")
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert "LANGSMITH_TRACING" not in os.environ

    def test_applies_stored_custom_project(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stored custom project is applied when none is already set."""
        import os

        from deepagents_code import auth_store

        for var in ("LANGSMITH_TRACING", "LANGSMITH_PROJECT"):
            monkeypatch.delenv(var, raising=False)
        auth_store.set_stored_key("langsmith", "lsv2_test", project="my-app")
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGSMITH_PROJECT"] == "my-app"

    def test_stored_project_does_not_override_env(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An explicit LANGSMITH_PROJECT wins over the stored project."""
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.setenv("LANGSMITH_PROJECT", "from-env")
        auth_store.set_stored_key("langsmith", "lsv2_test", project="my-app")
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGSMITH_PROJECT"] == "from-env"

    def test_replace_project_applies_stored_project_over_existing_env(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Immediate `/auth` save applies the latest stored project."""
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.setenv("LANGSMITH_PROJECT", "old-project")
        auth_store.set_stored_key("langsmith", "lsv2_test", project="my-app")
        _apply_stored_langsmith_tracing(replace_project=True)
        assert os.environ["LANGSMITH_PROJECT"] == "my-app"

    def test_replace_project_clears_existing_env_when_project_removed(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Immediate `/auth` save clears the old project when the field is blank."""
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.setenv("LANGSMITH_PROJECT", "old-project")
        auth_store.set_stored_key("langsmith", "lsv2_test")
        _apply_stored_langsmith_tracing(replace_project=True)
        assert "LANGSMITH_PROJECT" not in os.environ

    def test_immediate_auth_clear_restores_default_project(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Clearing `/auth` project updates the active traced session."""
        import os

        from deepagents_code import auth_store
        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        redaction_env: list[dict[str, str]] = []

        def capture_redaction_env() -> bool:
            redaction_env.append(dict(os.environ))
            return True

        monkeypatch.setenv("LANGSMITH_PROJECT", "old-project")
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        auth_store.set_stored_key("langsmith", "lsv2_test")
        with patch(
            "deepagents_code.config.configure_langsmith_secret_redaction",
            side_effect=capture_redaction_env,
        ) as configure_redaction:
            apply_stored_langsmith_auth(replace_project=True)
        assert os.environ["LANGSMITH_PROJECT"] == LANGSMITH_PROJECT_DEFAULT
        assert os.environ["LANGSMITH_TRACING"] == "true"
        configure_redaction.assert_called_once_with()
        assert redaction_env[0]["LANGSMITH_PROJECT"] == LANGSMITH_PROJECT_DEFAULT
        assert redaction_env[0]["LANGSMITH_TRACING"] == "true"

    def test_corrupt_store_warns_and_leaves_env_untouched(
        self,
        fake_state_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A corrupt credential file is logged and tracing is left untouched."""
        import os

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        fake_state_dir.mkdir(parents=True, exist_ok=True)
        (fake_state_dir / "auth.json").write_text("{ not json", encoding="utf-8")
        with caplog.at_level("WARNING", logger="deepagents_code.config"):
            _apply_stored_langsmith_tracing()
        assert "LANGSMITH_TRACING" not in os.environ
        assert any("may be corrupt" in r.getMessage() for r in caplog.records)

    def test_applies_stored_endpoint(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stored endpoint is applied to LANGSMITH_ENDPOINT at startup."""
        import os

        from deepagents_code import auth_store

        for var in ("LANGSMITH_TRACING", "LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"):
            monkeypatch.delenv(var, raising=False)
        auth_store.set_stored_key(
            "langsmith", "lsv2_test", base_url=LANGSMITH_EU_ENDPOINT
        )
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGSMITH_ENDPOINT"] == LANGSMITH_EU_ENDPOINT
        assert "LANGCHAIN_ENDPOINT" not in os.environ

    def test_stored_endpoint_does_not_override_env(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An explicit LANGSMITH_ENDPOINT wins over the stored endpoint at startup."""
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.delenv("LANGCHAIN_ENDPOINT", raising=False)
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://from-env.example.com")
        auth_store.set_stored_key(
            "langsmith", "lsv2_test", base_url=LANGSMITH_EU_ENDPOINT
        )
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGSMITH_ENDPOINT"] == "https://from-env.example.com"

    def test_no_stored_endpoint_leaves_env_endpoint_untouched(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stored key without an endpoint never clears an existing env endpoint."""
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://self-hosted.example.com")
        auth_store.set_stored_key("langsmith", "lsv2_test")
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGSMITH_ENDPOINT"] == "https://self-hosted.example.com"

    def test_alternate_env_endpoint_blocks_stored_endpoint(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A lone LANGCHAIN_ENDPOINT (the alternate) also wins over a stored one."""
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
        monkeypatch.setenv("LANGCHAIN_ENDPOINT", "https://alt-env.example.com")
        auth_store.set_stored_key(
            "langsmith", "lsv2_test", base_url=LANGSMITH_EU_ENDPOINT
        )
        _apply_stored_langsmith_tracing()
        # The stored endpoint must not be applied, and the alternate is preserved.
        assert "LANGSMITH_ENDPOINT" not in os.environ
        assert os.environ["LANGCHAIN_ENDPOINT"] == "https://alt-env.example.com"

    def test_replace_applies_stored_endpoint_over_existing_env(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Immediate `/auth` save replaces the endpoint and clears the alternate."""
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://old.example.com")
        monkeypatch.setenv("LANGCHAIN_ENDPOINT", "https://old-alt.example.com")
        auth_store.set_stored_key(
            "langsmith", "lsv2_test", base_url=LANGSMITH_EU_ENDPOINT
        )
        _apply_stored_langsmith_tracing(replace_project=True)
        assert os.environ["LANGSMITH_ENDPOINT"] == LANGSMITH_EU_ENDPOINT
        assert "LANGCHAIN_ENDPOINT" not in os.environ

    def test_replace_clears_endpoint_when_blank(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Immediate `/auth` save clears the endpoint when no base URL is stored."""
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://old.example.com")
        auth_store.set_stored_key("langsmith", "lsv2_test")
        _apply_stored_langsmith_tracing(replace_project=True)
        assert "LANGSMITH_ENDPOINT" not in os.environ

    def test_disabled_tracing_leaves_endpoint_unset(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A paused key (tracing off) never applies its stored endpoint.

        The endpoint is applied only after the enable decision, so a deliberate
        opt-out must not leak a stored endpoint into the process env.
        """
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
        monkeypatch.delenv("LANGCHAIN_ENDPOINT", raising=False)
        monkeypatch.setenv("LANGSMITH_TRACING", "false")
        auth_store.set_stored_key(
            "langsmith", "lsv2_test", base_url=LANGSMITH_EU_ENDPOINT
        )
        _apply_stored_langsmith_tracing()
        assert "LANGSMITH_ENDPOINT" not in os.environ

    def test_prefixed_key_override_leaves_stored_endpoint_unset(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A session-scoped key is not paired with a stale stored endpoint."""
        import os

        from deepagents_code import auth_store

        for var in ("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_prefixed")
        auth_store.set_stored_key(
            "langsmith", "lsv2_stored", base_url=LANGSMITH_EU_ENDPOINT
        )
        _apply_stored_langsmith_tracing()
        assert "LANGSMITH_ENDPOINT" not in os.environ
        assert "LANGCHAIN_ENDPOINT" not in os.environ

    def test_matching_prefixed_key_allows_stored_endpoint(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A scoped key with the stored value still uses the stored endpoint."""
        import os

        from deepagents_code import auth_store

        for var in ("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_stored")
        auth_store.set_stored_key(
            "langsmith", "lsv2_stored", base_url=LANGSMITH_EU_ENDPOINT
        )
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGSMITH_ENDPOINT"] == LANGSMITH_EU_ENDPOINT
        assert "LANGCHAIN_ENDPOINT" not in os.environ


class TestNormalizeLangSmithEndpoint:
    """Tests for normalize_langsmith_endpoint() and is_http_url()."""

    def test_us_alias(self) -> None:
        """`us` (any case) resolves to the canonical US endpoint."""
        assert normalize_langsmith_endpoint("us") == LANGSMITH_US_ENDPOINT
        assert normalize_langsmith_endpoint(" US ") == LANGSMITH_US_ENDPOINT

    def test_eu_alias(self) -> None:
        """`eu` (any case) resolves to the canonical EU endpoint."""
        assert normalize_langsmith_endpoint("eu") == LANGSMITH_EU_ENDPOINT
        assert normalize_langsmith_endpoint("Eu") == LANGSMITH_EU_ENDPOINT

    def test_full_url_passthrough(self) -> None:
        """A full URL is returned stripped and unchanged."""
        url = "https://langsmith.internal.example.com"
        assert normalize_langsmith_endpoint(f"  {url}  ") == url

    def test_empty(self) -> None:
        """Blank input returns an empty string."""
        assert normalize_langsmith_endpoint("   ") == ""

    def test_is_http_url(self) -> None:
        """Only http(s) URLs with a host pass validation."""
        assert is_http_url(LANGSMITH_EU_ENDPOINT) is True
        assert is_http_url("http://localhost:1984") is True
        assert is_http_url("ftp://example.com") is False
        assert is_http_url("not-a-url") is False
        assert is_http_url("https://") is False
        assert is_http_url("http://[::1") is False
        # A host with internal whitespace is malformed: rejecting it at save time
        # stops a wrong endpoint from being stored and silently dropping traces.
        assert is_http_url("https://exa mple.com") is False


class TestGetTracingStatus:
    """Tests for get_tracing_status()."""

    _CLEAN: ClassVar[dict[str, str]] = {
        "LANGSMITH_TRACING_V2": "",
        "LANGCHAIN_TRACING_V2": "",
        "LANGSMITH_TRACING": "",
        "LANGCHAIN_TRACING": "",
        "LANGSMITH_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGSMITH_ENDPOINT": "",
        "LANGCHAIN_ENDPOINT": "",
        "LANGSMITH_PROJECT": "",
        "DEEPAGENTS_CODE_LANGSMITH_PROJECT": "",
        "DEEPAGENTS_CODE_LANGSMITH_TRACING": "",
        "DEEPAGENTS_CODE_LANGCHAIN_TRACING_V2": "",
        "DEEPAGENTS_CODE_LANGSMITH_API_KEY": "",
        "DEEPAGENTS_CODE_LANGCHAIN_API_KEY": "",
        "LANGSMITH_REPLICA_PROJECTS": "",
        "DEEPAGENTS_CODE_LANGSMITH_REPLICA_PROJECTS": "",
        "LANGSMITH_PROFILE": "",
        "LANGSMITH_CONFIG_FILE": "/__deepagents_missing_langsmith_config__.json",
    }

    def test_disabled_when_no_flags(self) -> None:
        """A clean environment reports tracing off with configured project metadata."""
        from deepagents_code.config import get_tracing_status
        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        with patch.dict("os.environ", self._CLEAN, clear=False):
            status = get_tracing_status()
        assert status.enabled is False
        assert status.explicitly_disabled is False
        assert status.has_credentials is False
        assert status.endpoint is None
        assert status.project == LANGSMITH_PROJECT_DEFAULT
        assert status.project_is_default is True
        assert status.replica_project is None

    @pytest.mark.parametrize(
        ("var", "token", "expected_enabled", "expected_disabled"),
        [
            # Non-bridged flags (bare `env.get` path) set to each falsy token.
            ("LANGSMITH_TRACING_V2", "false", False, True),
            ("LANGSMITH_TRACING_V2", "0", False, True),
            ("LANGSMITH_TRACING_V2", "no", False, True),
            ("LANGSMITH_TRACING_V2", "off", False, True),
            ("LANGCHAIN_TRACING", "false", False, True),
            # Prefixed bridged falsy flag — exercises the prefix-aware
            # `_resolve_env_var_from` off path; doctor runs pre-bootstrap.
            ("DEEPAGENTS_CODE_LANGSMITH_TRACING", "false", False, True),
            # An empty flag reads as not configured, not an explicit opt-out.
            ("LANGSMITH_TRACING_V2", "", False, False),
            # An unrecognized token is neither enabled nor an explicit opt-out.
            ("LANGSMITH_TRACING_V2", "maybe", False, False),
            # A truthy flag is enabled and never reported as explicitly disabled.
            ("LANGSMITH_TRACING_V2", "true", True, False),
        ],
    )
    def test_explicit_disable_matrix(
        self,
        var: str,
        token: str,
        expected_enabled: bool,
        expected_disabled: bool,
    ) -> None:
        """Each flag/token combination resolves to the right tri-state.

        Exercises both the bare-`env.get` path (non-bridged vars) and the
        prefix-aware `_resolve_env_var_from` path (the prefixed bridged flag),
        across every recognized falsy token plus the empty and unrecognized
        cases.
        """
        from deepagents_code.config import get_tracing_status

        env = dict(self._CLEAN)
        env[var] = token
        with patch.dict("os.environ", env, clear=False):
            status = get_tracing_status()
        assert status.enabled is expected_enabled
        assert status.explicitly_disabled is expected_disabled

    def test_prefixed_flag_and_key_are_detected(self) -> None:
        """`DEEPAGENTS_CODE_`-prefixed tracing/key vars resolve like the runtime.

        `dcode doctor` runs before bootstrap bridges these to canonical names,
        so a user with only the supported prefixed vars must still read as
        enabled/configured with the prefixed project resolved.
        """
        from deepagents_code.config import get_tracing_status

        env = dict(self._CLEAN)
        env["DEEPAGENTS_CODE_LANGSMITH_TRACING"] = "true"
        env["DEEPAGENTS_CODE_LANGSMITH_API_KEY"] = "lsv2_test"
        env["DEEPAGENTS_CODE_LANGSMITH_PROJECT"] = "prefixed-proj"
        with patch.dict("os.environ", env, clear=False):
            status = get_tracing_status()
        assert status.enabled is True
        assert status.has_credentials is True
        assert status.project == "prefixed-proj"
        assert status.project_is_default is False

    def test_explicit_default_name_is_not_marked_default(self) -> None:
        """Explicitly setting the default name still counts as configured.

        `project_is_default` tracks provenance (was anything set), not a string
        match against the default. A user who sets the project to the same name
        as the built-in default must not get the `(default)` marker.
        """
        from deepagents_code.config import get_tracing_status
        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        env = dict(self._CLEAN)
        env["DEEPAGENTS_CODE_LANGSMITH_PROJECT"] = LANGSMITH_PROJECT_DEFAULT
        with patch.dict("os.environ", env, clear=False):
            status = get_tracing_status()
        assert status.project == LANGSMITH_PROJECT_DEFAULT
        assert status.project_is_default is False

    def test_dotenv_values_are_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Doctor tracing status sees the same dotenv values as bootstrap."""
        import deepagents_code.config as config_mod

        project = tmp_path / "project"
        project.mkdir()
        (project / ".env").write_text(
            "DEEPAGENTS_CODE_LANGSMITH_TRACING=true\n"
            "DEEPAGENTS_CODE_LANGSMITH_API_KEY=lsv2_dotenv\n"
            "DEEPAGENTS_CODE_LANGSMITH_PROJECT=dotenv-proj\n"
            "DEEPAGENTS_CODE_LANGSMITH_REPLICA_PROJECTS=replica\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(project)
        monkeypatch.setattr(
            config_mod,
            "_GLOBAL_DOTENV_PATH",
            tmp_path / "missing-global.env",
        )
        config_mod._dotenv_loaded_values.clear()

        with patch.dict("os.environ", {}, clear=True):
            status = config_mod.get_tracing_status()

        assert status.enabled is True
        assert status.has_credentials is True
        assert status.project == "dotenv-proj"
        assert status.replica_project == "replica"

    def test_dotenv_profile_credentials_are_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Doctor tracing status uses dotenv profile selectors for credentials."""
        import deepagents_code.config as config_mod

        langsmith = tmp_path / "langsmith.json"
        langsmith.write_text(
            "{"
            '"current_profile":"default",'
            '"profiles":{'
            '"default":{},'
            '"dotenv":{"api_key":"lsv2_profile","api_url":"http://localhost:1984"}'
            "}"
            "}",
            encoding="utf-8",
        )
        project = tmp_path / "project"
        project.mkdir()
        (project / ".env").write_text(
            "DEEPAGENTS_CODE_LANGSMITH_TRACING=true\n"
            f"LANGSMITH_CONFIG_FILE={langsmith}\n"
            "LANGSMITH_PROFILE=dotenv\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(project)
        monkeypatch.setattr(
            config_mod,
            "_GLOBAL_DOTENV_PATH",
            tmp_path / "missing-global.env",
        )
        config_mod._dotenv_loaded_values.clear()

        with patch.dict("os.environ", {}, clear=True):
            status = config_mod.get_tracing_status()

        assert status.enabled is True
        assert status.has_credentials is True
        assert status.endpoint == "http://localhost:1984"

    def test_empty_prefixed_flag_shadows_canonical(self) -> None:
        """An empty `DEEPAGENTS_CODE_` flag suppresses the canonical one.

        Mirrors `resolve_env_var`/bootstrap: a present-but-empty prefixed var
        disables tracing even when the canonical flag is truthy.
        """
        from deepagents_code.config import get_tracing_status

        env = dict(self._CLEAN)
        env["DEEPAGENTS_CODE_LANGSMITH_TRACING"] = ""
        env["LANGSMITH_TRACING"] = "true"
        with patch.dict("os.environ", env, clear=False):
            status = get_tracing_status()
        assert status.enabled is False
        assert status.explicitly_disabled is True

    def test_canonical_non_bridged_flag_enables(self) -> None:
        """A canonical, non-bridged flag (`LANGSMITH_TRACING_V2`) enables tracing."""
        from deepagents_code.config import get_tracing_status

        env = dict(self._CLEAN)
        env["LANGSMITH_TRACING_V2"] = "true"
        with patch.dict("os.environ", env, clear=False):
            status = get_tracing_status()
        assert status.enabled is True

    def test_keyless_custom_endpoint_resolves_project(self) -> None:
        """A keyless custom endpoint counts as active and resolves the project."""
        from deepagents_code.config import get_tracing_status
        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        env = dict(self._CLEAN)
        env["DEEPAGENTS_CODE_LANGSMITH_TRACING"] = "true"
        env["LANGSMITH_ENDPOINT"] = "http://localhost:1984"
        with patch.dict("os.environ", env, clear=False):
            status = get_tracing_status()
        assert status.enabled is True
        assert status.has_credentials is False
        assert status.endpoint == "http://localhost:1984"
        assert status.project == LANGSMITH_PROJECT_DEFAULT

    def test_profile_credentials_are_detected(self, tmp_path: Path) -> None:
        """A LangSmith profile API key counts as credentials (no env key needed)."""
        from deepagents_code.config import get_tracing_status
        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        config = tmp_path / "config.json"
        config.write_text(
            '{"current_profile":"default","profiles":{"default":{"api_key":"lsv2_profile"}}}',
            encoding="utf-8",
        )
        env = dict(self._CLEAN)
        env["DEEPAGENTS_CODE_LANGSMITH_TRACING"] = "true"
        env["LANGSMITH_CONFIG_FILE"] = str(config)
        with patch.dict("os.environ", env, clear=False):
            status = get_tracing_status()
        assert status.enabled is True
        assert status.has_credentials is True
        assert status.project == LANGSMITH_PROJECT_DEFAULT

    def test_project_resolved_when_enabled_without_auth(self) -> None:
        """Tracing auth state does not hide configured project metadata."""
        from deepagents_code.config import get_tracing_status
        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        env = dict(self._CLEAN)
        env["DEEPAGENTS_CODE_LANGSMITH_TRACING"] = "true"
        with patch.dict("os.environ", env, clear=False):
            status = get_tracing_status()
        assert status.enabled is True
        assert status.has_credentials is False
        assert status.project == LANGSMITH_PROJECT_DEFAULT

    def test_empty_prefixed_project_falls_through_to_canonical(self) -> None:
        """An empty prefixed project must not shadow a real `LANGSMITH_PROJECT`.

        Mirrors the manifest/runtime contract: `resolve_scalar` skips an empty
        `DEEPAGENTS_CODE_LANGSMITH_PROJECT` and uses bare `LANGSMITH_PROJECT`,
        unlike `resolve_env_var`, which would shadow it and report the default.
        """
        from deepagents_code.config import get_tracing_status

        env = dict(self._CLEAN)
        env["DEEPAGENTS_CODE_LANGSMITH_TRACING"] = "true"
        env["DEEPAGENTS_CODE_LANGSMITH_API_KEY"] = "lsv2_test"
        env["DEEPAGENTS_CODE_LANGSMITH_PROJECT"] = ""
        env["LANGSMITH_PROJECT"] = "prod"
        with patch.dict("os.environ", env, clear=False):
            status = get_tracing_status()
        assert status.project == "prod"
        assert status.project_is_default is False

    def test_reports_first_replica_project(self) -> None:
        """Only the first replica project is reported (server mirrors one)."""
        from deepagents_code.config import get_tracing_status

        env = dict(self._CLEAN)
        env["DEEPAGENTS_CODE_LANGSMITH_REPLICA_PROJECTS"] = "replica-a, replica-b"
        with patch.dict("os.environ", env, clear=False):
            status = get_tracing_status()
        assert status.replica_project == "replica-a"

    def test_enabled_and_explicitly_disabled_is_rejected(self) -> None:
        """The contradictory enabled/disabled pair fails loud at construction."""
        from deepagents_code.config import TracingStatus

        with pytest.raises(ValueError, match="both enabled and explicitly disabled"):
            TracingStatus(
                enabled=True,
                explicitly_disabled=True,
                has_credentials=False,
                endpoint=None,
                project=None,
                project_is_default=False,
                replica_project=None,
            )


class TestQuietSdkLogging:
    """Tests for _quiet_sdk_logging()."""

    def test_attaches_null_handler_without_debug(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without debug, SDK loggers get a NullHandler so logs stay off stderr."""
        from deepagents_code._env_vars import DEBUG

        monkeypatch.delenv(DEBUG, raising=False)
        for name in _QUIET_SDK_LOGGER_NAMES:
            logger = logging.getLogger(name)
            logger.handlers.clear()
            logger.setLevel(logging.NOTSET)
            monkeypatch.setattr(logger, "propagate", True)

        _quiet_sdk_logging()

        for name in _QUIET_SDK_LOGGER_NAMES:
            logger = logging.getLogger(name)
            handlers = logger.handlers
            assert any(isinstance(h, logging.NullHandler) for h in handlers)
            assert logger.level == logging.NOTSET
            # Propagation is left intact so a deliberately configured handler
            # (an embedding app's root handler, pytest's caplog) still receives
            # real SDK errors; the NullHandler alone keeps routine noise off the
            # last-resort stderr handler.
            assert logger.propagate is True

    def test_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Repeated calls do not stack duplicate handlers."""
        from deepagents_code._env_vars import DEBUG

        monkeypatch.delenv(DEBUG, raising=False)
        for name in _QUIET_SDK_LOGGER_NAMES:
            logging.getLogger(name).handlers.clear()

        _quiet_sdk_logging()
        _quiet_sdk_logging()

        for name in _QUIET_SDK_LOGGER_NAMES:
            handlers = logging.getLogger(name).handlers
            assert len(handlers) == 1

    def test_routes_harness_diagnostics_to_debug_log(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Debug mode configures a file handler for harness diagnostics."""
        from deepagents_code._env_vars import DEBUG

        monkeypatch.setenv(DEBUG, "1")
        harness_logger = logging.getLogger(
            "deepagents.profiles.harness.harness_profiles"
        )
        harness_logger.handlers.clear()
        monkeypatch.setattr(harness_logger, "propagate", True)

        with patch("deepagents_code._debug.configure_debug_logging") as configure:
            _quiet_sdk_logging()

        assert any(call.args == (harness_logger,) for call in configure.call_args_list)
        assert harness_logger.propagate is True

    def test_leaves_other_deepagents_loggers_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Actionable Deep Agents runtime warnings keep their normal routing."""
        from deepagents_code._env_vars import DEBUG

        monkeypatch.delenv(DEBUG, raising=False)
        runtime_logger = logging.getLogger("deepagents.backends.filesystem")
        runtime_logger.handlers.clear()
        monkeypatch.setattr(runtime_logger, "propagate", True)

        _quiet_sdk_logging()

        assert runtime_logger.handlers == []
        assert runtime_logger.propagate is True


class TestFetchLangsmithProjectUrl:
    """Tests for fetch_langsmith_project_url()."""

    def setup_method(self) -> None:
        """Clear LangSmith URL cache before each test."""
        reset_langsmith_url_cache()

    def test_returns_url_on_success(self) -> None:
        """Should return the project URL from the LangSmith client."""

        class FakeProject:
            url = "https://smith.langchain.com/o/org/projects/p/proj"

        with patch("langsmith.Client") as mock_client_cls:
            mock_client_cls.return_value.read_project.return_value = FakeProject()
            result = fetch_langsmith_project_url("my-project")

        assert result == "https://smith.langchain.com/o/org/projects/p/proj"

    def test_returns_none_on_error(self) -> None:
        """Should return None when the LangSmith client raises."""
        with patch("langsmith.Client") as mock_client_cls:
            mock_client_cls.return_value.read_project.side_effect = OSError("timeout")
            result = fetch_langsmith_project_url("my-project")

        assert result is None

    def test_returns_none_on_project_not_found(self) -> None:
        """Should return None when the project does not exist yet."""
        from langsmith.utils import LangSmithNotFoundError

        with patch("langsmith.Client") as mock_client_cls:
            mock_client_cls.return_value.read_project.side_effect = (
                LangSmithNotFoundError("Project angus-dacli not found")
            )
            result = fetch_langsmith_project_url("angus-dacli")

        assert result is None

    def test_returns_none_on_unexpected_exception(self) -> None:
        """Should return None on unexpected SDK exceptions."""
        with patch("langsmith.Client") as mock_client_cls:
            mock_client_cls.return_value.read_project.side_effect = TypeError(
                "unexpected SDK type error"
            )
            result = fetch_langsmith_project_url("my-project")

        assert result is None

    def test_returns_none_when_lookup_times_out(self) -> None:
        """Should return None when LangSmith lookup exceeds timeout."""
        with (
            patch(
                "deepagents_code.config._LANGSMITH_URL_LOOKUP_TIMEOUT_SECONDS",
                0.01,
            ),
            patch("langsmith.Client") as mock_client_cls,
        ):
            mock_client_cls.return_value.read_project.side_effect = lambda **_kwargs: (
                time.sleep(0.02)
            )
            result = fetch_langsmith_project_url("my-project")

        assert result is None

    def test_returns_none_when_url_is_none(self) -> None:
        """Should return None when the project has no URL."""

        class FakeProject:
            url = None

        with patch("langsmith.Client") as mock_client_cls:
            mock_client_cls.return_value.read_project.return_value = FakeProject()
            result = fetch_langsmith_project_url("my-project")

        assert result is None

    def test_caches_result_after_first_call(self) -> None:
        """Should only call the LangSmith client once for repeated invocations."""

        class FakeProject:
            url = "https://smith.langchain.com/o/org/projects/p/proj"

        with patch("langsmith.Client") as mock_client_cls:
            mock_client_cls.return_value.read_project.return_value = FakeProject()
            first = fetch_langsmith_project_url("my-project")
            second = fetch_langsmith_project_url("my-project")

        assert first == "https://smith.langchain.com/o/org/projects/p/proj"
        assert second == first
        mock_client_cls.assert_called_once()

    def test_retries_after_failure(self) -> None:
        """Should retry after failure instead of caching None."""
        with patch("langsmith.Client") as mock_client_cls:
            mock_client_cls.return_value.read_project.side_effect = OSError("timeout")
            first = fetch_langsmith_project_url("my-project")
            second = fetch_langsmith_project_url("my-project")

        assert first is None
        assert second is None
        assert mock_client_cls.return_value.read_project.call_count == 2

    def test_retries_when_url_is_none(self) -> None:
        """Should retry when the project URL is missing instead of caching None."""

        class FakeProject:
            url = None

        with patch("langsmith.Client") as mock_client_cls:
            mock_client_cls.return_value.read_project.return_value = FakeProject()
            first = fetch_langsmith_project_url("my-project")
            second = fetch_langsmith_project_url("my-project")

        assert first is None
        assert second is None
        assert mock_client_cls.return_value.read_project.call_count == 2

    def test_different_project_name_fetches_again(self) -> None:
        """Should fetch again when called with a different project name."""

        class FakeProjectA:
            url = "https://smith.langchain.com/o/org/projects/p/a"

        class FakeProjectB:
            url = "https://smith.langchain.com/o/org/projects/p/b"

        with patch("langsmith.Client") as mock_client_cls:
            mock_client_cls.return_value.read_project.side_effect = [
                FakeProjectA(),
                FakeProjectB(),
            ]
            first = fetch_langsmith_project_url("project-a")
            second = fetch_langsmith_project_url("project-b")

        assert first == "https://smith.langchain.com/o/org/projects/p/a"
        assert second == "https://smith.langchain.com/o/org/projects/p/b"
        assert mock_client_cls.return_value.read_project.call_count == 2

    def test_or_raise_raises_project_not_found(self) -> None:
        """A 404 from read_project raises LangSmithProjectNotFoundError."""
        from langsmith.utils import LangSmithNotFoundError

        with patch("langsmith.Client") as mock_client_cls:
            mock_client_cls.return_value.read_project.side_effect = (
                LangSmithNotFoundError("Project deepagents-code not found")
            )
            with pytest.raises(LangSmithProjectNotFoundError):
                fetch_langsmith_project_url_or_raise("deepagents-code")

    def test_or_raise_raises_api_error_on_other_failure(self) -> None:
        """A non-404 SDK error raises the generic LangSmithApiError."""
        with patch("langsmith.Client") as mock_client_cls:
            mock_client_cls.return_value.read_project.side_effect = OSError("boom")
            with pytest.raises(LangSmithApiError) as exc_info:
                fetch_langsmith_project_url_or_raise("my-project")
        assert not isinstance(exc_info.value, LangSmithProjectNotFoundError)


class TestBuildLangsmithThreadUrl:
    """Tests for build_langsmith_thread_url()."""

    def setup_method(self) -> None:
        """Clear LangSmith URL cache before each test."""
        reset_langsmith_url_cache()

    def test_returns_url_when_configured(self) -> None:
        """Should return a full thread URL when LangSmith is configured."""

        class FakeProject:
            url = "https://smith.langchain.com/o/org/projects/p/proj"

        with (
            patch(
                "deepagents_code.config.get_langsmith_project_name",
                return_value="my-project",
            ),
            patch("langsmith.Client") as mock_client_cls,
        ):
            mock_client_cls.return_value.read_project.return_value = FakeProject()
            result = build_langsmith_thread_url("thread-123")

        assert (
            result
            == "https://smith.langchain.com/o/org/projects/p/proj/t/thread-123?utm_source=deepagents-code"
        )

    def test_cached_url_is_available_without_another_lookup(self) -> None:
        """A resolved project URL can build later thread links synchronously."""

        class FakeProject:
            url = "https://smith.langchain.com/o/org/projects/p/proj"

        with (
            patch(
                "deepagents_code.config.get_langsmith_project_name",
                return_value="my-project",
            ),
            patch("langsmith.Client") as mock_client_cls,
        ):
            mock_client_cls.return_value.read_project.return_value = FakeProject()
            build_langsmith_thread_url("thread-123")
            result = get_cached_langsmith_thread_url("thread-456")

        assert (
            result
            == "https://smith.langchain.com/o/org/projects/p/proj/t/thread-456?utm_source=deepagents-code"
        )
        mock_client_cls.return_value.read_project.assert_called_once()

    def test_cached_url_does_not_trigger_initial_lookup(self) -> None:
        """A cache miss remains synchronous and never contacts LangSmith."""
        with (
            patch(
                "deepagents_code.config.get_langsmith_project_name",
                return_value="my-project",
            ),
            patch("langsmith.Client") as mock_client_cls,
        ):
            result = get_cached_langsmith_thread_url("thread-123")

        assert result is None
        mock_client_cls.assert_not_called()

    def test_strips_trailing_slash(self) -> None:
        """Should not produce double slashes when project URL has trailing slash."""

        class FakeProject:
            url = "https://smith.langchain.com/o/org/projects/p/proj/"

        with (
            patch(
                "deepagents_code.config.get_langsmith_project_name",
                return_value="my-project",
            ),
            patch("langsmith.Client") as mock_client_cls,
        ):
            mock_client_cls.return_value.read_project.return_value = FakeProject()
            result = build_langsmith_thread_url("thread-123")

        assert (
            result
            == "https://smith.langchain.com/o/org/projects/p/proj/t/thread-123?utm_source=deepagents-code"
        )

    def test_returns_none_when_no_project_name(self) -> None:
        """Should return None when LangSmith project name is not configured."""
        with patch(
            "deepagents_code.config.get_langsmith_project_name",
            return_value=None,
        ):
            result = build_langsmith_thread_url("thread-123")

        assert result is None

    def test_returns_none_when_fetch_fails(self) -> None:
        """Should return None when the project URL cannot be resolved."""
        with (
            patch(
                "deepagents_code.config.get_langsmith_project_name",
                return_value="my-project",
            ),
            patch("langsmith.Client") as mock_client_cls,
        ):
            mock_client_cls.return_value.read_project.side_effect = OSError("timeout")
            result = build_langsmith_thread_url("thread-123")

        assert result is None


class TestGetProviderKwargsConfigFallback:
    """Tests for _get_provider_kwargs() config-file fallback."""

    def setup_method(self) -> None:
        """Clear model config cache before each test."""
        clear_caches()

    def test_returns_base_url_from_config(self, tmp_path: Path) -> None:
        """Returns base_url from config for non-hardcoded provider."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.fireworks]
models = ["llama-v3p1-70b"]
base_url = "https://api.fireworks.ai/inference/v1"
api_key_env = "FIREWORKS_API_KEY"
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"FIREWORKS_API_KEY": "test-key"}, clear=True),
        ):
            kwargs = _get_provider_kwargs("fireworks")

        assert kwargs["base_url"] == "https://api.fireworks.ai/inference/v1"
        assert kwargs["api_key"] == "test-key"

    def test_returns_api_key_from_config(self, tmp_path: Path) -> None:
        """Returns resolved api_key from config-file api_key_env."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.together]
models = ["meta-llama/Llama-3-70b"]
api_key_env = "TOGETHER_API_KEY"
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"TOGETHER_API_KEY": "together-key"}, clear=True),
        ):
            kwargs = _get_provider_kwargs("together")

        assert kwargs["api_key"] == "together-key"
        assert "base_url" not in kwargs

    def test_stored_auth_base_url_reaches_kwargs_without_env_var(
        self, tmp_path: Path
    ) -> None:
        """A `/auth` endpoint reaches the `base_url` kwarg for a non-mapped provider.

        `baseten` has an API-key env var but no base-URL env var, so the stored
        endpoint resolves only through `get_base_url`'s store fallback. This is
        the end-to-end path that makes a saved base URL reach the model as the
        `base_url` constructor kwarg (which `ChatBaseten` accepts via its
        `base_url` alias) rather than being silently dropped.
        """
        from deepagents_code import auth_store

        state_dir = tmp_path / ".state"
        config_path = tmp_path / "config.toml"
        config_path.write_text("")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.object(model_config, "DEFAULT_STATE_DIR", state_dir),
            patch.dict("os.environ", {"BASETEN_API_KEY": "tk"}, clear=True),
        ):
            clear_caches()
            auth_store.set_stored_key(
                "baseten", "tk", base_url="https://proxy.example/v1"
            )
            kwargs = _get_provider_kwargs("baseten")

        assert kwargs["base_url"] == "https://proxy.example/v1"

    def test_prefixed_env_var_beats_canonical(self, tmp_path: Path) -> None:
        """DEEPAGENTS_CODE_ prefixed var overrides canonical in provider kwargs."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.fireworks]
models = ["llama-v3p1-70b"]
api_key_env = "FIREWORKS_API_KEY"
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict(
                "os.environ",
                {
                    "FIREWORKS_API_KEY": "canonical",
                    "DEEPAGENTS_CODE_FIREWORKS_API_KEY": "prefixed",
                },
                clear=True,
            ),
        ):
            kwargs = _get_provider_kwargs("fireworks")

        assert kwargs["api_key"] == "prefixed"

    def test_omits_api_key_when_env_not_set(self, tmp_path: Path) -> None:
        """Omits api_key when the env var is not set."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.fireworks]
models = ["llama-v3p1-70b"]
api_key_env = "FIREWORKS_API_KEY"
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {}, clear=True),
        ):
            kwargs = _get_provider_kwargs("fireworks")

        assert "api_key" not in kwargs

    def test_returns_empty_for_unknown_config_provider(self) -> None:
        """Returns empty dict for provider not in hardcoded map or config."""
        kwargs = _get_provider_kwargs("nonexistent_provider_xyz")
        assert kwargs == {}

    def test_unconfigured_providers_return_empty(self) -> None:
        """Providers without config or env credentials return empty kwargs."""
        with patch.dict("os.environ", {}, clear=True):
            kwargs = _get_provider_kwargs("anthropic")
            assert kwargs == {}

            kwargs = _get_provider_kwargs("google_genai")
            assert kwargs == {}

    def test_merges_config_params(self, tmp_path: Path) -> None:
        """Merges params from config with base_url and api_key."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.custom]
models = ["my-model"]
base_url = "https://my-endpoint.example.com"
api_key_env = "CUSTOM_KEY"

[models.providers.custom.params]
temperature = 0
max_tokens = 4096
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"CUSTOM_KEY": "secret"}, clear=True),
        ):
            kwargs = _get_provider_kwargs("custom")

        assert kwargs["temperature"] == 0
        assert kwargs["max_tokens"] == 4096
        assert kwargs["base_url"] == "https://my-endpoint.example.com"
        assert kwargs["api_key"] == "secret"

    def test_passes_model_name_for_per_model_params(self, tmp_path: Path) -> None:
        """Per-model params are merged when model_name is provided."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["qwen3:4b", "llama3"]

[models.providers.ollama.params]
temperature = 0
num_ctx = 8192

[models.providers.ollama.params."qwen3:4b"]
temperature = 0.5
num_ctx = 4000
""")
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            kwargs = _get_provider_kwargs("ollama", model_name="qwen3:4b")

        assert kwargs["temperature"] == pytest.approx(0.5)
        assert kwargs["num_ctx"] == 4000

    def test_model_name_none_uses_provider_params(self, tmp_path: Path) -> None:
        """model_name=None returns provider params without per-model merge."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["qwen3:4b"]

[models.providers.ollama.params]
temperature = 0

[models.providers.ollama.params."qwen3:4b"]
temperature = 0.5
""")
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            kwargs = _get_provider_kwargs("ollama")

        assert kwargs["temperature"] == 0

    def test_ollama_optional_api_key_sets_authorization_header(
        self,
        tmp_path: Path,
    ) -> None:
        """OLLAMA_API_KEY is forwarded through client_kwargs for cloud use."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["llama3"]
base_url = "https://ollama.example.com"
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"OLLAMA_API_KEY": "test-key"}, clear=True),
        ):
            kwargs = _get_provider_kwargs("ollama")

        assert kwargs["client_kwargs"]["headers"]["Authorization"] == (
            "Bearer test-key"
        )

    def test_ollama_prefixed_optional_api_key_overrides_canonical(
        self,
        tmp_path: Path,
    ) -> None:
        """The CLI-scoped Ollama key follows normal env override behavior."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["llama3"]
base_url = "https://ollama.example.com"
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict(
                "os.environ",
                {
                    "OLLAMA_API_KEY": "canonical",
                    "DEEPAGENTS_CODE_OLLAMA_API_KEY": "prefixed",
                },
                clear=True,
            ),
        ):
            kwargs = _get_provider_kwargs("ollama")

        assert kwargs["client_kwargs"]["headers"]["Authorization"] == (
            "Bearer prefixed"
        )

    def test_ollama_preserves_user_authorization_header(
        self,
        tmp_path: Path,
    ) -> None:
        """Existing Authorization header (any case) is not overwritten."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["llama3"]
base_url = "https://ollama.example.com"

[models.providers.ollama.params.llama3]
client_kwargs = { headers = { authorization = "Bearer user-supplied" } }
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"OLLAMA_API_KEY": "env-key"}, clear=True),
        ):
            kwargs = _get_provider_kwargs("ollama", model_name="llama3")

        headers = kwargs["client_kwargs"]["headers"]
        assert headers["authorization"] == "Bearer user-supplied"
        assert "Authorization" not in headers

    def test_ollama_preserves_unrelated_headers_and_client_kwargs(
        self,
        tmp_path: Path,
    ) -> None:
        """Sibling client_kwargs and headers entries survive Authorization injection."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["llama3"]
base_url = "https://ollama.example.com"

[models.providers.ollama.params.llama3]
client_kwargs = { timeout = 30, headers = { "X-Trace-Id" = "abc" } }
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"OLLAMA_API_KEY": "env-key"}, clear=True),
        ):
            kwargs = _get_provider_kwargs("ollama", model_name="llama3")

        client_kwargs = kwargs["client_kwargs"]
        assert client_kwargs["timeout"] == 30
        assert client_kwargs["headers"]["X-Trace-Id"] == "abc"
        assert client_kwargs["headers"]["Authorization"] == "Bearer env-key"

    def test_ollama_local_endpoint_does_not_inject_header(
        self,
        tmp_path: Path,
    ) -> None:
        """Without OLLAMA_API_KEY, no Authorization header is injected."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["llama3"]
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {}, clear=True),
        ):
            kwargs = _get_provider_kwargs("ollama")

        client_kwargs = kwargs.get("client_kwargs", {})
        headers = (
            client_kwargs.get("headers", {}) if isinstance(client_kwargs, dict) else {}
        )
        assert "Authorization" not in headers
        assert "authorization" not in headers

    def test_base_url_and_api_key_override_config_params(self, tmp_path: Path) -> None:
        """base_url/api_key from config fields override same keys in params."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.custom]
models = ["my-model"]
base_url = "https://correct-url.com"
api_key_env = "CUSTOM_KEY"

[models.providers.custom.params]
base_url = "https://wrong-url.com"
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"CUSTOM_KEY": "secret"}, clear=True),
        ):
            kwargs = _get_provider_kwargs("custom")

        # Explicit base_url field should win over kwargs.base_url
        assert kwargs["base_url"] == "https://correct-url.com"


def _make_init_chat_model_mock() -> Mock:
    """Return a `Mock` shaped like `init_chat_model`'s return value.

    Each `TestCreateModel*` test patches `langchain.chat_models.init_chat_model`
    and inspects `call_args`; the returned model needs `profile = None` so the
    downstream context-limit/modality extraction in `create_model` is a no-op.
    """
    mock_model = Mock()
    mock_model.profile = None
    return mock_model


@pytest.fixture
def _isolate_provider_profiles() -> Iterator[None]:
    """Snapshot/restore SDK `_PROVIDER_PROFILES` and CLI registration sentinel.

    The provider-profile registry is process-global. Tests that register
    custom profiles (or that exercise the CLI's lazy OpenRouter registration)
    must not leak state into other tests in the same session.
    """
    from deepagents.profiles.provider import provider_profiles

    from deepagents_code import config as cli_config

    saved_profiles = dict(provider_profiles._PROVIDER_PROFILES)
    saved_cli_flag = cli_config._cli_openrouter_profile_registered
    try:
        yield
    finally:
        provider_profiles._PROVIDER_PROFILES.clear()
        provider_profiles._PROVIDER_PROFILES.update(saved_profiles)
        cli_config._cli_openrouter_profile_registered = saved_cli_flag


class TestOpenRouterVersionCheck:
    """Tests for OpenRouter version enforcement via the SDK profile."""

    def setup_method(self) -> None:
        """Clear model config cache before each test."""
        clear_caches()

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_rejects_old_version(self, mock_init: Mock) -> None:
        """`create_model` wraps the version `ImportError` in `ModelConfigError`."""
        mock_init.return_value = _make_init_chat_model_mock()
        with (
            patch(
                "deepagents.profiles.provider._openrouter.pkg_version",
                return_value="0.0.1",
            ),
            pytest.raises(ModelConfigError, match="langchain-openrouter>="),
        ):
            create_model("openrouter:deepseek/deepseek-chat")

    @patch("langchain.chat_models.init_chat_model")
    def test_accepts_sufficient_version(self, mock_init: Mock) -> None:
        """`create_model` succeeds when version meets minimum."""
        from deepagents.profiles.provider._openrouter import OPENROUTER_MIN_VERSION

        mock_init.return_value = _make_init_chat_model_mock()
        with patch(
            "deepagents.profiles.provider._openrouter.pkg_version",
            return_value=OPENROUTER_MIN_VERSION,
        ):
            create_model("openrouter:deepseek/deepseek-chat")

        _, call_kwargs = mock_init.call_args
        assert "app_url" in call_kwargs

    @patch("langchain.chat_models.init_chat_model")
    def test_skipped_for_other_providers(self, mock_init: Mock) -> None:
        """Version check is not invoked for non-openrouter providers."""
        mock_init.return_value = _make_init_chat_model_mock()
        with patch(
            "deepagents.profiles.provider._openrouter.check_openrouter_version"
        ) as mock_check:
            create_model("openai:gpt-5.2")

        mock_check.assert_not_called()


class TestOpenRouterHeaders:
    """Tests for OpenRouter default attribution headers."""

    def setup_method(self) -> None:
        """Clear model config cache before each test."""
        clear_caches()

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_injects_attribution_kwargs(self, mock_init: Mock) -> None:
        """`create_model` injects `app_url`, `app_title`, `app_categories`."""
        mock_init.return_value = _make_init_chat_model_mock()

        create_model("openrouter:deepseek/deepseek-chat")

        _, call_kwargs = mock_init.call_args
        assert call_kwargs["app_url"] == "https://pypi.org/project/deepagents-code/"
        assert call_kwargs["app_title"] == "Deep Agents Code"
        assert call_kwargs["app_categories"] == ["cli-agent"]

    @patch("langchain.chat_models.init_chat_model")
    def test_per_model_attribution_overrides_defaults(
        self, mock_init: Mock, tmp_path: Path
    ) -> None:
        """Per-model `app_title` from `config.toml` overrides built-in default."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.openrouter]
models = ["deepseek/deepseek-chat"]

[models.providers.openrouter.params."deepseek/deepseek-chat"]
app_title = "My Custom App"
""")
        mock_init.return_value = _make_init_chat_model_mock()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            create_model("openrouter:deepseek/deepseek-chat")

        _, call_kwargs = mock_init.call_args
        assert call_kwargs["app_title"] == "My Custom App"
        # Built-in app_url should still be present
        assert call_kwargs["app_url"] == "https://pypi.org/project/deepagents-code/"

    @patch("langchain.chat_models.init_chat_model")
    def test_per_model_categories_override(
        self, mock_init: Mock, tmp_path: Path
    ) -> None:
        """Per-model `app_categories` from `config.toml` overrides built-in default."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.openrouter]
models = ["deepseek/deepseek-chat"]

[models.providers.openrouter.params."deepseek/deepseek-chat"]
app_categories = ["cloud-agent"]
""")
        mock_init.return_value = _make_init_chat_model_mock()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            create_model("openrouter:deepseek/deepseek-chat")

        _, call_kwargs = mock_init.call_args
        assert call_kwargs["app_categories"] == ["cloud-agent"]

    @patch("langchain.chat_models.init_chat_model")
    def test_no_attribution_for_other_providers(self, mock_init: Mock) -> None:
        """Other providers do not get OpenRouter attribution kwargs."""
        mock_init.return_value = _make_init_chat_model_mock()

        create_model("openai:gpt-5.2")

        _, call_kwargs = mock_init.call_args
        assert "app_url" not in call_kwargs
        assert "app_title" not in call_kwargs
        assert "app_categories" not in call_kwargs
        assert "openrouter_provider" not in call_kwargs

    @patch("langchain.chat_models.init_chat_model")
    def test_sdk_provider_routing_flows_through_cli_profile(
        self, mock_init: Mock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SDK's Azure-ignore default survives CLI profile chaining."""
        from deepagents.profiles.provider._openrouter import _OPENROUTER_ALLOW_AZURE_ENV

        monkeypatch.delenv(_OPENROUTER_ALLOW_AZURE_ENV, raising=False)
        mock_init.return_value = _make_init_chat_model_mock()

        create_model("openrouter:deepseek/deepseek-chat")

        _, call_kwargs = mock_init.call_args
        assert call_kwargs["openrouter_provider"] == {"ignore": ["azure"]}


class TestCreateModelForwardsProviderProfile:
    """Tests that `create_model` forwards profile kwargs to `init_chat_model`.

    Regression coverage for #2959: env-default and explicit OpenAI selections
    both need `use_responses_api=True` so the CLI's PDF-attachment path (which
    emits `type: "file"` content blocks) is routed through the Responses API
    instead of 400'ing against Chat Completions.
    """

    def setup_method(self) -> None:
        """Clear model config cache before each test."""
        clear_caches()

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )

    @patch("deepagents_code.config._get_default_model_spec")
    @patch("langchain.chat_models.init_chat_model")
    def test_env_default_openai_gets_use_responses_api(
        self, mock_init: Mock, mock_default: Mock
    ) -> None:
        """No-spec `create_model()` resolves to OpenAI with `use_responses_api=True`."""
        mock_default.return_value = "openai:gpt-5.2"
        mock_init.return_value = _make_init_chat_model_mock()

        create_model()

        _, call_kwargs = mock_init.call_args
        assert call_kwargs.get("use_responses_api") is True

    @patch("langchain.chat_models.init_chat_model")
    def test_explicit_openai_spec_gets_use_responses_api(self, mock_init: Mock) -> None:
        """Explicit `openai:*` selection also inherits the SDK Responses API default."""
        mock_init.return_value = _make_init_chat_model_mock()

        create_model("openai:gpt-5.2")

        _, call_kwargs = mock_init.call_args
        assert call_kwargs.get("use_responses_api") is True

    @patch("langchain.chat_models.init_chat_model")
    def test_model_params_override_profile_default(self, mock_init: Mock) -> None:
        """`--model-params` (`extra_kwargs`) wins over profile defaults."""
        mock_init.return_value = _make_init_chat_model_mock()

        create_model(
            "openai:gpt-5.2",
            extra_kwargs={"use_responses_api": False},
        )

        _, call_kwargs = mock_init.call_args
        assert call_kwargs.get("use_responses_api") is False

    @patch("langchain.chat_models.init_chat_model")
    def test_reasoning_effort_does_not_add_summary(self, mock_init: Mock) -> None:
        mock_init.return_value = _make_init_chat_model_mock()

        create_model(
            "openai:gpt-5.5",
            extra_kwargs={"reasoning_effort": "high"},
        )

        _, call_kwargs = mock_init.call_args
        assert call_kwargs["reasoning_effort"] == "high"
        assert "reasoning" not in call_kwargs

    @patch("langchain.chat_models.init_chat_model")
    def test_reasoning_effort_composes_with_configured_summary(
        self,
        mock_init: Mock,
        tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.openai]
models = ["gpt-5.5"]
[models.providers.openai.params."gpt-5.5".reasoning]
summary = "auto"
effort = "low"
""")
        mock_init.return_value = _make_init_chat_model_mock()

        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            create_model(
                "openai:gpt-5.5",
                extra_kwargs={"reasoning_effort": "high"},
            )

        _, call_kwargs = mock_init.call_args
        assert call_kwargs["reasoning"] == {"summary": "auto", "effort": "high"}
        assert "reasoning_effort" not in call_kwargs

    @patch("langchain.chat_models.init_chat_model")
    def test_explicit_native_reasoning_effort_keeps_precedence(
        self,
        mock_init: Mock,
    ) -> None:
        mock_init.return_value = _make_init_chat_model_mock()

        create_model(
            "openai:gpt-5.5",
            extra_kwargs={
                "reasoning_effort": "high",
                "reasoning": {"effort": "low", "summary": "auto"},
            },
        )

        _, call_kwargs = mock_init.call_args
        assert call_kwargs["reasoning"] == {"effort": "low", "summary": "auto"}
        assert "reasoning_effort" not in call_kwargs

    @patch("langchain.chat_models.init_chat_model")
    def test_config_toml_opt_out_wins_over_profile(
        self, mock_init: Mock, tmp_path: Path
    ) -> None:
        """`use_responses_api=false` in `config.toml` opts out of the default."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.openai.params]
use_responses_api = false
""")
        mock_init.return_value = _make_init_chat_model_mock()

        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            create_model("openai:gpt-5.2")

        _, call_kwargs = mock_init.call_args
        assert call_kwargs.get("use_responses_api") is False

    @patch("langchain.chat_models.init_chat_model")
    def test_anthropic_unaffected(self, mock_init: Mock) -> None:
        """Anthropic profile currently does not set `use_responses_api`."""
        mock_init.return_value = _make_init_chat_model_mock()

        create_model("anthropic:claude-sonnet-4-5")

        _, call_kwargs = mock_init.call_args
        assert "use_responses_api" not in call_kwargs

    @patch("langchain.chat_models.init_chat_model")
    def test_exact_model_profile_wins_over_provider_profile(
        self,
        mock_init: Mock,
        _isolate_provider_profiles: None,  # noqa: PT019
    ) -> None:
        """Per-model profile registration wins over the provider-wide profile.

        Pins the spec construction in `create_model` (`f"{provider}:{model_name}"`).
        A regression that drops the model-name suffix would silently fall back
        to the provider-wide registration and bypass exact-model overrides.
        """
        from deepagents.profiles.provider import (
            ProviderProfile,
            register_provider_profile,
        )

        register_provider_profile(
            "openai:gpt-5.2",
            ProviderProfile(init_kwargs={"temperature": 0.42}),
        )
        mock_init.return_value = _make_init_chat_model_mock()

        create_model("openai:gpt-5.2")
        _, exact_kwargs = mock_init.call_args
        assert exact_kwargs.get("temperature") == pytest.approx(0.42)

        mock_init.reset_mock()
        create_model("openai:gpt-5.5")
        _, other_kwargs = mock_init.call_args
        assert "temperature" not in other_kwargs

    @patch("langchain.chat_models.init_chat_model")
    def test_init_kwargs_factory_output_forwarded(
        self,
        mock_init: Mock,
        _isolate_provider_profiles: None,  # noqa: PT019
    ) -> None:
        """`init_kwargs_factory` output reaches `init_chat_model`.

        Direct coverage for the factory branch in `apply_provider_profile` —
        previously exercised only transitively via OpenRouter.
        """
        from deepagents.profiles.provider import (
            ProviderProfile,
            register_provider_profile,
        )

        register_provider_profile(
            "anthropic",
            ProviderProfile(init_kwargs_factory=lambda: {"max_tokens": 4096}),
        )
        mock_init.return_value = _make_init_chat_model_mock()

        create_model("anthropic:claude-sonnet-4-5")

        _, call_kwargs = mock_init.call_args
        assert call_kwargs.get("max_tokens") == 4096

    @patch("langchain.chat_models.init_chat_model")
    def test_pre_init_invoked_exactly_once(
        self,
        mock_init: Mock,
        _isolate_provider_profiles: None,  # noqa: PT019
    ) -> None:
        """`pre_init` runs once per `create_model` call (not duplicated by CLI path).

        Pins the consolidation: previously the CLI inline-called
        `check_openrouter_version` *and* the SDK profile's `pre_init` ran the
        same check, firing it twice. Only the profile path should run it now.
        """
        from deepagents.profiles.provider import (
            ProviderProfile,
            register_provider_profile,
        )

        pre_init_calls: list[str] = []
        register_provider_profile(
            "anthropic",
            ProviderProfile(pre_init=lambda spec: pre_init_calls.append(spec)),
        )
        mock_init.return_value = _make_init_chat_model_mock()

        create_model("anthropic:claude-sonnet-4-5")

        assert pre_init_calls == ["anthropic:claude-sonnet-4-5"]

    @patch("deepagents.profiles.provider.apply_provider_profile")
    @patch("langchain.chat_models.init_chat_model")
    def test_no_provider_skips_profile_lookup(
        self, mock_init: Mock, mock_apply: Mock
    ) -> None:
        """Bare model spec with no detected provider skips profile resolution.

        `detect_provider` returns `None` for unrecognized model names; the
        `if provider:` guard in `create_model` keeps the profile call out of
        that path so an empty spec is never sent to `apply_provider_profile`.
        """
        mock_init.return_value = _make_init_chat_model_mock()

        create_model("some-unknown-model-name")

        mock_apply.assert_not_called()

    @patch("langchain.chat_models.init_chat_model")
    def test_profile_pre_init_failure_wrapped_in_model_config_error(
        self,
        mock_init: Mock,
        _isolate_provider_profiles: None,  # noqa: PT019
    ) -> None:
        """Arbitrary `pre_init` exceptions surface as `ModelConfigError`.

        Without wrapping, a profile's `pre_init` failure bubbles up as a raw
        traceback to the user; the CLI's error path expects `ModelConfigError`
        for actionable rendering.
        """
        from deepagents.profiles.provider import (
            ProviderProfile,
            register_provider_profile,
        )

        def _broken_pre_init(_spec: str) -> None:
            msg = "boom"
            raise RuntimeError(msg)

        register_provider_profile(
            "anthropic",
            ProviderProfile(pre_init=_broken_pre_init),
        )
        mock_init.return_value = _make_init_chat_model_mock()

        with pytest.raises(ModelConfigError, match="provider profile"):
            create_model("anthropic:claude-sonnet-4-5")


class TestCreateModelFromClass:
    """Tests for _create_model_from_class() custom class factory."""

    def test_raises_on_invalid_class_path_format(self) -> None:
        """Raises ModelConfigError when class_path lacks colon."""
        from deepagents_code.model_config import ModelConfigError

        with pytest.raises(ModelConfigError, match="Invalid class_path"):
            _create_model_from_class("my_package.MyChatModel", "model", "provider", {})

    def test_raises_on_import_error(self) -> None:
        """Raises ModelConfigError when module cannot be imported."""
        from deepagents_code.model_config import ModelConfigError

        with pytest.raises(ModelConfigError, match="Could not import module"):
            _create_model_from_class(
                "nonexistent_package_xyz.models:MyModel", "model", "provider", {}
            )

    def test_raises_when_class_not_found_in_module(self) -> None:
        """Raises ModelConfigError when class doesn't exist in module."""
        from deepagents_code.model_config import ModelConfigError

        with pytest.raises(ModelConfigError, match="not found in module"):
            _create_model_from_class("os.path:NonExistentClass", "m", "p", {})

    def test_raises_when_not_base_chat_model_subclass(self) -> None:
        """Raises ModelConfigError when class is not a BaseChatModel."""
        from deepagents_code.model_config import ModelConfigError

        # os.path:join is a function, not a BaseChatModel subclass
        with pytest.raises(ModelConfigError, match="not a BaseChatModel subclass"):
            _create_model_from_class("os.path:sep", "m", "p", {})

    def test_instantiates_valid_subclass(self) -> None:
        """Successfully instantiates a valid BaseChatModel subclass."""
        from unittest.mock import MagicMock

        from langchain_core.callbacks import CallbackManagerForLLMRun
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import BaseMessage
        from langchain_core.outputs import ChatResult

        # Track what args the constructor receives
        captured: dict[str, object] = {}

        class FakeChatModel(BaseChatModel):
            """Minimal BaseChatModel subclass for testing."""

            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            def _generate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: object,
            ) -> ChatResult:
                msg = "not implemented"
                raise NotImplementedError(msg)

            @property
            def _llm_type(self) -> str:
                return "fake"

        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.MyChatModel = FakeChatModel
            mock_import.return_value = mock_module

            result = _create_model_from_class(
                "my_pkg:MyChatModel", "my-model", "custom", {"temp": 0}
            )

        assert isinstance(result, FakeChatModel)
        assert captured["model"] == "my-model"
        assert captured["temp"] == 0

    def test_raises_on_instantiation_error(self) -> None:
        """Raises ModelConfigError when constructor fails."""
        from unittest.mock import MagicMock

        from langchain_core.language_models import BaseChatModel

        from deepagents_code.model_config import ModelConfigError

        class BadModel(BaseChatModel):
            def __init__(self, **kwargs: object) -> None:
                pass

        with (
            patch("importlib.import_module") as mock_import,
            patch.object(BadModel, "__init__", side_effect=TypeError("bad args")),
        ):
            mock_module = MagicMock()
            mock_module.BadModel = BadModel
            mock_import.return_value = mock_module

            with pytest.raises(ModelConfigError, match="Failed to instantiate"):
                _create_model_from_class("my_pkg:BadModel", "model", "custom", {})


class TestCreateModelWithCustomClass:
    """Tests for create_model() using custom class_path from config."""

    def setup_method(self) -> None:
        """Clear model config cache before each test."""
        clear_caches()

    def test_create_model_uses_class_path(self, tmp_path: Path) -> None:
        """create_model dispatches to custom class when class_path is set."""
        from unittest.mock import MagicMock

        from langchain_core.language_models import BaseChatModel

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.custom]
class_path = "my_pkg.models:MyChatModel"
models = ["my-model"]

[models.providers.custom.params]
temperature = 0
""")
        mock_instance = MagicMock(spec=BaseChatModel)
        mock_instance.profile = None

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch(
                "deepagents_code.config._create_model_from_class",
                return_value=mock_instance,
            ) as mock_factory,
        ):
            result = create_model("custom:my-model")

        mock_factory.assert_called_once()
        call_args = mock_factory.call_args
        assert call_args[0][0] == "my_pkg.models:MyChatModel"
        assert call_args[0][1] == "my-model"
        assert call_args[0][2] == "custom"
        assert isinstance(result, ModelResult)
        assert result.model is mock_instance
        assert result.model_name == "my-model"
        assert result.provider == "custom"

    def test_configured_provider_takes_precedence_over_bedrock_inference(
        self, tmp_path: Path
    ) -> None:
        """A configured explicit provider is not treated as a bare Bedrock ID."""
        from unittest.mock import MagicMock

        from langchain_core.language_models import BaseChatModel

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers."meta.custom"]
class_path = "my_pkg.models:MyChatModel"
models = ["my-model"]
""")
        mock_instance = MagicMock(spec=BaseChatModel)
        mock_instance.profile = None

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch(
                "deepagents_code.config._create_model_from_class",
                return_value=mock_instance,
            ) as mock_factory,
        ):
            result = create_model("meta.custom:my-model")

        mock_factory.assert_called_once()
        assert mock_factory.call_args.args[1:3] == ("my-model", "meta.custom")
        assert result.model is mock_instance
        assert result.model_name == "my-model"
        assert result.provider == "meta.custom"

    def test_create_model_falls_through_without_class_path(
        self, tmp_path: Path
    ) -> None:
        """create_model uses init_chat_model when no class_path is set."""
        from unittest.mock import MagicMock

        from langchain_core.language_models import BaseChatModel

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.fireworks]
models = ["llama"]
api_key_env = "FIREWORKS_API_KEY"
""")
        mock_instance = MagicMock(spec=BaseChatModel)
        mock_instance.profile = None

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"FIREWORKS_API_KEY": "key"}, clear=False),
            patch(
                "deepagents_code.config._create_model_via_init",
                return_value=mock_instance,
            ) as mock_init,
        ):
            result = create_model("fireworks:llama")

        mock_init.assert_called_once()
        assert result.model is mock_instance


class TestCreateModelExtraKwargs:
    """Tests for create_model() with extra_kwargs from --model-params."""

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_extra_kwargs_passed_to_model(self, mock_init_chat_model: Mock) -> None:
        """extra_kwargs are forwarded to init_chat_model."""
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        create_model("anthropic:claude-sonnet-4-5", extra_kwargs={"temperature": 0.7})

        _, call_kwargs = mock_init_chat_model.call_args
        assert call_kwargs["temperature"] == pytest.approx(0.7)

    @patch("langchain.chat_models.init_chat_model")
    def test_extra_kwargs_override_config(
        self, mock_init_chat_model: Mock, tmp_path: Path
    ) -> None:
        """extra_kwargs override values from config file."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5"]

[models.providers.anthropic.params]
temperature = 0
max_tokens = 1024
""")
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        clear_caches()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            create_model(
                "anthropic:claude-sonnet-4-5",
                extra_kwargs={"temperature": 0.9},
            )

        _, call_kwargs = mock_init_chat_model.call_args
        # CLI kwarg wins over config
        assert call_kwargs["temperature"] == pytest.approx(0.9)
        # Config kwarg preserved when not overridden
        assert call_kwargs["max_tokens"] == 1024

    @patch("langchain.chat_models.init_chat_model")
    def test_none_extra_kwargs_is_noop(self, mock_init_chat_model: Mock) -> None:
        """extra_kwargs=None does not affect behavior."""
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        create_model("anthropic:claude-sonnet-4-5", extra_kwargs=None)
        mock_init_chat_model.assert_called_once()

    @patch("langchain.chat_models.init_chat_model")
    def test_empty_extra_kwargs_is_noop(self, mock_init_chat_model: Mock) -> None:
        """extra_kwargs={} does not affect behavior."""
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        create_model("anthropic:claude-sonnet-4-5", extra_kwargs={})
        mock_init_chat_model.assert_called_once()


class TestCreateModelEdgeCaseParsing:
    """Tests for create_model() edge-case spec parsing."""

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_leading_colon_treated_as_bare_model(
        self, mock_init_chat_model: Mock
    ) -> None:
        """Leading colon (e.g., ':claude-opus-4-6') is treated as bare model name."""
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        settings.anthropic_api_key = "test"
        try:
            result = create_model(":claude-opus-4-6")
        finally:
            settings.anthropic_api_key = None

        # Should have detected 'anthropic' provider and used 'claude-opus-4-6'
        assert result.model_name == "claude-opus-4-6"

    @patch("langchain.chat_models.init_chat_model")
    def test_versioned_bedrock_id_treated_as_bare_model(
        self, mock_init_chat_model: Mock
    ) -> None:
        """A Bedrock version suffix is not parsed as a provider separator."""
        model_id = "meta.llama3-70b-instruct-v1:0"
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        result = create_model(model_id)

        assert result.provider == "bedrock"
        assert result.model_name == model_id
        assert mock_init_chat_model.call_args.args == (model_id,)
        assert mock_init_chat_model.call_args.kwargs["model_provider"] == "bedrock"

    @patch("langchain.chat_models.init_chat_model")
    def test_versioned_bedrock_vendor_id_not_misparsed(
        self, mock_init_chat_model: Mock
    ) -> None:
        """A Bedrock vendor namespace is not split on its `:version` suffix.

        Regression: `mistral.` collides with the bare `mistral` prefix, so
        without dotted-namespace detection this ID parses to the garbage pair
        `provider='mistral.mistral-large-2402-v1', model='0'`.
        """
        model_id = "mistral.mistral-large-2402-v1:0"
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        result = create_model(model_id)

        assert result.provider == "bedrock"
        assert result.model_name == model_id
        assert mock_init_chat_model.call_args.args == (model_id,)
        assert mock_init_chat_model.call_args.kwargs["model_provider"] == "bedrock"

    @patch("langchain.chat_models.init_chat_model")
    def test_cross_region_bedrock_id_treated_as_bare_model(
        self, mock_init_chat_model: Mock
    ) -> None:
        """A cross-region inference-profile ID resolves to Bedrock intact."""
        model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        result = create_model(model_id)

        assert result.provider == "bedrock"
        assert result.model_name == model_id
        assert mock_init_chat_model.call_args.args == (model_id,)
        assert mock_init_chat_model.call_args.kwargs["model_provider"] == "bedrock"

    @patch("langchain.chat_models.init_chat_model")
    def test_non_versioned_bedrock_id_treated_as_bare_model(
        self, mock_init_chat_model: Mock
    ) -> None:
        """A Bedrock ID without a `:version` suffix still routes to Bedrock."""
        model_id = "amazon.titan-text-express-v1"
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        result = create_model(model_id)

        assert result.provider == "bedrock"
        assert result.model_name == model_id
        assert mock_init_chat_model.call_args.kwargs["model_provider"] == "bedrock"

    @patch("langchain.chat_models.init_chat_model")
    def test_explicit_provider_not_hijacked_by_bedrock(
        self, mock_init_chat_model: Mock
    ) -> None:
        """An explicit `provider:model` spec wins over Bedrock inference.

        `anthropic.` (dot) is a Bedrock namespace, but `anthropic:` (colon) is
        the explicit-provider syntax and must resolve to Anthropic, not Bedrock.
        """
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        result = create_model("anthropic:claude-sonnet-4-5")

        assert result.provider == "anthropic"
        assert result.model_name == "claude-sonnet-4-5"

    def test_trailing_colon_raises_error(self) -> None:
        """Trailing colon (e.g., 'anthropic:') raises ModelConfigError."""
        with pytest.raises(ModelConfigError, match="model name is required"):
            create_model("anthropic:")

    @patch("deepagents_code.config._get_default_model_spec")
    @patch("langchain.chat_models.init_chat_model")
    def test_empty_string_uses_default(
        self, mock_init_chat_model: Mock, mock_default: Mock
    ) -> None:
        """Empty string falls through to _get_default_model_spec."""
        mock_default.return_value = "openai:gpt-5.5"
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        create_model("")
        mock_default.assert_called_once()


class TestCreateModelViaInitImportError:
    """Tests for _create_model_via_init() ImportError handling."""

    @patch("langchain.chat_models.init_chat_model")
    def test_missing_package_error(self, mock_init: Mock) -> None:
        """Shows install hint when provider package is not installed."""
        from deepagents_code.model_config import MissingProviderPackageError

        mock_init.side_effect = ImportError(
            "No module named 'langchain_nvidia_ai_endpoints'"
        )
        with (
            patch("importlib.util.find_spec", return_value=None),
            pytest.raises(
                MissingProviderPackageError,
                match="Missing package for provider 'nvidia'",
            ) as exc_info,
        ):
            _create_model_via_init("nemotron", "nvidia", {})
        assert exc_info.value.provider == "nvidia"
        assert exc_info.value.package == "langchain-nvidia-ai-endpoints"
        # Subclasses ModelConfigError so existing handlers keep working.
        assert isinstance(exc_info.value, ModelConfigError)

    @patch("langchain.chat_models.init_chat_model")
    def test_missing_vertexai_package_uses_declared_extra(
        self, mock_init: Mock
    ) -> None:
        """Vertex AI provider id does not match its optional extra name."""
        from deepagents_code.model_config import MissingProviderPackageError

        mock_init.side_effect = ImportError(
            "No module named 'langchain_google_vertexai'"
        )
        with (
            patch("importlib.util.find_spec", return_value=None),
            patch(
                "deepagents_code.extras_info.extra_for_package",
                return_value="vertex",
            ) as mock_extra_for_package,
            pytest.raises(
                MissingProviderPackageError,
                match=r"Install: /install vertex",
            ) as exc_info,
        ):
            _create_model_via_init("claude-sonnet-4-5", "google_vertexai", {})
        mock_extra_for_package.assert_called_once_with(
            "langchain-google-vertexai", "deepagents-code"
        )
        assert exc_info.value.provider == "google_vertexai"
        assert exc_info.value.package == "langchain-google-vertexai"

    @patch("langchain.chat_models.init_chat_model")
    def test_installed_but_broken_import(self, mock_init: Mock) -> None:
        """Shows real error when package is installed but import fails internally."""
        mock_init.side_effect = ImportError("cannot import name 'foo' from 'bar'")
        mock_spec = Mock()
        with (
            patch("importlib.util.find_spec", return_value=mock_spec) as mock_find_spec,
            pytest.raises(
                ModelConfigError,
                match="installed but failed to import",
            ),
        ):
            _create_model_via_init("nemotron", "nvidia", {})
        mock_find_spec.assert_called_once_with("langchain_nvidia_ai_endpoints")

    @patch("langchain.chat_models.init_chat_model")
    def test_installed_but_broken_includes_original_error(
        self, mock_init: Mock
    ) -> None:
        """Original ImportError message is included when package is installed."""
        mock_init.side_effect = ImportError("some transitive dep missing")
        mock_spec = Mock()
        with (
            patch("importlib.util.find_spec", return_value=mock_spec),
            pytest.raises(ModelConfigError, match="some transitive dep missing"),
        ):
            _create_model_via_init("nemotron", "nvidia", {})

    @patch("langchain.chat_models.init_chat_model")
    def test_unknown_provider_fallback_package_name(
        self, mock_init: Mock, tmp_path, monkeypatch
    ) -> None:
        """Unknown provider falls back to langchain-{provider} package name.

        `install_package_command` reads the uv tool receipt, so it is isolated to
        a temporary tool root here; otherwise the hint degrades to the manual
        fallback (see the sibling unreadable-receipt test).
        """
        tmp_path.joinpath("uv-receipt.toml").write_text(
            '[tool]\nrequirements = [{ name = "deepagents-code" }]\n',
            encoding="utf-8",
        )
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        mock_init.side_effect = ImportError("no module")
        with (
            patch("importlib.util.find_spec", return_value=None),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value=set(),
            ),
            pytest.raises(
                ModelConfigError,
                match=(
                    "Install with: uv tool install --reinstall -U "
                    f"deepagents-code=={__version__} "
                    "--with langchain-custom_provider --prerelease allow"
                ),
            ),
        ):
            _create_model_via_init("some-model", "custom_provider", {})

    @patch("langchain.chat_models.init_chat_model")
    def test_unknown_provider_receipt_failure_falls_back_to_manual(
        self, mock_init: Mock, tmp_path, monkeypatch
    ) -> None:
        """An unreadable uv receipt degrades to the manual-install hint.

        Exercises the `ToolRequirementIntrospectionError` arm of the fallback:
        `install_package_command` reads the uv tool receipt, and a missing
        receipt must surface an actionable message instead of letting the error
        leak out of hint construction.
        """
        # tmp_path has no uv-receipt.toml, so the receipt read raises.
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        mock_init.side_effect = ImportError("no module")
        with (
            patch("importlib.util.find_spec", return_value=None),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value=set(),
            ),
            pytest.raises(
                ModelConfigError,
                match="Install the 'langchain-custom_provider' package manually",
            ),
        ):
            _create_model_via_init("some-model", "custom_provider", {})

    @patch("langchain.chat_models.init_chat_model")
    def test_unknown_provider_introspection_failure_falls_back_to_manual(
        self, mock_init: Mock
    ) -> None:
        """Unreadable extras metadata degrades to the manual-install hint.

        Exercises the `ExtrasIntrospectionError` arm of the fallback so the
        user still gets an actionable message instead of an unhandled error
        leaking out of hint construction.
        """
        from deepagents_code.extras_info import ExtrasIntrospectionError

        mock_init.side_effect = ImportError("no module")
        with (
            patch("importlib.util.find_spec", return_value=None),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                side_effect=ExtrasIntrospectionError("metadata unreadable"),
            ),
            pytest.raises(
                ModelConfigError,
                match="Install the 'langchain-custom_provider' package manually",
            ),
        ):
            _create_model_via_init("some-model", "custom_provider", {})

    @patch("langchain.chat_models.init_chat_model")
    def test_find_spec_raises_falls_back_to_missing(self, mock_init: Mock) -> None:
        """find_spec failure falls back to 'missing package' message."""
        mock_init.side_effect = ImportError("no module")
        with (
            patch(
                "importlib.util.find_spec",
                side_effect=ModuleNotFoundError("no parent"),
            ),
            pytest.raises(
                ModelConfigError,
                match="Missing package",
            ),
        ):
            _create_model_via_init("model", "dotted.provider", {})


class TestCreateModelViaInitUnknownProvider:
    """Tests for `UnknownProviderError` translation of langchain inference."""

    @patch("langchain.chat_models.init_chat_model")
    def test_value_error_with_empty_provider_becomes_unknown_provider_error(
        self, mock_init: Mock
    ) -> None:
        """Raise `UnknownProviderError` carrying the model spec and docs URL."""
        from deepagents_code.model_config import (
            PROVIDERS_DOCS_URL,
            UnknownProviderError,
        )

        mock_init.side_effect = ValueError(
            "Unable to infer model provider for model='mystery-model'."
        )
        with pytest.raises(UnknownProviderError) as exc_info:
            _create_model_via_init("mystery-model", "", {})

        assert exc_info.value.model_spec == "mystery-model"
        assert exc_info.value.docs_url == PROVIDERS_DOCS_URL
        # Plain message still mentions the URL for non-Textual surfaces.
        assert PROVIDERS_DOCS_URL in str(exc_info.value)

    @patch("langchain.chat_models.init_chat_model")
    def test_value_error_with_provider_stays_generic(self, mock_init: Mock) -> None:
        """Plain `ModelConfigError` when a provider was passed (not inference)."""
        from deepagents_code.model_config import UnknownProviderError

        mock_init.side_effect = ValueError("some other configuration problem")
        with pytest.raises(ModelConfigError) as exc_info:
            _create_model_via_init("claude-sonnet-4-5", "anthropic", {})

        assert not isinstance(exc_info.value, UnknownProviderError)
        assert "Invalid model configuration" in str(exc_info.value)


class TestDetectProvider:
    """Tests for detect_provider() auto-detection from model names."""

    @pytest.mark.parametrize(
        ("model_name", "expected"),
        [
            ("gpt-5.5", "openai"),
            ("gpt-5.2", "openai"),
            ("o1-preview", "openai"),
            ("o3-mini", "openai"),
            ("o4-mini", "openai"),
            ("text-davinci-003", "openai"),
            ("command-r-plus", "cohere"),
            ("amazon.titan-text-express-v1", "bedrock"),
            ("anthropic.claude-3-sonnet", "bedrock"),
            ("meta.llama3-70b-instruct-v1:0", "bedrock"),
            # Bedrock vendor namespaces that collide with the bare direct-API
            # prefixes below: the dotted form must win so the `:version` suffix
            # is not misparsed as a `provider:model` separator.
            ("mistral.mistral-large-2402-v1:0", "bedrock"),
            ("deepseek.r1-v1:0", "bedrock"),
            ("cohere.command-r-v1:0", "bedrock"),
            ("ai21.jamba-1-5-large-v1:0", "bedrock"),
            ("writer.palmyra-x5-v1:0", "bedrock"),
            # Structural detection covers vendors with no hardcoded entry.
            ("qwen.qwen3-32b-v1:0", "bedrock"),
            ("google.gemma-3-27b-v1:0", "bedrock"),
            # Cross-region inference-profile IDs front the vendor with a region.
            ("us.anthropic.claude-3-5-sonnet-20241022-v2:0", "bedrock"),
            ("eu.meta.llama3-2-3b-instruct-v1:0", "bedrock"),
            ("apac.anthropic.claude-3-5-sonnet-20241022-v2:0", "bedrock"),
            ("US.Anthropic.Claude-3-5-Sonnet-20241022-v2:0", "bedrock"),
            # A bare name that merely starts with a region token is not Bedrock.
            ("useful-model", None),
            ("mistral-large", "mistralai"),
            ("mixtral-8x7b-instruct", "mistralai"),
            ("deepseek-chat", "deepseek"),
            ("grok-4", "xai"),
            ("sonar-pro", "perplexity"),
            ("claude-sonnet-4-5", "anthropic"),
            ("claude-opus-4-5", "anthropic"),
            ("gemini-3.1-pro-preview", "google_genai"),
            ("nemotron-3-nano-30b-a3b", "nvidia"),
            ("nvidia/nemotron-3-nano-30b-a3b", "nvidia"),
            ("accounts/fireworks/models/kimi-k2p7-code", "fireworks"),
            ("accounts/fireworks/routers/kimi-k2p7-code", "fireworks"),
            ("Accounts/Fireworks/Models/Kimi-K2P7-Code", "fireworks"),
            ("accounts/openai/models/gpt-5.5", None),
            # A different account whose name merely starts with "fireworks"
            # must not resolve to Fireworks; the trailing slash in the prefix
            # is what anchors the match to the exact account namespace.
            ("accounts/fireworks-enterprise/models/kimi-k2p7-code", None),
            ("llama3", None),
            ("solar-pro", None),
            ("some-unknown-model", None),
        ],
    )
    def test_detect_known_patterns(self, model_name: str, expected: str | None) -> None:
        """detect_provider returns the correct provider for known patterns."""
        # Ensure both Anthropic and Google credentials are "available" so the
        # default paths are taken (not the Vertex AI fallbacks).
        settings.anthropic_api_key = "test"
        settings.google_api_key = "test"
        try:
            assert detect_provider(model_name) == expected
        finally:
            settings.anthropic_api_key = None
            settings.google_api_key = None

    def test_claude_falls_back_to_vertex_when_no_anthropic(self) -> None:
        """Claude models route to google_vertexai when only Vertex AI is configured."""
        settings.anthropic_api_key = None
        settings.google_cloud_project = "my-project"
        settings.google_api_key = None
        try:
            assert detect_provider("claude-sonnet-4-5") == "google_vertexai"
        finally:
            settings.google_cloud_project = None

    def test_gemini_falls_back_to_vertex_when_no_google(self) -> None:
        """Gemini models route to google_vertexai when only Vertex AI is configured."""
        settings.google_api_key = None
        settings.google_cloud_project = "my-project"
        try:
            assert detect_provider("gemini-3-pro") == "google_vertexai"
        finally:
            settings.google_cloud_project = None

    def test_gemini_prefers_google_genai_when_both_available(self) -> None:
        """Gemini prefers google_genai when both Google and Vertex AI are configured."""
        settings.google_api_key = "test"
        settings.google_cloud_project = "my-project"
        try:
            # has_vertex_ai is False when google_api_key is set, so this
            # tests the google_genai path which is preferred.
            assert detect_provider("gemini-3-pro") == "google_genai"
        finally:
            settings.google_api_key = None
            settings.google_cloud_project = None

    def test_case_insensitive(self) -> None:
        """detect_provider is case-insensitive."""
        settings.anthropic_api_key = "test"
        try:
            assert detect_provider("Claude-Sonnet-4-5") == "anthropic"
            assert detect_provider("gpt-5.5") == "openai"
        finally:
            settings.anthropic_api_key = None


class TestLazyModuleAttributes:
    """Tests for lazy `__getattr__` resolution of `settings` and `console`."""

    def test_getattr_returns_settings(self) -> None:
        """Module __getattr__ resolves 'settings' to a Settings instance."""
        from deepagents_code.config import _get_settings

        result = _get_settings()
        assert isinstance(result, Settings)

    def test_getattr_returns_console(self) -> None:
        """Module __getattr__ resolves 'console' to a Console instance."""
        from rich.console import Console

        from deepagents_code.config import _get_console

        result = _get_console()
        assert isinstance(result, Console)

    def test_getattr_raises_for_unknown(self) -> None:
        """Module __getattr__ raises AttributeError for unknown names."""
        import deepagents_code.config as config_mod

        with pytest.raises(AttributeError, match="no attribute"):
            getattr(config_mod, "nonexistent_attr_xyz")  # noqa: B009  # intentional __getattr__ test

    def test_ensure_bootstrap_is_idempotent(self) -> None:
        """_ensure_bootstrap is a no-op on second call."""
        from deepagents_code.config import _ensure_bootstrap

        # First call already ran (settings was imported above).
        # Calling again should be a harmless no-op.
        _ensure_bootstrap()
        assert isinstance(settings, Settings)

    def test_ensure_bootstrap_marks_done_on_failure(self) -> None:
        """_ensure_bootstrap sets flag even when the try body raises."""
        import deepagents_code.config as config_mod
        from deepagents_code.config import _ensure_bootstrap

        # Reset flag so bootstrap will re-enter
        original = config_mod._bootstrap_state.done
        config_mod._bootstrap_state.done = False

        try:
            with patch(
                "deepagents_code.config._load_dotenv", side_effect=RuntimeError("boom")
            ):
                _ensure_bootstrap()  # should warn, not raise

            # Flag must be set even after failure
            assert config_mod._bootstrap_state.done is True
        finally:
            config_mod._bootstrap_state.done = original

    def test_get_settings_returns_same_instance(self) -> None:
        """_get_settings caches in globals — two calls return the same object."""
        from deepagents_code.config import _get_settings

        a = _get_settings()
        b = _get_settings()
        assert a is b

    def test_ensure_bootstrap_langsmith_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_ensure_bootstrap copies DEEPAGENTS_CODE_LANGSMITH_PROJECT."""
        import deepagents_code.config as config_mod
        from deepagents_code.config import _ensure_bootstrap

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", "my-agent-project")
            monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
            ):
                _ensure_bootstrap()

            assert config_mod._bootstrap_state.original_langsmith_project is None
            import os

            assert os.environ["LANGSMITH_PROJECT"] == "my-agent-project"
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls

    def test_ensure_bootstrap_preserves_original_langsmith(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_ensure_bootstrap captures original LANGSMITH_PROJECT."""
        import deepagents_code.config as config_mod
        from deepagents_code.config import _ensure_bootstrap

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.setenv("LANGSMITH_PROJECT", "user-project")
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", "agent-project")

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
            ):
                _ensure_bootstrap()

            assert (
                config_mod._bootstrap_state.original_langsmith_project == "user-project"
            )
            import os

            assert os.environ["LANGSMITH_PROJECT"] == "agent-project"
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls

    def test_bootstrap_propagates_prefixed_langsmith_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prefixed LangSmith vars are copied to canonical names at bootstrap."""
        import deepagents_code.config as config_mod
        from deepagents_code.config import _ensure_bootstrap

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_test")
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")
            monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
            monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
            ):
                _ensure_bootstrap()

            import os

            assert os.environ["LANGSMITH_API_KEY"] == "lsv2_test"
            assert os.environ["LANGSMITH_TRACING"] == "true"
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls

    def test_bootstrap_overrides_canonical_with_prefixed_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prefixed value wins when both canonical and prefixed vars are set."""
        import deepagents_code.config as config_mod
        from deepagents_code.config import _ensure_bootstrap

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        original_api_keys = dict(config_mod._bootstrap_state.original_tracing_api_keys)
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_original")
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_override")
            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
            ):
                _ensure_bootstrap()

            import os

            # Prefixed value wins — canonical is overwritten.
            assert os.environ["LANGSMITH_API_KEY"] == "lsv2_override"
            # The original canonical value is saved for subprocess restoration.
            assert (
                config_mod._bootstrap_state.original_tracing_api_keys[
                    "LANGSMITH_API_KEY"
                ]
                == "lsv2_original"
            )
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls
            config_mod._bootstrap_state.original_tracing_api_keys = original_api_keys

    @pytest.mark.parametrize("canonical", ["LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"])
    def test_restore_user_tracing_api_keys_recovers_original_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, canonical: str
    ) -> None:
        """Shell subprocess env gets the caller's original API key, not the override.

        Parametrized over both members of `_TRACING_API_KEY_ENV_VARS` so the
        `LANGCHAIN_API_KEY` alias is covered, not just the primary var.
        """
        import os

        import deepagents_code.config as config_mod
        from deepagents_code.config import (
            _ensure_bootstrap,
            restore_user_tracing_api_keys,
        )

        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / ".state"
        )

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        original_tracing = dict(config_mod._bootstrap_state.original_tracing_env)
        original_api_keys = dict(config_mod._bootstrap_state.original_tracing_api_keys)
        config_mod._bootstrap_state.done = False

        try:
            # Isolate the key var under test so its sibling alias cannot mask it.
            for var in config_mod._TRACING_API_KEY_ENV_VARS:
                monkeypatch.delenv(var, raising=False)
                monkeypatch.delenv(f"DEEPAGENTS_CODE_{var}", raising=False)
            monkeypatch.setenv(canonical, "lsv2_original")
            monkeypatch.setenv(f"DEEPAGENTS_CODE_{canonical}", "lsv2_override")
            monkeypatch.setenv("LANGSMITH_TRACING", "true")
            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
            ):
                _ensure_bootstrap()

            # Bootstrap overwrote the canonical key with the prefixed value.
            assert os.environ[canonical] == "lsv2_override"

            shell_env = os.environ.copy()
            restore_user_tracing_api_keys(shell_env)

            # Shell subprocesses get the caller's original key back.
            assert shell_env[canonical] == "lsv2_original"
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls
            config_mod._bootstrap_state.original_tracing_env = original_tracing
            config_mod._bootstrap_state.original_tracing_api_keys = original_api_keys

    def test_restore_user_tracing_api_keys_drops_unset_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the caller had no canonical key, restore removes it from shell env."""
        import os

        import deepagents_code.config as config_mod
        from deepagents_code.config import (
            _ensure_bootstrap,
            restore_user_tracing_api_keys,
        )

        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / ".state"
        )

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        original_tracing = dict(config_mod._bootstrap_state.original_tracing_env)
        original_api_keys = dict(config_mod._bootstrap_state.original_tracing_api_keys)
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_prefixed")
            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
            ):
                _ensure_bootstrap()

            # Bootstrap propagated the prefixed key to canonical.
            assert os.environ["LANGSMITH_API_KEY"] == "lsv2_prefixed"

            shell_env = os.environ.copy()
            restore_user_tracing_api_keys(shell_env)

            # Caller had no key — the propagated value is removed from shell env.
            assert "LANGSMITH_API_KEY" not in shell_env
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls
            config_mod._bootstrap_state.original_tracing_env = original_tracing
            config_mod._bootstrap_state.original_tracing_api_keys = original_api_keys

    def test_restore_user_tracing_api_keys_pops_auth_stored_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `/auth`-stored key is bridged onto the env but popped for shells.

        The snapshot is captured *before* `apply_stored_langsmith_auth` bridges
        the stored key onto `LANGSMITH_API_KEY`, so the caller's original is
        `None` and restore pops the bridged key instead of leaking the agent's
        stored credential into `execute` subprocesses. Locks in the
        capture-before-bridge ordering that a bootstrap refactor could break.
        """
        import os

        import deepagents_code.config as config_mod
        from deepagents_code import auth_store
        from deepagents_code.config import (
            _ensure_bootstrap,
            restore_user_tracing_api_keys,
        )

        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / ".state"
        )

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        original_tracing = dict(config_mod._bootstrap_state.original_tracing_env)
        original_api_keys = dict(config_mod._bootstrap_state.original_tracing_api_keys)
        config_mod._bootstrap_state.done = False

        try:
            # No env or prefixed key — only a `/auth`-stored credential.
            for var in (
                "LANGSMITH_API_KEY",
                "LANGCHAIN_API_KEY",
                "DEEPAGENTS_CODE_LANGSMITH_API_KEY",
                "DEEPAGENTS_CODE_LANGCHAIN_API_KEY",
                "DEEPAGENTS_CODE_LANGSMITH_PROJECT",
            ):
                monkeypatch.delenv(var, raising=False)
            auth_store.set_stored_key("langsmith", "lsv2_stored")

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
            ):
                _ensure_bootstrap()

            # Bootstrap bridged the stored key onto the canonical env var...
            assert os.environ["LANGSMITH_API_KEY"] == "lsv2_stored"
            # ...but the caller had none, so the snapshot (taken before the
            # bridge) records it as absent.
            assert (
                config_mod._bootstrap_state.original_tracing_api_keys[
                    "LANGSMITH_API_KEY"
                ]
                is None
            )

            shell_env = os.environ.copy()
            restore_user_tracing_api_keys(shell_env)

            # The agent's stored credential is not leaked into shell subprocesses.
            assert "LANGSMITH_API_KEY" not in shell_env
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls
            config_mod._bootstrap_state.original_tracing_env = original_tracing
            config_mod._bootstrap_state.original_tracing_api_keys = original_api_keys

    def test_bootstrap_warns_on_conflicting_override(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A conflicting prefixed override logs a single explanatory warning."""
        import logging

        import deepagents_code.config as config_mod
        from deepagents_code.config import _ensure_bootstrap

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        original_api_keys = dict(config_mod._bootstrap_state.original_tracing_api_keys)
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_original")
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_override")
            monkeypatch.delenv(
                "DEEPAGENTS_CODE_SUPPRESS_ENV_OVERRIDE_WARNING", raising=False
            )
            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
                caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
            ):
                _ensure_bootstrap()

            warnings = [
                r.getMessage()
                for r in caplog.records
                if "DEEPAGENTS_CODE_LANGSMITH_API_KEY" in r.getMessage()
            ]
            assert len(warnings) == 1
            assert "DEEPAGENTS_CODE_SUPPRESS_ENV_OVERRIDE_WARNING=1" in warnings[0]
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls
            config_mod._bootstrap_state.original_tracing_api_keys = original_api_keys

    def test_bootstrap_suppresses_override_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The suppression flag silences the override warning but keeps the override."""
        import logging
        import os

        import deepagents_code.config as config_mod
        from deepagents_code.config import _ensure_bootstrap

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        original_api_keys = dict(config_mod._bootstrap_state.original_tracing_api_keys)
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_original")
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_override")
            monkeypatch.setenv("DEEPAGENTS_CODE_SUPPRESS_ENV_OVERRIDE_WARNING", "1")
            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
                caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
            ):
                _ensure_bootstrap()

            # Override still applies; only the warning is silenced.
            assert os.environ["LANGSMITH_API_KEY"] == "lsv2_override"
            assert not [
                r
                for r in caplog.records
                if "DEEPAGENTS_CODE_LANGSMITH_API_KEY" in r.getMessage()
            ]
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls
            config_mod._bootstrap_state.original_tracing_api_keys = original_api_keys

    def test_bootstrap_no_warning_when_values_match(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Matching canonical and prefixed values propagate without warning."""
        import logging
        import os

        import deepagents_code.config as config_mod
        from deepagents_code.config import _ensure_bootstrap

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        original_api_keys = dict(config_mod._bootstrap_state.original_tracing_api_keys)
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_same")
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_same")
            monkeypatch.delenv(
                "DEEPAGENTS_CODE_SUPPRESS_ENV_OVERRIDE_WARNING", raising=False
            )
            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
                caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
            ):
                _ensure_bootstrap()

            # No conflict, so no warning; the shared value stays in place.
            assert os.environ["LANGSMITH_API_KEY"] == "lsv2_same"
            assert not [
                r
                for r in caplog.records
                if "DEEPAGENTS_CODE_LANGSMITH_API_KEY" in r.getMessage()
            ]
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls
            config_mod._bootstrap_state.original_tracing_api_keys = original_api_keys

    def test_bootstrap_propagates_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty prefixed var propagates to canonical (explicit disable)."""
        import deepagents_code.config as config_mod
        from deepagents_code.config import _ensure_bootstrap

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "")
            monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
            ):
                _ensure_bootstrap()

            import os

            # Empty string propagated — lets user explicitly disable tracing.
            assert os.environ["LANGSMITH_TRACING"] == ""
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls

    def test_bootstrap_defaults_project_when_tracing_and_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tracing on with a key but no project defaults to deepagents-code.

        Exercises `_apply_default_langsmith_project` wired into the real
        `_ensure_bootstrap` flow (after the override and orphaned-tracing
        steps) — coverage the helper-level tests cannot provide.
        """
        import os

        import deepagents_code.config as config_mod
        from deepagents_code.config import _ensure_bootstrap
        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.setenv("LANGSMITH_TRACING", "true")
            monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
            monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
            ):
                _ensure_bootstrap()

            assert os.environ["LANGSMITH_PROJECT"] == LANGSMITH_PROJECT_DEFAULT
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls

    def test_bootstrap_keyless_tracing_leaves_project_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Keyless tracing is disabled first, so no default project is applied.

        Regression guard for the ordering between `_disable_orphaned_tracing`
        and `_apply_default_langsmith_project`: a tracing flag with no
        resolvable key must be flipped off *before* the default runs, so
        `LANGSMITH_PROJECT` is left unset (tracing never starts) rather than
        pointed at `deepagents-code`.
        """
        import os

        import deepagents_code.config as config_mod
        from deepagents_code.config import _ensure_bootstrap

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.setenv("LANGSMITH_TRACING", "true")
            monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
            monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
            monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
            monkeypatch.delenv("LANGCHAIN_ENDPOINT", raising=False)
            monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
                patch(
                    "deepagents_code.config._has_langsmith_profile_credentials",
                    return_value=False,
                ),
                patch(
                    "deepagents_code.config._has_langsmith_profile_custom_endpoint",
                    return_value=False,
                ),
            ):
                _ensure_bootstrap()

            assert "LANGSMITH_PROJECT" not in os.environ
            assert os.environ["LANGSMITH_TRACING"] == "false"
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls

    def test_bootstrap_stored_langsmith_key_keeps_tracing_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `/auth`-stored LangSmith key survives `_disable_orphaned_tracing`.

        End-to-end regression guard for the bootstrap ordering: a key stored via
        `/auth` (never exported to the env) must be bridged onto
        `LANGSMITH_API_KEY` and auto-enable tracing *before*
        `_disable_orphaned_tracing` runs, so the orphan guard sees the key and
        leaves tracing on. The helper-level `_apply_stored_langsmith_tracing`
        tests cannot catch a regression that reorders the two bootstrap steps.
        """
        import os

        import deepagents_code.config as config_mod
        from deepagents_code import auth_store
        from deepagents_code.config import _ensure_bootstrap

        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / ".state"
        )

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        original_tracing = dict(config_mod._bootstrap_state.original_tracing_env)
        original_api_keys = dict(config_mod._bootstrap_state.original_tracing_api_keys)
        config_mod._bootstrap_state.done = False

        try:
            for var in (
                "LANGSMITH_TRACING",
                "LANGCHAIN_TRACING_V2",
                "LANGSMITH_API_KEY",
                "LANGCHAIN_API_KEY",
                "LANGSMITH_PROJECT",
                "DEEPAGENTS_CODE_LANGSMITH_TRACING",
                "DEEPAGENTS_CODE_LANGSMITH_PROJECT",
            ):
                monkeypatch.delenv(var, raising=False)
            auth_store.set_stored_key("langsmith", "lsv2_stored")

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
                patch(
                    "deepagents_code.config._has_langsmith_profile_credentials",
                    return_value=False,
                ),
                patch(
                    "deepagents_code.config._has_langsmith_profile_custom_endpoint",
                    return_value=False,
                ),
            ):
                _ensure_bootstrap()

            # The stored key was bridged onto the canonical env var, and tracing
            # stayed on instead of being disabled as orphaned.
            assert os.environ["LANGSMITH_API_KEY"] == "lsv2_stored"
            assert os.environ["LANGSMITH_TRACING"] == "true"
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls
            config_mod._bootstrap_state.original_tracing_env = original_tracing
            config_mod._bootstrap_state.original_tracing_api_keys = original_api_keys

    def test_bootstrap_prefixed_langsmith_key_wins_over_stored_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A session-scoped LangSmith key remains authoritative at bootstrap."""
        import os

        import deepagents_code.config as config_mod
        from deepagents_code import auth_store
        from deepagents_code.config import _ensure_bootstrap

        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / ".state"
        )

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        original_tracing = dict(config_mod._bootstrap_state.original_tracing_env)
        original_api_keys = dict(config_mod._bootstrap_state.original_tracing_api_keys)
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_prefixed")
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")
            monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
            monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
            monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
            monkeypatch.delenv("LANGCHAIN_ENDPOINT", raising=False)
            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)
            auth_store.set_stored_key(
                "langsmith", "lsv2_stored", base_url=LANGSMITH_EU_ENDPOINT
            )

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
            ):
                _ensure_bootstrap()

            assert os.environ["LANGSMITH_API_KEY"] == "lsv2_prefixed"
            assert os.environ["LANGSMITH_TRACING"] == "true"
            assert "LANGSMITH_ENDPOINT" not in os.environ
            assert "LANGCHAIN_ENDPOINT" not in os.environ
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls
            config_mod._bootstrap_state.original_tracing_env = original_tracing
            config_mod._bootstrap_state.original_tracing_api_keys = original_api_keys

    def test_scoped_tracing_opt_out_restores_user_tracing_for_shell_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deep Agents Code opt-out does not leak into child command envs."""
        import os

        import deepagents_code.config as config_mod
        from deepagents_code import auth_store
        from deepagents_code.config import _ensure_bootstrap, restore_user_tracing_env

        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / ".state"
        )

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        original_tracing = dict(config_mod._bootstrap_state.original_tracing_env)
        original_api_keys = dict(config_mod._bootstrap_state.original_tracing_api_keys)
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "false")
            monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
            monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
            monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)
            auth_store.set_stored_key("langsmith", "lsv2_stored")

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
            ):
                _ensure_bootstrap()

            assert os.environ["LANGSMITH_TRACING"] == "false"
            assert os.environ["LANGCHAIN_TRACING_V2"] == "false"

            shell_env = os.environ.copy()
            restore_user_tracing_env(shell_env)

            assert "LANGSMITH_TRACING" not in shell_env
            assert shell_env["LANGCHAIN_TRACING_V2"] == "true"
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls
            config_mod._bootstrap_state.original_tracing_env = original_tracing
            config_mod._bootstrap_state.original_tracing_api_keys = original_api_keys


class TestApplyDefaultLangsmithProject:
    """Tests for _apply_default_langsmith_project()."""

    def test_defaults_when_tracing_on_and_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tracing on with no project set routes to the default project."""
        import os

        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

        _apply_default_langsmith_project()

        assert os.environ["LANGSMITH_PROJECT"] == LANGSMITH_PROJECT_DEFAULT

    def test_noop_when_project_already_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An existing LANGSMITH_PROJECT is never overwritten."""
        import os

        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_PROJECT", "user-project")

        _apply_default_langsmith_project()

        assert os.environ["LANGSMITH_PROJECT"] == "user-project"

    def test_noop_when_tracing_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No default is applied when tracing is not enabled."""
        import os

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

        _apply_default_langsmith_project()

        assert "LANGSMITH_PROJECT" not in os.environ


class TestFindDotenvFromStartPath:
    """Tests for _find_dotenv_from_start_path."""

    def test_finds_env_in_start_dir(self, tmp_path: Path) -> None:
        """Finds .env in the start directory itself."""
        from deepagents_code.config import _find_dotenv_from_start_path

        env_file = tmp_path / ".env"
        env_file.write_text("KEY=val")
        assert _find_dotenv_from_start_path(tmp_path) == env_file

    def test_finds_env_in_parent(self, tmp_path: Path) -> None:
        """Finds .env in a parent directory."""
        from deepagents_code.config import _find_dotenv_from_start_path

        env_file = tmp_path / ".env"
        env_file.write_text("KEY=val")
        child = tmp_path / "a" / "b"
        child.mkdir(parents=True)
        assert _find_dotenv_from_start_path(child) == env_file

    def test_returns_none_when_no_env(self, tmp_path: Path) -> None:
        """Returns None when no .env exists anywhere."""
        from deepagents_code.config import _find_dotenv_from_start_path

        child = tmp_path / "a"
        child.mkdir()
        # No .env anywhere under tmp_path — the search will keep going
        # to real parent dirs, but tmp_path itself has none
        result = _find_dotenv_from_start_path(child)
        # May find a real .env in parent dirs; just check it doesn't crash
        assert result is None or result.name == ".env"

    def test_continues_past_oserror_on_intermediate_dir(self, tmp_path: Path) -> None:
        """OSError on an intermediate .env candidate doesn't abort search."""
        from deepagents_code.config import _find_dotenv_from_start_path

        # Create .env in the grandparent
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=val")

        child = tmp_path / "sub"
        child.mkdir()

        # Patch is_file to raise OSError for the child's .env candidate
        original_is_file = Path.is_file

        def patched_is_file(self: Path) -> bool:
            if self == child / ".env":
                msg = "Permission denied"
                raise OSError(msg)
            return original_is_file(self)

        with patch.object(Path, "is_file", patched_is_file):
            result = _find_dotenv_from_start_path(child)

        # Should continue past the OSError and find .env in parent
        assert result == env_file


class TestDetectModePrefix:
    """Tests for `detect_mode_prefix`.

    This helper is the linchpin for routing typed prefixes to the correct
    mode. The longest-prefix-first invariant is critical: if `!!` ever loses
    to `!`, every `!!` command would silently route as a single-bang shell
    command and leak content to the model.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("!!ls", ("!!", "shell_incognito")),
            ("!!", ("!!", "shell_incognito")),
            ("!!!ls", ("!!", "shell_incognito")),
            ("!ls", ("!", "shell")),
            ("!", ("!", "shell")),
            ("/help", ("/", "command")),
            ("/", ("/", "command")),
        ],
    )
    def test_matches_known_prefixes(self, text: str, expected: tuple[str, str]) -> None:
        assert detect_mode_prefix(text) == expected

    @pytest.mark.parametrize(
        "text",
        ["", "ls", "echo hi", " !!ls", "\t!ls", "hello !!", "x!"],
    )
    def test_no_match_for_non_prefixed(self, text: str) -> None:
        assert detect_mode_prefix(text) is None

    def test_double_bang_wins_over_single_bang(self) -> None:
        """Regression guard: `!!` must beat `!` even if iteration order changes."""
        assert detect_mode_prefix("!!whoami") == ("!!", "shell_incognito")


class TestInterpreterSettings:
    """Tests for `[interpreter]` config.toml loading and validation."""

    def test_defaults_when_config_absent(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"  # does not exist
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            settings_obj = Settings.from_environment(start_path=tmp_path)

        assert settings_obj.enable_interpreter is True
        assert settings_obj.interpreter_timeout_seconds == pytest.approx(5.0)
        assert settings_obj.interpreter_memory_limit_mb == 64
        assert settings_obj.interpreter_max_ptc_calls == 256
        assert settings_obj.interpreter_max_result_chars == 4000
        assert settings_obj.interpreter_ptc == "safe"
        assert settings_obj.interpreter_ptc_acknowledge_unsafe is False

    def test_round_trip_through_toml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            """
[interpreter]
enable_interpreter = true
timeout_seconds = 12.5
memory_limit_mb = 128
max_ptc_calls = 64
max_result_chars = 8000
ptc = "safe"
ptc_acknowledge_unsafe = true
"""
        )
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            settings_obj = Settings.from_environment(start_path=tmp_path)

        assert settings_obj.enable_interpreter is True
        assert settings_obj.interpreter_timeout_seconds == pytest.approx(12.5)
        assert settings_obj.interpreter_memory_limit_mb == 128
        assert settings_obj.interpreter_max_ptc_calls == 64
        assert settings_obj.interpreter_max_result_chars == 8000
        assert settings_obj.interpreter_ptc == "safe"
        assert settings_obj.interpreter_ptc_acknowledge_unsafe is True

    def test_ptc_explicit_list_round_trip(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            """
[interpreter]
ptc = ["grep", "read_file"]
"""
        )
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            settings_obj = Settings.from_environment(start_path=tmp_path)

        assert settings_obj.interpreter_ptc == ["grep", "read_file"]

    def test_invalid_ptc_list_entry_falls_back(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            """
[interpreter]
ptc = [""]
"""
        )
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            settings_obj = Settings.from_environment(start_path=tmp_path)

        assert settings_obj.interpreter_ptc == "safe"

    def test_ptc_list_with_safe_preset_round_trip(self, tmp_path: Path) -> None:
        """`"safe"` is preserved as a list entry until agent-build expansion."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            """
[interpreter]
ptc = ["safe", "task"]
"""
        )
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            settings_obj = Settings.from_environment(start_path=tmp_path)

        assert settings_obj.interpreter_ptc == ["safe", "task"]

    def test_ptc_list_with_all_falls_back(self, tmp_path: Path) -> None:
        """`"all"` inside a list is rejected, falling back to the default."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            """
[interpreter]
ptc = ["all", "task"]
"""
        )
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            settings_obj = Settings.from_environment(start_path=tmp_path)

        assert settings_obj.interpreter_ptc == "safe"


class TestCreateModelCodex:
    """`create_model` dispatch for the ChatGPT-OAuth `openai_codex` provider.

    Covers the runtime path that turns a stored token into a working model —
    untested before this suite. All cases isolate the token store to a temp
    path and never touch the network.
    """

    def _plant_token(self, path: Path) -> None:
        """Write a valid (unexpired) token bundle at `path` with 0600 perms."""
        import json as _json
        from datetime import UTC, datetime, timedelta

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _json.dumps(
                {
                    "access_token": "fake_access",
                    "refresh_token": "fake_refresh",
                    "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                    "account_id": "acct_abc",
                    "plan_type": "pro",
                    "user_id": "user_xyz",
                    "id_token": None,
                }
            ),
            encoding="utf-8",
        )
        path.chmod(0o600)

    def test_missing_token_raises_missing_credentials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No stored token → `MissingCredentialsError` pointing at `/auth`."""
        from deepagents_code.integrations import openai_codex
        from deepagents_code.model_config import MissingCredentialsError

        monkeypatch.setattr(
            openai_codex, "default_store_path", lambda: tmp_path / "missing.json"
        )
        clear_caches()
        with pytest.raises(MissingCredentialsError) as exc_info:
            create_model("openai_codex:gpt-5.2-codex")
        # No env var to set; the recovery hint must route through `/auth`.
        assert exc_info.value.env_var is None
        assert "ChatGPT" in str(exc_info.value)

    def test_success_builds_codex_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stored token → a `_ChatOpenAICodex` under the codex provider."""
        from langchain_openai.chat_models.codex import _ChatOpenAICodex

        from deepagents_code.integrations import openai_codex

        path = tmp_path / "auth.json"
        self._plant_token(path)
        monkeypatch.setattr(openai_codex, "default_store_path", lambda: path)
        clear_caches()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`_ChatOpenAICodex` is experimental",
                category=UserWarning,
            )
            result = create_model(
                "openai_codex:gpt-5.2-codex",
                extra_kwargs={"http_socket_options": []},
            )
        assert isinstance(result.model, _ChatOpenAICodex)
        assert result.provider == "openai_codex"
        assert result.model_name == "gpt-5.2-codex"

    def test_reasoning_effort_composes_with_configured_summary(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from deepagents_code.integrations import openai_codex as codex_mod

        token_path = tmp_path / "auth.json"
        self._plant_token(token_path)
        monkeypatch.setattr(codex_mod, "default_store_path", lambda: token_path)
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.openai_codex]
models = ["gpt-5.5"]
[models.providers.openai_codex.params."gpt-5.5".reasoning]
summary = "auto"
effort = "low"
""")
        captured: dict[str, Any] = {}
        model = _make_init_chat_model_mock()

        def _capture(_model_name: str, /, **kwargs: Any) -> Any:  # noqa: ANN401
            captured.update(kwargs)
            return model

        monkeypatch.setattr(codex_mod, "build_chat_model", _capture)
        clear_caches()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            create_model(
                "openai_codex:gpt-5.5",
                extra_kwargs={"reasoning_effort": "high"},
            )

        assert captured["reasoning"] == {"summary": "auto", "effort": "high"}
        assert "reasoning_effort" not in captured

    def test_api_key_kwarg_is_stripped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A passed `api_key` must not reach the model; bearer is OAuth-only.

        `_ChatOpenAICodex` wires the OAuth token into `openai_api_key` as a
        callable, so the cleanest check is that the codex branch drops the
        `api_key` kwarg before it reaches `build_chat_model`.
        """
        from deepagents_code.integrations import openai_codex as codex_mod

        path = tmp_path / "auth.json"
        self._plant_token(path)
        monkeypatch.setattr(codex_mod, "default_store_path", lambda: path)

        captured: dict[str, Any] = {}
        real_build = codex_mod.build_chat_model

        def _capture(model_name: str, /, **kwargs: Any) -> Any:  # noqa: ANN401  # passthrough capture
            captured.update(kwargs)
            return real_build(model_name, **kwargs)

        monkeypatch.setattr(codex_mod, "build_chat_model", _capture)
        clear_caches()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`_ChatOpenAICodex` is experimental",
                category=UserWarning,
            )
            create_model(
                "openai_codex:gpt-5.2-codex",
                extra_kwargs={
                    "api_key": "sk-should-be-stripped",
                    "http_socket_options": [],
                },
            )
        assert "api_key" not in captured

    def test_expired_session_routes_to_auth(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A revoked refresh token → `MissingCredentialsError`, not generic.

        The codex branch must route `CodexAuthExpiredError` to the same
        sign-in recovery path as a missing token so the retry flow offers
        `/auth`, rather than wrapping it in a generic `ModelConfigError`.
        """
        from deepagents_code.integrations import openai_codex as codex_mod
        from deepagents_code.model_config import MissingCredentialsError

        path = tmp_path / "auth.json"
        self._plant_token(path)
        monkeypatch.setattr(codex_mod, "default_store_path", lambda: path)

        def _raise_expired(_model_name: str, /, **_kwargs: Any) -> Any:  # noqa: ANN401  # passthrough stub
            msg = "session expired"
            raise codex_mod.CodexAuthExpiredError(msg)

        monkeypatch.setattr(codex_mod, "build_chat_model", _raise_expired)
        clear_caches()
        with pytest.raises(MissingCredentialsError) as exc_info:
            create_model("openai_codex:gpt-5.2-codex")
        assert exc_info.value.env_var is None
        assert "expired" in str(exc_info.value).lower()

    def test_unexpected_build_error_wraps_as_model_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A genuinely unexpected build failure → `ModelConfigError` with spec.

        The broad catch-all is the last resort for construction failures that
        are neither missing nor expired credentials; it must name the spec
        rather than leak a raw traceback.
        """
        from deepagents_code.integrations import openai_codex as codex_mod
        from deepagents_code.model_config import ModelConfigError

        path = tmp_path / "auth.json"
        self._plant_token(path)
        monkeypatch.setattr(codex_mod, "default_store_path", lambda: path)

        def _boom(_model_name: str, /, **_kwargs: Any) -> Any:  # noqa: ANN401  # passthrough stub
            msg = "unexpected constructor failure"
            raise RuntimeError(msg)

        monkeypatch.setattr(codex_mod, "build_chat_model", _boom)
        clear_caches()
        with pytest.raises(ModelConfigError) as exc_info:
            create_model("openai_codex:gpt-5.2-codex")
        assert "openai_codex:gpt-5.2-codex" in str(exc_info.value)


class TestResolveGoalAutoAcceptCriteria:
    """Coverage for the config.py wrapper over the goals manifest option."""

    def test_missing_manifest_option_fails_closed_with_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A missing manifest entry disables auto-accept and logs why.

        The `option is None` branch only fires on a manifest regression (a
        renamed or dropped key), so it must fail closed to review *and* leave a
        trail rather than silently ignoring a user's saved preference.
        """
        from deepagents_code.config import resolve_goal_auto_accept_criteria

        with (
            patch(
                "deepagents_code.config_manifest.get_option",
                return_value=None,
            ) as get_option,
            caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
        ):
            result = resolve_goal_auto_accept_criteria()

        assert result == (False, "default")
        get_option.assert_called_once_with("goals.auto_accept_criteria")
        assert any(
            "goals.auto_accept_criteria" in record.getMessage()
            for record in caplog.records
        )

    def test_resolves_effective_preference_and_source(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The wrapper reports the resolved value and its config source."""
        from deepagents_code import config_manifest
        from deepagents_code._env_vars import GOAL_AUTO_ACCEPT_CRITERIA
        from deepagents_code.config import resolve_goal_auto_accept_criteria

        monkeypatch.setenv(GOAL_AUTO_ACCEPT_CRITERIA, "true")
        monkeypatch.setattr(config_manifest, "load_config_toml", dict)

        assert resolve_goal_auto_accept_criteria() == (
            True,
            f"env ({GOAL_AUTO_ACCEPT_CRITERIA})",
        )
