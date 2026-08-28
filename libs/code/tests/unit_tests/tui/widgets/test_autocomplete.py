"""Tests for autocomplete fuzzy search functionality."""

import asyncio
import logging
import subprocess
import threading
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from deepagents_code.command_registry import CommandEntry, get_slash_commands
from deepagents_code.tui.widgets import autocomplete as autocomplete_module
from deepagents_code.tui.widgets.autocomplete import (
    MAX_SUGGESTIONS,
    CompletionController,
    CompletionResult,
    FuzzyFileController,
    MultiCompletionManager,
    SlashCommandController,
    _fuzzy_score,
    _fuzzy_search,
    _get_git_executable,
    _get_project_files,
    _is_dotpath,
    _path_depth,
    _run_git_ls_files,
    _scope_files_to_cwd,
)


class TestFuzzyScore:
    """Tests for the _fuzzy_score function."""


class TestFuzzySearch:
    """Tests for the _fuzzy_search function."""

    @pytest.fixture
    def sample_files(self):
        """Sample file list for testing."""
        return [
            "README.md",
            "setup.py",
            "src/main.py",
            "src/utils.py",
            "src/helpers/string_utils.py",
            "tests/test_main.py",
            "tests/test_utils.py",
            ".github/workflows/ci.yml",
            ".gitignore",
            "docs/api.md",
        ]


class TestHelperFunctions:
    """Tests for helper functions."""


class TestSlashCommandController:
    """Tests for SlashCommandController."""

    @pytest.fixture
    def mock_view(self):
        """Create a mock CompletionView."""
        return MagicMock()

    @pytest.fixture
    def controller(self, mock_view):
        """Create a SlashCommandController with mock view."""
        return SlashCommandController(get_slash_commands(), mock_view)

    def test_substring_description_match_fresh(self, controller, mock_view):
        """Typing 'fresh' surfaces /clear via substring on 'Start a fresh thread'."""
        controller.on_text_changed("/fresh", 6)

        mock_view.render_completion_suggestions.assert_called()
        suggestions = mock_view.render_completion_suggestions.call_args[0][0]
        assert any("/clear" in s[0] for s in suggestions)


class TestScoreCommand:
    """Direct unit tests for SlashCommandController._score_command."""

    @staticmethod
    def score(search: str, cmd: str, desc: str, keywords: str = "") -> float:
        """Proxy score helper with explicit type signature for static analysis."""
        return SlashCommandController._score_command(search, cmd, desc, keywords)


class TestFuzzyFileControllerCanHandle:
    """Tests for FuzzyFileController.can_handle method."""

    @pytest.fixture
    def mock_view(self):
        """Create a mock CompletionView."""
        return MagicMock()

    @pytest.fixture
    def controller(self, mock_view, tmp_path):
        """Create a FuzzyFileController."""
        return FuzzyFileController(mock_view, cwd=tmp_path)

    def test_invalid_cursor_positions(self, controller):
        """Handles invalid cursor positions gracefully."""
        assert controller.can_handle("@file", 0) is False
        assert controller.can_handle("@file", -1) is False
        assert controller.can_handle("@file", 100) is False


class TestMultiCompletionManager:
    """Tests for MultiCompletionManager."""

    @pytest.fixture
    def mock_view(self):
        """Create a mock CompletionView."""
        return MagicMock()

    @pytest.fixture
    def manager(self, mock_view, tmp_path):
        """Create a MultiCompletionManager with both controllers."""
        slash_ctrl = SlashCommandController(get_slash_commands(), mock_view)
        file_ctrl = FuzzyFileController(mock_view, cwd=tmp_path)
        # Cast needed: lists are invariant, so the inferred type
        # list[SlashCommandController | FuzzyFileController] won't match
        # list[CompletionController] even though both satisfy the protocol.
        controllers = cast("list[CompletionController]", [slash_ctrl, file_ctrl])
        return MultiCompletionManager(controllers)


class TestSlashCommandControllerUpdateCommands:
    """Tests for SlashCommandController.update_commands()."""

    @pytest.fixture
    def mock_view(self) -> MagicMock:
        return MagicMock()


class TestSlashCommandControllerDisplaySeparation:
    """Popup shows the label but completion inserts the machine name."""

    @pytest.fixture
    def mock_view(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def controller(self, mock_view: MagicMock) -> SlashCommandController:
        commands = [
            CommandEntry(
                name="/skill:my-plugin:review",
                description="(my-plugin) Review code",
                hidden_keywords="my-plugin review",
                argument_hint="",
                display_name="/skill:review",
            ),
        ]
        return SlashCommandController(commands, mock_view)


class TestGetProjectFiles:
    """Tests for _get_project_files."""

    @staticmethod
    def _init_repo(root: Path) -> None:
        """Initialize a throwaway git repo with a test identity.

        Commit signing is disabled locally so commits succeed even when the
        host has `commit.gpgsign=true` set globally (no signing key is
        available in the throwaway repo).
        """
        for args in (
            ["init"],
            ["config", "user.email", "test@example.com"],
            ["config", "user.name", "Test"],
            ["config", "commit.gpgsign", "false"],
        ):
            subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    def test_includes_tracked_and_untracked_files(self, tmp_path: Path) -> None:
        """Both committed and untracked-but-not-ignored files are returned.

        Tracked files are listed before untracked ones so they rank ahead in
        completion results.
        """
        self._init_repo(tmp_path)
        (tmp_path / "tracked.py").write_text("x = 1\n")
        subprocess.run(
            ["git", "add", "tracked.py"], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        (tmp_path / "untracked.py").write_text("y = 2\n")

        files = _get_project_files(tmp_path)

        assert "tracked.py" in files
        assert "untracked.py" in files
        assert files.index("tracked.py") < files.index("untracked.py")

    def test_excludes_ignored_files(self, tmp_path: Path) -> None:
        """Files matched by .gitignore are not returned."""
        self._init_repo(tmp_path)
        (tmp_path / ".gitignore").write_text("ignored.py\n")
        (tmp_path / "ignored.py").write_text("z = 3\n")
        (tmp_path / "visible.py").write_text("a = 4\n")

        files = _get_project_files(tmp_path)

        assert "visible.py" in files
        assert "ignored.py" not in files

    def test_empty_git_listing_does_not_fall_back_to_glob(self, tmp_path: Path) -> None:
        """Successful empty git output is authoritative."""
        self._init_repo(tmp_path)
        (tmp_path / ".git" / "info" / "exclude").write_text("ignored.py\n")
        (tmp_path / "ignored.py").write_text("z = 3\n")

        files = _get_project_files(tmp_path)

        assert files == []

    def test_deduplicates_repeated_paths(self, tmp_path: Path) -> None:
        """A path emitted more than once by git ls-files appears only once.

        An unmerged (conflicted) file is reported once per merge stage by
        `git ls-files`, which is the real source of the duplicate entries the
        de-duplication guards against.
        """
        self._init_repo(tmp_path)
        conflict = tmp_path / "conflict.py"
        conflict.write_text("base\n")
        subprocess.run(
            ["git", "add", "conflict.py"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "base"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        base_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "checkout", "-b", "other"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        conflict.write_text("theirs\n")
        subprocess.run(
            ["git", "commit", "-am", "theirs"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", base_branch],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        conflict.write_text("ours\n")
        subprocess.run(
            ["git", "commit", "-am", "ours"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        # The merge fails with a conflict; the conflicted state is the point.
        subprocess.run(
            ["git", "merge", "other"],
            cwd=tmp_path,
            check=False,
            capture_output=True,
        )

        git_path = _get_git_executable()
        assert git_path is not None
        _, raw = _run_git_ls_files(git_path, tmp_path, [])
        # Premise: the conflicted path is reported more than once.
        assert raw.count("conflict.py") > 1

        files = _get_project_files(tmp_path)

        assert files.count("conflict.py") == 1

    def test_untracked_failure_keeps_tracked_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed untracked scan must not discard the tracked list.

        When the optional `--others --exclude-standard` call fails or times
        out, the already-successful tracked listing stays authoritative instead
        of falling back to the shallow glob walk.
        """
        self._init_repo(tmp_path)
        nested = tmp_path / "a" / "b" / "c" / "d" / "e"
        nested.mkdir(parents=True)
        deep = nested / "deep.py"
        deep.write_text("x = 1\n")
        subprocess.run(
            ["git", "add", "a"], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        real_run = autocomplete_module._run_git_ls_files

        def fake_run(
            git_path: str, root: Path, extra_args: list[str]
        ) -> tuple[bool, list[str]]:
            if "--others" in extra_args:
                return False, []
            return real_run(git_path, root, extra_args)

        monkeypatch.setattr(autocomplete_module, "_run_git_ls_files", fake_run)

        files = _get_project_files(tmp_path)

        assert "a/b/c/d/e/deep.py" in files

    def test_git_stderr_uses_stable_locale(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Git diagnostics must use the English locale expected by log filtering."""

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        captured: dict[str, object] = {}

        def fake_run(*_args: object, **kwargs: object) -> _Result:
            captured.update(kwargs)
            return _Result()

        monkeypatch.setenv("LC_ALL", "fr_FR.UTF-8")
        monkeypatch.setenv("AUTOCOMPLETE_TEST_ENV", "preserved")
        monkeypatch.setattr(autocomplete_module.subprocess, "run", fake_run)

        _run_git_ls_files("git", tmp_path, [])

        env = cast("dict[str, str]", captured["env"])
        assert env["LC_ALL"] == "C"
        assert env["AUTOCOMPLETE_TEST_ENV"] == "preserved"

    def test_non_repo_directory_is_quiet(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A non-Git directory (exit 128) must not emit a debug log.

        Running `git ls-files` outside a work tree is the expected trigger for
        the glob fallback, so the failure path stays silent.
        """
        git_path = _get_git_executable()
        assert git_path is not None

        with caplog.at_level(logging.DEBUG, logger="deepagents_code"):
            ok, files = _run_git_ls_files(git_path, tmp_path, [])

        assert ok is False
        assert files == []
        assert caplog.records == []

    def test_genuine_failure_logs_details(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A real Git failure logs root/cwd, args, exit code, and stderr."""

        class _Result:
            returncode = 129
            stdout = ""
            stderr = "fatal: unknown option `--bogus'\n"

        def fake_run(*_args: object, **_kwargs: object) -> _Result:
            return _Result()

        monkeypatch.setattr(autocomplete_module.subprocess, "run", fake_run)

        with caplog.at_level(logging.DEBUG, logger="deepagents_code"):
            ok, files = _run_git_ls_files("git", tmp_path, ["--bogus"])

        assert ok is False
        assert files == []
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert str(tmp_path) in message
        assert "--bogus" in message
        assert "exit=129" in message
        assert "unknown option" in message

    def test_failure_stderr_is_sanitized(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Control characters in git stderr are neutralized before logging."""

        class _Result:
            returncode = 1
            stdout = ""
            stderr = "fatal: broken\x1b[31mred\r\nsecond line\n"

        def fake_run(*_args: object, **_kwargs: object) -> _Result:
            return _Result()

        monkeypatch.setattr(autocomplete_module.subprocess, "run", fake_run)

        with caplog.at_level(logging.DEBUG, logger="deepagents_code"):
            _run_git_ls_files("git", tmp_path, [])

        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "\x1b" not in message
        assert "\r" not in message

    def test_glob_fallback_when_git_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without git, files are discovered via glob, excluding dotpaths."""
        monkeypatch.setattr(autocomplete_module, "_get_git_executable", lambda: None)
        (tmp_path / "visible.py").write_text("a = 1\n")
        (tmp_path / ".hidden.py").write_text("secret = 1\n")
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "mod.py").write_text("b = 2\n")

        files = _get_project_files(tmp_path)

        assert "visible.py" in files
        assert "pkg/mod.py" in files
        assert ".hidden.py" not in files


class TestFuzzyFileControllerScope:
    """Tests for cwd-scoped file completion behavior."""

    @pytest.fixture
    def mock_view(self):
        """Create a mock CompletionView."""
        return MagicMock()

    def test_scopes_git_file_list_to_cwd(self, mock_view, monkeypatch, tmp_path):
        """When cwd is nested, suggestions are scoped to that subtree."""
        project_root = tmp_path
        (project_root / ".git").mkdir()
        cwd = project_root / "apps" / "cli"
        cwd.mkdir(parents=True)

        mock_files = [
            "README.md",
            "apps/cli/main.py",
            "apps/cli/utils/helpers.py",
            "apps/web/index.ts",
        ]
        monkeypatch.setattr(
            autocomplete_module, "_get_project_files", lambda _root: mock_files
        )

        controller = FuzzyFileController(mock_view, cwd=cwd)
        assert controller._get_files() == ["main.py", "utils/helpers.py"]

        controller.on_text_changed("@", 1)
        suggestions = mock_view.render_completion_suggestions.call_args[0][0]
        labels = [label for label, _ in suggestions]
        assert "@main.py" in labels
        assert "@utils/helpers.py" in labels
        assert not any("apps/web" in label for label in labels)

    def test_keeps_project_root_scope_when_cwd_is_root(
        self, mock_view, monkeypatch, tmp_path
    ):
        """When cwd is project root, file list remains repo-relative."""
        (tmp_path / ".git").mkdir()
        mock_files = ["README.md", "apps/cli/main.py"]
        monkeypatch.setattr(
            autocomplete_module, "_get_project_files", lambda _root: mock_files
        )

        controller = FuzzyFileController(mock_view, cwd=tmp_path)
        assert controller._get_files() == mock_files

    def test_scopes_git_file_list_with_symlinked_cwd(
        self, mock_view, monkeypatch, tmp_path
    ):
        """Symlinked cwd should still scope suggestions to the resolved subtree."""
        project_root = tmp_path
        (project_root / ".git").mkdir()
        real_cwd = project_root / "apps" / "cli"
        real_cwd.mkdir(parents=True)
        symlink_cwd = project_root / "APPS_CLI_LINK"
        try:
            symlink_cwd.symlink_to(real_cwd, target_is_directory=True)
        except OSError:  # pragma: no cover - platform/permission dependent
            return

        mock_files = [
            "README.md",
            "apps/cli/main.py",
            "apps/cli/utils/helpers.py",
            "apps/web/index.ts",
        ]
        monkeypatch.setattr(
            autocomplete_module, "_get_project_files", lambda _root: mock_files
        )

        controller = FuzzyFileController(mock_view, cwd=symlink_cwd)
        assert controller._get_files() == ["main.py", "utils/helpers.py"]

    def test_excludes_sibling_with_shared_prefix(
        self, mock_view, monkeypatch, tmp_path
    ):
        """A sibling sharing a name prefix (apps/cli vs apps/cli-tools) is excluded.

        Guards the trailing slash in the scope prefix: without it, `apps/cli`
        would also match `apps/cli-tools/...`.
        """
        project_root = tmp_path
        (project_root / ".git").mkdir()
        cwd = project_root / "apps" / "cli"
        cwd.mkdir(parents=True)

        mock_files = [
            "apps/cli/main.py",
            "apps/cli-tools/runner.py",
        ]
        monkeypatch.setattr(
            autocomplete_module, "_get_project_files", lambda _root: mock_files
        )

        controller = FuzzyFileController(mock_view, cwd=cwd)
        assert controller._get_files() == ["main.py"]

    def test_empty_when_cwd_subtree_has_no_files(
        self, mock_view, monkeypatch, tmp_path
    ):
        """A nested cwd with no files under it yields an empty suggestion list."""
        project_root = tmp_path
        (project_root / ".git").mkdir()
        cwd = project_root / "apps" / "empty"
        cwd.mkdir(parents=True)

        mock_files = ["README.md", "apps/cli/main.py"]
        monkeypatch.setattr(
            autocomplete_module, "_get_project_files", lambda _root: mock_files
        )

        controller = FuzzyFileController(mock_view, cwd=cwd)
        assert controller._get_files() == []

    def test_refresh_cache_rescopes_to_cwd(self, mock_view, monkeypatch, tmp_path):
        """refresh_cache re-runs scoping against the latest file list."""
        project_root = tmp_path
        (project_root / ".git").mkdir()
        cwd = project_root / "apps" / "cli"
        cwd.mkdir(parents=True)

        files = ["apps/cli/main.py"]
        monkeypatch.setattr(
            autocomplete_module, "_get_project_files", lambda _root: files
        )

        controller = FuzzyFileController(mock_view, cwd=cwd)
        assert controller._get_files() == ["main.py"]

        files.append("apps/cli/added.py")
        controller.refresh_cache()
        assert controller._get_files() == ["main.py", "added.py"]

    def test_scope_helper_fails_closed_when_cwd_outside_root(self):
        """A cwd outside project_root returns [] rather than wrong-base paths.

        The input paths are project-root-relative; if cwd is not under the root
        they would resolve to the wrong base, so the helper fails closed.
        """
        files = ["src/main.py", "src/utils.py"]
        project_root = Path("/repo")
        cwd = Path("/elsewhere")

        assert _scope_files_to_cwd(files, project_root, cwd) == []

    def test_scope_helper_returns_unchanged_when_cwd_is_root(self):
        """A cwd equal to project_root leaves the repo-relative list unchanged."""
        files = ["src/main.py", "README.md"]
        root = Path("/repo")

        assert _scope_files_to_cwd(files, root, root) == files
