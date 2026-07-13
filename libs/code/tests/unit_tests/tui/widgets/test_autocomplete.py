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

    def test_exact_filename_match_at_start(self):
        """Exact match at start of filename gets highest score."""
        score = _fuzzy_score("main", "src/main.py")
        assert score > 140  # Should be ~150

    def test_exact_filename_match_anywhere(self):
        """Exact match anywhere in filename."""
        score = _fuzzy_score("test", "src/my_test_file.py")
        assert score > 90  # Should be ~100

    def test_word_boundary_match(self):
        """Match at word boundary (after _, -, .) gets bonus."""
        score_boundary = _fuzzy_score("test", "src/my_test.py")
        score_middle = _fuzzy_score("est", "src/mytest.py")
        assert score_boundary > score_middle

    def test_path_match_lower_than_filename(self):
        """Match in path scores lower than filename match."""
        filename_score = _fuzzy_score("utils", "utils.py")
        path_score = _fuzzy_score("utils", "src/utils/helper.py")
        assert filename_score > path_score

    def test_no_match_returns_low_score(self):
        """Completely unrelated strings get very low scores."""
        score = _fuzzy_score("xyz", "abc.py")
        assert score < 15  # Below MIN_FUZZY_SCORE threshold

    def test_case_insensitive(self):
        """Matching is case insensitive."""
        score_lower = _fuzzy_score("main", "Main.py")
        score_upper = _fuzzy_score("MAIN", "main.py")
        assert score_lower > 100
        assert score_upper > 100

    def test_shorter_paths_preferred(self):
        """Shorter paths get slightly higher scores for same match."""
        short_score = _fuzzy_score("test", "test.py")
        long_score = _fuzzy_score("test", "very/long/path/to/test.py")
        assert short_score > long_score

    def test_backslash_normalization(self):
        """Backslash-separated paths score the same as forward-slash paths."""
        forward = _fuzzy_score("helper", "src/utils/helper.py")
        backward = _fuzzy_score("helper", "src\\utils\\helper.py")
        assert backward == forward
        assert backward > 100  # Should be a strong filename match

    def test_mixed_separator_normalization(self):
        """Mixed forward/backslash paths are normalized before scoring."""
        score = _fuzzy_score("helper", "src/utils\\helper.py")
        assert score > 100  # Should extract filename correctly


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

    def test_empty_query_returns_root_files_first(self, sample_files):
        """Empty query returns files sorted by depth, then name."""
        results = _fuzzy_search("", sample_files, limit=5)
        # Root level files should come first
        assert results[0] in ["README.md", "setup.py"]
        assert all("/" not in r for r in results[:2])  # First items are root level

    def test_exact_match_ranked_first(self, sample_files):
        """Exact filename matches are ranked first."""
        results = _fuzzy_search("main", sample_files, limit=5)
        assert "src/main.py" in results[:2]

    def test_filters_dotfiles_by_default(self, sample_files):
        """Dotfiles are filtered out by default."""
        results = _fuzzy_search("git", sample_files, limit=10)
        assert not any(".git" in r for r in results)

    def test_includes_dotfiles_when_query_starts_with_dot(self, sample_files):
        """Dotfiles included when query starts with '.'."""
        results = _fuzzy_search(".git", sample_files, limit=10, include_dotfiles=True)
        assert any(".git" in r for r in results)

    def test_respects_limit(self, sample_files):
        """Results respect the limit parameter."""
        results = _fuzzy_search("", sample_files, limit=3)
        assert len(results) <= 3

    def test_filters_low_score_matches(self, sample_files):
        """Low score matches are filtered out."""
        results = _fuzzy_search("xyznonexistent", sample_files, limit=10)
        assert len(results) == 0

    def test_utils_matches_multiple_files(self, sample_files):
        """Query matching multiple files returns all matches."""
        results = _fuzzy_search("utils", sample_files, limit=10)
        assert len(results) >= 2
        assert any("utils.py" in r for r in results)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_dotpath_detects_dotfiles(self):
        """_is_dotpath correctly identifies dotfiles."""
        assert _is_dotpath(".gitignore") is True
        assert _is_dotpath(".github/workflows/ci.yml") is True
        assert _is_dotpath("src/.hidden/file.py") is True

    def test_is_dotpath_allows_normal_files(self):
        """_is_dotpath returns False for normal files."""
        assert _is_dotpath("src/main.py") is False
        assert _is_dotpath("README.md") is False
        assert _is_dotpath("tests/test_main.py") is False

    def test_path_depth_counts_slashes(self):
        """_path_depth correctly counts directory depth."""
        assert _path_depth("file.py") == 0
        assert _path_depth("src/file.py") == 1
        assert _path_depth("src/utils/file.py") == 2
        assert _path_depth("a/b/c/d/file.py") == 4


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

    def test_can_handle_slash_prefix(self, controller):
        """Handles text starting with /."""
        assert controller.can_handle("/", 1) is True
        assert controller.can_handle("/hel", 4) is True
        assert controller.can_handle("/help", 5) is True

    def test_cannot_handle_non_slash(self, controller):
        """Does not handle text not starting with /."""
        assert controller.can_handle("hello", 5) is False
        assert controller.can_handle("", 0) is False
        assert controller.can_handle("test /cmd", 9) is False

    def test_filters_commands_by_prefix(self, controller, mock_view):
        """Filters commands based on typed prefix."""
        controller.on_text_changed("/hel", 4)

        # Should have called render with /help suggestion
        mock_view.render_completion_suggestions.assert_called()
        suggestions = mock_view.render_completion_suggestions.call_args[0][0]
        assert any("/help" in s[0] for s in suggestions)

    def test_filters_version_command_by_prefix(self, controller, mock_view):
        """Filters /version command based on typed prefix."""
        controller.on_text_changed("/ver", 4)

        mock_view.render_completion_suggestions.assert_called()
        suggestions = mock_view.render_completion_suggestions.call_args[0][0]
        assert any("/version" in s[0] for s in suggestions)

    def test_shows_all_commands_on_slash_only(self, controller, mock_view):
        """Shows all commands when just / is typed."""
        controller.on_text_changed("/", 1)

        mock_view.render_completion_suggestions.assert_called()
        suggestions = mock_view.render_completion_suggestions.call_args[0][0]
        assert len(suggestions) == min(len(get_slash_commands()), MAX_SUGGESTIONS)

    def test_clears_on_no_match(self, controller, mock_view):
        """Clears suggestions when no commands match after having suggestions."""
        # First get some suggestions
        controller.on_text_changed("/h", 2)
        mock_view.render_completion_suggestions.assert_called()

        # Now type something that doesn't match - should clear
        controller.on_text_changed("/xyz", 4)
        mock_view.clear_completion_suggestions.assert_called()

    def test_reset_clears_state(self, controller, mock_view):
        """Reset clears suggestions and state."""
        controller.on_text_changed("/h", 2)
        controller.reset()

        mock_view.clear_completion_suggestions.assert_called()

    def test_suggestions_return_after_reset(self, controller, mock_view):
        """Suggestions reappear when text is re-entered after a reset."""
        controller.on_text_changed("/", 1)
        mock_view.render_completion_suggestions.assert_called()

        controller.reset()
        mock_view.reset_mock()

        # Re-entering "/" should show suggestions again
        controller.on_text_changed("/", 1)
        mock_view.render_completion_suggestions.assert_called()
        suggestions = mock_view.render_completion_suggestions.call_args[0][0]
        assert len(suggestions) == min(len(get_slash_commands()), MAX_SUGGESTIONS)

    def test_hidden_keyword_match_continue(self, controller, mock_view):
        """Typing 'continue' surfaces /threads via hidden keyword."""
        controller.on_text_changed("/continue", 9)

        mock_view.render_completion_suggestions.assert_called()
        suggestions = mock_view.render_completion_suggestions.call_args[0][0]
        assert any("/threads" in s[0] for s in suggestions)

    def test_substring_description_match_exit(self, controller, mock_view):
        """Typing 'exit' surfaces /quit via substring match on 'Exit app'."""
        controller.on_text_changed("/exit", 5)

        mock_view.render_completion_suggestions.assert_called()
        suggestions = mock_view.render_completion_suggestions.call_args[0][0]
        assert any("/quit" in s[0] for s in suggestions)

    def test_substring_description_match_new(self, controller, mock_view):
        """Typing 'new' surfaces /clear via substring on 'start new thread'."""
        controller.on_text_changed("/new", 4)

        mock_view.render_completion_suggestions.assert_called()
        suggestions = mock_view.render_completion_suggestions.call_args[0][0]
        assert any("/clear" in s[0] for s in suggestions)

    def test_substring_name_match(self, controller, mock_view):
        """Substring of command name (not prefix) surfaces the command."""
        controller.on_text_changed("/flo", 4)

        mock_view.render_completion_suggestions.assert_called()
        suggestions = mock_view.render_completion_suggestions.call_args[0][0]
        assert any("/offload" in s[0] for s in suggestions)

    def test_true_fuzzy_match_via_misspelling(self, controller, mock_view):
        """Misspelled command surfaces via SequenceMatcher ratio."""
        controller.on_text_changed("/hlep", 5)

        mock_view.render_completion_suggestions.assert_called()
        suggestions = mock_view.render_completion_suggestions.call_args[0][0]
        assert any("/help" in s[0] for s in suggestions)

    def test_prefix_match_ranks_first(self, controller, mock_view):
        """Prefix matches on command name rank above description matches."""
        controller.on_text_changed("/he", 3)

        mock_view.render_completion_suggestions.assert_called()
        suggestions = mock_view.render_completion_suggestions.call_args[0][0]
        # /help is a prefix match — should be first
        assert suggestions[0][0] == "/help"

    def test_no_match_clears(self, controller, mock_view):
        """Completely unrelated input clears suggestions."""
        controller.on_text_changed("/h", 2)
        mock_view.render_completion_suggestions.assert_called()

        controller.on_text_changed("/zzzzzzzzz", 10)
        mock_view.clear_completion_suggestions.assert_called()

    def test_space_dismisses_suggestions(self, controller, mock_view):
        """Typing a space after a command name dismisses the popup."""
        controller.on_text_changed("/model", 6)
        mock_view.render_completion_suggestions.assert_called()

        controller.on_text_changed("/model ", 7)
        mock_view.clear_completion_suggestions.assert_called()

    def test_space_with_args_stays_dismissed(self, controller, mock_view):
        """Suggestions stay dismissed while typing arguments after a space."""
        # First show suggestions, then dismiss with space
        controller.on_text_changed("/model", 6)
        mock_view.render_completion_suggestions.assert_called()
        mock_view.reset_mock()

        controller.on_text_changed("/model gpt-5.5", 13)
        mock_view.render_completion_suggestions.assert_not_called()

    def test_bare_slash_space_dismisses(self, controller, mock_view):
        """Typing '/ ' (slash then space, no command) dismisses the popup."""
        controller.on_text_changed("/", 1)
        mock_view.render_completion_suggestions.assert_called()

        controller.on_text_changed("/ ", 2)
        mock_view.clear_completion_suggestions.assert_called()

    def test_backspace_from_space_restores_suggestions(self, controller, mock_view):
        """Deleting the space after '/model ' re-shows suggestions."""
        controller.on_text_changed("/model", 6)
        mock_view.render_completion_suggestions.assert_called()

        controller.on_text_changed("/model ", 7)
        mock_view.clear_completion_suggestions.assert_called()
        mock_view.reset_mock()

        # Simulate backspace back to "/model"
        controller.on_text_changed("/model", 6)
        mock_view.render_completion_suggestions.assert_called()

    @pytest.mark.usefixtures("mock_view")
    def test_double_reset_is_safe(self, controller):
        """Calling reset twice does not raise or double-clear."""
        controller.on_text_changed("/", 1)
        controller.reset()
        # Second reset should be a no-op (suggestions already empty)
        controller.reset()

    def test_space_key_applies_selected_completion(self, controller, mock_view) -> None:
        """Pressing space with active suggestions applies the completion."""
        controller.on_text_changed("/hel", 4)
        mock_view.render_completion_suggestions.assert_called()

        event = MagicMock()
        event.key = "space"
        result = controller.on_key(event, "/hel", 4)

        assert result == CompletionResult.HANDLED
        mock_view.replace_completion_range.assert_called_once()
        # First positional arg is start=0, second is cursor_index=4,
        # third is the completed command name.
        args = mock_view.replace_completion_range.call_args[0]
        assert args[0] == 0
        assert args[1] == 4
        assert args[2] == "/help"

    def test_space_key_ignored_without_suggestions(self, controller) -> None:
        """Space returns IGNORED when there are no active suggestions."""
        event = MagicMock()
        event.key = "space"
        result = controller.on_key(event, "/zzz", 4)
        assert result == CompletionResult.IGNORED


class TestScoreCommand:
    """Direct unit tests for SlashCommandController._score_command."""

    @staticmethod
    def score(search: str, cmd: str, desc: str, keywords: str = "") -> float:
        """Proxy score helper with explicit type signature for static analysis."""
        return SlashCommandController._score_command(search, cmd, desc, keywords)

    def test_prefix_returns_200(self):
        assert self.score("hel", "/help", "Show help") == 200

    def test_substring_name_returns_150(self):
        assert self.score("omp", "/compact", "Offload conversation") == 150

    def test_substring_desc_word_boundary_returns_110(self):
        assert self.score("exit", "/quit", "Exit app") == 110

    def test_substring_desc_mid_word_returns_90(self):
        desc = "Free up context window space by offloading older messages"
        assert self.score("ex", "/offload", desc) == 90

    def test_no_match_returns_zero(self):
        assert self.score("zzzzz", "/help", "Show help") == 0

    def test_fuzzy_above_threshold(self):
        score = self.score("hlep", "/help", "Show help")
        assert 0 < score < 100  # fuzzy tier, not substring/prefix

    def test_hidden_keyword_prefix_match(self):
        assert (
            self.score("cont", "/threads", "Browse threads", "continue history") == 120
        )

    def test_hidden_keyword_substring_match(self):
        assert (
            self.score("hist", "/threads", "Browse threads", "continue history") == 120
        )

    def test_hidden_keyword_ignored_when_empty(self):
        assert self.score("cont", "/threads", "Browse threads", "") == 0

    def test_hidden_keyword_requires_min_length(self):
        """Single-char queries do not match hidden keywords."""
        assert self.score("c", "/threads", "Browse threads", "continue") == 0

    def test_tiers_ordering(self):
        """Prefix > substring-name > keyword > substring-desc > fuzzy."""
        prefix = self.score("hel", "/help", "Show help")
        substr_name = self.score("omp", "/compact", "Offload conversation")
        keyword = self.score("cont", "/threads", "Browse threads", "continue")
        desc_boundary = self.score("exit", "/quit", "Exit app")
        offload_desc = "Free up context window space by offloading older messages"
        desc_mid = self.score("ex", "/offload", offload_desc)
        fuzzy = self.score("hlep", "/help", "Show help")
        assert prefix > substr_name > keyword > desc_boundary > desc_mid > fuzzy > 0


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

    def test_handles_at_symbol(self, controller):
        """Handles text with @ symbol."""
        assert controller.can_handle("@", 1) is True
        assert controller.can_handle("@file", 5) is True
        assert controller.can_handle("look at @src/main.py", 20) is True

    def test_handles_at_mid_text(self, controller):
        """Handles @ in middle of text."""
        assert controller.can_handle("check @file", 11) is True
        assert controller.can_handle("see @", 5) is True

    def test_no_handle_without_at(self, controller):
        """Does not handle text without @."""
        assert controller.can_handle("hello", 5) is False
        assert controller.can_handle("", 0) is False

    def test_no_handle_at_after_cursor(self, controller):
        """Does not handle @ that's after cursor position."""
        assert controller.can_handle("hello @file", 5) is False

    def test_no_handle_space_after_at(self, controller):
        """Does not handle @ followed by space before cursor."""
        assert controller.can_handle("@ file", 6) is False
        assert controller.can_handle("@file name", 10) is False

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

    def test_activates_slash_controller_for_slash(self, manager):
        """Activates slash controller for / prefix."""
        manager.on_text_changed("/help", 5)
        assert manager._active is not None
        assert isinstance(manager._active, SlashCommandController)

    def test_activates_file_controller_for_at(self, manager):
        """Activates file controller for @ prefix."""
        manager.on_text_changed("@file", 5)
        assert manager._active is not None
        assert isinstance(manager._active, FuzzyFileController)

    def test_no_active_for_plain_text(self, manager):
        """No controller active for plain text."""
        manager.on_text_changed("hello world", 11)
        assert manager._active is None

    def test_switches_controllers(self, manager):
        """Switches between controllers as input changes."""
        manager.on_text_changed("/cmd", 4)
        assert isinstance(manager._active, SlashCommandController)

        manager.on_text_changed("@file", 5)
        assert isinstance(manager._active, FuzzyFileController)

    def test_reset_clears_active(self, manager):
        """Reset clears active controller."""
        manager.on_text_changed("/cmd", 4)
        manager.reset()
        assert manager._active is None

    def test_reactivates_after_reset(self, manager, mock_view):
        """Controller reactivates for new input after a full reset."""
        manager.on_text_changed("/", 1)
        assert isinstance(manager._active, SlashCommandController)

        manager.reset()
        assert manager._active is None
        mock_view.reset_mock()

        # Typing "/" again should reactivate the slash controller
        manager.on_text_changed("/", 1)
        assert isinstance(manager._active, SlashCommandController)
        mock_view.render_completion_suggestions.assert_called()

    def test_double_reset_is_safe(self, manager):
        """Calling reset when already inactive is a no-op."""
        manager.on_text_changed("/cmd", 4)
        manager.reset()
        manager.reset()
        assert manager._active is None


class TestSlashCommandControllerUpdateCommands:
    """Tests for SlashCommandController.update_commands()."""

    @pytest.fixture
    def mock_view(self) -> MagicMock:
        return MagicMock()

    def test_update_replaces_commands(self, mock_view: MagicMock) -> None:
        """update_commands() replaces the internal commands list."""
        initial = [CommandEntry("/help", "Show help", "", "")]
        controller = SlashCommandController(initial, mock_view)

        new_commands = [
            CommandEntry("/help", "Show help", "", ""),
            CommandEntry("/skill:web-research", "Research topics", "web-research", ""),
        ]
        controller.update_commands(new_commands)

        # Typing /skill: should now show the skill command
        controller.on_text_changed("/skill:", 7)
        mock_view.render_completion_suggestions.assert_called()
        suggestions = mock_view.render_completion_suggestions.call_args[0][0]
        assert any("/skill:web-research" in s[0] for s in suggestions)

    def test_update_resets_suggestions(self, mock_view: MagicMock) -> None:
        """update_commands() clears any active suggestions."""
        commands = [CommandEntry("/help", "Show help", "", "")]
        controller = SlashCommandController(commands, mock_view)
        controller.on_text_changed("/h", 2)
        mock_view.render_completion_suggestions.assert_called()

        controller.update_commands([CommandEntry("/quit", "Exit", "", "")])
        mock_view.clear_completion_suggestions.assert_called()

    def test_skill_commands_fuzzy_match(self, mock_view: MagicMock) -> None:
        """Skill commands match via hidden keywords."""
        commands = [
            CommandEntry("/help", "Show help", "", ""),
            CommandEntry(
                "/skill:code-review", "Review code changes", "code-review", ""
            ),
        ]
        controller = SlashCommandController(commands, mock_view)
        controller.on_text_changed("/code", 5)

        mock_view.render_completion_suggestions.assert_called()
        suggestions = mock_view.render_completion_suggestions.call_args[0][0]
        assert any("/skill:code-review" in s[0] for s in suggestions)


class TestFuzzyFileControllerSetCwd:
    """Tests for FuzzyFileController.set_cwd switching completion roots."""

    def test_set_cwd_defers_project_root_off_event_loop(self, tmp_path):
        """set_cwd roots at cwd immediately and defers project-root discovery."""
        proj = tmp_path / "proj"
        (proj / ".git").mkdir(parents=True)
        sub = proj / "sub"
        sub.mkdir()
        controller = FuzzyFileController(MagicMock(), cwd=tmp_path)

        controller.set_cwd(sub)

        # No blocking filesystem walk here: cwd is the provisional root and the
        # real root is resolved later in warm_cache().
        assert controller._cwd == sub
        assert controller._project_root == sub
        assert controller._project_root_pending is True

    async def test_warm_cache_resolves_pending_project_root(self, tmp_path):
        """warm_cache resolves the deferred project root to the git root."""
        proj = tmp_path / "proj"
        (proj / ".git").mkdir(parents=True)
        sub = proj / "sub"
        sub.mkdir()
        controller = FuzzyFileController(MagicMock(), cwd=tmp_path)
        controller.set_cwd(sub)

        await controller.warm_cache()

        assert controller._project_root == proj.resolve()
        assert controller._project_root_pending is False


class TestFuzzyFileControllerWarmCacheRace:
    """Overlapping cwd warmers must not let stale results win.

    Warmers are scheduled with `exclusive=False`, so a slow warmer for an old
    cwd can finish after a newer cwd switch. These tests force that ordering and
    assert the controller stays rooted/cache-warmed for the newest cwd.
    """

    async def test_stale_warmer_does_not_overwrite_project_root(
        self, tmp_path, monkeypatch
    ):
        """An older project-root lookup finishing last must not win."""
        proj_a = tmp_path / "a"
        sub_a = proj_a / "sub"
        sub_a.mkdir(parents=True)
        proj_b = tmp_path / "b"
        sub_b = proj_b / "sub"
        sub_b.mkdir(parents=True)

        entered_a = threading.Event()
        release_a = threading.Event()

        def fake_find(path: Path) -> Path | None:
            if path == sub_a:
                entered_a.set()
                release_a.wait(timeout=5)
                return proj_a
            if path == sub_b:
                return proj_b
            return None

        monkeypatch.setattr(autocomplete_module, "find_project_root", fake_find)
        monkeypatch.setattr(
            autocomplete_module,
            "_get_project_files",
            lambda root: [f"sub/{root.name}.py"],
        )

        controller = FuzzyFileController(MagicMock(), cwd=tmp_path)

        controller.set_cwd(sub_a)
        task_a = asyncio.create_task(controller.warm_cache())
        await asyncio.to_thread(entered_a.wait, 5)

        controller.set_cwd(sub_b)
        await controller.warm_cache()

        # The newer warmer fully resolved before the stale one is released.
        assert controller._project_root == proj_b
        assert controller._project_root_pending is False
        assert controller._file_cache == ["b.py"]

        release_a.set()
        await task_a

        # The stale warmer finished last but left the newer state untouched.
        assert controller._project_root == proj_b
        assert controller._project_root_pending is False
        assert controller._file_cache == ["b.py"]

    async def test_stale_warmer_does_not_overwrite_file_cache(
        self, tmp_path, monkeypatch
    ):
        """An older file-cache warm finishing last must not win."""
        sub_a = tmp_path / "a"
        sub_a.mkdir()
        sub_b = tmp_path / "b"
        sub_b.mkdir()

        entered_a = threading.Event()
        release_a = threading.Event()

        def fake_files(root: Path) -> list[str]:
            if root == sub_a:
                entered_a.set()
                release_a.wait(timeout=5)
                return ["a/file.py"]
            return ["b/file.py"]

        monkeypatch.setattr(autocomplete_module, "find_project_root", lambda _: None)
        monkeypatch.setattr(autocomplete_module, "_get_project_files", fake_files)

        controller = FuzzyFileController(MagicMock(), cwd=tmp_path)

        controller.set_cwd(sub_a)
        task_a = asyncio.create_task(controller.warm_cache())
        await asyncio.to_thread(entered_a.wait, 5)

        controller.set_cwd(sub_b)
        await controller.warm_cache()

        # The newer warmer fully resolved before the stale one is released.
        assert controller._project_root == sub_b
        assert controller._project_root_pending is False
        assert controller._file_cache == ["b/file.py"]

        release_a.set()
        await task_a

        # The stale warmer finished last but left the newer state untouched.
        assert controller._project_root == sub_b
        assert controller._project_root_pending is False
        assert controller._file_cache == ["b/file.py"]

    async def test_aba_stale_warmer_does_not_overwrite_file_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An older A warmer must not win after an A-to-B-to-A switch."""
        sub_a = tmp_path / "a"
        sub_a.mkdir()
        sub_b = tmp_path / "b"
        sub_b.mkdir()

        entered_old_a = threading.Event()
        release_old_a = threading.Event()
        lock = threading.Lock()
        a_calls = 0

        def fake_files(root: Path) -> list[str]:
            nonlocal a_calls

            if root == sub_a:
                with lock:
                    a_calls += 1
                    call = a_calls
                if call == 1:
                    entered_old_a.set()
                    release_old_a.wait(timeout=5)
                    return ["a/old.py"]
                return ["a/new.py"]
            return ["b/file.py"]

        monkeypatch.setattr(autocomplete_module, "find_project_root", lambda _: None)
        monkeypatch.setattr(autocomplete_module, "_get_project_files", fake_files)

        controller = FuzzyFileController(MagicMock(), cwd=tmp_path)

        controller.set_cwd(sub_a)
        old_task_a = asyncio.create_task(controller.warm_cache())
        await asyncio.to_thread(entered_old_a.wait, 5)

        controller.set_cwd(sub_b)
        await controller.warm_cache()

        controller.set_cwd(sub_a)
        await controller.warm_cache()
        assert controller._file_cache == ["a/new.py"]

        release_old_a.set()
        await old_task_a

        assert controller._project_root == sub_a
        assert controller._project_root_pending is False
        assert controller._file_cache == ["a/new.py"]

    async def test_force_warmer_does_not_overwrite_newer_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A force refresh superseded mid-walk must not clobber a newer cwd.

        `force=True` is a new way to reach the file-walk-and-swap block on an
        already-populated cache (a populated cache otherwise short-circuits), so
        a background refresh racing a `set_cwd` is a genuinely new scenario.
        """
        sub_a = tmp_path / "a"
        sub_a.mkdir()
        sub_b = tmp_path / "b"
        sub_b.mkdir()

        entered_force = threading.Event()
        release_force = threading.Event()
        lock = threading.Lock()
        a_calls = 0

        def fake_files(root: Path) -> list[str]:
            nonlocal a_calls

            if root == sub_a:
                with lock:
                    a_calls += 1
                    call = a_calls
                if call == 1:
                    # Initial warm populates the cache and returns immediately.
                    return ["a/file.py"]
                # The forced background refresh; block it mid-walk so a newer
                # cwd switch can supersede it.
                entered_force.set()
                release_force.wait(timeout=5)
                return ["a/stale.py"]
            return ["b/file.py"]

        monkeypatch.setattr(autocomplete_module, "find_project_root", lambda _: None)
        monkeypatch.setattr(autocomplete_module, "_get_project_files", fake_files)

        controller = FuzzyFileController(MagicMock(), cwd=tmp_path)
        controller.set_cwd(sub_a)
        await controller.warm_cache()
        assert controller._file_cache == ["a/file.py"]

        force_task = asyncio.create_task(controller.warm_cache(force=True))
        await asyncio.to_thread(entered_force.wait, 5)

        # The prior cache stays visible while the forced walk is in flight.
        assert controller._file_cache == ["a/file.py"]

        # A newer cwd switch supersedes the in-flight forced refresh.
        controller.set_cwd(sub_b)
        await controller.warm_cache()
        assert controller._file_cache == ["b/file.py"]

        release_force.set()
        await force_task

        # The stale forced walk finished last but dropped its result.
        assert controller._project_root == sub_b
        assert controller._project_root_pending is False
        assert controller._file_cache == ["b/file.py"]


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

    async def test_warm_cache_scopes_file_list_to_cwd(
        self, mock_view, monkeypatch, tmp_path
    ):
        """warm_cache scopes the cached file list to the resolved cwd subtree."""
        project_root = tmp_path
        (project_root / ".git").mkdir()
        cwd = project_root / "apps" / "cli"
        cwd.mkdir(parents=True)

        mock_files = [
            "README.md",
            "apps/cli/main.py",
            "apps/web/index.ts",
        ]
        monkeypatch.setattr(
            autocomplete_module, "_get_project_files", lambda _root: mock_files
        )

        controller = FuzzyFileController(mock_view, cwd=project_root)
        controller.set_cwd(cwd)
        await controller.warm_cache()

        assert controller._file_cache == ["main.py"]

    async def test_warm_cache_force_refreshes_populated_cache(
        self, mock_view, monkeypatch, tmp_path
    ):
        """warm_cache(force=True) re-walks and swaps in a fresh list."""
        project_root = tmp_path
        (project_root / ".git").mkdir()

        files = ["main.py"]
        monkeypatch.setattr(
            autocomplete_module, "_get_project_files", lambda _root: list(files)
        )

        controller = FuzzyFileController(mock_view, cwd=project_root)
        await controller.warm_cache()
        assert controller._file_cache == ["main.py"]

        files.append("added.py")
        await controller.warm_cache()
        assert controller._file_cache == ["main.py"]

        await controller.warm_cache(force=True)
        assert controller._file_cache == ["main.py", "added.py"]

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
