"""Tests for autocomplete fuzzy search functionality."""

from typing import cast
from unittest.mock import MagicMock

import pytest

from deepagents_cli.command_registry import SLASH_COMMANDS
from deepagents_cli.widgets.autocomplete import (
    MAX_SUGGESTIONS,
    CompletionController,
    FuzzyFileController,
    MultiCompletionManager,
    SlashCommandController,
    _fuzzy_score,
    _fuzzy_search,
    _is_dotpath,
    _path_depth,
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
        return SlashCommandController(SLASH_COMMANDS, mock_view)

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
        assert len(suggestions) == min(len(SLASH_COMMANDS), MAX_SUGGESTIONS)

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
        assert len(suggestions) == min(len(SLASH_COMMANDS), MAX_SUGGESTIONS)

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

        controller.on_text_changed("/model gpt-4o", 13)
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
        slash_ctrl = SlashCommandController(SLASH_COMMANDS, mock_view)
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
        initial = [("/help", "Show help", "")]
        controller = SlashCommandController(initial, mock_view)

        new_commands = [
            ("/help", "Show help", ""),
            ("/skill:web-research", "Research topics", "web-research"),
        ]
        controller.update_commands(new_commands)

        # Typing /skill: should now show the skill command
        controller.on_text_changed("/skill:", 7)
        mock_view.render_completion_suggestions.assert_called()
        suggestions = mock_view.render_completion_suggestions.call_args[0][0]
        assert any("/skill:web-research" in s[0] for s in suggestions)

    def test_update_resets_suggestions(self, mock_view: MagicMock) -> None:
        """update_commands() clears any active suggestions."""
        commands = [("/help", "Show help", "")]
        controller = SlashCommandController(commands, mock_view)
        controller.on_text_changed("/h", 2)
        mock_view.render_completion_suggestions.assert_called()

        controller.update_commands([("/quit", "Exit", "")])
        mock_view.clear_completion_suggestions.assert_called()

    def test_skill_commands_fuzzy_match(self, mock_view: MagicMock) -> None:
        """Skill commands match via hidden keywords."""
        commands = [
            ("/help", "Show help", ""),
            ("/skill:code-review", "Review code changes", "code-review"),
        ]
        controller = SlashCommandController(commands, mock_view)
        controller.on_text_changed("/code", 5)

        mock_view.render_completion_suggestions.assert_called()
        suggestions = mock_view.render_completion_suggestions.call_args[0][0]
        assert any("/skill:code-review" in s[0] for s in suggestions)
