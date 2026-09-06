"""Unit tests for the prompt search module's pure helpers.

These functions are shared by both prompt clipboard tiers, so a change here
moves the inline panel and the modal together.
"""

from __future__ import annotations

import pytest

from deepagents_code.tui.widgets.prompt_search import (
    PROMPT_SEARCH_MAX_HINT_ROWS,
    PROMPT_SEARCH_WINDOW,
    _window_bounds,
    filter_prompts,
    prompt_search_hint,
    prompt_title,
)


class TestPromptTitle:
    """Cover the one-row summary, including the degenerate inputs."""

    def test_uses_the_first_non_empty_line(self) -> None:
        assert prompt_title("first line\nsecond line") == "first line"

    def test_skips_leading_blank_lines(self) -> None:
        assert prompt_title("\n\n  real content\nmore") == "real content"

    def test_strips_surrounding_whitespace(self) -> None:
        assert prompt_title("   padded   ") == "padded"

    @pytest.mark.parametrize("prompt", ["", "   ", "\n", "\t\n  \n"])
    def test_whitespace_only_prompts_get_a_visible_placeholder(
        self, prompt: str
    ) -> None:
        """A blank row would be indistinguishable from its neighbours.

        `_read_history` preserves malformed lines verbatim, so a history file
        containing a blank line is reachable.
        """
        assert prompt_title(prompt) == "(empty prompt)"


class TestFilterPrompts:
    """Cover matching, which decides what both tiers show."""

    PROMPTS = ("fix the bug", "Add A Feature", "write TESTS")

    def test_matches_a_substring(self) -> None:
        assert filter_prompts(self.PROMPTS, "bug") == ["fix the bug"]

    @pytest.mark.parametrize("query", ["FIX", "Fix", "fIx"])
    def test_query_case_is_ignored(self, query: str) -> None:
        assert filter_prompts(self.PROMPTS, query) == ["fix the bug"]

    @pytest.mark.parametrize("query", ["feature", "FEATURE", "FeAtUrE"])
    def test_prompt_case_is_ignored(self, query: str) -> None:
        assert filter_prompts(self.PROMPTS, query) == ["Add A Feature"]

    def test_surrounding_whitespace_is_stripped_from_the_query(self) -> None:
        assert filter_prompts(self.PROMPTS, "  bug  ") == ["fix the bug"]

    @pytest.mark.parametrize("query", ["", "   ", "\t"])
    def test_a_blank_query_matches_everything(self, query: str) -> None:
        """Stripping to empty must widen to the full list, not empty it."""
        assert filter_prompts(self.PROMPTS, query) == list(self.PROMPTS)

    def test_no_match_returns_an_empty_list(self) -> None:
        assert filter_prompts(self.PROMPTS, "nothing here") == []

    def test_order_is_preserved(self) -> None:
        assert filter_prompts(self.PROMPTS, "e") == [
            "fix the bug",
            "Add A Feature",
            "write TESTS",
        ]


class TestWindowBounds:
    """Cover the mounted-row window around the selection."""

    def test_short_lists_are_mounted_whole(self) -> None:
        assert _window_bounds(10, 4) == (0, 10)

    def test_window_is_clamped_to_the_list_start(self) -> None:
        start, stop = _window_bounds(PROMPT_SEARCH_WINDOW * 2, 0)
        assert (start, stop) == (0, PROMPT_SEARCH_WINDOW)

    def test_window_is_clamped_to_the_list_end(self) -> None:
        total = PROMPT_SEARCH_WINDOW * 2
        start, stop = _window_bounds(total, total - 1)
        assert stop == total
        assert start == total - PROMPT_SEARCH_WINDOW

    @pytest.mark.parametrize(
        ("total", "selected"),
        [(0, 0), (1, 0), (200, 0), (200, 100), (200, 199)],
    )
    def test_bounds_are_always_orderable_and_in_range(
        self, total: int, selected: int
    ) -> None:
        start, stop = _window_bounds(total, selected)
        assert 0 <= start <= stop <= total

    def test_the_selection_stays_inside_the_window(self) -> None:
        total = PROMPT_SEARCH_WINDOW * 3
        for selected in range(total):
            start, stop = _window_bounds(total, selected)
            assert start <= selected < stop


class TestPromptSearchHint:
    """The hint's length is a documented constraint, not a free-form string."""

    def test_hint_fits_the_reserved_rows_at_the_narrowest_width(self) -> None:
        """The reserved height clamps at `PROMPT_SEARCH_MAX_HINT_ROWS`.

        A hint too long to wrap within that budget is silently clipped, so
        adding another chord to it needs to be a deliberate decision rather
        than a surprise at runtime.
        """
        narrowest_supported_width = 40
        hint = prompt_search_hint()
        assert len(hint) <= narrowest_supported_width * PROMPT_SEARCH_MAX_HINT_ROWS
