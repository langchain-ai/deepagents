"""Unit tests for HistoryManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from deepagents_cli.widgets.history import HistoryManager


@pytest.fixture
def history(tmp_path: Path) -> HistoryManager:
    """Create a HistoryManager with sample entries for substring tests."""
    hm = HistoryManager(tmp_path / "history.jsonl")
    for cmd in [
        "git checkout main",
        "docker compose up",
        "docker compose UP -d",
        "git status",
    ]:
        hm.add(cmd)
    hm.reset_navigation()
    return hm


@pytest.fixture
def simple_history(tmp_path: Path) -> HistoryManager:
    """Create a HistoryManager with simple seed entries."""
    mgr = HistoryManager(tmp_path / "history.jsonl")
    mgr._entries = ["first", "second", "third"]
    return mgr


class TestSubstringMatch:
    """Substring matching navigates to entries containing the query."""

    def test_matches_substring_anywhere(self, history: HistoryManager) -> None:
        entry = history.get_previous("up", query="up")
        assert entry == "docker compose UP -d"

        entry = history.get_previous("up", query="up")
        assert entry == "docker compose up"

    def test_skips_non_matching_entries(self, history: HistoryManager) -> None:
        entry = history.get_previous("up", query="up")
        assert entry == "docker compose UP -d"

        entry = history.get_previous("up", query="up")
        assert entry == "docker compose up"

        # No more matches
        entry = history.get_previous("up", query="up")
        assert entry is None

    def test_case_insensitive(self, history: HistoryManager) -> None:
        entry = history.get_previous("UP", query="UP")
        assert entry == "docker compose UP -d"

        entry = history.get_previous("UP", query="UP")
        assert entry == "docker compose up"


class TestEmptyQuery:
    """Empty query walks through all entries (backward compatible)."""

    def test_returns_all_entries_in_reverse(self, history: HistoryManager) -> None:
        entries = []
        entry = history.get_previous("", query="")
        while entry is not None:
            entries.append(entry)
            entry = history.get_previous("", query="")

        assert entries == [
            "git status",
            "docker compose UP -d",
            "docker compose up",
            "git checkout main",
        ]


class TestNoMatch:
    """Non-matching query returns None."""

    def test_returns_none(self, history: HistoryManager) -> None:
        entry = history.get_previous("xyz", query="xyz")
        assert entry is None

    def test_empty_history_returns_none(self, tmp_path: Path) -> None:
        mgr = HistoryManager(tmp_path / "empty.jsonl")
        assert mgr.get_previous("text", query="text") is None


class TestForwardNavigation:
    """`get_next()` reuses the stored query."""

    def test_respects_query(self, history: HistoryManager) -> None:
        # Navigate back twice
        history.get_previous("up", query="up")
        history.get_previous("up", query="up")

        # Navigate forward — should return next matching entry
        entry = history.get_next()
        assert entry == "docker compose UP -d"

    def test_full_forward_walk(self, history: HistoryManager) -> None:
        """Walk back to oldest match, then forward through all matches."""
        history.get_previous("x", query="compose")  # -> "docker compose UP -d"
        history.get_previous("x", query="compose")  # -> "docker compose up"
        assert history.get_previous("x", query="compose") is None

        assert history.get_next() == "docker compose UP -d"
        assert history.get_next() == "x"  # original input restored

    def test_restores_original_input(self, history: HistoryManager) -> None:
        history.get_previous("my input", query="up")

        # Navigate forward past newest match
        entry = history.get_next()
        assert entry == "my input"

    def test_get_next_without_previous_returns_none(
        self, history: HistoryManager
    ) -> None:
        assert history.get_next() is None


class TestResetClearsQuery:
    """`reset_navigation()` clears query state."""

    def test_reset_then_empty_query(self, history: HistoryManager) -> None:
        # Navigate with a query
        history.get_previous("up", query="up")
        history.reset_navigation()

        # After reset, empty query should walk all entries
        entry = history.get_previous("", query="")
        assert entry == "git status"


class TestWhitespaceQuery:
    """Whitespace-only query is treated as empty (matches everything)."""

    def test_whitespace_treated_as_empty(self, history: HistoryManager) -> None:
        entry = history.get_previous("", query="   ")
        assert entry == "git status"


class TestQueryCapturedOnce:
    """Query from first call is used; subsequent queries are ignored."""

    def test_subsequent_query_ignored(self, history: HistoryManager) -> None:
        entry = history.get_previous("compose", query="compose")
        assert entry == "docker compose UP -d"

        # Second call with different query — should still use "compose"
        entry = history.get_previous("compose", query="git")
        assert entry == "docker compose up"


class TestInHistoryProperty:
    """Test HistoryManager.in_history property."""

    def test_initial_state_is_false(self, tmp_path: Path) -> None:
        """in_history should be False before any navigation."""
        mgr = HistoryManager(tmp_path / "history.jsonl")
        assert mgr.in_history is False

    def test_true_after_get_previous(self, simple_history: HistoryManager) -> None:
        """in_history should be True after get_previous returns an entry."""
        entry = simple_history.get_previous("")
        assert entry is not None
        assert simple_history.in_history is True

    def test_true_while_browsing(self, simple_history: HistoryManager) -> None:
        """in_history should stay True while navigating through entries."""
        simple_history.get_previous("")
        assert simple_history.in_history is True

        simple_history.get_previous("")
        assert simple_history.in_history is True

    def test_false_after_get_next_past_end(
        self, simple_history: HistoryManager
    ) -> None:
        """in_history should be False after navigating past the newest entry."""
        simple_history.get_previous("current text")
        assert simple_history.in_history is True

        # Navigate forward past the end — returns to original input
        simple_history.get_next()
        assert simple_history.in_history is False

    def test_false_after_reset_navigation(self, simple_history: HistoryManager) -> None:
        """in_history should be False after explicit reset."""
        simple_history.get_previous("")
        assert simple_history.in_history is True

        simple_history.reset_navigation()
        assert simple_history.in_history is False

    def test_false_after_add(self, simple_history: HistoryManager) -> None:
        """in_history should be False after add() since it calls reset_navigation."""
        simple_history.get_previous("")
        assert simple_history.in_history is True

        simple_history.add("new entry")
        assert simple_history.in_history is False

    def test_in_history_stays_true_when_filtered_exhausted(
        self, history: HistoryManager
    ) -> None:
        """in_history stays True when a filtered query exhausts all matches."""
        history.get_previous("up", query="up")
        history.get_previous("up", query="up")
        history.get_previous("up", query="up")  # None — no more matches
        assert history.in_history is True

    def test_true_at_oldest_entry(self, simple_history: HistoryManager) -> None:
        """in_history should stay True when at the oldest entry with no older match."""
        # Navigate to oldest
        simple_history.get_previous("")
        simple_history.get_previous("")
        simple_history.get_previous("")
        assert simple_history.in_history is True

        # Try to go further back — returns None but stays in history
        result = simple_history.get_previous("")
        assert result is None
        assert simple_history.in_history is True
