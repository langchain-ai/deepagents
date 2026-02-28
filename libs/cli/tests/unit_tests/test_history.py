"""Unit tests for HistoryManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from deepagents_cli.widgets.history import HistoryManager


@pytest.fixture
def history(tmp_path: Path) -> HistoryManager:
    """Create a HistoryManager with a temp file and seed entries."""
    mgr = HistoryManager(tmp_path / "history.jsonl")
    mgr._entries = ["first", "second", "third"]
    return mgr


class TestInHistoryProperty:
    """Test HistoryManager.in_history property."""

    def test_initial_state_is_false(self, tmp_path: Path) -> None:
        """in_history should be False before any navigation."""
        mgr = HistoryManager(tmp_path / "history.jsonl")
        assert mgr.in_history is False

    def test_true_after_get_previous(self, history: HistoryManager) -> None:
        """in_history should be True after get_previous returns an entry."""
        entry = history.get_previous("")
        assert entry is not None
        assert history.in_history is True

    def test_true_while_browsing(self, history: HistoryManager) -> None:
        """in_history should stay True while navigating through entries."""
        history.get_previous("")
        assert history.in_history is True

        history.get_previous("")
        assert history.in_history is True

    def test_false_after_get_next_past_end(self, history: HistoryManager) -> None:
        """in_history should be False after navigating past the newest entry."""
        history.get_previous("current text")
        assert history.in_history is True

        # Navigate forward past the end — returns to original input
        history.get_next()
        assert history.in_history is False

    def test_false_after_reset_navigation(self, history: HistoryManager) -> None:
        """in_history should be False after explicit reset."""
        history.get_previous("")
        assert history.in_history is True

        history.reset_navigation()
        assert history.in_history is False

    def test_false_after_add(self, history: HistoryManager) -> None:
        """in_history should be False after add() since it calls reset_navigation."""
        history.get_previous("")
        assert history.in_history is True

        history.add("new entry")
        assert history.in_history is False

    def test_true_at_oldest_entry(self, history: HistoryManager) -> None:
        """in_history should stay True when at the oldest entry with no older match."""
        # Navigate to oldest
        history.get_previous("")
        history.get_previous("")
        history.get_previous("")
        assert history.in_history is True

        # Try to go further back — returns None but stays in history
        result = history.get_previous("")
        assert result is None
        assert history.in_history is True
