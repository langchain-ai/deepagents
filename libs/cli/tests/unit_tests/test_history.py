"""Tests for HistoryManager substring search."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from deepagents_cli.widgets.history import HistoryManager


@pytest.fixture
def history(tmp_path: Path) -> HistoryManager:
    """Create a HistoryManager with sample entries."""
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


class TestForwardNavigation:
    """`get_next()` reuses the stored query."""

    def test_respects_query(self, history: HistoryManager) -> None:
        # Navigate back twice
        history.get_previous("up", query="up")
        history.get_previous("up", query="up")

        # Navigate forward — should return next matching entry
        entry = history.get_next()
        assert entry == "docker compose UP -d"

    def test_restores_original_input(self, history: HistoryManager) -> None:
        history.get_previous("my input", query="up")

        # Navigate forward past newest match
        entry = history.get_next()
        assert entry == "my input"


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
