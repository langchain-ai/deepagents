"""Unit tests for HistoryManager."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from deepagents_code.tui.widgets.history import HistoryManager


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


class TestSkillInvocationHistory:
    """Explicit `/skill:<name>` commands are stored; other slash commands are not.

    History stores the raw submitted text before app-layer alias rewriting, so
    convenience aliases (e.g. `/remember`) are dropped here even though they
    resolve to skills downstream.
    """

    def test_skill_invocation_recallable(self, tmp_path: Path) -> None:
        """Skill invocations should be recallable via get_previous."""
        mgr = HistoryManager(tmp_path / "history.jsonl")
        mgr.add("some regular text")
        mgr.add("/skill:web-research find cats")
        mgr.reset_navigation()

        entry = mgr.get_previous("", query="")
        assert entry == "/skill:web-research find cats"

        entry = mgr.get_previous("", query="")
        assert entry == "some regular text"


class TestRecentPrompts:
    """Prompt snapshots refresh, deduplicate, and bound persisted history."""

    def test_refreshes_concurrent_appends(self, tmp_path: Path) -> None:
        history_file = tmp_path / "history.jsonl"
        mgr = HistoryManager(history_file)
        mgr.add("first")
        with history_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps("from another process") + "\n")

        assert mgr.recent_prompts() == ("from another process", "first")
        assert mgr.get_previous("") == "from another process"

    def test_failed_append_survives_refresh(self, tmp_path: Path) -> None:
        """A prompt that could not be written stays through a later refresh."""
        history_file = tmp_path / "history.jsonl"
        mgr = HistoryManager(history_file)
        mgr.add("persisted")

        # Point the manager at a path whose parent is a regular file, so the
        # append's mkdir/open raises OSError and the entry stays in memory.
        blocker = tmp_path / "blocker"
        blocker.touch()
        mgr.history_file = blocker / "history.jsonl"
        mgr.add("append failed")

        mgr.history_file = history_file
        with history_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps("from another process") + "\n")

        assert mgr.recent_prompts() == (
            "append failed",
            "from another process",
            "persisted",
        )
        assert mgr.get_previous("") == "append failed"

    def test_history_unreadable_resets_after_a_successful_read(
        self, tmp_path: Path
    ) -> None:
        """A sticky flag would report a transient failure forever.

        Both empty-state messages and the new degraded-list warning hang on
        this flag, so it has to clear once the file is readable again.
        """
        history_file = tmp_path / "history.jsonl"
        mgr = HistoryManager(history_file)
        mgr.add("persisted")
        assert mgr.history_unreadable is False

        # A directory in place of the file is an OSError on read.
        blocker = tmp_path / "blocker"
        blocker.mkdir()
        mgr.history_file = blocker
        mgr.recent_prompts()
        assert mgr.history_unreadable is True

        mgr.history_file = history_file
        mgr.recent_prompts()
        assert mgr.history_unreadable is False

    def test_history_unwritable_latches_on_a_failed_append(
        self, tmp_path: Path
    ) -> None:
        """A failed write is sticky: later prompts are memory-only too."""
        history_file = tmp_path / "history.jsonl"
        mgr = HistoryManager(history_file)
        mgr.add("persisted")
        assert mgr.history_unwritable is False

        blocker = tmp_path / "blocker"
        blocker.touch()
        mgr.history_file = blocker / "history.jsonl"
        mgr.add("append failed")

        assert mgr.history_unwritable is True

        # Stays true after a working path is restored: the earlier prompt is
        # still only in memory, so the session's history is still incomplete.
        mgr.history_file = history_file
        mgr.add("persisted again")
        assert mgr.history_unwritable is True

    def test_failed_duplicate_append_stays_newest(self, tmp_path: Path) -> None:
        """A failed repeat remains newer than its persisted occurrence."""
        history_file = tmp_path / "history.jsonl"
        mgr = HistoryManager(history_file)
        mgr.add("A")
        mgr.add("B")

        blocker = tmp_path / "blocker"
        blocker.touch()
        mgr.history_file = blocker / "history.jsonl"
        mgr.add("A")

        mgr.history_file = history_file
        assert mgr.recent_prompts() == ("A", "B")
        assert mgr.get_previous("") == "A"
        assert mgr.get_previous("") == "B"
        assert mgr.get_previous("") == "A"


class TestSubstringMatch:
    """Substring matching navigates to entries containing the query."""


class TestEmptyQuery:
    """Empty query walks through all entries (backward compatible)."""


class TestNoMatch:
    """Non-matching query returns None."""


class TestForwardNavigation:
    """`get_next()` reuses the stored query."""


class TestResetClearsQuery:
    """`reset_navigation()` clears query state."""


class TestWhitespaceQuery:
    """Whitespace-only query is treated as empty (matches everything)."""


class TestQueryCapturedOnce:
    """Query from first call is used; subsequent queries are ignored."""


class TestInHistoryProperty:
    """Test HistoryManager.in_history property."""
