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

    def test_skill_invocation_added(self, tmp_path: Path) -> None:
        """`/skill:web-research` should be stored in history."""
        mgr = HistoryManager(tmp_path / "history.jsonl")
        mgr.add("/skill:web-research find quantum computing")
        assert mgr._entries == ["/skill:web-research find quantum computing"]

    def test_skill_invocation_without_args_added(self, tmp_path: Path) -> None:
        """`/skill:remember` with no args should be stored."""
        mgr = HistoryManager(tmp_path / "history.jsonl")
        mgr.add("/skill:remember")
        assert mgr._entries == ["/skill:remember"]

    def test_mixed_case_skill_invocation_added(self, tmp_path: Path) -> None:
        """Mixed-case `/skill:` invocations should be stored as typed."""
        mgr = HistoryManager(tmp_path / "history.jsonl")
        mgr.add("/Skill:web-research find quantum computing")
        mgr.add("/SKILL:remember")
        assert mgr._entries == [
            "/Skill:web-research find quantum computing",
            "/SKILL:remember",
        ]

    def test_non_skill_slash_command_not_added(self, tmp_path: Path) -> None:
        """`/help` and other slash commands should not be stored."""
        mgr = HistoryManager(tmp_path / "history.jsonl")
        mgr.add("/help")
        mgr.add("/quit")
        mgr.add("/model")
        assert mgr._entries == []

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

    def test_skill_invocation_dedup(self, tmp_path: Path) -> None:
        """Duplicate skill invocations are deduplicated like normal entries."""
        mgr = HistoryManager(tmp_path / "history.jsonl")
        mgr.add("/skill:web-research find cats")
        mgr.add("/skill:web-research find cats")
        assert mgr._entries == ["/skill:web-research find cats"]

    def test_skill_listing_command_not_added(self, tmp_path: Path) -> None:
        """`/skill` without a colon is the listing command, not an invocation."""
        mgr = HistoryManager(tmp_path / "history.jsonl")
        mgr.add("/skill")
        assert mgr._entries == []

    def test_skill_prefix_collision_not_added(self, tmp_path: Path) -> None:
        """Slash commands that merely start with `/skill` (no colon) are dropped."""
        mgr = HistoryManager(tmp_path / "history.jsonl")
        mgr.add("/skilling")
        mgr.add("/skill-creator")
        assert mgr._entries == []

    def test_skill_invocation_empty_name_added(self, tmp_path: Path) -> None:
        """A bare `/skill:` (empty name) still matches the prefix and is stored.

        Upstream parsing rejects empty skill names, so this form is not expected
        in practice; the test pins current behavior to catch silent changes.
        """
        mgr = HistoryManager(tmp_path / "history.jsonl")
        mgr.add("/skill:")
        assert mgr._entries == ["/skill:"]

    def test_skill_alias_not_added(self, tmp_path: Path) -> None:
        """Convenience aliases are dropped because history precedes rewriting.

        A user typing `/remember` submits the raw alias, which is rewritten to
        `/skill:remember` only in the app layer after history capture.
        """
        mgr = HistoryManager(tmp_path / "history.jsonl")
        mgr.add("/remember something useful")
        assert mgr._entries == []

    def test_skill_invocation_persists_across_reload(self, tmp_path: Path) -> None:
        """Stored skill invocations survive into a new session via the file."""
        history_file = tmp_path / "history.jsonl"
        mgr = HistoryManager(history_file)
        mgr.add("/skill:web-research find cats")

        reloaded = HistoryManager(history_file)
        assert reloaded._entries == ["/skill:web-research find cats"]


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

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores read-only mode")
    def test_read_failure_preserves_session_prompts(self, tmp_path: Path) -> None:
        """A failed refresh keeps in-memory prompts instead of wiping them."""
        history_file = tmp_path / "history.jsonl"
        history_file.write_text(json.dumps("persisted") + "\n", encoding="utf-8")

        mgr = HistoryManager(history_file)
        mgr.add("session only")
        assert mgr._entries == ["persisted", "session only"]

        history_file.chmod(0o000)
        try:
            assert mgr.recent_prompts() == ("session only", "persisted")
            assert mgr._entries == ["persisted", "session only"]
        finally:
            history_file.chmod(0o600)

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

    def test_failed_persist_is_bounded_by_max_entries(self, tmp_path: Path) -> None:
        """Unpersisted entries stop being tracked once they age out."""
        history_file = tmp_path / "history.jsonl"
        blocker = tmp_path / "blocker"
        blocker.touch()
        mgr = HistoryManager(history_file, max_entries=3)
        mgr.history_file = blocker / "history.jsonl"

        for index in range(10):
            mgr.add(f"prompt {index}")
        assert len(mgr._failed_persist) == 10

        # The bounding runs on a successful read, so the file has to exist.
        history_file.write_text(json.dumps("from disk") + "\n", encoding="utf-8")
        mgr.history_file = history_file
        mgr.recent_prompts()

        # Only the newest bounded suffix can still reach the navigation window.
        assert len(mgr._failed_persist) <= mgr.max_entries

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

    def test_history_unwritable_latches_on_a_failed_compaction(
        self, tmp_path: Path
    ) -> None:
        """Compaction failures count too; they silently drop trimmed entries."""
        history_file = tmp_path / "history.jsonl"
        mgr = HistoryManager(history_file, max_entries=2)
        mgr.add("one")
        assert mgr.history_unwritable is False

        blocker = tmp_path / "blocker"
        blocker.touch()
        mgr.history_file = blocker / "history.jsonl"
        mgr._entries = [f"entry {index}" for index in range(6)]
        mgr._compact_history()

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
