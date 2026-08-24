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

    def test_returns_newest_first_and_deduplicates(self, tmp_path: Path) -> None:
        history_file = tmp_path / "history.jsonl"
        history_file.write_text(
            "".join(
                json.dumps(entry) + "\n"
                for entry in ("first", "duplicate", "second", "duplicate")
            ),
            encoding="utf-8",
        )

        mgr = HistoryManager(history_file)

        assert mgr.recent_prompts() == ("duplicate", "second", "first")

    def test_refreshes_concurrent_appends(self, tmp_path: Path) -> None:
        history_file = tmp_path / "history.jsonl"
        mgr = HistoryManager(history_file)
        mgr.add("first")
        with history_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps("from another process") + "\n")

        assert mgr.recent_prompts() == ("from another process", "first")
        assert mgr.get_previous("") == "from another process"

    def test_bounds_unique_entries_after_deduplication(self, tmp_path: Path) -> None:
        history_file = tmp_path / "history.jsonl"
        history_file.write_text(
            "".join(
                json.dumps(entry) + "\n"
                for entry in ("old unique", "repeat", "new unique", "repeat")
            ),
            encoding="utf-8",
        )

        mgr = HistoryManager(history_file, max_entries=3)

        assert mgr.recent_prompts() == ("repeat", "new unique", "old unique")

    def test_preserves_malformed_line_fallback(self, tmp_path: Path) -> None:
        history_file = tmp_path / "history.jsonl"
        history_file.write_text('"valid"\nnot-json\n', encoding="utf-8")

        mgr = HistoryManager(history_file)

        assert mgr.recent_prompts() == ("not-json", "valid")

    def test_includes_only_slash_commands_eligible_for_history(
        self, tmp_path: Path
    ) -> None:
        mgr = HistoryManager(tmp_path / "history.jsonl")
        mgr.add("regular prompt")
        mgr.add("/help")
        mgr.add("/skill:remember this")

        assert mgr.recent_prompts() == ("/skill:remember this", "regular prompt")

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

    def test_history_unreadable_is_false_when_the_file_is_absent(
        self, tmp_path: Path
    ) -> None:
        """An absent file and an unreadable one are opposite facts."""
        mgr = HistoryManager(tmp_path / "never-written.jsonl")
        assert mgr.recent_prompts() == ()
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
