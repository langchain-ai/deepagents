"""Unit tests for HistoryManager."""

from __future__ import annotations

import json
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

    def test_normal_absolute_path_added(self, tmp_path: Path) -> None:
        """Normal-mode absolute paths should be stored despite their slash."""
        mgr = HistoryManager(tmp_path / "history.jsonl")

        mgr.add("/tmp/assets what is in here", mode="normal")

        assert mgr.get_previous("") == "/tmp/assets what is in here"
        assert mgr.current_mode == "normal"

    def test_normal_mode_persists_across_reload(self, tmp_path: Path) -> None:
        """Reloaded path history retains its original submission mode."""
        history_file = tmp_path / "history.jsonl"
        mgr = HistoryManager(history_file)
        mgr.add("/tmp/assets", mode="normal")

        reloaded = HistoryManager(history_file)

        assert reloaded.get_previous("") == "/tmp/assets"
        assert reloaded.current_mode == "normal"

    def test_same_text_with_different_modes_is_not_deduplicated(
        self, tmp_path: Path
    ) -> None:
        """Mode metadata participates in adjacent-entry deduplication."""
        mgr = HistoryManager(tmp_path / "history.jsonl")
        mgr.add("same text", mode="normal")
        mgr.add("same text", mode="shell")

        assert mgr.get_previous("") == "same text"
        assert mgr.current_mode == "shell"
        assert mgr.get_previous("") == "same text"
        assert mgr.current_mode == "normal"


class TestStorageFormatCompatibility:
    """The on-disk shapes an older client sharing `history.jsonl` must survive."""

    def test_inferable_modes_stay_bare_strings(self, tmp_path: Path) -> None:
        """Only entries whose mode cannot be re-derived pay the object form."""
        history_file = tmp_path / "history.jsonl"
        mgr = HistoryManager(history_file)

        mgr.add("plain message", mode="normal")
        mgr.add("/skill:web-research cats", mode="command")
        mgr.add("!ls", mode="shell")
        # Only this one is ambiguous: the text looks exactly like a command.
        mgr.add("/tmp/assets", mode="normal")

        lines = history_file.read_text(encoding="utf-8").splitlines()
        assert lines[0] == '"plain message"'
        assert lines[1] == '"/skill:web-research cats"'
        assert lines[2] == '"!ls"'
        assert json.loads(lines[3]) == {"text": "/tmp/assets", "mode": "normal"}

    def test_compaction_preserves_both_shapes(self, tmp_path: Path) -> None:
        """A rewrite keeps modes attached to the right entries."""
        history_file = tmp_path / "history.jsonl"
        mgr = HistoryManager(history_file, max_entries=2)

        mgr.add("first", mode="normal")
        mgr.add("/tmp/assets", mode="normal")
        for index in range(6):
            mgr.add(f"filler {index}", mode="normal")

        reloaded = HistoryManager(history_file, max_entries=2)
        assert len(reloaded._entries) == len(reloaded._entry_modes)
        for entry, mode in zip(reloaded._entries, reloaded._entry_modes, strict=True):
            # Only the ambiguous path keeps stored metadata; everything else
            # is inferable and so round-trips as a bare string.
            assert mode == ("normal" if entry.startswith("/") else None)

    def test_legacy_and_new_lines_stay_index_aligned(self, tmp_path: Path) -> None:
        """A file written by two client versions loads without drift."""
        history_file = tmp_path / "history.jsonl"
        history_file.write_text(
            '"legacy one"\n{"text": "/tmp/assets", "mode": "normal"}\n"legacy two"\n',
            encoding="utf-8",
        )

        mgr = HistoryManager(history_file)

        assert mgr._entries == ["legacy one", "/tmp/assets", "legacy two"]
        assert mgr._entry_modes == [None, "normal", None]

    def test_unrecognized_mode_is_ignored(self, tmp_path: Path) -> None:
        """A bogus mode falls back to prefix detection instead of being trusted."""
        history_file = tmp_path / "history.jsonl"
        history_file.write_text(
            '{"text": "/tmp/assets", "mode": "wat"}\n', encoding="utf-8"
        )

        mgr = HistoryManager(history_file)

        assert mgr._entries == ["/tmp/assets"]
        assert mgr._entry_modes == [None]

    def test_malformed_entries_are_skipped_not_stringified(
        self, tmp_path: Path
    ) -> None:
        """A corrupt line is dropped rather than recalled as a Python repr."""
        history_file = tmp_path / "history.jsonl"
        history_file.write_text(
            '{"text": 123, "mode": "normal"}\n["a", "b"]\n"good entry"\n',
            encoding="utf-8",
        )

        mgr = HistoryManager(history_file)

        assert mgr._entries == ["good entry"]
        assert mgr._entry_modes == [None]


class TestDraftRestoreMode:
    """Walking past the newest entry restores the draft's own mode."""

    def test_restored_draft_keeps_its_mode(
        self, simple_history: HistoryManager
    ) -> None:
        """The draft comes back in the mode it was captured in."""
        assert (
            simple_history.get_previous("/tmp/assets", current_mode="normal")
            is not None
        )

        while (entry := simple_history.get_next()) != "/tmp/assets":
            assert entry is not None, "walked past the draft without restoring it"

        assert simple_history.current_mode == "normal"

    def test_stored_entry_mode_still_wins_while_navigating(
        self, tmp_path: Path
    ) -> None:
        """Draft capture does not overwrite the mode of real entries."""
        mgr = HistoryManager(tmp_path / "history.jsonl")
        mgr.add("!ls", mode="shell")

        assert mgr.get_previous("draft text", current_mode="normal") == "!ls"
        assert mgr.current_mode == "shell"

        assert mgr.get_next() == "draft text"
        assert mgr.current_mode == "normal"


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
