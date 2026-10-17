"""Command history manager for input persistence."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class HistoryManager:
    """Manages command history with file persistence.

    Uses append-only writes for concurrent safety. Multiple agents can
    safely write to the same history file without corruption.
    """

    def __init__(self, history_file: Path, max_entries: int = 100) -> None:
        """Initialize the history manager.

        Args:
            history_file: Path to the JSON-lines history file
            max_entries: Maximum number of entries to keep
        """
        self.history_file = history_file
        self.max_entries = max_entries
        self._entries: list[str] = []
        self._failed_persist: list[str] = []
        self._read_failed = False
        self._write_failed = False
        self._current_index: int = -1
        self._temp_input: str = ""
        self._query: str = ""
        self._load_history()

    def _read_history(self) -> list[str] | None:
        """Read all persisted entries, tolerating legacy malformed lines.

        Sets `history_unreadable` so callers can tell a file they cannot read
        from one that is simply not there yet. The two are opposite facts to
        report to the user, and both arrive here as `None`.

        Returns:
            Persisted entries in file order, or `None` when the file could
            not be read.
        """
        if not self.history_file.exists():
            self._read_failed = False
            return None

        try:
            with self.history_file.open("r", encoding="utf-8") as f:
                entries = []
                for raw_line in f:
                    line = raw_line.rstrip("\n\r")
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        entry = line
                    entries.append(entry if isinstance(entry, str) else str(entry))
                self._read_failed = False
                return entries
        except (OSError, UnicodeDecodeError):
            logger.warning(
                "Failed to read history from %s",
                self.history_file,
                exc_info=True,
            )
            self._read_failed = True
            return None

    @property
    def history_unreadable(self) -> bool:
        """Whether the last read failed, as opposed to finding no file.

        Returns:
            `True` when the history file exists but could not be read.
        """
        return self._read_failed

    @property
    def history_unwritable(self) -> bool:
        """Whether any write to the history file has failed this session.

        Sticky by design: once a write fails, prompts recorded from then on
        live only in memory and are lost at exit. Callers use this to say so
        once rather than letting the loss go unmentioned.

        Returns:
            `True` when an append or a compaction has failed.
        """
        return self._write_failed

    def _load_history(self) -> None:
        """Load the bounded navigation history from file."""
        self._entries = (self._read_history() or [])[-self.max_entries :]

    def recent_prompts(self) -> tuple[str, ...]:
        """Refresh and return unique prompts in newest-first order.

        On a read failure the in-memory entries are kept, so prompts whose
        file append failed earlier in the session are not destroyed. Check
        `history_unreadable` afterwards to tell an unreadable file from an
        empty one.

        Side effects: re-reads the file into the in-memory entries and resets
        up-arrow navigation.

        When the read succeeds the dedup runs over the persisted file rather
        than the bounded navigation window, so this can surface prompts older
        than up-arrow reaches. (The file is itself bounded: `_compact_history`
        truncates it at twice `max_entries`.) On a read failure the fallback is
        the navigation window, so that reach is lost along with it.

        Returns:
            A bounded immutable prompt snapshot.
        """
        entries = self._read_history()
        if entries is None:
            entries = list(self._entries)
        else:
            # Each failed append is a distinct occurrence, even when the same
            # text already exists on disk. Keep those occurrences in submission
            # order so the newest one remains newest after a refresh.
            entries.extend(self._failed_persist)
            self._entries = entries[-self.max_entries :]
            # Failed occurrences are appended after persisted history, so only
            # their newest bounded suffix can remain in the navigation window.
            self._failed_persist = self._failed_persist[-self.max_entries :]
        self.reset_navigation()

        recent: list[str] = []
        seen: set[str] = set()
        for entry in reversed(entries):
            if entry in seen:
                continue
            seen.add(entry)
            recent.append(entry)
            if len(recent) == self.max_entries:
                break
        return tuple(recent)

    def _append_to_file(self, text: str) -> bool:
        """Append a single entry to history file (concurrent-safe).

        Returns:
            `True` when the entry was persisted, `False` otherwise.
        """
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with self.history_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(text) + "\n")
        except OSError:
            logger.warning(
                "Failed to append history entry to %s",
                self.history_file,
                exc_info=True,
            )
            self._write_failed = True
            return False
        return True

    def _compact_history(self) -> None:
        """Rewrite history file to remove old entries.

        Only called when entries exceed 2x max_entries to minimize rewrites.
        """
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with self.history_file.open("w", encoding="utf-8") as f:
                for entry in self._entries:
                    f.write(json.dumps(entry) + "\n")
        except OSError:
            logger.warning(
                "Failed to compact history file %s",
                self.history_file,
                exc_info=True,
            )
            self._write_failed = True

    def add(self, text: str) -> None:
        """Add a command to history.

        Args:
            text: The command text to add
        """
        text = text.strip()
        # Skip empty input and slash commands, except the explicit
        # `/skill:<name>` form (case-insensitive), which is kept so users can
        # recall it with up-arrow. Note: history stores the raw submitted text
        # *before* app-layer alias rewriting, so convenience aliases such as
        # `/remember` (later rewritten to `/skill:remember`) are dropped here
        # despite being skill invocations.
        lower_text = text.lower()
        if not text or (text.startswith("/") and not lower_text.startswith("/skill:")):
            return

        # Skip duplicates of the last entry
        if self._entries and self._entries[-1] == text:
            return

        self._entries.append(text)

        # Append to file (fast, concurrent-safe). The entry stays in memory
        # even when the write fails, so up-arrow recall keeps working for the
        # session; `_failed_persist` lets a later `recent_prompts` refresh
        # merge it back instead of dropping it.
        if not self._append_to_file(text):
            self._failed_persist.append(text)

        # Compact only when we have 2x max entries (rare operation)
        if len(self._entries) > self.max_entries * 2:
            self._entries = self._entries[-self.max_entries :]
            self._compact_history()

        self.reset_navigation()

    def get_previous(self, current_input: str, *, query: str = "") -> str | None:
        """Get the previous history entry matching a substring query.

        The query is captured on the first call of a navigation session
        (when `_current_index == -1`) and reused for all subsequent calls until
        `reset_navigation`. Passing a different value on later calls has
        no effect.

        Args:
            current_input: Current input text. Saved only on the first call of a
                navigation session; ignored on subsequent calls.
            query: Substring to match against history entries.
                Captured once on the first call of a navigation session.

        Returns:
            Previous matching entry or `None`.
        """
        if not self._entries:
            return None

        # Save current input and capture query on first navigation
        if self._current_index == -1:
            self._temp_input = current_input
            self._current_index = len(self._entries)
            self._query = query.strip().lower()

        # Search backwards for matching entry
        for i in range(self._current_index - 1, -1, -1):
            if not self._query or self._query in self._entries[i].lower():
                self._current_index = i
                return self._entries[i]

        return None

    def get_next(self) -> str | None:
        """Get the next history entry matching the stored query.

        Uses the query captured by the most recent `get_previous` call.

        Returns:
            The next matching entry, or the original input when past the newest
                match.

                `None` if not currently navigating history.
        """
        if self._current_index == -1:
            return None

        # Search forwards for matching entry
        for i in range(self._current_index + 1, len(self._entries)):
            if not self._query or self._query in self._entries[i].lower():
                self._current_index = i
                return self._entries[i]

        # Return to original input at the end
        result = self._temp_input
        self.reset_navigation()
        return result

    @property
    def in_history(self) -> bool:
        """Whether currently navigating history entries."""
        return self._current_index >= 0

    def reset_navigation(self) -> None:
        """Reset navigation state."""
        self._current_index = -1
        self._temp_input = ""
        self._query = ""
