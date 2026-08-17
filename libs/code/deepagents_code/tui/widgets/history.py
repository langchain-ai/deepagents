"""Command history manager for input persistence."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from deepagents_code.config import MODE_PREFIXES, detect_mode_prefix

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_VALID_MODES: frozenset[str] = frozenset(MODE_PREFIXES) | {"normal"}
"""Submission modes accepted from storage.

Anything else is a corrupted or future-format line; it is dropped rather than
trusted, because a bogus mode silently changes how a recalled entry submits.
"""


def _mode_is_recoverable(text: str, mode: str) -> bool:
    """Return whether `mode` can be re-derived from `text` alone.

    Storage keeps such entries as bare JSON strings so older clients -- which
    read the same `history.jsonl` and have no concept of the object form --
    keep rendering them as text instead of a Python repr. Only entries whose
    mode genuinely cannot be inferred (a normal-mode absolute path, which looks
    exactly like a command) pay the compatibility cost of the object form.

    Args:
        text: Submitted text, including any mode prefix.
        mode: Submission mode recorded for the entry.

    Returns:
        `True` when a reader that only inspects the text would infer `mode`.
    """
    detected = detect_mode_prefix(text)
    inferred = detected[1] if detected is not None else "normal"
    return inferred == mode


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
        # Parallel arrays: `_entry_modes[i]` is the mode recorded for
        # `_entries[i]`, or `None` for legacy entries that carry no metadata.
        # Every mutation must keep the two the same length and order --
        # `current_mode` and `_compact_history` both index across them, so a
        # drift would attach one entry's mode to another's text.
        self._entries: list[str] = []
        self._entry_modes: list[str | None] = []
        self._current_index: int = -1
        self._temp_input: str = ""
        # Mode of the in-progress draft saved alongside `_temp_input`. Restoring
        # the draft must restore its mode too: a dropped path is normal-mode
        # text that re-detection would otherwise read back as a slash command.
        self._temp_mode: str | None = None
        # Mode belonging to the value most recently returned by `get_previous`
        # or `get_next`. Derived explicitly rather than from `_current_index`,
        # because `get_next` resets navigation before returning the draft.
        self._navigation_mode: str | None = None
        self._query: str = ""
        self._load_history()

    def _load_history(self) -> None:
        """Load history from file."""
        if not self.history_file.exists():
            return

        try:
            with self.history_file.open("r", encoding="utf-8") as f:
                entries: list[str] = []
                modes: list[str | None] = []
                for raw_line in f:
                    line = raw_line.rstrip("\n\r")
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        entry = line
                    if isinstance(entry, dict):
                        text = entry.get("text")
                        if not isinstance(text, str):
                            # A dict without usable text is corrupt or from a
                            # future format. Stringifying it would put a Python
                            # repr in the recall UI, so drop it and say so.
                            logger.warning(
                                "Skipping malformed history entry in %s: %r",
                                self.history_file,
                                entry,
                            )
                            continue
                        entries.append(text)
                        modes.append(self._validated_mode(entry.get("mode")))
                    elif isinstance(entry, str):
                        entries.append(entry)
                        modes.append(None)
                    else:
                        logger.warning(
                            "Skipping non-string history entry in %s: %r",
                            self.history_file,
                            entry,
                        )
                        continue
                self._entries = entries[-self.max_entries :]
                self._entry_modes = modes[-self.max_entries :]
        except (OSError, UnicodeDecodeError):
            logger.warning(
                "Failed to load history from %s; starting with empty history",
                self.history_file,
                exc_info=True,
            )
            self._entries = []
            self._entry_modes = []

    @staticmethod
    def _validated_mode(mode: object) -> str | None:
        """Return `mode` when it is a recognized submission mode, else `None`.

        Args:
            mode: Raw `mode` value read from a stored entry.

        Returns:
            The mode string, or `None` for absent, non-string, or unrecognized
            values (which then fall back to prefix detection at recall time).
        """
        if mode is None:
            return None
        if isinstance(mode, str) and mode in _VALID_MODES:
            return mode
        logger.warning("Ignoring unrecognized history entry mode: %r", mode)
        return None

    @staticmethod
    def _serialize_entry(text: str, mode: str | None) -> str | dict[str, str]:
        """Return the on-disk form for one entry.

        Two shapes share the file: a bare JSON string (the historical format,
        still used whenever the mode is implied by the text) and a
        `{"text", "mode"}` object for the entries that genuinely need metadata.
        `_load_history` reads both. Keeping the string form as the common case
        is what lets an older client share `history.jsonl` without rendering
        every recalled line as a Python repr.

        Args:
            text: Submitted text, including any mode prefix.
            mode: Submission mode when known.

        Returns:
            The JSON-serializable entry to write.
        """
        if mode is None or _mode_is_recoverable(text, mode):
            return text
        return {"text": text, "mode": mode}

    def _append_to_file(self, text: str, mode: str | None) -> None:
        """Append a single entry to history file (concurrent-safe).

        Args:
            text: Submitted text, including any mode prefix.
            mode: Submission mode when known. See `_serialize_entry` for how
                the two on-disk shapes are chosen.
        """
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with self.history_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(self._serialize_entry(text, mode)) + "\n")
        except OSError:
            logger.warning(
                "Failed to append history entry to %s",
                self.history_file,
                exc_info=True,
            )

    def _compact_history(self) -> None:
        """Rewrite history file to remove old entries.

        Only called when entries exceed 2x max_entries to minimize rewrites.
        """
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with self.history_file.open("w", encoding="utf-8") as f:
                for index, text in enumerate(self._entries):
                    mode = (
                        self._entry_modes[index]
                        if index < len(self._entry_modes)
                        else None
                    )
                    f.write(json.dumps(self._serialize_entry(text, mode)) + "\n")
        except OSError:
            logger.warning(
                "Failed to compact history file %s",
                self.history_file,
                exc_info=True,
            )

    def add(self, text: str, *, mode: str | None = None) -> None:
        """Add a command to history.

        Args:
            text: The command text to add
            mode: Submission mode when known. Normal-mode absolute paths are
                retained even though they begin with `/`.
        """
        text = text.strip()
        # Skip empty input and slash commands, except the explicit
        # `/skill:<name>` form (case-insensitive), which is kept so users can
        # recall it with up-arrow. Note: history stores the raw submitted text
        # *before* app-layer alias rewriting, so convenience aliases such as
        # `/remember` (later rewritten to `/skill:remember`) are dropped here
        # despite being skill invocations.
        lower_text = text.lower()
        is_command = mode == "command" if mode is not None else text.startswith("/")
        if not text or (is_command and not lower_text.startswith("/skill:")):
            return

        # Defensive: the two lists are written together everywhere, so they can
        # only diverge if a future edit appends to `_entries` alone. Pad rather
        # than let the index alignment `current_mode` depends on silently slip.
        if len(self._entry_modes) < len(self._entries):
            self._entry_modes.extend(
                [None] * (len(self._entries) - len(self._entry_modes))
            )
        # Mode is part of an entry's meaning: the same slash-prefixed text may
        # be a literal dropped path in normal mode or a command.
        if (
            self._entries
            and self._entries[-1] == text
            and self._entry_modes[-1] == mode
        ):
            return

        self._entries.append(text)
        self._entry_modes.append(mode)

        # Append to file (fast, concurrent-safe)
        self._append_to_file(text, mode)

        # Compact only when we have 2x max entries (rare operation)
        if len(self._entries) > self.max_entries * 2:
            self._entries = self._entries[-self.max_entries :]
            self._entry_modes = self._entry_modes[-self.max_entries :]
            self._compact_history()

        self.reset_navigation()

    def get_previous(
        self, current_input: str, *, query: str = "", current_mode: str | None = None
    ) -> str | None:
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
            current_mode: Mode the saved draft was in. Captured with
                `current_input` and handed back by `current_mode` when
                `get_next` walks past the newest entry, so restoring the draft
                restores the mode it was typed in rather than re-detecting it.

        Returns:
            Previous matching entry or `None`.
        """
        if not self._entries:
            return None

        # Save current input and capture query on first navigation
        if self._current_index == -1:
            self._temp_input = current_input
            self._temp_mode = current_mode
            self._current_index = len(self._entries)
            self._query = query.strip().lower()

        # Search backwards for matching entry
        for i in range(self._current_index - 1, -1, -1):
            if not self._query or self._query in self._entries[i].lower():
                self._current_index = i
                self._navigation_mode = self._mode_at(i)
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
                self._navigation_mode = self._mode_at(i)
                return self._entries[i]

        # Return to original input at the end. `reset_navigation` clears the
        # saved mode, so re-publish it afterwards: the caller reads
        # `current_mode` after this returns, and the restored draft is the
        # user's own text, not a stored entry to re-detect.
        result = self._temp_input
        restored_mode = self._temp_mode
        self.reset_navigation()
        self._navigation_mode = restored_mode
        return result

    def _mode_at(self, index: int) -> str | None:
        """Return the mode recorded for `index`, tolerating a short mode list.

        Args:
            index: Index into `_entries`.

        Returns:
            The recorded mode, or `None` when none is stored for that entry.
        """
        if 0 <= index < len(self._entry_modes):
            return self._entry_modes[index]
        return None

    @property
    def in_history(self) -> bool:
        """Whether currently navigating history entries."""
        return self._current_index >= 0

    @property
    def current_mode(self) -> str | None:
        """Submission mode for the value most recently returned by navigation.

        Covers the restored draft as well as stored entries, so a caller can
        read it directly after either `get_previous` or `get_next`.
        """
        return self._navigation_mode

    def reset_navigation(self) -> None:
        """Reset navigation state."""
        self._current_index = -1
        self._temp_input = ""
        self._temp_mode = None
        self._navigation_mode = None
        self._query = ""
