"""Command history manager for input persistence."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - used at runtime in type hints


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
        self._current_index: int = -1
        self._temp_input: str = ""
        self._load_history()

    def _load_history(self) -> None:
        """Load history from file."""
        if not self.history_file.exists():
            return

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
                self._entries = entries[-self.max_entries :]
        except (OSError, UnicodeDecodeError):
            self._entries = []

    def _append_to_file(self, text: str) -> None:
        """Append a single entry to history file (concurrent-safe)."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with self.history_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(text) + "\n")
        except OSError:
            pass

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
            pass

    def add(self, text: str) -> None:
        """Add a command to history.

        Args:
            text: The command text to add
        """
        text = text.strip()
        # Skip empty or slash commands
        if not text or text.startswith("/"):
            return

        # Skip duplicates of the last entry
        if self._entries and self._entries[-1] == text:
            return

        self._entries.append(text)

        # Append to file (fast, concurrent-safe)
        self._append_to_file(text)

        # Compact only when we have 2x max entries (rare operation)
        if len(self._entries) > self.max_entries * 2:
            self._entries = self._entries[-self.max_entries :]
            self._compact_history()

        self.reset_navigation()

    def get_previous(self, current_input: str, prefix: str = "") -> str | None:
        """Get the previous history entry.

        Args:
            current_input: Current input text (saved on first navigation)
            prefix: Optional prefix to filter entries

        Returns:
            Previous matching entry or None
        """
        if not self._entries:
            return None

        # Save current input on first navigation
        if self._current_index == -1:
            self._temp_input = current_input
            self._current_index = len(self._entries)

        # Search backwards for matching entry
        for i in range(self._current_index - 1, -1, -1):
            if self._entries[i].startswith(prefix):
                self._current_index = i
                return self._entries[i]

        return None

    def get_next(self, prefix: str = "") -> str | None:
        """Get the next history entry.

        Args:
            prefix: Optional prefix to filter entries

        Returns:
            Next matching entry, original input at end, or None
        """
        if self._current_index == -1:
            return None

        # Search forwards for matching entry
        for i in range(self._current_index + 1, len(self._entries)):
            if self._entries[i].startswith(prefix):
                self._current_index = i
                return self._entries[i]

        # Return to original input at the end
        result = self._temp_input
        self.reset_navigation()
        return result

    def reset_navigation(self) -> None:
        """Reset navigation state."""
        self._current_index = -1
        self._temp_input = ""
