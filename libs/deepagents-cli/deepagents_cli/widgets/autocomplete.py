"""Autocomplete widgets for @ mentions and / commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual_autocomplete import AutoComplete, DropdownItem, TargetState

if TYPE_CHECKING:
    from textual.widgets import Input


# Built-in slash commands (prefix shows description)
SLASH_COMMANDS = [
    DropdownItem(main="/help", prefix="Show help"),
    DropdownItem(main="/clear", prefix="Clear chat"),
    DropdownItem(main="/quit", prefix="Exit app"),
    DropdownItem(main="/exit", prefix="Exit app"),
    DropdownItem(main="/tokens", prefix="Token usage"),
]


class TriggerAutoComplete(AutoComplete):
    """Autocomplete that triggers on @ or / characters.

    This subclass of AutoComplete provides:
    - / commands: Shows available slash commands
    - @ mentions: Shows files in the current directory for context

    The key pattern is overriding get_search_string() to only return
    text after the trigger character, allowing normal typing without
    autocomplete interference.
    """

    def __init__(
        self,
        target: Input,
        *,
        cwd: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the autocomplete.

        Args:
            target: The Input widget to attach to
            cwd: Current working directory for file suggestions
            **kwargs: Additional args passed to AutoComplete
        """
        self._cwd = Path(cwd) if cwd else Path.cwd()
        super().__init__(target, candidates=self._get_candidates, **kwargs)

    def _get_candidates(self, state: TargetState) -> list[DropdownItem]:
        """Get autocomplete candidates based on current input.

        This is called each time the input changes. We check if the user
        is typing a / command or @ mention and return appropriate suggestions.
        """
        text = state.text
        cursor = state.cursor

        # Get text up to cursor
        text_before_cursor = text[:cursor]

        # Check for / command trigger at start of input
        if text_before_cursor.startswith("/"):
            search = text_before_cursor[1:].lower()  # Text after /
            return [
                item
                for item in SLASH_COMMANDS
                if search in item.main.lower() or search in (str(item.prefix) or "").lower()
            ]

        # Check for @ mention trigger
        # Find the last @ before cursor, ensure it's at start or after whitespace
        at_pos = text_before_cursor.rfind("@")
        if at_pos >= 0 and (at_pos == 0 or text_before_cursor[at_pos - 1].isspace()):
            search = text_before_cursor[at_pos + 1 :].lower()
            return self._get_file_candidates(search)

        return []

    def _get_file_candidates(self, search: str) -> list[DropdownItem]:
        """Get file candidates for @ mentions.

        Args:
            search: The search string after @

        Returns:
            List of file/directory suggestions
        """
        candidates = []

        try:
            # List files in cwd
            for item in self._cwd.iterdir():
                name = item.name

                # Skip hidden files unless searching for them
                if name.startswith(".") and not search.startswith("."):
                    continue

                # Filter by search string
                if search and search.lower() not in name.lower():
                    continue

                # Determine prefix (type indicator) and display name
                if item.is_dir():
                    prefix = "dir"
                    display = f"@{name}/"
                else:
                    ext = item.suffix.lower()
                    prefix = ext[1:] if ext else "file"
                    display = f"@{name}"

                candidates.append(
                    DropdownItem(
                        main=display,
                        prefix=prefix,
                    )
                )

            # Sort: directories first, then files, alphabetically
            candidates.sort(key=lambda x: (not x.main.endswith("/"), x.main.lower()))

        except OSError:
            pass

        return candidates[:20]  # Limit to 20 suggestions

    def get_search_string(self, state: TargetState) -> str:
        """Get the string to use for filtering candidates.

        For / commands: return text after /
        For @ mentions: return text after the last @

        This is the key method that makes trigger-based autocomplete work.
        """
        text = state.text
        cursor = state.cursor
        text_before_cursor = text[:cursor]

        # / command at start
        if text_before_cursor.startswith("/"):
            return text_before_cursor[1:]

        # @ mention - find last @
        at_pos = text_before_cursor.rfind("@")
        if at_pos >= 0 and (at_pos == 0 or text_before_cursor[at_pos - 1].isspace()):
            return text_before_cursor[at_pos + 1 :]

        return ""

    def apply_completion(self, value: str) -> None:
        """Apply the selected completion to the input.

        Args:
            value: The selected completion value (e.g., "/help" or "@file.py")
        """
        if not self.target:
            return

        text = self.target.value
        cursor = self.target.cursor_position

        # For / commands at start, replace entire input
        if text.startswith("/"):
            self.target.value = value + " "
            self.target.cursor_position = len(self.target.value)
            return

        # For @ mentions, find and replace the @mention being typed
        text_before_cursor = text[:cursor]
        at_pos = text_before_cursor.rfind("@")

        if at_pos >= 0:
            # Replace from @ to cursor with the completion
            before = text[:at_pos]
            after = text[cursor:]
            self.target.value = before + value + " " + after
            self.target.cursor_position = at_pos + len(value) + 1
