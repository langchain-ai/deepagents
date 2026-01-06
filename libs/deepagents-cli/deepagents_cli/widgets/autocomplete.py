"""Autocomplete system for @ mentions and / commands.

This is a custom implementation that handles trigger-based completion
for slash commands (/) and file mentions (@).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from textual import events


class CompletionResult(StrEnum):
    """Result of handling a key event in the completion system."""

    IGNORED = "ignored"  # Key not handled, let default behavior proceed
    HANDLED = "handled"  # Key handled, prevent default
    SUBMIT = "submit"  # Key triggers submission (e.g., Enter on slash command)


class CompletionView(Protocol):
    """Protocol for views that can display completion suggestions."""

    def render_completion_suggestions(
        self, suggestions: list[tuple[str, str]], selected_index: int
    ) -> None:
        """Render the completion suggestions popup.

        Args:
            suggestions: List of (label, description) tuples
            selected_index: Index of currently selected item
        """
        ...

    def clear_completion_suggestions(self) -> None:
        """Hide/clear the completion suggestions popup."""
        ...

    def replace_completion_range(self, start: int, end: int, replacement: str) -> None:
        """Replace text in the input from start to end with replacement.

        Args:
            start: Start index in the input text
            end: End index in the input text
            replacement: Text to insert
        """
        ...


class CompletionController(Protocol):
    """Protocol for completion controllers."""

    def can_handle(self, text: str, cursor_index: int) -> bool:
        """Check if this controller can handle the current input state."""
        ...

    def on_text_changed(self, text: str, cursor_index: int) -> None:
        """Called when input text changes."""
        ...

    def on_key(self, event: events.Key, text: str, cursor_index: int) -> CompletionResult:
        """Handle a key event. Returns how the event was handled."""
        ...

    def reset(self) -> None:
        """Reset/clear the completion state."""
        ...


# ============================================================================
# Slash Command Completion
# ============================================================================

# Built-in slash commands with descriptions
SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show help"),
    ("/clear", "Clear chat"),
    ("/quit", "Exit app"),
    ("/exit", "Exit app"),
    ("/tokens", "Token usage"),
]

MAX_SUGGESTIONS = 10


class SlashCommandController:
    """Controller for / slash command completion."""

    def __init__(
        self,
        commands: list[tuple[str, str]],
        view: CompletionView,
    ) -> None:
        """Initialize the slash command controller.

        Args:
            commands: List of (command, description) tuples
            view: View to render suggestions to
        """
        self._commands = commands
        self._view = view
        self._suggestions: list[tuple[str, str]] = []
        self._selected_index = 0

    def can_handle(self, text: str, cursor_index: int) -> bool:  # noqa: ARG002
        """Handle input that starts with /."""
        return text.startswith("/")

    def reset(self) -> None:
        """Clear suggestions."""
        if self._suggestions:
            self._suggestions.clear()
            self._selected_index = 0
            self._view.clear_completion_suggestions()

    def on_text_changed(self, text: str, cursor_index: int) -> None:
        """Update suggestions when text changes."""
        if cursor_index < 0 or cursor_index > len(text):
            self.reset()
            return

        if not self.can_handle(text, cursor_index):
            self.reset()
            return

        # Get the search string (text after /)
        search = text[1:cursor_index].lower()

        # Filter commands that match
        suggestions = [
            (cmd, desc) for cmd, desc in self._commands if cmd.lower().startswith("/" + search)
        ]

        if len(suggestions) > MAX_SUGGESTIONS:
            suggestions = suggestions[:MAX_SUGGESTIONS]

        if suggestions:
            self._suggestions = suggestions
            self._selected_index = 0
            self._view.render_completion_suggestions(self._suggestions, self._selected_index)
        else:
            self.reset()

    def on_key(  # noqa: PLR0911
        self, event: events.Key, _text: str, cursor_index: int
    ) -> CompletionResult:
        """Handle key events for navigation and selection."""
        if not self._suggestions:
            return CompletionResult.IGNORED

        match event.key:
            case "tab":
                if self._apply_selected_completion(cursor_index):
                    return CompletionResult.HANDLED
                return CompletionResult.IGNORED
            case "enter":
                if self._apply_selected_completion(cursor_index):
                    return CompletionResult.SUBMIT
                return CompletionResult.HANDLED
            case "down":
                self._move_selection(1)
                return CompletionResult.HANDLED
            case "up":
                self._move_selection(-1)
                return CompletionResult.HANDLED
            case "escape":
                self.reset()
                return CompletionResult.HANDLED
            case _:
                return CompletionResult.IGNORED

    def _move_selection(self, delta: int) -> None:
        """Move selection up or down."""
        if not self._suggestions:
            return
        count = len(self._suggestions)
        self._selected_index = (self._selected_index + delta) % count
        self._view.render_completion_suggestions(self._suggestions, self._selected_index)

    def _apply_selected_completion(self, cursor_index: int) -> bool:
        """Apply the currently selected completion."""
        if not self._suggestions:
            return False

        command, _ = self._suggestions[self._selected_index]
        # Replace from start to cursor with the command
        self._view.replace_completion_range(0, cursor_index, command)
        self.reset()
        return True


# ============================================================================
# Path/File Completion
# ============================================================================


class PathCompletionController:
    """Controller for @ file path completion."""

    def __init__(
        self,
        view: CompletionView,
        cwd: Path | None = None,
    ) -> None:
        """Initialize the path completion controller.

        Args:
            view: View to render suggestions to
            cwd: Current working directory for file listing
        """
        self._view = view
        self._cwd = cwd or Path.cwd()
        self._suggestions: list[tuple[str, str]] = []
        self._selected_index = 0

    def can_handle(self, text: str, cursor_index: int) -> bool:
        """Handle input that contains @ not followed by space."""
        if cursor_index <= 0 or cursor_index > len(text):
            return False

        before_cursor = text[:cursor_index]
        if "@" not in before_cursor:
            return False

        at_index = before_cursor.rfind("@")
        if cursor_index <= at_index:
            return False

        # Fragment from @ to cursor must not contain spaces
        fragment = before_cursor[at_index:cursor_index]
        return bool(fragment) and " " not in fragment

    def reset(self) -> None:
        """Clear suggestions."""
        if self._suggestions:
            self._suggestions.clear()
            self._selected_index = 0
            self._view.clear_completion_suggestions()

    def on_text_changed(self, text: str, cursor_index: int) -> None:
        """Update suggestions when text changes."""
        if not self.can_handle(text, cursor_index):
            self.reset()
            return

        before_cursor = text[:cursor_index]
        at_index = before_cursor.rfind("@")
        search = before_cursor[at_index + 1 :]

        suggestions = self._get_file_suggestions(search)

        if suggestions:
            self._suggestions = suggestions
            self._selected_index = 0
            self._view.render_completion_suggestions(self._suggestions, self._selected_index)
        else:
            self.reset()

    def _get_file_suggestions(self, search: str) -> list[tuple[str, str]]:
        """Get file/directory suggestions based on search string."""
        suggestions: list[tuple[str, str]] = []

        try:
            # Determine base path and search pattern
            if "/" in search:
                # User is navigating into a subdirectory
                parts = search.rsplit("/", 1)
                base_path = self._cwd / parts[0]
                pattern = parts[1].lower()
            else:
                base_path = self._cwd
                pattern = search.lower()

            if not base_path.exists() or not base_path.is_dir():
                return []

            for item in sorted(base_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                name = item.name

                # Skip hidden files unless searching for them
                if name.startswith(".") and not pattern.startswith("."):
                    continue

                # Filter by pattern
                if pattern and pattern not in name.lower():
                    continue

                # Build the full relative path from search context
                rel_path = search.rsplit("/", 1)[0] + "/" + name if "/" in search else name

                # Determine type indicator
                if item.is_dir():
                    label = f"@{rel_path}/"
                    type_hint = "dir"
                else:
                    label = f"@{rel_path}"
                    ext = item.suffix.lower()
                    type_hint = ext[1:] if ext else "file"

                suggestions.append((label, type_hint))

                if len(suggestions) >= MAX_SUGGESTIONS:
                    break

        except OSError:
            pass

        return suggestions

    def on_key(  # noqa: PLR0911
        self, event: events.Key, text: str, cursor_index: int
    ) -> CompletionResult:
        """Handle key events for navigation and selection."""
        if not self._suggestions:
            return CompletionResult.IGNORED

        match event.key:
            case "tab" | "enter":
                if self._apply_selected_completion(text, cursor_index):
                    return CompletionResult.HANDLED
                return CompletionResult.IGNORED
            case "down":
                self._move_selection(1)
                return CompletionResult.HANDLED
            case "up":
                self._move_selection(-1)
                return CompletionResult.HANDLED
            case "escape":
                self.reset()
                return CompletionResult.HANDLED
            case _:
                return CompletionResult.IGNORED

    def _move_selection(self, delta: int) -> None:
        """Move selection up or down."""
        if not self._suggestions:
            return
        count = len(self._suggestions)
        self._selected_index = (self._selected_index + delta) % count
        self._view.render_completion_suggestions(self._suggestions, self._selected_index)

    def _apply_selected_completion(self, text: str, cursor_index: int) -> bool:
        """Apply the currently selected completion."""
        if not self._suggestions:
            return False

        label, _ = self._suggestions[self._selected_index]
        before_cursor = text[:cursor_index]
        at_index = before_cursor.rfind("@")

        if at_index < 0:
            return False

        # Replace from @ to cursor with the completion
        self._view.replace_completion_range(at_index, cursor_index, label)
        self.reset()
        return True


# ============================================================================
# Multi-Completion Manager
# ============================================================================


class MultiCompletionManager:
    """Manages multiple completion controllers, delegating to the active one."""

    def __init__(self, controllers: list[CompletionController]) -> None:
        """Initialize with a list of controllers.

        Args:
            controllers: List of completion controllers (checked in order)
        """
        self._controllers = controllers
        self._active: CompletionController | None = None

    def on_text_changed(self, text: str, cursor_index: int) -> None:
        """Handle text change, activating the appropriate controller."""
        # Find the first controller that can handle this input
        candidate = None
        for controller in self._controllers:
            if controller.can_handle(text, cursor_index):
                candidate = controller
                break

        # No controller can handle - reset if we had one active
        if candidate is None:
            if self._active is not None:
                self._active.reset()
                self._active = None
            return

        # Switch to new controller if different
        if candidate is not self._active:
            if self._active is not None:
                self._active.reset()
            self._active = candidate

        # Let the active controller process the change
        candidate.on_text_changed(text, cursor_index)

    def on_key(self, event: events.Key, text: str, cursor_index: int) -> CompletionResult:
        """Handle key event, delegating to active controller."""
        if self._active is None:
            return CompletionResult.IGNORED
        return self._active.on_key(event, text, cursor_index)

    def reset(self) -> None:
        """Reset all controllers."""
        if self._active is not None:
            self._active.reset()
            self._active = None
