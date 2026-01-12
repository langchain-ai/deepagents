"""Autocomplete system for @ mentions and / commands.

This is a custom implementation that handles trigger-based completion
for slash commands (/) and file mentions (@).
"""

from __future__ import annotations

import subprocess
from difflib import SequenceMatcher
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

SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show help"),
    ("/clear", "Clear chat and start new session"),
    ("/quit", "Exit app"),
    ("/exit", "Exit app"),
    ("/tokens", "Token usage"),
    ("/threads", "Show session info"),
    ("/version", "Show version"),
]
"""Built-in slash commands with descriptions."""

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
# Fuzzy File Completion (from project root)
# ============================================================================

# Constants for fuzzy file completion
_MAX_FALLBACK_FILES = 1000
_MIN_FUZZY_RATIO = 0.4
_MIN_FUZZY_SCORE = 15  # Minimum score to include in results


def _find_project_root(start_path: Path) -> Path:
    """Find git root or return start_path."""
    current = start_path.resolve()
    for parent in [current, *list(current.parents)]:
        if (parent / ".git").exists():
            return parent
    return start_path


def _get_project_files(root: Path) -> list[str]:
    """Get project files using git ls-files or fallback to glob."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],  # noqa: S607
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            files = result.stdout.strip().split("\n")
            return [f for f in files if f]  # Filter empty strings
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Fallback: simple glob (limited depth to avoid slowness)
    files = []
    try:
        for pattern in ["*", "*/*", "*/*/*", "*/*/*/*"]:
            for p in root.glob(pattern):
                if p.is_file() and not any(part.startswith(".") for part in p.parts):
                    files.append(str(p.relative_to(root)))
                if len(files) >= _MAX_FALLBACK_FILES:
                    break
            if len(files) >= _MAX_FALLBACK_FILES:
                break
    except OSError:
        pass
    return files


def _fuzzy_score(query: str, candidate: str) -> float:  # noqa: PLR0911
    """Score a candidate against query. Higher = better match."""
    query_lower = query.lower()
    candidate_lower = candidate.lower()

    # Extract filename for matching (prioritize filename over full path)
    filename = candidate.rsplit("/", 1)[-1].lower()
    filename_start = candidate_lower.rfind("/") + 1

    # Check filename first (higher priority)
    if query_lower in filename:
        idx = filename.find(query_lower)
        # Bonus for being at start of filename
        if idx == 0:
            return 150 + (1 / len(candidate))
        # Bonus for word boundary in filename
        if idx > 0 and filename[idx - 1] in "_-.":
            return 120 + (1 / len(candidate))
        return 100 + (1 / len(candidate))

    # Check full path
    if query_lower in candidate_lower:
        idx = candidate_lower.find(query_lower)
        # At start of filename
        if idx == filename_start:
            return 80 + (1 / len(candidate))
        # At word boundary in path
        if idx == 0 or candidate[idx - 1] in "/_-.":
            return 60 + (1 / len(candidate))
        return 40 + (1 / len(candidate))

    # Fuzzy match on filename only (more relevant)
    filename_ratio = SequenceMatcher(None, query_lower, filename).ratio()
    if filename_ratio > _MIN_FUZZY_RATIO:
        return filename_ratio * 30

    # Fallback: fuzzy on full path
    ratio = SequenceMatcher(None, query_lower, candidate_lower).ratio()
    return ratio * 15


def _is_dotpath(path: str) -> bool:
    """Check if path contains dotfiles/dotdirs (e.g., .github/...)."""
    return any(part.startswith(".") for part in path.split("/"))


def _path_depth(path: str) -> int:
    """Get depth of path (number of / separators)."""
    return path.count("/")


def _fuzzy_search(
    query: str, candidates: list[str], limit: int = 10, *, include_dotfiles: bool = False
) -> list[str]:
    """Return top matches sorted by score.

    Args:
        query: Search query
        candidates: List of file paths to search
        limit: Max results to return
        include_dotfiles: Whether to include dotfiles (default False)
    """
    # Filter dotfiles unless explicitly searching for them
    filtered = candidates if include_dotfiles else [c for c in candidates if not _is_dotpath(c)]

    if not query:
        # Empty query: show root-level files first, sorted by depth then name
        sorted_files = sorted(filtered, key=lambda p: (_path_depth(p), p.lower()))
        return sorted_files[:limit]

    scored = [(score, c) for c in filtered if (score := _fuzzy_score(query, c)) >= _MIN_FUZZY_SCORE]
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:limit]]


class FuzzyFileController:
    """Controller for @ file completion with fuzzy matching from project root."""

    def __init__(
        self,
        view: CompletionView,
        cwd: Path | None = None,
    ) -> None:
        """Initialize the fuzzy file controller.

        Args:
            view: View to render suggestions to
            cwd: Starting directory to find project root from
        """
        self._view = view
        self._cwd = cwd or Path.cwd()
        self._project_root = _find_project_root(self._cwd)
        self._suggestions: list[tuple[str, str]] = []
        self._selected_index = 0
        self._file_cache: list[str] | None = None

    def _get_files(self) -> list[str]:
        """Get cached file list or refresh."""
        if self._file_cache is None:
            self._file_cache = _get_project_files(self._project_root)
        return self._file_cache

    def refresh_cache(self) -> None:
        """Force refresh of file cache."""
        self._file_cache = None

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

        suggestions = self._get_fuzzy_suggestions(search)

        if suggestions:
            self._suggestions = suggestions
            self._selected_index = 0
            self._view.render_completion_suggestions(self._suggestions, self._selected_index)
        else:
            self.reset()

    def _get_fuzzy_suggestions(self, search: str) -> list[tuple[str, str]]:
        """Get fuzzy file suggestions."""
        files = self._get_files()
        # Include dotfiles only if query starts with "."
        include_dots = search.startswith(".")
        matches = _fuzzy_search(search, files, limit=MAX_SUGGESTIONS, include_dotfiles=include_dots)

        suggestions: list[tuple[str, str]] = []
        for path in matches:
            # Get file extension for type hint
            ext = Path(path).suffix.lower()
            type_hint = ext[1:] if ext else "file"
            suggestions.append((f"@{path}", type_hint))

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


# Keep old name as alias for backwards compatibility
PathCompletionController = FuzzyFileController


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
