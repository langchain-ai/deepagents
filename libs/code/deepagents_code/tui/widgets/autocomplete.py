"""Autocomplete system for @ mentions and / commands.

This is a custom implementation that handles trigger-based completion
for slash commands (/) and file mentions (@).
"""

from __future__ import annotations

import asyncio
import logging
import shutil

# S404: subprocess is required for git ls-files to get project file list
import subprocess  # noqa: S404
from difflib import SequenceMatcher
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from deepagents_code.project_utils import find_project_root

logger = logging.getLogger(__name__)


def _get_git_executable() -> str | None:
    """Get full path to git executable using shutil.which().

    Returns:
        Full path to git executable, or None if not found.
    """
    return shutil.which("git")


if TYPE_CHECKING:
    from textual import events

    from deepagents_code.command_registry import CommandEntry


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

    def on_key(
        self, event: events.Key, text: str, cursor_index: int
    ) -> CompletionResult:
        """Handle a key event. Returns how the event was handled."""
        ...

    def reset(self) -> None:
        """Reset/clear the completion state."""
        ...


# ============================================================================
# Slash Command Completion
# ============================================================================


MAX_SUGGESTIONS = 10
"""UI cap so the completion popup doesn't get unwieldy."""

_MIN_SLASH_FUZZY_SCORE = 25
"""Minimum score for slash-command fuzzy matches."""

_MIN_DESC_SEARCH_LEN = 2
"""Minimum query length to search command descriptions (avoids single-char noise)."""


class SlashCommandController:
    """Controller for / slash command completion."""

    def __init__(
        self,
        commands: list[CommandEntry],
        view: CompletionView,
    ) -> None:
        """Initialize the slash command controller.

        Args:
            commands: List of `CommandEntry` instances.
            view: View to render suggestions to.
        """
        self._commands = commands
        self._view = view
        self._suggestions: list[tuple[str, str]] = []
        self._selected_index = 0

    def update_commands(self, commands: list[CommandEntry]) -> None:
        """Replace the commands list and reset suggestions.

        Used to merge dynamically discovered skill commands with
        the static command registry at runtime.

        Args:
            commands: New list of `CommandEntry` instances.
        """
        self._commands = commands
        self.reset()

    @staticmethod
    def can_handle(text: str, cursor_index: int) -> bool:  # noqa: ARG004  # Required by AutocompleteProvider interface
        """Handle input that starts with /.

        Returns:
            True if text starts with slash, indicating a command.
        """
        return text.startswith("/")

    def reset(self) -> None:
        """Clear suggestions."""
        if self._suggestions:
            self._suggestions.clear()
            self._selected_index = 0
            self._view.clear_completion_suggestions()

    def name_prefix_matches(self, text: str, cursor_index: int) -> list[CommandEntry]:
        """Return commands whose names start with the current slash query."""
        if cursor_index < 0 or cursor_index > len(text):
            return []
        if not self.can_handle(text, cursor_index):
            return []

        search = text[1:cursor_index].lower()
        if not search or " " in search:
            return []

        return [
            entry
            for entry in self._commands
            if entry.name.lstrip("/").lower().startswith(search)
        ]

    @staticmethod
    def _score_command(search: str, cmd: str, desc: str, keywords: str = "") -> float:
        """Score a command against a search string. Higher = better match.

        Args:
            search: Lowercase search string (without leading `/`).
            cmd: Command name (e.g. `'/help'`).
            desc: Command description text.
            keywords: Space-separated hidden keywords for matching.

        Returns:
            Score value where higher indicates better match quality.
        """
        if not search:
            return 0.0
        name = cmd.lstrip("/").lower()
        lower_desc = desc.lower()
        # Prefix match on command name — highest priority
        if name.startswith(search):
            return 200.0
        # Substring match on command name
        if search in name:
            return 150.0
        # Hidden keyword match — treated like a word-boundary description match
        if keywords and len(search) >= _MIN_DESC_SEARCH_LEN:
            for kw in keywords.lower().split():
                if kw.startswith(search) or search in kw:
                    return 120.0
        # Substring match on description (require ≥2 chars to avoid single-letter noise)
        if len(search) >= _MIN_DESC_SEARCH_LEN and search in lower_desc:
            idx = lower_desc.find(search)
            # Word-boundary bonus: match at start of description or after a space
            if idx == 0 or lower_desc[idx - 1] == " ":
                return 110.0
            return 90.0
        # Fuzzy match via SequenceMatcher on name + desc
        name_ratio = SequenceMatcher(None, search, name).ratio()
        desc_ratio = SequenceMatcher(None, search, lower_desc).ratio()
        best = max(name_ratio * 60, desc_ratio * 30)
        return best if best >= _MIN_SLASH_FUZZY_SCORE else 0.0

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

        # Space means the user finished picking a command — dismiss popup
        if " " in search:
            self.reset()
            return

        if not search:
            # No search text — show all commands (display only cmd + desc)
            suggestions = [(entry.name, entry.description) for entry in self._commands][
                :MAX_SUGGESTIONS
            ]
        else:
            # Score and filter commands using fuzzy matching
            scored = [
                (score, entry.name, entry.description)
                for entry in self._commands
                if (
                    score := self._score_command(
                        search, entry.name, entry.description, entry.hidden_keywords
                    )
                )
                > 0
            ]
            scored.sort(key=lambda x: -x[0])
            suggestions = [(cmd, desc) for _, cmd, desc in scored[:MAX_SUGGESTIONS]]

        if suggestions:
            self._suggestions = suggestions
            self._selected_index = 0
            self._view.render_completion_suggestions(
                self._suggestions, self._selected_index
            )
        else:
            self.reset()

    def on_key(
        self, event: events.Key, _text: str, cursor_index: int
    ) -> CompletionResult:
        """Handle key events for navigation and selection.

        Returns:
            CompletionResult indicating how the key was handled.
        """
        if not self._suggestions:
            return CompletionResult.IGNORED

        match event.key:
            case "tab" | "space":
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
        self._view.render_completion_suggestions(
            self._suggestions, self._selected_index
        )

    def _apply_selected_completion(self, cursor_index: int) -> bool:
        """Apply the currently selected completion.

        Returns:
            True if completion was applied, False if no suggestions.
        """
        if not self._suggestions:
            return False

        command, _ = self._suggestions[self._selected_index]
        # Replace from start to cursor with the command
        self._view.replace_completion_range(0, cursor_index, command)
        self.reset()
        return True

    def apply_name_prefix_completion(
        self, match: CommandEntry, cursor_index: int
    ) -> None:
        """Apply a command-name prefix match.

        Args:
            match: Command entry to apply.
            cursor_index: Cursor index in completion-space coordinates.
        """
        self._view.replace_completion_range(0, cursor_index, match.name)
        self.reset()


# ============================================================================
# Fuzzy File Completion (scoped to current working directory)
# ============================================================================

# Constants for fuzzy file completion
_MAX_FALLBACK_FILES = 1000
"""Hard cap on files returned by the non-git glob fallback."""

_MIN_FUZZY_SCORE = 15
"""Minimum score to include in file-completion results."""

_MIN_FUZZY_RATIO = 0.4
"""SequenceMatcher threshold for filename-only fuzzy matches."""


def _run_git_ls_files(
    git_path: str, root: Path, extra_args: list[str]
) -> tuple[bool, list[str]]:
    """Run `git ls-files` with the given arguments and return file paths.

    Args:
        git_path: Full path to the git executable.
        root: Directory to run the command in.
        extra_args: Flags appended after `ls-files`, e.g.
            `["--others", "--exclude-standard"]`.

    Returns:
        Tuple of success status and relative file paths. Success is `False`
            when git could not be run or exited non-zero, signalling the caller
            to fall back to a glob walk.
    """
    try:
        # S603: git_path validated via shutil.which(); ls-files args are
        # caller-supplied literals.
        result = subprocess.run(  # noqa: S603
            [git_path, "ls-files", *extra_args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        logger.debug("git ls-files %s failed to run", extra_args, exc_info=True)
        return False, []
    if result.returncode != 0:
        logger.debug("git ls-files %s exited with %d", extra_args, result.returncode)
        return False, []
    return True, [f for f in result.stdout.strip().split("\n") if f]


def _get_project_files(root: Path) -> list[str]:
    """Get project files using git ls-files or fallback to glob.

    Includes both tracked files and untracked files that are not ignored
    (via `--others --exclude-standard`), so freshly created files surface
    in `@` completion without needing to be committed first.

    Returns:
        List of relative file paths from project root.
    """
    git_path = _get_git_executable()
    if git_path:
        tracked_ok, tracked = _run_git_ls_files(git_path, root, [])
        if tracked_ok:
            # The untracked scan is optional; if it fails or times out, keep the
            # already-successful tracked list rather than dropping to the glob
            # fallback (which only walks a few levels deep).
            _, untracked = _run_git_ls_files(
                git_path, root, ["--others", "--exclude-standard"]
            )
            seen: set[str] = set()
            files: list[str] = []
            for f in (*tracked, *untracked):
                if f not in seen:
                    seen.add(f)
                    files.append(f)
            return files

    # Fallback: simple glob (limited depth to avoid slowness)
    files = []
    try:
        for pattern in ["*", "*/*", "*/*/*", "*/*/*/*"]:
            for p in root.glob(pattern):
                if p.is_file() and not any(part.startswith(".") for part in p.parts):
                    files.append(p.relative_to(root).as_posix())
                if len(files) >= _MAX_FALLBACK_FILES:
                    break
            if len(files) >= _MAX_FALLBACK_FILES:
                break
    except OSError:
        logger.debug("glob fallback failed for %s", root, exc_info=True)
    return files


def _fuzzy_score(query: str, candidate: str) -> float:
    """Score a candidate against query. Higher = better match.

    Returns:
        Score value where higher indicates better match quality.
    """
    query_lower = query.lower()
    # Normalize path separators for cross-platform support
    candidate_normalized = candidate.replace("\\", "/")
    candidate_lower = candidate_normalized.lower()

    # Extract filename for matching (prioritize filename over full path)
    filename = candidate_normalized.rsplit("/", 1)[-1].lower()
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
    """Check if path contains dotfiles/dotdirs (e.g., .github/...).

    Returns:
        True if path contains hidden directories or files.
    """
    return any(part.startswith(".") for part in path.split("/"))


def _path_depth(path: str) -> int:
    """Get depth of path (number of / separators).

    Returns:
        Number of path separators in the path.
    """
    return path.count("/")


def _fuzzy_search(
    query: str,
    candidates: list[str],
    limit: int = 10,
    *,
    include_dotfiles: bool = False,
) -> list[str]:
    """Return top matches sorted by score.

    Args:
        query: Search query
        candidates: List of file paths to search
        limit: Max results to return
        include_dotfiles: Whether to include dotfiles (default False)

    Returns:
        List of matching file paths sorted by relevance score.
    """
    # Filter dotfiles unless explicitly searching for them
    filtered = (
        candidates
        if include_dotfiles
        else [c for c in candidates if not _is_dotpath(c)]
    )

    if not query:
        # Empty query: show root-level files first, sorted by depth then name
        sorted_files = sorted(filtered, key=lambda p: (_path_depth(p), p.lower()))
        return sorted_files[:limit]

    scored = [
        (score, c)
        for c in filtered
        if (score := _fuzzy_score(query, c)) >= _MIN_FUZZY_SCORE
    ]
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:limit]]


def _scope_files_to_cwd(files: list[str], project_root: Path, cwd: Path) -> list[str]:
    """Scope a project-root-relative file list to paths under `cwd`.

    Args:
        files: File paths relative to `project_root` (as produced by
            `_get_project_files`).
        project_root: Directory the `files` paths are relative to.
        cwd: Directory to scope suggestions to.

    Returns:
        Paths rewritten relative to `cwd`, filtered to that subtree (possibly
        empty), when `cwd` is nested under `project_root`. The input list
        unchanged when `cwd` equals `project_root`. An empty list when `cwd` is
        not under `project_root`: the paths are project-root-relative and would
        resolve to the wrong base from `cwd`, so fail closed rather than offer
        misleading suggestions.
    """
    if cwd == project_root:
        return files
    try:
        relative_cwd = cwd.relative_to(project_root).as_posix()
    except ValueError:
        return []
    prefix = f"{relative_cwd}/"
    return [path[len(prefix) :] for path in files if path.startswith(prefix)]


class FuzzyFileController:
    """Controller for @ file completion with fuzzy matching from current cwd."""

    def __init__(
        self,
        view: CompletionView,
        cwd: Path | None = None,
    ) -> None:
        """Initialize the fuzzy file controller.

        Args:
            view: View to render suggestions to
            cwd: Current working directory for file completion scope
        """
        self._view = view
        self._cwd = (cwd or Path.cwd()).resolve()
        self._project_root = find_project_root(self._cwd) or self._cwd
        self._suggestions: list[tuple[str, str]] = []
        self._selected_index = 0
        self._file_cache: list[str] | None = None
        # When True, `_project_root` is a provisional value (set synchronously by
        # `set_cwd`) and the real project root is resolved off the event loop in
        # `warm_cache`. See `set_cwd` for why discovery is deferred.
        self._project_root_pending = False
        self._cache_generation = 0

    def _get_files(self) -> list[str]:
        """Get cached file list or refresh.

        Returns:
            List of project file paths.
        """
        if self._file_cache is None:
            files = _get_project_files(self._project_root)
            self._file_cache = _scope_files_to_cwd(files, self._project_root, self._cwd)
        return self._file_cache

    def refresh_cache(self) -> None:
        """Force refresh of file cache."""
        self._cache_generation += 1
        self._file_cache = None

    def set_cwd(self, cwd: Path) -> None:
        """Switch completion roots to a new cwd.

        Roots completion at `cwd` immediately and invalidates the file cache.
        Project-root discovery (`find_project_root`) walks the filesystem, so it
        is deferred to `warm_cache` (which runs in a worker thread) rather than
        run here on the event loop. Until then `cwd` is used as a provisional
        root, which is a safe narrower scope.
        """
        self._cache_generation += 1
        self._cwd = cwd.resolve()
        self._project_root = self._cwd
        self._project_root_pending = True
        self._file_cache = None
        self.reset()

    async def warm_cache(self, *, force: bool = False) -> None:
        """Pre-populate the file cache off the event loop.

        Also resolves a project root deferred by `set_cwd`, so the blocking
        filesystem walk runs in a worker thread instead of on the event loop.

        Warmers are scheduled non-exclusively, so quick cwd/cache invalidations
        can run concurrently. The generation is snapshotted once before the first
        await and re-checked after each await, so a warmer whose generation has
        been superseded drops its results instead of overwriting controller state
        belonging to a newer generation. (Snapshotting again before the second
        await would defeat the guard: it would match the post-supersession
        generation and let a stale-root file walk win.)

        Args:
            force: Re-walk and swap in a fresh file list even when the cache is
                already populated. Used by the periodic background refresh so
                files created or deleted mid-session surface in `@` completion.
                The existing cache stays visible until the new walk completes.
        """
        cwd = self._cwd
        generation = self._cache_generation
        if self._project_root_pending:
            root = await asyncio.to_thread(find_project_root, cwd)
            if generation != self._cache_generation:
                # A newer cwd/cache invalidation superseded this warmer.
                return
            resolved = root or cwd
            if resolved != self._project_root:
                # The real root differs from the provisional `cwd`; drop any
                # cache built against the narrower scope.
                self._file_cache = None
            self._project_root = resolved
            self._project_root_pending = False
        if not force and self._file_cache is not None:
            return
        project_root = self._project_root
        # Best-effort: on failure the existing cache (if any) stays in place. A
        # cold cache (`_file_cache is None`) is later filled synchronously by
        # `_get_files()`; a force refresh that fails simply leaves the prior
        # list visible. Log at debug so a recurring background refresh failure
        # (the 30s timer) is diagnosable rather than silently stale.
        try:
            files = await asyncio.to_thread(_get_project_files, project_root)
            if generation == self._cache_generation:
                self._file_cache = _scope_files_to_cwd(files, project_root, cwd)
        except Exception:  # best-effort refresh; prior cache is the fallback
            logger.debug("File-cache warm failed for %s", project_root, exc_info=True)

    @staticmethod
    def can_handle(text: str, cursor_index: int) -> bool:
        """Handle input that contains @ not followed by space.

        Returns:
            True if cursor is after @ and within a file mention context.
        """
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
            self._view.render_completion_suggestions(
                self._suggestions, self._selected_index
            )
        else:
            self.reset()

    def _get_fuzzy_suggestions(self, search: str) -> list[tuple[str, str]]:
        """Get fuzzy file suggestions.

        Returns:
            List of (label, type_hint) tuples for matching files.
        """
        files = self._get_files()
        # Include dotfiles only if query starts with "."
        include_dots = search.startswith(".")
        matches = _fuzzy_search(
            search, files, limit=MAX_SUGGESTIONS, include_dotfiles=include_dots
        )

        suggestions: list[tuple[str, str]] = []
        for path in matches:
            # Get file extension for type hint
            ext = Path(path).suffix.lower()
            type_hint = ext[1:] if ext else "file"
            suggestions.append((f"@{path}", type_hint))

        return suggestions

    def on_key(
        self, event: events.Key, text: str, cursor_index: int
    ) -> CompletionResult:
        """Handle key events for navigation and selection.

        Returns:
            CompletionResult indicating how the key was handled.
        """
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
        self._view.render_completion_suggestions(
            self._suggestions, self._selected_index
        )

    def _apply_selected_completion(self, text: str, cursor_index: int) -> bool:
        """Apply the currently selected completion.

        Returns:
            True if completion was applied, False if no suggestions or invalid state.
        """
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

    def on_key(
        self, event: events.Key, text: str, cursor_index: int
    ) -> CompletionResult:
        """Handle key event, delegating to active controller.

        Returns:
            CompletionResult from active controller, or IGNORED if none active.
        """
        if self._active is None:
            return CompletionResult.IGNORED
        return self._active.on_key(event, text, cursor_index)

    def reset(self) -> None:
        """Reset all controllers."""
        if self._active is not None:
            self._active.reset()
            self._active = None
