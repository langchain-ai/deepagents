"""Chat input widget for deepagents-cli with autocomplete and history support."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static, TextArea
from textual.widgets.text_area import Selection

from deepagents_cli.config import (
    COLORS,
    MODE_PREFIXES,
    CharsetMode,
    _detect_charset_mode,
    get_glyphs,
)
from deepagents_cli.widgets.autocomplete import (
    SLASH_COMMANDS,
    CompletionResult,
    FuzzyFileController,
    MultiCompletionManager,
    SlashCommandController,
)
from deepagents_cli.widgets.history import HistoryManager

logger = logging.getLogger(__name__)

_PREFIX_TO_MODE: dict[str, str] = {v: k for k, v in MODE_PREFIXES.items()}
"""Reverse lookup: trigger character -> mode name."""

_IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[image \d+\]")
"""Pattern for detecting image placeholder tokens in the text area.

Used to locate tokens for atomic backspace/delete handling.
"""

if TYPE_CHECKING:
    from textual import events
    from textual.app import ComposeResult
    from textual.events import Click

    from deepagents_cli.input import ImageTracker


class CompletionOption(Static):
    """A clickable completion option in the autocomplete popup."""

    DEFAULT_CSS = """
    CompletionOption {
        height: 1;
        padding: 0 1;
    }

    CompletionOption:hover {
        background: $surface-lighten-1;
    }

    CompletionOption.completion-option-selected {
        background: $primary;
        text-style: bold;
    }

    CompletionOption.completion-option-selected:hover {
        background: $primary-lighten-1;
    }
    """

    class Clicked(Message):
        """Message sent when a completion option is clicked."""

        def __init__(self, index: int) -> None:
            """Initialize with the clicked option index."""
            super().__init__()
            self.index = index

    def __init__(
        self,
        label: str,
        description: str,
        index: int,
        is_selected: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the completion option.

        Args:
            label: The main label text (e.g., command name or file path)
            description: Secondary description text
            index: Index of this option in the suggestions list
            is_selected: Whether this option is currently selected
            **kwargs: Additional arguments for parent
        """
        super().__init__(**kwargs)
        self._label = label
        self._description = description
        self._index = index
        self._is_selected = is_selected

    def on_mount(self) -> None:
        """Set up the option display on mount."""
        self._update_display()

    def _update_display(self) -> None:
        """Update the display text and styling."""
        glyphs = get_glyphs()
        cursor = f"{glyphs.cursor} " if self._is_selected else "  "

        if self._description:
            text = f"{cursor}[bold]{self._label}[/bold]  [dim]{self._description}[/dim]"
        else:
            text = f"{cursor}[bold]{self._label}[/bold]"

        self.update(text)

        if self._is_selected:
            self.add_class("completion-option-selected")
        else:
            self.remove_class("completion-option-selected")

    def set_selected(self, *, selected: bool) -> None:
        """Update the selected state of this option."""
        if self._is_selected != selected:
            self._is_selected = selected
            self._update_display()

    def on_click(self, event: Click) -> None:
        """Handle click on this option."""
        event.stop()
        self.post_message(self.Clicked(self._index))


class CompletionPopup(VerticalScroll):
    """Popup widget that displays completion suggestions as clickable options."""

    DEFAULT_CSS = """
    CompletionPopup {
        display: none;
        height: auto;
        max-height: 12;
    }
    """

    class OptionClicked(Message):
        """Message sent when a completion option is clicked."""

        def __init__(self, index: int) -> None:
            """Initialize with the clicked option index."""
            super().__init__()
            self.index = index

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the completion popup."""
        super().__init__(**kwargs)
        self.can_focus = False
        self._options: list[CompletionOption] = []
        self._selected_index = 0

    def update_suggestions(
        self, suggestions: list[tuple[str, str]], selected_index: int
    ) -> None:
        """Update the popup with new suggestions."""
        if not suggestions:
            self.hide()
            return

        self._selected_index = selected_index
        # Store pending update and schedule async rebuild
        self._pending_suggestions = suggestions
        self._pending_selected = selected_index
        self.call_after_refresh(self._rebuild_options)
        self.show()

    async def _rebuild_options(self) -> None:
        """Rebuild option widgets from pending suggestions."""
        suggestions = getattr(self, "_pending_suggestions", [])
        selected_index = getattr(self, "_pending_selected", 0)

        if not suggestions:
            return

        # Remove existing options
        await self.remove_children()
        self._options.clear()

        # Create new options
        for idx, (label, description) in enumerate(suggestions):
            option = CompletionOption(
                label=label,
                description=description,
                index=idx,
                is_selected=(idx == selected_index),
            )
            self._options.append(option)
            await self.mount(option)

        # Scroll selected option into view
        if 0 <= selected_index < len(self._options):
            self._options[selected_index].scroll_visible()

    def update_selection(self, selected_index: int) -> None:
        """Update which option is selected without rebuilding the list."""
        if self._selected_index == selected_index:
            return

        # Deselect previous
        if 0 <= self._selected_index < len(self._options):
            self._options[self._selected_index].set_selected(selected=False)

        # Select new
        self._selected_index = selected_index
        if 0 <= selected_index < len(self._options):
            self._options[selected_index].set_selected(selected=True)
            self._options[selected_index].scroll_visible()

    def on_completion_option_clicked(self, event: CompletionOption.Clicked) -> None:
        """Handle click on a completion option."""
        event.stop()
        self.post_message(self.OptionClicked(event.index))

    def hide(self) -> None:
        """Hide the popup."""
        self._pending_suggestions = []
        self.styles.display = "none"  # type: ignore[assignment]  # Textual accepts string display values at runtime

    def show(self) -> None:
        """Show the popup."""
        self.styles.display = "block"


class ChatTextArea(TextArea):
    """TextArea subclass with custom key handling for chat input."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding(
            "shift+enter,ctrl+j,alt+enter,ctrl+enter",
            "insert_newline",
            "New Line",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+a",
            "select_all_text",
            "Select All",
            show=False,
            priority=True,
        ),
        # Mac Cmd+Z/Cmd+Shift+Z for undo/redo (in addition to Ctrl+Z/Y)
        Binding("cmd+z,super+z", "undo", "Undo", show=False, priority=True),
        Binding("cmd+shift+z,super+shift+z", "redo", "Redo", show=False, priority=True),
    ]

    _navigating_history: bool
    """Transient guard set `True` only while `ChatInput` replaces text with a
    history entry.

    Prevents `watch_text` from treating the programmatic replacement as user
    typing (which would trigger autocomplete, etc.).
    """

    _in_history: bool
    """Persistent flag that stays `True` while the user is browsing history.

    Relaxes cursor-boundary checks so Up/Down work from either end of
    the text.

    Reset to `False` when navigating past the newest entry, submitting,
    or clearing.
    """

    class Submitted(Message):
        """Message sent when text is submitted."""

        def __init__(self, value: str) -> None:
            """Initialize with submitted value."""
            self.value = value
            super().__init__()

    class HistoryPrevious(Message):
        """Request previous history entry."""

        def __init__(self, current_text: str) -> None:
            """Initialize with current text for saving."""
            self.current_text = current_text
            super().__init__()

    class HistoryNext(Message):
        """Request next history entry."""

    class PastedPaths(Message):
        """Message sent when paste payload resolves to file paths."""

        def __init__(self, raw_text: str, paths: list[Path]) -> None:
            """Initialize with raw pasted text and parsed file paths."""
            self.raw_text = raw_text
            self.paths = paths
            super().__init__()

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the chat text area."""
        # Remove placeholder if passed, TextArea doesn't support it the same way
        kwargs.pop("placeholder", None)
        super().__init__(**kwargs)
        self._navigating_history = False
        self._in_history = False
        self._completion_active = False
        self._app_has_focus = True

    def set_app_focus(self, *, has_focus: bool) -> None:
        """Set whether the app should show the cursor as active.

        When has_focus=False (e.g., agent is running), disables cursor blink
        so the cursor doesn't flash while waiting for a response.
        """
        self._app_has_focus = has_focus
        self.cursor_blink = has_focus
        if has_focus and not self.has_focus:
            self.call_after_refresh(self.focus)

    def set_completion_active(self, *, active: bool) -> None:
        """Set whether completion suggestions are visible."""
        self._completion_active = active

    def action_insert_newline(self) -> None:
        """Insert a newline character."""
        self.insert("\n")

    def action_select_all_text(self) -> None:
        """Select all text in the text area."""
        if not self.text:
            return
        # Select from start to end
        lines = self.text.split("\n")
        end_row = len(lines) - 1
        end_col = len(lines[end_row])
        self.selection = Selection(start=(0, 0), end=(end_row, end_col))

    async def _on_key(self, event: events.Key) -> None:
        """Handle key events."""
        # Modifier+Enter inserts newline (Ctrl+J is most reliable across terminals)
        if event.key in {"shift+enter", "ctrl+j", "alt+enter", "ctrl+enter"}:
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return

        if event.key == "backspace" and self._delete_image_placeholder(backwards=True):
            event.prevent_default()
            event.stop()
            return

        if event.key == "delete" and self._delete_image_placeholder(backwards=False):
            event.prevent_default()
            event.stop()
            return

        # If completion is active, let parent handle navigation keys
        if self._completion_active and event.key in {"up", "down", "tab", "enter"}:
            # Prevent TextArea's default behavior (e.g., Enter inserting newline)
            # but let event bubble to ChatInput for completion handling
            event.prevent_default()
            return

        # Plain Enter submits
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            value = self.text.strip()
            if value:
                self.post_message(self.Submitted(value))
            return

        # Up/Down arrow: only navigate history at input boundaries.
        # Up requires cursor at position (0, 0); Down requires cursor at
        # the very end.  When already browsing history, either boundary
        # allows navigation in both directions.
        if event.key in {"up", "down"}:
            row, col = self.cursor_location
            text = self.text
            lines = text.split("\n")
            last_row = len(lines) - 1
            at_start = row == 0 and col == 0
            at_end = row == last_row and col == len(lines[last_row])
            navigate = (
                event.key == "up" and (at_start or (self._in_history and at_end))
            ) or (event.key == "down" and (at_end or (self._in_history and at_start)))

            if navigate:
                event.prevent_default()
                event.stop()
                self._navigating_history = True
                if event.key == "up":
                    self.post_message(self.HistoryPrevious(self.text))
                else:
                    self.post_message(self.HistoryNext())
                return

        await super()._on_key(event)

    def _delete_image_placeholder(self, *, backwards: bool) -> bool:
        """Delete a full image placeholder token in one keypress.

        Args:
            backwards: Whether the delete action is backwards (`backspace`) or
                forwards (`delete`).

        Returns:
            `True` when a placeholder token was deleted.
        """
        if not self.text or not self.selection.is_empty:
            return False

        cursor_offset = self.document.get_index_from_location(self.cursor_location)  # type: ignore[attr-defined]  # Document has this method; DocumentBase stub is narrower
        span = self._find_image_placeholder_span(cursor_offset, backwards=backwards)
        if span is None:
            return False

        start, end = span
        start_location = self.document.get_location_from_index(start)  # type: ignore[attr-defined]  # Document has this method; DocumentBase stub is narrower
        end_location = self.document.get_location_from_index(end)  # type: ignore[attr-defined]
        self.delete(start_location, end_location)
        self.move_cursor(start_location)
        return True

    def _find_image_placeholder_span(
        self, cursor_offset: int, *, backwards: bool
    ) -> tuple[int, int] | None:
        """Return placeholder span to delete for current cursor and key direction.

        Args:
            cursor_offset: Character offset of the cursor from the start of text.
            backwards: Whether the delete action is backwards (backspace) or
                forwards (delete).
        """
        text = self.text
        for match in _IMAGE_PLACEHOLDER_PATTERN.finditer(text):
            start, end = match.span()
            if backwards:
                # Cursor is inside token or right after a trailing space inserted
                # with the token.
                if start < cursor_offset <= end:
                    return start, end
                if cursor_offset > 0:
                    previous_index = cursor_offset - 1
                    if (
                        previous_index < len(text)
                        and previous_index == end
                        and text[previous_index].isspace()
                    ):
                        return start, cursor_offset
            elif start <= cursor_offset < end:
                return start, end
        return None

    async def _on_paste(self, event: events.Paste) -> None:
        """Handle paste events and detect dragged file paths."""
        from deepagents_cli.input import parse_pasted_file_paths

        paths = parse_pasted_file_paths(event.text)
        if not paths:
            # Don't call super() here — Textual's MRO dispatch already calls
            # TextArea._on_paste after this handler returns. Calling super()
            # would insert the text a second time, duplicating the paste.
            return

        event.prevent_default()
        event.stop()
        self.post_message(self.PastedPaths(event.text, paths))

    def set_text_from_history(self, text: str) -> None:
        """Set text from history navigation."""
        self._navigating_history = True
        self.text = text
        # Move cursor to end
        lines = text.split("\n")
        last_row = len(lines) - 1
        last_col = len(lines[last_row])
        self.move_cursor((last_row, last_col))
        self._navigating_history = False

    def clear_text(self) -> None:
        """Clear the text area."""
        self._in_history = False
        self.text = ""
        self.move_cursor((0, 0))


class _CompletionViewAdapter:
    """Translate completion-space replacements to text-area coordinates."""

    def __init__(self, chat_input: ChatInput) -> None:
        """Initialize adapter with its owning `ChatInput`."""
        self._chat_input = chat_input

    def render_completion_suggestions(
        self, suggestions: list[tuple[str, str]], selected_index: int
    ) -> None:
        """Delegate suggestion rendering to `ChatInput`."""
        self._chat_input.render_completion_suggestions(suggestions, selected_index)

    def clear_completion_suggestions(self) -> None:
        """Delegate completion clearing to `ChatInput`."""
        self._chat_input.clear_completion_suggestions()

    def replace_completion_range(self, start: int, end: int, replacement: str) -> None:
        """Map completion indices to text-area indices before replacing text."""
        self._chat_input.replace_completion_range(
            self._chat_input._completion_index_to_text_index(start),
            self._chat_input._completion_index_to_text_index(end),
            replacement,
        )


class ChatInput(Vertical):
    """Chat input widget with prompt, multi-line text, autocomplete, and history.

    Features:
    - Multi-line input with TextArea
    - Enter to submit, Ctrl+J for newlines (reliable across terminals)
    - Up/Down arrows for command history at input boundaries (start/end of text)
    - Autocomplete for @ (files) and / (commands)
    """

    DEFAULT_CSS = """
    ChatInput {
        height: auto;
        min-height: 3;
        max-height: 25;
        padding: 0;
        background: $surface;
        border: solid $primary;
    }

    ChatInput.mode-bash {
        border: solid __MODE_BASH__;
    }

    ChatInput.mode-command {
        border: solid __MODE_CMD__;
    }

    ChatInput .input-row {
        height: auto;
        width: 100%;
    }

    ChatInput .input-prompt {
        width: 3;
        height: 1;
        padding: 0 1;
        color: $primary;
        text-style: bold;
    }

    ChatInput.mode-bash .input-prompt {
        color: __MODE_BASH__;
    }

    ChatInput.mode-command .input-prompt {
        color: __MODE_CMD__;
    }

    ChatInput ChatTextArea {
        width: 1fr;
        height: auto;
        min-height: 1;
        max-height: 8;
        border: none;
        background: transparent;
        padding: 0;
    }

    ChatInput ChatTextArea:focus {
        border: none;
    }
    """.replace("__MODE_BASH__", COLORS["mode_bash"]).replace(
        "__MODE_CMD__", COLORS["mode_command"]
    )

    class Submitted(Message):
        """Message sent when input is submitted."""

        def __init__(self, value: str, mode: str = "normal") -> None:
            """Initialize with value and mode."""
            super().__init__()
            self.value = value
            self.mode = mode

    class ModeChanged(Message):
        """Message sent when input mode changes."""

        def __init__(self, mode: str) -> None:
            """Initialize with new mode."""
            super().__init__()
            self.mode = mode

    mode: reactive[str] = reactive("normal")

    def __init__(
        self,
        cwd: str | Path | None = None,
        history_file: Path | None = None,
        image_tracker: ImageTracker | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the chat input widget.

        Args:
            cwd: Current working directory for file completion
            history_file: Path to history file (default: ~/.deepagents/history.jsonl)
            image_tracker: Optional tracker for attached images
            **kwargs: Additional arguments for parent
        """
        super().__init__(**kwargs)
        self._cwd = Path(cwd) if cwd else Path.cwd()
        self._image_tracker = image_tracker
        self._text_area: ChatTextArea | None = None
        self._popup: CompletionPopup | None = None
        self._completion_manager: MultiCompletionManager | None = None
        self._completion_view: _CompletionViewAdapter | None = None

        # Guard flag: set True before programmatically stripping the mode
        # prefix character so the resulting text-change event does not
        # re-evaluate mode.
        self._stripping_prefix = False

        # When the user submits, we clear the text area which fires a
        # text-change event. Without this guard the tracker would see the
        # now-empty text, assume all images were deleted, and discard them
        # before the app has a chance to send them. Each submit bumps the
        # counter by one; the next text-change event decrements it and
        # skips the sync.
        self._skip_image_sync_events = 0

        # Number of virtual prefix characters currently injected for
        # completion controller calls (0 for normal, 1 for bash/command).
        self._completion_prefix_len = 0

        # Track current suggestions for click handling
        self._current_suggestions: list[tuple[str, str]] = []
        self._current_selected_index = 0

        # Set up history manager
        if history_file is None:
            history_file = Path.home() / ".deepagents" / "history.jsonl"
        self._history = HistoryManager(history_file)

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual widget method convention
        """Compose the chat input layout.

        Yields:
            Widgets for the input row and completion popup.
        """
        with Horizontal(classes="input-row"):
            yield Static(">", classes="input-prompt", id="prompt")
            yield ChatTextArea(id="chat-input")

        yield CompletionPopup(id="completion-popup")

    def on_mount(self) -> None:
        """Initialize components after mount."""
        if _detect_charset_mode() == CharsetMode.ASCII:
            self.styles.border = ("ascii", "cyan")

        self._text_area = self.query_one("#chat-input", ChatTextArea)
        self._popup = self.query_one("#completion-popup", CompletionPopup)

        # Both controllers implement the CompletionController protocol but have
        # different concrete types; the list-item warning is a false positive.
        self._completion_view = _CompletionViewAdapter(self)
        self._completion_manager = MultiCompletionManager(
            [
                SlashCommandController(SLASH_COMMANDS, self._completion_view),
                FuzzyFileController(self._completion_view, cwd=self._cwd),
            ]  # type: ignore[list-item]  # Controller types are compatible at runtime
        )

        self._text_area.focus()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Detect input mode and update completions."""
        text = event.text_area.text
        self._sync_image_tracker_to_text(text)

        # History handlers explicitly decide mode and stripped display text.
        # Skip mode detection here so recalled entries don't inherit stale mode.
        if self._text_area and self._text_area._navigating_history:
            if self._completion_manager:
                self._completion_manager.reset()
            self.scroll_visible()
            return

        # Checked after the guards above so we skip the (potentially slow)
        # filesystem lookup when the text change came from history navigation
        # or prefix stripping, which never need path detection.
        is_path_payload = self._is_dropped_path_payload(text)

        # Guard: skip mode re-detection after we programmatically stripped
        # a prefix character.
        if self._stripping_prefix:
            self._stripping_prefix = False
        elif text and text[0] in _PREFIX_TO_MODE:
            if text[0] == "/" and is_path_payload:
                # Absolute dropped paths stay normal input, not slash-command mode.
                if self.mode != "normal":
                    self.mode = "normal"
            else:
                # Detected a mode-trigger prefix (e.g. "!" or "/").
                # Strip it unconditionally -- even when already in the correct
                # mode -- because completion controllers may write replacement
                # text that re-includes the trigger character.  The
                # _stripping_prefix guard prevents the resulting change event
                # from looping back here.
                detected = _PREFIX_TO_MODE[text[0]]
                if self.mode != detected:
                    self.mode = detected
                self._strip_mode_prefix()
                return
        # Update completion suggestions using completion-space text/cursor.
        if self._completion_manager and self._text_area:
            if is_path_payload:
                self._completion_manager.reset()
            else:
                vtext, vcursor = self._completion_text_and_cursor()
                self._completion_manager.on_text_changed(vtext, vcursor)

        # Scroll input into view when content changes (handles text wrap)
        self.scroll_visible()

    @staticmethod
    def _is_existing_path_payload(text: str) -> bool:
        """Return whether text is a dropped-path payload for existing files."""
        from deepagents_cli.input import parse_pasted_file_paths

        if len(text) < 2:  # noqa: PLR2004  # Need at least '/' + one char
            return False
        return bool(parse_pasted_file_paths(text))

    def _is_dropped_path_payload(self, text: str) -> bool:
        """Return whether current text looks like a dropped file-path payload."""
        if not text:
            return False
        if self._is_existing_path_payload(text):
            return True
        if self.mode == "command":
            candidate = f"/{text.lstrip('/')}"
            return self._is_existing_path_payload(candidate)
        return False

    def _strip_mode_prefix(self) -> None:
        """Remove the first character (mode trigger) from the text area.

        Sets the `_stripping_prefix` guard so the resulting text-change event is
        not misinterpreted as new input.
        """
        if not self._text_area:
            return
        if self._stripping_prefix:
            logger.warning(
                "Previous _stripping_prefix guard was never cleared; "
                "resetting. This may indicate a missed text-change event."
            )
        text = self._text_area.text
        if not text:
            return
        row, col = self._text_area.cursor_location
        self._stripping_prefix = True
        self._text_area.text = text[1:]
        if row == 0 and col > 0:
            col -= 1
        self._text_area.move_cursor((row, col))

    def _completion_text_and_cursor(self) -> tuple[str, int]:
        """Return controller-facing text/cursor in completion space.

        Also updates `_completion_prefix_len` so that subsequent calls to
        `_completion_index_to_text_index` use the matching offset.
        """
        if not self._text_area:
            self._completion_prefix_len = 0
            return "", 0

        text = self._text_area.text
        cursor = self._get_cursor_offset()
        prefix = MODE_PREFIXES.get(self.mode, "")
        self._completion_prefix_len = len(prefix)

        if prefix:
            return prefix + text, cursor + len(prefix)
        return text, cursor

    def _completion_index_to_text_index(self, index: int) -> int:
        """Translate completion-space index into text-area index.

        Args:
            index: Cursor/index position in completion space.

        Returns:
            Clamped index in text-area space.
        """
        if not self._text_area:
            return 0

        mapped = index - self._completion_prefix_len
        text_len = len(self._text_area.text)
        if mapped < 0 or mapped > text_len:
            logger.warning(
                "Completion index %d mapped to %d, outside [0, %d]; "
                "clamping (prefix_len=%d, mode=%s)",
                index,
                mapped,
                text_len,
                self._completion_prefix_len,
                self.mode,
            )
        return max(0, min(mapped, text_len))

    def _submit_value(self, value: str) -> None:
        """Prepend mode prefix, save to history, post message, and reset input.

        This is the single path for all submission flows so the prefix-prepend +
        history + post + clear + mode-reset logic stays in one place.

        Args:
            value: The stripped text to submit (without mode prefix).
        """
        if not value:
            return

        if self._completion_manager:
            self._completion_manager.reset()

        value = self._replace_submitted_paths_with_images(value)

        # Prepend mode prefix so the app layer receives the original trigger
        # form (e.g. "!ls", "/help"). The value may already contain the prefix
        # when a completion controller wrote it back into the text area before
        # the strip handler ran.
        prefix = MODE_PREFIXES.get(self.mode, "")
        if prefix and not value.startswith(prefix):
            value = prefix + value

        self._history.add(value)
        self.post_message(self.Submitted(value, self.mode))

        if self._text_area:
            # Preserve submission-time attachments until adapter consumes them.
            self._skip_image_sync_events += 1
            self._text_area.clear_text()
        self.mode = "normal"

    def _sync_image_tracker_to_text(self, text: str) -> None:
        """Keep tracked images aligned with placeholder tokens in input text.

        Args:
            text: Current text in the input area.
        """
        if not self._image_tracker:
            return
        if self._skip_image_sync_events:
            if self._skip_image_sync_events < 0:
                logger.warning(
                    "_skip_image_sync_events is negative (%d); resetting to 0",
                    self._skip_image_sync_events,
                )
                self._skip_image_sync_events = 0
            else:
                self._skip_image_sync_events -= 1
            return
        self._image_tracker.sync_to_text(text)

    def on_chat_text_area_submitted(self, event: ChatTextArea.Submitted) -> None:
        """Handle text submission.

        Always posts the Submitted event - the app layer decides whether to
        process immediately or queue based on agent status.
        """
        self._submit_value(event.value)

    def on_chat_text_area_history_previous(
        self, event: ChatTextArea.HistoryPrevious
    ) -> None:
        """Handle history previous request."""
        entry = self._history.get_previous(event.current_text)
        if entry is not None and self._text_area:
            mode, display_text = self._history_entry_mode_and_text(entry)
            self.mode = mode
            self._text_area.set_text_from_history(display_text)
        elif self._text_area:
            self._text_area._navigating_history = False
        # Keep text area's _in_history in sync with the history manager.
        if self._text_area:
            self._text_area._in_history = self._history.in_history

    def on_chat_text_area_history_next(
        self,
        event: ChatTextArea.HistoryNext,  # noqa: ARG002  # Textual event handler signature
    ) -> None:
        """Handle history next request."""
        entry = self._history.get_next()
        if entry is not None and self._text_area:
            mode, display_text = self._history_entry_mode_and_text(entry)
            self.mode = mode
            self._text_area.set_text_from_history(display_text)
        elif self._text_area:
            self._text_area._navigating_history = False
        # Keep text area's _in_history in sync with the history manager.
        # When the user presses Down past the newest entry, get_next()
        # resets navigation internally, so in_history becomes False.
        if self._text_area:
            self._text_area._in_history = self._history.in_history

    def on_chat_text_area_pasted_paths(self, event: ChatTextArea.PastedPaths) -> None:
        """Handle paste payloads that resolve to dropped file paths."""
        if not self._text_area:
            return

        self._insert_pasted_paths(event.raw_text, event.paths)

    def handle_external_paste(self, pasted: str) -> bool:
        """Handle paste text from app-level routing when input is not focused.

        When the text area is mounted, the paste is always consumed: file paths
        are attached as images, and plain text is inserted directly.

        Args:
            pasted: Raw pasted text payload.

        Returns:
            `True` when the text area is mounted and the paste was inserted,
                `False` if the widget is not yet composed.
        """
        if not self._text_area:
            return False

        from deepagents_cli.input import parse_pasted_file_paths

        paths = parse_pasted_file_paths(pasted)
        if paths:
            self._insert_pasted_paths(pasted, paths)
        else:
            self._text_area.insert(pasted)

        self._text_area.focus()
        return True

    def _insert_pasted_paths(self, raw_text: str, paths: list[Path]) -> None:
        """Insert pasted path payload, attaching images when possible.

        Args:
            raw_text: Original paste payload text.
            paths: Resolved file paths parsed from the payload.
        """
        if not self._text_area:
            return
        replacement, attached = self._build_path_replacement(
            raw_text, paths, add_trailing_space=True
        )
        if attached:
            self._text_area.insert(replacement)
            return
        self._text_area.insert(raw_text)

    def _build_path_replacement(
        self,
        raw_text: str,
        paths: list[Path],
        *,
        add_trailing_space: bool,
    ) -> tuple[str, bool]:
        """Build replacement text for dropped paths and attach any images.

        Args:
            raw_text: Original paste payload text.
            paths: Resolved file paths parsed from the payload.
            add_trailing_space: Whether to append a trailing space after the
                last token when paths are separated by spaces.

        Returns:
            Tuple of `(replacement, attached)` where `attached` indicates whether
            at least one image attachment was created.
        """
        if not self._image_tracker:
            return raw_text, False

        from deepagents_cli.image_utils import get_image_from_path

        parts: list[str] = []
        attached = False
        for path in paths:
            image_data = get_image_from_path(path)
            if image_data is None:
                logger.debug("Could not load image from dropped path: %s", path)
                parts.append(str(path))
                continue
            parts.append(self._image_tracker.add_image(image_data))
            attached = True

        if not attached:
            return raw_text, False

        separator = "\n" if "\n" in raw_text else " "
        replacement = separator.join(parts)
        if separator == " " and add_trailing_space:
            replacement += " "
        return replacement, True

    def _replace_submitted_paths_with_images(self, value: str) -> str:
        """Replace dropped-path payloads in submitted text with image placeholders.

        Args:
            value: Stripped submitted text (without mode prefix).

        Returns:
            Submitted text with image placeholders when attachment succeeded.
        """
        from deepagents_cli.input import parse_pasted_file_paths

        paths = parse_pasted_file_paths(value)
        candidate = value

        # Recovery path: if command mode stripped the leading slash from an
        # absolute dropped path, rehydrate it before resolving attachments.
        if not paths and self.mode == "command":
            prefixed = f"/{value.lstrip('/')}"
            paths = parse_pasted_file_paths(prefixed)
            if paths:
                candidate = prefixed
                logger.debug(
                    "Recovering stripped absolute path; resetting mode from "
                    "'command' to 'normal'"
                )
                self.mode = "normal"

        if paths:
            replacement, attached = self._build_path_replacement(
                candidate, paths, add_trailing_space=False
            )
            if attached:
                return replacement.strip()
        return value

    @staticmethod
    def _history_entry_mode_and_text(entry: str) -> tuple[str, str]:
        """Return mode and stripped display text for a history entry.

        Args:
            entry: Raw entry value read from history storage.

        Returns:
            Tuple of `(mode, display_text)` where mode-trigger prefixes are
                removed from `display_text`.
        """
        for prefix, mode in _PREFIX_TO_MODE.items():
            # Small dict; loop is fine. No need to over-engineer right now
            if entry.startswith(prefix):
                return mode, entry[len(prefix) :]
        return "normal", entry

    async def on_key(self, event: events.Key) -> None:
        """Handle key events for completion navigation."""
        if not self._completion_manager or not self._text_area:
            return

        # Backspace on empty input exits the current mode (e.g. command/bash)
        if (
            event.key == "backspace"
            and not self._text_area.text
            and self.mode != "normal"
        ):
            self._completion_manager.reset()
            self.mode = "normal"
            event.prevent_default()
            event.stop()
            return

        text, cursor = self._completion_text_and_cursor()
        result = self._completion_manager.on_key(event, text, cursor)

        match result:
            case CompletionResult.HANDLED:
                event.prevent_default()
                event.stop()
            case CompletionResult.SUBMIT:
                event.prevent_default()
                event.stop()
                self._submit_value(self._text_area.text.strip())
            case CompletionResult.IGNORED if event.key == "enter":
                # Handle Enter when completion is not active (bash/normal modes)
                value = self._text_area.text.strip()
                if value:
                    event.prevent_default()
                    event.stop()
                    self._submit_value(value)

    def _get_cursor_offset(self) -> int:
        """Get the cursor offset as a single integer.

        Returns:
            Cursor position as character offset from start of text.
        """
        if not self._text_area:
            return 0

        text = self._text_area.text
        row, col = self._text_area.cursor_location

        if not text:
            return 0

        lines = text.split("\n")
        row = max(0, min(row, len(lines) - 1))
        col = max(0, col)

        offset = sum(len(lines[i]) + 1 for i in range(row))
        return offset + min(col, len(lines[row]))

    def watch_mode(self, mode: str) -> None:
        """Post mode changed message and update prompt indicator."""
        try:
            prompt = self.query_one("#prompt", Static)
        except NoMatches:
            return
        self.remove_class("mode-bash", "mode-command")
        prefix = MODE_PREFIXES.get(mode)
        if prefix:
            prompt.update(prefix)
            self.add_class(f"mode-{mode}")
        else:
            prompt.update(">")
        self.post_message(self.ModeChanged(mode))

    def focus_input(self) -> None:
        """Focus the input field."""
        if self._text_area:
            self._text_area.focus()

    @property
    def value(self) -> str:
        """Get the current input value.

        Returns:
            Current text in the input field.
        """
        if self._text_area:
            return self._text_area.text
        return ""

    @value.setter
    def value(self, val: str) -> None:
        """Set the input value."""
        if self._text_area:
            self._text_area.text = val

    @property
    def input_widget(self) -> ChatTextArea | None:
        """Get the underlying TextArea widget.

        Returns:
            The ChatTextArea widget or None if not mounted.
        """
        return self._text_area

    def set_disabled(self, *, disabled: bool) -> None:
        """Enable or disable the input widget."""
        if self._text_area:
            self._text_area.disabled = disabled
            if disabled:
                self._text_area.blur()
                if self._completion_manager:
                    self._completion_manager.reset()

    def set_cursor_active(self, *, active: bool) -> None:
        """Set whether the cursor should be actively blinking.

        When active=False (e.g., agent is working), disables cursor blink
        so the cursor doesn't flash while waiting for a response.
        """
        if self._text_area:
            self._text_area.set_app_focus(has_focus=active)

    def dismiss_completion(self) -> bool:
        """Dismiss completion: clear view and reset controller state.

        Returns:
            True if completion was active and has been dismissed.
        """
        if not self._current_suggestions:
            return False
        if self._completion_manager:
            self._completion_manager.reset()
        # Always clear local state so the popup is hidden even if the
        # manager's active controller was already None (no-op reset).
        self.clear_completion_suggestions()
        return True

    # =========================================================================
    # CompletionView protocol implementation
    # =========================================================================

    def render_completion_suggestions(
        self, suggestions: list[tuple[str, str]], selected_index: int
    ) -> None:
        """Render completion suggestions in the popup."""
        # Track suggestions locally for click handling
        self._current_suggestions = suggestions
        self._current_selected_index = selected_index

        if self._popup:
            self._popup.update_suggestions(suggestions, selected_index)
        # Tell TextArea that completion is active so it yields navigation keys
        if self._text_area:
            self._text_area.set_completion_active(active=bool(suggestions))

    def clear_completion_suggestions(self) -> None:
        """Clear/hide the completion popup."""
        self._current_suggestions = []
        self._current_selected_index = 0

        if self._popup:
            self._popup.hide()
        # Tell TextArea that completion is no longer active
        if self._text_area:
            self._text_area.set_completion_active(active=False)

    def on_completion_popup_option_clicked(
        self, event: CompletionPopup.OptionClicked
    ) -> None:
        """Handle click on a completion option."""
        if not self._current_suggestions or not self._text_area:
            return

        index = event.index
        if index < 0 or index >= len(self._current_suggestions):
            return

        # Get the selected completion
        label, _ = self._current_suggestions[index]
        text = self._text_area.text
        cursor = self._get_cursor_offset()

        # Determine replacement range based on completion type.
        # Slash completions use completion-space coordinates and are translated
        # through the completion view adapter.
        if label.startswith("/"):
            if self._completion_view is None:
                logger.warning(
                    "Slash completion clicked but _completion_view is not "
                    "initialized; this indicates a widget lifecycle issue."
                )
                return
            _, virtual_cursor = self._completion_text_and_cursor()
            self._completion_view.replace_completion_range(0, virtual_cursor, label)
        elif label.startswith("@"):
            # File mention: replace from @ to cursor
            at_index = text[:cursor].rfind("@")
            if at_index >= 0:
                self.replace_completion_range(at_index, cursor, label)

        # Reset completion state
        if self._completion_manager:
            self._completion_manager.reset()

        # Re-focus the text input after click
        self._text_area.focus()

    def replace_completion_range(self, start: int, end: int, replacement: str) -> None:
        """Replace text in the input field."""
        if not self._text_area:
            return

        text = self._text_area.text

        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))

        prefix = text[:start]
        suffix = text[end:]

        # Add space after completion unless it's a directory path
        if replacement.endswith("/"):
            insertion = replacement
        else:
            insertion = replacement + " " if not suffix.startswith(" ") else replacement

        new_text = f"{prefix}{insertion}{suffix}"
        self._text_area.text = new_text

        # Calculate new cursor position and move cursor
        new_offset = start + len(insertion)
        lines = new_text.split("\n")
        remaining = new_offset
        for row, line in enumerate(lines):
            if remaining <= len(line):
                self._text_area.move_cursor((row, remaining))
                break
            remaining -= len(line) + 1
