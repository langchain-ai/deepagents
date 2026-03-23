"""Chat input widget for deepagents-cli with autocomplete and history support."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static, TextArea

from deepagents_cli import theme
from deepagents_cli.command_registry import SLASH_COMMANDS
from deepagents_cli.config import (
    MODE_DISPLAY_GLYPHS,
    MODE_PREFIXES,
    PREFIX_TO_MODE,
    get_glyphs,
    is_ascii_mode,
)
from deepagents_cli.input import IMAGE_PLACEHOLDER_PATTERN, VIDEO_PLACEHOLDER_PATTERN
from deepagents_cli.widgets.autocomplete import (
    CompletionResult,
    FuzzyFileController,
    MultiCompletionManager,
    SlashCommandController,
)
from deepagents_cli.widgets.history import HistoryManager

logger = logging.getLogger(__name__)


def _default_history_path() -> Path:
    """Return the default history file path.

    Extracted as a function so tests can monkeypatch it to a temp path,
    preventing test runs from polluting `~/.deepagents/history.jsonl`.
    """
    return Path.home() / ".deepagents" / "history.jsonl"


_PASTE_BURST_CHAR_GAP_SECONDS = 0.03
"""Maximum time between chars to treat input as a paste-like burst."""

_PASTE_BURST_FLUSH_DELAY_SECONDS = 0.08
"""Idle timeout before flushing buffered burst text."""

_PASTE_BURST_START_CHARS = {"'", '"'}
"""Characters that can start dropped-path payloads."""

_BACKSLASH_ENTER_GAP_SECONDS = 0.15
"""Maximum gap between a `\\` key and a following `enter` key to treat the
pair as a terminal-emitted shift+enter sequence.

Some terminals (e.g. VSCode's built-in terminal) send a literal backslash
followed by enter when the user presses shift+enter.  The gap is
generous (150 ms) because the terminal emits both characters nearly
simultaneously; a human deliberately typing `\\` then pressing Enter would
have a much larger gap."""

if TYPE_CHECKING:
    from textual import events
    from textual.app import ComposeResult
    from textual.events import Click
    from textual.timer import Timer

    from deepagents_cli.input import MediaTracker, ParsedPastedPathPayload


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
        color: $background;
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
            content = Content.from_markup(
                f"{cursor}[bold]$label[/bold]  [dim]$desc[/dim]",
                label=self._label,
                desc=self._description,
            )
        else:
            content = Content.from_markup(
                f"{cursor}[bold]$label[/bold]", label=self._label
            )

        self.update(content)

        if self._is_selected:
            self.add_class("completion-option-selected")
        else:
            self.remove_class("completion-option-selected")

    def set_selected(self, *, selected: bool) -> None:
        """Update the selected state of this option."""
        if self._is_selected != selected:
            self._is_selected = selected
            self._update_display()

    def set_content(
        self, label: str, description: str, index: int, *, is_selected: bool
    ) -> None:
        """Replace label, description, index, and selection in-place."""
        self._label = label
        self._description = description
        self._index = index
        self._is_selected = is_selected
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
        self._pending_suggestions: list[tuple[str, str]] = []
        self._pending_selected: int = 0
        self._rebuild_generation: int = 0

    def update_suggestions(
        self, suggestions: list[tuple[str, str]], selected_index: int
    ) -> None:
        """Update the popup with new suggestions."""
        if not suggestions:
            self.hide()
            return

        self._selected_index = selected_index
        self._pending_suggestions = suggestions
        self._pending_selected = selected_index
        # Increment generation so stale callbacks from prior calls are skipped.
        self._rebuild_generation += 1
        gen = self._rebuild_generation
        # show() deferred to _rebuild_options to avoid a flash of stale content.
        self.call_after_refresh(lambda: self._rebuild_options(gen))

    async def _rebuild_options(self, generation: int) -> None:
        """Rebuild option widgets from pending suggestions.

        Reuses existing DOM nodes where possible to avoid flicker from
        a full teardown/mount cycle while the popup is visible.

        Args:
            generation: Caller's generation counter; skipped if superseded.
        """
        if generation != self._rebuild_generation:
            return

        suggestions = self._pending_suggestions
        selected_index = self._pending_selected

        if not suggestions:
            self.hide()
            return

        existing = len(self._options)
        needed = len(suggestions)

        # Update existing widgets in-place
        for i in range(min(existing, needed)):
            label, desc = suggestions[i]
            self._options[i].set_content(
                label, desc, i, is_selected=(i == selected_index)
            )

        # DOM mutations: trim extras / mount new widgets
        try:
            if existing > needed:
                for option in self._options[needed:]:
                    await option.remove()
                del self._options[needed:]

            if needed > existing:
                new_widgets: list[CompletionOption] = []
                for idx in range(existing, needed):
                    label, desc = suggestions[idx]
                    option = CompletionOption(
                        label=label,
                        description=desc,
                        index=idx,
                        is_selected=(idx == selected_index),
                    )
                    new_widgets.append(option)
                self._options.extend(new_widgets)
                await self.mount(*new_widgets)
        except Exception:
            logger.exception("Failed to rebuild completion popup; hiding to recover")
            self._options = []
            with contextlib.suppress(Exception):
                await self.remove_children()
            self.hide()
            return

        self.show()

        if 0 <= selected_index < len(self._options):
            self._options[selected_index].scroll_visible()

    def update_selection(self, selected_index: int) -> None:
        """Update which option is selected without rebuilding the list."""
        # Keep pending state in sync so an in-flight _rebuild_options uses
        # the latest selection.
        self._pending_selected = selected_index

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
        self._rebuild_generation += 1  # Cancel any in-flight rebuild
        self.styles.display = "none"  # type: ignore[assignment]  # Textual accepts string display values at runtime

    def show(self) -> None:
        """Show the popup."""
        self.styles.display = "block"


class ChatTextArea(TextArea):
    """TextArea subclass with custom key handling for chat input."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding(
            "shift+enter,alt+enter,ctrl+enter",
            "insert_newline",
            "New Line",
            show=False,
            priority=True,
        ),
    ]
    """Key bindings for the chat text area.

    These are the single source of truth for shortcut keys. `_NEWLINE_KEYS`
    is derived from this list so that `_on_key` stays in sync automatically.
    """

    _NEWLINE_KEYS: ClassVar[frozenset[str]] = frozenset(
        key
        for b in BINDINGS
        if b.action == "insert_newline"
        for key in b.key.split(",")
    )
    """Flattened set of keys that insert a newline, derived from `BINDINGS`."""

    _skip_history_change_events: int
    """Counter incremented before a history-driven text replacement so the
    resulting `TextArea.Changed` event (which fires on the next message-loop
    iteration) can be suppressed.  `ChatInput.on_text_area_changed` decrements
    the counter.
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

    class Typing(Message):
        """Posted when the user presses a printable key or backspace.

        Relayed by `ChatInput` as `ChatInput.Typing` for the app to track
        typing activity.
        """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the chat text area."""
        # Remove placeholder if passed, TextArea doesn't support it the same way
        kwargs.pop("placeholder", None)
        super().__init__(**kwargs)
        self._skip_history_change_events = 0
        self._in_history = False
        self._completion_active = False
        self._app_has_focus = True
        # Buffer quote-prefixed high-frequency key bursts from terminals that
        # emulate paste via rapid key events instead of dispatching a paste
        # event.
        self._paste_burst_buffer = ""
        self._paste_burst_last_char_time: float | None = None
        self._paste_burst_timer: Timer | None = None
        # See _BACKSLASH_ENTER_GAP_SECONDS for context.
        self._backslash_pending_time: float | None = None

    def set_app_focus(self, *, has_focus: bool) -> None:
        """Set whether the app should show the cursor as active.

        When has_focus=False (e.g., agent is running), disables cursor blink
        so the cursor doesn't flash while waiting for a response.
        """
        self._app_has_focus = has_focus
        self._backslash_pending_time = None
        self.cursor_blink = has_focus
        if has_focus and not self.has_focus:
            self.call_after_refresh(self.focus)

    def set_completion_active(self, *, active: bool) -> None:
        """Set whether completion suggestions are visible."""
        self._completion_active = active

    def action_insert_newline(self) -> None:
        """Insert a newline character."""
        self.insert("\n")

    def _cancel_paste_burst_timer(self) -> None:
        """Cancel any scheduled paste-burst flush timer."""
        if self._paste_burst_timer is None:
            return
        self._paste_burst_timer.stop()
        self._paste_burst_timer = None

    def _schedule_paste_burst_flush(self) -> None:
        """Schedule idle-time flush for buffered paste-burst text."""
        self._cancel_paste_burst_timer()
        self._paste_burst_timer = self.set_timer(
            _PASTE_BURST_FLUSH_DELAY_SECONDS, self._flush_paste_burst
        )

    def _start_paste_burst(self, char: str, now: float) -> None:
        """Start buffering a paste-like keystroke burst."""
        self._paste_burst_buffer = char
        self._paste_burst_last_char_time = now
        self._schedule_paste_burst_flush()

    def _append_paste_burst(self, text: str, now: float) -> None:
        """Append text to an active paste-burst buffer."""
        if not self._paste_burst_buffer:
            self._start_paste_burst(text, now)
            return
        self._paste_burst_buffer += text
        self._paste_burst_last_char_time = now
        self._schedule_paste_burst_flush()

    def _should_start_paste_burst(self, char: str) -> bool:
        """Return whether a keypress should start paste-burst buffering.

        Restricting to quote-prefixed input at an empty cursor reduces false
        positives for normal typing and slash-command entry.
        """
        if char not in _PASTE_BURST_START_CHARS:
            return False
        if self.text or not self.selection.is_empty:
            return False
        row, col = self.cursor_location
        return row == 0 and col == 0

    async def _flush_paste_burst(self) -> None:
        """Flush buffered burst text through dropped-path parsing.

        When parsing fails, the buffered text is inserted unchanged so regular
        typing behavior is preserved.
        """
        payload = self._paste_burst_buffer
        self._paste_burst_buffer = ""
        self._paste_burst_last_char_time = None
        self._cancel_paste_burst_timer()
        if not payload:
            return

        from deepagents_cli.input import parse_pasted_path_payload

        try:
            parsed = await asyncio.to_thread(parse_pasted_path_payload, payload)
        except Exception:  # noqa: BLE001  # Treat thread failure as non-path text
            parsed = None
        if parsed is not None:
            self.post_message(self.PastedPaths(payload, parsed.paths))
            return

        self.insert(payload)

    def _delete_preceding_backslash(self) -> bool:
        """Delete the backslash character immediately before the cursor.

        Caller must ensure a backslash is expected at this position. The
        method verifies the character before deleting it.

        Returns:
            `True` if a backslash was found and deleted, `False` otherwise.
        """
        row, col = self.cursor_location
        if col > 0:
            start = (row, col - 1)
            if self.document.get_text_range(start, self.cursor_location) == "\\":
                self.delete(start, self.cursor_location)
                return True
        elif row > 0:
            prev_line = self.document.get_line(row - 1)
            start = (row - 1, len(prev_line) - 1)
            end = (row - 1, len(prev_line))
            if self.document.get_text_range(start, end) == "\\":
                self.delete(start, self.cursor_location)
                return True
        return False

    async def _on_key(self, event: events.Key) -> None:
        """Handle key events."""
        # VS Code 1.110 incorrectly sends space as a CSI u escape code
        # (`\x1b[32u`) instead of a plain ` ` character.  Textual parses
        # this as Key(key='space', character=None, is_printable=False), so
        # the TextArea never inserts the space.  Per the kitty keyboard
        # protocol spec, keys that generate text (like space) should NOT
        # use CSI u encoding — VS Code is the outlier here.
        #
        # This workaround should be safe to keep indefinitely: once VS Code or
        # Textual fixes the issue upstream, `character` will be `' '` and
        # this branch simply won't match.
        #
        # Upstream: https://github.com/Textualize/textual/issues/6408
        if event.key == "space" and event.character is None:
            event.prevent_default()
            event.stop()
            self.insert(" ")
            self.post_message(self.Typing())
            return

        now = time.monotonic()

        # Signal typing activity for printable keys and backspace so the app
        # can defer approval widgets while the user is actively editing.
        if event.is_printable or event.key == "backspace":
            self.post_message(self.Typing())

        if self._paste_burst_buffer:
            if event.key == "enter":
                self._append_paste_burst("\n", now)
                event.prevent_default()
                event.stop()
                return

            if event.is_printable and event.character is not None:
                last_time = self._paste_burst_last_char_time
                if (
                    last_time is not None
                    and (now - last_time) <= _PASTE_BURST_CHAR_GAP_SECONDS
                ):
                    self._append_paste_burst(event.character, now)
                    event.prevent_default()
                    event.stop()
                    return

            await self._flush_paste_burst()

        if (
            event.is_printable
            and event.character is not None
            and self._should_start_paste_burst(event.character)
        ):
            self._start_paste_burst(event.character, now)
            event.prevent_default()
            event.stop()
            return

        # Some terminals (e.g. VSCode built-in) send a literal backslash
        # followed by enter for shift+enter.  When enter arrives shortly
        # after a backslash, delete the backslash and insert a newline.
        if (
            event.key == "enter"
            and not self._completion_active
            and self._backslash_pending_time is not None
            and (now - self._backslash_pending_time) <= _BACKSLASH_ENTER_GAP_SECONDS
        ):
            self._backslash_pending_time = None
            if self._delete_preceding_backslash():
                event.prevent_default()
                event.stop()
                self.insert("\n")
                return
        self._backslash_pending_time = None

        if event.key == "backslash" and event.character == "\\":
            self._backslash_pending_time = now

        # Modifier+Enter inserts newline — keys derived from BINDINGS
        if event.key in self._NEWLINE_KEYS:
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
        # Check both image and video placeholders
        for pattern in (IMAGE_PLACEHOLDER_PATTERN, VIDEO_PLACEHOLDER_PATTERN):
            for match in pattern.finditer(text):
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
        self._backslash_pending_time = None
        if self._paste_burst_buffer:
            await self._flush_paste_burst()

        from deepagents_cli.input import parse_pasted_path_payload

        try:
            parsed = await asyncio.to_thread(parse_pasted_path_payload, event.text)
        except Exception:  # noqa: BLE001  # Treat thread failure as non-path text
            parsed = None
        if parsed is None:
            # Don't call super() here — Textual's MRO dispatch already calls
            # TextArea._on_paste after this handler returns. Calling super()
            # would insert the text a second time, duplicating the paste.
            return

        event.prevent_default()
        event.stop()
        self.post_message(self.PastedPaths(event.text, parsed.paths))

    def set_text_from_history(self, text: str) -> None:
        """Set text from history navigation."""
        self._paste_burst_buffer = ""
        self._paste_burst_last_char_time = None
        self._cancel_paste_burst_timer()
        self._backslash_pending_time = None
        self._skip_history_change_events += 1
        self.text = text
        # Move cursor to end
        lines = text.split("\n")
        last_row = len(lines) - 1
        last_col = len(lines[last_row])
        self.move_cursor((last_row, last_col))

    def clear_text(self) -> None:
        """Clear the text area."""
        self._in_history = False
        # Increment (not reset) so any pending Changed event from a prior
        # set_text_from_history is still suppressed, plus one for the
        # self.text = "" assignment below.
        self._skip_history_change_events += 1
        self._paste_burst_buffer = ""
        self._paste_burst_last_char_time = None
        self._cancel_paste_burst_timer()
        self._backslash_pending_time = None
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
    - Enter to submit, modifier key for newlines (see `config.newline_shortcut`)
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

    ChatInput.mode-shell {
        border: solid $mode-bash;
    }

    ChatInput.mode-command {
        border: solid $mode-command;
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

    ChatInput.mode-shell .input-prompt {
        color: $mode-bash;
    }

    ChatInput.mode-command .input-prompt {
        color: $mode-command;
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
    """
    """Border and prompt glyph change color per mode for immediate visual feedback."""

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

    class Typing(Message):
        """Posted when the user presses a printable key or backspace in the input.

        The app uses this to delay approval widgets while the user is actively
        typing, preventing accidental key presses (e.g. `y`, `n`) from
        triggering approval decisions.
        """

    mode: reactive[str] = reactive("normal")

    def __init__(
        self,
        cwd: str | Path | None = None,
        history_file: Path | None = None,
        image_tracker: MediaTracker | None = None,
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
        self._slash_controller: SlashCommandController | None = None

        # Guard flag: set True before programmatically stripping the mode
        # prefix character so the resulting text-change event does not
        # re-evaluate mode.
        self._stripping_prefix = False

        # When the user submits, we clear the text area which fires a
        # text-change event. Without this guard the tracker would see the
        # now-empty text, assume all media were deleted, and discard them
        # before the app has a chance to send them. Each submit bumps the
        # counter by one; the next text-change event decrements it and
        # skips the sync.
        self._skip_media_sync_events = 0

        # Number of virtual prefix characters currently injected for
        # completion controller calls (0 for normal, 1 for shell/command).
        self._completion_prefix_len = 0

        # Guard flag: set while replacing a dropped path payload with an
        # inline image placeholder so the resulting change event doesn't
        # immediately recurse into the same replacement path.
        self._applying_inline_path_replacement = False

        # Track previous text length so we can skip expensive filesystem
        # path detection on single-character typing (normal keystrokes).
        # Dropped paths are always multi-character paste/insert events.
        self._prev_text_len = 0

        # Track current suggestions for click handling
        self._current_suggestions: list[tuple[str, str]] = []
        self._current_selected_index = 0

        # Set up history manager
        if history_file is None:
            history_file = _default_history_path()
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
        if is_ascii_mode():
            colors = theme.get_theme_colors(self)
            self.styles.border = ("ascii", colors.primary)

        self._text_area = self.query_one("#chat-input", ChatTextArea)
        self._popup = self.query_one("#completion-popup", CompletionPopup)

        # Both controllers implement the CompletionController protocol but have
        # different concrete types; the list-item warning is a false positive.
        self._completion_view = _CompletionViewAdapter(self)
        self._file_controller = FuzzyFileController(
            self._completion_view, cwd=self._cwd
        )
        self._slash_controller = SlashCommandController(
            SLASH_COMMANDS, self._completion_view
        )
        self._completion_manager = MultiCompletionManager(
            [
                self._slash_controller,
                self._file_controller,
            ]  # type: ignore[list-item]  # Controller types are compatible at runtime
        )

        self.run_worker(
            self._file_controller.warm_cache(),
            exclusive=False,
            exit_on_error=False,
        )
        self._text_area.focus()

    def update_slash_commands(self, commands: list[tuple[str, str, str]]) -> None:
        """Update the slash command controller's command list.

        Called by the app after discovering skills to merge static
        commands with dynamic `/skill:` entries.

        Args:
            commands: Full list of `(command, description, hidden_keywords)` tuples.
        """
        if self._slash_controller:
            self._slash_controller.update_commands(commands)
        else:
            logger.warning(
                "Cannot update slash commands: controller not initialized "
                "(widget not yet mounted)"
            )

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Detect input mode and update completions."""
        text = event.text_area.text
        self._sync_media_tracker_to_text(text)

        # Detect whether this change looks like a multi-character paste/drop
        # rather than a single keystroke. Filesystem-backed path detection is
        # expensive (stat syscalls) and only relevant for drop/paste payloads,
        # so we skip it entirely for normal single-char typing.
        text_len = len(text)
        is_bulk_change = abs(text_len - self._prev_text_len) > 1
        self._prev_text_len = text_len

        # History handlers explicitly decide mode and stripped display text.
        # Skip mode detection here so recalled entries don't inherit stale mode.
        if self._text_area and self._text_area._skip_history_change_events > 0:
            self._text_area._skip_history_change_events -= 1
            if self._completion_manager:
                self._completion_manager.reset()
            self.scroll_visible()
            return
        if self._text_area and self._text_area._skip_history_change_events < 0:
            logger.warning(
                "_skip_history_change_events is negative (%d); resetting to 0",
                self._text_area._skip_history_change_events,
            )
            self._text_area._skip_history_change_events = 0

        if self._applying_inline_path_replacement:
            self._applying_inline_path_replacement = False
        elif is_bulk_change and self._apply_inline_dropped_path_replacement(text):
            return

        # Only run the (potentially slow) filesystem-backed path detection
        # when the change looks like a paste/drop (multi-character insert).
        # Single-character typing changes skip this entirely.
        is_path_payload = is_bulk_change and self._is_dropped_path_payload(text)

        # Guard: skip mode re-detection after we programmatically stripped
        # a prefix character.
        if self._stripping_prefix:
            self._stripping_prefix = False
        elif text and text[0] in PREFIX_TO_MODE:
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
                detected = PREFIX_TO_MODE[text[0]]
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
    def _parse_dropped_path_payload(
        text: str, *, allow_leading_path: bool = False
    ) -> ParsedPastedPathPayload | None:
        """Parse dropped-path payload text through a single parser entrypoint.

        Returns:
            Parsed payload details, otherwise `None`.
        """
        from deepagents_cli.input import parse_pasted_path_payload

        return parse_pasted_path_payload(text, allow_leading_path=allow_leading_path)

    def _parse_dropped_path_payload_with_command_recovery(
        self, text: str, *, allow_leading_path: bool = False
    ) -> tuple[str, ParsedPastedPathPayload | None]:
        """Parse payload and recover stripped leading slash in command mode.

        Args:
            text: Input text to parse.
            allow_leading_path: Whether to parse leading path + suffix payloads.

        Returns:
            Tuple of `(candidate_text, parsed_payload)`.
        """
        candidate = text
        parsed = self._parse_dropped_path_payload(
            text, allow_leading_path=allow_leading_path
        )
        if parsed is not None:
            return candidate, parsed

        if self.mode != "command":
            return candidate, None

        prefixed = f"/{text.lstrip('/')}"
        parsed = self._parse_dropped_path_payload(
            prefixed, allow_leading_path=allow_leading_path
        )
        if parsed is None:
            return candidate, None

        logger.debug(
            "Recovering stripped absolute path; resetting mode from "
            "'command' to 'normal'"
        )
        self.mode = "normal"
        return prefixed, parsed

    def _extract_leading_dropped_path_with_command_recovery(
        self, text: str
    ) -> tuple[str, tuple[Path, int] | None]:
        """Extract a leading dropped-path token with command-mode recovery.

        Args:
            text: Input text to parse.

        Returns:
            Tuple of `(candidate_text, leading_match)`, where `leading_match` is
            `(path, token_end)` when extraction succeeds, otherwise `None`.
        """
        from deepagents_cli.input import extract_leading_pasted_file_path

        leading_match = extract_leading_pasted_file_path(text)
        candidate = text
        if leading_match is not None:
            return candidate, leading_match

        if self.mode != "command":
            return candidate, None

        prefixed = f"/{text.lstrip('/')}"
        leading_match = extract_leading_pasted_file_path(prefixed)
        if leading_match is None:
            return candidate, None

        logger.debug(
            "Recovering stripped absolute leading path; resetting mode "
            "from 'command' to 'normal'"
        )
        self.mode = "normal"
        return prefixed, leading_match

    @staticmethod
    def _is_existing_path_payload(text: str) -> bool:
        """Return whether text is a dropped-path payload for existing files."""
        from deepagents_cli.input import parse_pasted_path_payload

        if len(text) < 2:  # noqa: PLR2004  # Need at least '/' + one char
            return False
        return parse_pasted_path_payload(text, allow_leading_path=True) is not None

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
            self._skip_media_sync_events += 1
            self._text_area.clear_text()
        self.mode = "normal"

    def _sync_media_tracker_to_text(self, text: str) -> None:
        """Keep tracked media aligned with placeholder tokens in input text.

        Args:
            text: Current text in the input area.
        """
        if not self._image_tracker:
            return
        if self._skip_media_sync_events:
            if self._skip_media_sync_events < 0:
                logger.warning(
                    "_skip_media_sync_events is negative (%d); resetting to 0",
                    self._skip_media_sync_events,
                )
                self._skip_media_sync_events = 0
            else:
                self._skip_media_sync_events -= 1
            return
        self._image_tracker.sync_to_text(text)

    def on_chat_text_area_typing(
        self,
        event: ChatTextArea.Typing,  # noqa: ARG002  # Textual event handler signature
    ) -> None:
        """Relay typing activity to the app as `ChatInput.Typing`."""
        self.post_message(self.Typing())

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
        entry = self._history.get_previous(event.current_text, query=event.current_text)
        if entry is not None and self._text_area:
            mode, display_text = self._history_entry_mode_and_text(entry)
            self.mode = mode
            self._text_area.set_text_from_history(display_text)
        # No-match path: don't reset the counter — a pending Changed event
        # from a prior set_text_from_history call may still be in flight.
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
        # No-match path: don't reset the counter — a pending Changed event
        # from a prior set_text_from_history call may still be in flight.
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

        parsed = self._parse_dropped_path_payload(pasted)
        if parsed is None:
            self._text_area.insert(pasted)
        else:
            self._insert_pasted_paths(pasted, parsed.paths)

        self._text_area.focus()
        return True

    def _apply_inline_dropped_path_replacement(self, text: str) -> bool:
        """Replace full dropped-path payload text with image placeholders.

        Some terminals insert drag-and-drop payloads as plain text rather than
        dispatching a dedicated paste event. When the current text resolves to
        one or more file paths and at least one path is an image, rewrite the
        text inline to `[image N]` placeholders.

        Args:
            text: Current text area content.

        Returns:
            `True` if text was rewritten inline, otherwise `False`.
        """
        if not self._text_area:
            return False

        parsed = self._parse_dropped_path_payload(text)
        if parsed is None:
            return False

        replacement, attached = self._build_path_replacement(
            text, parsed.paths, add_trailing_space=True
        )
        if not attached or replacement == text:
            return False

        self._applying_inline_path_replacement = True
        self._text_area.text = replacement
        lines = replacement.split("\n")
        self._text_area.move_cursor((len(lines) - 1, len(lines[-1])))
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
            at least one media attachment (image or video) was created.
        """
        if not self._image_tracker:
            return raw_text, False

        from deepagents_cli.media_utils import (
            IMAGE_EXTENSIONS,
            MAX_MEDIA_BYTES,
            VIDEO_EXTENSIONS,
            ImageData,
            get_media_from_path,
        )

        parts: list[str] = []
        attached = False
        for path in paths:
            media = get_media_from_path(path)
            if media is not None:
                kind = "image" if isinstance(media, ImageData) else "video"
                parts.append(self._image_tracker.add_media(media, kind))
                attached = True
                continue

            # Check if it looked like media but failed validation
            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS or suffix in VIDEO_EXTENSIONS:
                label = "Video" if suffix in VIDEO_EXTENSIONS else "Image"
                try:
                    size = path.stat().st_size
                    if size > MAX_MEDIA_BYTES:
                        msg = (
                            f"{label} too large: {path.name} "
                            f"({size // (1024 * 1024)} MB, max "
                            f"{MAX_MEDIA_BYTES // (1024 * 1024)} MB)"
                        )
                    else:
                        msg = f"Could not attach {label.lower()}: {path.name}"
                except OSError as exc:
                    logger.debug("Failed to stat media file %s: %s", path, exc)
                    msg = f"Could not attach {label.lower()}: {path.name}"
                self.app.notify(msg, severity="warning", timeout=5, markup=False)

            # Not a supported media file, keep as path
            logger.debug("Could not load media from dropped path: %s", path)
            parts.append(str(path))

        if not attached:
            return raw_text, False

        separator = "\n" if "\n" in raw_text else " "
        replacement = separator.join(parts)
        if separator == " " and add_trailing_space:
            replacement += " "
        return replacement, True

    def _replace_submitted_paths_with_images(self, value: str) -> str:
        """Replace dropped-path payloads in submitted text with image placeholders.

        Handles both full-path payloads and leading-path-with-suffix payloads
        (for example, `'<path>' what is this?`). When command mode previously
        stripped a leading slash, this method also retries with the slash
        restored before giving up.

        Args:
            value: Stripped submitted text (without mode prefix).

        Returns:
            Submitted text with image placeholders when attachment succeeded.
        """
        candidate, parsed = self._parse_dropped_path_payload_with_command_recovery(
            value, allow_leading_path=True
        )
        if parsed is None:
            return value

        if parsed.token_end is None:
            replacement, attached = self._build_path_replacement(
                candidate, parsed.paths, add_trailing_space=False
            )
            if attached:
                return replacement.strip()
            # Even when full-payload parsing resolves, still retry explicit
            # leading-token extraction before giving up.
            candidate, leading_match = (
                self._extract_leading_dropped_path_with_command_recovery(value)
            )
            if leading_match is None:
                return value
            leading_path, token_end = leading_match
        else:
            leading_path = parsed.paths[0]
            token_end = parsed.token_end

        replacement, attached = self._build_path_replacement(
            str(leading_path), [leading_path], add_trailing_space=False
        )
        if attached:
            suffix = candidate[token_end:].lstrip()
            if suffix:
                return f"{replacement.strip()} {suffix}".strip()
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
        for prefix, mode in PREFIX_TO_MODE.items():
            # Small dict; loop is fine. No need to over-engineer right now
            if entry.startswith(prefix):
                return mode, entry[len(prefix) :]
        return "normal", entry

    async def on_key(self, event: events.Key) -> None:
        """Handle key events for completion navigation."""
        if not self._completion_manager or not self._text_area:
            return

        # Backspace at cursor position 0 (or on empty input) exits the
        # current mode (e.g. command/shell).  When the cursor is at the very
        # start of the text area, backspace is a no-op for the underlying
        # widget, so without this guard the user would be stuck in the mode.
        if (
            event.key == "backspace"
            and self.mode != "normal"
            and self._get_cursor_offset() == 0
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
                # Handle Enter when completion is not active (shell/normal modes)
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
            logger.warning("watch_mode: #prompt widget not found")
            self.post_message(self.ModeChanged(mode))
            return
        self.remove_class("mode-shell", "mode-command")
        glyph = MODE_DISPLAY_GLYPHS.get(mode)
        if glyph:
            prompt.update(glyph)
            self.add_class(f"mode-{mode}")
        else:
            if mode != "normal":
                logger.warning(
                    "No display glyph for mode %r; falling back to '>'",
                    mode,
                )
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

    def exit_mode(self) -> bool:
        """Exit the current input mode (command/shell) back to normal.

        Returns:
            True if mode was non-normal and has been reset.
        """
        if self.mode == "normal":
            return False
        self.mode = "normal"
        if self._completion_manager:
            self._completion_manager.reset()
        self.clear_completion_suggestions()
        return True

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
        prev_suggestions = self._current_suggestions
        self._current_suggestions = suggestions
        self._current_selected_index = selected_index

        if self._popup:
            # If only the selection changed (same items), skip full rebuild
            if suggestions == prev_suggestions:
                self._popup.update_selection(selected_index)
            else:
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
