"""Chat input widget for deepagents-code with autocomplete and history support."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, assert_never

from rich.cells import cell_len
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual.app import NoScreen
from textual.color import Color
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.geometry import Offset, Size
from textual.highlight import highlight
from textual.message import Message
from textual.reactive import reactive
from textual.strip import Strip
from textual.widgets import Input, Static, TextArea

from deepagents_code import theme
from deepagents_code.command_registry import CommandEntry, get_slash_commands
from deepagents_code.config import (
    MODE_DISPLAY_GLYPHS,
    MODE_PREFIXES,
    detect_mode_prefix,
    get_glyphs,
    is_ascii_mode,
)
from deepagents_code.input import (
    IMAGE_PLACEHOLDER_PATTERN,
    VIDEO_PLACEHOLDER_PATTERN,
)
from deepagents_code.paste_collapse import (
    PASTE_PLACEHOLDER_PATTERN,
    PastedContent,
    count_lines,
    expand_paste_refs,
    format_paste_ref,
    should_collapse_paste,
)
from deepagents_code.tui.widgets._paste_textarea import (
    PasteBurstTextArea,
    _collapse_pastes_enabled,
)
from deepagents_code.tui.widgets.autocomplete import (
    CompletionResult,
    FuzzyFileController,
    MultiCompletionManager,
    SlashCommandController,
)
from deepagents_code.tui.widgets.history import HistoryManager
from deepagents_code.tui.widgets.prompt_search import (
    PROMPT_SEARCH_PANEL_ROWS,
    PromptSearchInput,
    PromptSearchPanel,
    filter_prompts,
    prompt_title,
)

logger = logging.getLogger(__name__)


def _default_history_path() -> Path:
    """Return the default history file path.

    Extracted as a function so tests can monkeypatch it to a temp path,
    preventing test runs from polluting `~/.deepagents/.state/history.jsonl`.
    """
    from deepagents_code.model_config import DEFAULT_STATE_DIR

    return DEFAULT_STATE_DIR / "history.jsonl"


_LOCK_KEYS = frozenset({"caps_lock", "num_lock", "scroll_lock"})
"""Lock keys that must never insert text.

Under the kitty keyboard protocol with associated-text reporting (VS Code's
xterm.js and others), pressing a lock key arrives as a `Key` event whose
`character` is the text that *would* have been produced by the next key —
e.g. pressing Caps Lock reports `key='caps_lock'`, `character='A'`. Textual's
parser does not strip this, so `TextArea` inserts a stray letter. We drop
these events entirely. Terminals encode lock keys in several shapes (iTerm2
notably differs from kitty/Ghostty); `_textual_patches.py` is the canonical
reference and neutralizes every shape at the parser. See the kitty keyboard
protocol spec (functional key definitions) for background.
"""


_FILE_CACHE_WORKER_GROUP = "file-cache"
"""Textual worker group for all `@` file-completion cache warmers."""

_CHAT_INPUT_AUTO_MAX_HEIGHT = 8
"""Rows the composer grows to on its own before the draft starts scrolling.

Also caps the manual-resize floor: a drag will not shrink the composer below
the visible draft, but that refusal itself stops at this many rows, so a long
draft cannot pin the composer open. Interpolated into `ChatInput.DEFAULT_CSS`
as the `ChatTextArea` `max-height` so the stylesheet and the sizing math cannot
drift apart.
"""

_CHAT_INPUT_BORDER_CORNER_COLUMNS = 2
"""Columns the resize handle gives up so both top border corners stay drawn.

The handle occludes whatever it covers (see `ChatInputResizeHandle.render`), so
it is inset one column at each end — paired with `offset: 1 0` in the CSS — to
leave the box's corner glyphs visible.
"""

_CHAT_INPUT_BOX_MAX_HEIGHT = 25
"""Rows the bordered input box may occupy for its own content and border.

`ChatInputBox._apply_manual_height` subtracts the gutter and every inline panel
row (completion popup and prompt search) from this to derive the composer
budget, so it is load-bearing arithmetic rather than a cosmetic cap.

This is the composer's budget, not the box's rendered ceiling: the `#input-box`
`max-height` in `ChatInput.DEFAULT_CSS` interpolates
`_CHAT_INPUT_BOX_MAX_HEIGHT + PROMPT_SEARCH_PANEL_ROWS`, so a fully expanded
prompt search panel fits inside the border instead of pushing the draft out.
"""

_CHAT_INPUT_MANUAL_MAX_HEIGHT = 20
"""Rows a drag may stretch the composer to, before screen-size clamping.

Five rows short of `_CHAT_INPUT_BOX_MAX_HEIGHT` so that a composer dragged to
its limit still leaves room inside the box for the border and a few rows of
completion popup, rather than immediately fighting the popup for space.
"""

_CHAT_INPUT_RESERVED_SCREEN_ROWS = 7
"""Screen rows kept away from the composer text so the app stays usable.

Subtracted from the screen height to bound a manual resize. It counts rows
*outside* the composer's own text area but excludes the input box's 2-row
border, which is charged separately via `gutter.height`. So on a 24-row
terminal the composer maxes at 17 rows, the bordered box occupies 19, and about
5 rows remain for the transcript and status bar.
"""

_CHAT_INPUT_RESIZE_HOVER_LIGHTEN = 0.15
"""How far to lighten the resize line while the pointer is over it.

Enough to read as a distinct affordance against the same-colored box border it
sits on, but below the point where the top border looks like a different UI
element from the other three sides.
"""

_COMPLETION_POPUP_MAX_HEIGHT = 12
"""Rows the completion popup may occupy inside the input box.

`ChatInputBox` reserves this much space when fitting a manual composer height,
so it must match what the popup actually renders. Interpolated into
`CompletionPopup.DEFAULT_CSS` as its `max-height` to keep the two in step.
"""

_DOUBLE_CLICK_CHAIN = 2
"""Textual `Click.chain` count that marks a click as a double-click.

Compared with `>=`, so triple and faster clicks also toggle rather than being
silently ignored partway through a rapid click sequence.
"""

_REFOCUS_CLICK_SUPPRESS_WINDOW_SECONDS = 0.3
"""Window after a terminal focus regain during which a click only refocuses.

When the terminal window is unfocused and the user clicks back in, we rely on
the OS delivering a `FocusIn` (Textual `AppFocus`) before the mouse report —
the same FocusIn support `on_app_focus` documents. Terminals without it never
arm suppression, so clicks just behave normally (the cursor moves). A
mouse-down landing within this window after the focus regain is treated as
focus-only so the cursor stays put instead of jumping to the click location.

The window trades off two failure modes: too small and a genuine refocus click
leaks through and moves the cursor (the bug this guards against); too large and
an intentional click made shortly after refocusing is wrongly suppressed. 0.3s
comfortably covers the FocusIn-to-mouse-report latency while staying below a
deliberate click-pause-click interaction.
"""

_FILE_CACHE_REFRESH_INTERVAL_SECONDS = 30.0
"""How often to refresh the `@` file-completion cache in the background.

The cache is pre-warmed on mount and re-warmed on cwd switches, but files
created or deleted mid-session would otherwise stay stale until the next switch.
A periodic refresh keeps `@` suggestions current; the walk runs off the event
loop and swaps in atomically, so it never blocks typing."""

if TYPE_CHECKING:
    from textual import events
    from textual.app import ComposeResult
    from textual.events import Click
    from textual.screen import Screen

    from deepagents_code.config_manifest import CursorStyle
    from deepagents_code.input import MediaTracker, ParsedPastedPathPayload


def _should_collapse_chat_paste(text: str) -> bool:
    """Return whether pasted chat text should be collapsed."""
    return detect_mode_prefix(text) is None and should_collapse_paste(text)


_PASTE_COLLAPSED_TOAST = "Large paste collapsed. Paste again to expand."
"""Toast shown when a paste collapses into a `[Pasted text #N]` placeholder.

Emitted only for a new collapse, not when a repeat paste expands an existing
placeholder back to full text.
"""


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
        display_label = self._label.removeprefix("/")
        if self._description:
            content = Content.from_markup(
                "[bold]$label[/bold]  [dim]$desc[/dim]",
                label=display_label,
                desc=self._description,
            )
        else:
            content = Content.from_markup("[bold]$label[/bold]", label=display_label)

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


InputAction = Literal["clear", "copy"]
"""Closed set of actions an `InputActionButton` can dispatch."""


class InputActionButton(Static):
    """Small clickable button shown at the right edge of the chat input row.

    Provides discoverable mouse alternatives to keyboard shortcuts for
    clearing (`[ X ]`) and copying (`[ COPY ]`) the current draft.
    """

    DEFAULT_CSS = """
    InputActionButton {
        height: 1;
        margin: 0 0 0 1;
        text-style: bold;
    }

    InputActionButton.input-action-clear {
        width: 5;
        color: $error;
    }

    InputActionButton.input-action-copy {
        width: 8;
        color: $primary;
    }

    InputActionButton.input-action-clear:hover {
        background: $error;
        color: auto;
    }

    InputActionButton.input-action-copy:hover {
        background: $primary;
        color: auto;
    }
    """

    class Clicked(Message):
        """Message sent when an input action button is clicked."""

        def __init__(self, action: InputAction) -> None:
            """Initialize with the action identifier (`clear` or `copy`)."""
            super().__init__()
            self.action = action

    @property
    def allow_select(self) -> bool:
        """Disable terminal text selection for the action label."""
        return False

    def __init__(self, label: str, action: InputAction, **kwargs: Any) -> None:
        """Initialize the button with a label and an action identifier."""
        super().__init__(label, markup=False, **kwargs)
        self._action = action

    def on_click(self, event: Click) -> None:
        """Relay the click as a typed `Clicked` message."""
        event.stop()
        self.post_message(self.Clicked(self._action))


class CompletionPopup(VerticalScroll):
    """Popup widget that displays completion suggestions as clickable options."""

    DEFAULT_CSS = f"""
    CompletionPopup {{
        display: none;
        height: auto;
        max-height: {_COMPLETION_POPUP_MAX_HEIGHT};
    }}
    """

    class RowsChanged(Message):
        """Message sent when the popup's visible row count changes.

        Deduplicated by the sender, so receiving this always means the rendered
        row count actually moved.
        """

        def __init__(self, rows: int) -> None:
            """Initialize with the visible row count (0 when hidden)."""
            super().__init__()
            self.rows = rows

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
        self._reported_rows = 0
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
        # show() is still deferred to _rebuild_options to avoid stale content,
        # but the rebuild runs before the next paint so prompt and popup changes
        # appear in the same frame.
        self.call_next(lambda: self._rebuild_options(gen))

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

        # The DOM mutations above can await, during which a hide() (or a newer
        # rebuild) bumps the generation to cancel this one. The top-of-function
        # guard ran before that await, so re-check here: without it a stale
        # rebuild would re-show a popup that was dismissed mid-flight (e.g. when
        # a completion is applied and the popup hidden in the same key press).
        if generation != self._rebuild_generation:
            return

        self.show(len(self._options))

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

    def _report_rows(self, rows: int) -> None:
        """Announce the popup's rendered row count when it changes."""
        if self._reported_rows != rows:
            self._reported_rows = rows
            self.post_message(self.RowsChanged(rows))

    def hide(self) -> None:
        """Hide the popup."""
        self._pending_suggestions = []
        self._rebuild_generation += 1  # Cancel any in-flight rebuild
        self.styles.display = "none"  # ty: ignore[invalid-assignment]  # Textual accepts string display values at runtime
        self._report_rows(0)

    def show(self, suggestion_count: int) -> None:
        """Show the popup and report how many rows it will render.

        Args:
            suggestion_count: Number of suggestions about to be displayed. Taken
                as an argument rather than read from internal state so the
                reported height is tied to the caller's list, which is the thing
                `ChatInputBox` must reserve space for.
        """
        self.styles.display = "block"
        self._report_rows(min(suggestion_count, _COMPLETION_POPUP_MAX_HEIGHT))


class ChatTextArea(PasteBurstTextArea):
    """TextArea subclass with custom key handling for chat input.

    Modifier-Enter / Ctrl+J newline bindings and the VSCode backslash+enter
    fallback are inherited from `PasteBurstTextArea`.
    """

    _skip_history_change_events: int
    """Counter incremented before a history-driven text replacement so the
    resulting `TextArea.Changed` event (which fires on the next message-loop
    iteration) can be suppressed.  `ChatInput.on_text_area_changed` decrements
    the counter.
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

    class Typing(Message):
        """Posted when the user presses a printable key or backspace.

        Relayed by `ChatInput` as `ChatInput.Typing` for the app to track
        typing activity.
        """

    argument_hint: reactive[str] = reactive("")
    """Inline slash-command argument hint rendered at the end of the line."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the chat text area."""
        # Remove placeholder if passed, TextArea doesn't support it the same way
        kwargs.pop("placeholder", None)
        super().__init__(**kwargs)
        self._chat_input_owner: ChatInput | None = None
        self._skip_history_change_events = 0
        self._completion_active = False
        self._burst_payload_keeps_leading_slash = False
        # Paste-burst and backslash-pending state is initialized by
        # PasteBurstTextArea.__init__.
        # Tracks terminal focus so a click that re-focuses the window only
        # restores focus instead of also moving the cursor. See
        # `_REFOCUS_CLICK_SUPPRESS_WINDOW_SECONDS`.
        self._app_blurred = False
        self._refocus_time: float | None = None
        self._shell_highlighting = False
        self._highlighted_source = ""
        self._highlighted_lines: list[Content] | None = None

    def set_shell_highlighting(self, *, enabled: bool) -> None:
        """Enable or disable shell syntax highlighting for this text area.

        Args:
            enabled: Whether to style input as shell syntax.
        """
        if self._shell_highlighting == enabled:
            return
        self._shell_highlighting = enabled
        self._highlighted_source = ""
        self._highlighted_lines = None
        self._line_cache.clear()
        self.refresh()

    def on_focus(self) -> None:
        """Keep an open prompt search focused when the composer is clicked."""
        owner = self._chat_input_owner
        if owner is not None and owner._prompt_search_active:
            owner.focus_input()

    def _render_line(self, y: int) -> Strip:
        """Render a line, keeping shell token colors visible on the cursor line.

        `TextArea._render_line` stylizes the whole cursor line with
        `theme.cursor_line_style`, which under the default `css` theme resolves
        to a style carrying the widget's text color. That foreground is painted
        after (and on top of) the shell syntax spans produced by `get_line`,
        flattening every token on the cursor line to a single color. Clear just
        the foreground while shell highlighting is active so the cursor-line
        background tint - and any text styles such as bold - still apply over
        the token colors.

        Mutating the theme in place is safe because `TextArea._set_theme` keeps
        a per-widget copy (`dataclasses.replace`), so this never touches the
        shared builtin theme or another `TextArea`. Rendering is synchronous
        and `apply_css` runs outside this window, so the `finally` restore is
        sufficient.

        Args:
            y: Y coordinate of the line relative to the widget region.

        Returns:
            The rendered line.
        """
        theme = self._theme
        cursor_line_style = theme.cursor_line_style if theme else None
        if (
            not self._shell_highlighting
            or cursor_line_style is None
            or cursor_line_style.color is None
        ):
            return super()._render_line(y)

        # Restore on the exception path too; otherwise the widget keeps a
        # foreground-less cursor line for the rest of the session.
        theme.cursor_line_style = cursor_line_style.without_color + Style(
            bgcolor=cursor_line_style.bgcolor
        )
        try:
            return super()._render_line(y)
        finally:
            theme.cursor_line_style = cursor_line_style

    def get_line(self, line_index: int) -> Text:
        """Return one input line, with shell syntax styles when enabled.

        Args:
            line_index: Index of the line to return.

        Returns:
            The line as Rich text, styled per shell token when highlighting is
            active. Falls back to the unstyled base implementation if
            highlighting fails, so the text shown always matches the document.
        """
        if not self._shell_highlighting:
            return super().get_line(line_index)

        source = self.text
        lines = self._highlighted_lines
        if lines is None or source != self._highlighted_source:
            language = "batch" if sys.platform == "win32" else "sh"
            try:
                # `tab_size=1` keeps span offsets aligned with the document's
                # raw character offsets: Pygments expands tabs (default 8),
                # while `_render_line` does its own tab expansion downstream.
                highlighted = highlight(source, language=language, tab_size=1)
                lines = list(highlighted.split("\n", allow_blank=True))
            except Exception:
                # This runs inside the render loop, where an uncaught exception
                # tears down the whole app. Degrade to unhighlighted text and
                # stop retrying every frame.
                logger.exception(
                    "Shell highlighting failed for a %d-character draft; "
                    "falling back to unhighlighted text",
                    len(source),
                )
                self._shell_highlighting = False
                self._highlighted_source = ""
                self._highlighted_lines = None
                return super().get_line(line_index)
            # `highlight()` normalizes via `"\n".join(code.splitlines())`, which
            # drops the trailing empty line that `Document` keeps. Pad so line
            # indices stay in step with the document.
            lines.extend([Content("")] * max(0, self.document.line_count - len(lines)))
            # Only commit the cache marker once both fallible steps succeeded;
            # advancing it earlier would serve the previous draft's lines
            # forever on the cache-hit path.
            self._highlighted_source = source
            self._highlighted_lines = lines

        if not 0 <= line_index < len(lines):
            logger.warning(
                "Shell highlight covers %d lines, not line %d (document has "
                "%d); rendering it unhighlighted",
                len(lines),
                line_index,
                self.document.line_count,
            )
            return super().get_line(line_index)

        line = Text(end="", no_wrap=True)
        for segment in lines[line_index].render_segments(self.visual_style):
            line.append(segment.text, segment.style)
        return line

    def render_line(self, y: int) -> Strip:
        """Render a single line, appending any argument hint at line end.

        The built-in `TextArea.suggestion` renders at the cursor position,
        but slash-command argument hints should stay attached to the end of the
        command text regardless of cursor movement.

        Args:
            y: Y Coordinate of line relative to the widget region.

        Returns:
            A rendered line.
        """
        strip = super().render_line(y)
        if not self._should_render_argument_hint():
            return strip

        line_info = self._get_visual_line_info(y)
        if line_info is None:
            return strip

        line_index, section_offset = line_info
        if not self._is_argument_hint_section(line_index, section_offset):
            return strip

        content_cells = self._get_section_cell_length(line_index, section_offset)
        if content_cells >= strip.cell_length:
            return strip

        prefix = strip.crop(0, content_cells)
        suffix = strip.crop(content_cells, strip.cell_length)
        suffix_width = suffix.cell_length
        cursor_on_hint = self._cursor_at_argument_hint_anchor(line_index)
        if cursor_on_hint and suffix_width > 0:
            suffix = suffix.crop(1, suffix.cell_length)

        hint_strip = self._build_argument_hint_strip(cursor_on_hint=cursor_on_hint)
        tail = Strip.join([hint_strip, suffix]).crop(0, suffix_width)
        return Strip.join([prefix, tail])

    def _should_render_argument_hint(self) -> bool:
        """Return whether the inline argument hint should be rendered."""
        return bool(
            self.argument_hint and (self.has_focus or not self.hide_suggestion_on_blur)
        )

    def _get_visual_line_info(self, y: int) -> tuple[int, int] | None:
        """Map a widget-relative y coordinate to wrapped line metadata.

        Returns:
            Tuple of `(line_index, section_offset)` for the wrapped line at `y`,
            otherwise `None` when `y` is outside the wrapped document.
        """
        _scroll_x, scroll_y = self.scroll_offset
        absolute_y = scroll_y + y
        # Private Textual API (verified against textual 3.x); revisit on
        # major Textual upgrades.
        try:
            offset_map = self.wrapped_document._offset_to_line_info
        except AttributeError:
            logger.warning(
                "WrappedDocument._offset_to_line_info not found; "
                "argument hint rendering disabled (Textual API change?)"
            )
            return None
        if absolute_y < 0 or absolute_y >= len(offset_map):
            return None
        entry = offset_map[absolute_y]
        expected_length = 2  # (line_index, section_offset)
        if not isinstance(entry, tuple) or len(entry) != expected_length:
            logger.warning("Unexpected offset_map entry: %r", entry)
            return None
        return entry

    def _is_argument_hint_section(self, line_index: int, section_offset: int) -> bool:
        """Return whether a wrapped section owns the end-of-line hint."""
        if line_index != self.document.line_count - 1:
            return False
        return section_offset == len(self.wrapped_document.get_offsets(line_index))

    def _get_section_cell_length(self, line_index: int, section_offset: int) -> int:
        """Return the rendered cell width of a wrapped text section."""
        wrapped_sections = self.wrapped_document.get_sections(line_index)
        if section_offset < 0 or section_offset >= len(wrapped_sections):
            return 0
        section_text = wrapped_sections[section_offset].expandtabs(self.indent_width)
        return cell_len(section_text)

    def _cursor_at_argument_hint_anchor(self, line_index: int) -> bool:
        """Return whether the cursor currently sits on the hint anchor."""
        if not self._draw_cursor or not self.show_cursor or not self.has_focus:
            return False
        cursor_row, cursor_column = self.selection.end
        if cursor_row != line_index:
            return False
        return cursor_column == len(self.document.get_line(line_index))

    def _build_argument_hint_strip(self, *, cursor_on_hint: bool) -> Strip:
        """Build a strip for the current argument hint text.

        Returns:
            A `Strip` containing the current argument hint, with cursor styling
            applied to the first hint character when the cursor sits on the
            hint anchor.
        """
        hint = self.argument_hint
        hint_style = self.get_component_rich_style("text-area--suggestion")
        if not cursor_on_hint or not hint:
            return Strip([Segment(hint, hint_style)], cell_length=cell_len(hint))

        ta_theme = self._theme
        cursor_style = ta_theme.cursor_style if ta_theme else None
        first_style = hint_style if cursor_style is None else hint_style + cursor_style
        segments = [Segment(hint[0], first_style)]
        if len(hint) > 1:
            segments.append(Segment(hint[1:], hint_style))
        return Strip(segments, cell_length=cell_len(hint))

    def scroll_cursor_visible(
        self, center: bool = False, animate: bool = False
    ) -> Offset:
        """Scroll to make the cursor visible, guarding against cursor/document desync.

        Textual's `WrappedDocument.location_to_offset` has an off-by-one in its
        line-index clamp (`len(...)` instead of `len(...) - 1`). When a reactive
        watcher (e.g. `_watch_show_vertical_scrollbar`) fires between a document
        replacement and cursor update, the stale cursor location triggers a
        `ValueError`. Guard here since `scroll_cursor_visible` is the sole
        caller of `_recompute_cursor_offset`.

        Args:
            center: Whether the cursor should be scrolled to the center.
            animate: Whether to animate while scrolling.

        Returns:
            The scroll offset applied, or `Offset(0, 0)` on desync.
        """
        try:
            return super().scroll_cursor_visible(center=center, animate=animate)
        except (
            ValueError
        ):  # WrappedDocument.get_offsets off-by-one clamp in location_to_offset
            logger.warning(
                "Cursor/document desync in scroll_cursor_visible "
                "(cursor=%s, doc_lines=%d); skipping scroll",
                self.cursor_location,
                self.document.line_count,
            )
            return Offset(0, 0)

    def set_app_focus(self, *, has_focus: bool) -> None:
        """Set whether the app should show the cursor as active.

        Args:
            has_focus: Whether the app input should be focused.
        """
        self._backslash_pending_time = None
        if has_focus and not self.has_focus:
            self.call_after_refresh(self.focus)

    def _notify_app_blur(self) -> None:
        """Record that the terminal window lost OS focus."""
        self._app_blurred = True

    def _notify_app_focus(self) -> None:
        """Record that the terminal window regained OS focus via a focus event.

        Stamps the regain time so the click that re-focused the window (which
        arrives just after the focus event) can be treated as focus-only.
        """
        if self._app_blurred:
            self._refocus_time = time.monotonic()
            self._app_blurred = False

    def _consume_refocus_click(self) -> bool:
        """Return whether the current mouse-down only re-focuses the window.

        `_refocus_time` is only cleared here, so a focus regain that is never
        followed by a text-area click leaves the stamp set. The gap check
        bounds that staleness: an old stamp exceeds the window and returns
        `False` (clearing it), so a much later click is never suppressed.
        """
        refocus_time = self._refocus_time
        if refocus_time is None:
            return False
        self._refocus_time = None
        gap = time.monotonic() - refocus_time
        return gap <= _REFOCUS_CLICK_SUPPRESS_WINDOW_SECONDS

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        """Position the cursor on click, except when the click re-focuses the app.

        A mouse-down landing within a short window after a terminal focus
        regain only restores focus and leaves the cursor where it was.

        Deliberately shadows Textual's private `TextArea._on_mouse_down` to gate
        cursor positioning; verified against Textual 8.2.8. Re-verify on every
        Textual bump that early-returning before `super()` still leaves no
        selection/capture state set.
        """
        if self._consume_refocus_click():
            event.stop()
            event.prevent_default()
            return
        await super()._on_mouse_down(event)

    def set_completion_active(self, *, active: bool) -> None:
        """Set whether completion suggestions are visible."""
        self._completion_active = active

    def action_insert_newline(self) -> None:
        """Insert a newline character."""
        self.insert("\n")
        # TextArea's built-in cursor-visible scroll runs before the widget
        # reflows for the new row, so it sees stale dimensions and is a no-op
        # when the cursor would land below `max-height`. Re-issue after
        # refresh so it stays in view.
        self.call_after_refresh(self.scroll_cursor_visible)

    def _refresh_scrollbars(self) -> None:
        """Refresh scrollbars without flashing a transient vertical bar.

        `TextArea` grows its `virtual_size` height the moment a row is inserted,
        a frame before this `height: auto` widget's container reflows to match.
        The base `_refresh_scrollbars` decides vertical visibility by comparing
        `virtual_size.height` against the stale `self._container_size.height`,
        so for that one frame the freshly inserted row looks like overflow and
        the scrollbar flashes on, then off once the container catches up.

        The widget only ever truly overflows once its content exceeds the height
        it settles at — its resolved `max-height` (the layout chain above it is
        all `height: auto`, so it always grows to `min(content, max-height)`).
        Feed the base method that settled height instead of the mid-reflow one,
        so the bar appears only on genuine overflow and never flashes. All other
        base behavior (horizontal bar, anti-oscillation, scroll updates) is left
        untouched.

        Deliberately overrides Textual's private `_refresh_scrollbars` and
        swaps the private `_container_size`; verified against Textual 8.2.8.
        Re-verify these attribute names on every Textual bump.
        """
        bound = self._settled_content_height()
        if bound is None:
            super()._refresh_scrollbars()
            return

        original = self._container_size
        # Never report a viewport smaller than the settled height; `max(...)`
        # also guards the unlikely case where the real container is already
        # larger than the bound, so we only ever raise the comparison height.
        corrected_height = max(original.height, min(self.virtual_size.height, bound))
        if corrected_height == original.height:
            super()._refresh_scrollbars()
            return

        self._container_size = Size(original.width, corrected_height)
        try:
            super()._refresh_scrollbars()
        finally:
            self._container_size = original

    def _settled_content_height(self) -> int | None:
        """Return the content-row height this widget settles at, if knowable.

        Returns `None` (so the caller defers to the base behavior) unless the
        vertical overflow is `auto` and `max-height` resolves to a fixed cell
        count, the only case where the flash-suppression bound is well-defined.
        """
        styles = self.styles
        if styles.overflow_y != "auto" or not styles.has_rule("max_height"):
            return None
        max_height = styles.max_height
        cells = max_height.cells if max_height is not None else None
        if cells is None:
            return None
        # box-sizing is border-box by default, so subtract border/padding to get
        # the content-row count the base method compares `virtual_size` against.
        return max(1, cells - self.gutter.height)

    def _cursor_at_visual_top(self) -> bool:
        """Return whether the cursor cannot move up further."""
        try:
            return self.get_cursor_up_location() == self.cursor_location
        except ValueError:
            # `WrappedDocument.location_to_offset` can raise during a brief
            # text/cursor desync window (see `scroll_cursor_visible` guard).
            # Treat as "not at top" so TextArea moves the cursor instead of
            # firing history navigation on a transient state.
            return False

    def _cursor_at_visual_bottom(self) -> bool:
        """Return whether the cursor cannot move down further."""
        try:
            return self.get_cursor_down_location() == self.cursor_location
        except ValueError:
            return False

    def action_cursor_up(self, select: bool = False) -> None:
        """Move cursor up, or navigate to the previous history entry at top.

        When `select` is true or a selection is active, falls through to
        TextArea's default so shift+up extends selection rather than
        triggering navigation. History fires only when moving up cannot
        advance the cursor — handled via the wrapped-document navigator so
        soft-wrap is respected.
        """
        if not select and self.selection.is_empty and self._cursor_at_visual_top():
            self.post_message(self.HistoryPrevious(self.text))
            return
        super().action_cursor_up(select)

    def action_cursor_down(self, select: bool = False) -> None:
        """Move cursor down, or navigate to the next history entry at bottom.

        Mirrors `action_cursor_up`: defers to TextArea on selection or when
        the cursor still has somewhere to move; otherwise fires history.
        """
        if not select and self.selection.is_empty and self._cursor_at_visual_bottom():
            self.post_message(self.HistoryNext())
            return
        super().action_cursor_down(select)

    def _in_slash_command_context(self) -> bool:
        """Return whether the current input is composing a slash command."""
        owner = self._chat_input_owner
        if owner is not None and owner.mode == "command":
            return True
        return self.text.startswith("/")

    def _paste_collapse_enabled(self) -> bool:
        """Return whether large pastes should be collapsed into placeholders.

        Reads the owning `ChatInput`'s resolved preference, defaulting to
        enabled when the owner is not yet attached.
        """
        owner = self._chat_input_owner
        return owner is None or owner._collapse_pastes

    async def _dispatch_burst_payload(self, payload: str) -> None:
        """Route a flushed burst through dropped-path and large-paste checks.

        Routed payloads are applied through the owner synchronously rather than
        posted to it, so the payload is in the document before this returns. A
        posted message lands at the tail of this widget's queue, behind any
        keystroke the terminal has already delivered, which would insert that
        character ahead of the paste.

        When parsing fails, or there is no owner to route through, the buffered
        text is inserted unchanged so regular typing behavior is preserved.
        """
        from deepagents_code.input import parse_pasted_path_payload

        keeps_leading_slash = self._burst_payload_keeps_leading_slash
        self._burst_payload_keeps_leading_slash = False
        owner = self._chat_input_owner
        if owner is not None:
            # Cleared up front so the verbatim-insert path below cannot leave a
            # previous payload's answer standing for
            # `_payload_supplied_trailing_space`.
            owner._paste_appended_trailing_space = False

        try:
            parsed = await asyncio.to_thread(parse_pasted_path_payload, payload)
        except Exception:
            # The parser guards its own filesystem probes, but
            # `_resolve_with_unicode_space_variants` calls `expanduser()` and
            # `Path.cwd()` unguarded, so a deleted working directory or an
            # unresolvable home still surfaces here.  Leave a breadcrumb (the
            # message never carries the paste content) instead of swallowing it,
            # then fall through to normal text handling.  Logged at warning (not
            # debug) so it actually surfaces in production.
            logger.warning(
                "Path-payload parsing failed; treating burst as text",
                exc_info=True,
            )
            parsed = None
        if owner is not None:
            if parsed is not None:
                applied = owner.apply_paste_payload(payload, parsed.paths)
            elif self._paste_collapse_enabled() and _should_collapse_chat_paste(
                payload
            ):
                applied = owner.apply_paste_payload(payload, None)
            else:
                applied = False
            if applied:
                return

        if keeps_leading_slash and owner is not None:
            # Consumed by the change handler this insert triggers, suppressing the
            # mode re-detection that would otherwise strip the restored `/`.
            owner.suppress_next_prefix_detection()
        self.insert(payload)
        # A multi-line payload adds rows the same way `action_insert_newline`
        # does, and needs the same post-refresh scroll for the same reason: the
        # built-in scroll sees stale dimensions and leaves the cursor off screen.
        if "\n" in payload:
            self.call_after_refresh(self.scroll_cursor_visible)

    def _burst_run_payload_for_dispatch(self, payload: str) -> str:
        """Restore a virtual command prefix when a burst is an absolute path.

        A `/` typed at offset 0 switches the input into command mode and is never
        inserted, so a dropped absolute path replayed as key events loses its
        leading separator. Restoring it lets the run be recognized as a path.

        The restore is deliberately narrow, because a payload rewritten here is
        also what takes the input *out* of command mode
        (`_on_burst_run_promoted`). Asking `looks_like_dropped_payload` about the
        `/`-prefixed candidate cannot decide this — that function is a leading-
        token check, so prepending `/` makes it vacuously true for any text. Three
        conditions stand in for it instead:

        - The run must start at document offset 0, i.e. it is the text that
          directly followed the consumed `/` rather than a later burst.
        - The payload must contain its own separator, so `help` stays a command
          name while `private/tmp/x` reads as a path tail.
        - Nothing before that separator may be whitespace, which keeps a command
          with a path argument (`read src/main.py`) from qualifying.

        Returns:
            The payload with the consumed leading slash restored, or the payload
            unchanged when it does not look like the tail of an absolute path.
        """
        owner = self._chat_input_owner
        if owner is None or owner.mode != "command":
            return payload
        cursor_offset = self.document.get_index_from_location(self.cursor_location)  # ty: ignore[unresolved-attribute]  # Document has this method; DocumentBase stub is narrower
        if cursor_offset != len(payload) or not self.text.startswith(payload):
            return payload
        head, separator, _ = payload.partition("/")
        if not separator or any(char.isspace() for char in head):
            return payload
        # Exactly one `/` was consumed, so exactly one is restored: `lstrip("/")`
        # would eat a second leading slash and silently drop a character from a
        # `//host/share` payload.
        return f"/{payload}"

    def _on_burst_run_promoted(
        self, visible_payload: str, dispatch_payload: str
    ) -> None:
        """Leave command mode when promotion recovered a leading path slash."""
        if visible_payload == dispatch_payload:
            return
        # The payload now leads with the restored `/`, so re-inserting it at
        # offset 0 would trip mode-prefix detection a second time and strip the
        # slash again — losing the character for good on the paths that do not
        # resolve on disk. Flag it so the insert suppresses that detection.
        self._burst_payload_keeps_leading_slash = True
        owner = self._chat_input_owner
        if owner is not None and owner.mode == "command":
            owner.mode = "normal"

    def _reset_paste_burst_state(self) -> None:
        """Reset burst tracking, including the restored-slash flag.

        The flag describes the buffered payload that `super()` is about to
        discard, so it must not outlive it: a stale `True` would suppress the next
        burst's legitimate mode re-detection.
        """
        self._burst_payload_keeps_leading_slash = False
        super()._reset_paste_burst_state()

    def _payload_supplied_trailing_space(self) -> bool:
        """Return whether the flush appended a trailing space of its own.

        An attached dropped-path payload gets a trailing space from
        `_build_path_replacement`; inserting the pending space as well would
        double it. The question is what the flush *did*, not what the document
        happens to end with — a verbatim payload that merely ends in a space
        would otherwise swallow the user's real keystroke.
        """
        owner = self._chat_input_owner
        return owner is not None and owner._paste_appended_trailing_space

    async def _on_key(self, event: events.Key) -> None:
        """Handle key events."""
        # Lock keys (Caps Lock, Num Lock, Scroll Lock) must never type. The
        # kitty parser patch in `_textual_patches.py` already neutralizes these
        # at the source; this is defense-in-depth in case a lock key still
        # arrives with associated text (e.g. if that patch failed to install or
        # a future terminal bypasses it). Note this only shields the chat input
        # — if the parser patch silently no-ops, other widgets stay broken. The
        # key may carry modifier prefixes (e.g. 'ctrl+caps_lock'), so match on
        # the final '+'-delimited token.
        if event.key.rsplit("+", 1)[-1] in _LOCK_KEYS:
            event.prevent_default()
            event.stop()
            return

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
            # This branch bypasses the burst helpers below, so it has to drive
            # them itself: a space inside a replayed paste must reach the buffer
            # (if one is live) or the run tracker (if not), otherwise the
            # tracker's text diverges from the document and the run is discarded,
            # losing grouping for that stretch of the paste.
            space_now = time.monotonic()
            if self._paste_burst_buffer and self._append_recent_paste_burst_text(
                " ", space_now
            ):
                self.post_message(self.Typing())
                return
            # The burst (if any) had gone idle, so this space follows the paste
            # rather than belonging to it. Flushing applies the payload before it
            # returns, so the space inserted next lands after it.
            if self._paste_burst_buffer:
                await self._flush_paste_burst()
                if self._payload_supplied_trailing_space():
                    self.post_message(self.Typing())
                    return
            self.insert(" ")
            self._note_printable_burst_keystroke(" ", space_now)
            self.post_message(self.Typing())
            # The space is in the document, so the run may now qualify — a path
            # payload can end on a space, and a long single-line paste can cross
            # the length threshold here.
            self._check_burst_run_for_promotion()
            return

        now = time.monotonic()

        # Signal typing activity for printable keys and backspace so the app
        # can defer approval widgets while the user is actively editing.
        if event.is_printable or event.key == "backspace":
            self.post_message(self.Typing())

        # While the inline prompt search is open, printable keys feed the
        # panel's query -- the focused `PromptSearchInput` consumes them and
        # `ChatInput.on_input_changed` re-filters -- so they must never be
        # mistaken for paste-burst replay.
        prompt_search_active = (
            self._chat_input_owner is not None
            and self._chat_input_owner._prompt_search_active
        )

        if not prompt_search_active and await self._absorb_key_into_burst(event, now):
            event.prevent_default()
            event.stop()
            return

        # Track rapid keystroke runs so terminals without bracketed paste keep
        # embedded newlines grouped without delaying ordinary text insertion.
        if not prompt_search_active:
            self._track_burst_run(event, now)

        # A mode trigger (`!`, `!!`, `/`) typed at the very start of an
        # unselected input switches modes. Handle it before TextArea inserts the
        # character so the trigger never flashes on screen for a frame before
        # the change handler would strip it.
        if (
            event.is_printable
            and event.character is not None
            and self.cursor_location == (0, 0)
            and self.selection.is_empty
            and self._chat_input_owner is not None
            and self._chat_input_owner.handle_mode_prefix_keystroke(event.character)
        ):
            event.prevent_default()
            event.stop()
            # `_track_burst_run` above already counted this character, but it is
            # consumed as a mode switch rather than inserted. Drop the run so the
            # tracker does not claim a character the document never received.
            self._reset_paste_burst_run()
            return

        # Some terminals (e.g. VSCode built-in) send a literal backslash
        # followed by enter for shift+enter.  When enter arrives shortly
        # after a backslash, delete the backslash and insert a newline.  The
        # fallback is inactive while completion is active.
        if self._consume_backslash_enter_newline(
            event, now, enabled=not self._completion_active
        ):
            return

        self._track_backslash_pending(event, now)

        # Modifier+Enter inserts newline — keys derived from BINDINGS
        if self._consume_modifier_newline(event):
            return

        if event.key == "backspace" and self._delete_placeholder_token(backwards=True):
            event.prevent_default()
            event.stop()
            return

        # While the inline prompt search is open, the panel owns the keyboard.
        # Route the key to the search handler before TextArea defaults (Enter
        # newline, printable insertion) consume it; the draft stays frozen for
        # the whole session.
        if prompt_search_active:
            if (
                self._chat_input_owner is not None
                and self._chat_input_owner._handle_prompt_search_key(event)
            ):
                return
            # Keys the search ignores must not edit the draft either.
            event.prevent_default()
            event.stop()
            return

        # If completion is active, let parent handle navigation keys.
        # Space is included so that slash-command completion can accept the
        # selected suggestion via the same code path as Tab (avoiding a
        # frame-lag between the popup hiding and the argument hint appearing).
        # When the active controller ignores the space (e.g. file completion),
        # ChatInput.on_key inserts it manually.
        if self._completion_active and event.key in {
            "up",
            "down",
            "tab",
            "enter",
            "space",
        }:
            # Prevent TextArea's default behavior (e.g., Enter inserting newline)
            # but let event bubble to ChatInput for completion handling
            event.prevent_default()
            # `space` is the one printable key here, so `_track_burst_run` above
            # counted it while this branch inserts nothing. Drop the run so the
            # tracker does not claim a character the document never received —
            # the same reason as the mode-prefix branch above.
            if event.is_printable:
                self._reset_paste_burst_run()
            return

        # Plain Enter submits, unless a recent keystroke burst suggests this
        # newline is part of a paste replayed as key events. In that case the
        # visible run is pulled off screen into the paste buffer along with this
        # newline, and the window is kept alive so the rest of the paste stays
        # grouped instead of submitting mid-stream. The text reappears when the
        # burst flushes — possibly as a `[Pasted text #N]` placeholder.
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            if not prompt_search_active and self._consume_enter_as_burst_newline(now):
                return
            if (
                self._chat_input_owner is not None
                and self._chat_input_owner._handle_stale_slash_enter()
            ):
                return
            value = self.text.strip()
            if value:
                self.post_message(self.Submitted(value))
            return

        await super()._on_key(event)

        # Must follow `super()._on_key`: promotion verifies the run against the
        # document, so the current character has to be in it already.
        if not prompt_search_active:
            self._check_burst_run_for_promotion()

    def action_delete_right(self) -> None:
        """Delete a bound placeholder atomically or the next character."""
        if not self._delete_placeholder_token(backwards=False):
            super().action_delete_right()

    def action_delete_word_left(self) -> None:
        """Delete a bound placeholder atomically or the previous word."""
        if not self._delete_placeholder_token(backwards=True):
            super().action_delete_word_left()

    def _delete_placeholder_token(self, *, backwards: bool) -> bool:
        """Delete a full placeholder token (image, video, or paste) in one keypress.

        Args:
            backwards: Whether the delete action is backwards (`backspace`) or
                forwards (`delete`).

        Returns:
            `True` when a placeholder token was deleted.
        """
        if not self.text or not self.selection.is_empty:
            return False

        cursor_offset = self.document.get_index_from_location(self.cursor_location)  # ty: ignore[unresolved-attribute]  # Document has this method; DocumentBase stub is narrower
        span = self._find_placeholder_span(cursor_offset, backwards=backwards)
        if span is None:
            return False

        start, end = span
        start_location = self.document.get_location_from_index(start)  # ty: ignore[unresolved-attribute]  # Document has this method; DocumentBase stub is narrower
        end_location = self.document.get_location_from_index(end)  # ty: ignore[unresolved-attribute]
        self.delete(start_location, end_location)
        self.move_cursor(start_location)
        return True

    def _bound_media_placeholders(self) -> set[str]:
        """Return placeholder tokens bound to currently tracked media.

        Returns:
            The set of `[image N]`/`[video N]` tokens for media the tracker is
                actually holding. Empty when there is no owner/tracker.
        """
        owner = self._chat_input_owner
        tracker = owner._image_tracker if owner is not None else None
        if tracker is None:
            return set()
        placeholders = {img.placeholder for img in tracker.images}
        placeholders.update(video.placeholder for video in tracker.videos)
        return placeholders

    def _bound_paste_ids(self) -> set[int]:
        """Return paste ids that have backing content in the owner.

        Returns:
            The set of paste ids present in `ChatInput._pasted_contents`. Empty
                when there is no owner.
        """
        owner = self._chat_input_owner
        if owner is None:
            return set()
        return set(owner._pasted_contents)

    def _find_placeholder_span(
        self, cursor_offset: int, *, backwards: bool
    ) -> tuple[int, int] | None:
        """Return placeholder span to delete for current cursor and key direction.

        Covers image, video, and collapsed-paste placeholders so each deletes as
        a single atomic token.  Paste placeholders carry backing content in
        `ChatInput._pasted_contents`; that map is intentionally left untouched
        here so an undo can restore the token with its content (it is cleared
        only at submit).

        Only tokens bound to real attachments are treated as atomic: image/video
        placeholders must correspond to a tracked media item and paste
        placeholders to an entry in `ChatInput._pasted_contents`. Placeholder-
        shaped text the user typed by hand (e.g. literally typing ``[image 2]``)
        is left as ordinary text and edits character by character.

        Args:
            cursor_offset: Character offset of the cursor from the start of text.
            backwards: Whether the delete action is backwards (backspace) or
                forwards (delete).

        Returns:
            The `(start, end)` character span of the placeholder to delete, or
                `None` when the cursor is not adjacent to a bound placeholder
                token.
        """
        text = self.text
        media_placeholders = self._bound_media_placeholders()
        pasted_ids = self._bound_paste_ids()
        for pattern in (
            IMAGE_PLACEHOLDER_PATTERN,
            VIDEO_PLACEHOLDER_PATTERN,
            PASTE_PLACEHOLDER_PATTERN,
        ):
            for match in pattern.finditer(text):
                if pattern is PASTE_PLACEHOLDER_PATTERN:
                    if int(match.group(1)) not in pasted_ids:
                        continue
                elif match.group(0) not in media_placeholders:
                    continue
                start, end = match.span()
                if backwards:
                    # Cursor is inside token or right after a trailing space inserted
                    # with the token.
                    if start < cursor_offset <= end:
                        return start, end
                    if cursor_offset > 0:
                        previous_index = cursor_offset - 1
                        # Swallow trailing whitespace with the token, except for
                        # a newline: backspacing a line break should rejoin the
                        # lines without deleting the placeholder.
                        if (
                            previous_index < len(text)
                            and previous_index == end
                            and text[previous_index].isspace()
                            and text[previous_index] != "\n"
                        ):
                            return start, cursor_offset
                elif start <= cursor_offset < end:
                    return start, end
        return None

    def replace_placeholder_with_text(self, paste_id: int, content: str) -> bool:
        """Replace a `[Pasted text #id]` placeholder with full text in place.

        Used when the same content is pasted again: the compact placeholder is
        expanded back to the original text where it sits, preserving surrounding
        input.

        Args:
            paste_id: The paste id whose placeholder should be expanded.
            content: The full text to insert where the placeholder was.

        Returns:
            `True` when a matching placeholder was found and replaced.
        """
        for match in PASTE_PLACEHOLDER_PATTERN.finditer(self.text):
            if int(match.group(1)) != paste_id:
                continue
            start, end = match.span()
            start_location = self.document.get_location_from_index(start)  # ty: ignore[unresolved-attribute]  # Document has this method; DocumentBase stub is narrower
            end_location = self.document.get_location_from_index(end)  # ty: ignore[unresolved-attribute]
            self.delete(start_location, end_location)
            self.insert(content, start_location)
            return True
        return False

    async def _on_paste(self, event: events.Paste) -> None:
        """Handle paste events, detecting file paths and large pastes."""
        self._backslash_pending_time = None
        if self._paste_burst_buffer:
            await self._flush_paste_burst()

        from deepagents_code.input import parse_pasted_path_payload

        try:
            parsed = await asyncio.to_thread(parse_pasted_path_payload, event.text)
        except Exception:
            # See _flush_paste_burst: swallowing here would silently break the
            # drag-drop-file path, so log a breadcrumb and fall through to text.
            logger.debug(
                "Path-payload parsing failed; treating paste as text",
                exc_info=True,
            )
            parsed = None
        owner = self._chat_input_owner
        if parsed is not None and owner is not None:
            event.prevent_default()
            event.stop()
            owner.apply_paste_payload(event.text, parsed.paths)
            return

        if (
            owner is not None
            and self._paste_collapse_enabled()
            and _should_collapse_chat_paste(event.text)
        ):
            # Intercept the paste so Textual's default _on_paste doesn't insert
            # the full text. The owner stores the content and inserts a compact
            # placeholder instead — applied here rather than posted, so a
            # keystroke queued behind this paste cannot overtake it.
            event.prevent_default()
            event.stop()
            owner.apply_paste_payload(event.text, None)
            return

        # Don't call super() here — Textual's MRO dispatch already calls
        # TextArea._on_paste after this handler returns. Calling super()
        # would insert the text a second time, duplicating the paste.

    def set_text_from_history(self, text: str, *, cursor_at_end: bool = True) -> None:
        """Set text from history navigation.

        Args:
            text: The history entry text to load.
            cursor_at_end: Place the cursor at the end of the loaded text
                (use for down-navigation, so the next down press continues
                forward through history). When `False`, place at the start
                so the next up press continues backward. Defaults to `True`
                to preserve historical cursor-at-end behavior for callers
                that don't specify a direction.
        """
        self._reset_paste_burst_state()
        self._skip_history_change_events += 1
        self.text = text
        # The suppressed Changed event (see above) is what would normally toggle
        # the clear/copy buttons, so sync them now to hide/show in the same frame
        # the text swaps — otherwise an emptied draft keeps the buttons for a frame.
        self._sync_owner_action_buttons(text)
        if cursor_at_end:
            self.move_cursor_to_end()
        else:
            self.move_cursor((0, 0))

    def move_cursor_to_end(self) -> None:
        """Move the cursor to the end of the current text."""
        lines = self.text.split("\n")
        last_row = len(lines) - 1
        self.move_cursor((last_row, len(lines[last_row])))

    def clear_text(self) -> None:
        """Clear the text area."""
        # Increment (not reset) so any pending Changed event from a prior
        # set_text_from_history is still suppressed, plus one for the
        # self.text = "" assignment below.
        self._skip_history_change_events += 1
        self._reset_paste_burst_state()
        self.text = ""
        # Hide the clear/copy buttons in the same frame the draft empties; the
        # suppressed Changed event would otherwise leave them for an extra frame.
        self._sync_owner_action_buttons("")
        self.move_cursor((0, 0))

    def _sync_owner_action_buttons(self, text: str) -> None:
        """Match the owner's clear/copy buttons to programmatically set text.

        History/clear text swaps suppress the `Changed` event that normally
        drives button visibility, so the owner is updated directly to keep the
        buttons in lockstep with the draft (matching the `Changed`-path gate).
        """
        owner = self._chat_input_owner
        if owner is not None:
            owner._set_action_buttons_visible(visible=bool(text.strip()))

    def discard_text(self) -> bool:
        """Clear the draft via an undoable edit (restorable with ctrl+z).

        Unlike `clear_text`, the deletion is recorded in the undo history and
        the resulting `Changed` event is allowed to propagate, so completion
        and argument-hint state stay in sync.

        Returns:
            `True` when there was text to clear.
        """
        if not self.text:
            return False
        self._reset_paste_burst_state()
        self.clear()
        return True


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
        # The completion controller returns the full command name (e.g.
        # "/remember") in completion space, but the TextArea only contains
        # text after the virtual mode prefix (e.g. "/" in command mode).
        # Strip the prefix to avoid double-insertion.
        prefix = MODE_PREFIXES.get(self._chat_input.mode, "")
        if prefix and replacement.startswith(prefix):
            replacement = replacement[len(prefix) :]
        self._chat_input.replace_completion_range(
            self._chat_input._completion_index_to_text_index(start),
            self._chat_input._completion_index_to_text_index(end),
            replacement,
        )


def _manual_height_ceiling(screen_height: int) -> int:
    """Return the largest composer height a manual resize may request.

    Args:
        screen_height: Current screen height in rows.

    Returns:
        The manual-resize ceiling, at least 1 row.
    """
    return max(
        1,
        min(
            _CHAT_INPUT_MANUAL_MAX_HEIGHT,
            screen_height - _CHAT_INPUT_RESERVED_SCREEN_ROWS,
        ),
    )


def _content_height_floor(text_area: ChatTextArea) -> int:
    """Return the rows the draft currently occupies, capped at auto growth.

    Uses `virtual_size`, so soft-wrapped lines count as the rows they actually
    render as rather than as one row per newline. A drag cannot shrink the
    composer below this, which stops resizing from hiding visible text.

    Args:
        text_area: The composer whose draft height to measure.

    Returns:
        The floor in rows, between 1 and `_CHAT_INPUT_AUTO_MAX_HEIGHT`.
    """
    return max(1, min(text_area.virtual_size.height, _CHAT_INPUT_AUTO_MAX_HEIGHT))


class ChatInputBox(Vertical):
    """Bordered box that owns the chat composer size.

    Sizing is either automatic (content-driven, the default) or manual, and
    `_requested_height` is the discriminator: `None` means automatic.

    A manual size separates *intent* from what is *rendered*. The requested
    height is what the user dragged to and is preserved verbatim; the applied
    height is re-derived on every relayout by fitting that request between a
    content floor and whichever ceiling is currently tightest. So a completion
    popup can transiently squeeze the composer and it springs back when the
    popup closes, and a terminal shrink does not destroy the request.

    `ChatInput` composes the children and drives this box through
    `set_manual_height`, `toggle_expanded`, and `refresh_content_height`.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize sizing state."""
        super().__init__(**kwargs)
        self._completion_rows = 0
        self._prompt_search_rows = 0
        self._requested_height: int | None = None
        self._applied_height: int | None = None

    def on_mount(self) -> None:
        """Re-fit the composer whenever the screen relayouts.

        A manual height pins the composer, so once it has been squeezed by a
        smaller terminal nothing about this box's own geometry changes when the
        terminal grows back -- it never receives another `Resize`. The screen's
        layout-refresh signal does still fire, so that is what restores the
        height the user asked for.
        """
        self.screen.screen_layout_refresh_signal.subscribe(
            self, self._on_layout_refresh
        )

    def _on_layout_refresh(self, _screen: Screen) -> None:
        """Re-fit a manual height against the new layout."""
        self._apply_manual_height()

    def _composer(self) -> ChatTextArea | None:
        """Return the composer, or `None` when the box is mid-teardown.

        Returns:
            The child text area, or `None` if it is no longer mounted.
        """
        try:
            return self.query_one(ChatTextArea)
        except NoMatches:
            logger.warning("ChatInputBox: composer not found; skipping height update")
            return None

    def _screen_height(self) -> int | None:
        """Return the screen height, or `None` when detached from a screen.

        Returns:
            The screen's row count, or `None` if this box has no screen.
        """
        try:
            return self.screen.size.height
        except NoScreen:
            logger.warning("ChatInputBox: no screen; skipping height update")
            return None

    def set_manual_height(self, height: int) -> None:
        """Request a manual composer height and render it.

        The request is stored clamped only by the screen ceiling; the popup and
        content constraints are applied at render time so they stay reversible.

        Args:
            height: Desired composer height in rows.
        """
        screen_height = self._screen_height()
        if screen_height is None:
            return
        self._requested_height = max(
            1, min(height, _manual_height_ceiling(screen_height))
        )
        self._apply_manual_height()

    def refresh_content_height(self) -> None:
        """Re-fit a manual height after the draft's rendered height changes."""
        if self._requested_height is not None:
            self._apply_manual_height()

    def _apply_manual_height(self) -> None:
        """Render the requested height against the current constraints.

        Three ceilings compete, and the tightest wins:

        - the screen ceiling, which reserves rows for the rest of the app;
        - the box ceiling, `_CHAT_INPUT_BOX_MAX_HEIGHT` minus the border gutter
          and any inline panels (completion popup, prompt search), so a panel
          renders inside the border rather than overflowing it;
        - the same screen budget again but with the panels subtracted, since a
          panel adds rows to the box that the plain screen ceiling ignores.

        The result is then floored by the visible draft, so a manual height
        never hides text. Both `height` and `max_height` are set because
        `max_height` alone would leave Textual's `auto` growth in charge.
        """
        if self._requested_height is None:
            return
        text_area = self._composer()
        screen_height = self._screen_height()
        if text_area is None or screen_height is None:
            return
        panel_rows = self._completion_rows + self._prompt_search_rows
        # The plain screen cap already allows for the composer's own gutter; a
        # panel needs it reserved explicitly because it adds rows to the box.
        panel_gutter = self.gutter.height if panel_rows else 0
        screen_available = (
            screen_height - _CHAT_INPUT_RESERVED_SCREEN_ROWS - panel_rows - panel_gutter
        )
        available = min(
            _manual_height_ceiling(screen_height),
            max(
                1,
                _CHAT_INPUT_BOX_MAX_HEIGHT - self.gutter.height - panel_rows,
            ),
            max(1, screen_available),
        )
        minimum = min(_content_height_floor(text_area), available)
        height = max(minimum, min(self._requested_height, available))
        # Writing styles schedules another layout, which republishes the signal
        # this method runs from. Bail when nothing moved so the two cannot feed
        # each other.
        if height == self._applied_height:
            return
        self._applied_height = height
        text_area.styles.height = height
        text_area.styles.max_height = height
        text_area.call_after_refresh(text_area.scroll_cursor_visible)

    def _reset_height(self) -> None:
        """Restore content-driven composer sizing."""
        self._requested_height = None
        self._applied_height = None
        text_area = self._composer()
        if text_area is None:
            return
        text_area.styles.height = "auto"
        text_area.styles.max_height = _CHAT_INPUT_AUTO_MAX_HEIGHT
        text_area.call_after_refresh(text_area.scroll_cursor_visible)

    def _manual_height_is_visible(self) -> bool:
        """Whether a manual height renders taller than automatic sizing would.

        A drag that lands at or below the draft's own height is floored by
        `_content_height_floor`, so it renders exactly as automatic sizing does
        even though a request is stored. Distinguishing the two keeps a toggle
        from having no visible effect.

        Returns:
            True when a manual height is set and is what pins the composer.
        """
        text_area = self._composer()
        if self._requested_height is None or self._applied_height is None:
            return False
        if text_area is None:
            return False
        return self._applied_height > _content_height_floor(text_area)

    def toggle_expanded(self) -> None:
        """Expand to the manual-height ceiling, or drop a manual height.

        Keys off whether a manual height is *visible* rather than merely stored.
        A request floored by the draft renders identically to automatic sizing,
        so collapsing it would leave the composer where it already is and the
        gesture would read as broken -- expanding is the only move that
        responds. That also covers the stray row of travel a press can emit
        before the second half of a double-click lands.
        """
        if self._manual_height_is_visible():
            self._reset_height()
        else:
            # Clamped to the screen ceiling inside `set_manual_height`, so
            # asking for the absolute maximum lands on whatever fits now.
            self.set_manual_height(_CHAT_INPUT_MANUAL_MAX_HEIGHT)

    def on_completion_popup_rows_changed(
        self, event: CompletionPopup.RowsChanged
    ) -> None:
        """Fit a manual composer around the completion popup."""
        self._completion_rows = event.rows
        self._apply_manual_height()
        event.stop()

    def on_prompt_search_panel_rows_changed(
        self, event: PromptSearchPanel.RowsChanged
    ) -> None:
        """Fit a manual composer around the prompt search panel."""
        self._prompt_search_rows = event.rows
        self._apply_manual_height()
        event.stop()

    def on_resize(self, _event: events.Resize) -> None:
        """Re-fit a manual height after this box is resized.

        Deliberately re-renders rather than re-requesting: the stored request
        survives, so shrinking the terminal and growing it back restores the
        height the user actually chose.
        """
        self._apply_manual_height()


class ChatInputResizeHandle(Static):
    """Drag target docked over the chat input's top border.

    Only the background is transparent: the handle sits on the layer above the
    box's border and would blank the line it covers, so `render` repaints the
    horizontal rule itself.

    The handle reports pointer geometry and knows nothing about heights; the
    parent decides what a drag means.
    """

    ALLOW_SELECT = False

    class DragStarted(Message):
        """Message sent when a resize drag begins."""

    class Dragged(Message):
        """Message sent with the current drag delta."""

        def __init__(self, delta: int) -> None:
            """Initialize with a row delta, cumulative since the drag started.

            Positive is upward — screen Y grows downward, so this is the
            negation of the raw pointer movement.
            """
            super().__init__()
            self.delta = delta

    class DragEnded(Message):
        """Message sent when a resize drag stops, however it stopped."""

    class HoverChanged(Message):
        """Message sent when resize hover feedback changes."""

        def __init__(self, highlighted: bool) -> None:
            """Initialize with the new hover state."""
            super().__init__()
            self.highlighted = highlighted

    class ToggleExpanded(Message):
        """Message sent when expanded sizing should be toggled."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize drag state."""
        super().__init__("", markup=False, **kwargs)
        self._drag_start_y: int | None = None
        self._highlighted = False

    def render(self) -> str:
        """Render the border line beneath the drag target.

        Returns:
            A charset-compatible horizontal rule spanning the handle.
        """
        return get_glyphs().box_horizontal * self.size.width

    def _set_highlighted(self, *, highlighted: bool) -> None:
        """Publish top-border hover changes."""
        if self._highlighted != highlighted:
            self._highlighted = highlighted
            self.post_message(self.HoverChanged(highlighted))

    def on_enter(self, event: events.Enter) -> None:
        """Highlight the resize target on hover."""
        if event.node is self:
            self._set_highlighted(highlighted=True)

    def on_leave(self, event: events.Leave) -> None:
        """Remove hover feedback, but never mid-drag.

        A drag routinely travels outside the handle; dropping the highlight
        there would make the border flicker off the moment resizing starts.
        """
        if event.node is self and self._drag_start_y is None:
            self._set_highlighted(highlighted=False)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Begin resizing from a left press on the handle."""
        if event.button != 1:
            return
        self._drag_start_y = event.screen_y
        self._set_highlighted(highlighted=True)
        self.capture_mouse()
        self.post_message(self.DragStarted())
        event.stop()
        event.prevent_default()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        """Publish movement during a captured drag."""
        if self._drag_start_y is None:
            return
        delta = self._drag_start_y - event.screen_y
        # A double-click registers only when both presses land on the same cell,
        # which is exactly when the pointer drifts away and back — emitting a
        # zero-delta move. Reporting it would pin the composer to a manual height
        # the user never asked for, freezing the auto-growth they expect as they
        # keep typing.
        if delta:
            self.post_message(self.Dragged(delta))
        event.stop()
        event.prevent_default()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        """Finish an active left-button resize.

        Hover feedback drops here rather than being re-derived from the pointer
        position: releasing the capture makes Textual recompute what the mouse
        is over and deliver a `Leave` anyway, and the next real movement back
        onto the handle re-highlights it through `on_enter`.
        """
        if self._drag_start_y is None or event.button != 1:
            return
        self._end_drag(highlighted=False)
        event.stop()
        event.prevent_default()

    def _end_drag(self, *, highlighted: bool, notify: bool = True) -> None:
        """Clear drag state, drop mouse capture, and settle hover feedback.

        Args:
            highlighted: Hover state to leave the handle in.
            notify: Whether to announce `DragEnded`. Suppressed during teardown,
                where the message pump is closing and nothing can receive it.
        """
        was_dragging = self._drag_start_y is not None
        self._drag_start_y = None
        self.release_mouse()
        self._set_highlighted(highlighted=highlighted)
        if was_dragging and notify:
            self.post_message(self.DragEnded())

    def on_mouse_release(self, _event: events.MouseRelease) -> None:
        """Abandon a drag when the app revokes this widget's mouse capture.

        Without this the handle keeps a phantom drag alive: `on_leave` refuses
        to clear the highlight, and later pointer movement resizes from a stale
        baseline with no press behind it.
        """
        if self._drag_start_y is not None:
            logger.debug("Resize drag ended without a mouse up; clearing drag state")
        self._end_drag(highlighted=False)

    def on_hide(self, _event: events.Hide) -> None:
        """Abandon a drag when the composer is hidden mid-gesture."""
        self._end_drag(highlighted=False)

    def on_click(self, event: Click) -> None:
        """Toggle expanded sizing on a double-click (or a faster chain)."""
        if event.button == 1 and event.chain >= _DOUBLE_CLICK_CHAIN:
            self.post_message(self.ToggleExpanded())
            event.stop()
            event.prevent_default()

    def on_unmount(self) -> None:
        """Release mouse capture and hover state during teardown."""
        self._end_drag(highlighted=False, notify=False)


class ChatInput(Vertical):
    """Chat input widget with prompt, multi-line text, autocomplete, and history.

    Features:
    - Multi-line input with TextArea
    - Enter to submit, modifier key for newlines (see `config.newline_shortcut`)
    - Up/Down arrows for command history at input boundaries (start/end of text)
    - Autocomplete for @ (files) and / (commands)
    - Drag the top border to resize the composer; double-click it to expand to
      the maximum height, or to drop a manual height back to content-driven
      sizing
    """

    DEFAULT_CSS = (
        """
    ChatInput {
        height: auto;
        layers: base actions;
    }

    ChatInput #input-box {
        height: auto;
        min-height: 3;
        padding: 0;
        background: $background;
        border: solid $primary;
    }

    ChatInput.mode-shell #input-box {
        border: solid $mode-bash;
    }

    ChatInput.mode-command #input-box {
        border: solid $mode-command;
    }

    ChatInput.mode-shell-incognito #input-box {
        border: solid $mode-incognito;
        border-title-color: $mode-incognito;
        border-title-style: bold;
    }

    /* Pre-mount default only. `_sync_resize_handle_color` sets the color as an
       inline style, which outranks this rule, because the hover highlight must
       persist while a drag travels outside the handle -- something `:hover`
       cannot express. Mode-specific rules here would be dead code, so mode
       colors live in that method instead. */
    ChatInput #input-resize-handle {
        layer: actions;
        dock: left;
        width: 1;
        height: 1;
        offset: 1 0;
        background: transparent;
        color: $primary;
        pointer: ns-resize;
    }

    /* Action buttons float on their own z-layer over the top border line, so
       they cost no content row and never overlap the draft text. The row docks
       to the right edge and sizes to its buttons (`width: auto`), overlaying
       only the right portion of the border line and leaving the rest clear. */
    ChatInput #input-actions {
        layer: actions;
        dock: right;
        width: auto;
        height: 1;
        margin-right: 1;
        display: none;
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

    ChatInput.mode-shell-incognito .input-prompt {
        color: $mode-incognito;
    }

    ChatInput ChatTextArea {
        width: 1fr;
        height: auto;
        min-height: 1;
        border: none;
        background: transparent;
        padding: 0;
    }

    ChatInput ChatTextArea.cursor-underline .text-area--cursor {
        background: transparent;
        color: $text;
        text-style: underline;
    }

    ChatInput ChatTextArea:focus {
        border: none;
    }
    """
        f"""
    /* Sizes the composer sizing math depends on, interpolated from the module
       constants so the stylesheet cannot silently drift from the arithmetic in
       `ChatInputBox`. Appended rather than inlined above to keep the rest of
       this block free of doubled braces. */
    ChatInput #input-box {{
        max-height: {_CHAT_INPUT_BOX_MAX_HEIGHT + PROMPT_SEARCH_PANEL_ROWS};
    }}

    ChatInput ChatTextArea {{
        max-height: {_CHAT_INPUT_AUTO_MAX_HEIGHT};
    }}
    """
    )
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
            history_file: Override path for persisted input history.
                Resolved by `_default_history_path()` when `None`.
            image_tracker: Optional tracker for attached images
            **kwargs: Additional arguments for parent
        """
        super().__init__(**kwargs)
        self._cwd = Path(cwd) if cwd else Path.cwd()
        self._image_tracker = image_tracker
        self._input_box: ChatInputBox | None = None
        self._resize_handle: ChatInputResizeHandle | None = None
        self._resize_hovered = False
        self._resize_start_height: int | None = None
        self._action_buttons: Horizontal | None = None
        self._text_area: ChatTextArea | None = None
        self._popup: CompletionPopup | None = None
        self._completion_manager: MultiCompletionManager | None = None
        self._completion_view: _CompletionViewAdapter | None = None
        self._slash_controller: SlashCommandController | None = None

        # Collapsed paste storage: paste_id → full content.  When a large paste
        # arrives, the full text is stored here and a compact
        # `[Pasted text #N +M lines]` placeholder is inserted into the text
        # area instead.  At submission the placeholder is expanded back.
        self._pasted_contents: dict[int, PastedContent] = {}
        self._next_paste_id = 1

        # Whether large pastes are collapsed into `[Pasted text #N +M lines]`
        # placeholders.
        # Gated by `display.collapse_pastes` (env / `[ui].collapse_pastes`);
        # when disabled, pasted text is inserted verbatim.
        self._collapse_pastes = _collapse_pastes_enabled()

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

        # Whether the most recent `apply_paste_payload` appended its own
        # trailing space. Read by `ChatTextArea._payload_supplied_trailing_space`
        # to decide whether a pending space keystroke would double it.
        self._paste_appended_trailing_space = False

        # Text area content from the previous Changed event. Used to skip
        # blocking filesystem path-detection on single-keystroke edits while
        # still detecting replacement edits that insert a full path payload.
        self._prev_text = ""

        # Track current suggestions for click handling
        self._current_suggestions: list[tuple[str, str]] = []
        self._current_selected_index = 0

        # Command name (without /) → argument hint for inline ghost text
        self._argument_hints: dict[str, str] = {}
        # Runtime hints that depend on session state, kept separate so rebuilding
        # slash commands after skill discovery cannot replace them.
        self._argument_hint_overrides: dict[str, str] = {}

        # Inline prompt search (first Ctrl+R tier). `None` draft means inactive;
        # the snapshot is what Escape restores. The panel widget itself is
        # grabbed in `on_mount` (it is composed with the input box).
        self._prompt_search: PromptSearchPanel | None = None
        self._prompt_search_draft: str | None = None
        self._prompt_search_cursor: tuple[int, int] | None = None
        self._prompt_search_query = ""
        self._prompt_search_prompts: tuple[str, ...] = ()
        self._prompt_search_filtered: list[str] = []
        self._prompt_search_index = 0
        # Latches once the "history is not being saved" warning has been shown,
        # so a read-only home directory costs one toast rather than one per
        # submission.
        self._warned_history_unwritable = False

        # Set up history manager
        if history_file is None:
            history_file = _default_history_path()
        self._history = HistoryManager(history_file)

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual widget method convention
        """Compose the chat input layout.

        Yields:
            Widgets for the input row and completion popup.
        """
        # The bordered box owns the prompt, text area, and completion popup so
        # the action buttons (a sibling) can float on its top border line; a
        # widget can only render on its sibling's border, not its parent's.
        input_box = ChatInputBox(id="input-box")
        with input_box:
            with Horizontal(classes="input-row"):
                yield Static(">", classes="input-prompt", id="prompt")
                yield ChatTextArea(id="chat-input")
            yield CompletionPopup(id="completion-popup")
            yield PromptSearchPanel(id="prompt-search-panel")

        yield ChatInputResizeHandle(id="input-resize-handle")

        # Action buttons float on their own z-layer over the top border line so
        # they cost no content row and never overlap the draft text.
        with Horizontal(id="input-actions"):
            yield InputActionButton(
                "[ X ]",
                "clear",
                id="clear-button",
                classes="input-action input-action-clear",
            )
            yield InputActionButton(
                "[ COPY ]",
                "copy",
                id="copy-button",
                classes="input-action input-action-copy",
            )

    def on_mount(self) -> None:
        """Initialize components after mount."""
        self._input_box = self.query_one("#input-box", ChatInputBox)
        self._resize_handle = self.query_one(
            "#input-resize-handle", ChatInputResizeHandle
        )
        self._action_buttons = self.query_one("#input-actions", Horizontal)
        if is_ascii_mode():
            colors = theme.get_theme_colors(self)
            self._input_box.styles.border = ("ascii", colors.primary)

        self._text_area = self.query_one("#chat-input", ChatTextArea)
        self._popup = self.query_one("#completion-popup", CompletionPopup)
        self._prompt_search = self.query_one("#prompt-search-panel", PromptSearchPanel)
        self._text_area._chat_input_owner = self
        self._text_area.set_shell_highlighting(
            enabled=self.mode in {"shell", "shell_incognito"}
        )

        # Both controllers implement the CompletionController protocol but have
        # different concrete types; the list-item warning is a false positive.
        self._completion_view = _CompletionViewAdapter(self)
        self._file_controller = FuzzyFileController(
            self._completion_view, cwd=self._cwd
        )
        self._slash_controller = SlashCommandController(
            get_slash_commands(), self._completion_view
        )
        self._completion_manager = MultiCompletionManager(
            [
                self._slash_controller,
                self._file_controller,
            ]  # ty: ignore[invalid-argument-type]  # Controller types are compatible at runtime
        )

        self._rebuild_argument_hints(get_slash_commands())

        self._warm_file_cache()
        self.set_interval(
            _FILE_CACHE_REFRESH_INTERVAL_SECONDS,
            self._refresh_file_cache,
        )
        self.call_after_refresh(self._sync_resize_handle_geometry)
        self.watch(self.app, "theme", self._on_theme_change, init=False)
        self._sync_resize_handle_color()
        self._text_area.focus()

    def _on_theme_change(self) -> None:
        """Recolor the resize handle when the app theme changes."""
        self._sync_resize_handle_color()

    def _sync_resize_handle_geometry(self) -> None:
        """Inset the resize handle so border corners remain visible."""
        if self._input_box is None or self._resize_handle is None:
            return
        # `Widget.region` swallows NoScreen/NoWidget and returns a null region,
        # so a zero width means "not laid out yet" rather than a real size.
        # Writing it through would collapse the whole drag target to one column.
        box_width = self._input_box.region.width
        if not box_width:
            logger.debug("Chat input box not laid out yet; deferring handle geometry")
            return
        self._resize_handle.styles.width = max(
            1,
            box_width - _CHAT_INPUT_BORDER_CORNER_COLUMNS,
        )

    def on_resize(self, _event: events.Resize) -> None:
        """Keep the resize handle aligned after layout changes."""
        self.call_after_refresh(self._sync_resize_handle_geometry)

    def _sync_resize_handle_color(self) -> None:
        """Match the resize line to the active input mode and hover state."""
        if self._resize_handle is None:
            return
        colors = theme.get_theme_colors(self)
        mode_colors = {
            "shell": colors.mode_bash,
            "command": colors.mode_command,
            "shell_incognito": colors.mode_incognito,
        }
        if self.mode != "normal" and self.mode not in mode_colors:
            # A mode added to config without a color here would render the resize
            # line in the default color while the border and prompt follow the
            # new mode -- a mismatch that is hard to trace back to this method.
            logger.warning(
                "No resize handle color for mode %r; falling back to primary",
                self.mode,
            )
        color = Color.parse(mode_colors.get(self.mode, colors.primary))
        if self._resize_hovered:
            color = color.lighten(_CHAT_INPUT_RESIZE_HOVER_LIGHTEN)
        self._resize_handle.styles.color = color

    def _set_resize_highlighted(self, *, highlighted: bool) -> None:
        """Toggle resize hover feedback on the interior border line."""
        self._resize_hovered = highlighted
        self._sync_resize_handle_color()

    def on_chat_input_resize_handle_drag_started(
        self, event: ChatInputResizeHandle.DragStarted
    ) -> None:
        """Record the composer height at the start of a drag."""
        if self._text_area is not None:
            self._resize_start_height = max(1, self._text_area.size.height)
        event.stop()

    def on_chat_input_resize_handle_dragged(
        self, event: ChatInputResizeHandle.Dragged
    ) -> None:
        """Apply a drag delta to the composer height."""
        if self._resize_start_height is None:
            # No baseline means no DragStarted arrived; resizing from a stale
            # one would jump the composer to an unrelated size.
            logger.debug("Ignoring resize drag with no recorded start height")
            event.stop()
            return
        if self._input_box is not None:
            self._input_box.set_manual_height(self._resize_start_height + event.delta)
        event.stop()

    def on_chat_input_resize_handle_drag_ended(
        self, event: ChatInputResizeHandle.DragEnded
    ) -> None:
        """Drop the drag baseline so a later delta cannot reuse it."""
        self._resize_start_height = None
        event.stop()

    def on_chat_input_resize_handle_hover_changed(
        self, event: ChatInputResizeHandle.HoverChanged
    ) -> None:
        """Update top-border hover feedback."""
        self._set_resize_highlighted(highlighted=event.highlighted)
        event.stop()

    def on_chat_input_resize_handle_toggle_expanded(
        self, event: ChatInputResizeHandle.ToggleExpanded
    ) -> None:
        """Expand the composer, or drop a manual height back to automatic."""
        if self._input_box is not None:
            self._input_box.toggle_expanded()
        event.stop()

    def _warm_file_cache(self, *, force: bool = False, exclusive: bool = False) -> None:
        """Schedule an `@` file-completion cache warmer.

        No-ops before `on_mount` wires up the file controller (the periodic
        refresh interval can fire during teardown or a partial mount).

        Args:
            force: Re-walk even when the cache is already populated. The prior
                cache stays visible until the new walk completes.
            exclusive: Cancel any other in-flight warmer in the shared worker
                group before starting, so a slow walk is superseded by the next
                tick rather than stacking overlapping walks. Used by the
                periodic refresh; the on-mount/cwd-switch warmers run
                non-exclusively so a quick invalidation can warm concurrently.
        """
        file_controller = getattr(self, "_file_controller", None)
        if file_controller is None:
            return
        self.run_worker(
            file_controller.warm_cache(force=force),
            exclusive=exclusive,
            group=_FILE_CACHE_WORKER_GROUP,
            exit_on_error=False,
        )

    def _refresh_file_cache(self) -> None:
        """Re-warm the `@` file-completion cache off the event loop."""
        self._warm_file_cache(force=True, exclusive=True)

    def set_cwd(self, cwd: str | Path) -> None:
        """Update file completion to use a new cwd.

        Re-roots the file controller and schedules a background cache warm so
        the project-root walk runs off the event loop.
        """
        self._cwd = Path(cwd)
        file_controller = getattr(self, "_file_controller", None)
        if file_controller is not None:
            file_controller.set_cwd(self._cwd)
            self._warm_file_cache()

    def update_slash_commands(self, commands: list[CommandEntry]) -> None:
        """Update the slash command controller's command list.

        Called by the app after discovering skills to merge static
        commands with dynamic `/skill:` entries.

        Args:
            commands: Full list of `CommandEntry` instances.
        """
        if self._slash_controller:
            self._slash_controller.update_commands(commands)
            self._rebuild_argument_hints(commands)
        else:
            logger.warning(
                "Cannot update slash commands: controller not initialized "
                "(widget not yet mounted)"
            )

    def set_argument_hint_override(self, command: str, hint: str | None) -> None:
        """Set, suppress, or restore a runtime slash-command argument hint.

        Args:
            command: Slash command name, with or without the leading `/`.
            hint: Replacement hint, an empty string to suppress the registered
                hint, or `None` to restore it.
        """
        name = command.removeprefix("/")
        if hint is None:
            self._argument_hint_overrides.pop(name, None)
        else:
            self._argument_hint_overrides[name] = hint
        self._update_argument_hint()

    def _rebuild_argument_hints(self, commands: list[CommandEntry]) -> None:
        """Rebuild the command-name -> argument-hint lookup.

        Args:
            commands: Current list of `CommandEntry` instances.
        """
        self._argument_hints = {
            entry.name.removeprefix("/"): entry.argument_hint
            for entry in commands
            if entry.argument_hint
        }

    def _update_argument_hint(self) -> None:
        """Show or clear inline ghost text for slash-command argument hints.

        Sets `ChatTextArea.argument_hint` when the input is a known slash
        command followed by a trailing space with no args typed yet. Both
        spacebar and Tab completion produce this state (Tab goes through
        `replace_completion_range` which appends a trailing space).
        """
        if not self._text_area:
            return

        if self.mode == "command":
            text = self._text_area.text
            if text.endswith(" ") and text.count(" ") == 1:
                command = text[:-1]
                hint = self._argument_hint_overrides.get(command)
                if hint is None:
                    hint = self._argument_hints.get(command, "")
                if hint:
                    self._text_area.argument_hint = hint
                    return

        self._text_area.argument_hint = ""

    def _set_action_buttons_visible(self, *, visible: bool) -> None:
        """Show or hide the clear/copy action buttons on the input border.

        Only writes `display` when it actually changes. Mutating it on every
        keystroke would trigger a layout reflow each time, which perturbs the
        completion popup's deferred (`call_after_refresh`) show/hide ordering.
        """
        if self._action_buttons is not None and self._action_buttons.display != visible:
            self._action_buttons.display = visible

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Detect input mode and update completions."""
        text = event.text_area.text
        # Reveal the clear/copy buttons only when there is a meaningful draft to
        # act on, so an empty input keeps a clean, uncluttered border.
        # Whitespace-only input (e.g. stray spaces or newlines) has nothing
        # worth clearing or copying, so it stays hidden too. Done before the
        # early returns below so recalled-history text shows them as well.
        # NOTE: this `strip()` gate is deliberately stricter than the keyboard
        # paths (esc+esc clear, Ctrl+C copy), which act on the raw value so a
        # whitespace-only draft is still clearable/copyable without the buttons.
        self._set_action_buttons_visible(visible=bool(text.strip()))
        if self._input_box is not None:
            self._input_box.refresh_content_height()
        # Drag-drop / bracketed paste arrive as one Changed event with a
        # multi-character inserted span. Normal typing arrives one character at
        # a time. Checking the changed span (rather than net length delta)
        # preserves replacement edits where selected text is replaced by a path
        # of similar length.
        should_check_path_payload = self._should_check_path_payload(text)
        previous_text = self._prev_text
        self._sync_media_tracker_to_text(
            text, previous_text=previous_text, cursor_offset=self._get_cursor_offset()
        )
        self._prev_text = text

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
        elif should_check_path_payload and self._apply_inline_dropped_path_replacement(
            text
        ):
            return

        # Checked after the guards above so we skip the (potentially slow)
        # filesystem lookup when the text change came from history navigation
        # or prefix stripping, which never need path detection.
        is_path_payload = should_check_path_payload and self._is_dropped_path_payload(
            text
        )

        # Guard: skip mode re-detection after we programmatically stripped
        # a prefix character.
        if self._stripping_prefix:
            self._stripping_prefix = False
        elif detected_prefix := detect_mode_prefix(text):
            prefix, raw_detected = detected_prefix
            detected, strip_length = self._resolve_prefix_mode(prefix, raw_detected)
            if prefix == "/" and is_path_payload:
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
                if self.mode != detected:
                    self.mode = detected
                if strip_length:
                    self._strip_mode_prefix(strip_length)
                # Fall through to update completion suggestions in the same
                # refresh cycle as the mode/glyph change rather than waiting
                # for the next text-change event caused by the prefix strip.
                # Note: the strip's text-change event will also call
                # on_text_changed (idempotently) since _stripping_prefix only
                # skips mode detection, not the completion block below.
        # Set inline argument hint before the completion manager runs so
        # the suggestion is ready in the same render pass that hides the popup.
        self._update_argument_hint()

        # Update completion suggestions using completion-space text/cursor.
        if self._completion_manager and self._text_area:
            if is_path_payload:
                self._completion_manager.reset()
            else:
                vtext, vcursor = self._completion_text_and_cursor()
                self._completion_manager.on_text_changed(vtext, vcursor)

        # Scroll input into view when content changes (handles text wrap)
        self.scroll_visible()

    def _should_check_path_payload(self, text: str) -> bool:
        """Return whether a text change may contain a pasted path payload."""
        old = self._prev_text
        if text == old:
            return False

        prefix_len = 0
        max_prefix_len = min(len(old), len(text))
        while prefix_len < max_prefix_len and old[prefix_len] == text[prefix_len]:
            prefix_len += 1

        old_suffix = len(old)
        text_suffix = len(text)
        while (
            old_suffix > prefix_len
            and text_suffix > prefix_len
            and old[old_suffix - 1] == text[text_suffix - 1]
        ):
            old_suffix -= 1
            text_suffix -= 1

        inserted_len = text_suffix - prefix_len
        return inserted_len > 1

    @staticmethod
    def _parse_dropped_path_payload(
        text: str, *, allow_leading_path: bool = False
    ) -> ParsedPastedPathPayload | None:
        """Parse dropped-path payload text through a single parser entrypoint.

        Returns:
            Parsed payload details, otherwise `None`.
        """
        from deepagents_code.input import parse_pasted_path_payload

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
        from deepagents_code.input import extract_leading_pasted_file_path

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
        if len(text) < 2:  # noqa: PLR2004  # Need at least '/' + one char
            return False
        from deepagents_code.input import parse_pasted_path_payload

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

    def _resolve_prefix_mode(self, prefix: str, detected: str) -> tuple[str, int]:
        """Resolve target mode and strip length for a detected mode prefix.

        Applies the `!`/`!!` state machine relative to the current mode.

        Returns:
            Tuple of `(target_mode, strip_length)`.
        """
        strip_length = len(prefix)
        if self.mode == "shell" and detected == "shell":
            # First `!` was stripped on entry to shell mode, so this `!` is the
            # second bang of `!!`. Promote to incognito and consume it.
            detected = "shell_incognito"
        elif self.mode == "shell_incognito" and detected == "shell":
            # Already in incognito; an extra `!` is part of the command body.
            # Skip the strip-and-demote path that would drop back to shell mode.
            detected = "shell_incognito"
            strip_length = 0
        return detected, strip_length

    def handle_mode_prefix_keystroke(self, char: str) -> bool:
        """Switch input mode for a mode trigger typed at the start of the input.

        Handles the switch before `TextArea` inserts the character so the
        trigger (`!`, `!!`, `/`) never flashes on screen for a frame before the
        change handler would strip it.

        Returns:
            True if the keystroke was consumed as a mode selector without
            inserting the character, otherwise False.
        """
        # The first slash enters command mode without being inserted. A second
        # slash at the same offset can be the leading separator of a UNC-style
        # path replayed as key events, so retain it rather than consuming both
        # characters as mode triggers.
        if char == "/" and self.mode == "command":
            self.suppress_next_prefix_detection()
            return False

        detected_prefix = detect_mode_prefix(char)
        if detected_prefix is None:
            return False
        prefix, raw_detected = detected_prefix
        detected, strip_length = self._resolve_prefix_mode(prefix, raw_detected)
        if not strip_length:
            # An extra `!` inside an incognito command body is literal text.
            return False
        if self.mode != detected:
            self.mode = detected
        # No text changed, so run the same hint/completion refresh that
        # on_text_area_changed performs after stripping a typed prefix.
        self._update_argument_hint()
        if self._completion_manager and self._text_area:
            vtext, vcursor = self._completion_text_and_cursor()
            self._completion_manager.on_text_changed(vtext, vcursor)
        self.scroll_visible()
        return True

    def _strip_mode_prefix(self, length: int = 1) -> None:
        """Remove the mode trigger from the text area.

        Sets the `_stripping_prefix` guard so the resulting text-change event is
        not misinterpreted as new input.

        Args:
            length: Number of leading characters to strip (matches the trigger
                length detected by `detect_mode_prefix`).
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
        self._text_area.text = text[length:]
        if row == 0 and col > 0:
            col = max(0, col - length)
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

        if 0 <= index <= self._completion_prefix_len:
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

    def _handle_stale_slash_enter(self) -> bool:
        """Refresh stale slash completions during an Enter-key race.

        Returns:
            `True` when Enter was handled by applying a single visible
            suggestion or by showing multiple visible suggestions.
        """
        if self.mode != "command" or self._text_area is None:
            return False

        slash_controller = self._slash_controller
        if slash_controller is None:
            return False

        if self._text_area._completion_active:
            return False

        text, cursor = self._completion_text_and_cursor()
        if not text.startswith("/"):
            return False

        matches = slash_controller.name_prefix_matches(text, cursor)
        if not matches:
            return False

        completion_manager = self._completion_manager
        if completion_manager is None:
            logger.warning(
                "Slash controller is initialized without completion manager; "
                "stale slash Enter cannot refresh completions."
            )
            return False

        completion_manager.on_text_changed(text, cursor)
        if len(matches) == 1:
            slash_controller.apply_name_prefix_completion(matches[0], cursor)
            self._submit_value(self._text_area.text.strip())
            return True
        return True

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

        # Expand collapsed paste placeholders back to their full content so the
        # agent receives the original text, not the compact reference.
        value = expand_paste_refs(value, self._pasted_contents)
        value = self._replace_submitted_paths_with_images(value)

        mode = self.mode
        if mode == "normal":
            detected = detect_mode_prefix(value)
            if detected is not None:
                _, mode = detected

        # Prepend mode prefix so the app layer receives the original trigger
        # form (e.g. "!ls", "/help"). The value may already contain the prefix
        # when a completion controller wrote it back into the text area before
        # the strip handler ran.
        prefix = MODE_PREFIXES.get(mode, "")
        if prefix and not value.startswith(prefix):
            value = prefix + value

        # Placeholder spans were captured against the raw draft; the transforms
        # above (whitespace strip, paste expansion, path substitution, prefix)
        # shifted offsets. Re-map spans onto the final submitted text so the
        # adapter strips the correct display token from the model-facing message
        # instead of a same-looking literal the user typed.
        if self._text_area is not None and self._image_tracker is not None:
            self._image_tracker.remap_spans_to_text(
                value, previous_text=self._text_area.text
            )

        self._history.add(value)
        self._warn_if_history_unwritable()
        self.post_message(self.Submitted(value, mode))

        if self._text_area:
            # Preserve submission-time attachments until adapter consumes them.
            self._skip_media_sync_events += 1
            self._text_area.clear_text()
        # Clear only after submit. Ordinary edits are undoable, so removing
        # backing content earlier can strand a restored placeholder.  The input
        # and its paste map are emptied together here, so IDs can safely restart
        # at 1 for the next message.
        self._pasted_contents.clear()
        self._next_paste_id = 1
        self.mode = "normal"

    def _sync_media_tracker_to_text(
        self,
        text: str,
        *,
        previous_text: str | None = None,
        cursor_offset: int | None = None,
    ) -> None:
        """Keep tracked media aligned with placeholder tokens in input text.

        Args:
            text: Current text in the input area.
            previous_text: Previous text in the input area.
            cursor_offset: Current cursor offset in the input area.
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
        self._image_tracker.sync_to_text(
            text, previous_text=previous_text, cursor_offset=cursor_offset
        )

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
            # Cursor at top so pressing up again continues backward through
            # history without the user having to navigate to the first row.
            self._text_area.set_text_from_history(display_text, cursor_at_end=False)
        else:
            # No matching older entry — surface the boundary so the user
            # doesn't think their keypress was lost.
            self.app.bell()

    def on_chat_text_area_history_next(
        self,
        event: ChatTextArea.HistoryNext,  # noqa: ARG002  # Textual event handler signature
    ) -> None:
        """Handle history next request."""
        entry = self._history.get_next()
        if entry is not None and self._text_area:
            mode, display_text = self._history_entry_mode_and_text(entry)
            self.mode = mode
            # Cursor at end so pressing down again continues forward through
            # history.
            self._text_area.set_text_from_history(display_text, cursor_at_end=True)
        else:
            self.app.bell()

    def apply_paste_payload(self, text: str, paths: list[Path] | None) -> bool:
        """Apply an already-parsed paste payload to the input.

        Callers apply a payload through this method rather than posting it as a
        message so it reaches the document synchronously. Textual appends a
        posted message to the tail of the receiving widget's FIFO queue, so a
        keystroke the terminal already delivered would be handled first and land
        ahead of the paste.

        Args:
            text: Raw payload text.
            paths: Resolved dropped paths, or `None` to collapse `text` into a
                `[Pasted text #N]` placeholder.

        Returns:
            `True` when the payload was applied. `False` when there is no text
            area to apply it to, in which case the caller still owns the text.
        """
        if not self._text_area:
            return False
        if paths is not None:
            self._paste_appended_trailing_space = self._insert_pasted_paths(text, paths)
        else:
            self._collapse_and_insert_paste(text)
            self._paste_appended_trailing_space = False
        return True

    def suppress_next_prefix_detection(self) -> None:
        """Skip mode-prefix detection for the next text change.

        Used when inserting text that legitimately starts with a mode trigger, so
        the change handler does not consume that character. Shares the guard with
        `_strip_mode_prefix`, which reports a guard left uncleared by a missed
        change event.
        """
        self._stripping_prefix = True

    def handle_external_paste(self, pasted: str) -> bool:
        """Handle paste text from app-level routing when input is not focused.

        When the text area is mounted, the paste is always consumed: file paths
        are attached as images, large text is collapsed into a placeholder,
        and remaining plain text is inserted directly.

        Args:
            pasted: Raw pasted text payload.

        Returns:
            `True` when the text area is mounted and the paste was inserted,
                `False` if the widget is not yet composed.
        """
        if not self._text_area:
            return False

        parsed = self._parse_dropped_path_payload(pasted)
        if parsed is not None:
            self.apply_paste_payload(pasted, parsed.paths)
        elif self._collapse_pastes and _should_collapse_chat_paste(pasted):
            self.apply_paste_payload(pasted, None)
        else:
            self._text_area.insert(pasted)

        self._text_area.focus()
        return True

    def _collapse_and_insert_paste(self, text: str) -> None:
        """Store full paste content and insert a compact placeholder.

        Pasting content identical to a visible already-collapsed placeholder
        expands that placeholder back to the full text in place instead of
        adding a second placeholder — a repeat paste is treated as a request to
        see the content in full.

        Args:
            text: The full pasted text to collapse.
        """
        if not self._text_area:
            logger.debug("Dropping collapsed paste: text area not mounted")
            return
        visible_ids = {
            int(match.group(1))
            for match in PASTE_PLACEHOLDER_PATTERN.finditer(self._text_area.text)
        }
        match_id = next(
            (
                pid
                for pid, stored in self._pasted_contents.items()
                if pid in visible_ids and stored.content == text
            ),
            None,
        )
        if match_id is not None and self._text_area.replace_placeholder_with_text(
            match_id, text
        ):
            return
        paste_id = self._next_paste_id
        self._next_paste_id += 1
        self._pasted_contents[paste_id] = PastedContent(content=text)
        placeholder = format_paste_ref(paste_id, count_lines(text))
        self._text_area.insert(placeholder)
        self.app.notify(_PASTE_COLLAPSED_TOAST, timeout=5, markup=False)

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
        self._text_area.move_cursor_to_end()
        return True

    def _insert_pasted_paths(self, raw_text: str, paths: list[Path]) -> bool:
        """Insert pasted path payload, attaching images when possible.

        Args:
            raw_text: Original paste payload text.
            paths: Resolved file paths parsed from the payload.

        Returns:
            `True` when the inserted text carries a trailing space that
            `_build_path_replacement` appended. Unattached payloads are inserted
            verbatim, so they never do.
        """
        if not self._text_area:
            return False
        replacement, attached = self._build_path_replacement(
            raw_text, paths, add_trailing_space=True
        )
        if attached:
            self._text_area.insert(replacement)
            return replacement.endswith(" ")
        self._text_area.insert(raw_text)
        return False

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

        from deepagents_code.media_utils import (
            MAX_MEDIA_BYTES,
            VIDEO_EXTENSIONS,
            ImageData,
            get_media_from_path,
            is_media_path,
        )

        parts: list[str] = []
        attached = False
        for path in paths:
            media = get_media_from_path(path)
            if media is not None:
                kind = "image" if isinstance(media, ImageData) else "video"
                existing_text = self._text_area.text if self._text_area else raw_text
                parts.append(
                    self._image_tracker.add_media(
                        media,
                        kind,
                        existing_text=existing_text,
                    )
                )
                attached = True
                continue

            # Check if it looked like media but failed validation
            suffix = path.suffix.lower()
            if is_media_path(path):
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
        if mode_match := detect_mode_prefix(entry):
            prefix, mode = mode_match
            return mode, entry[len(prefix) :]
        return "normal", entry

    async def on_key(self, event: events.Key) -> None:
        """Handle key events for completion navigation."""
        if not self._completion_manager or not self._text_area:
            return

        # The inline prompt search owns the keyboard while open; this must run
        # before completion routing so arrows/enter reach the panel rather
        # than the autocomplete controllers. Returning unconditionally matters:
        # a key the search does not own (Backspace with a non-empty query) must
        # still bubble to the query input's own bindings rather than fall
        # through to the mode-exit branch and the completion manager, which
        # would edit the composer behind the open panel.
        if self._prompt_search_active:
            self._handle_prompt_search_key(event)
            return

        # Backspace at the start of a mode prompt exits the current mode. Prefix
        # characters are mode selectors, not hidden draft text, so exiting the
        # mode does not restore `/`, `!`, or `!!` into the input.
        if (
            event.key == "backspace"
            and self.mode != "normal"
            and self._get_cursor_offset() == 0
            and not self._text_area.text
        ):
            # Schedule the popup reset alongside the prompt/style update so both
            # visual changes land before the next paint.
            def _deferred_reset() -> None:
                if self._completion_manager is not None:
                    self._completion_manager.reset()

            self.call_next(_deferred_reset)
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
            case CompletionResult.IGNORED if event.key == "space":
                # Space was intercepted (prevent_default) so the active
                # controller could attempt completion. The controller
                # declined (e.g. file completion), so insert the space that
                # TextArea would have inserted normally.
                self._text_area.insert(" ")
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
        """Post mode changed message and update prompt indicator.

        The prompt glyph update is scheduled for the next message-loop turn so
        callers which also schedule popup work can coalesce both visual changes
        before the next paint.
        """
        # Keep inline argument hints in sync for mode-only transitions
        # (for example, exiting command mode via Escape or backspace).
        self._update_argument_hint()
        if self._text_area is not None:
            self._text_area.set_shell_highlighting(
                enabled=mode in {"shell", "shell_incognito"}
            )

        glyph = MODE_DISPLAY_GLYPHS.get(mode)
        if not glyph and mode != "normal":
            logger.warning(
                "No display glyph for mode %r; falling back to '>'",
                mode,
            )

        def _apply() -> None:
            self.remove_class("mode-shell", "mode-command", "mode-shell-incognito")
            if glyph:
                class_name = (
                    "mode-shell-incognito"
                    if mode == "shell_incognito"
                    else f"mode-{mode}"
                )
                self.add_class(class_name)
            try:
                prompt = self.query_one("#prompt", Static)
            except NoMatches:
                logger.warning("watch_mode._apply: prompt widget not found")
                if mode == "shell_incognito":
                    # Privacy-sensitive: surface a visible warning so the user
                    # never types an incognito command without confirmation
                    # that the mode is active.
                    app = getattr(self, "app", None)
                    if app is not None:
                        with contextlib.suppress(Exception):
                            app.notify(
                                "Incognito mode UI failed to render; "
                                "switching back to normal input.",
                                severity="warning",
                                markup=False,
                            )
                    self.mode = "normal"
                # The handle color is an inline style, so backing out of a mode
                # does not repaint it on its own; a stale incognito-colored line
                # would outlive the mode it advertises.
                self._sync_resize_handle_color()
                return
            prompt.update(glyph or ">")
            if self._input_box is not None:
                self._input_box.border_title = (
                    "incognito" if mode == "shell_incognito" else None
                )
            self._sync_resize_handle_color()

        self.call_next(_apply)
        self.post_message(self.ModeChanged(mode))

    def focus_input(self) -> None:
        """Focus the input field."""
        if self._prompt_search_active and self._prompt_search is not None:
            if self._prompt_search.focus_query():
                return
            # The session is open but the query input is not mounted yet, so
            # keys would route into a panel the user cannot type in. Worth a
            # trace: this combination is a lifecycle bug, not a normal state.
            logger.warning("Prompt search is active but its query input is not mounted")
        if self._text_area:
            self._text_area.focus()

    @property
    def value(self) -> str:
        """Current input value.

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

    def set_value_at_end(self, val: str) -> bool:
        """Set the input value and place the cursor at the end of the text.

        Returns:
            `True` when the value was written, `False` when the text area is
            unavailable and the value could not be set. Callers that surface a
            "moved to input" toast should gate it on this so the toast never
            claims a write that did not happen.
        """
        if not self._text_area:
            return False
        self._text_area.text = val
        self._text_area.move_cursor_to_end()
        return True

    def recent_prompts(self) -> tuple[str, ...]:
        """Refresh and return submitted prompts in newest-first order.

        Returns:
            An immutable snapshot of recent unique prompts.
        """
        return self._history.recent_prompts()

    def prompt_history_error(self) -> str | None:
        """Describe why the last prompt refresh came back empty, if it failed.

        Returns:
            A message naming the unreadable history file, or `None` when the
            file was read (including when it does not exist yet).
        """
        if not self._history.history_unreadable:
            return None
        return f"Could not read prompt history from {self._history.history_file}"

    def insert_at_cursor(self, text: str) -> bool:
        """Insert text at the current cursor through the undoable edit path.

        Returns:
            Whether the text area was available for insertion.
        """
        if not self._text_area:
            return False
        self._text_area.insert(text)
        return True

    @property
    def _prompt_search_active(self) -> bool:
        """Whether the inline prompt search panel is open.

        A `None` draft is the discriminator: the snapshot only exists for the
        lifetime of a search session.
        """
        return self._prompt_search_draft is not None

    def open_prompt_search(self) -> Literal["inline", "modal", "noop"]:
        """Open the inline prompt search, or escalate an open one to the modal.

        First call shows the inline panel with a fresh prompt snapshot, seeds
        its filter from the current draft, and saves that draft for
        cancel-restore; a call while the panel is already open means the second
        Ctrl+R tier.

        Returns:
            `"inline"` when the panel opened, `"modal"` when the caller should
            open the full `PromptClipboardScreen`, or `"noop"` when the
            composer is unavailable.
        """
        if self._text_area is None or self._prompt_search is None:
            return "noop"
        if self._prompt_search_active:
            return "modal"
        if self._current_suggestions:
            # Completion owns the shared panel rows; inserting the search panel
            # between the popup and the input row would break the completion
            # flow's keyboard assumptions, so the modal serves this case.
            return "modal"

        self._prompt_search_draft = self._text_area.text
        self._prompt_search_cursor = self._text_area.cursor_location
        # Both tiers go through the public accessor so they always show the
        # same snapshot.
        self._prompt_search_prompts = self.recent_prompts()
        self._prompt_search_query = self._text_area.text
        self._prompt_search_index = 0
        self._warn_if_history_unreadable()
        self._refresh_prompt_search_panel()
        # The query field is a real Input, so the blinking cursor lives in it
        # while the search is open rather than in the frozen draft above.
        self._prompt_search.focus_query()
        return "inline"

    def escalate_prompt_search(self) -> str:
        """Return the modal filter and close any open inline panel.

        An active inline search supplies its current query. When autocomplete
        sends Ctrl+R directly to the modal tier, the current chat input is the
        query instead.

        Returns:
            The text to seed the modal's filter with.
        """
        query = (
            self._prompt_search_query
            if self._prompt_search_active
            else (self._text_area.text if self._text_area is not None else "")
        )
        self._close_prompt_search(restore_draft=False, refocus=False)
        return query

    def _close_prompt_search(
        self, *, restore_draft: bool, refocus: bool = True
    ) -> None:
        """Hide the inline panel and clear search state.

        Args:
            restore_draft: Whether to restore the cursor from the
                `open_prompt_search` snapshot. Escape and empty-query Backspace
                do; insert and modal escalation leave the composer alone.
            refocus: Whether to return focus to the composer. Escalating to the
                modal skips this so the modal's own filter input takes focus.
        """
        draft = self._prompt_search_draft
        cursor = self._prompt_search_cursor
        self._prompt_search_draft = None
        self._prompt_search_cursor = None
        self._prompt_search_query = ""
        self._prompt_search_prompts = ()
        self._prompt_search_filtered = []
        self._prompt_search_index = 0
        if self._prompt_search is not None:
            self._prompt_search.hide()
        # Only the cursor needs restoring. Edits made while the panel was open
        # are kept, and in the unedited case the text already equals the
        # snapshot -- assigning it back would be a no-op that clears the undo
        # history, because `TextArea.text` aliases `load_text`. The cursor goes
        # back to the snapshot position, matching readline/codex cancel
        # semantics rather than jumping to the end.
        if (
            restore_draft
            and draft is not None
            and cursor is not None
            and self._text_area is not None
            and self._text_area.text == draft
        ):
            self._text_area.move_cursor(cursor)
        if refocus:
            self.focus_input()

    def _refresh_prompt_search_panel(self) -> None:
        """Filter the snapshot by the query and re-render the panel."""
        if self._prompt_search is None:
            return
        self._prompt_search_filtered = filter_prompts(
            self._prompt_search_prompts, self._prompt_search_query
        )
        self._prompt_search_index = max(
            0, min(self._prompt_search_index, len(self._prompt_search_filtered) - 1)
        )
        # Pass every filtered title, not just the first visible page: the panel
        # renders a window around the selection, and a row that is not mounted
        # cannot be scrolled into view — which previously made arrow moves past
        # row 5 invisible and left scrolling with nothing to do.
        titles = [prompt_title(prompt) for prompt in self._prompt_search_filtered]
        empty: str | None
        if self._prompt_search_filtered:
            empty = None
        elif self._prompt_search_prompts:
            empty = "No matching prompts."
        else:
            # Never claim the history is empty when it is only unreadable.
            empty = (
                self.prompt_history_error()
                or "No prompts yet. Submitted prompts appear here."
            )
        self._prompt_search.update_state(
            self._prompt_search_query, titles, self._prompt_search_index, empty
        )

    def _warn_if_history_unreadable(self) -> None:
        """Warn when the shown prompts are a degraded fallback.

        The empty-state message already names an unreadable history file, but
        only when there is nothing to list. `recent_prompts` falls back to this
        session's entries on a read failure, so the usual outcome is a
        *non-empty* list that looks like a complete history and is silently
        truncated. Warn whenever the read failed and there is something to
        show, since that is the case the empty state cannot cover.
        """
        error = self.prompt_history_error()
        if error is None or not self._prompt_search_prompts:
            return
        self.app.notify(
            f"{error}; showing this session's prompts only",
            severity="warning",
            markup=False,
        )

    def _warn_if_history_unwritable(self) -> None:
        """Say once that prompts are no longer being saved.

        A failed append keeps the entry in memory, so up-arrow and the prompt
        clipboard keep working and nothing on screen suggests a problem. The
        prompts are still lost at exit, and the clipboard advertises itself as
        durable history, so the loss has to be stated rather than only logged.
        """
        if self._warned_history_unwritable or not self._history.history_unwritable:
            return
        self._warned_history_unwritable = True
        self.app.notify(
            f"Could not save prompt history to {self._history.history_file}; "
            "this session's prompts will be lost when it ends",
            severity="warning",
            markup=False,
        )

    def _prompt_search_insert_selected(self) -> None:
        """Insert the selected prompt into the draft and close the panel."""
        prompt = self._prompt_search_filtered[self._prompt_search_index]
        self._close_prompt_search(restore_draft=False)
        if self._text_area is not None:
            self._text_area.insert(prompt)
            self._text_area.scroll_cursor_visible()

    def _handle_prompt_search_key(self, event: events.Key) -> bool:
        """Handle one key while the inline prompt search is open.

        Runs from `ChatInput.on_key` (ahead of completion routing) and from
        `ChatTextArea._on_key` (ahead of the TextArea's own editing defaults,
        which is what the no-`await` rule below protects); it may not
        `await`, because the panel rebuild it triggers is already message-pumped
        through `PromptSearchPanel.call_next`, so synchronous handling loses
        no frames.

        Args:
            event: The key event, bubbling from either the search query input
                (the usual case, since `open_prompt_search` focuses it) or the
                text area when focus never moved.

        Returns:
            `True` when the panel consumed the key and stopped it here.
                `False` means the panel does not own the key and the caller
                decides: `ChatInput.on_key` lets it bubble to the query
                input's own bindings, while `ChatTextArea._on_key` suppresses
                it so an unrecognized key cannot edit the frozen draft behind
                the panel.
        """
        if event.key == "escape":
            self._close_prompt_search(restore_draft=True)
            event.prevent_default()
            event.stop()
            return True

        if event.key in {"enter", "tab"}:
            event.prevent_default()
            event.stop()
            if self._prompt_search_filtered:
                self._prompt_search_insert_selected()
            elif self.app is not None:
                self.app.bell()
            return True

        if event.key in {"up", "down"}:
            event.prevent_default()
            event.stop()
            last = len(self._prompt_search_filtered) - 1
            if last < 0:
                return True
            new_index = (
                max(0, self._prompt_search_index - 1)
                if event.key == "up"
                else min(last, self._prompt_search_index + 1)
            )
            if new_index != self._prompt_search_index:
                self._prompt_search_index = new_index
                if self._prompt_search is not None:
                    self._prompt_search.update_selection(new_index)
            return True

        return False

    def on_prompt_search_input_abandon_search(
        self, event: PromptSearchInput.AbandonSearch
    ) -> None:
        """Close the search, restoring the draft, on empty-query Backspace."""
        if not self._prompt_search_active:
            return
        event.stop()
        self._close_prompt_search(restore_draft=True)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter prompts as the query input's text changes."""
        if not self._prompt_search_active:
            return
        if not isinstance(event.input, PromptSearchInput):
            return
        self.post_message(self.Typing())
        self._prompt_search_query = event.value
        self._prompt_search_index = 0
        self._refresh_prompt_search_panel()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Insert the selected prompt when the query input is submitted."""
        if not self._prompt_search_active:
            return
        if not isinstance(event.input, PromptSearchInput):
            return
        event.stop()
        if self._prompt_search_filtered:
            self._prompt_search_insert_selected()
        elif self.app is not None:
            self.app.bell()

    def on_prompt_search_panel_option_selected(
        self, event: PromptSearchPanel.OptionSelected
    ) -> None:
        """Select a clicked prompt row; Enter is still required to insert."""
        event.stop()
        if not self._prompt_search_active:
            return
        if (
            0 <= event.index < len(self._prompt_search_filtered)
            and event.index != self._prompt_search_index
        ):
            self._prompt_search_index = event.index
            if self._prompt_search is not None:
                self._prompt_search.update_selection(event.index)

    def discard_text(self) -> bool:
        """Clear the draft, keeping it restorable via undo (ctrl+z).

        Returns:
            `True` when there was text to clear.
        """
        if self._text_area is None:
            return False
        if self._text_area.text:
            self._skip_media_sync_events += 1
        return self._text_area.discard_text()

    def on_input_action_button_clicked(self, event: InputActionButton.Clicked) -> None:
        """Handle clicks on the `[ X ]` / `[ COPY ]` input buttons."""
        event.stop()
        if event.action == "clear":
            self._clear_via_button()
        elif event.action == "copy":
            self._copy_via_button()
        else:
            assert_never(event.action)

    def _clear_via_button(self) -> None:
        """Clear the draft from the `[ X ]` button (undoable with ctrl+z).

        Also exits any active slash/shell mode, unlike the Esc-driven clear.
        """
        cleared = self.discard_text()
        self.exit_mode()
        if cleared:
            self.app.notify("Input cleared (ctrl+z to undo)", timeout=3, markup=False)
        if self._text_area is not None:
            self._text_area.focus()

    def _copy_via_button(self) -> None:
        """Copy the current draft to the clipboard from the `[ COPY ]` button."""
        from deepagents_code.clipboard import copy_text_with_feedback

        text = expand_paste_refs(self.value, self._pasted_contents)
        if text:
            copy_text_with_feedback(
                self.app,
                text,
                failure_noun="input",
                success_message="Input copied to clipboard",
            )
        # Refocus the input so clicking the button never strands focus on the
        # (non-focusable) button.
        if self._text_area is not None:
            self._text_area.focus()

    @property
    def input_widget(self) -> ChatTextArea | None:
        """Underlying `TextArea` widget.

        Returns:
            The `ChatTextArea` widget or `None` if not mounted.
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
        """Toggle input focus state (e.g., unfocus while agent is working).

        Args:
            active: Whether the input should be focused and accepting input.
        """
        if self._text_area:
            self._text_area.set_app_focus(has_focus=active)

    def set_cursor_style(self, *, style: CursorStyle) -> None:
        """Set the input cursor's visual style.

        Args:
            style: Whether to render a block or underlined character cell.
        """
        if self._text_area is not None:
            self._text_area.set_class(style == "underline", "cursor-underline")

    def set_cursor_blink(self, *, blink: bool) -> None:
        """Toggle the input's cursor blink without changing focus.

        Args:
            blink: Whether the cursor should blink.
        """
        if self._text_area is not None:
            self._text_area.cursor_blink = blink

    def _notify_app_blur(self) -> None:
        """Tell the text area the terminal window lost OS focus."""
        if self._text_area is not None:
            self._text_area._notify_app_blur()

    def _notify_app_focus(self) -> None:
        """Tell the text area the terminal window regained OS focus."""
        if self._text_area is not None:
            self._text_area._notify_app_focus()

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

        # Completion selections should render their final inline hint
        # immediately, without waiting for the subsequent Changed event.
        self._update_argument_hint()
