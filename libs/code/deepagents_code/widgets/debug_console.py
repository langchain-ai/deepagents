r"""Read-only in-app Debug Console modal.

Toggled with `Ctrl+\` (or the hidden `/debug` command), this overlay shows a
point-in-time session/runtime snapshot plus a live tail of recent
`deepagents_code.*` log records sourced from the in-memory ring buffer in
`_debug_buffer`. It never mutates session state.
"""

from __future__ import annotations

import bisect
import logging
from typing import TYPE_CHECKING, ClassVar, Literal, cast

from rich.segment import Segment
from rich.style import Style as RichStyle
from textual.binding import Binding, BindingType
from textual.cache import LRUCache
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.geometry import Size
from textual.screen import ModalScreen
from textual.scroll_view import ScrollView
from textual.strip import Strip
from textual.widgets import Select, Static
from textual.widgets._select import (  # noqa: PLC2701  # needed to keep Tab navigation inside the open Select overlay
    SelectCurrent,
    SelectOverlay,
)

from deepagents_code import theme
from deepagents_code._debug_buffer import InMemoryLogRecord, get_log_buffer
from deepagents_code.clipboard import copy_text_to_clipboard
from deepagents_code.unicode_security import sanitize_control_chars

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from textual import events
    from textual.app import ComposeResult

DEBUG_TOGGLE_KEY = "ctrl+backslash"
r"""Textual key name for the `Ctrl+\` chord that toggles the console."""

_REFRESH_INTERVAL = 0.5
"""Seconds between log-tail refresh ticks."""

_FILTER_SELECT_ID = "debug-level-filter"
FilterValue = Literal[
    "all",
    "min:DEBUG",
    "min:INFO",
    "min:WARNING",
    "min:ERROR",
    "min:CRITICAL",
    "only:DEBUG",
    "only:INFO",
    "only:WARNING",
    "only:ERROR",
    "only:CRITICAL",
]
_BASE_FILTER_OPTIONS: tuple[tuple[str, FilterValue], ...] = (
    ("All", "all"),
    ("INFO", "min:INFO"),
    ("WARNING", "min:WARNING"),
    ("ERROR", "min:ERROR"),
    ("CRITICAL", "min:CRITICAL"),
    ("Only INFO", "only:INFO"),
    ("Only WARNING", "only:WARNING"),
    ("Only ERROR", "only:ERROR"),
    ("Only CRITICAL", "only:CRITICAL"),
)
_DEBUG_FILTER_OPTIONS: tuple[tuple[str, FilterValue], ...] = (
    ("DEBUG", "min:DEBUG"),
    ("Only DEBUG", "only:DEBUG"),
)
_LEVEL_VALUES = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}
_LEVEL_STYLES = {
    "DEBUG": "dim",
    "INFO": "cyan",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold red",
}
_EMPTY_STYLE = None


def _sanitize_display_text(text: str, *, keep_newlines: bool = False) -> str:
    """Return text safe to render in the debug console."""
    return sanitize_control_chars(
        text,
        keep_newlines=keep_newlines,
        collapse_whitespace=False,
    )


def _debug_records_enabled() -> bool:
    """Return whether the package logger can emit DEBUG records."""
    return logging.getLogger("deepagents_code").isEnabledFor(logging.DEBUG)


def _filter_options() -> tuple[tuple[str, FilterValue], ...]:
    """Return level filter options valid for the current logging configuration."""
    if not _debug_records_enabled():
        return _BASE_FILTER_OPTIONS
    all_option = _BASE_FILTER_OPTIONS[:1]
    rest = _BASE_FILTER_OPTIONS[1:]
    return (*all_option, *_DEBUG_FILTER_OPTIONS, *rest)


def _record_matches_filter(
    record: InMemoryLogRecord, level_filter: FilterValue
) -> bool:
    """Return whether *record* should be visible for *level_filter*."""
    if level_filter == "all":
        return True
    mode, selected_level = level_filter.split(":", maxsplit=1)
    if mode == "only":
        return record.level == selected_level
    return _LEVEL_VALUES.get(record.level, 0) >= _LEVEL_VALUES[selected_level]


def _record_to_content(record: InMemoryLogRecord) -> Content:
    """Render a structured log record as styled Textual content.

    Returns:
        Styled content for the log view.
    """
    timestamp = _sanitize_display_text(record.timestamp)
    level = _sanitize_display_text(record.level)
    logger = _sanitize_display_text(record.logger)
    message = _sanitize_display_text(record.message, keep_newlines=True)
    level_style = _LEVEL_STYLES.get(record.level, "dim")
    return Content.assemble(
        (timestamp, "dim"),
        " ",
        (f"{level:<8}", level_style),
        " ",
        (logger, "dim"),
        " ",
        message,
    )


class _LogLevelOverlay(SelectOverlay):
    """Select overlay that treats Tab and Shift+Tab like down and up arrows."""

    def key_tab(self, event: events.Key) -> None:
        """Move the highlighted option down while the menu is open."""
        event.prevent_default()
        event.stop()
        self.action_cursor_down()

    def key_shift_tab(self, event: events.Key) -> None:
        """Move the highlighted option up while the menu is open."""
        event.prevent_default()
        event.stop()
        self.action_cursor_up()

    def key_escape(self, event: events.Key) -> None:
        """Close the dropdown without dismissing the debug console."""
        event.prevent_default()
        event.stop()
        self.action_dismiss()

    def check_consume_key(self, key: str, character: str | None = None) -> bool:
        """Prevent screen-level focus traversal while the menu is open.

        Returns:
            `True` when this overlay should handle the key itself.
        """
        return key in {"escape", "tab", "shift+tab"} or super().check_consume_key(
            key, character
        )


class _LogLevelSelect(Select[FilterValue]):
    """Level dropdown whose open menu treats Tab like arrow navigation."""

    def compose(self) -> ComposeResult:
        """Compose the select with a Tab-aware overlay.

        Yields:
            Current value display and dropdown overlay widgets.
        """
        yield SelectCurrent(self.prompt)
        yield _LogLevelOverlay(type_to_search=self._type_to_search).data_bind(
            compact=Select.compact
        )


class _DebugLogView(ScrollView, can_focus=True):
    """Scrollable styled log view with logical-record hover and click handling."""

    def __init__(
        self,
        on_copy_record: Callable[[InMemoryLogRecord], None],
        *,
        widget_id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=widget_id, classes=classes)
        self._on_copy_record = on_copy_record
        self._records: list[InMemoryLogRecord] = []
        self._notice: Content | None = None
        self._contents: list[Content] = []
        self._wrap_counts: list[int] = []
        self._wrap_prefix: list[int] = [0]
        self._total_visual = 0
        self._cached_width = 0
        self._hover_index: int | None = None
        self._selected_index: int | None = None
        self._render_line_cache: LRUCache[
            tuple[int, int, int, int | None, int | None], Strip
        ] = LRUCache(1024)

    @property
    def line_count(self) -> int:
        """The current visual line count."""
        return self._total_visual

    @property
    def records(self) -> Sequence[InMemoryLogRecord]:
        """The currently visible logical records."""
        return self._records

    def set_records(
        self, records: Sequence[InMemoryLogRecord], *, scroll_end: bool = True
    ) -> None:
        """Replace the visible records and optionally scroll to the bottom."""
        self._notice = None
        self._records = list(records)
        self._hover_index = None
        self._selected_index = self._coerce_selected_index(self._selected_index)
        self._rebuild_contents()
        self._reflow()
        if scroll_end:
            self.scroll_end(animate=False, immediate=True, x_axis=False)

    def append_records(self, records: Sequence[InMemoryLogRecord]) -> None:
        """Append records to the visible view."""
        if not records:
            return
        if self._notice is not None:
            self.set_records(records)
            return
        at_bottom = self.is_vertical_scroll_end
        start = len(self._contents)
        self._records.extend(records)
        self._contents.extend(_record_to_content(record) for record in records)
        width = self._cached_width or self.size.width
        if width <= 0:
            self._recompute_prefix()
            self.refresh()
            return
        self._cached_width = width
        counts = [
            self._wrap_count(content, width) for content in self._contents[start:]
        ]
        self._wrap_counts.extend(counts)
        self._recompute_prefix()
        self.virtual_size = Size(width, self._total_visual)
        self._render_line_cache.clear()
        self.refresh()
        if at_bottom:
            self.scroll_end(animate=False, immediate=True, x_axis=False)

    def clear_records(self) -> None:
        """Clear the visible records."""
        self._notice = None
        self._records.clear()
        self._contents.clear()
        self._wrap_counts.clear()
        self._wrap_prefix = [0]
        self._total_visual = 0
        self._hover_index = None
        self._selected_index = None
        self._render_line_cache.clear()
        self.virtual_size = Size(self.size.width, 0)
        self.refresh()

    def show_notice(self, message: str) -> None:
        """Render a one-line notice in place of log records."""
        self._records.clear()
        self._notice = Content.styled(_sanitize_display_text(message), "dim italic")
        self._hover_index = None
        self._selected_index = None
        self._rebuild_contents()
        self._reflow()

    def _rebuild_contents(self) -> None:
        if self._notice is not None:
            self._contents = [self._notice]
            return
        self._contents = [_record_to_content(record) for record in self._records]

    @staticmethod
    def _wrap_count(content: Content, width: int) -> int:
        if width <= 0:
            return 1
        return max(1, len(content.wrap(width)))

    def _recompute_prefix(self) -> None:
        self._wrap_prefix = [0]
        for count in self._wrap_counts:
            self._wrap_prefix.append(self._wrap_prefix[-1] + count)
        self._total_visual = self._wrap_prefix[-1]

    def _reflow(self) -> None:
        width = self.size.width
        if width <= 0:
            width = self._cached_width
        if width <= 0:
            self._wrap_counts = [1 for _content in self._contents]
            self._recompute_prefix()
            self.refresh()
            return
        self._cached_width = width
        self._render_line_cache.clear()
        self._wrap_counts = [
            self._wrap_count(content, width) for content in self._contents
        ]
        self._recompute_prefix()
        self.virtual_size = Size(width, self._total_visual)
        self.refresh()

    def _content_index_at_visual_y(self, visual_y: int) -> int | None:
        if visual_y < 0 or visual_y >= self._total_visual:
            return None
        index = bisect.bisect_right(self._wrap_prefix, visual_y) - 1
        if 0 <= index < len(self._contents):
            return index
        return None

    def _record_at_visual_y(self, visual_y: int) -> InMemoryLogRecord | None:
        if self._notice is not None:
            return None
        index = self._content_index_at_visual_y(visual_y)
        if index is None or index >= len(self._records):
            return None
        return self._records[index]

    def _coerce_selected_index(self, index: int | None) -> int | None:
        if not self._records:
            return None
        if index is None:
            return None
        return min(max(index, 0), len(self._records) - 1)

    def _select_record(self, index: int) -> None:
        if not self._records:
            return
        self._selected_index = min(max(index, 0), len(self._records) - 1)
        self._hover_index = None
        self._render_line_cache.clear()
        self._scroll_selected_visible()
        self.refresh()

    def _scroll_selected_visible(self) -> None:
        if self._selected_index is None or not self._wrap_prefix:
            return
        start = self._wrap_prefix[self._selected_index]
        end = self._wrap_prefix[self._selected_index + 1] - 1
        _scroll_x, scroll_y = self.scroll_offset
        height = max(self.size.height, 1)
        if start < scroll_y:
            self.scroll_to(y=start, animate=False, immediate=True)
        elif end >= scroll_y + height:
            self.scroll_to(y=end - height + 1, animate=False, immediate=True)

    def _copy_selected_record(self) -> None:
        if self._selected_index is None:
            if not self._records:
                return
            self._selected_index = len(self._records) - 1
        record = self._records[self._selected_index]
        self._on_copy_record(record)

    def render_line(self, y: int) -> Strip:
        _scroll_x, scroll_y = self.scroll_offset
        abs_y = scroll_y + y
        width = self.size.width
        key = (
            abs_y,
            width,
            self._cached_width,
            self._hover_index,
            self._selected_index,
        )
        cached = self._render_line_cache.get(key)
        if cached is not None:
            return cached
        if abs_y >= self._total_visual:
            return Strip.blank(width, self.rich_style)

        content_index = self._content_index_at_visual_y(abs_y)
        if content_index is None:
            return Strip.blank(width, self.rich_style)
        content = self._contents[content_index]
        row_style: RichStyle | None = None
        if self._selected_index == content_index and self._notice is None:
            colors = theme.get_theme_colors(self)
            row_style = RichStyle(
                color=colors.background,
                bgcolor=colors.primary,
                bold=True,
            )
        elif self._hover_index == content_index and self._notice is None:
            colors = theme.get_theme_colors(self)
            row_style = RichStyle(bgcolor=colors.panel)
        wrapped = content.wrap(self._cached_width or width)
        base = self._wrap_prefix[content_index]
        line = wrapped[abs_y - base] if abs_y - base < len(wrapped) else Content()
        segments = [
            segment
            if segment.style is not None
            else Segment(segment.text, _EMPTY_STYLE)
            for segment in line.render_segments(end="")
        ]
        strip = Strip(segments, line.cell_length).crop_extend(0, width, self.rich_style)
        if row_style is not None:
            strip = Strip(
                Segment.apply_style(strip, None, row_style),
                strip.cell_length,
            )
        self._render_line_cache[key] = strip
        return strip

    def notify_style_update(self) -> None:
        """Clear cached render lines after a style update."""
        super().notify_style_update()
        self._render_line_cache.clear()

    def on_resize(self, event: events.Resize) -> None:
        """Re-wrap log entries when the view width changes."""
        if event.size.width != self._cached_width:
            self._reflow()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        """Highlight the logical log record under the pointer."""
        _scroll_x, scroll_y = self.scroll_offset
        hover_index = self._content_index_at_visual_y(scroll_y + event.y)
        if self._notice is not None:
            hover_index = None
        self.styles.pointer = "pointer" if hover_index is not None else "default"
        if hover_index == self._hover_index:
            return
        self._hover_index = hover_index
        self._render_line_cache.clear()
        self.refresh()

    def on_leave(self) -> None:
        """Clear hover highlighting when the pointer leaves the log."""
        self.styles.pointer = "default"
        if self._hover_index is None:
            return
        self._hover_index = None
        self._render_line_cache.clear()
        self.refresh()

    def on_focus(self) -> None:
        """Select the latest log record when keyboard focus enters the log."""
        if self._selected_index is None and self._records:
            self._select_record(len(self._records) - 1)

    def key_up(self, event: events.Key) -> None:
        """Move keyboard selection to the previous logical log record."""
        event.prevent_default()
        event.stop()
        if not self._records:
            return
        index = (
            len(self._records) if self._selected_index is None else self._selected_index
        )
        self._select_record(index - 1)

    def key_down(self, event: events.Key) -> None:
        """Move keyboard selection to the next logical log record."""
        event.prevent_default()
        event.stop()
        if not self._records:
            return
        index = -1 if self._selected_index is None else self._selected_index
        self._select_record(index + 1)

    def key_enter(self, event: events.Key) -> None:
        """Copy the selected logical log record."""
        event.prevent_default()
        event.stop()
        self._copy_selected_record()

    def key_tab(self, event: events.Key) -> None:
        """Move focus from the log to the level filter."""
        event.prevent_default()
        event.stop()
        self.screen.focus_next("#debug-level-filter, #debug-log")

    def key_shift_tab(self, event: events.Key) -> None:
        """Move focus from the log to the level filter."""
        event.prevent_default()
        event.stop()
        self.screen.focus_previous("#debug-level-filter, #debug-log")

    def on_click(self, event: events.Click) -> None:
        """Copy the clicked logical log record."""
        _scroll_x, scroll_y = self.scroll_offset
        record = self._record_at_visual_y(scroll_y + event.y)
        if record is None:
            return
        index = self._content_index_at_visual_y(scroll_y + event.y)
        if index is not None:
            self._select_record(index)
        event.stop()
        self._on_copy_record(record)


class DebugConsoleScreen(ModalScreen[None]):
    """Modal showing a session snapshot and a live tail of recent log records."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close", show=False),
        Binding(DEBUG_TOGGLE_KEY, "close", "Close", show=False, priority=True),
        Binding("ctrl+l", "clear_view", "Clear view", show=False, priority=True),
        Binding("c", "copy", "Copy", show=False, priority=True),
    ]
    """Close, clear-the-view, and copy bindings (all `priority` for the modal)."""

    CSS = """
    DebugConsoleScreen {
        align: center middle;
    }

    DebugConsoleScreen > Vertical {
        width: 100;
        max-width: 95%;
        height: 85%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    DebugConsoleScreen .debug-console-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    DebugConsoleScreen .debug-console-snapshot {
        margin-bottom: 1;
    }

    DebugConsoleScreen .debug-console-toolbar {
        height: auto;
        margin-bottom: 1;
    }

    DebugConsoleScreen .debug-console-filter-label {
        width: auto;
        content-align: center middle;
        color: $text-muted;
        margin-right: 1;
    }

    DebugConsoleScreen #debug-level-filter {
        width: 18;
    }

    DebugConsoleScreen .debug-console-log {
        height: 1fr;
        min-height: 5;
        scrollbar-gutter: stable;
        background: $background;
        border: solid $primary;
        overflow-x: hidden;
        overflow-y: scroll;
    }

    DebugConsoleScreen .debug-console-log:focus {
        border: solid $primary-lighten-2;
    }

    DebugConsoleScreen .debug-console-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }
    """

    def __init__(self, snapshot: Sequence[tuple[str, str]]) -> None:
        """Initialize with a captured *snapshot* of session/runtime fields.

        Args:
            snapshot: Ordered `(label, value)` pairs rendered in the header.
        """
        super().__init__()
        self._snapshot = list(snapshot)
        self._records: list[InMemoryLogRecord] = []
        # Absolute index of the next unrendered log record (incremental writes).
        self._rendered_upto = 0
        # Absolute index floor for Copy: records retained since the last clear.
        self._view_floor = 0
        # One-shot guard so the "buffer unavailable" notice is written only once.
        self._missing_notice_shown = False
        self._level_filter: FilterValue = "all"

    def compose(self) -> ComposeResult:
        """Lay out the title, snapshot, filter, log tail, and key-hint footer.

        Yields:
            The child widgets composing the console.
        """
        with Vertical():
            yield Static("Debug Console", classes="debug-console-title")
            yield Static(self._render_snapshot(), classes="debug-console-snapshot")
            with Horizontal(classes="debug-console-toolbar"):
                yield Static("Level", classes="debug-console-filter-label")
                yield _LogLevelSelect(
                    _filter_options(),
                    value="all",
                    allow_blank=False,
                    id=_FILTER_SELECT_ID,
                    compact=True,
                )
            yield _DebugLogView(
                self._copy_record,
                widget_id="debug-log",
                classes="debug-console-log",
            )
            yield Static(self._render_help(), classes="debug-console-help")

    def on_mount(self) -> None:
        """Start the refresh timer and render the current buffer contents."""
        self.set_interval(_REFRESH_INTERVAL, self._poll_logs)
        self._poll_logs()
        self.call_after_refresh(self.query_one("#debug-log", _DebugLogView).focus)

    def key_tab(self, event: events.Key) -> None:
        """Cycle focus between the level filter and log lines."""
        if self._level_select().expanded:
            return
        event.prevent_default()
        event.stop()
        self.focus_next("#debug-level-filter, #debug-log")

    def key_shift_tab(self, event: events.Key) -> None:
        """Cycle focus between the log lines and level filter."""
        if self._level_select().expanded:
            return
        event.prevent_default()
        event.stop()
        self.focus_previous("#debug-level-filter, #debug-log")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Refresh visible records when the log-level filter changes."""
        if event.select.id != _FILTER_SELECT_ID:
            return
        value = str(event.value)
        if value == self._level_filter:
            return
        self._level_filter = cast("FilterValue", value)
        self._refresh_log_view(scroll_end=True)

    def _render_snapshot(self) -> Content:
        """Build the right-aligned `label: value` snapshot block.

        Returns:
            The formatted snapshot block.
        """
        if not self._snapshot:
            return Content.styled("(no session data)", "dim italic")
        width = max(len(label) for label, _ in self._snapshot)
        lines = [
            Content.assemble((f"{label:>{width}}  ", "bold"), value)
            for label, value in self._snapshot
        ]
        return Content("\n").join(lines)

    @staticmethod
    def _render_help() -> Content:
        """Build the footer key-hint line.

        Returns:
            The formatted key-hint line.
        """
        return Content.styled(
            "Esc close · Ctrl+L clear view · c copy visible logs · click copy line",
            "dim italic",
        )

    def _poll_logs(self) -> None:
        """Append any log records emitted since the last tick to the log view."""
        log = self.query_one("#debug-log", _DebugLogView)
        buffer = get_log_buffer()
        if buffer is None:
            if not self._missing_notice_shown:
                log.show_notice("(log buffer unavailable)")
                self._missing_notice_shown = True
            return
        records, total = buffer.snapshot_records_since(self._rendered_upto)
        self._records.extend(records)
        self._rendered_upto = total
        visible_records = [
            record
            for record in records
            if _record_matches_filter(record, self._level_filter)
        ]
        log.append_records(visible_records)

    def _refresh_log_view(self, *, scroll_end: bool) -> None:
        """Rebuild the log view using the current filter."""
        visible_records = [
            record
            for record in self._records
            if _record_matches_filter(record, self._level_filter)
        ]
        self.query_one("#debug-log", _DebugLogView).set_records(
            visible_records, scroll_end=scroll_end
        )

    def action_clear_view(self) -> None:
        """Clear the on-screen log view; the in-memory buffer keeps accruing."""
        self.query_one("#debug-log", _DebugLogView).clear_records()
        self._records.clear()
        buffer = get_log_buffer()
        if buffer is not None:
            total = buffer.total_emitted
            self._rendered_upto = total
            self._view_floor = total

    def action_copy(self) -> None:
        """Copy visible retained log records since the last clear to the clipboard."""
        lines = [
            record.plain_line
            for record in self._records
            if _record_matches_filter(record, self._level_filter)
        ]
        self._copy_lines(lines, empty_message="No visible log lines to copy")

    def _copy_record(self, record: InMemoryLogRecord) -> None:
        """Copy a clicked logical log record to the clipboard."""
        self._copy_lines([record.plain_line], empty_message="No log line to copy")

    def _level_select(self) -> Select[FilterValue]:
        """Return the level-filter dropdown."""
        return cast(
            "Select[FilterValue]", self.query_one("#debug-level-filter", Select)
        )

    def _copy_lines(self, lines: Sequence[str], *, empty_message: str) -> None:
        """Copy lines to clipboard with user-visible feedback."""
        text = "\n".join(lines)
        if not text:
            self.app.notify(
                empty_message, severity="information", timeout=2, markup=False
            )
            return
        success, error = copy_text_to_clipboard(self.app, text)
        if success:
            self.app.notify(
                "Debug log copied", severity="information", timeout=2, markup=False
            )
            return
        suffix = f": {error}" if error else ""
        self.app.notify(
            f"Failed to copy debug log{suffix}",
            severity="warning",
            timeout=3,
            markup=False,
        )

    def action_close(self) -> None:
        """Close the open level dropdown, or close the debug console."""
        level_select = self._level_select()
        if level_select.expanded:
            level_select.expanded = False
            level_select.focus()
            return
        self.dismiss(None)
