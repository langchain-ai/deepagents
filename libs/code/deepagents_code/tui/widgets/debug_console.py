r"""Read-only in-app Debug Console modal.

Toggled with `Ctrl+\` (or the hidden `/debug` command), this overlay shows a
point-in-time session/runtime snapshot plus a live tail of recent
`deepagents_code.*` log records sourced from the in-memory ring buffer in
`_debug_buffer`. It never mutates session state.
"""

from __future__ import annotations

import asyncio
import bisect
import logging
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple, cast, get_args

from rich.segment import Segment
from rich.style import Style as RichStyle
from textual.binding import Binding, BindingType
from textual.cache import LRUCache
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.geometry import Offset, Size
from textual.screen import ModalScreen
from textual.scroll_view import ScrollView
from textual.strip import Strip
from textual.style import Style as TStyle
from textual.widgets import Checkbox, Select, Static
from textual.widgets._select import (  # noqa: PLC2701  # needed to keep Tab navigation inside the open Select overlay
    SelectCurrent,
    SelectOverlay,
)

from deepagents_code import theme
from deepagents_code._debug import LOG_LEVELS
from deepagents_code._debug_buffer import (
    DEFAULT_CAPACITY,
    InMemoryLogRecord,
    get_log_buffer,
    retention_bucket_for_level,
)
from deepagents_code.clipboard import copy_text_to_clipboard
from deepagents_code.tui.widgets._links import open_style_link
from deepagents_code.unicode_security import sanitize_control_chars

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from textual import events
    from textual.app import ComposeResult

logger = logging.getLogger(__name__)

DEBUG_TOGGLE_KEY = "ctrl+backslash"
r"""Textual key name for the `Ctrl+\` chord that toggles the console."""


class SnapshotField(NamedTuple):
    """A single row in the console's session snapshot.

    The four named fields keep the display strings and their interaction metadata
    explicit at construction sites. `copyable` opts a row into click-to-copy, and
    `thread_id` enables a resolvable `(langsmith)` trace link for the thread row.
    """

    label: str
    value: str
    copyable: bool = False
    """Whether `value` can be clicked to copy it to the clipboard."""
    thread_id: str | None = None
    """A LangSmith thread id whose ``(langsmith)`` trace link is appended to the
    row once the URL resolves. `None` disables the link."""


_SNAPSHOT_COPY_META = "snapshot_copy"
"""Meta key marking a snapshot span whose text is copied on click."""

_SNAPSHOT_COPY_LABEL_META = "snapshot_copy_label"
"""Meta key carrying the snapshot field label used in the copy toast."""


_REFRESH_INTERVAL = 0.5
"""Seconds between log-tail refresh ticks."""

_RECORD_LIMIT = DEFAULT_CAPACITY
"""Maximum records retained per level by an open debug console view.

Matches the buffer's per-level `deque` bound so the console mirrors the buffer's
level-partitioned retention instead of re-flattening it into a single window."""

_FILTER_SELECT_ID = "debug-level-filter"
_CLICK_TO_COPY_ID = "debug-click-to-copy"
"""Id of the checkbox that opts click-to-copy in for the console."""
_CLICK_TO_COPY_DEFAULT = False
"""Whether click-to-copy is enabled before the user toggles the checkbox."""
_FOCUS_CYCLE = f"#{_FILTER_SELECT_ID}, #{_CLICK_TO_COPY_ID}, #debug-log"
"""Tab-cycle selector spanning the toolbar controls and the log view."""
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
_VALID_FILTER_VALUES: frozenset[str] = frozenset(get_args(FilterValue))
"""Every legal `FilterValue`, used to validate values crossing the Select
boundary before they are trusted as a `FilterValue`."""
_LEVEL_STYLES = {
    "DEBUG": "dim",
    "INFO": "cyan",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold red",
}
_EMPTY_STYLE = RichStyle()


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
    threshold = LOG_LEVELS.get(selected_level)
    if threshold is None:
        # An unrecognized level should never reach here (FilterValue enumerates
        # only LOG_LEVELS keys), but a diagnostic must not hide records on a bad
        # filter: show everything rather than raise on the poll timer.
        return True
    return record.levelno >= threshold


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
        click_to_copy: bool = _CLICK_TO_COPY_DEFAULT,
    ) -> None:
        super().__init__(id=widget_id, classes=classes)
        self._on_copy_record = on_copy_record
        self.click_to_copy = click_to_copy
        """Whether clicking a log line copies it. Enter always copies."""
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
            # Not yet sized (e.g. first poll before layout). Assume one visual
            # line per new content so `_wrap_counts` stays 1:1 with `_contents`;
            # the first `on_resize` reflow recomputes real counts. Skipping this
            # would leave the new records out of `_wrap_prefix`/`_total_visual`
            # and invisible until that resize.
            self._wrap_counts.extend(1 for _ in self._contents[start:])
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
        """Move focus from the log to the next toolbar control."""
        event.prevent_default()
        event.stop()
        self.screen.focus_next(_FOCUS_CYCLE)

    def key_shift_tab(self, event: events.Key) -> None:
        """Move focus from the log to the previous toolbar control."""
        event.prevent_default()
        event.stop()
        self.screen.focus_previous(_FOCUS_CYCLE)

    def on_click(self, event: events.Click) -> None:
        """Select the clicked log record, copying it when click-to-copy is on."""
        _scroll_x, scroll_y = self.scroll_offset
        record = self._record_at_visual_y(scroll_y + event.y)
        if record is None:
            return
        index = self._content_index_at_visual_y(scroll_y + event.y)
        if index is not None:
            self._select_record(index)
        event.stop()
        if self.click_to_copy:
            self._on_copy_record(record)


def _snapshot_copy_target(style: object) -> tuple[str, str] | None:
    """Return the copy text and field label from a snapshot span style, if any.

    Args:
        style: The Textual event style under the pointer/click.

    Returns:
        `(text, label)` when the span carries a copy marker, else `None`.
    """
    meta = getattr(style, "meta", None)
    if not isinstance(meta, dict):
        return None
    text = meta.get(_SNAPSHOT_COPY_META)
    if not isinstance(text, str) or not text:
        return None
    label = meta.get(_SNAPSHOT_COPY_LABEL_META)
    if not isinstance(label, str) or not label:
        return None
    return text, label


def _snapshot_copy_success_message(label: str) -> str:
    """Build the toast shown after copying a snapshot field value.

    Args:
        label: The snapshot row label (e.g. `"Thread"`, `"Version"`).

    Returns:
        A short success toast for the copied field.
    """
    # The thread row is labeled "Thread" in the snapshot, but the value users
    # copy is specifically the thread id — keep that wording for the toast.
    if label == "Thread":
        return "Thread ID copied"
    return f"{label} copied"


class _SnapshotView(Static):
    """Snapshot header that copies marked spans and opens link spans on click."""

    # Match WelcomeBanner: disabling auto_links avoids a hover-refresh flicker
    # loop caused by link styles getting a fresh random id on every render.
    auto_links = False

    def __init__(
        self,
        on_copy: Callable[[str, str], None],
        *,
        classes: str | None = None,
    ) -> None:
        """Initialize with a callback used to copy a clicked span's text.

        Args:
            on_copy: Called with `(text, label)` when a copyable span is clicked.
            classes: Optional space-separated CSS classes.
        """
        super().__init__(classes=classes)
        self._on_copy = on_copy

    def on_click(self, event: events.Click) -> None:
        """Copy a marked span or open a link span under the click.

        Copyable snapshot spans (e.g. the thread id) always copy on click; the
        console's "Click to copy" checkbox governs only the log lines, never the
        snapshot.
        """
        if getattr(event.style, "link", None):
            open_style_link(event)
            return
        target = _snapshot_copy_target(event.style)
        if target is not None:
            event.stop()
            text, label = target
            self._on_copy(text, label)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        """Show a hand pointer over clickable spans and reset it elsewhere."""
        clickable = bool(getattr(event.style, "link", None)) or (
            _snapshot_copy_target(event.style) is not None
        )
        self.styles.pointer = "pointer" if clickable else "default"

    def on_leave(self) -> None:
        """Reset the pointer shape when the mouse leaves the snapshot."""
        self.styles.pointer = "default"


class DebugConsoleScreen(ModalScreen[None]):
    """Modal showing a session snapshot and a live tail of recent log records."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close", show=False),
        Binding(DEBUG_TOGGLE_KEY, "close", "Close", show=False, priority=True),
        Binding("ctrl+l", "clear_view", "Clear view", show=False, priority=True),
        # Not `priority`: a priority `c` would pre-empt type-to-search in the
        # open level dropdown (e.g. typing "c" to reach CRITICAL). The log view
        # has no `c` binding, so it still bubbles up to this copy action.
        Binding("c", "copy", "Copy", show=False),
    ]
    """The toggle-key close (`ctrl+backslash`) and `ctrl+l` clear-view are
    `priority`. Escape close and `c` copy are deliberately *not* `priority`:
    Escape must reach the open level dropdown's overlay first so it closes only
    the menu (a priority Escape would tear down the whole console instead), and
    `c` must not pre-empt the dropdown's type-to-search."""

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

    DebugConsoleScreen .debug-console-click-to-copy {
        margin-left: 2;
        color: $text-muted;
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

    def __init__(
        self,
        snapshot: Sequence[SnapshotField],
        *,
        cleared_upto: int = 0,
        on_clear: Callable[[int], None] | None = None,
        click_to_copy: bool = _CLICK_TO_COPY_DEFAULT,
        on_click_to_copy_change: Callable[[bool], None] | None = None,
    ) -> None:
        """Initialize with a captured *snapshot* of session/runtime fields.

        Args:
            snapshot: Ordered `SnapshotField` rows rendered in the header.
            cleared_upto: Absolute emission index a prior `Ctrl+L` cleared up to.
                The console starts rendering from here so a clear persists across
                close/reopen; records emitted after it still appear.
            on_clear: Invoked with the new clear cursor whenever `Ctrl+L` clears
                the view, letting the owner persist it for the next open.
            click_to_copy: Initial state of the "Click to copy" checkbox,
                restored from the persisted preference.
            on_click_to_copy_change: Called with the new value whenever the
                checkbox is toggled, so the host can persist the preference.
        """
        super().__init__()
        self._snapshot = list(snapshot)
        self._records: list[InMemoryLogRecord] = []
        # Absolute index of the next unrendered log record (incremental writes),
        # seeded from any persisted clear so reopening honors the last Ctrl+L.
        self._rendered_upto = cleared_upto
        self._on_clear = on_clear
        # One-shot guard so the "buffer unavailable" notice is written only once.
        self._missing_notice_shown = False
        self._level_filter: FilterValue = "all"
        self._click_to_copy = click_to_copy
        self._on_click_to_copy_change = on_click_to_copy_change
        # Seed links resolved elsewhere in this process (normally the welcome
        # banner) so reopening the console does not briefly render without one.
        self._langsmith_urls = self._cached_langsmith_urls()

    def _cached_langsmith_urls(self) -> dict[str, str]:
        """Return immediately available LangSmith URLs for snapshot threads."""
        from deepagents_code.config import get_cached_langsmith_thread_url

        urls: dict[str, str] = {}
        thread_ids = {field.thread_id for field in self._snapshot if field.thread_id}
        for thread_id in thread_ids:
            try:
                url = get_cached_langsmith_thread_url(thread_id)
            except Exception:  # a diagnostic overlay must always be able to open
                logger.warning(
                    "Cached LangSmith thread URL lookup errored for %r",
                    thread_id,
                    exc_info=True,
                )
                continue
            if url:
                urls[thread_id] = url
        return urls

    def compose(self) -> ComposeResult:
        """Lay out the title, snapshot, filter, log tail, and key-hint footer.

        Yields:
            The child widgets composing the console.
        """
        with Vertical():
            yield Static("Debug Console", classes="debug-console-title")
            snapshot_view = _SnapshotView(
                self._copy_snapshot_value,
                classes="debug-console-snapshot",
            )
            snapshot_view.update(self._render_snapshot())
            yield snapshot_view
            with Horizontal(classes="debug-console-toolbar"):
                yield Static("Level", classes="debug-console-filter-label")
                yield _LogLevelSelect(
                    _filter_options(),
                    value="all",
                    allow_blank=False,
                    id=_FILTER_SELECT_ID,
                    compact=True,
                )
                yield Checkbox(
                    "Click to copy",
                    value=self._click_to_copy,
                    id=_CLICK_TO_COPY_ID,
                    compact=True,
                    classes="debug-console-click-to-copy",
                )
            yield _DebugLogView(
                self._copy_record,
                widget_id="debug-log",
                classes="debug-console-log",
                click_to_copy=self._click_to_copy,
            )
            yield Static(self._render_help(), classes="debug-console-help")

    def on_mount(self) -> None:
        """Start the refresh timer and render the current buffer contents."""
        self.set_interval(_REFRESH_INTERVAL, self._poll_logs)
        self._poll_logs()
        self._resolve_langsmith_links()
        self.call_after_refresh(self.query_one("#debug-log", _DebugLogView).focus)

    def _resolve_langsmith_links(self) -> None:
        """Kick off background resolution of `(langsmith)` links for the snapshot."""
        thread_ids = {
            field.thread_id
            for field in self._snapshot
            if field.thread_id and field.thread_id not in self._langsmith_urls
        }
        for thread_id in thread_ids:
            self.run_worker(
                self._fetch_langsmith_link(thread_id),
                exclusive=False,
                group="debug-console-langsmith",
            )

    async def _fetch_langsmith_link(self, thread_id: str) -> None:
        """Resolve a thread's LangSmith URL and re-render the snapshot.

        Follows the welcome banner's thread + short-timeout pattern so an
        unreachable LangSmith never blocks the console, but splits error
        handling by expectedness: an expected timeout/I/O failure degrades
        quietly to no link, while an unexpected error is logged loudly so a
        genuine resolution bug is not hidden inside the diagnostic overlay.
        """
        from deepagents_code.config import build_langsmith_thread_url

        try:
            url = await asyncio.wait_for(
                asyncio.to_thread(build_langsmith_thread_url, thread_id),
                timeout=2.0,
            )
        except (TimeoutError, OSError):
            # Expected: the outer timeout fired or a network error escaped the
            # helper. A passive convenience link merely fails to appear.
            logger.debug(
                "LangSmith thread URL lookup timed out/failed for %r",
                thread_id,
                exc_info=True,
            )
            return
        except Exception:  # a diagnostic overlay must not crash on a lookup bug
            # Unexpected: a real defect in URL resolution. WARNING (not DEBUG) so
            # the traceback lands in the always-on in-memory buffer and is visible
            # in the console itself; the package logger sits at INFO by default,
            # which drops DEBUG.
            logger.warning(
                "LangSmith thread URL lookup errored unexpectedly for %r",
                thread_id,
                exc_info=True,
            )
            return
        if url:
            self._langsmith_urls[thread_id] = url
            self._refresh_snapshot()

    def _refresh_snapshot(self) -> None:
        """Re-render the snapshot header in place (e.g. after a link resolves)."""
        from textual.css.query import NoMatches

        try:
            self.query_one(".debug-console-snapshot", _SnapshotView).update(
                self._render_snapshot()
            )
        except NoMatches:
            # The console was dismissed before the worker returned.
            logger.debug("Debug console snapshot refresh skipped (widget unavailable)")

    def key_tab(self, event: events.Key) -> None:
        """Cycle focus between the toolbar controls and log lines."""
        if self._level_select().expanded:
            return
        event.prevent_default()
        event.stop()
        self.focus_next(_FOCUS_CYCLE)

    def key_shift_tab(self, event: events.Key) -> None:
        """Cycle focus between the log lines and toolbar controls."""
        if self._level_select().expanded:
            return
        event.prevent_default()
        event.stop()
        self.focus_previous(_FOCUS_CYCLE)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Dismiss transient control state when the user clicks outside it.

        Clicking a focusable control already moves focus, but clicking a
        non-focusable area (the snapshot, labels, help, or empty modal space)
        does not. Mirror that outside-click behavior for the open level dropdown
        and the focused "Click to copy" checkbox.
        """
        offset = event.screen_offset
        select = self._level_select()
        if select.expanded and not self._point_in_level_select(select, offset):
            overlay = select.query_one(SelectOverlay)
            select.expanded = False
            # Re-focus the select only when focus is still trapped on the now
            # hidden overlay; if the click already moved focus to another
            # control, leave it there.
            if self.focused is overlay:
                select.focus()
        checkbox = self.query_one(f"#{_CLICK_TO_COPY_ID}", Checkbox)
        if self.focused is checkbox and not checkbox.region.contains(
            offset.x, offset.y
        ):
            self.set_focus(None)

    @staticmethod
    def _point_in_level_select(select: Select[FilterValue], offset: Offset) -> bool:
        """Return whether *offset* falls on the select box or its open overlay."""
        if select.region.contains(offset.x, offset.y):
            return True
        overlay = select.query_one(SelectOverlay)
        return overlay.display and overlay.region.contains(offset.x, offset.y)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Refresh visible records when the log-level filter changes."""
        if event.select.id != _FILTER_SELECT_ID:
            return
        value = str(event.value)
        if value == self._level_filter:
            return
        if value not in _VALID_FILTER_VALUES:
            # The Select only offers known options, so this is unreachable in
            # practice; validate anyway so an unexpected value degrades to the
            # current filter instead of being trusted as a FilterValue.
            logger.warning("Ignoring unknown debug level filter %r", value)
            return
        self._level_filter = cast("FilterValue", value)
        self._refresh_log_view(scroll_end=True)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Toggle click-to-copy for the log lines.

        The checkbox governs only the log lines; copyable snapshot spans (e.g.
        the thread id) always copy on click regardless of this setting.
        """
        if event.checkbox.id != _CLICK_TO_COPY_ID:
            return
        self._click_to_copy = event.value
        self.query_one("#debug-log", _DebugLogView).click_to_copy = event.value
        if self._on_click_to_copy_change is not None:
            self._on_click_to_copy_change(event.value)

    def _render_snapshot(self) -> Content:
        """Build the right-aligned `label: value` snapshot block.

        Returns:
            The formatted snapshot block.
        """
        if not self._snapshot:
            return Content.styled("(no session data)", "dim italic")
        width = max(len(field.label) for field in self._snapshot)
        lines = [self._render_snapshot_row(field, width) for field in self._snapshot]
        return Content("\n").join(lines)

    def _render_snapshot_row(self, field: SnapshotField, width: int) -> Content:
        """Render a single snapshot row, wiring up copy and link spans.

        Args:
            field: The snapshot field to render.
            width: Column width the labels are right-aligned to.

        Returns:
            The formatted row content.
        """
        parts: list[str | tuple[str, str | TStyle]] = [
            (f"{field.label:>{width}}  ", "bold")
        ]
        if field.copyable and field.value:
            parts.append(
                (
                    field.value,
                    TStyle.from_meta(
                        {
                            _SNAPSHOT_COPY_META: field.value,
                            _SNAPSHOT_COPY_LABEL_META: field.label,
                        }
                    ),
                )
            )
        else:
            parts.append(field.value)
        url = self._langsmith_urls.get(field.thread_id) if field.thread_id else None
        if url:
            parts.extend(("  ", ("(langsmith)", TStyle(link=url))))
        return Content.assemble(*parts)

    @staticmethod
    def _render_help() -> Content:
        """Build the footer key-hint line.

        Returns:
            The formatted key-hint line.
        """
        return Content.styled(
            "Esc close · Ctrl+L clear view · c copy visible logs · Enter copy line",
            "dim italic",
        )

    def _poll_logs(self) -> None:
        """Append log records emitted since the last tick, guarding the timer.

        Runs on a repeating `set_interval` timer, so an unhandled exception here
        would propagate out of the callback and tear down the whole host app.
        A diagnostic overlay must degrade instead: a tick that races teardown
        (`NoMatches`) is logged at DEBUG and skipped, and any other failure
        degrades the tail to a notice rather than crashing the app it exists to
        inspect.
        """
        from textual.css.query import NoMatches

        try:
            self._poll_logs_once()
        except NoMatches:
            # Expected when a queued tick races console teardown: the log widget
            # is already gone. Logged at DEBUG (not swallowed outright) so a
            # genuine missing/mis-typed-widget bug still leaves a breadcrumb in
            # the buffer instead of silently rendering nothing forever.
            logger.debug("Debug console poll skipped (widget unavailable)")
            return
        except Exception:  # a diagnostic must never crash the app it inspects
            logger.warning("Debug console log poll failed", exc_info=True)
            try:
                self.query_one("#debug-log", _DebugLogView).show_notice(
                    "(log tail unavailable)"
                )
            except Exception:  # best-effort notice; never re-raise from here
                logger.debug("Debug console poll-error notice failed", exc_info=True)

    def _poll_logs_once(self) -> None:
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
        pruned = self._prune_records()
        self._rendered_upto = total
        if pruned:
            self._refresh_log_view(scroll_end=log.is_vertical_scroll_end)
            return
        log.append_records(self._visible_records(records))

    def _prune_records(self) -> bool:
        """Trim retained records to the ring buffer capacity, per level.

        Mirrors the buffer's level-partitioned retention: each standard level
        keeps at most `_RECORD_LIMIT` records, while custom levels share the
        buffer's fallback bucket. Only the oldest entries of an over-capacity
        bucket are dropped; chronological order is preserved.

        Returns:
            `True` when records were pruned.
        """
        counts: dict[str, int] = {}
        for record in self._records:
            bucket = retention_bucket_for_level(record.level)
            counts[bucket] = counts.get(bucket, 0) + 1
        overflow = {
            level: count - _RECORD_LIMIT
            for level, count in counts.items()
            if count > _RECORD_LIMIT
        }
        if not overflow:
            return False
        kept: list[InMemoryLogRecord] = []
        for record in self._records:
            bucket = retention_bucket_for_level(record.level)
            remaining = overflow.get(bucket, 0)
            if remaining > 0:
                overflow[bucket] = remaining - 1
                continue
            kept.append(record)
        self._records = kept
        return True

    def _visible_records(
        self, records: Sequence[InMemoryLogRecord]
    ) -> list[InMemoryLogRecord]:
        """Return the subset of *records* matching the current level filter."""
        return [
            record
            for record in records
            if _record_matches_filter(record, self._level_filter)
        ]

    def _refresh_log_view(self, *, scroll_end: bool) -> None:
        """Rebuild the log view using the current filter."""
        self.query_one("#debug-log", _DebugLogView).set_records(
            self._visible_records(self._records), scroll_end=scroll_end
        )

    def action_clear_view(self) -> None:
        """Clear the on-screen log view; the in-memory buffer keeps accruing.

        Advances the render cursor past everything emitted so far and reports it
        via `on_clear` so the owner can persist the clear across close/reopen.
        """
        self.query_one("#debug-log", _DebugLogView).clear_records()
        self._records.clear()
        buffer = get_log_buffer()
        if buffer is not None:
            self._rendered_upto = buffer.total_emitted
        if self._on_clear is not None:
            self._on_clear(self._rendered_upto)

    def action_copy(self) -> None:
        """Copy visible retained log records since the last clear to the clipboard."""
        lines = [record.plain_line for record in self._visible_records(self._records)]
        self._copy_lines(lines, empty_message="No visible log lines to copy")

    def _copy_record(self, record: InMemoryLogRecord) -> None:
        """Copy a clicked logical log record to the clipboard."""
        self._copy_lines([record.plain_line], empty_message="No log line to copy")

    def _copy_snapshot_value(self, text: str, label: str) -> None:
        """Copy a clicked snapshot value to the clipboard.

        Args:
            text: The field value to put on the clipboard.
            label: The snapshot row label used to word the success toast.
        """
        self._copy_lines(
            [text],
            empty_message="Nothing to copy",
            success_message=_snapshot_copy_success_message(label),
        )

    def _level_select(self) -> Select[FilterValue]:
        """Return the level-filter dropdown."""
        return cast(
            "Select[FilterValue]", self.query_one("#debug-level-filter", Select)
        )

    def _copy_lines(
        self,
        lines: Sequence[str],
        *,
        empty_message: str,
        success_message: str = "Debug log copied",
    ) -> None:
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
                success_message, severity="information", timeout=2, markup=False
            )
            return
        suffix = f": {error}" if error else ""
        self.app.notify(
            f"Failed to copy{suffix}",
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
