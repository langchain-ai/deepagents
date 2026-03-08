"""Interactive thread selector screen for /threads command."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
from typing import TYPE_CHECKING, ClassVar, cast

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.fuzzy import Matcher
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Checkbox, Input, Static

if TYPE_CHECKING:
    from collections.abc import Mapping

    from textual.app import ComposeResult
    from textual.events import Click, Key

from deepagents_cli.config import (
    CharsetMode,
    _detect_charset_mode,
    build_langsmith_thread_url,
    get_glyphs,
)
from deepagents_cli.sessions import ThreadInfo
from deepagents_cli.widgets._links import open_style_link

logger = logging.getLogger(__name__)

_COL_TID = 10
_COL_AGENT = 12
_COL_MSGS = 4
_COL_BRANCH = 16
_COL_TIMESTAMP = None
_MAX_SEARCH_TEXT_LEN = 200
_COL_PROMPT = None
_AUTO_WIDTH_COLUMNS = {"agent_name", "created_at", "updated_at"}
_COLUMN_ORDER = (
    "thread_id",
    "agent_name",
    "messages",
    "created_at",
    "updated_at",
    "git_branch",
    "initial_prompt",
)
_COLUMN_WIDTHS: dict[str, int | None] = {
    "thread_id": _COL_TID,
    "agent_name": _COL_AGENT,
    "messages": _COL_MSGS,
    "created_at": _COL_TIMESTAMP,
    "updated_at": _COL_TIMESTAMP,
    "git_branch": _COL_BRANCH,
    "initial_prompt": _COL_PROMPT,
}
_COLUMN_LABELS = {
    "thread_id": "Thread ID",
    "agent_name": "Agent",
    "messages": "Msgs",
    "created_at": "Created",
    "updated_at": "Updated",
    "git_branch": "Branch",
    "initial_prompt": "Prompt",
}
_COLUMN_TOGGLE_LABELS = {
    "thread_id": "Thread ID",
    "agent_name": "Agent Name",
    "messages": "# Messages",
    "created_at": "Created At",
    "updated_at": "Updated At",
    "git_branch": "Git Branch",
    "initial_prompt": "Initial Prompt",
}
# Reserved for future right-aligned columns (e.g., message counts).
_RIGHT_ALIGNED_COLUMNS: set[str] = set()
_SWITCH_ID_PREFIX = "thread-column-"
_SORT_SWITCH_ID = "thread-sort-toggle"
_RELATIVE_TIME_SWITCH_ID = "thread-relative-time"
_CELL_PADDING_RIGHT = 1


def _apply_column_width(
    cell: Static, key: str, column_widths: Mapping[str, int | None]
) -> None:
    """Apply an explicit width to a table cell when one is configured.

    Args:
        cell: The cell widget to size.
        key: Column key for the cell.
        column_widths: Effective column widths for the current table state.
    """
    width = column_widths.get(key)
    if width is not None:
        cell.styles.width = width
        if key in _AUTO_WIDTH_COLUMNS:
            cell.styles.min_width = width


def _active_sort_key(sort_by_updated: bool) -> str:
    """Return the active timestamp field used for sorting."""
    return "updated_at" if sort_by_updated else "created_at"


def _visible_column_keys(columns: dict[str, bool]) -> list[str]:
    """Return visible columns in the on-screen order.

    Args:
        columns: Column visibility settings keyed by column name.

    Returns:
        Visible column keys in display order.
    """
    return [key for key in _COLUMN_ORDER if columns.get(key)]


def _collapse_whitespace(value: str) -> str:
    """Normalize a text value onto a single display line.

    Args:
        value: Raw text to display in a single cell.

    Returns:
        The input text collapsed to a single line.
    """
    return " ".join(value.split())


def _truncate_value(value: str, width: int | None) -> str:
    """Trim text to fit a fixed-width column.

    Args:
        value: Raw cell text.
        width: Maximum column width, or `None` for no truncation.

    Returns:
        The possibly truncated display string.
    """
    if width is None:
        return value

    display = _collapse_whitespace(value)
    if len(display) <= width:
        return display

    glyphs = get_glyphs()
    ellipsis = glyphs.ellipsis
    if width <= len(ellipsis):
        return display[:width]
    return display[: width - len(ellipsis)] + ellipsis


def _format_column_value(
    thread: ThreadInfo, key: str, *, relative_time: bool = False
) -> str:
    """Return the display text for one thread column.

    Args:
        thread: Thread metadata for the row.
        key: Column key to format.
        relative_time: Use relative timestamps instead of absolute.

    Returns:
        Formatted display text for the column cell.
    """
    from deepagents_cli.sessions import format_relative_timestamp, format_timestamp

    fmt = format_relative_timestamp if relative_time else format_timestamp

    value: str
    if key == "thread_id":
        # Strip UUID separators in the compact table preview so truncation
        # never leaves a dangling trailing hyphen in the thread ID column.
        value = thread["thread_id"].replace("-", "")
    elif key == "agent_name":
        value = thread.get("agent_name") or "unknown"
    elif key == "messages":
        raw_count = thread.get("message_count")
        value = str(raw_count) if raw_count is not None else "..."
    elif key == "created_at":
        value = fmt(thread.get("created_at"))
    elif key == "updated_at":
        value = fmt(thread.get("updated_at"))
    elif key == "git_branch":
        value = thread.get("git_branch") or ""
    elif key == "initial_prompt":
        value = _collapse_whitespace(thread.get("initial_prompt") or "")
    else:
        value = ""

    return _truncate_value(value, _COLUMN_WIDTHS[key])


def _format_header_label(key: str) -> str:
    """Return the rendered header label for a column."""
    return _truncate_value(_COLUMN_LABELS[key], _COLUMN_WIDTHS[key])


def _header_cell_classes(key: str, *, sort_key: str) -> str:
    """Return CSS classes for a header cell.

    Args:
        key: Column key for the header cell.
        sort_key: Currently active sort column.

    Returns:
        Space-delimited classes for the header cell widget.
    """
    classes = f"thread-cell thread-cell-{key}"
    if key == sort_key:
        classes += " thread-cell-sorted"
    return classes


class ThreadOption(Horizontal):
    """A clickable thread option in the selector."""

    def __init__(
        self,
        thread: ThreadInfo,
        index: int,
        *,
        columns: dict[str, bool],
        column_widths: Mapping[str, int | None],
        selected: bool,
        current: bool,
        relative_time: bool = False,
        classes: str = "",
    ) -> None:
        """Initialize a thread option row.

        Args:
            thread: Thread metadata for the row.
            index: The index of this option in the filtered list.
            columns: Column visibility settings.
            column_widths: Effective widths for the visible columns.
            selected: Whether the row is highlighted.
            current: Whether the row is the active thread.
            relative_time: Use relative timestamps.
            classes: CSS classes for styling.
        """
        super().__init__(classes=classes)
        self.thread = thread
        self.thread_id = thread["thread_id"]
        self.index = index
        self._columns = dict(columns)
        self._column_widths = dict(column_widths)
        self._selected = selected
        self._current = current
        self._relative_time = relative_time

    class Clicked(Message):
        """Message sent when a thread option is clicked."""

        def __init__(self, thread_id: str, index: int) -> None:
            """Initialize the Clicked message.

            Args:
                thread_id: The thread identifier.
                index: The index of the clicked option.
            """
            super().__init__()
            self.thread_id = thread_id
            self.index = index

    def compose(self) -> ComposeResult:
        """Compose the row cells.

        Yields:
            Static cells for each visible column.
        """
        yield Static(
            self._cursor_text(),
            classes="thread-cell thread-cell-cursor",
            markup=False,
        )
        for key in _visible_column_keys(self._columns):
            cell = Static(
                _format_column_value(
                    self.thread, key, relative_time=self._relative_time
                ),
                classes=f"thread-cell thread-cell-{key}",
                expand=key == "initial_prompt",
                markup=False,
            )
            _apply_column_width(cell, key, self._column_widths)
            yield cell

    def _cursor_text(self) -> str:
        """Return the cursor indicator for the row."""
        return get_glyphs().cursor if self._selected else ""

    def set_selected(self, selected: bool) -> None:
        """Update row selection styling without rebuilding the row.

        Args:
            selected: Whether the row should be highlighted.
        """
        self._selected = selected
        if selected:
            self.add_class("thread-option-selected")
        else:
            self.remove_class("thread-option-selected")

        try:
            cursor = self.query_one(".thread-cell-cursor", Static)
        except NoMatches:
            return
        cursor.update(self._cursor_text())

    def on_click(self, event: Click) -> None:
        """Handle click on this option.

        Args:
            event: The click event.
        """
        event.stop()
        self.post_message(self.Clicked(self.thread_id, self.index))


class DeleteThreadConfirmScreen(ModalScreen[bool]):
    """Confirmation modal shown before deleting a thread."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "confirm", "Confirm", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    CSS = """
    DeleteThreadConfirmScreen {
        align: center middle;
    }

    DeleteThreadConfirmScreen > Vertical {
        width: 50;
        height: auto;
        background: $surface;
        border: solid red;
        padding: 1 2;
    }

    DeleteThreadConfirmScreen .thread-confirm-text {
        text-align: center;
        margin-bottom: 1;
    }

    DeleteThreadConfirmScreen .thread-confirm-help {
        text-align: center;
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(self, thread_id: str) -> None:
        """Initialize the confirmation modal.

        Args:
            thread_id: Thread ID the user is being asked to delete.
        """
        super().__init__()
        self._delete_thread_id = thread_id

    def compose(self) -> ComposeResult:
        """Compose the confirmation dialog.

        Yields:
            Widgets for the delete confirmation prompt.
        """
        with Vertical(id="delete-confirm"):
            yield Static(
                f"Delete thread [bold]{self._delete_thread_id}[/bold]?",
                classes="thread-confirm-text",
            )
            yield Static(
                "Enter to confirm, Esc to cancel",
                classes="thread-confirm-help",
            )

    def action_confirm(self) -> None:
        """Confirm deletion."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Cancel deletion."""
        self.dismiss(False)


class ThreadSelectorScreen(ModalScreen[str | None]):
    """Modal dialog for browsing and resuming threads.

    Displays recent threads with keyboard navigation, fuzzy search,
    configurable columns, and delete support.

    Returns a `thread_id` string on selection, or `None` on cancel.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False, priority=True),
        Binding("k", "move_up", "Up", show=False, priority=True),
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("j", "move_down", "Down", show=False, priority=True),
        Binding("pageup", "page_up", "Page up", show=False, priority=True),
        Binding("pagedown", "page_down", "Page down", show=False, priority=True),
        Binding("enter", "select", "Select", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        Binding("ctrl+d", "delete_thread", "Delete", show=False, priority=True),
        Binding("tab", "focus_next_filter", "Next filter", show=False, priority=True),
        Binding(
            "shift+tab",
            "focus_previous_filter",
            "Previous filter",
            show=False,
            priority=True,
        ),
    ]

    CSS = """
    ThreadSelectorScreen {
        align: center middle;
    }

    ThreadSelectorScreen #thread-selector-shell {
        width: 100%;
        max-width: 98%;
        height: 90%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    ThreadSelectorScreen .thread-selector-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    ThreadSelectorScreen #thread-filter {
        margin-bottom: 1;
        border: solid $primary-lighten-2;
    }

    ThreadSelectorScreen #thread-filter:focus {
        border: solid $primary;
    }

    ThreadSelectorScreen .thread-selector-body {
        height: 1fr;
    }

    ThreadSelectorScreen .thread-table-pane {
        width: 1fr;
        min-width: 40;
        height: 1fr;
    }

    ThreadSelectorScreen .thread-controls {
        width: 28;
        min-width: 24;
        height: 1fr;
        margin-left: 1;
        padding-left: 1;
        border-left: solid $primary-lighten-2;
    }

    ThreadSelectorScreen .thread-controls-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    ThreadSelectorScreen .thread-controls-help {
        color: $text-muted;
        margin-bottom: 1;
    }

    ThreadSelectorScreen .thread-column-toggle {
        width: 1fr;
        height: auto;
    }

    ThreadSelectorScreen .thread-list-header {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        text-style: bold;
        width: 100%;
        overflow-x: hidden;
    }

    ThreadSelectorScreen .thread-list-header .thread-cell-sorted {
        color: $primary;
    }

    ThreadSelectorScreen .thread-list {
        height: 1fr;
        min-height: 5;
        scrollbar-gutter: stable;
        background: $background;
    }

    ThreadSelectorScreen .thread-option {
        height: 1;
        width: 100%;
        padding: 0 1;
        overflow-x: hidden;
    }

    ThreadSelectorScreen .thread-option:hover {
        background: $surface-lighten-1;
    }

    ThreadSelectorScreen .thread-option-selected {
        background: $primary;
        text-style: bold;
    }

    ThreadSelectorScreen .thread-option-selected:hover {
        background: $primary-lighten-1;
    }

    ThreadSelectorScreen .thread-option-current {
        text-style: italic;
    }

    ThreadSelectorScreen .thread-cell {
        height: 1;
        padding-right: 1;
    }

    ThreadSelectorScreen .thread-cell-cursor {
        width: 2;
        color: $primary;
    }

    ThreadSelectorScreen .thread-cell-thread_id {
        width: 10;
    }

    ThreadSelectorScreen .thread-cell-agent_name {
        width: auto;
        overflow-x: hidden;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    ThreadSelectorScreen .thread-cell-messages {
        width: 4;
    }

    ThreadSelectorScreen .thread-cell-created_at,
    ThreadSelectorScreen .thread-cell-updated_at {
        width: auto;
    }

    ThreadSelectorScreen .thread-cell-git_branch {
        width: 17;
        overflow-x: hidden;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    ThreadSelectorScreen .thread-cell-initial_prompt {
        width: 1fr;
        min-width: 1;
        overflow-x: hidden;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    ThreadSelectorScreen .thread-selector-help {
        height: auto;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }

    ThreadSelectorScreen .thread-empty {
        color: $text-muted;
        text-align: center;
        margin-top: 2;
    }

    """

    def __init__(
        self,
        current_thread: str | None = None,
        *,
        thread_limit: int | None = None,
        initial_threads: list[ThreadInfo] | None = None,
    ) -> None:
        """Initialize the `ThreadSelectorScreen`.

        Args:
            current_thread: The currently active thread ID (to highlight).
            thread_limit: Maximum number of rows to fetch when querying DB.
            initial_threads: Optional preloaded rows to render immediately.
        """
        super().__init__()
        self._current_thread = current_thread
        self._thread_limit = thread_limit
        self._threads: list[ThreadInfo] = (
            [ThreadInfo(**thread) for thread in initial_threads]
            if initial_threads is not None
            else []
        )
        self._filtered_threads: list[ThreadInfo] = list(self._threads)
        self._has_initial_threads = initial_threads is not None
        self._selected_index = 0
        self._option_widgets: list[ThreadOption] = []
        self._filter_text = ""
        self._confirming_delete = False
        self._render_lock = asyncio.Lock()
        self._filter_input: Input | None = None
        self._filter_controls: list[Input | Checkbox] | None = None

        from deepagents_cli.model_config import (
            load_thread_columns,
            load_thread_relative_time,
            load_thread_sort_order,
        )

        self._columns = load_thread_columns()
        self._relative_time = load_thread_relative_time()
        self._sort_by_updated = load_thread_sort_order() == "updated_at"

        self._sync_selected_index()
        self._column_widths = self._compute_column_widths()

    @staticmethod
    def _switch_id(column_key: str) -> str:
        """Return the DOM id for a column toggle switch."""
        return f"{_SWITCH_ID_PREFIX}{column_key}"

    @staticmethod
    def _switch_column_key(switch_id: str | None) -> str | None:
        """Extract the column key from a switch id.

        Args:
            switch_id: Widget id for a switch in the control panel.

        Returns:
            The corresponding column key, or `None` for unrelated ids.
        """
        if not switch_id or not switch_id.startswith(_SWITCH_ID_PREFIX):
            return None
        return switch_id.removeprefix(_SWITCH_ID_PREFIX)

    def _sync_selected_index(self) -> None:
        """Select the current thread when it exists in the loaded rows."""
        self._selected_index = 0
        for i, thread in enumerate(self._filtered_threads):
            if thread["thread_id"] == self._current_thread:
                self._selected_index = i
                break

    def _build_title(self, thread_url: str | None = None) -> str | Text:
        """Build the title, optionally with a clickable thread ID link.

        Args:
            thread_url: LangSmith thread URL. When provided, the thread ID is
                rendered as a clickable hyperlink.

        Returns:
            Plain string or Rich `Text` with an embedded hyperlink.
        """
        if not self._current_thread:
            return "Select Thread"
        if thread_url:
            return Text.assemble(
                "Select Thread (current: ",
                (self._current_thread, Style(color="cyan", link=thread_url)),
                ")",
            )
        return f"Select Thread (current: {self._current_thread})"

    def _build_help_text(self) -> str:
        """Build the footer help text for the selector.

        Returns:
            Footer guidance for the active selector bindings.
        """
        glyphs = get_glyphs()
        lines = (
            f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate"
            f" {glyphs.bullet} Enter select"
            f" {glyphs.bullet} Tab/Shift+Tab focus options"
            f" {glyphs.bullet} Space toggle option"
            f" {glyphs.bullet} Ctrl+D delete"
            f" {glyphs.bullet} Esc cancel"
        )
        limit = self._effective_thread_limit()
        if len(self._threads) >= limit:
            lines += (
                f"\nShowing last {limit} threads. "
                "Set DA_CLI_RECENT_THREADS to override."
            )
        return lines

    def _effective_thread_limit(self) -> int:
        """Return the resolved thread limit for display purposes."""
        if self._thread_limit is not None:
            return self._thread_limit
        from deepagents_cli.sessions import get_thread_limit

        return get_thread_limit()

    def _format_sort_toggle_label(self) -> str:
        """Return the control-panel sort label for the toggle switch."""
        label = "Updated At" if self._sort_by_updated else "Created At"
        return f"Sort by {label}"

    def _get_filter_input(self) -> Input:
        """Return the cached search input widget."""
        if self._filter_input is None:
            self._filter_input = self.query_one("#thread-filter", Input)
        return self._filter_input

    def _filter_focus_order(self) -> list[Input | Checkbox]:
        """Return the cached tab order for filter controls in the side panel."""
        if self._filter_controls is None:
            filter_input = self._get_filter_input()
            sort_switch = self.query_one(f"#{_SORT_SWITCH_ID}", Checkbox)
            relative_switch = self.query_one(f"#{_RELATIVE_TIME_SWITCH_ID}", Checkbox)
            column_switches = [
                self.query_one(f"#{self._switch_id(key)}", Checkbox)
                for key in _COLUMN_ORDER
            ]
            self._filter_controls = [
                filter_input,
                sort_switch,
                relative_switch,
                *column_switches,
            ]
        return self._filter_controls

    def compose(self) -> ComposeResult:
        """Compose the screen layout.

        Yields:
            Widgets for the thread selector UI.
        """
        with Vertical(id="thread-selector-shell"):
            yield Static(
                self._build_title(), classes="thread-selector-title", id="thread-title"
            )

            yield Input(
                placeholder="Type to search threads...",
                select_on_focus=False,
                id="thread-filter",
            )

            with Horizontal(classes="thread-selector-body"):
                with Vertical(classes="thread-table-pane"):
                    with Horizontal(
                        classes="thread-list-header",
                        id="thread-header",
                    ):
                        yield Static("", classes="thread-cell thread-cell-cursor")
                        sort_key = _active_sort_key(self._sort_by_updated)
                        for key in _visible_column_keys(self._columns):
                            cell = Static(
                                _format_header_label(key),
                                classes=_header_cell_classes(key, sort_key=sort_key),
                                expand=key == "initial_prompt",
                                markup=False,
                            )
                            _apply_column_width(cell, key, self._column_widths)
                            yield cell

                    with VerticalScroll(classes="thread-list"):
                        if self._has_initial_threads:
                            if self._filtered_threads:
                                self._option_widgets, _ = self._create_option_widgets()
                                yield from self._option_widgets
                            else:
                                yield Static(
                                    "[dim]No threads found[/dim]",
                                    classes="thread-empty",
                                )
                        else:
                            yield Static(
                                "[dim]Loading threads...[/dim]",
                                classes="thread-empty",
                                id="thread-loading",
                            )

                with Vertical(classes="thread-controls"):
                    yield Static("Options", classes="thread-controls-title")
                    yield Static(
                        (
                            "Tab through sort and column toggles. "
                            "Column visibility persists between sessions."
                        ),
                        classes="thread-controls-help",
                        markup=False,
                    )
                    yield Checkbox(
                        self._format_sort_toggle_label(),
                        self._sort_by_updated,
                        id=_SORT_SWITCH_ID,
                        classes="thread-column-toggle",
                        compact=True,
                    )
                    yield Checkbox(
                        "Relative Timestamps",
                        self._relative_time,
                        id=_RELATIVE_TIME_SWITCH_ID,
                        classes="thread-column-toggle",
                        compact=True,
                    )
                    for key in _COLUMN_ORDER:
                        yield Checkbox(
                            _COLUMN_TOGGLE_LABELS[key],
                            self._columns.get(key, False),
                            id=self._switch_id(key),
                            classes="thread-column-toggle",
                            compact=True,
                        )

            yield Static(
                self._build_help_text(),
                classes="thread-selector-help",
                id="thread-help",
            )

    async def on_mount(self) -> None:
        """Fetch threads, configure border for ASCII terminals, and build the list."""
        if _detect_charset_mode() == CharsetMode.ASCII:
            container = self.query_one("#thread-selector-shell", Vertical)
            container.styles.border = ("ascii", "green")

        filter_input = self._get_filter_input()
        self._filter_focus_order()
        filter_input.focus()

        if self._has_initial_threads:
            self.call_after_refresh(self._scroll_selected_into_view)
            if self._current_thread:
                self._resolve_thread_url()

        # _load_threads replaces self._threads and schedules background
        # enrichment (message counts, initial prompts) after load completes.
        # Avoid eagerly scheduling enrichment on stale initial_threads that
        # will be replaced.
        self.run_worker(
            self._load_threads, exclusive=True, group="thread-selector-load"
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter threads as user types.

        Args:
            event: The input changed event.
        """
        self._filter_text = event.value
        self._schedule_filter_and_rebuild()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key when filter input is focused.

        Args:
            event: The input submitted event.
        """
        event.stop()
        self.action_select()

    def on_key(self, event: Key) -> None:
        """Return focus to search when letters are typed from other controls.

        Args:
            event: The key event.
        """
        if self._confirming_delete:
            return

        filter_input = self._get_filter_input()
        if filter_input.has_focus:
            return

        character = event.character
        if not character or not character.isalpha():
            return

        filter_input.focus()
        filter_input.insert_text_at_cursor(character)
        self.set_timer(0.01, self._collapse_search_selection)
        event.stop()

    def _collapse_search_selection(self) -> None:
        """Place the search cursor at the end without an active selection."""
        filter_input = self._get_filter_input()
        filter_input.selection = type(filter_input.selection).cursor(
            len(filter_input.value)
        )

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Route sort, relative-time, and column-visibility checkbox changes.

        Args:
            event: The checkbox change event.
        """
        if event.checkbox.id == _SORT_SWITCH_ID:
            if self._sort_by_updated == event.value:
                return
            self._sort_by_updated = event.value
            self._apply_sort()
            self._sync_selected_index()
            self._update_help_widgets()
            self._schedule_list_rebuild()

            self._persist_sort_order("updated_at" if event.value else "created_at")
            return

        if event.checkbox.id == _RELATIVE_TIME_SWITCH_ID:
            if self._relative_time == event.value:
                return
            self._relative_time = event.value

            from deepagents_cli.model_config import save_thread_relative_time

            self.run_worker(
                asyncio.to_thread(save_thread_relative_time, event.value),
                group="thread-selector-save",
            )
            self._schedule_list_rebuild()
            return

        column_key = self._switch_column_key(event.checkbox.id)
        if column_key is None or column_key not in self._columns:
            return
        if self._columns[column_key] == event.value:
            return

        self._columns[column_key] = event.value
        self._apply_sort()
        self._sync_selected_index()
        self._update_help_widgets()
        if event.value and column_key in {"messages", "initial_prompt"}:
            self._schedule_checkpoint_enrichment()

        from deepagents_cli.model_config import save_thread_columns

        snapshot = dict(self._columns)
        self.run_worker(
            asyncio.to_thread(save_thread_columns, snapshot),
            group="thread-selector-save",
        )
        self._schedule_list_rebuild()

    def _update_filtered_list(self) -> None:
        """Update filtered threads based on search text using fuzzy matching."""
        query = self._filter_text.strip()
        if not query:
            self._filtered_threads = list(self._threads)
            self._apply_sort()
            self._sync_selected_index()
            self._column_widths = self._compute_column_widths()
            return

        tokens = query.split()
        try:
            matchers = [Matcher(token, case_sensitive=False) for token in tokens]
            scored: list[tuple[float, ThreadInfo]] = []
            for thread in self._threads:
                search_text = self._get_search_text(thread)
                scores = [matcher.match(search_text) for matcher in matchers]
                if all(score > 0 for score in scores):
                    scored.append((min(scores), thread))
        except Exception:
            logger.warning(
                "Fuzzy matcher failed for query %r, falling back to full list",
                query,
                exc_info=True,
            )
            self._filtered_threads = list(self._threads)
            self._apply_sort()
            self._sync_selected_index()
            self._column_widths = self._compute_column_widths()
            return

        sort_key = _active_sort_key(self._sort_by_updated)
        self._filtered_threads = [
            thread
            for _, thread in sorted(
                scored,
                key=lambda item: (
                    item[0],
                    item[1].get(sort_key) or "",
                    item[1].get("updated_at") or "",
                    item[1]["thread_id"],
                ),
                reverse=True,
            )
        ]
        self._selected_index = 0
        self._column_widths = self._compute_column_widths()

    def _compute_column_widths(self) -> dict[str, int | None]:
        """Return effective widths for the current table state.

        The auto-width columns stay dynamic, but they must share one width
        across the header and all visible rows. Textual's `width: auto`
        computes per-widget widths, so the screen computes those shared widths
        from the visible data instead.

        Returns:
            Dict mapping column keys to their effective pixel widths, with
                `None` for flex columns.
        """
        widths = dict(_COLUMN_WIDTHS)

        for key in _AUTO_WIDTH_COLUMNS:
            if not self._columns.get(key):
                continue
            labels = [_format_header_label(key)]
            labels.extend(
                _format_column_value(thread, key, relative_time=self._relative_time)
                for thread in self._filtered_threads
            )
            widths[key] = max((cell_len(label) for label in labels), default=1) + (
                _CELL_PADDING_RIGHT
            )

        return widths

    @staticmethod
    def _get_search_text(thread: ThreadInfo) -> str:
        """Build searchable text from thread fields.

        The result is capped at `_MAX_SEARCH_TEXT_LEN` characters so that
        Textual's fuzzy `Matcher` (which uses recursive backtracking) does
        not hit exponential performance on long initial prompts with
        repeated characters.

        Args:
            thread: Thread metadata.

        Returns:
            Concatenated searchable string, truncated to a safe length.
        """
        parts = [
            thread["thread_id"],
            thread.get("agent_name") or "",
            thread.get("git_branch") or "",
            thread.get("initial_prompt") or "",
        ]
        text = " ".join(parts)
        return text[:_MAX_SEARCH_TEXT_LEN]

    def _schedule_filter_and_rebuild(self) -> None:
        """Queue a filter + rebuild, coalescing rapid keystrokes."""
        self.run_worker(
            self._filter_and_build,
            exclusive=True,
            group="thread-selector-render",
        )

    async def _filter_and_build(self) -> None:
        """Run fuzzy filtering in a thread then rebuild the list."""
        query = self._filter_text.strip()
        threads = list(self._threads)
        sort_by_updated = self._sort_by_updated

        filtered = await asyncio.to_thread(
            self._compute_filtered, query, threads, sort_by_updated
        )
        self._filtered_threads = filtered
        if query:
            self._selected_index = 0
        else:
            self._sync_selected_index()
        self._column_widths = self._compute_column_widths()
        await self._build_list(recompute_widths=False)

    @staticmethod
    def _compute_filtered(
        query: str,
        threads: list[ThreadInfo],
        sort_by_updated: bool,
    ) -> list[ThreadInfo]:
        """Compute filtered thread list off the main thread.

        Args:
            query: Current search query text.
            threads: Full thread list snapshot.
            sort_by_updated: Whether to sort by `updated_at`.

        Returns:
            Filtered and sorted thread list.
        """
        sort_key = _active_sort_key(sort_by_updated)

        if not query:
            result = list(threads)
            result.sort(key=lambda t: t.get(sort_key) or "", reverse=True)
            return result

        tokens = query.split()
        try:
            matchers = [Matcher(token, case_sensitive=False) for token in tokens]
            scored: list[tuple[float, ThreadInfo]] = []
            for thread in threads:
                search_text = ThreadSelectorScreen._get_search_text(thread)
                scores = [matcher.match(search_text) for matcher in matchers]
                if all(score > 0 for score in scores):
                    scored.append((min(scores), thread))
        except Exception:
            logger.warning(
                "Fuzzy matcher failed for query %r, falling back to full list",
                query,
                exc_info=True,
            )
            result = list(threads)
            result.sort(key=lambda t: t.get(sort_key) or "", reverse=True)
            return result

        return [
            thread
            for _, thread in sorted(
                scored,
                key=lambda item: (
                    item[0],
                    item[1].get(sort_key) or "",
                    item[1].get("updated_at") or "",
                    item[1]["thread_id"],
                ),
                reverse=True,
            )
        ]

    def _schedule_list_rebuild(self) -> None:
        """Queue a list rebuild, coalescing rapid updates."""
        self.run_worker(
            self._build_list,
            exclusive=True,
            group="thread-selector-render",
        )

    def _pending_checkpoint_fields(self) -> tuple[bool, bool]:
        """Return which visible checkpoint-derived fields still need loading."""
        load_counts = self._columns.get("messages", False) and any(
            "message_count" not in thread for thread in self._threads
        )
        load_prompts = self._columns.get("initial_prompt", False) and any(
            "initial_prompt" not in thread for thread in self._threads
        )
        return load_counts, load_prompts

    async def _populate_visible_checkpoint_details(self) -> tuple[bool, bool]:
        """Load any still-missing checkpoint-derived fields for visible columns.

        Returns:
            Tuple indicating whether message counts and prompts were requested.
        """
        from deepagents_cli.sessions import populate_thread_checkpoint_details

        load_counts, load_prompts = self._pending_checkpoint_fields()
        if not load_counts and not load_prompts:
            return False, False

        await populate_thread_checkpoint_details(
            self._threads,
            include_message_count=load_counts,
            include_initial_prompt=load_prompts,
        )
        return load_counts, load_prompts

    def _schedule_checkpoint_enrichment(self) -> None:
        """Schedule one checkpoint-enrichment pass for missing row fields."""
        has_missing_counts, has_missing_prompts = self._pending_checkpoint_fields()
        if not has_missing_counts and not has_missing_prompts:
            return
        self.run_worker(
            self._load_checkpoint_details,
            exclusive=True,
            group="thread-selector-checkpoints",
        )

    async def _load_threads(self) -> None:
        """Load thread rows first, then kick off background enrichment."""
        from deepagents_cli.sessions import (
            apply_cached_thread_initial_prompts,
            apply_cached_thread_message_counts,
            list_threads,
        )

        try:
            limit = self._thread_limit
            if limit is None:
                from deepagents_cli.sessions import get_thread_limit

                limit = get_thread_limit()
            self._threads = await list_threads(limit=limit, include_message_count=False)
        except (OSError, sqlite3.Error) as exc:
            logger.exception("Failed to load threads for thread selector")
            await self._show_mount_error(str(exc))
            return
        except Exception as exc:
            logger.exception("Unexpected error loading threads for thread selector")
            await self._show_mount_error(str(exc))
            return

        apply_cached_thread_message_counts(self._threads)
        apply_cached_thread_initial_prompts(self._threads)
        if not self._has_initial_threads:
            try:
                await self._populate_visible_checkpoint_details()
            except (OSError, sqlite3.Error):
                logger.debug(
                    "Could not preload checkpoint details for thread selector",
                    exc_info=True,
                )
            except Exception:
                logger.warning(
                    "Unexpected error preloading checkpoint details "
                    "for thread selector",
                    exc_info=True,
                )
        self._update_filtered_list()
        self._sync_selected_index()

        await self._build_list()

        self._schedule_checkpoint_enrichment()

        if self._current_thread:
            self._resolve_thread_url()

    async def _load_checkpoint_details(self) -> None:
        """Populate checkpoint-derived thread fields in one background pass."""
        if not self._threads:
            return

        try:
            _, load_prompts = await self._populate_visible_checkpoint_details()
        except (OSError, sqlite3.Error):
            logger.debug(
                "Could not load checkpoint details for thread selector",
                exc_info=True,
            )
            return
        except Exception:
            logger.warning(
                "Unexpected error loading checkpoint details for thread selector",
                exc_info=True,
            )
            return

        if load_prompts and self._filter_text.strip():
            # Prompts may affect fuzzy match results; rebuild the filtered
            # list but preserve the user's cursor position.
            saved_tid = (
                self._filtered_threads[self._selected_index]["thread_id"]
                if self._selected_index < len(self._filtered_threads)
                else None
            )
            self._update_filtered_list()
            if saved_tid is not None:
                for i, thread in enumerate(self._filtered_threads):
                    if thread["thread_id"] == saved_tid:
                        self._selected_index = i
                        break
            self._schedule_list_rebuild()
        else:
            self._refresh_cell_labels()

    def _refresh_cell_labels(self) -> None:
        """Update visible cell text in-place without rebuilding the DOM."""
        for widget in self._option_widgets:
            thread = widget.thread
            for key in _visible_column_keys(self._columns):
                try:
                    cell = widget.query_one(f".thread-cell-{key}", Static)
                except NoMatches:
                    continue
                cell.update(
                    _format_column_value(thread, key, relative_time=self._relative_time)
                )

    def _resolve_thread_url(self) -> None:
        """Start exclusive background worker to resolve LangSmith thread URL."""
        self.run_worker(
            self._fetch_thread_url, exclusive=True, group="thread-selector-url"
        )

    async def _fetch_thread_url(self) -> None:
        """Resolve the LangSmith URL and update the title with a clickable link."""
        if not self._current_thread:
            return
        try:
            thread_url = await asyncio.wait_for(
                asyncio.to_thread(build_langsmith_thread_url, self._current_thread),
                timeout=2.0,
            )
        except (TimeoutError, OSError):
            logger.debug(
                "Could not resolve LangSmith thread URL for '%s'",
                self._current_thread,
                exc_info=True,
            )
            return
        except Exception:
            logger.debug(
                "Unexpected error resolving LangSmith thread URL for '%s'",
                self._current_thread,
                exc_info=True,
            )
            return
        if thread_url:
            try:
                title_widget = self.query_one("#thread-title", Static)
                title_widget.update(self._build_title(thread_url))
            except NoMatches:
                logger.debug(
                    "Title widget #thread-title not found; "
                    "thread selector may have been dismissed during URL resolution"
                )

    async def _show_mount_error(self, detail: str) -> None:
        """Display an error message inside the thread list and refocus.

        Args:
            detail: Human-readable error detail to show.
        """
        try:
            async with self._render_lock:
                scroll = self.query_one(".thread-list", VerticalScroll)
                await scroll.remove_children()
                await scroll.mount(
                    Static(
                        (
                            f"[red]Failed to load threads: {detail}. "
                            "Press Esc to close.[/red]"
                        ),
                        classes="thread-empty",
                    )
                )
        except Exception:
            logger.warning(
                "Could not display error message in thread selector UI",
                exc_info=True,
            )
        self.focus()

    async def _build_list(self, *, recompute_widths: bool = True) -> None:
        """Build the thread option widgets.

        Args:
            recompute_widths: Whether to recalculate shared column widths first.
        """
        async with self._render_lock:
            try:
                scroll = self.query_one(".thread-list", VerticalScroll)
            except NoMatches:
                return

            if recompute_widths:
                self._column_widths = self._compute_column_widths()
            with self.app.batch_update():
                await scroll.remove_children()
                self._update_help_widgets()

                if not self._filtered_threads:
                    self._option_widgets = []
                    await scroll.mount(
                        Static(
                            "[dim]No threads found[/dim]",
                            classes="thread-empty",
                        )
                    )
                    return

                self._option_widgets, selected_widget = self._create_option_widgets()
                await scroll.mount(*self._option_widgets)

            if selected_widget:
                self.call_after_refresh(self._scroll_selected_into_view)

    def _create_option_widgets(self) -> tuple[list[ThreadOption], ThreadOption | None]:
        """Build option widgets from filtered threads without mounting.

        Returns:
            Tuple of all option widgets and the currently selected widget.
        """
        widgets: list[ThreadOption] = []
        selected_widget: ThreadOption | None = None

        for i, thread in enumerate(self._filtered_threads):
            is_current = thread["thread_id"] == self._current_thread
            is_selected = i == self._selected_index

            classes = "thread-option"
            if is_selected:
                classes += " thread-option-selected"
            if is_current:
                classes += " thread-option-current"

            widget = ThreadOption(
                thread=thread,
                index=i,
                columns=self._columns,
                column_widths=self._column_widths,
                selected=is_selected,
                current=is_current,
                relative_time=self._relative_time,
                classes=classes,
            )
            widgets.append(widget)
            if is_selected:
                selected_widget = widget

        return widgets, selected_widget

    def _scroll_selected_into_view(self) -> None:
        """Scroll selected option into view without animation."""
        if not self._option_widgets:
            return
        if self._selected_index >= len(self._option_widgets):
            return
        try:
            scroll = self.query_one(".thread-list", VerticalScroll)
        except NoMatches:
            return

        if self._selected_index == 0:
            scroll.scroll_home(animate=False)
        else:
            self._option_widgets[self._selected_index].scroll_visible(animate=False)

    def _update_help_widgets(self) -> None:
        """Update visible header and help text after state changes."""
        self._schedule_header_rebuild()

        try:
            help_widget = self.query_one("#thread-help", Static)
            help_widget.update(self._build_help_text())
        except NoMatches:
            logger.debug("Help widget #thread-help not found during update")

        with contextlib.suppress(NoMatches):
            sort_checkbox = self.query_one(f"#{_SORT_SWITCH_ID}", Checkbox)
            sort_checkbox.label = self._format_sort_toggle_label()
            if sort_checkbox.value != self._sort_by_updated:
                sort_checkbox.value = self._sort_by_updated

    def _schedule_header_rebuild(self) -> None:
        """Queue a header rebuild to reflect column/sort changes."""
        self.run_worker(
            self._rebuild_header,
            exclusive=True,
            group="thread-selector-header",
        )

    async def _rebuild_header(self) -> None:
        """Replace header cells to match current visible columns."""
        try:
            header = self.query_one("#thread-header", Horizontal)
        except NoMatches:
            return
        sort_key = _active_sort_key(self._sort_by_updated)
        self._column_widths = self._compute_column_widths()
        with self.app.batch_update():
            await header.remove_children()
            cells: list[Static] = [Static("", classes="thread-cell thread-cell-cursor")]
            for key in _visible_column_keys(self._columns):
                cell = Static(
                    _format_header_label(key),
                    classes=_header_cell_classes(key, sort_key=sort_key),
                    expand=key == "initial_prompt",
                    markup=False,
                )
                _apply_column_width(cell, key, self._column_widths)
                cells.append(cell)
            await header.mount(*cells)

    def _apply_sort(self) -> None:
        """Sort filtered threads by the active sort key."""
        key = _active_sort_key(self._sort_by_updated)
        self._filtered_threads.sort(
            key=lambda thread: thread.get(key) or "", reverse=True
        )

    def _move_selection(self, delta: int) -> None:
        """Move selection by delta, updating only the affected rows.

        Args:
            delta: Positions to move (negative for up, positive for down).
        """
        if not self._filtered_threads or not self._option_widgets:
            return

        count = len(self._filtered_threads)
        old_index = self._selected_index
        new_index = (old_index + delta) % count
        self._selected_index = new_index

        self._option_widgets[old_index].set_selected(False)
        self._option_widgets[new_index].set_selected(True)

        if new_index == 0:
            scroll = self.query_one(".thread-list", VerticalScroll)
            scroll.scroll_home(animate=False)
        else:
            self._option_widgets[new_index].scroll_visible()

    def action_move_up(self) -> None:
        """Move selection up."""
        if self._confirming_delete:
            return
        self._move_selection(-1)

    def action_move_down(self) -> None:
        """Move selection down."""
        if self._confirming_delete:
            return
        self._move_selection(1)

    def _visible_page_size(self) -> int:
        """Return the number of thread options that fit in one visual page.

        Returns:
            Number of thread options per page, at least 1.
        """
        default_page_size = 10
        try:
            scroll = self.query_one(".thread-list", VerticalScroll)
            height = scroll.size.height
        except NoMatches:
            logger.debug(
                "Thread list widget not found in _visible_page_size; "
                "using default page size %d",
                default_page_size,
            )
            return default_page_size
        if height <= 0:
            return default_page_size
        return max(1, height)

    def action_page_up(self) -> None:
        """Move selection up by one visible page."""
        if self._confirming_delete or not self._filtered_threads:
            return
        page = self._visible_page_size()
        target = max(0, self._selected_index - page)
        delta = target - self._selected_index
        if delta != 0:
            self._move_selection(delta)

    def action_page_down(self) -> None:
        """Move selection down by one visible page."""
        if self._confirming_delete or not self._filtered_threads:
            return
        count = len(self._filtered_threads)
        page = self._visible_page_size()
        target = min(count - 1, self._selected_index + page)
        delta = target - self._selected_index
        if delta != 0:
            self._move_selection(delta)

    def action_select(self) -> None:
        """Confirm the highlighted thread and dismiss the selector."""
        if self._confirming_delete:
            return
        if self._filtered_threads:
            thread_id = self._filtered_threads[self._selected_index]["thread_id"]
            self.dismiss(thread_id)

    def action_focus_next_filter(self) -> None:
        """Move focus through the filter and column-toggle controls."""
        if self._confirming_delete:
            return
        controls = self._filter_focus_order()
        focused = self.focused
        if focused not in controls:
            controls[0].focus()
            return

        index = controls.index(cast("Input | Checkbox", focused))
        controls[(index + 1) % len(controls)].focus()

    def action_focus_previous_filter(self) -> None:
        """Move focus backward through the filter and column-toggle controls."""
        if self._confirming_delete:
            return
        controls = self._filter_focus_order()
        focused = self.focused
        if focused not in controls:
            controls[-1].focus()
            return

        index = controls.index(cast("Input | Checkbox", focused))
        controls[(index - 1) % len(controls)].focus()

    def action_toggle_sort(self) -> None:
        """Toggle sort between updated_at and created_at."""
        if self._confirming_delete:
            return
        self._sort_by_updated = not self._sort_by_updated
        self._apply_sort()
        self._sync_selected_index()
        self._update_help_widgets()
        self._schedule_list_rebuild()

        self._persist_sort_order(
            "updated_at" if self._sort_by_updated else "created_at"
        )

    def _persist_sort_order(self, order: str) -> None:
        """Save sort-order preference to config, notifying on failure."""

        async def _save() -> None:
            from deepagents_cli.model_config import save_thread_sort_order

            ok = await asyncio.to_thread(save_thread_sort_order, order)
            if not ok:
                self.app.notify("Could not save sort preference", severity="warning")

        self.run_worker(_save(), group="thread-selector-save")

    def action_delete_thread(self) -> None:
        """Show delete confirmation for the highlighted thread."""
        if self._confirming_delete or not self._filtered_threads:
            return
        self._confirming_delete = True
        thread = self._filtered_threads[self._selected_index]
        tid = thread["thread_id"]
        self.app.push_screen(
            DeleteThreadConfirmScreen(tid),
            lambda confirmed: self._on_delete_confirmed(tid, confirmed),
        )

    @property
    def is_delete_confirmation_open(self) -> bool:
        """Return whether the delete confirmation overlay is visible."""
        return self._confirming_delete

    def _on_delete_confirmed(self, thread_id: str, confirmed: bool | None) -> None:
        """Handle the result from the delete confirmation modal.

        Args:
            thread_id: Thread ID that was targeted.
            confirmed: Whether deletion was confirmed.
        """
        self._confirming_delete = False
        if confirmed:
            self.run_worker(
                self._handle_delete_confirm(thread_id),
                group="thread-delete-execute",
            )
            return
        with contextlib.suppress(NoMatches):
            self._get_filter_input().focus()

    async def _handle_delete_confirm(self, thread_id: str) -> None:
        """Execute thread deletion after confirmation.

        Args:
            thread_id: Thread ID to delete.
        """
        from deepagents_cli.sessions import delete_thread

        preferred_thread_id: str | None = None
        if self._selected_index + 1 < len(self._filtered_threads):
            preferred_thread_id = self._filtered_threads[self._selected_index + 1][
                "thread_id"
            ]
        elif self._selected_index > 0:
            preferred_thread_id = self._filtered_threads[self._selected_index - 1][
                "thread_id"
            ]

        try:
            await delete_thread(thread_id)
        except (OSError, sqlite3.Error):
            logger.warning("Failed to delete thread %s", thread_id, exc_info=True)
            self.app.notify(
                f"Failed to delete thread {thread_id[:8]}",
                severity="error",
                timeout=3,
            )
            with contextlib.suppress(NoMatches):
                self.query_one("#thread-filter", Input).focus()
            return

        self._threads = [
            thread for thread in self._threads if thread["thread_id"] != thread_id
        ]
        self._update_filtered_list()
        if preferred_thread_id is not None:
            for index, thread in enumerate(self._filtered_threads):
                if thread["thread_id"] == preferred_thread_id:
                    self._selected_index = index
                    break
        if self._selected_index >= len(self._filtered_threads):
            self._selected_index = max(0, len(self._filtered_threads) - 1)
        await self._build_list()
        with contextlib.suppress(NoMatches):
            self.query_one("#thread-filter", Input).focus()

    def on_click(self, event: Click) -> None:  # noqa: PLR6301  # Textual event handler
        """Open Rich-style hyperlinks on single click."""
        open_style_link(event)

    def on_thread_option_clicked(self, event: ThreadOption.Clicked) -> None:
        """Handle click on a thread option.

        Args:
            event: The clicked message with thread ID and index.
        """
        if self._confirming_delete:
            return
        if 0 <= event.index < len(self._filtered_threads):
            self._selected_index = event.index
            self.dismiss(event.thread_id)

    def action_cancel(self) -> None:
        """Cancel the selection."""
        self.dismiss(None)
