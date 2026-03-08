"""Interactive thread selector screen for /threads command."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from typing import TYPE_CHECKING, ClassVar

from rich.style import Style
from rich.text import Text
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.fuzzy import Matcher
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Input, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Click

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
_COL_AGENT = 14
_COL_MSGS = 4
_COL_BRANCH = 16
_COL_TIMESTAMP = 16
_COL_PROMPT = 30


class ThreadOption(Static):
    """A clickable thread option in the selector."""

    def __init__(
        self,
        label: str,
        thread_id: str,
        index: int,
        *,
        classes: str = "",
    ) -> None:
        """Initialize a thread option.

        Args:
            label: The display text for the option.
            thread_id: The thread identifier.
            index: The index of this option in the list.
            classes: CSS classes for styling.
        """
        super().__init__(label, classes=classes)
        self.thread_id = thread_id
        self.index = index

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

    def on_click(self, event: Click) -> None:
        """Handle click on this option.

        Args:
            event: The click event.
        """
        event.stop()
        self.post_message(self.Clicked(self.thread_id, self.index))


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
        Binding("tab", "toggle_sort", "Toggle sort", show=False, priority=True),
    ]

    CSS = """
    ThreadSelectorScreen {
        align: center middle;
    }

    ThreadSelectorScreen > Vertical {
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

    ThreadSelectorScreen .thread-list-header {
        height: 1;
        padding: 0 2 0 1;
        color: $text-muted;
        text-style: bold;
    }

    ThreadSelectorScreen .thread-list {
        height: 1fr;
        min-height: 5;
        scrollbar-gutter: stable;
        background: $background;
    }

    ThreadSelectorScreen .thread-option {
        height: 1;
        padding: 0 1;
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

    ThreadSelectorScreen .thread-confirm-overlay {
        align: center middle;
    }

    ThreadSelectorScreen .thread-confirm-box {
        width: 50;
        height: auto;
        background: $surface;
        border: solid red;
        padding: 1 2;
    }

    ThreadSelectorScreen .thread-confirm-text {
        text-align: center;
        margin-bottom: 1;
    }

    ThreadSelectorScreen .thread-confirm-help {
        text-align: center;
        color: $text-muted;
        text-style: italic;
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
        self._sort_by_updated = True
        self._confirming_delete = False

        from deepagents_cli.model_config import load_thread_columns

        self._columns = load_thread_columns()

        self._sync_selected_index()

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

    def compose(self) -> ComposeResult:
        """Compose the screen layout.

        Yields:
            Widgets for the thread selector UI.
        """
        glyphs = get_glyphs()

        with Vertical():
            yield Static(
                self._build_title(), classes="thread-selector-title", id="thread-title"
            )

            yield Input(
                placeholder="Type to search threads...",
                id="thread-filter",
            )

            yield Static(
                self._format_header(), classes="thread-list-header", id="thread-header"
            )

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

            sort_label = "updated" if self._sort_by_updated else "created"
            help_text = (
                f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate"
                f" {glyphs.bullet} Enter select"
                f" {glyphs.bullet} Tab sort ({sort_label})"
                f" {glyphs.bullet} Ctrl+D delete"
                f" {glyphs.bullet} Esc cancel"
            )
            yield Static(help_text, classes="thread-selector-help", id="thread-help")

    async def on_mount(self) -> None:
        """Fetch threads, configure border for ASCII terminals, and build the list."""
        if _detect_charset_mode() == CharsetMode.ASCII:
            container = self.query_one(Vertical)
            container.styles.border = ("ascii", "green")

        filter_input = self.query_one("#thread-filter", Input)
        filter_input.focus()

        if self._has_initial_threads:
            self.call_after_refresh(self._scroll_selected_into_view)
            self._schedule_message_count_load()
            self._schedule_initial_prompt_load()
            if self._current_thread:
                self._resolve_thread_url()
            self.run_worker(
                self._load_threads, exclusive=True, group="thread-selector-load"
            )
            return

        self.run_worker(
            self._load_threads, exclusive=True, group="thread-selector-load"
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter threads as user types.

        Args:
            event: The input changed event.
        """
        self._filter_text = event.value
        self._update_filtered_list()
        self.call_after_refresh(self._rebuild_list_from_filter)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key when filter input is focused.

        Args:
            event: The input submitted event.
        """
        event.stop()
        self.action_select()

    def _update_filtered_list(self) -> None:
        """Update filtered threads based on search text using fuzzy matching."""
        query = self._filter_text.strip()
        if not query:
            self._filtered_threads = list(self._threads)
            self._apply_sort()
            self._sync_selected_index()
            return

        tokens = query.split()
        try:
            matchers = [Matcher(token, case_sensitive=False) for token in tokens]
            scored: list[tuple[float, ThreadInfo]] = []
            for thread in self._threads:
                search_text = self._get_search_text(thread)
                scores = [m.match(search_text) for m in matchers]
                if all(s > 0 for s in scores):
                    scored.append((min(scores), thread))
        except Exception:
            logger.warning(
                "Fuzzy matcher failed for query %r, falling back to full list",
                query,
                exc_info=True,
            )
            self._filtered_threads = list(self._threads)
            self._sync_selected_index()
            return

        self._filtered_threads = [thread for _, thread in sorted(scored, reverse=True)]
        self._selected_index = 0

    @staticmethod
    def _get_search_text(thread: ThreadInfo) -> str:
        """Build searchable text from thread fields.

        Args:
            thread: Thread metadata.

        Returns:
            Concatenated searchable string.
        """
        parts = [
            thread["thread_id"],
            thread.get("agent_name") or "",
            thread.get("initial_prompt") or "",
            thread.get("git_branch") or "",
        ]
        return " ".join(parts)

    async def _rebuild_list_from_filter(self) -> None:
        """Rebuild the list after filter changes."""
        await self._build_list()

    def _schedule_message_count_load(self) -> None:
        """Schedule background message-count loading when counts are missing."""
        has_missing_counts = self._threads and any(
            "message_count" not in thread for thread in self._threads
        )
        if has_missing_counts:
            self.run_worker(
                self._load_message_counts,
                exclusive=True,
                group="thread-selector-counts",
            )

    def _schedule_initial_prompt_load(self) -> None:
        """Schedule background initial-prompt loading when prompts are missing."""
        has_missing = self._threads and any(
            "initial_prompt" not in thread for thread in self._threads
        )
        if has_missing:
            self.run_worker(
                self._load_initial_prompts,
                exclusive=True,
                group="thread-selector-prompts",
            )

    async def _load_threads(self) -> None:
        """Load thread rows first, then kick off background enrichment."""
        from deepagents_cli.sessions import (
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
        self._update_filtered_list()
        self._sync_selected_index()

        await self._build_list()

        self._schedule_message_count_load()
        self._schedule_initial_prompt_load()

        if self._current_thread:
            self._resolve_thread_url()

    async def _load_message_counts(self) -> None:
        """Populate thread message counts in background and refresh labels."""
        from deepagents_cli.sessions import populate_thread_message_counts

        if not self._threads:
            return

        try:
            await populate_thread_message_counts(self._threads)
        except (OSError, sqlite3.Error):
            logger.debug(
                "Could not load message counts for thread selector",
                exc_info=True,
            )
            return
        except Exception:
            logger.warning(
                "Unexpected error loading message counts for thread selector",
                exc_info=True,
            )
            return

        self._refresh_labels()

    async def _load_initial_prompts(self) -> None:
        """Populate initial prompts in background and refresh labels."""
        from deepagents_cli.sessions import populate_thread_initial_prompts

        if not self._threads:
            return

        try:
            await populate_thread_initial_prompts(self._threads)
        except (OSError, sqlite3.Error):
            logger.debug(
                "Could not load initial prompts for thread selector",
                exc_info=True,
            )
            return
        except Exception:
            logger.warning(
                "Unexpected error loading initial prompts for thread selector",
                exc_info=True,
            )
            return

        self._update_filtered_list()
        self._refresh_labels()

    def _refresh_labels(self) -> None:
        """Refresh row labels after background data loads complete."""
        if not self._filtered_threads or not self._option_widgets:
            return

        for index, thread in enumerate(self._filtered_threads):
            if index >= len(self._option_widgets):
                break
            widget = self._option_widgets[index]
            widget.update(
                self._format_option_label(
                    thread,
                    selected=index == self._selected_index,
                    current=thread["thread_id"] == self._current_thread,
                    columns=self._columns,
                )
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
            scroll = self.query_one(".thread-list", VerticalScroll)
            await scroll.remove_children()
            await scroll.mount(
                Static(
                    f"[red]Failed to load threads: {detail}. Press Esc to close.[/red]",
                    classes="thread-empty",
                )
            )
        except Exception:
            logger.warning(
                "Could not display error message in thread selector UI",
                exc_info=True,
            )
        self.focus()

    async def _build_list(self) -> None:
        """Build the thread option widgets."""
        scroll = self.query_one(".thread-list", VerticalScroll)
        await scroll.remove_children()

        try:
            header_widget = self.query_one("#thread-header", Static)
            header_widget.update(self._format_header())
        except NoMatches:
            pass

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
            self._scroll_selected_into_view()

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

            label = self._format_option_label(
                thread,
                selected=is_selected,
                current=is_current,
                columns=self._columns,
            )
            widget = ThreadOption(
                label=label,
                thread_id=thread["thread_id"],
                index=i,
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

    def _format_header(self) -> str:
        """Build the column header label.

        Returns:
            Formatted header string with column names.
        """
        parts = ["  "]
        if self._columns.get("thread_id"):
            parts.append(f"{'Thread':<{_COL_TID}}  ")
        if self._columns.get("agent_name"):
            parts.append(f"{'Agent':<{_COL_AGENT}}  ")
        if self._columns.get("messages"):
            parts.append(f"{'Msgs':>{_COL_MSGS}}  ")
        if self._columns.get("git_branch"):
            parts.append(f"{'Branch':<{_COL_BRANCH}}  ")
        if self._sort_by_updated and self._columns.get("updated_at"):
            parts.append(f"{'Updated':<{_COL_TIMESTAMP}}  ")
        elif not self._sort_by_updated and self._columns.get("created_at"):
            parts.append(f"{'Created':<{_COL_TIMESTAMP}}  ")
        if self._columns.get("initial_prompt"):
            parts.append("Prompt")
        return "".join(parts)

    @staticmethod
    def _format_option_label(
        thread: ThreadInfo,
        *,
        selected: bool,
        current: bool,
        columns: dict[str, bool] | None = None,
    ) -> str:
        """Build the display label for a thread option.

        Args:
            thread: Thread metadata from `list_threads`.
            selected: Whether this option is currently highlighted.
            current: Whether this is the active thread.
            columns: Column visibility settings.

        Returns:
            Rich-markup label string.
        """
        from deepagents_cli.model_config import THREAD_COLUMN_DEFAULTS
        from deepagents_cli.sessions import format_timestamp

        cols = columns if columns is not None else THREAD_COLUMN_DEFAULTS

        glyphs = get_glyphs()
        cursor = f"{glyphs.cursor} " if selected else "  "

        parts = [cursor]

        if cols.get("thread_id"):
            tid = thread["thread_id"][:_COL_TID]
            parts.append(f"{tid:<{_COL_TID}}  ")

        if cols.get("agent_name"):
            agent = (thread.get("agent_name") or "unknown")[:_COL_AGENT]
            parts.append(f"{agent:<{_COL_AGENT}}  ")

        if cols.get("messages"):
            raw_count = thread.get("message_count")
            msgs = str(raw_count) if raw_count is not None else "..."
            parts.append(f"{msgs:>{_COL_MSGS}}  ")

        if cols.get("git_branch"):
            branch = (thread.get("git_branch") or "")[:_COL_BRANCH]
            parts.append(f"{branch:<{_COL_BRANCH}}  ")

        if cols.get("updated_at"):
            timestamp = format_timestamp(thread.get("updated_at"))
            parts.append(f"{timestamp:<{_COL_TIMESTAMP}}  ")
        elif cols.get("created_at"):
            timestamp = format_timestamp(thread.get("created_at"))
            parts.append(f"{timestamp:<{_COL_TIMESTAMP}}  ")

        if cols.get("initial_prompt"):
            prompt = thread.get("initial_prompt") or ""
            if len(prompt) > _COL_PROMPT:
                prompt = prompt[: _COL_PROMPT - 1] + glyphs.ellipsis
            parts.append(prompt)

        label = "".join(parts)
        if current:
            label += " [dim](current)[/dim]"
        return label

    def _apply_sort(self) -> None:
        """Sort filtered threads by the active sort key."""
        key = "updated_at" if self._sort_by_updated else "created_at"
        self._filtered_threads.sort(key=lambda t: t.get(key) or "", reverse=True)

    def _move_selection(self, delta: int) -> None:
        """Move selection by delta, re-rendering only the old and new widgets.

        Args:
            delta: Positions to move (negative for up, positive for down).
        """
        if not self._filtered_threads or not self._option_widgets:
            return

        count = len(self._filtered_threads)
        old_index = self._selected_index
        new_index = (old_index + delta) % count
        self._selected_index = new_index

        old_widget = self._option_widgets[old_index]
        old_widget.remove_class("thread-option-selected")
        old_thread = self._filtered_threads[old_index]
        old_widget.update(
            self._format_option_label(
                old_thread,
                selected=False,
                current=old_thread["thread_id"] == self._current_thread,
                columns=self._columns,
            )
        )

        new_widget = self._option_widgets[new_index]
        new_widget.add_class("thread-option-selected")
        new_thread = self._filtered_threads[new_index]
        new_widget.update(
            self._format_option_label(
                new_thread,
                selected=True,
                current=new_thread["thread_id"] == self._current_thread,
                columns=self._columns,
            )
        )

        if new_index == 0:
            scroll = self.query_one(".thread-list", VerticalScroll)
            scroll.scroll_home(animate=False)
        else:
            new_widget.scroll_visible()

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

    def action_toggle_sort(self) -> None:
        """Toggle sort between updated_at and created_at."""
        if self._confirming_delete:
            return
        self._sort_by_updated = not self._sort_by_updated

        old_columns = dict(self._columns)
        if self._sort_by_updated:
            self._columns["updated_at"] = True
            self._columns["created_at"] = False
        else:
            self._columns["created_at"] = True
            self._columns["updated_at"] = False

        if old_columns != self._columns:
            from deepagents_cli.model_config import save_thread_columns

            save_thread_columns(self._columns)

        self._apply_sort()
        self._sync_selected_index()

        glyphs = get_glyphs()
        sort_label = "updated" if self._sort_by_updated else "created"
        help_text = (
            f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate"
            f" {glyphs.bullet} Enter select"
            f" {glyphs.bullet} Tab sort ({sort_label})"
            f" {glyphs.bullet} Ctrl+D delete"
            f" {glyphs.bullet} Esc cancel"
        )
        try:
            help_widget = self.query_one("#thread-help", Static)
            help_widget.update(help_text)
        except NoMatches:
            pass

        self.call_after_refresh(self._rebuild_list_from_filter)

    def action_delete_thread(self) -> None:
        """Show delete confirmation for the highlighted thread."""
        if self._confirming_delete or not self._filtered_threads:
            return
        self._confirming_delete = True
        thread = self._filtered_threads[self._selected_index]
        tid = thread["thread_id"]
        self.run_worker(self._show_delete_confirm(tid), group="thread-delete-confirm")

    async def _show_delete_confirm(self, thread_id: str) -> None:
        """Show a delete confirmation overlay.

        Args:
            thread_id: Thread ID to delete.
        """
        from textual.containers import Center

        overlay = Vertical(
            Static(
                f"Delete thread [bold]{thread_id}[/bold]?",
                classes="thread-confirm-text",
            ),
            Static(
                "Enter to confirm, Esc to cancel",
                classes="thread-confirm-help",
            ),
            classes="thread-confirm-box",
            id="delete-confirm",
        )
        wrapper = Center(
            overlay, classes="thread-confirm-overlay", id="confirm-overlay"
        )
        await self.mount(wrapper)

    async def _handle_delete_confirm(self, thread_id: str) -> None:
        """Execute thread deletion after confirmation.

        Args:
            thread_id: Thread ID to delete.
        """
        from deepagents_cli.sessions import delete_thread

        try:
            await delete_thread(thread_id)
        except (OSError, sqlite3.Error):
            logger.warning("Failed to delete thread %s", thread_id, exc_info=True)

        self._threads = [t for t in self._threads if t["thread_id"] != thread_id]
        self._update_filtered_list()
        if self._selected_index >= len(self._filtered_threads):
            self._selected_index = max(0, len(self._filtered_threads) - 1)
        await self._build_list()

    async def _dismiss_delete_confirm(self) -> None:
        """Remove the delete confirmation overlay."""
        self._confirming_delete = False
        try:
            overlay = self.query_one("#confirm-overlay")
            await overlay.remove()
        except NoMatches:
            pass

    async def on_key(self, event: object) -> None:
        """Handle key events for delete confirmation.

        Args:
            event: The key event.
        """
        if not self._confirming_delete:
            return

        key = getattr(event, "key", "")
        if key == "enter":
            event.stop()  # type: ignore[union-attr]
            if self._filtered_threads:
                tid = self._filtered_threads[self._selected_index]["thread_id"]
                await self._dismiss_delete_confirm()
                await self._handle_delete_confirm(tid)
        elif key == "escape":
            event.stop()  # type: ignore[union-attr]
            await self._dismiss_delete_confirm()

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
        if self._confirming_delete:
            self.run_worker(
                self._dismiss_delete_confirm(), group="thread-delete-cancel"
            )
            return
        self.dismiss(None)
