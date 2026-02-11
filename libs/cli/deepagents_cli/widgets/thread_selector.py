"""Interactive thread selector screen for /resume command."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.events import (
    Click,  # noqa: TC002 - needed at runtime for Textual event dispatch
)
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

from deepagents_cli.config import CharsetMode, _detect_charset_mode, get_glyphs
from deepagents_cli.sessions import _format_timestamp, list_threads


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
    """Full-screen modal for thread selection.

    Displays recent threads with keyboard navigation. The current thread
    is highlighted.

    Returns a `thread_id` string on selection, or `None` on cancel.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False, priority=True),
        Binding("k", "move_up", "Up", show=False, priority=True),
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("j", "move_down", "Down", show=False, priority=True),
        Binding("tab", "move_down", "Down", show=False, priority=True),
        Binding("shift+tab", "move_up", "Up", show=False, priority=True),
        Binding("pageup", "page_up", "Page up", show=False, priority=True),
        Binding("pagedown", "page_down", "Page down", show=False, priority=True),
        Binding("enter", "select", "Select", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    CSS = """
    ThreadSelectorScreen {
        align: center middle;
    }

    ThreadSelectorScreen > Vertical {
        width: 80;
        max-width: 90%;
        height: 80%;
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
        height: 1;
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

    def __init__(self, current_thread: str | None = None) -> None:
        """Initialize the `ThreadSelectorScreen`.

        Args:
            current_thread: The currently active thread ID (to highlight).
        """
        super().__init__()
        self._current_thread = current_thread
        self._threads: list[dict] = []
        self._selected_index = 0
        self._option_widgets: list[ThreadOption] = []

    def compose(self) -> ComposeResult:
        """Compose the screen layout.

        Yields:
            Widgets for the thread selector UI.
        """
        glyphs = get_glyphs()

        with Vertical():
            title = "Select Thread"
            if self._current_thread:
                title = f"Select Thread (current: {self._current_thread})"
            yield Static(title, classes="thread-selector-title")

            with VerticalScroll(classes="thread-list"):
                yield Static(
                    "[dim]Loading threads...[/dim]",
                    classes="thread-empty",
                    id="thread-loading",
                )

            help_text = (
                f"{glyphs.arrow_up}/{glyphs.arrow_down}/tab navigate "
                f"{glyphs.bullet} Enter select {glyphs.bullet} Esc cancel"
            )
            yield Static(help_text, classes="thread-selector-help")

    async def on_mount(self) -> None:
        """Fetch threads and build the list on mount."""
        if _detect_charset_mode() == CharsetMode.ASCII:
            container = self.query_one(Vertical)
            container.styles.border = ("ascii", "green")

        self._threads = await list_threads(limit=20, include_message_count=True)

        # Find current thread index
        for i, t in enumerate(self._threads):
            if t["thread_id"] == self._current_thread:
                self._selected_index = i
                break

        await self._build_list()
        self.focus()

    async def _build_list(self) -> None:
        """Build the thread option widgets."""
        scroll = self.query_one(".thread-list", VerticalScroll)
        await scroll.remove_children()
        self._option_widgets = []

        if not self._threads:
            await scroll.mount(
                Static(
                    "[dim]No threads found[/dim]",
                    classes="thread-empty",
                )
            )
            return

        widgets: list[ThreadOption] = []
        selected_widget: ThreadOption | None = None

        for i, thread in enumerate(self._threads):
            is_current = thread["thread_id"] == self._current_thread
            is_selected = i == self._selected_index

            classes = "thread-option"
            if is_selected:
                classes += " thread-option-selected"
            if is_current:
                classes += " thread-option-current"

            label = self._format_option_label(
                thread, selected=is_selected, current=is_current
            )
            widget = ThreadOption(
                label=label,
                thread_id=thread["thread_id"],
                index=i,
                classes=classes,
            )
            widgets.append(widget)
            self._option_widgets.append(widget)

            if is_selected:
                selected_widget = widget

        await scroll.mount(*widgets)

        if selected_widget:
            if self._selected_index == 0:
                scroll.scroll_home(animate=False)
            else:
                selected_widget.scroll_visible(animate=False)

    @staticmethod
    def _format_option_label(
        thread: dict,
        *,
        selected: bool,
        current: bool,
    ) -> str:
        """Build the display label for a thread option.

        Args:
            thread: Thread dict with `thread_id`, `agent_name`, `updated_at`,
                and optionally `message_count`.
            selected: Whether this option is currently highlighted.
            current: Whether this is the active thread.

        Returns:
            Rich-markup label string.
        """
        glyphs = get_glyphs()
        cursor = f"{glyphs.cursor} " if selected else "  "
        tid = thread["thread_id"]
        agent = thread.get("agent_name") or "unknown"
        msgs = thread.get("message_count", 0)
        timestamp = _format_timestamp(thread.get("updated_at"))

        parts = [f"{cursor}{tid}  {agent}  {msgs} msgs"]
        if timestamp:
            parts[0] += f"  {timestamp}"

        suffix = " [dim](current)[/dim]" if current else ""
        return parts[0] + suffix

    def _move_selection(self, delta: int) -> None:
        """Move selection by delta, updating only the affected widgets.

        Args:
            delta: Number of positions to move (-1 for up, +1 for down).
        """
        if not self._threads or not self._option_widgets:
            return

        count = len(self._threads)
        old_index = self._selected_index
        new_index = (old_index + delta) % count
        self._selected_index = new_index

        old_widget = self._option_widgets[old_index]
        old_widget.remove_class("thread-option-selected")
        old_thread = self._threads[old_index]
        old_widget.update(
            self._format_option_label(
                old_thread,
                selected=False,
                current=old_thread["thread_id"] == self._current_thread,
            )
        )

        new_widget = self._option_widgets[new_index]
        new_widget.add_class("thread-option-selected")
        new_thread = self._threads[new_index]
        new_widget.update(
            self._format_option_label(
                new_thread,
                selected=True,
                current=new_thread["thread_id"] == self._current_thread,
            )
        )

        if new_index == 0:
            scroll = self.query_one(".thread-list", VerticalScroll)
            scroll.scroll_home(animate=False)
        else:
            new_widget.scroll_visible()

    def action_move_up(self) -> None:
        """Move selection up."""
        self._move_selection(-1)

    def action_move_down(self) -> None:
        """Move selection down."""
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
        except Exception:  # noqa: BLE001
            return default_page_size
        if height <= 0:
            return default_page_size
        return max(1, height)

    def action_page_up(self) -> None:
        """Move selection up by one visible page."""
        if not self._threads:
            return
        page = self._visible_page_size()
        target = max(0, self._selected_index - page)
        delta = target - self._selected_index
        if delta != 0:
            self._move_selection(delta)

    def action_page_down(self) -> None:
        """Move selection down by one visible page."""
        if not self._threads:
            return
        count = len(self._threads)
        page = self._visible_page_size()
        target = min(count - 1, self._selected_index + page)
        delta = target - self._selected_index
        if delta != 0:
            self._move_selection(delta)

    def action_select(self) -> None:
        """Select the current thread."""
        if self._threads:
            thread_id = self._threads[self._selected_index]["thread_id"]
            self.dismiss(thread_id)

    def on_thread_option_clicked(self, event: ThreadOption.Clicked) -> None:
        """Handle click on a thread option.

        Args:
            event: The click event with thread info.
        """
        self._selected_index = event.index
        self.dismiss(event.thread_id)

    def action_cancel(self) -> None:
        """Cancel the selection."""
        self.dismiss(None)
