"""Searchable local prompt history modal."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from deepagents_code.config import get_glyphs
from deepagents_code.tui.widgets.prompt_search import filter_prompts, prompt_title

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Click


class PromptRow(Static):
    """One prompt summary row.

    Public so its `Clicked` message resolves to the idiomatic
    `on_prompt_row_clicked` handler; nothing outside this module builds one.
    """

    class Clicked(Message):
        """Message sent when a prompt row is clicked."""

        def __init__(self, index: int) -> None:
            """Initialize with the clicked row index."""
            super().__init__()
            self.index = index

    def __init__(self, prompt: str, index: int, *, selected: bool) -> None:
        """Initialize the row with its title, list position, and selection."""
        classes = "prompt-row prompt-row-selected" if selected else "prompt-row"
        super().__init__(self._content(prompt), classes=classes)
        self._index = index

    def on_click(self, event: Click) -> None:
        """Announce the click; Enter is still required to insert the prompt.

        Posting a message rather than calling back into the screen keeps the
        row independent of what contains it, and matches how
        `PromptSearchOption` reports clicks in the inline panel.
        """
        event.stop()
        self.post_message(self.Clicked(self._index))

    @staticmethod
    def _content(prompt: str) -> Content:
        """Build a single-line literal-text title.

        Rows show only the title; the full prompt renders in the preview pane.
        The row is one cell high, so overlong titles clip visually; the
        `text-overflow: ellipsis` rule on `.prompt-row` marks the clipped end.
        The title comes from the inline panel's helper so both tiers summarize a
        prompt the same way.

        Returns:
            Safe Textual content for the row.
        """
        return Content.styled(prompt_title(prompt), "bold")


class PromptClipboardScreen(ModalScreen[str | None]):
    """Filter, preview, copy, or select a previously submitted prompt."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False, priority=True),
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("tab", "select", "Insert", show=False, priority=True),
        Binding("enter", "select", "Insert", show=False, priority=True),
        Binding("ctrl+c", "copy", "Copy", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    CSS_PATH = "prompt_clipboard.tcss"

    def __init__(
        self,
        prompts: tuple[str, ...],
        initial_query: str = "",
        *,
        empty_message: str | None = None,
    ) -> None:
        """Initialize the modal with a newest-first prompt snapshot.

        Args:
            prompts: Unique prompts, newest first.
            initial_query: Filter text to seed the search input with, either
                from the chat input or an inline search query.
            empty_message: Replaces the "no prompts yet" text when `prompts` is
                empty for a reason worth naming, such as an unreadable history
                file.
        """
        super().__init__()
        self._prompts = prompts
        self._filtered = list(prompts)
        self._filter_value = ""
        self._initial_query = initial_query
        self._empty_message = empty_message
        self._selected_index = 0
        self._rows: list[PromptRow] = []

    @override
    def compose(self) -> ComposeResult:
        """Compose the filter, results, preview, and keyboard help.

        Yields:
            Widgets for the prompt clipboard.
        """
        with Vertical():
            yield Static("Prompt Clipboard", classes="prompt-title")
            yield Input(placeholder="Search submitted prompts", id="prompt-filter")
            # Rows mount directly into the scroll container: a nested
            # auto-height wrapper is clamped to the list's max-height, which
            # hides the overflow and leaves the list unscrollable.
            yield VerticalScroll(id="prompt-list")
            yield Static("Preview", classes="prompt-preview-label")
            with VerticalScroll(id="prompt-preview-scroll"):
                yield Static("", id="prompt-preview")
            glyphs = get_glyphs()
            sep = f"  {glyphs.bullet}  "
            yield Static(
                sep.join(
                    (
                        f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate",
                        "Tab/Enter insert",
                        "Ctrl+C copy",
                        "Esc cancel",
                    )
                ),
                classes="prompt-help",
            )

    async def on_mount(self) -> None:
        """Render the initial rows and focus search."""
        if self._initial_query:
            search = self.query_one("#prompt-filter", Input)
            search.value = self._initial_query
            self._apply_filter(self._initial_query)
        await self._render_rows()
        search = self.query_one("#prompt-filter", Input)
        search.focus()
        if self._initial_query:
            search.cursor_position = len(self._initial_query)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter prompts using the current search value."""
        self._apply_filter(event.value)
        self.call_after_refresh(self._render_rows)

    def _apply_filter(self, value: str) -> None:
        """Synchronize filtered prompts with literal search text."""
        self._filter_value = value
        self._filtered = list(filter_prompts(self._prompts, value))
        self._selected_index = 0

    def _sync_filter_value(self) -> None:
        """Consume a filter edit whose Changed message is still queued."""
        value = self.query_one("#prompt-filter", Input).value
        if value == self._filter_value:
            return
        # The highlighted row is what the user is acting on, so follow that
        # prompt into the new list rather than snapping to index 0 and copying
        # or inserting something they never selected.
        selected = (
            self._filtered[self._selected_index]
            if 0 <= self._selected_index < len(self._filtered)
            else None
        )
        self._apply_filter(value)
        if selected is not None and selected in self._filtered:
            self._selected_index = self._filtered.index(selected)
        self.call_after_refresh(self._render_rows)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Insert the selected prompt when search receives Enter."""
        event.stop()
        self.action_select()

    async def _render_rows(self) -> None:
        rows_list = self.query_one("#prompt-list", VerticalScroll)
        await rows_list.remove_children()
        self._rows = []
        if not self._filtered:
            if self._prompts:
                message = "No matching prompts."
            else:
                message = (
                    self._empty_message
                    or "No prompts yet. Submitted prompts appear here."
                )
            await rows_list.mount(Static(Content.styled(message, "dim")))
            self.query_one("#prompt-preview", Static).update("")
            return

        self._rows = [
            PromptRow(prompt, index, selected=index == self._selected_index)
            for index, prompt in enumerate(self._filtered)
        ]
        await rows_list.mount(*self._rows)
        self._update_preview()

    def _update_preview(self) -> None:
        preview = self.query_one("#prompt-preview", Static)
        if not self._filtered:
            preview.update("")
            return
        preview.update(Content(self._filtered[self._selected_index]))

    def on_prompt_row_clicked(self, event: PromptRow.Clicked) -> None:
        """Move the selection to the clicked row."""
        event.stop()
        self._select_row(event.index)

    def _select_row(self, index: int) -> None:
        """Move the selection to `index`, as if navigated to by key.

        Enter is still required to insert. Ignores clicks that arrive while a
        filter edit is still queued, since the row indexes then describe a
        different list than `_filtered`.
        """
        self._sync_filter_value()
        if (
            not self._filtered
            or len(self._rows) != len(self._filtered)
            or index == self._selected_index
            or not 0 <= index < len(self._rows)
        ):
            return
        previous = self._selected_index
        self._selected_index = index
        self._rows[previous].remove_class("prompt-row-selected")
        self._rows[index].add_class("prompt-row-selected")
        self._update_preview()

    def _apply_selection(self, index: int) -> None:
        """Move the selection to an already-rendered row.

        Preconditions, established by every caller before it gets here:
        `_rows` mirrors `_filtered` one-to-one, and both `index` and
        `_selected_index` are in range. Callers settle any queued filter edit
        (`_sync_filter_value`, then `_render_rows` if the lengths disagree) to
        guarantee that, which is why this indexes without guarding. A new
        caller that skips those steps will raise `IndexError` here.

        Args:
            index: Row to select. Must be a valid index into `_rows`.
        """
        self._rows[self._selected_index].remove_class("prompt-row-selected")
        self._selected_index = index
        selected = self._rows[index]
        selected.add_class("prompt-row-selected")
        selected.scroll_visible(animate=False)
        self._update_preview()

    async def _move(self, delta: int) -> None:
        # Every user-facing action settles a queued filter edit first. Without
        # this, an edit whose `Changed` message has not been dispatched yet
        # leaves `_filtered` and `_rows` both describing the pre-edit list --
        # the lengths match, the guard below does not fire, and the arrow keys
        # navigate a list the user has already filtered away from.
        self._sync_filter_value()
        if not self._filtered:
            self.app.bell()
            return
        if len(self._rows) != len(self._filtered):
            # A filter edit already updated `_filtered` but its deferred
            # `_render_rows` has not landed, so the rows still describe the old
            # list. Indexing them with a bound taken from the new one raises,
            # so settle the render before moving.
            await self._render_rows()
        previous = self._selected_index
        new_index = max(0, min(previous + delta, len(self._filtered) - 1))
        if previous == new_index:
            self.app.bell()
            return
        self._apply_selection(new_index)

    async def action_move_up(self) -> None:
        """Move selection toward newer prompts."""
        await self._move(-1)

    async def action_move_down(self) -> None:
        """Move selection toward older prompts."""
        await self._move(1)

    def action_select(self) -> None:
        """Dismiss with the selected prompt."""
        self._sync_filter_value()
        if self._filtered:
            self.dismiss(self._filtered[self._selected_index])
        else:
            self.app.bell()

    def action_copy(self) -> None:
        """Copy the selected prompt without dismissing the modal."""
        self._sync_filter_value()
        if not self._filtered:
            self.app.bell()
            return
        from deepagents_code.clipboard import copy_text_with_feedback

        copy_text_with_feedback(
            self.app,
            self._filtered[self._selected_index],
            failure_noun="prompt",
            success_message="Prompt copied to clipboard",
        )

    def action_cancel(self) -> None:
        """Dismiss without selecting a prompt."""
        self.dismiss(None)
