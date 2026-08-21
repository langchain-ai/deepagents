"""Searchable local prompt history modal."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Input, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

_TITLE_MAX = 72
_EXCERPT_MAX = 120


class _PromptRow(Static):
    """One prompt summary row."""

    def __init__(self, prompt: str, *, selected: bool) -> None:
        classes = "prompt-row prompt-row-selected" if selected else "prompt-row"
        super().__init__(self._content(prompt), classes=classes)

    @staticmethod
    def _content(prompt: str) -> Content:
        """Build a bounded literal-text title and excerpt.

        Returns:
            Safe Textual content for the row.
        """
        nonempty = [line.strip() for line in prompt.splitlines() if line.strip()]
        title = nonempty[0] if nonempty else prompt.strip()
        if len(title) > _TITLE_MAX:
            title = title[: _TITLE_MAX - 1] + "…"
        excerpt = " ".join(prompt.split())
        if len(excerpt) > _EXCERPT_MAX:
            excerpt = excerpt[: _EXCERPT_MAX - 1] + "…"
        return Content.assemble(
            (title or "(empty prompt)", "bold"), (f"\n{excerpt}", "dim")
        )


class PromptClipboardScreen(ModalScreen[str | None]):
    """Filter, preview, copy, or select a previously submitted prompt."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False, priority=True),
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("enter", "select", "Insert", show=False, priority=True),
        Binding("ctrl+c", "copy", "Copy", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    CSS_PATH = "prompt_clipboard.tcss"

    def __init__(self, prompts: tuple[str, ...]) -> None:
        """Initialize the modal with a newest-first prompt snapshot."""
        super().__init__()
        self._prompts = prompts
        self._filtered = list(prompts)
        self._filter_value = ""
        self._selected_index = 0
        self._rows: list[_PromptRow] = []

    @override
    def compose(self) -> ComposeResult:
        """Compose the filter, results, preview, and keyboard help.

        Yields:
            Widgets for the prompt clipboard.
        """
        with Vertical():
            yield Static("Prompt Clipboard", classes="prompt-title")
            yield Input(placeholder="Search submitted prompts", id="prompt-filter")
            with VerticalScroll(id="prompt-list"):
                yield Container(id="prompt-rows")
            yield Static("Preview", classes="prompt-preview-label")
            with VerticalScroll(id="prompt-preview-scroll"):
                yield Static("", id="prompt-preview")
            yield Static(
                "↑/↓ navigate  •  Enter insert  •  Ctrl+C copy  •  Esc cancel",
                classes="prompt-help",
            )

    async def on_mount(self) -> None:
        """Render the initial rows and focus search."""
        await self._render_rows()
        self.query_one("#prompt-filter", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter prompts using the current search value."""
        self._apply_filter(event.value)
        self.call_after_refresh(self._render_rows)

    def _apply_filter(self, value: str) -> None:
        """Synchronize filtered prompts with literal search text."""
        self._filter_value = value
        query = value.strip().casefold()
        self._filtered = [
            prompt for prompt in self._prompts if query in prompt.casefold()
        ]
        self._selected_index = 0

    def _sync_filter_value(self) -> None:
        """Consume a filter edit whose Changed message is still queued."""
        value = self.query_one("#prompt-filter", Input).value
        if value != self._filter_value:
            self._apply_filter(value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Insert the selected prompt when search receives Enter."""
        event.stop()
        self.action_select()

    async def _render_rows(self) -> None:
        container = self.query_one("#prompt-rows", Container)
        await container.remove_children()
        self._rows = []
        if not self._filtered:
            message = (
                "No matching prompts."
                if self._prompts
                else "No prompts yet. Submitted prompts appear here."
            )
            await container.mount(Static(Content.styled(message, "dim")))
            self.query_one("#prompt-preview", Static).update("")
            return

        self._rows = [
            _PromptRow(prompt, selected=index == self._selected_index)
            for index, prompt in enumerate(self._filtered)
        ]
        await container.mount(*self._rows)
        self._update_preview()

    def _update_preview(self) -> None:
        preview = self.query_one("#prompt-preview", Static)
        if not self._filtered:
            preview.update("")
            return
        preview.update(Content(self._filtered[self._selected_index]))

    def _move(self, delta: int) -> None:
        if not self._filtered:
            self.app.bell()
            return
        previous = self._selected_index
        self._selected_index = max(
            0, min(self._selected_index + delta, len(self._filtered) - 1)
        )
        if previous == self._selected_index:
            self.app.bell()
            return
        self._rows[previous].remove_class("prompt-row-selected")
        selected = self._rows[self._selected_index]
        selected.add_class("prompt-row-selected")
        selected.scroll_visible()
        self._update_preview()

    def action_move_up(self) -> None:
        """Move selection toward newer prompts."""
        self._move(-1)

    def action_move_down(self) -> None:
        """Move selection toward older prompts."""
        self._move(1)

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
