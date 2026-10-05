"""Editable confirmation for a generated thread name."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from deepagents_code.sessions import MAX_THREAD_NAME_LENGTH, validate_thread_name

if TYPE_CHECKING:
    from textual.app import ComposeResult


class ThreadNameScreen(ModalScreen[str | None]):
    """Confirm or edit a suggested name without changing the thread yet."""

    CSS_PATH = "thread_name.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    def __init__(self, name: str) -> None:
        """Initialize the editable proposal."""
        super().__init__()
        self._name = name

    @override
    def compose(self) -> ComposeResult:
        """Yield the editable name field and confirmation hints."""
        with Vertical():
            yield Static("Rename thread", classes="thread-name-heading")
            yield Input(self._name, max_length=MAX_THREAD_NAME_LENGTH, id="thread-name")
            yield Static("Enter save / Esc cancel", classes="thread-name-help")
            yield Static("", id="thread-name-error", markup=False)

    def on_mount(self) -> None:
        """Focus the proposed name for editing."""
        field = self.query_one(Input)
        field.focus()
        field.cursor_position = len(field.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Accept only a valid, printable single-line name."""
        event.stop()
        try:
            name = validate_thread_name(event.value)
        except ValueError as exc:
            self.query_one("#thread-name-error", Static).update(str(exc))
            return
        self.dismiss(name)

    def action_cancel(self) -> None:
        """Dismiss without saving the proposal."""
        self.dismiss(None)
