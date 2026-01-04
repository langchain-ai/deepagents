"""Chat input widget for deepagents-cli."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Input, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


class ChatInput(Horizontal):
    """Chat input widget with prompt indicator and text input."""

    DEFAULT_CSS = """
    ChatInput {
        height: auto;
        min-height: 3;
        max-height: 10;
        padding: 0;
        background: $surface;
        border: solid $primary;
    }

    ChatInput .input-prompt {
        width: 3;
        height: 1;
        padding: 0 1;
        color: $primary;
        text-style: bold;
    }

    ChatInput Input {
        width: 1fr;
        border: none;
        background: transparent;
        padding: 0;
    }

    ChatInput Input:focus {
        border: none;
    }
    """

    class Submitted(Message):
        """Message sent when input is submitted."""

        def __init__(self, value: str, mode: str = "normal") -> None:
            """Initialize the submitted message.

            Args:
                value: The submitted text
                mode: The input mode ("normal", "bash", "command")
            """
            super().__init__()
            self.value = value
            self.mode = mode

    class ModeChanged(Message):
        """Message sent when input mode changes."""

        def __init__(self, mode: str) -> None:
            """Initialize the mode changed message.

            Args:
                mode: The new input mode
            """
            super().__init__()
            self.mode = mode

    mode: reactive[str] = reactive("normal")

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the chat input widget."""
        super().__init__(**kwargs)
        self._input: Input | None = None

    def compose(self) -> ComposeResult:
        """Compose the chat input layout."""
        yield Static(">", classes="input-prompt", id="prompt")
        yield Input(placeholder="Type a message...", id="chat-input")

    def on_mount(self) -> None:
        """Focus the input when mounted."""
        self._input = self.query_one("#chat-input", Input)
        self._input.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Detect input mode based on first character."""
        text = event.value
        if text.startswith("!"):
            self.mode = "bash"
        elif text.startswith("/"):
            self.mode = "command"
        else:
            self.mode = "normal"

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        value = event.value.strip()
        if value:
            self.post_message(self.Submitted(value, self.mode))
            event.input.value = ""
            self.mode = "normal"

    def watch_mode(self, mode: str) -> None:
        """Post mode changed message when mode changes."""
        self.post_message(self.ModeChanged(mode))

    def focus_input(self) -> None:
        """Focus the input field."""
        if self._input:
            self._input.focus()

    @property
    def value(self) -> str:
        """Get the current input value."""
        if self._input:
            return self._input.value
        return ""

    @value.setter
    def value(self, val: str) -> None:
        """Set the input value."""
        if self._input:
            self._input.value = val

    @property
    def input_widget(self) -> Input | None:
        """Get the underlying Input widget for autocomplete attachment."""
        return self._input

    def set_disabled(self, disabled: bool) -> None:
        """Enable or disable the input widget.

        When disabled, the input cannot receive focus or input.

        Args:
            disabled: True to disable, False to enable
        """
        if self._input:
            self._input.disabled = disabled
            if disabled:
                # Blur the input when disabling
                self._input.blur()
            else:
                # When re-enabling, don't auto-focus - let caller decide
                pass
