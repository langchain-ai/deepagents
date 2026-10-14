"""Floating action for asking about selected conversation text."""

from textual.events import Click, MouseDown, MouseUp
from textual.message import Message
from textual.widgets import Static


class SelectionAction(Static):
    """Clickable selection tooltip that retains text while focus changes."""

    DEFAULT_CSS = """
    SelectionAction {
        layer: tooltip;
        position: absolute;
        width: 18;
        height: 1;
        padding: 0 1;
        background: $panel;
        color: $text;
        text-style: bold;
    }
    SelectionAction:hover {
        background: $accent;
        color: $text-accent;
    }
    """

    class Ask(Message):
        """Request insertion of the captured selection into the chat input."""

        def __init__(self, text: str) -> None:
            """Capture the selected text."""
            super().__init__()
            self.text = text

    def __init__(self, text: str) -> None:
        """Create the action with a snapshot of the selection."""
        super().__init__("Ask about this", markup=False)
        self.selected_text = text

    @property
    def allow_select(self) -> bool:
        """Keep the action label out of terminal selections."""
        return False

    @staticmethod
    def on_mouse_down(event: MouseDown) -> None:
        """Keep the selection action from triggering app mouse handling."""
        event.stop()

    @staticmethod
    def on_mouse_up(event: MouseUp) -> None:
        """Preserve the captured selection until the click is handled."""
        event.stop()

    def on_click(self, event: Click) -> None:
        """Send the captured text to the chat input."""
        event.stop()
        self.post_message(self.Ask(self.selected_text))
        self.remove()
