"""Floating action for asking about selected conversation text."""

from collections.abc import Iterable

from textual.events import Click, MouseDown, MouseUp
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

from deepagents_code.tui.widgets.messages import DiffMessage, UserMessage

_TRANSCRIPT_IDS = frozenset(
    {
        "assistant-content",
        "reasoning-body",
        "task-desc",
        "args-full",
        "output-preview",
        "output-full",
    }
)
"""Widget ids whose subtree holds message or tool content, not chrome."""


def is_transcript_selection(widgets: Iterable[Widget]) -> bool:
    """Return whether every selected widget shows transcript text.

    Headers, toggles, hints, status lines, notices, and inline prompts in the
    conversation are UI chrome, so a selection that touches one is rejected.
    """
    return all(_is_transcript_widget(widget) for widget in widgets)


def _is_transcript_widget(widget: Widget) -> bool:
    """Return whether `widget` shows transcript text in the main conversation.

    A `diff-header` widget is always chrome. Otherwise `widget` or one of its
    ancestors must mark transcript content, and `#messages` must contain it,
    so messages rendered elsewhere (for example in `/btw`) do not qualify.
    """
    if widget.has_class("diff-header"):
        return False
    matched = False
    node = widget
    while node is not None:
        if node.id == "messages":
            return matched
        matched = matched or (
            isinstance(node, (UserMessage, DiffMessage))
            or node.id in _TRANSCRIPT_IDS
            or node.has_class("tool-args")
        )
        node = node.parent
    return False


class SelectionAction(Static):
    """Clickable selection tooltip that retains text while focus changes."""

    DEFAULT_CSS = """
    SelectionAction {
        layer: selection-action;
        position: absolute;
        width: 18;
        height: 1;
        padding: 0 1;
        background: $primary;
        color: $background;
        text-style: bold;
    }
    SelectionAction:hover {
        background: $primary-lighten-1;
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
