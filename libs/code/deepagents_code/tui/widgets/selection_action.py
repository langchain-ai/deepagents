"""Floating action for asking about selected conversation text."""

from collections.abc import Collection

from textual.events import Click, MouseUp
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

from deepagents_code.tui.widgets.messages import DiffMessage, UserMessage

_TRANSCRIPT_IDS = frozenset(
    {
        "assistant-content",
        "reasoning-body",
        "task-desc",
        "output-preview",
        "output-full",
    }
)
"""Widget ids whose subtree holds message or tool content, not chrome."""

_LABEL = "Ask about this"

SELECTION_ACTION_WIDTH = len(_LABEL) + 2
"""Cells the action occupies: its label plus one cell of padding per side."""


def is_transcript_selection(widgets: Collection[Widget]) -> bool:
    """Return whether a non-empty selection shows only transcript text.

    Headers, toggles, hints, status lines, notices, and inline prompts in the
    conversation are UI chrome, so a selection that touches one is rejected.
    `UserMessage` and `DiffMessage` count as transcript as a whole, apart from
    a diff header.
    """
    return bool(widgets) and all(_is_transcript_widget(widget) for widget in widgets)


def _is_transcript_widget(widget: Widget) -> bool:
    """Return whether `widget` shows transcript text in the main conversation.

    A `diff-header` widget is always chrome. Otherwise `widget` or one of its
    ancestors must mark transcript content, and `#messages` must contain it,
    so message widgets rendered anywhere else, such as on another screen, do
    not qualify.
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


def quote_after(draft: str, text: str) -> str:
    """Return `text` as a Markdown blockquote to append after `draft`.

    A non-empty draft gets exactly one blank line before the quote, and the
    quote ends with a blank line so the user can type below it.
    """
    quote = "\n".join("> " + line for line in text.split("\n"))
    trailing = len(draft) - len(draft.rstrip("\n"))
    separator = "\n" * max(0, 2 - trailing) if draft else ""
    return separator + quote + "\n\n"


class SelectionAction(Static):
    """Floating "Ask about this" button for the current selection.

    It snapshots the selected text because clicking it clears the screen
    selection before the `Click` arrives.
    """

    DEFAULT_CSS = """
    SelectionAction {
        layer: selection-action;
        position: absolute;
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
        super().__init__(_LABEL, markup=False)
        self.selected_text = text
        self.styles.width = SELECTION_ACTION_WIDTH

    @property
    def allow_select(self) -> bool:
        """Keep the action label out of terminal selections."""
        return False

    @staticmethod
    def on_mouse_up(event: MouseUp) -> None:
        """Stop the app's mouse-up handler from dismissing this before `Click`."""
        event.stop()

    def on_click(self, event: Click) -> None:
        """Send the captured text to the chat input."""
        event.stop()
        self.post_message(self.Ask(self.selected_text))
