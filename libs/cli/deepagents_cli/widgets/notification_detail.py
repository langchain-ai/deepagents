"""Generic detail modal for a single pending notification.

Used by `NotificationCenterScreen` when the user drills into an entry
whose payload does not have a dedicated modal (e.g. missing-dependency
notices). Update-available notifications continue to use
`UpdateAvailableScreen`, which adds a changelog row on top of the
action list.

Dismisses with the selected `ActionId`, or `None` on Esc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Click

    from deepagents_cli.notifications import (
        ActionId,
        NotificationAction,
        PendingNotification,
    )

from deepagents_cli import theme
from deepagents_cli.config import get_glyphs, is_ascii_mode


class DetailActionActivated(Message):
    """Posted when an `_ActionOption` is clicked with the mouse."""

    def __init__(self, action_id: ActionId) -> None:
        """Initialize the message.

        Args:
            action_id: The action the clicked row represents.
        """
        super().__init__()
        self.action_id = action_id


class _ActionOption(Static):
    """Clickable single-line action row."""

    def __init__(self, action: NotificationAction, widget_id: str) -> None:
        """Initialize the action row widget.

        Args:
            action: The action this row represents.
            widget_id: DOM id assigned to the widget.
        """
        super().__init__(id=widget_id, classes="nd-action")
        self._action = action
        self._is_selected = False
        self.update(self._render())

    @property
    def action(self) -> NotificationAction:
        """Return the underlying action."""
        return self._action

    def set_selected(self, selected: bool) -> None:
        """Toggle selection styling.

        Args:
            selected: Whether this row is currently under the cursor.
        """
        if self._is_selected == selected:
            return
        self._is_selected = selected
        self.set_class(selected, "-selected")
        self.update(self._render())

    def _render(self) -> Content:
        glyphs = get_glyphs()
        cursor = glyphs.cursor if self._is_selected else " "
        text = f"{cursor} {self._action.label}"
        if self._action.primary:
            return Content.styled(text, "bold")
        return Content(text)

    def on_click(self, event: Click) -> None:
        """Dispatch a click as a `DetailActionActivated` message."""
        event.stop()
        self.post_message(DetailActionActivated(self._action.action_id))


class NotificationDetailScreen(ModalScreen["ActionId | None"]):
    """Modal displaying a single notification's title, body, and actions.

    Activation returns the chosen `ActionId` via `dismiss()`; Esc
    returns `None` so the caller can keep the underlying notification
    center open.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Back", show=False),
        Binding("up", "move_up", "Up", show=False, priority=True),
        Binding("k", "move_up", "Up", show=False, priority=True),
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("j", "move_down", "Down", show=False, priority=True),
        Binding("tab", "move_down", "Next", show=False, priority=True),
        Binding("shift+tab", "move_up", "Previous", show=False, priority=True),
        Binding("enter", "activate", "Select", show=False, priority=True),
    ]

    CSS = """
    NotificationDetailScreen {
        align: center middle;
    }

    NotificationDetailScreen > Vertical {
        width: 68;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    NotificationDetailScreen .nd-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    NotificationDetailScreen .nd-body {
        color: $text-muted;
        margin-bottom: 1;
    }

    NotificationDetailScreen .nd-action {
        height: auto;
        padding: 0 1;
        color: $text;
    }

    NotificationDetailScreen .nd-action:hover {
        background: $surface-lighten-1;
    }

    NotificationDetailScreen .nd-action.-selected {
        background: $surface-lighten-1;
    }

    NotificationDetailScreen .nd-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }
    """

    def __init__(self, entry: PendingNotification) -> None:
        """Initialize the screen.

        Args:
            entry: Notification to render. Its actions drive the row
                list; dismissing via an action returns the chosen
                `ActionId` to the caller.
        """
        super().__init__()
        self._entry = entry
        self._options: list[_ActionOption] = []
        self._selected = 0

    def compose(self) -> ComposeResult:
        """Compose the modal layout.

        Yields:
            Title, optional body, one `_ActionOption` per action, and
            a help footer.
        """
        glyphs = get_glyphs()
        with Vertical():
            yield Static(self._entry.title, classes="nd-title", markup=False)
            if self._entry.body:
                yield Static(self._entry.body, classes="nd-body", markup=False)
            for idx, action in enumerate(self._entry.actions):
                option = _ActionOption(action, f"nd-row-{idx}")
                self._options.append(option)
                yield option
            help_text = (
                f"{glyphs.arrow_up}/{glyphs.arrow_down} or Tab navigate "
                f"{glyphs.bullet} Enter select "
                f"{glyphs.bullet} Esc back"
            )
            yield Static(help_text, classes="nd-help")

    def on_mount(self) -> None:
        """Apply ASCII borders and highlight the primary action by default."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.primary)
        if not self._options:
            return
        primary_idx = next(
            (i for i, a in enumerate(self._entry.actions) if a.primary),
            0,
        )
        self._set_selected(primary_idx)

    def _set_selected(self, new_index: int) -> None:
        """Move the selection cursor to *new_index*.

        Raises:
            IndexError: When *new_index* is outside `0..len(self._options)`.
        """
        if not self._options:
            return
        if not 0 <= new_index < len(self._options):
            msg = f"selection {new_index} out of range 0..{len(self._options)}"
            raise IndexError(msg)
        if new_index != self._selected:
            self._options[self._selected].set_selected(selected=False)
        self._selected = new_index
        self._options[new_index].set_selected(selected=True)

    def action_move_up(self) -> None:
        """Move the cursor up one row (wraps at the top)."""
        if not self._options:
            return
        self._set_selected((self._selected - 1) % len(self._options))

    def action_move_down(self) -> None:
        """Move the cursor down one row (wraps at the bottom)."""
        if not self._options:
            return
        self._set_selected((self._selected + 1) % len(self._options))

    def action_activate(self) -> None:
        """Dismiss with the highlighted action's id."""
        if not self._options:
            self.dismiss(None)
            return
        self.dismiss(self._options[self._selected].action.action_id)

    def action_cancel(self) -> None:
        """Close without firing any action."""
        self.dismiss(None)

    def on_detail_action_activated(self, message: DetailActionActivated) -> None:
        """Handle a mouse click on an action row."""
        message.stop()
        self.dismiss(message.action_id)
