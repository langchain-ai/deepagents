"""Notification center modal for pending actionable notices.

Surfaces a list of `PendingNotification` entries as cards with
keyboard-selectable action rows. Selecting an action dismisses the
modal with a `NotificationActionResult` describing the user's choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Click

    from deepagents_cli.notifications import NotificationAction, PendingNotification

from deepagents_cli import theme
from deepagents_cli.config import get_glyphs, is_ascii_mode


@dataclass(frozen=True)
class NotificationActionResult:
    """Dismissal payload identifying which action the user picked.

    The screen returns this via `dismiss()` when the user selects an
    action; it returns `None` when the user cancels with Esc.
    """

    key: str
    """Registry key of the notification the action was picked for."""

    action_id: str
    """Identifier of the chosen `NotificationAction`."""


@dataclass(frozen=True)
class _ActionRow:
    """Flat entry in the navigation cursor's linear list of actions."""

    notification_key: str
    """Registry key of the notification this row belongs to."""

    action: NotificationAction
    """The action (label + id) rendered on this row."""

    widget_id: str
    """DOM id assigned to the `_ActionOption` widget for this row."""


class ActionActivated(Message):
    """Posted when an `_ActionOption` is clicked with the mouse."""

    def __init__(self, row: _ActionRow) -> None:
        """Initialize the message.

        Args:
            row: The action row whose widget was clicked.
        """
        super().__init__()
        self.row = row


class _ActionOption(Static):
    """Clickable single-line action row rendered inside a notification card."""

    def __init__(self, row: _ActionRow) -> None:
        """Initialize the action row widget.

        Args:
            row: The action row this widget represents.
        """
        super().__init__(id=row.widget_id, classes="nc-action")
        self._row = row
        self._is_selected = False
        self.update(self._render())

    @property
    def row(self) -> _ActionRow:
        """Return the underlying action row."""
        return self._row

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
        text = f"{cursor} {self._row.action.label}"
        if self._row.action.primary:
            return Content.styled(text, "bold")
        return Content(text)

    def on_click(self, event: Click) -> None:
        """Dispatch a click as an `ActionActivated` message."""
        event.stop()
        self.post_message(ActionActivated(self._row))


class NotificationCenterScreen(ModalScreen[NotificationActionResult | None]):
    """Modal screen listing pending notifications with per-row actions.

    Displays each `PendingNotification` as a card (title, body, action
    rows). A single linear cursor navigates across all action rows so
    up/down moves intuitively through the flattened list regardless of
    which card the action belongs to. Enter fires the highlighted
    action; Esc dismisses without an action.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Close", show=False),
        Binding("up", "move_up", "Up", show=False, priority=True),
        Binding("k", "move_up", "Up", show=False, priority=True),
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("j", "move_down", "Down", show=False, priority=True),
        Binding("enter", "activate", "Select", show=False, priority=True),
    ]

    CSS = """
    NotificationCenterScreen {
        align: center middle;
        background: transparent;
    }

    NotificationCenterScreen > Vertical {
        width: 72;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    NotificationCenterScreen .nc-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    NotificationCenterScreen VerticalScroll {
        height: auto;
        max-height: 24;
    }

    NotificationCenterScreen .nc-card-title {
        text-style: bold;
    }

    NotificationCenterScreen .nc-card-body {
        color: $text-muted;
        margin-bottom: 1;
    }

    NotificationCenterScreen .nc-action {
        height: auto;
        padding: 0 1;
        color: $text;
    }

    NotificationCenterScreen .nc-action.-selected {
        background: $surface-lighten-1;
    }

    NotificationCenterScreen .nc-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }

    NotificationCenterScreen .nc-separator {
        height: 1;
        color: $text-muted;
        margin: 0 1;
    }
    """

    def __init__(self, notifications: list[PendingNotification]) -> None:
        """Initialize the screen with a snapshot of pending notifications.

        Args:
            notifications: Entries to render. Order is preserved.
        """
        super().__init__()
        self._notifications = notifications
        self._rows_by_key: dict[str, list[_ActionRow]] = {}
        self._rows: list[_ActionRow] = []
        for card_idx, notif in enumerate(notifications):
            card_rows: list[_ActionRow] = []
            for action_idx, action in enumerate(notif.actions):
                widget_id = f"nc-row-{card_idx}-{action_idx}"
                row = _ActionRow(notif.key, action, widget_id)
                card_rows.append(row)
                self._rows.append(row)
            self._rows_by_key[notif.key] = card_rows
        self._selected: int = 0
        self._option_widgets: dict[str, _ActionOption] = {}

    def compose(self) -> ComposeResult:
        """Compose the modal layout.

        Yields:
            The title widget, one card per pending notification (separated
            by divider lines), and a help footer.
        """
        glyphs = get_glyphs()
        with Vertical():
            yield Static("Notifications", classes="nc-title")
            with VerticalScroll():
                last_idx = len(self._notifications) - 1
                for card_idx, notif in enumerate(self._notifications):
                    yield from self._compose_card(notif, is_last=card_idx == last_idx)
            help_text = (
                f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate "
                f"{glyphs.bullet} Enter select "
                f"{glyphs.bullet} Esc close"
            )
            yield Static(help_text, classes="nc-help")

    def _compose_card(
        self, notif: PendingNotification, *, is_last: bool
    ) -> ComposeResult:
        """Render one notification card (title, body, action rows).

        Args:
            notif: The entry to render.
            is_last: When `False`, append a separator line after the card.

        Yields:
            The title, body (if present), one `_ActionOption` per action,
            and an optional separator.
        """
        yield Static(notif.title, classes="nc-card-title", markup=False)
        if notif.body:
            yield Static(notif.body, classes="nc-card-body", markup=False)
        for row in self._rows_by_key[notif.key]:
            option = _ActionOption(row)
            self._option_widgets[row.widget_id] = option
            yield option
        if not is_last:
            yield Static("─" * 60, classes="nc-separator")

    def on_mount(self) -> None:
        """Apply ASCII borders and highlight the first action row."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.primary)
        if self._rows:
            option = self._option_widgets.get(self._rows[0].widget_id)
            if option is not None:
                option.set_selected(selected=True)
                option.scroll_visible()

    def _set_selected(self, new_index: int) -> None:
        """Move the selection cursor to *new_index*.

        Raises:
            IndexError: If *new_index* is outside `0..len(self._rows)`.
        """
        if not self._rows or new_index == self._selected:
            return
        if not 0 <= new_index < len(self._rows):
            msg = f"selection {new_index} out of range 0..{len(self._rows)}"
            raise IndexError(msg)
        prev = self._option_widgets.get(self._rows[self._selected].widget_id)
        if prev is not None:
            prev.set_selected(selected=False)
        self._selected = new_index
        target = self._option_widgets.get(self._rows[new_index].widget_id)
        if target is not None:
            target.set_selected(selected=True)
            target.scroll_visible()

    def action_move_up(self) -> None:
        """Move the cursor up one row (wraps at the top)."""
        if not self._rows:
            return
        self._set_selected((self._selected - 1) % len(self._rows))

    def action_move_down(self) -> None:
        """Move the cursor down one row (wraps at the bottom)."""
        if not self._rows:
            return
        self._set_selected((self._selected + 1) % len(self._rows))

    def action_activate(self) -> None:
        """Fire the highlighted action."""
        if not self._rows:
            self.dismiss(None)
            return
        row = self._rows[self._selected]
        self.dismiss(
            NotificationActionResult(row.notification_key, row.action.action_id)
        )

    def action_cancel(self) -> None:
        """Close without firing any action."""
        self.dismiss(None)

    def on_action_activated(self, message: ActionActivated) -> None:
        """Handle a mouse click on an action row."""
        message.stop()
        self.dismiss(
            NotificationActionResult(
                message.row.notification_key, message.row.action.action_id
            )
        )
