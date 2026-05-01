"""Notification center modal for pending actionable notices.

Surfaces a list of `PendingNotification` entries as single-line rows.
Selecting a row drills into a dedicated detail modal
(`UpdateAvailableScreen` for update entries, `NotificationDetailScreen`
otherwise) stacked on top of the center. When the detail modal
dismisses with any non-SUPPRESS action the center dismisses with a
`NotificationActionResult` so the app layer can dispatch; SUPPRESS is
handled in place via `NotificationSuppressRequested` so the remaining
notifications stay reachable. When the detail cancels, the center
stays open on the list.
"""

from __future__ import annotations

import logging
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

    from deepagents_cli.notifications import PendingNotification

from deepagents_cli import theme
from deepagents_cli.config import get_glyphs, is_ascii_mode
from deepagents_cli.notifications import ActionId, UpdateAvailablePayload

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NotificationActionResult:
    """Dismissal payload identifying which action the user picked.

    The screen returns this via `dismiss()` when the user drills into
    a notification and selects an action; it returns `None` when the
    user cancels with Esc without committing to an action.
    """

    key: str
    """Registry key of the notification the action was picked for."""

    action_id: ActionId
    """Identifier of the chosen `NotificationAction`."""


class NotificationRowClicked(Message):
    """Posted when a notification row is clicked with the mouse."""

    def __init__(self, key: str) -> None:
        """Initialize the message.

        Args:
            key: Registry key of the clicked notification. Using a key
                instead of an index keeps the message valid across
                `reload()` rebuilds, which replace the row list.
        """
        super().__init__()
        self.key = key


class NotificationSuppressRequested(Message):
    """Posted when the user picks SUPPRESS from a notification's detail modal.

    The center does not dismiss on SUPPRESS because the remaining
    notifications should still be reachable in place. The app handles
    this message by running the suppress dispatch and calling
    `NotificationCenterScreen.reload` with the refreshed registry
    snapshot.
    """

    def __init__(self, key: str) -> None:
        """Initialize the message.

        Args:
            key: Registry key of the notification being suppressed.
        """
        super().__init__()
        self.key = key


class _NotificationRow(Static):
    """Clickable single-line row displaying a notification's title."""

    def __init__(self, notification: PendingNotification, index: int) -> None:
        """Initialize the row widget.

        Args:
            notification: The entry to render.
            index: Position in the parent's list.
        """
        super().__init__(id=f"nc-row-{index}", classes="nc-row")
        self._notification = notification
        self._index = index
        self._is_selected = False
        self.update(self._render())

    @property
    def notification(self) -> PendingNotification:
        """Return the underlying notification."""
        return self._notification

    @property
    def index(self) -> int:
        """Return the row index in the parent list."""
        return self._index

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
        return Content.assemble(
            f"{cursor} ",
            (self._notification.title, "bold"),
        )

    def on_click(self, event: Click) -> None:
        """Dispatch a click as a `NotificationRowClicked` message."""
        event.stop()
        self.post_message(NotificationRowClicked(self._notification.key))


class NotificationCenterScreen(ModalScreen[NotificationActionResult | None]):
    """Modal listing pending notifications with drill-in details.

    Each `PendingNotification` is a single row. Up/Down (or j/k)
    moves the cursor; Enter or click pushes a detail modal for the
    highlighted entry. The detail modal carries the action list and
    dismisses with an `ActionId` or `None`. Esc on the center returns
    `None`.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Close", show=False),
        Binding("up", "move_up", "Up", show=False, priority=True),
        Binding("k", "move_up", "Up", show=False, priority=True),
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("j", "move_down", "Down", show=False, priority=True),
        Binding("tab", "move_down", "Next", show=False, priority=True),
        Binding("enter", "activate", "Open", show=False, priority=True),
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

    NotificationCenterScreen .nc-row {
        height: 1;
        padding: 0 1;
        color: $text;
    }

    NotificationCenterScreen .nc-row:hover {
        background: $surface-lighten-1;
    }

    NotificationCenterScreen .nc-row.-selected {
        background: $surface-lighten-1;
    }

    NotificationCenterScreen .nc-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }
    """

    def __init__(self, notifications: list[PendingNotification]) -> None:
        """Initialize the screen with a snapshot of pending notifications.

        Args:
            notifications: Entries to render. Order is preserved.
        """
        super().__init__()
        self._notifications = notifications
        self._selected: int = 0
        self._rows: list[_NotificationRow] = []
        self._drilling = False

    def compose(self) -> ComposeResult:
        """Compose the modal layout.

        Yields:
            The title widget, one row per pending notification, and a
            help footer.
        """
        glyphs = get_glyphs()
        with Vertical():
            yield Static("Notifications", classes="nc-title")
            with VerticalScroll():
                for idx, notif in enumerate(self._notifications):
                    row = _NotificationRow(notif, idx)
                    self._rows.append(row)
                    yield row
            help_text = (
                f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate "
                f"{glyphs.bullet} Enter open "
                f"{glyphs.bullet} Esc close"
            )
            yield Static(help_text, classes="nc-help")

    def on_mount(self) -> None:
        """Apply ASCII borders and highlight the first row."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.primary)
        if self._rows:
            self._rows[0].set_selected(selected=True)
            self._rows[0].scroll_visible()

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
        self._rows[self._selected].set_selected(selected=False)
        self._selected = new_index
        self._rows[new_index].set_selected(selected=True)
        self._rows[new_index].scroll_visible()

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
        """Drill into the highlighted notification."""
        if not self._rows:
            self.dismiss(None)
            return
        self._drill_into(self._notifications[self._selected])

    def action_cancel(self) -> None:
        """Close without firing any action."""
        self.dismiss(None)

    def on_notification_row_clicked(self, message: NotificationRowClicked) -> None:
        """Handle a mouse click on a notification row."""
        message.stop()
        index = next(
            (i for i, n in enumerate(self._notifications) if n.key == message.key),
            None,
        )
        if index is None:
            # Row was rebuilt out from under the click (reload race);
            # surface at debug level so regressions stay diagnosable.
            logger.debug("Ignoring click on unknown notification key %r", message.key)
            return
        self._set_selected(index)
        self._drill_into(self._notifications[index])

    def _drill_into(self, entry: PendingNotification) -> None:
        """Push a detail modal for *entry*.

        Guarded against reentry so a rapid double-activation (e.g.
        keyboard repeat) does not stack two detail modals.

        Args:
            entry: The notification to drill into.
        """
        if self._drilling:
            return
        detail_screen = self._detail_screen_for(entry)
        self._drilling = True

        def handle_detail(action_id: ActionId | None) -> None:
            self._drilling = False
            if action_id is None:
                return
            if action_id == ActionId.SUPPRESS:
                # Keep the center open; the app handles dispatch and
                # calls `reload` with the refreshed list. Rationale is
                # in `NotificationSuppressRequested`'s class docstring.
                self.post_message(NotificationSuppressRequested(entry.key))
                return
            self.dismiss(NotificationActionResult(entry.key, action_id))

        try:
            self.app.push_screen(detail_screen, handle_detail)
        except Exception:
            # push_screen raising would otherwise leave `_drilling`
            # permanently True and wedge the center.
            self._drilling = False
            raise

    async def reload(self, notifications: list[PendingNotification]) -> None:
        """Rebuild the row list from a refreshed snapshot.

        Preserves cursor position by key when possible; falls back to
        clamping the previous index into the new bounds. Dismisses the
        screen with `None` when the list is empty.

        Args:
            notifications: Current pending entries to display.
        """
        if not notifications:
            self.dismiss(None)
            return
        prev_key: str | None = None
        if self._notifications and 0 <= self._selected < len(self._notifications):
            prev_key = self._notifications[self._selected].key
        new_selected = next(
            (i for i, n in enumerate(notifications) if n.key == prev_key),
            min(self._selected, len(notifications) - 1) if prev_key else 0,
        )

        scroll = self.query_one(VerticalScroll)
        await scroll.remove_children()
        new_rows = [
            _NotificationRow(notif, idx) for idx, notif in enumerate(notifications)
        ]
        await scroll.mount(*new_rows)
        self._rows = new_rows
        self._notifications = notifications
        self._selected = new_selected
        new_rows[new_selected].set_selected(selected=True)
        new_rows[new_selected].scroll_visible()

    @staticmethod
    def _detail_screen_for(
        entry: PendingNotification,
    ) -> ModalScreen[ActionId | None]:
        """Pick the appropriate detail modal for *entry*'s payload.

        Update-available entries use the dedicated
        `UpdateAvailableScreen` (which adds a changelog row); all
        other payloads use the generic `NotificationDetailScreen`.

        Returns:
            A `ModalScreen` whose `dismiss()` payload is the selected
            `ActionId` or `None` when the user cancels.
        """
        if isinstance(entry.payload, UpdateAvailablePayload):
            from deepagents_cli.widgets.update_available import UpdateAvailableScreen

            return UpdateAvailableScreen(entry)
        from deepagents_cli.widgets.notification_detail import NotificationDetailScreen

        return NotificationDetailScreen(entry)
