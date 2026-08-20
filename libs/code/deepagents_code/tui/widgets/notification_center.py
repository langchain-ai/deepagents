"""Notification hub for pending notices and warning preferences.

Surfaces `PendingNotification` entries as single-line rows plus a persistent
settings destination, including when no notices are pending. Selecting a notice
drills into a dedicated detail modal
(`UpdateAvailableScreen` for update entries, `NotificationDetailScreen`
otherwise) stacked on top of the center. When the detail modal
dismisses with a terminal action (one that closes the center) the
center dismisses with a `NotificationActionResult` so the app layer can
dispatch. Actions that must keep the center open are handled in place:
SUPPRESS via
`NotificationSuppressRequested` (so the remaining notifications stay
reachable) and actions in `IN_PLACE_ACTIONS` via
`NotificationActionRequested` (so a follow-up modal, e.g. the API-key
prompt, stacks on top and Esc returns to the center). When the detail
cancels, the center stays open on the list.
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

    from deepagents_code.notifications import PendingNotification

from deepagents_code import theme
from deepagents_code.config import get_glyphs, is_ascii_mode
from deepagents_code.notifications import ActionId, UpdateAvailablePayload

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


class NotificationSettingsRequested(Message):
    """Posted when the user opens settings from the notification center."""


IN_PLACE_ACTIONS: frozenset[ActionId] = frozenset({ActionId.ENTER_API_KEY})
"""Actions handled in place without dismissing the center.

Each opens a follow-up modal on top of the center, so the center stays
mounted and Esc in that modal returns here (rationale in
`NotificationActionRequested`). SUPPRESS is also handled in place but
routes through its own `NotificationSuppressRequested` message, so it is
deliberately excluded from this set.
"""


class NotificationActionRequested(Message):
    """Posted for an action that opens a follow-up modal in place.

    Some actions (those in `IN_PLACE_ACTIONS`, currently `ENTER_API_KEY`)
    push another modal, such as the API-key prompt, on top of the
    still-open center. Dismissing the center first would drop that stack,
    so Esc in the follow-up modal would fall through to the base screen
    instead of returning here. The app handles this message by dispatching
    the action while the center stays mounted, then reloading it with the
    refreshed registry snapshot.
    """

    def __init__(self, key: str, action_id: ActionId) -> None:
        """Initialize the message.

        Args:
            key: Registry key of the notification the action targets.
            action_id: The in-place action the user selected. Must be a
                member of `IN_PLACE_ACTIONS`.

        Raises:
            ValueError: If `action_id` is not an in-place action, which
                would be a programmer error (the message is only meant to
                carry actions that keep the center open).
        """
        super().__init__()
        if action_id not in IN_PLACE_ACTIONS:
            msg = f"{action_id} is not an in-place action"
            raise ValueError(msg)
        self.key = key
        self.action_id = action_id


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
        """Underlying notification."""
        return self._notification

    @property
    def index(self) -> int:
        """Row index in the parent list."""
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


class _NotificationSettingsRow(Static):
    """Selectable row that opens notification warning preferences."""

    def __init__(self, index: int) -> None:
        """Initialize the settings row.

        Args:
            index: Position in the center's selectable rows.
        """
        super().__init__(id="nc-settings", classes="nc-row")
        self._index = index
        self._is_selected = False
        self.update(self._render())

    @property
    def index(self) -> int:
        """Row index in the center's selectable rows."""
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
            ("Notification settings", "bold"),
        )

    def on_click(self, event: Click) -> None:
        """Request notification settings when clicked."""
        event.stop()
        self.post_message(NotificationSettingsRequested())


class NotificationCenterScreen(ModalScreen[NotificationActionResult | None]):
    """Shared hub for pending notifications and warning preferences.

    Each `PendingNotification` is a single row followed by a persistent
    settings row. Up/Down (or j/k) moves the cursor; Enter or click opens
    the highlighted destination. Notification details carry an `ActionId`
    or `None`, while settings stay stacked over the hub. Esc returns `None`.
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

    NotificationCenterScreen .nc-section {
        height: 1;
        color: $text-muted;
        text-style: bold;
        margin-top: 1;
    }

    NotificationCenterScreen .nc-section-first {
        margin-top: 0;
    }

    NotificationCenterScreen .nc-empty {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        text-style: italic;
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
        self._rows: list[_NotificationRow | _NotificationSettingsRow] = []
        self._drilling = False

    def _build_list(self) -> list[Static]:
        """Build list widgets and refresh the selectable row index.

        Returns:
            Static widgets for the notification hub's scrollable list.
        """
        widgets: list[Static] = [
            Static("Pending", classes="nc-section nc-section-first")
        ]
        rows: list[_NotificationRow | _NotificationSettingsRow] = []
        if self._notifications:
            for idx, notification in enumerate(self._notifications):
                row = _NotificationRow(notification, idx)
                rows.append(row)
                widgets.append(row)
        else:
            widgets.append(Static("No pending notifications.", classes="nc-empty"))
        settings = _NotificationSettingsRow(len(rows))
        rows.append(settings)
        widgets.extend(
            [
                Static("Preferences", classes="nc-section"),
                settings,
            ]
        )
        self._rows = rows
        return widgets

    def compose(self) -> ComposeResult:
        """Compose the modal layout.

        Yields:
            Pending notifications, the settings row, and navigation help.
        """
        glyphs = get_glyphs()
        with Vertical():
            yield Static("Notifications", classes="nc-title")
            with VerticalScroll():
                yield from self._build_list()
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
        """Open the highlighted notification or settings destination."""
        if not self._rows:
            return
        row = self._rows[self._selected]
        if isinstance(row, _NotificationSettingsRow):
            self.post_message(NotificationSettingsRequested())
            return
        self._drill_into(row.notification)

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
            if action_id in IN_PLACE_ACTIONS:
                # Keep the center open so the follow-up modal (e.g. the
                # API-key prompt) stacks on top and Esc returns here.
                # Rationale is in `NotificationActionRequested`'s docstring.
                self.post_message(NotificationActionRequested(entry.key, action_id))
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
        """Rebuild the hub from a refreshed notification snapshot.

        Preserves a notification selection by key and keeps the settings row
        selected across refreshes. When a selected notification disappears,
        the cursor clamps to the remaining notification rows before falling
        back to settings.

        Args:
            notifications: Current pending entries to display.
        """
        prev_key: str | None = None
        settings_selected = False
        if self._rows and 0 <= self._selected < len(self._rows):
            selected_row = self._rows[self._selected]
            if isinstance(selected_row, _NotificationSettingsRow):
                settings_selected = True
            else:
                prev_key = selected_row.notification.key

        self._notifications = notifications
        scroll = self.query_one(VerticalScroll)
        await scroll.remove_children()
        await scroll.mount(*self._build_list())

        if settings_selected:
            new_selected = len(self._rows) - 1
        elif prev_key is not None:
            new_selected = next(
                (
                    i
                    for i, notification in enumerate(notifications)
                    if notification.key == prev_key
                ),
                min(self._selected, len(notifications) - 1)
                if notifications
                else len(self._rows) - 1,
            )
        else:
            new_selected = 0
        self._selected = new_selected
        self._rows[new_selected].set_selected(selected=True)
        self._rows[new_selected].scroll_visible()

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
            from deepagents_code.tui.widgets.update_available import (
                UpdateAvailableScreen,
            )

            return UpdateAvailableScreen(entry)
        from deepagents_code.tui.widgets.notification_detail import (
            NotificationDetailScreen,
        )

        return NotificationDetailScreen(entry)
