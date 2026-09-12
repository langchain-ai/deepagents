"""Notification hub for pending notices and warning preferences.

Surfaces `PendingNotification` entries as single-line rows plus an expandable
settings section, including when no notices are pending. Selecting a notice
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
cancels, the center stays open on the list. Warning preferences expand
inline under "Notification settings" so toggles never leave the hub.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Checkbox, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Click

    from deepagents_code.notifications import PendingNotification

from deepagents_code import theme
from deepagents_code._paths import PATHS
from deepagents_code.config import get_glyphs, is_ascii_mode
from deepagents_code.notifications import ActionId, UpdateAvailablePayload
from deepagents_code.tui.key_hints import modal_navigation_hint
from deepagents_code.tui.widgets.notification_settings import WARNING_TOGGLES

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
    """Posted when the user toggles the settings disclosure row.

    The app responds by refreshing the open center's settings state from
    config; the center also handles the message itself so toggles stay
    responsive in isolation (e.g. widget tests on a bare host app).
    """


_PRELOAD_TIMEOUT_SECONDS = 3.0
"""How long an expand waits on the mount-time preferences read.

Generous for a local config read, short enough that a stalled filesystem
surfaces as a toast instead of an unresponsive settings row.
"""


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
    """Selectable disclosure row for the inline warning-preferences section."""

    def __init__(self, index: int, *, expanded: bool = False) -> None:
        """Initialize the settings row.

        Args:
            index: Position in the center's selectable rows.
            expanded: Whether the settings section is currently expanded.
        """
        super().__init__(id="nc-settings", classes="nc-row")
        self._index = index
        self._expanded = expanded
        self._is_selected = False
        self.update(self._render())

    @property
    def index(self) -> int:
        """Row index in the center's selectable rows."""
        return self._index

    def set_expanded(self, expanded: bool) -> None:
        """Redraw the leading glyph for the section's new state.

        Args:
            expanded: Whether the settings section is now expanded.
        """
        if self._expanded == expanded:
            return
        self._expanded = expanded
        self.update(self._render())

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
        # One leading glyph carries both the cursor and the disclosure
        # affordance, so a toggle changes a single position instead of two.
        # While expanded the disclosure glyph deliberately takes the
        # cursor's place: focus has moved to the checkboxes, so a row
        # cursor there would point at something that is not selected.
        glyphs = get_glyphs()
        if self._expanded:
            cursor = glyphs.disclosure_expanded
        elif self._is_selected:
            cursor = glyphs.cursor
        else:
            cursor = " "
        return Content.assemble(
            f"{cursor} ",
            ("Notification settings", "bold"),
        )

    def on_click(self, event: Click) -> None:
        """Request notification settings when clicked."""
        event.stop()
        self.post_message(NotificationSettingsRequested())


class _NotificationSettingsGroup(Vertical):
    """Container for the expanded warning checkboxes.

    Owns up/down/tab navigation so the keys cycle between this section's
    checkboxes instead of falling through to the center's row cursor or
    dropping focus. The center's priority cursor bindings are disabled while
    a checkbox is focused (see `NotificationCenterScreen.check_action`), so
    these run unshadowed.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "focus_previous", "Previous", show=False),
        Binding("k", "focus_previous", "Previous", show=False),
        Binding("down", "focus_next", "Next", show=False),
        Binding("j", "focus_next", "Next", show=False),
        Binding("tab", "focus_next", "Next", show=False),
        Binding("shift+tab", "focus_previous", "Previous", show=False),
    ]

    def _checkboxes(self) -> list[Checkbox]:
        """The mounted warning checkboxes in order.

        Returns:
            The `Checkbox` widgets in this group, in display order.
        """
        return list(self.query(Checkbox))

    def _cycle(self, step: int) -> None:
        """Move focus *step* checkboxes forward/back, wrapping at the ends.

        Args:
            step: `1` to advance, `-1` to go back.
        """
        checkboxes = self._checkboxes()
        if not checkboxes:
            return
        focused = self.app.focused
        if isinstance(focused, Checkbox) and focused in checkboxes:
            index = (checkboxes.index(focused) + step) % len(checkboxes)
        else:
            index = 0 if step > 0 else len(checkboxes) - 1
        checkboxes[index].focus()

    def action_focus_next(self) -> None:
        """Move focus to the next checkbox (wraps)."""
        self._cycle(1)

    def action_focus_previous(self) -> None:
        """Move focus to the previous checkbox (wraps)."""
        self._cycle(-1)


class NotificationCenterScreen(ModalScreen[NotificationActionResult | None]):
    """Shared hub for pending notifications and warning preferences.

    Each `PendingNotification` is a single row followed by a settings
    disclosure row. Up/Down (or j/k) moves the cursor; Enter or click drills
    into a notification or toggles the inline settings section. Expanded
    settings hand key focus to their first checkbox (Space/Enter toggle); Esc
    there collapses back to the row cursor. Esc on the row cursor returns
    `None`.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Close", show=False, priority=True),
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

    NotificationCenterScreen #nc-settings-group {
        height: auto;
    }

    NotificationCenterScreen #nc-settings-group Checkbox {
        margin: 0;
        border: none;
        &:focus {
            border: none;
        }
    }

    NotificationCenterScreen .nc-help {
        dock: bottom;
        height: auto;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }
    """

    def __init__(
        self,
        notifications: list[PendingNotification],
        suppressed: set[str] | None = None,
    ) -> None:
        """Initialize the screen with a snapshot of pending notifications.

        Args:
            notifications: Entries to render. Order is preserved.
            suppressed: Currently suppressed warning keys, used to render
                checkbox state when the settings section expands. `None`
                means the app has not supplied settings state; the screen
                preloads it from config on mount so the first expand has
                the values ready.
        """
        super().__init__()
        self._notifications = notifications
        self._suppressed = suppressed
        self._selected: int = 0
        self._rows: list[_NotificationRow | _NotificationSettingsRow] = []
        self._drilling = False
        self._settings_expanded = False
        self._settings_loading = False
        self._settings_transitioning = False
        self._settings_preloaded = asyncio.Event()

    @property
    def settings_expanded(self) -> bool:
        """Whether the inline warning preferences are currently expanded."""
        return self._settings_expanded

    @property
    def settings_checkbox_focused(self) -> bool:
        """Whether key focus is on one of the settings checkboxes.

        Used by `DeepAgentsApp.action_toggle_auto_approve` to route
        shift+tab into reverse checkbox traversal instead of the row cursor.
        """
        return self._settings_has_focus()

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
        settings = _NotificationSettingsRow(len(rows), expanded=self._settings_expanded)
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
        with Vertical():
            yield Static("Notifications", classes="nc-title")
            with VerticalScroll():
                yield from self._build_list()
            yield Static(self._help_text(), classes="nc-help")

    def on_mount(self) -> None:
        """Apply ASCII borders, highlight the first row, preload settings."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.primary)
        if self._rows:
            self._rows[0].set_selected(selected=True)
            self._rows[0].scroll_visible()
        if self._suppressed is None:
            # Read the suppressed keys now so the first expand can mount the
            # checkboxes immediately. Loading on expand would leave the pane
            # open but unfocused for the duration of the config read, which
            # also delayed the footer hint's "Space/Enter toggle" verb.
            self.run_worker(
                self._preload_settings(),
                group="nc-preload",
                # A preferences read must not be able to kill the session.
                # The default would route any escape to `WorkerFailed`.
                exit_on_error=False,
            )

    async def _preload_settings(self) -> None:
        """Load suppressed keys in the background so expand need not wait."""
        self._settings_loading = True
        try:
            suppressed = await self._load_suppressed()
            if self._suppressed is None:
                self._suppressed = suppressed
        finally:
            # Release the waiter even when the read raises or the worker is
            # cancelled. `_expand_settings` blocks on this event, so a missed
            # `set()` would wedge the settings row for the life of the screen.
            self._settings_loading = False
            self._settings_preloaded.set()

    def _settings_has_focus(self) -> bool:
        """Whether key focus is inside the expanded settings checkboxes.

        Returns:
            `True` when the focused widget is one of the settings checkboxes.
        """
        return isinstance(self.app.focused, Checkbox)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Stand the screen's key bindings down while a checkbox has focus.

        The screen-level `enter`/`up`/`down`/`tab` bindings are priority, so
        without this gate they consume the key before the focused `Checkbox`
        can use it: Enter/Space must toggle (Textual binds them on
        `ToggleButton`), and up/down/tab must step between checkboxes.
        Returning `False` disables the screen binding for that dispatch so
        the key reaches the checkbox / focus system.

        Args:
            action: The action name being dispatched.
            parameters: The action's parameters.

        Returns:
            `False` to disable the binding, otherwise the superclass verdict.
        """
        if not (self._settings_expanded and self._settings_has_focus()):
            return super().check_action(action, parameters)
        # While a settings checkbox is focused it owns the keys: Enter/Space
        # toggle (Textual's `ToggleButton` binding), and up/down/tab step
        # between checkboxes via normal focus movement. The screen's priority
        # cursor bindings would otherwise swallow all of these.
        if action in {"activate", "move_up", "move_down"}:
            return False
        return super().check_action(action, parameters)

    def action_activate(self) -> None:
        """Drill into the highlighted notification or toggle settings."""
        if self._settings_expanded and self._settings_has_focus():
            # Focused checkboxes own Enter/Space for toggling.
            return
        if not self._rows:
            return
        row = self._rows[self._selected]
        if isinstance(row, _NotificationSettingsRow):
            self.post_message(NotificationSettingsRequested())
            return
        self._drill_into(row.notification)

    def action_cancel(self) -> None:
        """Collapse expanded settings, else close without firing an action."""
        if self._settings_expanded:
            self.run_worker(self._toggle_settings(), group="nc-settings")
            return
        self.dismiss(None)

    def _help_text(self) -> str:
        glyphs = get_glyphs()
        if self._settings_expanded:
            return (
                f"{modal_navigation_hint(glyphs)} "
                f"{glyphs.bullet} Space/Enter toggle "
                f"{glyphs.bullet} Esc collapse"
            )
        return (
            f"{modal_navigation_hint(glyphs)} "
            f"{glyphs.bullet} Enter open "
            f"{glyphs.bullet} Esc close"
        )

    def _refresh_help(self) -> None:
        self.query_one(".nc-help", Static).update(self._help_text())

    def _first_checkbox(self) -> Checkbox | None:
        """The first settings checkbox, if the section is mounted.

        Returns:
            The first `Checkbox` in the settings group, or `None` when the
            section is collapsed.
        """
        return next(iter(self.query(Checkbox)), None)

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

    def action_focus_previous(self) -> None:
        """Step checkbox focus backward for the app's shift+tab router.

        The app-level priority `shift+tab -> toggle_auto_approve` binding
        wins dispatch, and `action_toggle_auto_approve` routes cursor-style
        modals to `action_move_up` via `_SupportsReverseNav`. With settings
        expanded that would only move the row cursor and strand focus on the
        first checkbox, so the app branches here instead (see
        `DeepAgentsApp.action_toggle_auto_approve`).
        """
        group = next(iter(self.query(_NotificationSettingsGroup)), None)
        if group is not None:
            group.action_focus_previous()

    def on_notification_settings_requested(
        self, message: NotificationSettingsRequested
    ) -> None:
        """Toggle the settings section from the disclosure row.

        Handles the message locally so the section stays responsive when
        the app has no handler of its own (e.g. widget tests on a bare
        host app).
        """
        message.stop()
        self.run_worker(self._toggle_settings(), group="nc-settings")

    async def _toggle_settings(self) -> None:
        """Expand or collapse the inline warning-preferences section.

        Serialized against re-entry: the mount/unmount awaits yield, so a
        rapid second toggle (e.g. keyboard-repeat Esc) would otherwise start
        a second `nc-settings` worker — which is not exclusive by default —
        and the expand branch would mount a duplicate `#nc-settings-group`
        while the first worker's removal is still in flight, raising
        `DuplicateIds` and killing the app via `WorkerFailed`.
        """
        if self._settings_transitioning:
            return
        self._settings_transitioning = True
        try:
            if self._settings_expanded:
                await self._collapse_settings()
            else:
                await self._expand_settings()
        finally:
            self._settings_transitioning = False

    def _settings_row(self) -> _NotificationSettingsRow:
        """The settings disclosure row (always the last selectable row).

        Returns:
            The `_NotificationSettingsRow` at the bottom of the row list.
        """
        row = self._rows[-1]
        assert isinstance(row, _NotificationSettingsRow)  # noqa: S101
        return row

    async def _await_preload(self) -> set[str] | None:
        """Wait for the mount preload to publish the suppressed keys.

        Bounded so a stalled config read (an unresponsive network home
        directory, a held lock) cannot block the expand forever. On timeout
        or a failed read the section stays collapsed rather than rendering
        every warning as enabled, which is the unsafe direction to lie in.

        Returns:
            The preloaded suppressed keys, or `None` when the read timed out
            or failed and the expand must not go on.
        """
        try:
            async with asyncio.timeout(_PRELOAD_TIMEOUT_SECONDS):
                await self._settings_preloaded.wait()
        except TimeoutError:
            logger.warning("Notification settings preload did not complete in time")
        if self._suppressed is not None:
            return self._suppressed
        # The preload released the event without keys, so it raised or was
        # cancelled; `_load_suppressed` handles its own errors and returns a
        # set, so reaching here means an unexpected failure.
        if self.is_mounted:
            self.app.notify(
                "Could not read notification preferences. Reopen to retry.",
                severity="warning",
                timeout=6,
                markup=False,
            )
        return None

    async def _expand_settings(self) -> None:
        """Mount the warning checkboxes under the settings row.

        Uses the suppressed keys supplied at construction, or the ones
        `on_mount` preloaded. Reaching the wait below means the preload has
        not stored them yet, so that path always blocks; it is bounded so a
        stalled config read cannot wedge the row. A `reload()` that rebuilt
        the row list while the section was open calls this to remount the
        checkboxes, so an already-`_settings_expanded` state only skips a
        *mounted* section.
        """
        if self._settings_expanded and self.query("#nc-settings-group"):
            return
        # Drop a stale group that a `reload()` rebuild or an interrupted
        # collapse left mounted, so the mount below cannot raise
        # `DuplicateIds` on `#nc-settings-group`.
        for stale in self.query("#nc-settings-group"):
            await stale.remove()
        suppressed = self._suppressed
        if suppressed is None:
            suppressed = await self._await_preload()
            if suppressed is None:
                return
            # A rapid Esc while the config read was in flight already
            # dismissed the screen; expanding a dead screen would raise.
            if not self.is_mounted:
                return

        scroll = self.query_one(VerticalScroll)
        group = _NotificationSettingsGroup(id="nc-settings-group")
        # Batch so the display timer cannot paint between the mount and the
        # focus handoff; an unbatched timer would otherwise show the pane
        # open with nothing focused (and the footer hint still on its
        # collapsed verbs) for a frame before the focused checkbox lands.
        with self.app.batch_update():
            await scroll.mount(group)
            checkboxes = [
                Checkbox(
                    label,
                    value=key not in suppressed,
                    id=f"ns-{key}",
                )
                for key, label in WARNING_TOGGLES
            ]
            await group.mount(*checkboxes)

            self._settings_expanded = True
            self._settings_row().set_expanded(True)
            self._refresh_help()
            self._set_selected(len(self._rows) - 1)
            if checkboxes:
                # `Widget.focus()` defers via `app.call_later`.
                # `Screen.set_focus` applies in this same turn, so the pane
                # cannot paint open with nothing focused.
                self.set_focus(checkboxes[0])
            self._settings_row().scroll_visible()

    async def _collapse_settings(self) -> None:
        """Unmount the warning checkboxes and return focus to the row."""
        if not self._settings_expanded:
            return
        self._settings_expanded = False
        for group in self.query("#nc-settings-group"):
            await group.remove()
        self._settings_row().set_expanded(False)
        self.focus()
        self._rows[self._selected].scroll_visible()
        self._refresh_help()

    async def _load_suppressed(self) -> set[str]:
        """Read suppressed warning keys from config off the event loop.

        Falls back to empty (all warnings shown) with a warning toast. Note
        that `is_warning_suppressed` already swallows an unreadable or
        malformed config and returns `False` per key, so the ordinary config
        failure defaults silently and never reaches the handler here; this
        catches the rest.

        Returns:
            The set of suppressed warning keys from `config.toml`.
        """
        suppressed: set[str] = set()
        try:
            # Imported inside the try: a failure here must degrade to
            # defaults like any other read failure, not escape the worker.
            from deepagents_code.model_config import is_warning_suppressed

            for key, _ in WARNING_TOGGLES:
                if await asyncio.to_thread(is_warning_suppressed, key):
                    suppressed.add(key)
        except Exception:
            logger.warning("Failed to read notification settings", exc_info=True)
            # Guarded: a raise from the recovery path would replace the
            # handled error and escape as an unhandled one.
            with contextlib.suppress(Exception):
                self.app.notify(
                    "Could not read notification preferences. Showing defaults.",
                    severity="warning",
                    timeout=6,
                    markup=False,
                )
            return set()
        return suppressed

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Persist warning suppression toggle to config.toml on change."""
        event.stop()
        checkbox_id = event.checkbox.id
        if not checkbox_id or not checkbox_id.startswith("ns-"):
            return
        key = checkbox_id.removeprefix("ns-")
        enabled = event.value

        async def _persist() -> None:
            from deepagents_code.model_config import (
                suppress_warning,
                unsuppress_warning,
            )

            try:
                if enabled:
                    ok = await asyncio.to_thread(unsuppress_warning, key)
                else:
                    ok = await asyncio.to_thread(suppress_warning, key)
            except Exception:
                logger.warning(
                    "Failed to persist notification setting for %r",
                    key,
                    exc_info=True,
                )
                ok = False
            if not ok:
                # Roll the box back to what is actually on disk. Leaving it
                # showing the requested state would claim a warning is armed
                # when it is still suppressed — the unsafe direction to lie in.
                # `prevent` keeps the rollback from re-entering this handler.
                with event.checkbox.prevent(Checkbox.Changed):
                    event.checkbox.value = not enabled
                self.app.notify(
                    "Could not save notification preference. "
                    "Check file permissions for "
                    f"{PATHS.display(PATHS.profile.config_file)}.",
                    severity="warning",
                    timeout=6,
                    markup=False,
                )
            if self._suppressed is not None:
                if event.checkbox.value:
                    self._suppressed.discard(key)
                else:
                    self._suppressed.add(key)

        self.call_later(_persist)

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

    @staticmethod
    def _preload_follow_up(entry: PendingNotification) -> None:
        """Load deferred follow-up UI before its detail screen becomes active."""
        if any(action.action_id == ActionId.ENTER_API_KEY for action in entry.actions):
            import_module("deepagents_code.tui.widgets.auth")

    def _drill_into(self, entry: PendingNotification) -> None:
        """Push a detail modal for *entry*.

        Guarded against reentry so a rapid double-activation (e.g.
        keyboard repeat) does not stack two detail modals.

        Args:
            entry: The notification to drill into.
        """
        if self._drilling:
            return
        self._preload_follow_up(entry)
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
        back to settings. Warning preferences are refreshed from config so an
        in-place suppression is reflected if the user expands settings next.
        An expanded settings section is remounted so the refresh never silently
        collapses it.

        Args:
            notifications: Current pending entries to display.
        """
        self._suppressed = await self._load_suppressed()

        prev_key: str | None = None
        settings_selected = False
        if self._rows and 0 <= self._selected < len(self._rows):
            selected_row = self._rows[self._selected]
            if isinstance(selected_row, _NotificationSettingsRow):
                settings_selected = True
            else:
                prev_key = selected_row.notification.key
        was_expanded = self._settings_expanded

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

        if was_expanded:
            # The rebuild above dropped the settings group with the old rows;
            # remount the checkboxes so the section stays open.
            await self._expand_settings()

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
