"""Registry of pending actionable notifications.

Stores plain data for notices the user can act on from a dedicated
modal screen. The registry is deliberately UI-agnostic: UI routing
(toast click, keybinds) lives in the app layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class NotificationKind(StrEnum):
    """Category of actionable notification."""

    MISSING_DEP = "missing_dep"
    """A recommended optional dependency is not installed or configured."""

    UPDATE_AVAILABLE = "update_available"
    """A newer version of `deepagents-cli` is available on PyPI."""


class ActionId(StrEnum):
    """Stable identifiers for notification actions dispatched by the app."""

    SUPPRESS = "suppress"
    """Persist a suppression entry so the notice isn't re-raised."""

    COPY_INSTALL = "copy_install"
    """Copy the install command to the system clipboard."""

    OPEN_WEBSITE = "open_website"
    """Open the associated URL in the user's browser."""

    INSTALL = "install"
    """Run the upgrade command via `perform_upgrade`."""

    SKIP_ONCE = "skip_once"
    """Clear the notified marker so the update re-toasts next launch."""

    SKIP_VERSION = "skip_version"
    """Mark this version as notified; silence until a newer version ships."""


@dataclass(frozen=True)
class NotificationAction:
    """One button/action row in the notification modal."""

    action_id: ActionId
    label: str
    primary: bool = False
    """Whether this action is the default (Enter-bound) choice."""


@dataclass(frozen=True)
class MissingDepPayload:
    """Typed payload for a `MISSING_DEP` notification."""

    tool: str
    """Name of the missing tool (e.g. `"ripgrep"`, `"tavily"`)."""

    install_command: str | None = None
    """Shell command that installs the tool, when one is known."""

    url: str | None = None
    """Install guide or sign-up URL, used when no direct command exists."""


@dataclass(frozen=True)
class UpdateAvailablePayload:
    """Typed payload for an `UPDATE_AVAILABLE` notification."""

    latest: str
    """PyPI version string the user is being prompted to install."""

    upgrade_cmd: str
    """Shell command that upgrades to `latest`."""


Payload = MissingDepPayload | UpdateAvailablePayload


@dataclass
class PendingNotification:
    """A single notice waiting for user action.

    `kind` is derived from `payload`: adding a new notification category
    requires introducing a new payload dataclass, which in turn forces
    every dispatcher to handle it (or fail exhaustiveness checks).
    """

    key: str
    """Stable identifier used to dedupe and to remove the notice once handled."""

    title: str
    """One-line heading shown in the modal."""

    body: str
    """Longer description shown below the title.

    May contain install instructions, links, or version info.
    """

    actions: tuple[NotificationAction, ...]
    """Available actions, rendered as rows in the modal."""

    payload: Payload
    """Kind-specific typed data consumed by the action dispatcher."""

    toast_identity: str | None = None
    """The `Notification.identity` of the originating toast.

    Set by the app when the toast is posted; used to route toast clicks
    back to this entry.
    """

    @property
    def kind(self) -> NotificationKind:
        """Return the notification kind derived from `payload`."""
        if isinstance(self.payload, MissingDepPayload):
            return NotificationKind.MISSING_DEP
        return NotificationKind.UPDATE_AVAILABLE


class NotificationRegistry:
    """In-memory store of pending notifications.

    Instance-scoped (one per app) so test apps don't pollute each other.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._entries: dict[str, PendingNotification] = {}
        self._toast_to_key: dict[str, str] = {}

    def add(self, notification: PendingNotification) -> None:
        """Register a new notification or replace an existing one with the same key.

        Replacing is intentional: re-registering with the same key
        refreshes the entry rather than stacking duplicates.

        Args:
            notification: Entry to add or replace.
        """
        existing = self._entries.get(notification.key)
        if existing is not None and existing.toast_identity is not None:
            self._toast_to_key.pop(existing.toast_identity, None)
        self._entries[notification.key] = notification
        if notification.toast_identity is not None:
            self._toast_to_key[notification.toast_identity] = notification.key

    def remove(self, key: str) -> PendingNotification | None:
        """Remove a notification by key.

        Args:
            key: Registry key of the entry to remove.

        Returns:
            The removed entry, or `None` when *key* was not registered.
        """
        entry = self._entries.pop(key, None)
        if entry is not None and entry.toast_identity is not None:
            self._toast_to_key.pop(entry.toast_identity, None)
        return entry

    def get(self, key: str) -> PendingNotification | None:
        """Return the notification for *key*, or `None` when not registered."""
        return self._entries.get(key)

    def bind_toast(self, key: str, toast_identity: str) -> None:
        """Attach a Textual toast identity to an existing notification.

        Logs a warning when *key* is unknown — this only happens if a
        caller binds a toast without first `add`-ing the entry, which is
        a programming error.

        Args:
            key: Registry key of the entry.
            toast_identity: `Notification.identity` of the originating toast.
        """
        entry = self._entries.get(key)
        if entry is None:
            logger.warning("bind_toast called for unknown key %r; ignoring", key)
            return
        if entry.toast_identity is not None:
            self._toast_to_key.pop(entry.toast_identity, None)
        entry.toast_identity = toast_identity
        self._toast_to_key[toast_identity] = key

    def key_for_toast(self, toast_identity: str) -> str | None:
        """Return the registered key for *toast_identity*, or `None`."""
        return self._toast_to_key.get(toast_identity)

    def is_actionable_toast(self, toast_identity: str) -> bool:
        """Return whether a click on *toast_identity* should open the modal."""
        return toast_identity in self._toast_to_key

    def list_all(self) -> list[PendingNotification]:
        """Return all pending notifications in insertion order."""
        return list(self._entries.values())

    def __len__(self) -> int:
        """Return the number of pending notifications."""
        return len(self._entries)

    def __bool__(self) -> bool:
        """Return `True` when at least one notification is pending."""
        return bool(self._entries)

    def clear(self) -> None:
        """Remove all entries. Primarily useful for tests."""
        self._entries.clear()
        self._toast_to_key.clear()
