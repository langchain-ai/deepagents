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
    """Clear the notified marker so the update modal re-opens next launch."""

    SKIP_VERSION = "skip_version"
    """Mark this version as notified; silence until a newer version ships."""


@dataclass(frozen=True)
class NotificationAction:
    """One button/action row in the notification modal."""

    action_id: ActionId
    label: str
    primary: bool = False
    """Whether to render this action in bold.

    `UpdateAvailableScreen` also uses this flag to place the initial
    cursor on the primary row; `NotificationCenterScreen` always starts
    on the first row regardless.
    """


@dataclass(frozen=True)
class MissingDepPayload:
    """Typed payload for a missing-dependency notification."""

    tool: str
    """Name of the missing tool (e.g. `"ripgrep"`, `"tavily"`)."""

    install_command: str | None = None
    """Shell command that installs the tool, when one is known."""

    url: str | None = None
    """Install guide or sign-up URL, used when no direct command exists."""


@dataclass(frozen=True)
class UpdateAvailablePayload:
    """Typed payload for an update-available notification."""

    latest: str
    """PyPI version string the user is being prompted to install."""

    upgrade_cmd: str
    """Shell command that upgrades to `latest`."""


Payload = MissingDepPayload | UpdateAvailablePayload


@dataclass(frozen=True)
class PendingNotification:
    """A single notice waiting for user action.

    Immutable value object: the registry owns the
    key-to-toast-identity binding (see `NotificationRegistry`) so
    external callers cannot corrupt click-routing indices by mutating
    notifications after construction.
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

    def __post_init__(self) -> None:
        """Enforce basic invariants at construction time.

        Raises:
            ValueError: If `key` is empty, `actions` is empty, or more
                than one action is marked `primary=True`.
        """
        if not self.key:
            msg = "PendingNotification.key must be non-empty"
            raise ValueError(msg)
        if not self.actions:
            msg = f"PendingNotification {self.key!r} must declare at least one action"
            raise ValueError(msg)
        primaries = sum(1 for a in self.actions if a.primary)
        if primaries > 1:
            msg = (
                f"PendingNotification {self.key!r} has {primaries} primary actions; "
                "at most one is allowed"
            )
            raise ValueError(msg)


class NotificationRegistry:
    """In-memory store of pending notifications.

    Instance-scoped (one per app) so test apps don't pollute each other.
    Owns the bidirectional key-to-toast-identity binding so callers
    cannot accidentally desynchronize the click-routing indices.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._entries: dict[str, PendingNotification] = {}
        self._key_to_toast: dict[str, str] = {}
        self._toast_to_key: dict[str, str] = {}

    def add(self, notification: PendingNotification) -> None:
        """Register a new notification or replace an existing one with the same key.

        Replacing is intentional: re-registering with the same key
        refreshes the entry rather than stacking duplicates. Any
        previously bound toast identity is dropped, since the old toast
        has been dismissed or replaced by the caller.

        Args:
            notification: Entry to add or replace.
        """
        old_identity = self._key_to_toast.pop(notification.key, None)
        if old_identity is not None:
            self._toast_to_key.pop(old_identity, None)
        self._entries[notification.key] = notification

    def remove(self, key: str) -> PendingNotification | None:
        """Remove a notification by key.

        Args:
            key: Registry key of the entry to remove.

        Returns:
            The removed entry, or `None` when *key* was not registered.
        """
        entry = self._entries.pop(key, None)
        identity = self._key_to_toast.pop(key, None)
        if identity is not None:
            self._toast_to_key.pop(identity, None)
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
        if key not in self._entries:
            logger.warning("bind_toast called for unknown key %r; ignoring", key)
            return
        prev = self._key_to_toast.get(key)
        if prev is not None:
            self._toast_to_key.pop(prev, None)
        self._key_to_toast[key] = toast_identity
        self._toast_to_key[toast_identity] = key

    def toast_identity_for(self, key: str) -> str | None:
        """Return the toast identity bound to *key*, or `None`."""
        return self._key_to_toast.get(key)

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
        self._key_to_toast.clear()
        self._toast_to_key.clear()
