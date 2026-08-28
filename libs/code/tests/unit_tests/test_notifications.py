"""Unit tests for `NotificationRegistry` and payload types."""

from __future__ import annotations

import logging

import pytest

from deepagents_code.notifications import (
    ActionId,
    MissingDepPayload,
    NotificationAction,
    NotificationRegistry,
    PendingNotification,
    UpdateAvailablePayload,
)


def _dep_entry(
    key: str = "dep:ripgrep",
    *,
    tool: str = "ripgrep",
) -> PendingNotification:
    return PendingNotification(
        key=key,
        title=f"{tool} missing",
        body=f"Install {tool}",
        actions=(NotificationAction(ActionId.SUPPRESS, "Don't show", primary=True),),
        payload=MissingDepPayload(tool=tool),
    )


def _update_entry(
    *,
    latest: str = "1.0.0",
) -> PendingNotification:
    return PendingNotification(
        key="update:available",
        title=f"Update available: v{latest}",
        body=f"v{latest} is available.",
        actions=(NotificationAction(ActionId.INSTALL, "Install now", primary=True),),
        payload=UpdateAvailablePayload(
            latest=latest, upgrade_cmd="uv tool upgrade deepagents-code"
        ),
    )


class TestPendingNotificationInvariants:
    """Invariants enforced by `PendingNotification.__post_init__`."""


class TestNotificationRegistry:
    """Tests for add / remove / toast-binding semantics."""
