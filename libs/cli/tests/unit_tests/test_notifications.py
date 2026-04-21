"""Unit tests for `NotificationRegistry` and payload types."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deepagents_cli.notifications import (
    ActionId,
    MissingDepPayload,
    NotificationAction,
    NotificationKind,
    NotificationRegistry,
    PendingNotification,
    UpdateAvailablePayload,
)

if TYPE_CHECKING:
    import pytest


def _dep_entry(
    key: str = "dep:ripgrep",
    *,
    tool: str = "ripgrep",
    toast_identity: str | None = None,
) -> PendingNotification:
    return PendingNotification(
        key=key,
        title=f"{tool} missing",
        body=f"Install {tool}",
        actions=(NotificationAction(ActionId.SUPPRESS, "Don't show", primary=True),),
        payload=MissingDepPayload(tool=tool),
        toast_identity=toast_identity,
    )


def _update_entry(
    *,
    latest: str = "1.0.0",
    toast_identity: str | None = None,
) -> PendingNotification:
    return PendingNotification(
        key="update:available",
        title=f"Update available: v{latest}",
        body=f"v{latest} is available.",
        actions=(NotificationAction(ActionId.INSTALL, "Install now", primary=True),),
        payload=UpdateAvailablePayload(latest=latest, upgrade_cmd="pip install"),
        toast_identity=toast_identity,
    )


class TestDerivedKind:
    """`PendingNotification.kind` is derived from the payload type."""

    def test_missing_dep_payload_maps_to_missing_dep_kind(self) -> None:
        assert _dep_entry().kind is NotificationKind.MISSING_DEP

    def test_update_payload_maps_to_update_kind(self) -> None:
        assert _update_entry().kind is NotificationKind.UPDATE_AVAILABLE


class TestNotificationRegistry:
    """Tests for add / remove / toast-binding semantics."""

    def test_add_and_list_preserves_insertion_order(self) -> None:
        reg = NotificationRegistry()
        reg.add(_dep_entry("dep:ripgrep"))
        reg.add(_dep_entry("dep:tavily", tool="tavily"))
        reg.add(_update_entry())

        assert [e.key for e in reg.list_all()] == [
            "dep:ripgrep",
            "dep:tavily",
            "update:available",
        ]
        assert len(reg) == 3
        assert bool(reg) is True

    def test_add_with_same_key_replaces_entry(self) -> None:
        reg = NotificationRegistry()
        reg.add(_dep_entry("dep:ripgrep", toast_identity="toast-1"))
        reg.add(_dep_entry("dep:ripgrep", toast_identity="toast-2"))

        assert len(reg) == 1
        assert reg.key_for_toast("toast-1") is None
        assert reg.key_for_toast("toast-2") == "dep:ripgrep"

    def test_add_replacement_with_no_toast_clears_old_toast_index(self) -> None:
        reg = NotificationRegistry()
        reg.add(_dep_entry("dep:ripgrep", toast_identity="toast-1"))
        reg.add(_dep_entry("dep:ripgrep", toast_identity=None))

        assert reg.key_for_toast("toast-1") is None
        assert len(reg) == 1

    def test_remove_returns_entry_and_clears_toast_index(self) -> None:
        reg = NotificationRegistry()
        entry = _dep_entry("dep:ripgrep", toast_identity="toast-1")
        reg.add(entry)

        removed = reg.remove("dep:ripgrep")
        assert removed is entry
        assert reg.key_for_toast("toast-1") is None
        assert reg.get("dep:ripgrep") is None
        assert not reg

    def test_remove_unknown_key_returns_none(self) -> None:
        reg = NotificationRegistry()
        assert reg.remove("dep:missing") is None

    def test_bind_toast_routes_click_back_to_key(self) -> None:
        reg = NotificationRegistry()
        reg.add(_update_entry())
        reg.bind_toast("update:available", "toast-42")

        assert reg.is_actionable_toast("toast-42") is True
        assert reg.key_for_toast("toast-42") == "update:available"
        assert reg.is_actionable_toast("toast-other") is False

    def test_bind_toast_replaces_previous_identity(self) -> None:
        reg = NotificationRegistry()
        reg.add(_update_entry())
        reg.bind_toast("update:available", "toast-1")
        reg.bind_toast("update:available", "toast-2")

        assert reg.key_for_toast("toast-1") is None
        assert reg.key_for_toast("toast-2") == "update:available"

    def test_bind_toast_for_unknown_key_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        reg = NotificationRegistry()
        with caplog.at_level(logging.WARNING, logger="deepagents_cli.notifications"):
            reg.bind_toast("dep:missing", "toast-stray")
        assert reg.is_actionable_toast("toast-stray") is False
        assert any(
            "bind_toast called for unknown key" in record.message
            for record in caplog.records
        )

    def test_clear_removes_everything(self) -> None:
        reg = NotificationRegistry()
        reg.add(_dep_entry("dep:ripgrep", toast_identity="toast-1"))
        reg.add(_dep_entry("dep:tavily", tool="tavily", toast_identity="toast-2"))

        reg.clear()
        assert len(reg) == 0
        assert reg.key_for_toast("toast-1") is None
        assert reg.key_for_toast("toast-2") is None

    def test_empty_registry_is_falsy(self) -> None:
        reg = NotificationRegistry()
        assert bool(reg) is False
        assert reg.list_all() == []
