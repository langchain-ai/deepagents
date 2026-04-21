"""Unit tests for `NotificationRegistry` and payload types."""

from __future__ import annotations

import logging

import pytest

from deepagents_cli.notifications import (
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
        payload=UpdateAvailablePayload(latest=latest, upgrade_cmd="pip install"),
    )


class TestPendingNotificationInvariants:
    """Invariants enforced by `PendingNotification.__post_init__`."""

    def test_empty_key_raises(self) -> None:
        with pytest.raises(ValueError, match="key must be non-empty"):
            PendingNotification(
                key="",
                title="t",
                body="b",
                actions=(NotificationAction(ActionId.SUPPRESS, "x"),),
                payload=MissingDepPayload(tool="x"),
            )

    def test_empty_actions_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one action"):
            PendingNotification(
                key="k",
                title="t",
                body="b",
                actions=(),
                payload=MissingDepPayload(tool="x"),
            )

    def test_multiple_primary_actions_raises(self) -> None:
        with pytest.raises(ValueError, match="primary actions"):
            PendingNotification(
                key="k",
                title="t",
                body="b",
                actions=(
                    NotificationAction(ActionId.INSTALL, "a", primary=True),
                    NotificationAction(ActionId.SKIP_ONCE, "b", primary=True),
                ),
                payload=UpdateAvailablePayload(latest="1", upgrade_cmd="c"),
            )

    def test_pending_notification_is_frozen(self) -> None:
        """Public invariants: the dataclass is immutable after construction."""
        from dataclasses import FrozenInstanceError

        entry = _dep_entry()
        with pytest.raises(FrozenInstanceError):
            entry.key = "other"  # type: ignore[misc]


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

    def test_add_with_same_key_replaces_entry_and_drops_toast_binding(self) -> None:
        reg = NotificationRegistry()
        reg.add(_dep_entry("dep:ripgrep"))
        reg.bind_toast("dep:ripgrep", "toast-1")
        reg.add(_dep_entry("dep:ripgrep"))

        assert len(reg) == 1
        assert reg.key_for_toast("toast-1") is None
        assert reg.toast_identity_for("dep:ripgrep") is None

    def test_remove_returns_entry_and_clears_toast_index(self) -> None:
        reg = NotificationRegistry()
        entry = _dep_entry("dep:ripgrep")
        reg.add(entry)
        reg.bind_toast("dep:ripgrep", "toast-1")

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
        assert reg.toast_identity_for("update:available") == "toast-42"
        assert reg.is_actionable_toast("toast-other") is False

    def test_bind_toast_replaces_previous_identity(self) -> None:
        reg = NotificationRegistry()
        reg.add(_update_entry())
        reg.bind_toast("update:available", "toast-1")
        reg.bind_toast("update:available", "toast-2")

        assert reg.key_for_toast("toast-1") is None
        assert reg.key_for_toast("toast-2") == "update:available"
        assert reg.toast_identity_for("update:available") == "toast-2"

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
        reg.add(_dep_entry("dep:ripgrep"))
        reg.add(_dep_entry("dep:tavily", tool="tavily"))
        reg.bind_toast("dep:ripgrep", "toast-1")
        reg.bind_toast("dep:tavily", "toast-2")

        reg.clear()
        assert len(reg) == 0
        assert reg.key_for_toast("toast-1") is None
        assert reg.key_for_toast("toast-2") is None
        assert reg.toast_identity_for("dep:ripgrep") is None

    def test_empty_registry_is_falsy(self) -> None:
        reg = NotificationRegistry()
        assert bool(reg) is False
        assert reg.list_all() == []
