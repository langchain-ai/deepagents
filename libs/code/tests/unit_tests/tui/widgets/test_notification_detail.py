"""Tests for `NotificationDetailScreen`."""

from __future__ import annotations

import pytest
from textual.app import App

from deepagents_code.notifications import (
    ActionId,
    MissingDepPayload,
    NotificationAction,
    PendingNotification,
)
from deepagents_code.tui.widgets.notification_detail import (
    NotificationDetailScreen,
    _ActionOption,
)


def _dep_entry() -> PendingNotification:
    return PendingNotification(
        key="dep:ripgrep",
        title="ripgrep is not installed",
        body="Install with: brew install ripgrep",
        actions=(
            NotificationAction(
                ActionId.COPY_INSTALL, "Copy install command", primary=True
            ),
            NotificationAction(ActionId.OPEN_WEBSITE, "Open install guide"),
            NotificationAction(ActionId.SUPPRESS, "Don't show notification again"),
        ),
        payload=MissingDepPayload(
            tool="ripgrep",
            install_command="brew install ripgrep",
            url="https://example.com",
        ),
    )


class TestNotificationDetailScreen:
    """Drill-target behavior for non-update notifications."""

    async def test_escape_dismisses_with_none(self) -> None:
        """Esc returns `None` so the caller can return to the center."""
        results: list[ActionId | None] = []

        app = App()
        async with app.run_test() as pilot:

            def on_result(result: ActionId | None) -> None:
                results.append(result)

            app.push_screen(NotificationDetailScreen(_dep_entry()), on_result)
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

        assert results == [None]
