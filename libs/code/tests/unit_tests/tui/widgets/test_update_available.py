"""Tests for `UpdateAvailableScreen`."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from textual.app import App

from deepagents_code.notifications import (
    ActionId,
    NotificationAction,
    PendingNotification,
    UpdateAvailablePayload,
)
from deepagents_code.tui.widgets.update_available import (
    UpdateAvailableScreen,
    _ChangelogOption,
)


def _update_entry() -> PendingNotification:
    return PendingNotification(
        key="update:available",
        title="Update available",
        body="v2.0.0 is available.\nCurrently installed: 1.0.0.",
        actions=(
            NotificationAction(ActionId.INSTALL, "Install now", primary=True),
            NotificationAction(ActionId.SKIP_ONCE, "Remind me next launch"),
            NotificationAction(ActionId.SKIP_VERSION, "Skip this version"),
        ),
        payload=UpdateAvailablePayload(
            latest="2.0.0", upgrade_cmd="uv tool upgrade deepagents-code"
        ),
    )


class TestUpdateAvailableScreen:
    """Focused modal-behavior tests for the dedicated update modal."""

    async def test_escape_dismisses_with_none(self) -> None:
        """Esc closes the modal without firing any action."""
        results: list[ActionId | None] = []

        app = App()
        async with app.run_test() as pilot:

            def on_result(result: ActionId | None) -> None:
                results.append(result)

            app.push_screen(UpdateAvailableScreen(_update_entry()), on_result)
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

        assert results == [None]
