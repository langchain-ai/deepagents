"""Tests for `NotificationDetailScreen`."""

from __future__ import annotations

import pytest
from textual.app import App

from deepagents_cli.notifications import (
    ActionId,
    MissingDepPayload,
    NotificationAction,
    PendingNotification,
)
from deepagents_cli.widgets.notification_detail import (
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

    async def test_enter_dismisses_with_primary_action(self) -> None:
        """Enter on mount returns the primary action."""
        results: list[ActionId | None] = []

        app = App()
        async with app.run_test() as pilot:

            def on_result(result: ActionId | None) -> None:
                results.append(result)

            app.push_screen(NotificationDetailScreen(_dep_entry()), on_result)
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

        assert results == [ActionId.COPY_INSTALL]

    @pytest.mark.parametrize("key", ["down", "j", "tab"])
    async def test_down_or_j_or_tab_advances(self, key: str) -> None:
        """Forward nav from the primary lands on the second action."""
        results: list[ActionId | None] = []

        app = App()
        async with app.run_test() as pilot:

            def on_result(result: ActionId | None) -> None:
                results.append(result)

            app.push_screen(NotificationDetailScreen(_dep_entry()), on_result)
            await pilot.pause()
            await pilot.press(key)
            await pilot.press("enter")
            await pilot.pause()

        assert results == [ActionId.OPEN_WEBSITE]

    @pytest.mark.parametrize("key", ["up", "k", "shift+tab"])
    async def test_up_or_k_or_shift_tab_wraps(self, key: str) -> None:
        """Backward nav from the primary wraps to the last action."""
        results: list[ActionId | None] = []

        app = App()
        async with app.run_test() as pilot:

            def on_result(result: ActionId | None) -> None:
                results.append(result)

            app.push_screen(NotificationDetailScreen(_dep_entry()), on_result)
            await pilot.pause()
            await pilot.press(key)
            await pilot.press("enter")
            await pilot.pause()

        assert results == [ActionId.SUPPRESS]

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

    async def test_click_on_action_dismisses_with_that_action(self) -> None:
        """Clicking an action row dismisses with its id."""
        results: list[ActionId | None] = []

        app = App()
        async with app.run_test() as pilot:

            def on_result(result: ActionId | None) -> None:
                results.append(result)

            screen = NotificationDetailScreen(_dep_entry())
            app.push_screen(screen, on_result)
            await pilot.pause()
            options = list(screen.query(_ActionOption))
            assert len(options) == 3
            await pilot.click(options[2])
            await pilot.pause()

        assert results == [ActionId.SUPPRESS]
