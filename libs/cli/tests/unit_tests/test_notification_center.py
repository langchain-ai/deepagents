"""Tests for `NotificationCenterScreen` and its action flow."""

from __future__ import annotations

import pytest
from textual.app import App

from deepagents_cli.notifications import (
    ActionId,
    MissingDepPayload,
    NotificationAction,
    PendingNotification,
    UpdateAvailablePayload,
)
from deepagents_cli.widgets.notification_center import (
    NotificationActionResult,
    NotificationCenterScreen,
    _ActionOption,
)


def _dep_entry(key: str = "dep:ripgrep") -> PendingNotification:
    return PendingNotification(
        key=key,
        title="ripgrep is not installed",
        body="Install with: brew install ripgrep",
        actions=(
            NotificationAction(
                ActionId.COPY_INSTALL, "Copy install command", primary=True
            ),
            NotificationAction(ActionId.SUPPRESS, "Don't show this again"),
        ),
        payload=MissingDepPayload(
            tool="ripgrep", install_command="brew install ripgrep"
        ),
    )


def _update_entry() -> PendingNotification:
    return PendingNotification(
        key="update:available",
        title="Update available: v2.0.0",
        body="v2.0.0 is available (current: v1.0.0).",
        actions=(
            NotificationAction(ActionId.INSTALL, "Install now", primary=True),
            NotificationAction(ActionId.SKIP_ONCE, "Remind me next launch"),
            NotificationAction(ActionId.SKIP_VERSION, "Skip this version"),
        ),
        payload=UpdateAvailablePayload(latest="2.0.0", upgrade_cmd="pip install"),
    )


class TestNotificationCenterScreen:
    """Focused modal-behavior tests."""

    async def test_flattens_actions_across_cards(self) -> None:
        screen = NotificationCenterScreen([_dep_entry(), _update_entry()])
        assert [row.action.action_id for row in screen._rows] == [
            ActionId.COPY_INSTALL,
            ActionId.SUPPRESS,
            ActionId.INSTALL,
            ActionId.SKIP_ONCE,
            ActionId.SKIP_VERSION,
        ]

    async def test_widget_ids_are_collision_free_across_duplicate_keys(self) -> None:
        """Enumerated widget ids survive keys that would sanitize identically."""
        entry_a = _dep_entry(key="dep:foo")
        entry_b = _dep_entry(key="dep-foo")
        screen = NotificationCenterScreen([entry_a, entry_b])

        widget_ids = [row.widget_id for row in screen._rows]
        assert len(widget_ids) == len(set(widget_ids))

    async def test_enter_dismisses_with_primary_action(self) -> None:
        results: list[NotificationActionResult | None] = []

        app = App()
        async with app.run_test() as pilot:

            def on_result(result: NotificationActionResult | None) -> None:
                results.append(result)

            app.push_screen(NotificationCenterScreen([_dep_entry()]), on_result)
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

        assert results == [
            NotificationActionResult("dep:ripgrep", ActionId.COPY_INSTALL),
        ]

    @pytest.mark.parametrize("key", ["down", "j"])
    async def test_down_or_j_then_enter_picks_second_action(self, key: str) -> None:
        results: list[NotificationActionResult | None] = []

        app = App()
        async with app.run_test() as pilot:

            def on_result(result: NotificationActionResult | None) -> None:
                results.append(result)

            app.push_screen(NotificationCenterScreen([_dep_entry()]), on_result)
            await pilot.pause()
            await pilot.press(key)
            await pilot.press("enter")
            await pilot.pause()

        assert results == [
            NotificationActionResult("dep:ripgrep", ActionId.SUPPRESS),
        ]

    @pytest.mark.parametrize("key", ["up", "k"])
    async def test_up_or_k_wraps_to_last(self, key: str) -> None:
        results: list[NotificationActionResult | None] = []

        app = App()
        async with app.run_test() as pilot:

            def on_result(result: NotificationActionResult | None) -> None:
                results.append(result)

            app.push_screen(NotificationCenterScreen([_dep_entry()]), on_result)
            await pilot.pause()
            await pilot.press(key)
            await pilot.press("enter")
            await pilot.pause()

        assert results == [
            NotificationActionResult("dep:ripgrep", ActionId.SUPPRESS),
        ]

    async def test_down_wraps_between_cards(self) -> None:
        results: list[NotificationActionResult | None] = []

        app = App()
        async with app.run_test() as pilot:

            def on_result(result: NotificationActionResult | None) -> None:
                results.append(result)

            app.push_screen(
                NotificationCenterScreen([_dep_entry(), _update_entry()]),
                on_result,
            )
            await pilot.pause()
            # copy_install -> suppress -> install
            await pilot.press("down")
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

        assert results == [
            NotificationActionResult("update:available", ActionId.INSTALL),
        ]

    async def test_escape_dismisses_with_none(self) -> None:
        results: list[NotificationActionResult | None] = []

        app = App()
        async with app.run_test() as pilot:

            def on_result(result: NotificationActionResult | None) -> None:
                results.append(result)

            app.push_screen(NotificationCenterScreen([_dep_entry()]), on_result)
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

        assert results == [None]

    async def test_click_on_action_option_dismisses(self) -> None:
        results: list[NotificationActionResult | None] = []

        app = App()
        async with app.run_test() as pilot:

            def on_result(result: NotificationActionResult | None) -> None:
                results.append(result)

            screen = NotificationCenterScreen([_dep_entry()])
            app.push_screen(screen, on_result)
            await pilot.pause()
            options = list(screen.query(_ActionOption))
            assert len(options) == 2
            await pilot.click(options[1])
            await pilot.pause()

        assert results == [
            NotificationActionResult("dep:ripgrep", ActionId.SUPPRESS),
        ]
