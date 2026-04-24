"""Tests for `UpdateAvailableScreen`."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from textual.app import App

from deepagents_cli.notifications import (
    ActionId,
    NotificationAction,
    PendingNotification,
    UpdateAvailablePayload,
)
from deepagents_cli.widgets.update_available import (
    UpdateAvailableScreen,
    _ChangelogOption,
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


class TestUpdateAvailableScreen:
    """Focused modal-behavior tests for the dedicated update modal."""

    async def test_enter_dismisses_with_primary_action(self) -> None:
        """Pressing enter on mount returns the primary (Install now) action."""
        results: list[ActionId | None] = []

        app = App()
        async with app.run_test() as pilot:

            def on_result(result: ActionId | None) -> None:
                results.append(result)

            app.push_screen(UpdateAvailableScreen(_update_entry()), on_result)
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

        assert results == [ActionId.INSTALL]

    @pytest.mark.parametrize("key", ["down", "j", "tab"])
    async def test_down_or_j_or_tab_then_enter_picks_second_action(
        self, key: str
    ) -> None:
        """down/j/tab advances from the primary row to SKIP_ONCE."""
        results: list[ActionId | None] = []

        app = App()
        async with app.run_test() as pilot:

            def on_result(result: ActionId | None) -> None:
                results.append(result)

            app.push_screen(UpdateAvailableScreen(_update_entry()), on_result)
            await pilot.pause()
            await pilot.press(key)
            await pilot.press("enter")
            await pilot.pause()

        assert results == [ActionId.SKIP_ONCE]

    @pytest.mark.parametrize("key", ["up", "k", "shift+tab"])
    async def test_up_or_k_or_shift_tab_selects_changelog(self, key: str) -> None:
        """From the primary row, backward nav lands on the changelog row."""
        screen = UpdateAvailableScreen(_update_entry())
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press(key)
            await pilot.pause()
            assert isinstance(screen._options[screen._selected], _ChangelogOption)

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

    async def test_primary_action_selected_by_default(self) -> None:
        """Mount places the cursor on the `primary=True` action, not on changelog."""
        screen = UpdateAvailableScreen(_update_entry())
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(screen)
            await pilot.pause()
            option = screen._options[screen._selected]
            assert not isinstance(option, _ChangelogOption)
            assert option.action.action_id == ActionId.INSTALL

    async def test_changelog_enter_opens_url_and_keeps_modal_open(self) -> None:
        """Activating the changelog row opens the URL without dismissing."""
        from deepagents_cli._version import CHANGELOG_URL

        screen = UpdateAvailableScreen(_update_entry())
        app = App()
        with patch("webbrowser.open", return_value=True) as mock_open:
            async with app.run_test() as pilot:
                app.push_screen(screen)
                await pilot.pause()
                await pilot.press("shift+tab")
                await pilot.press("enter")
                await pilot.pause()
                assert app.screen is screen

        mock_open.assert_called_once_with(CHANGELOG_URL)

    async def test_changelog_click_opens_url_and_keeps_modal_open(self) -> None:
        """Mouse-clicking the changelog row opens the URL without dismissing."""
        from deepagents_cli._version import CHANGELOG_URL

        screen = UpdateAvailableScreen(_update_entry())
        app = App()
        with patch("webbrowser.open", return_value=True) as mock_open:
            async with app.run_test() as pilot:
                app.push_screen(screen)
                await pilot.pause()
                await pilot.click("#ua-changelog")
                await pilot.pause()
                assert app.screen is screen

        mock_open.assert_called_once_with(CHANGELOG_URL)

    async def test_action_click_does_not_dismiss(self) -> None:
        """Mouse-clicking an action row is a no-op.

        Install / skip-once / skip-version are keyboard-only to avoid
        accidentally triggering irreversible actions.
        """
        results: list[ActionId | None] = []

        app = App()
        async with app.run_test() as pilot:

            def on_result(result: ActionId | None) -> None:
                results.append(result)

            app.push_screen(UpdateAvailableScreen(_update_entry()), on_result)
            await pilot.pause()
            await pilot.click("#ua-row-0")
            await pilot.pause()
            assert isinstance(app.screen, UpdateAvailableScreen)

        assert results == []

    async def test_tab_wraps_from_last_action_to_changelog(self) -> None:
        """Tab from `Skip this version` wraps to the changelog row."""
        screen = UpdateAvailableScreen(_update_entry())
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("tab")
            await pilot.press("tab")
            await pilot.press("tab")
            await pilot.pause()
            assert isinstance(screen._options[screen._selected], _ChangelogOption)
