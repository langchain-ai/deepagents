"""Tests for NotificationSettingsScreen."""

from __future__ import annotations

from textual.app import App

from deepagents_code.tui.widgets.notification_settings import (
    NotificationSettingsScreen,
)


class _NotificationSettingsHost(App[None]):
    """Minimal host app for mounting `NotificationSettingsScreen` in tests."""


async def test_notification_settings_dims_underlying_content() -> None:
    """The modal must inherit the translucent `ModalScreen` backdrop.

    Like the selector modals, the notification settings dialog should dim the
    content underneath rather than render a fully transparent overlay. The
    alpha is in (0, 1) only under a non-ansi theme, so pin `textual-dark`.
    """
    app = _NotificationSettingsHost()
    async with app.run_test() as pilot:
        app.theme = "textual-dark"
        await pilot.pause()
        await app.push_screen(NotificationSettingsScreen(suppressed=set()))
        await pilot.pause()
        assert 0 < app.screen.styles.background.a < 1
