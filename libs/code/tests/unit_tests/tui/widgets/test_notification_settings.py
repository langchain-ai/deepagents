"""Tests for NotificationSettingsScreen."""

from __future__ import annotations

from textual.app import App
from textual.widgets import Checkbox

from deepagents_code.approval_mode import YOLO_WARNING_KEY
from deepagents_code.model_config import is_warning_suppressed
from deepagents_code.tui.widgets.notification_settings import (
    WARNING_TOGGLES,
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


async def test_yolo_warning_is_toggleable() -> None:
    """Unchecking the YOLO row persists the suppression key to `config.toml`."""
    assert any(key == YOLO_WARNING_KEY for key, _ in WARNING_TOGGLES)

    app = _NotificationSettingsHost()
    async with app.run_test() as pilot:
        await app.push_screen(NotificationSettingsScreen(suppressed=set()))
        await pilot.pause()

        checkbox = app.screen.query_one(f"#ns-{YOLO_WARNING_KEY}", Checkbox)
        assert checkbox.value is True
        checkbox.value = False
        await pilot.pause()
        await pilot.pause()

        assert is_warning_suppressed(YOLO_WARNING_KEY) is True
