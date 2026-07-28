"""Tests for NotificationSettingsScreen."""

from __future__ import annotations

from unittest.mock import patch

from textual.app import App
from textual.widgets import Checkbox, Static

from deepagents_code.approval_mode import YOLO_WARNING_KEY
from deepagents_code.model_config import is_warning_suppressed, suppress_warning
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


async def test_enter_toggles_focused_warning_without_closing() -> None:
    """Enter must toggle the focused warning, matching the footer hint.

    Textual's `ToggleButton` binds `enter,space` to its toggle action, so Enter
    is a documented second toggle key here. A screen-level `enter` binding
    would silently steal it, so assert the real keypress path.
    """
    app = _NotificationSettingsHost()
    async with app.run_test() as pilot:
        screen = NotificationSettingsScreen(suppressed=set())
        await app.push_screen(screen)
        await pilot.pause()
        focused = screen.query(Checkbox).first()
        assert app.focused is focused
        assert focused.value is True

        await pilot.press("enter")
        await pilot.pause()

        assert focused.value is False
        assert app.screen is screen


async def test_help_footer_documents_both_toggle_keys() -> None:
    """The footer advertises Enter alongside Space so the hint is complete."""
    app = _NotificationSettingsHost()
    async with app.run_test() as pilot:
        screen = NotificationSettingsScreen(suppressed=set())
        await app.push_screen(screen)
        await pilot.pause()

        help_text = str(screen.query_one(".ns-help", Static).content)

        assert "Space/Enter toggle" in help_text
        assert "Esc close" in help_text


async def test_yolo_warning_is_toggleable() -> None:
    """Toggling the YOLO row round-trips the suppression key in `config.toml`.

    The re-check direction matters as much as the mute: a user who silences
    the toast needs a working way to get it back.
    """
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

        checkbox.value = True
        await pilot.pause()
        await pilot.pause()

        assert is_warning_suppressed(YOLO_WARNING_KEY) is False


async def test_suppressed_key_renders_unchecked() -> None:
    """A key already in `[warnings].suppress` renders its row unchecked."""
    app = _NotificationSettingsHost()
    async with app.run_test() as pilot:
        await app.push_screen(NotificationSettingsScreen(suppressed={YOLO_WARNING_KEY}))
        await pilot.pause()

        checkbox = app.screen.query_one(f"#ns-{YOLO_WARNING_KEY}", Checkbox)
        assert checkbox.value is False


async def test_failed_unsuppress_reverts_the_checkbox() -> None:
    """A failed write must not leave the box claiming the warning is armed.

    Showing checked while the key is still suppressed is the unsafe direction
    to diverge in: the user believes they re-enabled a safety warning that
    will never fire.
    """
    assert suppress_warning(YOLO_WARNING_KEY) is True

    app = _NotificationSettingsHost()
    async with app.run_test() as pilot:
        await app.push_screen(NotificationSettingsScreen(suppressed={YOLO_WARNING_KEY}))
        await pilot.pause()

        checkbox = app.screen.query_one(f"#ns-{YOLO_WARNING_KEY}", Checkbox)
        assert checkbox.value is False

        with patch(
            "deepagents_code.model_config.unsuppress_warning",
            return_value=False,
        ):
            checkbox.value = True
            await pilot.pause()
            await pilot.pause()

        assert checkbox.value is False
        assert is_warning_suppressed(YOLO_WARNING_KEY) is True
