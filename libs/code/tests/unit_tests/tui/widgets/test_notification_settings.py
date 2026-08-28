"""Tests for the notification center's inline settings section."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from textual.app import App
from textual.widgets import Checkbox, Static

from deepagents_code.approval_mode import YOLO_WARNING_KEY
from deepagents_code.config import get_glyphs
from deepagents_code.model_config import is_warning_suppressed, suppress_warning
from deepagents_code.tui.widgets import notification_center
from deepagents_code.tui.widgets.notification_center import NotificationCenterScreen
from deepagents_code.tui.widgets.notification_settings import WARNING_TOGGLES


class _NotificationSettingsHost(App[None]):
    """Minimal host app for mounting `NotificationCenterScreen` in tests."""


async def test_failed_preload_leaves_the_row_usable_not_wedged() -> None:
    """A raising preload must release the waiter instead of hanging expand.

    `_settings_preloaded` is set in a `finally`, so an expand that blocks on
    it returns rather than waiting forever with `_settings_transitioning`
    stuck. The section stays collapsed instead of rendering every warning as
    enabled, which would be the unsafe direction to lie in.
    """

    async def failing_load(  # noqa: RUF029  # awaited by the preload worker
        _screen: NotificationCenterScreen,
    ) -> set[str]:
        msg = "config unreadable"
        raise RuntimeError(msg)

    app = _NotificationSettingsHost()
    with patch.object(NotificationCenterScreen, "_load_suppressed", failing_load):
        async with app.run_test() as pilot:
            screen = NotificationCenterScreen([])
            await app.push_screen(screen)
            await screen._settings_preloaded.wait()

            await pilot.press("enter")
            await pilot.pause()

            assert not screen.settings_expanded
            assert not list(screen.query(Checkbox))
            # A second press must still be accepted, not blocked by a
            # transition flag left set by an abandoned expand.
            await pilot.press("enter")
            await pilot.pause()
            assert not screen.settings_expanded


async def test_stalled_preload_times_out_instead_of_blocking_expand() -> None:
    """A config read that never returns must not hang the expand forever.

    Simulates an unresponsive filesystem: the read is held open past the
    timeout, so the expand gives up and leaves the section collapsed rather
    than waiting on an event that may never arrive.
    """
    release = asyncio.Event()

    async def stalled_load(_screen: NotificationCenterScreen) -> set[str]:
        await release.wait()
        return set()

    app = _NotificationSettingsHost()
    with (
        patch.object(NotificationCenterScreen, "_load_suppressed", stalled_load),
        patch.object(notification_center, "_PRELOAD_TIMEOUT_SECONDS", 0.05),
    ):
        async with app.run_test() as pilot:
            screen = NotificationCenterScreen([])
            await app.push_screen(screen)
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()
            await asyncio.sleep(0.1)
            await pilot.pause()

            assert not screen.settings_expanded
            assert not list(screen.query(Checkbox))
            # The load-bearing assertion: an expand still parked on the
            # event would leave this set, wedging every later toggle. A
            # collapsed section alone does not distinguish "gave up" from
            # "still waiting".
            assert not screen._settings_transitioning
        release.set()


async def test_first_toggle_waits_for_settings_preload() -> None:
    """The first Enter is honored even while the config read is in flight."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_load(_screen: NotificationCenterScreen) -> set[str]:
        started.set()
        await release.wait()
        return {YOLO_WARNING_KEY}

    app = _NotificationSettingsHost()
    with patch.object(NotificationCenterScreen, "_load_suppressed", blocked_load):
        async with app.run_test() as pilot:
            screen = NotificationCenterScreen([])
            await app.push_screen(screen)
            await started.wait()

            await pilot.press("enter")
            await pilot.pause()
            assert not screen.settings_expanded

            release.set()
            await pilot.pause()
            await pilot.pause()

            assert screen.settings_expanded
            checkbox = screen.query_one(f"#ns-{YOLO_WARNING_KEY}", Checkbox)
            assert checkbox.value is False


async def test_reload_refreshes_preloaded_settings() -> None:
    """A reload picks up a suppression saved after the initial preload."""
    suppressed: set[str] = set()

    async def load(  # noqa: RUF029  # awaited by the production reload path
        _screen: NotificationCenterScreen,
    ) -> set[str]:
        return set(suppressed)

    app = _NotificationSettingsHost()
    with patch.object(NotificationCenterScreen, "_load_suppressed", load):
        async with app.run_test() as pilot:
            screen = NotificationCenterScreen([])
            await app.push_screen(screen)
            await screen._settings_preloaded.wait()
            suppressed.add(YOLO_WARNING_KEY)

            await screen.reload([])
            await pilot.press("enter")
            await pilot.pause()

            checkbox = screen.query_one(f"#ns-{YOLO_WARNING_KEY}", Checkbox)
            assert checkbox.value is False


async def test_notification_center_dims_underlying_content() -> None:
    """The modal must inherit the translucent `ModalScreen` backdrop.

    Like the selector modals, the notification hub should dim the content
    underneath rather than render a fully transparent overlay. The alpha is
    in (0, 1) only under a non-ansi theme, so pin `textual-dark`.
    """
    app = _NotificationSettingsHost()
    async with app.run_test() as pilot:
        app.theme = "textual-dark"
        await pilot.pause()
        await app.push_screen(NotificationCenterScreen([], suppressed=set()))
        await pilot.pause()
        assert 0 < app.screen.styles.background.a < 1


async def test_enter_toggles_focused_warning_without_collapsing() -> None:
    """Enter must toggle the focused warning, matching the footer hint.

    Textual's `ToggleButton` binds `enter,space` to its toggle action, so Enter
    is a documented second toggle key here. A screen-level `enter` binding
    would silently steal it, so assert the real keypress path.
    """
    app = _NotificationSettingsHost()
    async with app.run_test() as pilot:
        screen = NotificationCenterScreen([], suppressed=set())
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.press("enter")  # expand the settings section
        await pilot.pause()
        focused = screen.query(Checkbox).first()
        assert app.focused is focused
        assert focused.value is True

        await pilot.press("enter")
        await pilot.pause()

        assert focused.value is False
        assert app.screen is screen
        assert screen.settings_expanded


async def test_help_footer_documents_both_toggle_keys_when_expanded() -> None:
    """The footer advertises Enter alongside Space so the hint is complete."""
    app = _NotificationSettingsHost()
    async with app.run_test() as pilot:
        screen = NotificationCenterScreen([], suppressed=set())
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        help_text = str(screen.query_one(".nc-help", Static).content)

        assert "Tab/Shift+Tab navigate" in help_text
        assert "Space/Enter toggle" in help_text
        assert "Esc collapse" in help_text


async def test_settings_row_leading_glyph_marks_expanded_state() -> None:
    """The disclosure affordance lives in the leading glyph, not a suffix."""
    glyphs = get_glyphs()
    app = _NotificationSettingsHost()
    async with app.run_test() as pilot:
        screen = NotificationCenterScreen([], suppressed=set())
        await app.push_screen(screen)
        await pilot.pause()

        row = screen.query_one("#nc-settings", Static)
        assert str(row.content) == f"{glyphs.cursor} Notification settings"

        await pilot.press("enter")
        await pilot.pause()

        assert str(row.content) == f"{glyphs.disclosure_expanded} Notification settings"

        await pilot.press("escape")
        await pilot.pause()

        # Back to the cursor: a stale expanded glyph over a closed pane is
        # the affordance bug this row's rendering exists to avoid.
        assert str(row.content) == f"{glyphs.cursor} Notification settings"


def test_cold_cache_warning_is_listed() -> None:
    """The advisory spend gate can be disabled from `/notifications`."""
    assert any(key == "cold-cache" for key, _ in WARNING_TOGGLES)


async def test_yolo_warning_is_toggleable() -> None:
    """Toggling the YOLO row round-trips the suppression key in `config.toml`.

    The re-check direction matters as much as the mute: a user who silences
    the toast needs a working way to get it back.
    """
    assert any(key == YOLO_WARNING_KEY for key, _ in WARNING_TOGGLES)

    app = _NotificationSettingsHost()
    async with app.run_test() as pilot:
        screen = NotificationCenterScreen([], suppressed=set())
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        checkbox = screen.query_one(f"#ns-{YOLO_WARNING_KEY}", Checkbox)
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
        screen = NotificationCenterScreen([], suppressed={YOLO_WARNING_KEY})
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        checkbox = screen.query_one(f"#ns-{YOLO_WARNING_KEY}", Checkbox)
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
        screen = NotificationCenterScreen([], suppressed={YOLO_WARNING_KEY})
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        checkbox = screen.query_one(f"#ns-{YOLO_WARNING_KEY}", Checkbox)
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
