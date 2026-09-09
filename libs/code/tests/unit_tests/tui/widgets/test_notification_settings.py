"""Tests for the notification center's inline settings section."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from textual.app import App
from textual.widgets import Checkbox, Static

from deepagents_code.approval_mode import YOLO_WARNING_KEY
from deepagents_code.config import get_glyphs
from deepagents_code.tui.widgets import notification_center
from deepagents_code.tui.widgets.notification_center import NotificationCenterScreen


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
