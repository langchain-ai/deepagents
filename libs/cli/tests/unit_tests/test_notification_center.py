"""Tests for `NotificationCenterScreen` and its drill-in flow."""

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
    NotificationSuppressRequested,
    _NotificationRow,
)
from deepagents_cli.widgets.notification_detail import NotificationDetailScreen
from deepagents_cli.widgets.update_available import UpdateAvailableScreen


def _dep_entry(key: str = "dep:ripgrep") -> PendingNotification:
    return PendingNotification(
        key=key,
        title="ripgrep is not installed",
        body="Install with: brew install ripgrep",
        actions=(
            NotificationAction(
                ActionId.COPY_INSTALL, "Copy install command", primary=True
            ),
            NotificationAction(ActionId.SUPPRESS, "Don't show notification again"),
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
    """Drill-in behavior tests for the list-of-notifications modal."""

    async def test_renders_one_row_per_notification(self) -> None:
        """Each pending entry shows up as a single `_NotificationRow`."""
        app = App()
        screen = NotificationCenterScreen([_dep_entry(), _update_entry()])
        async with app.run_test() as pilot:
            app.push_screen(screen)
            await pilot.pause()
            rows = list(screen.query(_NotificationRow))
            assert [r.notification.key for r in rows] == [
                "dep:ripgrep",
                "update:available",
            ]

    async def test_widget_ids_are_collision_free_across_duplicate_keys(self) -> None:
        """Enumerated widget ids survive keys that would sanitize identically."""
        entry_a = _dep_entry(key="dep:foo")
        entry_b = _dep_entry(key="dep-foo")
        app = App()
        screen = NotificationCenterScreen([entry_a, entry_b])
        async with app.run_test() as pilot:
            app.push_screen(screen)
            await pilot.pause()
            widget_ids = [r.id for r in screen.query(_NotificationRow)]
            assert len(widget_ids) == len(set(widget_ids))

    async def test_enter_drills_into_missing_dep_detail(self) -> None:
        """Enter on a missing-dep row pushes `NotificationDetailScreen`."""
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(NotificationCenterScreen([_dep_entry()]))
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, NotificationDetailScreen)

    async def test_enter_drills_into_update_available_screen(self) -> None:
        """Enter on an update row pushes the dedicated `UpdateAvailableScreen`."""
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(NotificationCenterScreen([_update_entry()]))
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, UpdateAvailableScreen)

    async def test_detail_action_dismisses_center_with_result(self) -> None:
        """Selecting an action in the detail dismisses the center with it."""
        results: list[NotificationActionResult | None] = []

        app = App()
        async with app.run_test() as pilot:

            def on_result(result: NotificationActionResult | None) -> None:
                results.append(result)

            app.push_screen(NotificationCenterScreen([_dep_entry()]), on_result)
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            # The detail opens with the primary (COPY_INSTALL) selected.
            await pilot.press("enter")
            await pilot.pause()

        assert results == [
            NotificationActionResult("dep:ripgrep", ActionId.COPY_INSTALL),
        ]

    async def test_suppress_keeps_center_open_and_posts_message(self) -> None:
        """SUPPRESS from the detail posts a message and leaves the center up."""
        messages: list[NotificationSuppressRequested] = []

        class _App(App):
            def on_notification_suppress_requested(
                self, message: NotificationSuppressRequested
            ) -> None:
                messages.append(message)

        app = _App()
        screen = NotificationCenterScreen([_dep_entry(), _update_entry()])
        async with app.run_test() as pilot:
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("enter")  # drill into the first (dep) entry
            await pilot.pause()
            assert isinstance(app.screen, NotificationDetailScreen)
            # Primary is COPY_INSTALL; SUPPRESS is second in _dep_entry.
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            # Center is still the active screen; no dismissal fired.
            assert isinstance(app.screen, NotificationCenterScreen)

        assert [m.key for m in messages] == ["dep:ripgrep"]

    async def test_reload_rebuilds_rows_and_preserves_selection_by_key(self) -> None:
        """`reload` re-renders the list and keeps the cursor on the same entry."""
        dep = _dep_entry()
        update = _update_entry()
        app = App()
        screen = NotificationCenterScreen([dep, update])
        async with app.run_test() as pilot:
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("down")  # select row 1 (update)
            await pilot.pause()
            assert screen._selected == 1

            extra = PendingNotification(
                key="dep:tavily",
                title="tavily missing",
                body="",
                actions=(NotificationAction(ActionId.SUPPRESS, "Don't show"),),
                payload=MissingDepPayload(tool="tavily", url="https://tavily.com"),
            )
            await screen.reload([extra, update])
            await pilot.pause()

            keys = [r.notification.key for r in screen.query(_NotificationRow)]
            assert keys == ["dep:tavily", "update:available"]
            # Cursor follows the previously-selected key into its new position.
            assert screen._selected == 1

    async def test_reload_clamps_cursor_when_selected_entry_is_gone(self) -> None:
        """`reload` clamps to the last row when the selected key was removed."""
        dep = _dep_entry()
        update = _update_entry()
        app = App()
        screen = NotificationCenterScreen([dep, update])
        async with app.run_test() as pilot:
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("down")  # cursor on update (index 1)
            await pilot.pause()

            await screen.reload([dep])  # update removed
            await pilot.pause()

            assert [r.notification.key for r in screen.query(_NotificationRow)] == [
                "dep:ripgrep"
            ]
            assert screen._selected == 0

    async def test_reload_with_empty_list_dismisses_center(self) -> None:
        """`reload([])` closes the center with None."""
        results: list[NotificationActionResult | None] = []
        app = App()

        def on_result(result: NotificationActionResult | None) -> None:
            results.append(result)

        screen = NotificationCenterScreen([_dep_entry()])
        async with app.run_test() as pilot:
            app.push_screen(screen, on_result)
            await pilot.pause()
            await screen.reload([])
            await pilot.pause()

        assert results == [None]

    async def test_detail_esc_returns_to_center(self) -> None:
        """Esc in the detail modal keeps the notification center open."""
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(NotificationCenterScreen([_dep_entry()]))
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, NotificationDetailScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, NotificationCenterScreen)

    @pytest.mark.parametrize("key", ["down", "j", "tab"])
    async def test_down_or_j_or_tab_moves_cursor_to_next_row(self, key: str) -> None:
        """Navigating down highlights the second notification."""
        app = App()
        screen = NotificationCenterScreen([_dep_entry(), _update_entry()])
        async with app.run_test() as pilot:
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press(key)
            await pilot.pause()
            assert screen._selected == 1

    @pytest.mark.parametrize("key", ["up", "k"])
    async def test_up_or_k_wraps_to_last_row(self, key: str) -> None:
        """Navigating up from row 0 wraps to the last notification."""
        app = App()
        screen = NotificationCenterScreen([_dep_entry(), _update_entry()])
        async with app.run_test() as pilot:
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press(key)
            await pilot.pause()
            assert screen._selected == 1

    async def test_escape_dismisses_with_none(self) -> None:
        """Esc on the center (no detail open) returns `None`."""
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

    async def test_click_on_row_drills_in(self) -> None:
        """Clicking a row drills into its detail modal."""
        app = App()
        screen = NotificationCenterScreen([_dep_entry(), _update_entry()])
        async with app.run_test() as pilot:
            app.push_screen(screen)
            await pilot.pause()
            rows = list(screen.query(_NotificationRow))
            assert len(rows) == 2
            await pilot.click(rows[1])
            await pilot.pause()
            assert isinstance(app.screen, UpdateAvailableScreen)

    async def test_detail_screen_dispatch_maps_update_payload(self) -> None:
        """_detail_screen_for returns UpdateAvailableScreen for update entries."""
        screen = NotificationCenterScreen._detail_screen_for(_update_entry())
        assert isinstance(screen, UpdateAvailableScreen)

    async def test_detail_screen_dispatch_maps_missing_dep_payload(self) -> None:
        """_detail_screen_for returns NotificationDetailScreen for dep entries."""
        screen = NotificationCenterScreen._detail_screen_for(_dep_entry())
        assert isinstance(screen, NotificationDetailScreen)

    async def test_rapid_double_activation_only_pushes_one_detail(self) -> None:
        """Reentry guard prevents stacking two detail modals on keyboard repeat."""
        app = App()
        screen = NotificationCenterScreen([_dep_entry()])
        async with app.run_test() as pilot:
            app.push_screen(screen)
            await pilot.pause()
            # Drive _drill_into twice synchronously (no pause between)
            # — the second call must be dropped by the `_drilling` guard.
            screen.action_activate()
            screen.action_activate()
            await pilot.pause()

            detail_screens = [
                s for s in app._screen_stack if isinstance(s, NotificationDetailScreen)
            ]
            assert len(detail_screens) == 1
