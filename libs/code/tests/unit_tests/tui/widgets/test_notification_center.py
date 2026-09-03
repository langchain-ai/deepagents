"""Tests for `NotificationCenterScreen` and its drill-in flow."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Checkbox, Static

from deepagents_code.notifications import (
    ActionId,
    MissingDepPayload,
    NotificationAction,
    PendingNotification,
)
from deepagents_code.tui.widgets import notification_center
from deepagents_code.tui.widgets.notification_center import (
    NotificationActionRequested,
    NotificationActionResult,
    NotificationCenterScreen,
)
from deepagents_code.tui.widgets.notification_detail import NotificationDetailScreen


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


def _service_entry(key: str = "dep:tavily") -> PendingNotification:
    return PendingNotification(
        key=key,
        title="Web search disabled",
        body="No Tavily API key is set.",
        actions=(
            NotificationAction(ActionId.ENTER_API_KEY, "Enter API key", primary=True),
            NotificationAction(ActionId.OPEN_WEBSITE, "Open tavily.com"),
            NotificationAction(ActionId.SUPPRESS, "Don't show notification again"),
        ),
        payload=MissingDepPayload(tool="tavily", url="https://tavily.com"),
    )


class TestNotificationCenterScreen:
    """Drill-in behavior tests for the list-of-notifications modal."""

    async def test_esc_while_expanded_collapses_before_closing(self) -> None:
        """Esc collapses expanded settings first; a second Esc dismisses."""
        results: list[NotificationActionResult | None] = []
        app = App()
        screen = NotificationCenterScreen([], suppressed=set())
        async with app.run_test() as pilot:
            app.push_screen(screen, results.append)
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert screen.settings_expanded

            await pilot.press("escape")
            await pilot.pause()

            assert app.screen is screen
            assert not screen.settings_expanded
            assert not list(screen.query(Checkbox))
            assert results == []

            await pilot.press("escape")
            await pilot.pause()

        assert results == [None]

    async def test_settings_expand_collapse_round_trip_restores_help(self) -> None:
        """The footer hint tracks expansion so the Esc verb stays accurate."""
        app = App()
        screen = NotificationCenterScreen([], suppressed=set())
        async with app.run_test() as pilot:
            app.push_screen(screen)
            await pilot.pause()
            help_widget = screen.query_one(".nc-help", Static)
            assert "Esc close" in str(help_widget.content)

            await pilot.press("enter")
            await pilot.pause()
            assert "Esc collapse" in str(screen.query_one(".nc-help", Static).content)

            await pilot.press("escape")
            await pilot.pause()
            assert "Esc close" in str(screen.query_one(".nc-help", Static).content)

    async def test_rapid_double_esc_does_not_duplicate_settings_group(self) -> None:
        """Two Esc presses in one batch must not mount a second settings group.

        `run_worker(..., group="nc-settings")` is not exclusive by default,
        so without serialization the second Esc starts an expand while the
        first Esc's collapse is still awaiting `remove()` — mounting a
        duplicate `#nc-settings-group` raises `DuplicateIds` and kills the
        app via `WorkerFailed`.
        """
        app = App()
        screen = NotificationCenterScreen([], suppressed=set())
        async with app.run_test() as pilot:
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert screen.settings_expanded

            await pilot.press("escape", "escape")
            for _ in range(20):
                await pilot.pause()

            assert len(screen.query("#nc-settings-group")) <= 1

    async def test_enter_preloads_api_key_screen_before_detail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """API-key UI loads while the center still covers the base screen."""
        imports: list[tuple[str, object]] = []
        app = App()
        screen = NotificationCenterScreen([_service_entry()])

        def capture_import(name: str) -> None:
            imports.append((name, app.screen))

        monkeypatch.setattr(notification_center, "import_module", capture_import)
        async with app.run_test() as pilot:
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, NotificationDetailScreen)

        assert imports == [("deepagents_code.tui.widgets.auth", screen)]

    async def test_in_place_action_keeps_center_open_and_posts_message(self) -> None:
        """ENTER_API_KEY posts a request message and leaves the center up."""
        messages: list[NotificationActionRequested] = []

        class _App(App):
            def on_notification_action_requested(
                self, message: NotificationActionRequested
            ) -> None:
                messages.append(message)

        app = _App()
        screen = NotificationCenterScreen([_service_entry()])
        async with app.run_test() as pilot:
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("enter")  # drill into the tavily entry
            await pilot.pause()
            assert isinstance(app.screen, NotificationDetailScreen)
            # ENTER_API_KEY is the primary (first) action, selected on mount.
            await pilot.press("enter")
            await pilot.pause()

            # Center is still the active screen; no dismissal fired.
            assert isinstance(app.screen, NotificationCenterScreen)

        assert [(m.key, m.action_id) for m in messages] == [
            ("dep:tavily", ActionId.ENTER_API_KEY)
        ]

    def test_action_requested_rejects_non_in_place_action(self) -> None:
        """Constructing the message with a terminal action fails fast."""
        with pytest.raises(ValueError, match="not an in-place action"):
            NotificationActionRequested("dep:ripgrep", ActionId.INSTALL)

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
