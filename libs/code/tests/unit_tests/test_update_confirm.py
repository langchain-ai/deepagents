"""Tests for the `/update` dependency-refresh confirmation modal."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Static

from deepagents_code.widgets.update_confirm import RefreshDependenciesConfirmScreen


class _RefreshConfirmTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("base")


class TestRefreshDependenciesConfirmScreen:
    """Behavior tests for `RefreshDependenciesConfirmScreen`."""

    async def test_enter_dismisses_with_true(self) -> None:
        """Pressing Enter confirms the refresh."""
        app = _RefreshConfirmTestApp()
        async with app.run_test() as pilot:
            outcomes: list[bool | None] = []
            app.push_screen(RefreshDependenciesConfirmScreen(), outcomes.append)
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert outcomes == [True]

    async def test_escape_dismisses_with_false(self) -> None:
        """Pressing Esc cancels (no implicit refresh)."""
        app = _RefreshConfirmTestApp()
        async with app.run_test() as pilot:
            outcomes: list[bool | None] = []
            app.push_screen(RefreshDependenciesConfirmScreen(), outcomes.append)
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert outcomes == [False]

    async def test_action_cancel_dismisses_with_false(self) -> None:
        """`action_cancel` cancels — the path taken by the app's Esc handler."""
        app = _RefreshConfirmTestApp()
        async with app.run_test() as pilot:
            outcomes: list[bool | None] = []
            screen = RefreshDependenciesConfirmScreen()
            app.push_screen(screen, outcomes.append)
            await pilot.pause()
            screen.action_cancel()
            await pilot.pause()
            assert outcomes == [False]
