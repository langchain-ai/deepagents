"""Tests for launch initialization screens."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Input, Static

from deepagents_cli.widgets.launch_init import (
    LaunchCountdownScreen,
    LaunchNameScreen,
    LaunchWaitingScreen,
    _normalize_name,
)


class LaunchNameTestApp(App[None]):
    """Test app for `LaunchNameScreen`."""

    def __init__(self) -> None:
        super().__init__()
        self.result: str | None = None
        self.dismissed = False

    def compose(self) -> ComposeResult:
        """Compose a minimal host app."""
        yield Container(id="main")

    def show_name_screen(self) -> None:
        """Open the launch name screen."""

        def handle_result(result: str | None) -> None:
            self.result = result
            self.dismissed = True

        self.push_screen(LaunchNameScreen(), handle_result)

    def show_waiting_screen(self, *, ready_to_start: bool = False) -> None:
        """Open the launch waiting screen."""
        self.push_screen(LaunchWaitingScreen(ready_to_start=ready_to_start))

    def show_countdown_screen(self, seconds: int) -> LaunchCountdownScreen:
        """Open the launch countdown screen."""
        screen = LaunchCountdownScreen(seconds)
        self.push_screen(screen)
        return screen


class TestLaunchNameScreen:
    """Tests for launch name entry."""

    async def test_name_input_autofocuses(self) -> None:
        """The name field should be focused on mount so users can type immediately."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_name_screen()
            await pilot.pause()

            name_input = app.screen.query_one("#launch-name-input", Input)
            assert name_input.has_focus

    async def test_submit_returns_normalized_name(self) -> None:
        """Submitting a name should dismiss with the trimmed, title-cased value."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_name_screen()
            await pilot.pause()

            await pilot.press("space", "a", "d", "a", "space", "enter")
            await pilot.pause()

        assert app.dismissed is True
        assert app.result == "Ada"

    async def test_submit_title_cases_multiple_lowercase_words(self) -> None:
        """Lowercase full names should be returned in title case."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_name_screen()
            await pilot.pause()

            await pilot.press(
                "a", "d", "a", "space", "l", "o", "v", "e", "l", "a", "c", "e", "enter"
            )
            await pilot.pause()

        assert app.dismissed is True
        assert app.result == "Ada Lovelace"

    async def test_submit_empty_name_warns_and_does_not_dismiss(self) -> None:
        """Submitting empty input should keep the modal open with a warning toast."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_name_screen()
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

        assert app.dismissed is False

    async def test_escape_does_not_dismiss(self) -> None:
        """Escape should not dismiss the modal so a name is always captured."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_name_screen()
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert app.dismissed is False
            assert isinstance(app.screen, LaunchNameScreen)


class TestLaunchWaitingScreen:
    """Tests for launch waiting copy."""

    async def test_ready_to_start_initial_state(self) -> None:
        """The modal can open directly in the post-model-ready state."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_waiting_screen(ready_to_start=True)
            await pilot.pause()

            title = app.screen.query_one("#launch-wait-title", Static)
            copy = app.screen.query_one("#launch-wait-copy", Static)

            assert str(title.render()) == "Waiting to start..."
            assert str(copy.render()) == "The controller will start the round."


class TestLaunchCountdownScreen:
    """Tests for launch countdown copy."""

    async def test_countdown_value_updates(self) -> None:
        """The modal should render and update the visible countdown value."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            screen = app.show_countdown_screen(5)
            await pilot.pause()

            value = app.screen.query_one("#launch-countdown-value", Static)
            assert str(value.render()) == "5..."

            screen.set_seconds(4)
            await pilot.pause()

            assert str(value.render()) == "4..."


class TestNormalizeName:
    """Direct unit tests for `_normalize_name`."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ada", "Ada"),
            ("ada lovelace", "Ada Lovelace"),
            ("  ada  ", "Ada"),
            ("Ada", "Ada"),
            ("ADA", "ADA"),
            ("aDa", "aDa"),
            ("Ada Lovelace", "Ada Lovelace"),
            ("", ""),
            ("   ", ""),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        """Title-case lowercase input; preserve user-typed casing otherwise."""
        assert _normalize_name(raw) == expected
