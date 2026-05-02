"""Tests for launch initialization screens."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Container

from deepagents_cli.widgets.launch_init import LaunchNameScreen


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


class TestLaunchNameScreen:
    """Tests for launch name entry."""

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

    async def test_escape_cancels(self) -> None:
        """Escape should cancel the setup flow."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_name_screen()
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

        assert app.dismissed is True
        assert app.result is None
