"""Tests for launch initialization screens."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Input

from deepagents_cli.widgets.launch_init import LaunchNameScreen, _normalize_name


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
