"""Tests for the blocked-resume action modal."""

import html

from textual.app import App

from deepagents_code.tui.modals.resume_blocked import (
    ResumeBlockedChoice,
    ResumeBlockedScreen,
)


class _Host(App[None]):
    """Minimal host recording modal results."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[ResumeBlockedChoice | None] = []

    def open(self) -> None:
        """Open a representative blocked-resume prompt."""
        self.push_screen(
            ResumeBlockedScreen("Thread stale[bold] cannot be resumed.\u202e"),
            self.results.append,
        )


async def test_choices_start_new_or_exit() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        app.open()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.results == ["new"]

        app.open()
        await pilot.pause()
        await pilot.press("down", "enter")
        await pilot.pause()
        assert app.results == ["new", "exit"]

        app.open()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app.results == ["new", "exit", "exit"]


async def test_renders_reason_as_sanitized_plain_text() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        app.open()
        await pilot.pause()
        rendered = html.unescape(app.export_screenshot()).replace("\xa0", " ")

    assert "Thread stale[bold] cannot be resumed." in rendered
    assert "\u202e" not in rendered
    assert "Start a new session" in rendered
    assert "Enter: select" in rendered
    assert "Esc: exit" in rendered
