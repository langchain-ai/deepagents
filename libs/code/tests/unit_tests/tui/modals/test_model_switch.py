"""Tests for the large-context model-switch warning modal."""

from textual.app import App
from textual.widgets import Static

from deepagents_code.tui.modals.model_switch import ModelSwitchWarningScreen


class _Host(App[None]):
    """Minimal host recording modal results."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[bool | None] = []

    def open(self, *, approximate: bool = False) -> ModelSwitchWarningScreen:
        """Open a representative warning screen."""
        screen = ModelSwitchWarningScreen(
            current_model="anthropic:claude[old]",
            target_model="openai:gpt[new]",
            context_tokens=124_000,
            threshold=100_000,
            approximate=approximate,
        )
        self.push_screen(screen, self.results.append)
        return screen


async def test_enter_confirms_and_escape_cancels() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        app.open()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.results == [True]

        app.open()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app.results == [True, False]


async def test_click_does_not_authorize_switch() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        app.open()
        await pilot.pause()
        await pilot.click(".model-switch-warning-body")
        await pilot.pause()
        assert app.results == []


async def test_dynamic_copy_renders_literally_and_marks_approximation() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        screen = app.open(approximate=True)
        await pilot.pause()
        body = screen.query_one(".model-switch-warning-body", Static)
        rendered = str(body.content)
        assert "approximately 124.0K" in rendered
        assert "anthropic:claude[old]" in rendered
        assert "openai:gpt[new]" in rendered
