"""Tests for the searchable prompt clipboard modal."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Container

from deepagents_code.tui.modals.prompt_clipboard import PromptClipboardScreen


class _PromptClipboardApp(App[None]):
    """Minimal host recording prompt clipboard results."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[str | None] = []

    def compose(self) -> ComposeResult:
        yield Container()

    def open(
        self,
        prompts: tuple[str, ...],
        initial_query: str = "",
        *,
        empty_message: str | None = None,
    ) -> PromptClipboardScreen:
        screen = PromptClipboardScreen(
            prompts, initial_query, empty_message=empty_message
        )
        self.push_screen(screen, self.results.append)
        return screen


class TestPromptClipboardScreen:
    """Keyboard interaction and literal rendering tests."""

    async def test_escape_cancels(self) -> None:
        app = _PromptClipboardApp()
        async with app.run_test() as pilot:
            app.open(("saved prompt",))
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert app.results == [None]

    async def test_tab_inserts_selected_prompt(self) -> None:
        """Tab behaves like Enter instead of paging through results."""
        app = _PromptClipboardApp()
        async with app.run_test() as pilot:
            app.open(("newest", "oldest"))
            await pilot.pause()

            await pilot.press("down")
            await pilot.press("tab")
            await pilot.pause()

            assert app.results == ["oldest"]
