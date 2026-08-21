"""Tests for the searchable prompt clipboard modal."""

from __future__ import annotations

from unittest.mock import patch

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Input, Static

from deepagents_code.tui.modals.prompt_clipboard import PromptClipboardScreen


class _PromptClipboardApp(App[None]):
    """Minimal host recording prompt clipboard results."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[str | None] = []

    def compose(self) -> ComposeResult:
        yield Container()

    def open(self, prompts: tuple[str, ...]) -> PromptClipboardScreen:
        screen = PromptClipboardScreen(prompts)
        self.push_screen(screen, self.results.append)
        return screen


class TestPromptClipboardScreen:
    """Keyboard interaction and literal rendering tests."""

    async def test_filter_autofocus_and_typing_c(self) -> None:
        app = _PromptClipboardApp()
        async with app.run_test() as pilot:
            screen = app.open(("alpha", "contains c", "charlie"))
            await pilot.pause()

            search = screen.query_one("#prompt-filter", Input)
            assert screen.focused is search

            await pilot.press("c")
            await pilot.pause()

            assert search.value == "c"
            assert screen._filtered == ["contains c", "charlie"]

    async def test_navigation_preview_and_enter_selection(self) -> None:
        app = _PromptClipboardApp()
        async with app.run_test() as pilot:
            screen = app.open(("newest", "older\nwith details"))
            await pilot.pause()

            await pilot.press("down")
            await pilot.pause()
            preview = screen.query_one("#prompt-preview", Static)
            assert str(preview.content) == "older\nwith details"

            await pilot.press("enter")
            await pilot.pause()

            assert app.results == ["older\nwith details"]

    async def test_immediate_enter_uses_latest_unrendered_filter(self) -> None:
        app = _PromptClipboardApp()
        async with app.run_test() as pilot:
            screen = app.open(("saved prompt",))
            await pilot.pause()
            search = screen.query_one("#prompt-filter", Input)

            search.value = "no match"
            screen.action_select()

            assert app.results == []
            assert app.screen is screen

    async def test_escape_cancels(self) -> None:
        app = _PromptClipboardApp()
        async with app.run_test() as pilot:
            app.open(("saved prompt",))
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert app.results == [None]

    async def test_markup_like_prompt_renders_literally(self) -> None:
        prompt = "[bold red]do not style[/]\n<literal>"
        app = _PromptClipboardApp()
        async with app.run_test() as pilot:
            screen = app.open((prompt,))
            await pilot.pause()

            preview = screen.query_one("#prompt-preview", Static)
            assert str(preview.content) == prompt
            assert str(screen._rows[0].content).startswith("[bold red]")

    async def test_empty_state_fits_narrow_terminal(self) -> None:
        app = _PromptClipboardApp()
        async with app.run_test(size=(45, 18)) as pilot:
            screen = app.open(())
            await pilot.pause()

            rows = screen.query_one("#prompt-rows", Container)
            empty = rows.query_one(Static)
            assert (
                str(empty.content) == "No prompts yet. Submitted prompts appear here."
            )
            assert screen.region.width <= 45

    async def test_ctrl_c_copies_selected_prompt_without_dismissing(self) -> None:
        app = _PromptClipboardApp()
        async with app.run_test() as pilot:
            screen = app.open(("copy me",))
            await pilot.pause()

            with patch(
                "deepagents_code.clipboard.copy_text_with_feedback"
            ) as copy_text:
                await pilot.press("ctrl+c")
                await pilot.pause()

            copy_text.assert_called_once_with(
                app,
                "copy me",
                failure_noun="prompt",
                success_message="Prompt copied to clipboard",
            )
            assert app.screen is screen
