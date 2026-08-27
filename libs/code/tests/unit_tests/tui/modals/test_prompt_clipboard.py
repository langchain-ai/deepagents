"""Tests for the searchable prompt clipboard modal."""

from __future__ import annotations

from unittest.mock import patch

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import Input, Static

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

    async def test_hovering_a_row_does_not_move_the_selection(self) -> None:
        """Hover is visual only (:hover CSS): arrows move from the selection."""
        app = _PromptClipboardApp()
        async with app.run_test() as pilot:
            screen = app.open(("newest", "middle", "oldest"))
            await pilot.pause()
            await pilot.pause()

            await pilot.hover(screen._rows[2])
            await pilot.pause()

            # Textual tracks the hovered widget for :hover styling; selection
            # and preview stay put.
            assert screen._rows[2].mouse_hover
            assert screen._selected_index == 0
            preview = screen.query_one("#prompt-preview", Static)
            assert str(preview.content) == "newest"

            # Keyboard navigation resumes from the selection, not the hover.
            await pilot.press("down")
            await pilot.pause()
            assert screen._selected_index == 1
            assert str(preview.content) == "middle"
            assert app.results == []
