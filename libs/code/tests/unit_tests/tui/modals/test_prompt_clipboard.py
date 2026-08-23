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

    async def test_row_shows_only_first_line_of_multiline_prompt(self) -> None:
        """Rows are one line; continuation lines belong to the preview pane."""
        prompt = "title line\ndetails here"
        app = _PromptClipboardApp()
        async with app.run_test() as pilot:
            screen = app.open((prompt,))
            await pilot.pause()

            assert str(screen._rows[0].content) == "title line"

    async def test_single_line_prompt_row_does_not_repeat_text(self) -> None:
        """A one-line prompt renders as one line, not a duplicated excerpt."""
        app = _PromptClipboardApp()
        async with app.run_test() as pilot:
            screen = app.open(("single line prompt",))
            await pilot.pause()

            assert str(screen._rows[0].content) == "single line prompt"
            assert screen._rows[0].region.height == 1

    async def test_empty_state_fits_narrow_terminal(self) -> None:
        app = _PromptClipboardApp()
        async with app.run_test(size=(45, 18)) as pilot:
            screen = app.open(())
            await pilot.pause()

            rows = screen.query_one("#prompt-list", VerticalScroll)
            empty = rows.query_one(Static)
            assert (
                str(empty.content) == "No prompts yet. Submitted prompts appear here."
            )
            assert screen.region.width <= 45

    async def test_list_scrolls_when_rows_exceed_max_height(self) -> None:
        """Rows past the visible cap must overflow the list, not be clipped."""
        app = _PromptClipboardApp()
        async with app.run_test(size=(80, 40)) as pilot:
            screen = app.open(tuple(f"prompt {index}" for index in range(30)))
            await pilot.pause()
            await pilot.pause()

            rows_list = screen.query_one("#prompt-list", VerticalScroll)
            assert len(screen._rows) == 30
            assert rows_list.is_scrollable
            assert rows_list.virtual_size.height > rows_list.region.height
            assert rows_list.max_scroll_y > 0

            await pilot.press("down")
            await pilot.pause()
            assert screen._rows[1].region.height > 0

    async def test_preview_matches_list_width_when_rows_scroll(self) -> None:
        """The preview box spans the same width as the scrollable prompt list."""
        app = _PromptClipboardApp()
        async with app.run_test(size=(80, 40)) as pilot:
            app.open(tuple(f"prompt {index}" for index in range(30)))
            await pilot.pause()
            await pilot.pause()

            rows_list = app.screen.query_one("#prompt-list", VerticalScroll)
            preview = app.screen.query_one("#prompt-preview-scroll", VerticalScroll)
            assert rows_list.show_vertical_scrollbar
            assert preview.region.width == rows_list.region.width

    async def test_modal_height_is_constant_while_previewing(self) -> None:
        """Selecting prompts of different lengths must not resize the modal."""
        app = _PromptClipboardApp()
        async with app.run_test(size=(80, 40)) as pilot:
            screen = app.open(("short", "long\n" + "line\n" * 20))
            await pilot.pause()

            outer = screen.query_one(Vertical)
            preview = screen.query_one("#prompt-preview-scroll", VerticalScroll)
            initial_outer = outer.region.height
            initial_preview = preview.region.height

            await pilot.press("down")
            await pilot.pause()

            assert str(screen.query_one("#prompt-preview", Static).content).startswith(
                "long"
            )
            assert outer.region.height == initial_outer
            assert preview.region.height == initial_preview
            assert preview.is_scrollable

    async def test_list_fills_fixed_modal_and_scrolls(self) -> None:
        """The list takes the fixed modal's leftover space and clips rows."""
        app = _PromptClipboardApp()
        async with app.run_test(size=(80, 18)) as pilot:
            screen = app.open(tuple(f"prompt {index}" for index in range(30)))
            await pilot.pause()
            await pilot.pause()

            rows_list = screen.query_one("#prompt-list", VerticalScroll)
            outer = screen.query_one(Vertical)
            help_ = screen.query_one(".prompt-help", Static)
            assert rows_list.region.height < 10
            assert rows_list.show_vertical_scrollbar
            assert help_.region.bottom <= outer.region.bottom

    async def test_tab_and_shift_tab_do_not_move_focus(self) -> None:
        """Focus keys are swallowed; the search input keeps focus throughout."""
        app = _PromptClipboardApp()
        async with app.run_test() as pilot:
            screen = app.open(("alpha", "beta"))
            await pilot.pause()
            search = screen.query_one("#prompt-filter", Input)

            await pilot.press("tab")
            await pilot.pause()
            assert screen.focused is search
            assert screen._selected_index == 0

            await pilot.press("shift+tab")
            await pilot.pause()
            assert screen.focused is search
            assert screen._selected_index == 0

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
