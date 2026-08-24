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

    async def test_short_terminal_keeps_filter_and_preview_on_screen(self) -> None:
        """A terminal shorter than the modal must not bury the controls.

        The modal's `height: 80%` alone can be shorter than the fixed
        controls (filter, labels, borders/padding, wrapped help), which placed
        the preview and key help entirely below the screen. The `min-height`
        clamp makes Textual center the overflowing modal and clip it top and
        bottom instead, keeping the filter on screen and clipping the preview
        last.
        """
        app = _PromptClipboardApp()
        async with app.run_test(size=(45, 10)) as pilot:
            screen = app.open(tuple(f"prompt {index}" for index in range(30)))
            await pilot.pause()
            await pilot.pause()

            outer = screen.query_one(Vertical)
            filter_ = screen.query_one("#prompt-filter", Input)
            preview = screen.query_one("#prompt-preview-scroll", VerticalScroll)
            help_ = screen.query_one(".prompt-help", Static)
            assert outer.region.height == 13
            assert filter_.region.y >= 0
            assert filter_.region.bottom <= screen.size.height
            assert preview.region.y >= 0
            assert preview.region.bottom <= screen.size.height
            assert help_.region.bottom <= outer.region.bottom

    async def test_tab_and_shift_tab_page_selection(self) -> None:
        """Tab/Shift+Tab page through results; the filter input keeps focus."""
        app = _PromptClipboardApp()
        async with app.run_test() as pilot:
            screen = app.open(tuple(f"prompt {i}" for i in range(12)))
            await pilot.pause()
            search = screen.query_one("#prompt-filter", Input)

            await pilot.press("tab")
            await pilot.pause()
            assert screen.focused is search
            assert screen._selected_index == 5

            await pilot.press("shift+tab")
            await pilot.pause()
            assert screen.focused is search
            assert screen._selected_index == 0

            # Page up at the top stays put rather than wrapping or erroring.
            await pilot.press("shift+tab")
            await pilot.pause()
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

    async def test_initial_query_seeds_the_filter(self) -> None:
        """Escalating from the inline panel carries the typed query over."""
        app = _PromptClipboardApp()
        async with app.run_test() as pilot:
            screen = app.open(("fix the bug", "add feature", "fix tests"), "fix")
            await pilot.pause()
            await pilot.pause()

            search = screen.query_one("#prompt-filter", Input)
            assert search.value == "fix"
            # The cursor sits at the end so typing extends the carried query.
            assert search.cursor_position == len("fix")
            assert screen._filtered == ["fix the bug", "fix tests"]
            rows = screen.query_one("#prompt-list", VerticalScroll).children
            assert len(rows) == 2

    async def test_initial_query_matching_nothing_shows_the_empty_state(self) -> None:
        app = _PromptClipboardApp()
        async with app.run_test() as pilot:
            screen = app.open(("alpha", "beta"), "zzz")
            await pilot.pause()
            await pilot.pause()

            assert screen._filtered == []
            rows = screen.query_one("#prompt-list", VerticalScroll).children
            assert len(rows) == 1
            assert "No matching prompts." in str(rows[0].render())

    async def test_moving_after_a_queued_filter_edit_does_not_raise(self) -> None:
        """Ctrl+C leaves the modal open, so its re-filter must not desync rows.

        `action_copy` consumes a filter edit whose `Changed` message is still
        queued, which widens `_filtered` without re-rendering `_rows`. Paging
        then indexed the stale row list and raised `IndexError`.
        """
        app = _PromptClipboardApp()
        async with app.run_test() as pilot:
            screen = app.open(tuple(f"prompt {index}" for index in range(12)))
            await pilot.pause()

            search = screen.query_one("#prompt-filter", Input)
            search.value = "prompt 1"
            screen._apply_filter("prompt 1")
            await pilot.pause()
            assert len(screen._rows) == len(screen._filtered)

            # Widen the filter behind the modal's back, exactly as a queued
            # `Input.Changed` does, then act on it without letting the render
            # land first.
            search.value = "prompt"
            with patch("deepagents_code.clipboard.copy_text_with_feedback"):
                screen.action_copy()
            assert len(screen._rows) != len(screen._filtered)

            await screen.action_page_older()
            await pilot.pause()

            assert len(screen._rows) == len(screen._filtered)
            assert 0 <= screen._selected_index < len(screen._rows)

    async def test_queued_filter_edit_keeps_the_highlighted_prompt(self) -> None:
        """A consumed filter edit must not silently re-point the selection.

        Snapping to index 0 made Ctrl+C copy a prompt the user never
        highlighted, under a success toast.
        """
        app = _PromptClipboardApp()
        async with app.run_test() as pilot:
            screen = app.open(("alpha one", "alpha two", "beta"))
            await pilot.pause()

            await pilot.press("down")
            await pilot.pause()
            selected = screen._filtered[screen._selected_index]
            assert selected == "alpha two"

            search = screen.query_one("#prompt-filter", Input)
            search.value = "alpha"
            with patch(
                "deepagents_code.clipboard.copy_text_with_feedback"
            ) as copy_text:
                screen.action_copy()

            copy_text.assert_called_once()
            assert copy_text.call_args.args[1] == selected

    async def test_empty_message_replaces_the_no_prompts_text(self) -> None:
        """An unreadable history file must not read as "no prompts yet"."""
        app = _PromptClipboardApp()
        async with app.run_test() as pilot:
            screen = app.open((), empty_message="Could not read prompt history from x")
            await pilot.pause()

            rows = screen.query_one("#prompt-list", VerticalScroll).children
            assert len(rows) == 1
            message = str(rows[0].render())
            assert "Could not read prompt history" in message
            assert "No prompts yet" not in message

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

    async def test_clicking_a_row_selects_but_does_not_submit(self) -> None:
        """Click moves the selection; only Enter dismisses the modal."""
        app = _PromptClipboardApp()
        async with app.run_test() as pilot:
            screen = app.open(("newest", "middle", "oldest"))
            await pilot.pause()
            await pilot.pause()

            await pilot.click(screen._rows[2])
            await pilot.pause()

            assert screen._selected_index == 2
            assert "prompt-row-selected" in screen._rows[2].classes
            assert "prompt-row-selected" not in screen._rows[0].classes
            preview = screen.query_one("#prompt-preview", Static)
            assert str(preview.content) == "oldest"
            assert app.results == []
            assert app.screen is screen

            await pilot.press("enter")
            await pilot.pause()
            assert app.results == ["oldest"]
