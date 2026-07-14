"""Unit tests for the goal review widget."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from textual import events
from textual.app import App, ComposeResult
from textual.widgets import Markdown, Static

import deepagents_code
from deepagents_code.tui.widgets.goal_review import (
    GoalReviewMenu,
    GoalReviewResult,
    GoalReviewTextArea,
)

if TYPE_CHECKING:
    import pytest


class _GoalReviewTestApp(App[None]):
    CSS_PATH = Path(deepagents_code.__file__).resolve().parent / "app.tcss"

    def compose(self) -> ComposeResult:
        yield GoalReviewMenu("add refresh tokens", "- tests pass", id="goal-review")


class _GoalAmendmentReviewTestApp(App[None]):
    CSS_PATH = Path(deepagents_code.__file__).resolve().parent / "app.tcss"

    def compose(self) -> ComposeResult:
        yield GoalReviewMenu(
            "add refresh tokens with rotation",
            "- tests pass\n- rotation works",
            amendment=True,
            id="goal-review",
        )


class TestGoalReviewMenu:
    """Tests for goal criteria review interactions."""

    async def test_menu_receives_focus_on_mount(self) -> None:
        """The menu must receive focus so navigation and quick keys work."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)

            assert menu.has_focus

    async def test_markdown_omits_goal_text(self) -> None:
        """The review widget should show criteria without restating the goal."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            markdown = app.query_one(".goal-review-markdown", Markdown)

            assert "add refresh tokens" not in markdown.source
            assert "- tests pass" in markdown.source
            assert "Proposed criteria" in markdown.source

    async def test_amendment_review_shows_objective_and_criteria(self) -> None:
        """Amendment review should expose both canonical fields before approval."""
        app = _GoalAmendmentReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            markdown = app.query_one(".goal-review-markdown", Markdown)
            title = app.query_one(".goal-review-title", Static)

            assert "Proposed objective" in markdown.source
            assert "add refresh tokens with rotation" in markdown.source
            assert "rotation works" in markdown.source
            assert "amendment" in str(title.content).lower()

    async def test_accept_resolves_accepted(self) -> None:
        """Accept should resolve with the accepted result."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            future: asyncio.Future[GoalReviewResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            menu.action_accept()

            assert await future == {"type": "accepted"}

    async def test_terminal_result_resolves_future_only_once(self) -> None:
        """Later actions must not override the first goal-review result."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            future: asyncio.Future[GoalReviewResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            menu.action_accept()
            menu.action_cancel()
            menu.action_edit()

            assert await future == {"type": "accepted"}
            assert menu.display is False

    async def test_option_highlight_syncs_with_selection(self) -> None:
        """The selected-option class tracks the cursor and clears in edit mode."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            options = menu._option_widgets
            selected_class = "goal-review-option-selected"

            # Mount highlights the first option only.
            assert options[0].has_class(selected_class)
            assert not any(o.has_class(selected_class) for o in options[1:])

            await pilot.press("down")
            await pilot.pause()

            # Highlight follows the cursor to the next option.
            assert options[1].has_class(selected_class)
            assert not options[0].has_class(selected_class)

            menu.action_edit()
            await pilot.pause()

            # With the editor open no option is highlighted, but the cursor
            # glyph stays on the active option so position is not lost.
            assert not any(o.has_class(selected_class) for o in options)
            assert options[1].selected

    async def test_edit_prefills_and_submits_revised_criteria(self) -> None:
        """Edit should prefill generated criteria and submit revisions."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            future: asyncio.Future[GoalReviewResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            menu.action_edit()
            text_input = menu.query_one(".goal-review-edit-input", GoalReviewTextArea)
            assert text_input.display is True
            assert text_input.text == "- tests pass"

            text_input.text = "- tests pass\n- docs updated"
            text_input.focus()
            await pilot.press("enter")

            assert await future == {
                "type": "edited",
                "criteria": "- tests pass\n- docs updated",
            }

    async def test_reject_with_message_submits_feedback(self) -> None:
        """Reject with message should submit feedback for regeneration."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            future: asyncio.Future[GoalReviewResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            menu.action_reject_with_message()
            text_input = menu.query_one(".goal-review-edit-input", GoalReviewTextArea)
            assert text_input.display is True
            assert text_input.text == ""

            text_input.text = "include docs and migration notes"
            text_input.focus()
            await pilot.press("enter")

            assert await future == {
                "type": "rejected",
                "message": "include docs and migration notes",
            }

    async def test_edit_expands_collapsed_paste_on_submit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A collapsed paste in the editor expands in the submitted criteria."""
        from deepagents_code.tui.widgets import _paste_textarea as paste_textarea_module

        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: True
        )
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            future: asyncio.Future[GoalReviewResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            menu.action_edit()
            text_input = menu.query_one(".goal-review-edit-input", GoalReviewTextArea)
            text_input.text = ""
            text_input.focus()

            big = "- crit\n" * 5
            await text_input._on_paste(events.Paste(big))
            await pilot.pause()
            assert text_input.text == "[Pasted text #1 +5 lines]"

            await pilot.press("enter")

            assert await future == {"type": "edited", "criteria": big.strip()}

    async def test_keypress_accept_resolves_accepted(self) -> None:
        """The accept quick-key resolves through the real binding dispatch."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            future: asyncio.Future[GoalReviewResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.press("y")

            assert await future == {"type": "accepted"}

    async def test_keypress_reject_enters_reject_mode(self) -> None:
        """The reject quick-key opens the feedback editor without resolving."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            future: asyncio.Future[GoalReviewResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.press("r")

            text_input = menu.query_one(".goal-review-edit-input", GoalReviewTextArea)
            assert text_input.display is True
            assert text_input.text == ""
            assert future.done() is False

    async def test_keypress_cancel_resolves_cancelled(self) -> None:
        """The cancel quick-key resolves through the real binding dispatch."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            future: asyncio.Future[GoalReviewResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.press("n")

            assert await future == {"type": "cancelled"}

    async def test_keypress_escape_resolves_cancelled(self) -> None:
        """Escape from the menu (not edit mode) cancels the proposal."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            future: asyncio.Future[GoalReviewResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.press("escape")

            assert await future == {"type": "cancelled"}

    async def test_arrow_navigation_then_enter_selects_highlighted(self) -> None:
        """Down+Enter dispatches `action_select` to the highlighted option (edit)."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            future: asyncio.Future[GoalReviewResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.press("down", "enter")

            text_input = menu.query_one(".goal-review-edit-input", GoalReviewTextArea)
            assert menu._selected == 1
            assert text_input.display is True
            assert future.done() is False

    async def test_edit_mode_keeps_quick_keys_in_text_input(self) -> None:
        """Quick-key characters should type text instead of triggering menu actions."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            future: asyncio.Future[GoalReviewResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            menu.action_edit()
            text_input = menu.query_one(".goal-review-edit-input", GoalReviewTextArea)
            text_input.text = ""
            await pilot.press("y", "e", "n")

            assert text_input.text == "yen"
            assert future.done() is False
            assert text_input.display is True

    async def test_edit_mode_preserves_text_area_navigation_keys(self) -> None:
        """Backspace and arrow keys should keep normal TextArea behavior."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            future: asyncio.Future[GoalReviewResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            menu.action_edit()
            text_input = menu.query_one(".goal-review-edit-input", GoalReviewTextArea)
            text_input.text = ""
            await pilot.press("a", "b", "left", "backspace", "c")

            assert text_input.text == "cb"
            assert future.done() is False
            assert text_input.display is True

    async def test_cancel_closes_edit_before_cancelling_proposal(self) -> None:
        """Esc from edit mode should return to menu before cancelling the proposal."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            future: asyncio.Future[GoalReviewResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            menu.action_edit()
            text_input = menu.query_one(".goal-review-edit-input", GoalReviewTextArea)
            await pilot.pause()
            assert text_input.has_focus

            await pilot.press("escape")
            await pilot.pause()

            assert future.done() is False
            assert text_input.display is False
            assert menu.has_focus

            await pilot.press("escape")

            assert await future == {"type": "cancelled"}

    async def test_empty_edit_does_not_submit(self) -> None:
        """Submitting blank edited criteria should keep the editor open."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            future: asyncio.Future[GoalReviewResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            menu.action_edit()
            text_input = menu.query_one(".goal-review-edit-input", GoalReviewTextArea)
            text_input.text = "   "
            menu._submit_edit()

            assert future.done() is False
            assert text_input.display is True
            help_widget = menu.query_one(".goal-review-help", Static)
            assert "Enter some criteria" in str(help_widget.content)

    async def test_empty_rejection_does_not_submit(self) -> None:
        """Submitting blank rejection feedback should keep the editor open."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            future: asyncio.Future[GoalReviewResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            menu.action_reject_with_message()
            text_input = menu.query_one(".goal-review-edit-input", GoalReviewTextArea)
            text_input.text = "  \n  "
            menu._submit_rejection()

            assert future.done() is False
            assert text_input.display is True
            help_widget = menu.query_one(".goal-review-help", Static)
            assert "Enter some feedback" in str(help_widget.content)

    async def test_blur_does_not_dismiss_proposal(self) -> None:
        """Losing focus must not resolve the proposal future."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            future: asyncio.Future[GoalReviewResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            menu.on_blur(events.Blur())

            assert future.done() is False
            assert menu.display is True
