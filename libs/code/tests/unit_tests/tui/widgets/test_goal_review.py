"""Unit tests for the goal review widget."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.widgets import Markdown, Static

import deepagents_code
from deepagents_code.tui.widgets.ask_user import AskUserTextArea
from deepagents_code.tui.widgets.goal_review import GoalReviewMenu, GoalReviewResult


class _GoalReviewTestApp(App[None]):
    CSS_PATH = Path(deepagents_code.__file__).resolve().parent / "app.tcss"

    def compose(self) -> ComposeResult:
        yield GoalReviewMenu("add refresh tokens", "- tests pass", id="goal-review")


class TestGoalReviewMenu:
    """Tests for goal criteria review interactions."""

    async def test_markdown_omits_goal_text(self) -> None:
        """The review widget should show criteria without restating the goal."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            markdown = app.query_one(".goal-review-markdown", Markdown)

            assert "add refresh tokens" not in markdown.source
            assert "- tests pass" in markdown.source
            assert "Proposed criteria" in markdown.source

    async def test_actions_explain_execution_and_feedback(self) -> None:
        """Every review action should state its effect clearly."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            options = [
                str(widget.content)
                for widget in app.query(".goal-review-option").results(Static)
            ]

        assert any("Accept and run (y)" in option for option in options)
        assert any(
            "Accept and run automatically with unrestricted permissions (a)" in option
            for option in options
        )
        assert any("Edit (e)" in option for option in options)
        assert any("Reject with feedback (r)" in option for option in options)
        assert any("Cancel (n)" in option for option in options)

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

    async def test_accept_unrestricted_resolves_automatic_result(self) -> None:
        """Automatic acceptance should return its distinct execution decision."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            future: asyncio.Future[GoalReviewResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            menu.action_accept_unrestricted()

            assert await future == {"type": "accepted_unrestricted"}

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
            text_input = menu.query_one(".goal-review-edit-input", AskUserTextArea)
            assert text_input.display is True
            assert text_input.text == "- tests pass"

            text_input.text = "- tests pass\n- docs updated"
            menu._submit_edit()

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
            text_input = menu.query_one(".goal-review-edit-input", AskUserTextArea)
            assert text_input.display is True
            assert text_input.text == ""

            text_input.text = "include docs and migration notes"
            menu._submit_rejection()

            assert await future == {
                "type": "rejected",
                "message": "include docs and migration notes",
            }

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

    async def test_keypress_automatic_acceptance_resolves_distinct_result(self) -> None:
        """The automatic quick-key should enable the unattended execution path."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            future: asyncio.Future[GoalReviewResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.press("a")

            assert await future == {"type": "accepted_unrestricted"}

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

            text_input = menu.query_one(".goal-review-edit-input", AskUserTextArea)
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
        """Arrow navigation dispatches the highlighted edit action."""
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            future: asyncio.Future[GoalReviewResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.press("down", "down", "enter")

            text_input = menu.query_one(".goal-review-edit-input", AskUserTextArea)
            assert menu._selected == 2
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
            text_input = menu.query_one(".goal-review-edit-input", AskUserTextArea)
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
            text_input = menu.query_one(".goal-review-edit-input", AskUserTextArea)
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
            text_input = menu.query_one(".goal-review-edit-input", AskUserTextArea)
            menu.action_cancel()

            assert future.done() is False
            assert text_input.display is False

            menu.action_cancel()

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
            text_input = menu.query_one(".goal-review-edit-input", AskUserTextArea)
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
            text_input = menu.query_one(".goal-review-edit-input", AskUserTextArea)
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
