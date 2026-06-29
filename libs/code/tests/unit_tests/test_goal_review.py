"""Unit tests for the goal review widget."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.app import App, ComposeResult

from deepagents_code.widgets.ask_user import AskUserTextArea
from deepagents_code.widgets.goal_review import GoalReviewMenu, GoalReviewResult


class _GoalReviewTestApp(App[None]):
    CSS_PATH = Path(__file__).resolve().parents[2] / "deepagents_code" / "app.tcss"

    def compose(self) -> ComposeResult:
        yield GoalReviewMenu("add refresh tokens", "- tests pass", id="goal-review")


class TestGoalReviewMenu:
    """Tests for goal criteria review interactions."""

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
