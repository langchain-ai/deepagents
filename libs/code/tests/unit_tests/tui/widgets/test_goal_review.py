"""Unit tests for the goal review widget."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from textual import events
from textual.app import App, ComposeResult
from textual.widgets import Markdown, Static

import deepagents_code
from deepagents_code.goal_state_limits import GOAL_APPLICATION_CHAR_LIMIT
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

    async def test_text_editor_help_names_configured_editor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both goal-review text modes should name the configured editor."""
        monkeypatch.setenv("VISUAL", "nvim")
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            help_widget = menu.query_one(".goal-review-help", Static)

            menu.action_edit()
            assert "Ctrl+G edit in nvim" in str(help_widget.content)

            menu.action_cancel()
            menu.action_reject_with_message()
            assert "Ctrl+G edit in nvim" in str(help_widget.content)

    async def test_newline_hint_uses_terminal_shortcut(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Newline hints use the terminal-aware shortcut, not a hardcoded key.

        On terminals that cannot report Shift+Enter (e.g. macOS Terminal.app),
        `newline_shortcut` returns `Ctrl+J`/`Option+Enter`; the goal editor must
        advertise that key rather than a Shift+Enter that would submit instead.
        """
        from deepagents_code import config as config_module

        # `newline_hint` resolves `newline_shortcut` via a call-time
        # `from deepagents_code.config import newline_shortcut`, so patch the
        # name on the config module it actually looks up.
        monkeypatch.setattr(config_module, "newline_shortcut", lambda: "Ctrl+J")
        app = _GoalReviewTestApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#goal-review", GoalReviewMenu)
            help_widget = menu.query_one(".goal-review-help", Static)

            menu.action_edit()
            assert "Ctrl+J newline" in str(help_widget.content)
            assert "Shift+Enter" not in str(help_widget.content)

            menu.action_cancel()
            menu.action_reject_with_message()
            assert "Ctrl+J newline" in str(help_widget.content)

            menu._hint_empty_submission("criteria")
            assert "Ctrl+J newline" in str(help_widget.content)

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
            # Post through the App so Textual's MRO dispatch reaches the
            # base handlers that perform the insert.
            pilot.app.post_message(events.Paste(big))
            await pilot.pause()
            assert text_input.text == "[Pasted text #1 +5 lines]"

            await pilot.press("enter")

            assert await future == {"type": "edited", "criteria": big.strip()}

    async def test_regenerate_expands_collapsed_paste_on_submit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A collapsed paste in the feedback box expands in the regenerate message."""
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

            menu.action_reject_with_message()
            text_input = menu.query_one(".goal-review-edit-input", GoalReviewTextArea)
            text_input.focus()

            big = "- feedback\n" * 5
            # Post through the App so Textual's MRO dispatch reaches the
            # base handlers that perform the insert.
            pilot.app.post_message(events.Paste(big))
            await pilot.pause()
            assert text_input.text == "[Pasted text #1 +5 lines]"

            await pilot.press("enter")

            assert await future == {"type": "rejected", "message": big.strip()}

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
