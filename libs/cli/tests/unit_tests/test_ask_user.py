"""Tests for ask_user tool integration in the CLI."""

import asyncio

from textual.app import App, ComposeResult
from textual.widgets import Input

from deepagents_cli.tool_display import format_tool_display
from deepagents_cli.widgets.ask_user import AskUserMenu


class _AskUserTestApp(App[None]):
    def __init__(self, questions: list[dict[str, object]]) -> None:
        super().__init__()
        self._questions = questions

    def compose(self) -> ComposeResult:
        yield AskUserMenu(self._questions, id="ask-user-menu")


class TestAskUserToolDisplay:
    """Tests for ask_user formatting in tool_display."""

    def test_format_single_question(self) -> None:
        result = format_tool_display(
            "ask_user",
            {
                "questions": [
                    {"question": "What is your name?", "type": "text"},
                ]
            },
        )
        assert "ask_user" in result
        assert "1 question" in result

    def test_format_multiple_questions(self) -> None:
        result = format_tool_display(
            "ask_user",
            {
                "questions": [
                    {"question": "Name?", "type": "text"},
                    {
                        "question": "Color?",
                        "type": "multiple_choice",
                        "choices": [{"value": "red"}, {"value": "blue"}],
                    },
                ]
            },
        )
        assert "ask_user" in result
        assert "2 questions" in result

    def test_format_empty_questions(self) -> None:
        result = format_tool_display("ask_user", {"questions": []})
        assert "ask_user" in result
        assert "0 questions" in result

    def test_format_no_questions_key(self) -> None:
        result = format_tool_display("ask_user", {})
        assert "ask_user" in result


class TestAskUserMenu:
    async def test_text_input_receives_focus_on_mount(self) -> None:
        """The text Input must have focus after mount so the user can type."""
        app = _AskUserTestApp([{"question": "What is your name?", "type": "text"}])

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            text_input = menu.query_one(".ask-user-text-input", Input)
            assert text_input.has_focus

    async def test_multiple_choice_question_widget_receives_focus_on_mount(
        self,
    ) -> None:
        """The _QuestionWidget must have focus so arrow/enter bindings work."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick one",
                    "type": "multiple_choice",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                }
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            qw = menu._question_widgets[0]
            assert qw.has_focus

    async def test_text_question_submits_typed_answer(self) -> None:
        app = _AskUserTestApp([{"question": "What is your name?", "type": "text"}])

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            future: asyncio.Future[dict[str, object]] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.pause()
            text_input = menu.query_one(".ask-user-text-input", Input)
            text_input.value = "Alice"
            text_input.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {"type": "answered", "answers": ["Alice"]}

    async def test_escape_cancels_and_resolves_future(self) -> None:
        app = _AskUserTestApp([{"question": "Name?", "type": "text"}])

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            future: asyncio.Future[dict[str, object]] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.pause()
            menu.focus()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

            assert future.done()
            assert future.result() == {"type": "cancelled"}

    async def test_multiple_choice_submits_without_text_input(self) -> None:
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick one",
                    "type": "multiple_choice",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                }
            ]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            future: asyncio.Future[dict[str, object]] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {"type": "answered", "answers": ["red"]}

    async def test_multiple_choice_other_accepts_custom_text(self) -> None:
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick one",
                    "type": "multiple_choice",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                }
            ]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            future: asyncio.Future[dict[str, object]] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.pause()
            await pilot.press("down")
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            other_input = menu.query_one(".ask-user-other-input", Input)
            other_input.value = "green"
            other_input.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {"type": "answered", "answers": ["green"]}

    async def test_enter_advances_sequentially_through_mc_questions(self) -> None:
        """Enter on a MC question should advance to the next, not skip."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Color?",
                    "type": "multiple_choice",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                },
                {
                    "question": "Size?",
                    "type": "multiple_choice",
                    "choices": [{"value": "S"}, {"value": "M"}, {"value": "L"}],
                },
                {"question": "Name?", "type": "text"},
            ]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            future: asyncio.Future[dict[str, object]] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.pause()
            # Q1 (MC) — first question should be active
            qw0 = menu._question_widgets[0]
            assert qw0.has_focus
            assert qw0.has_class("ask-user-question-active")

            # Press Enter to confirm Q1 default ("red") → should advance to Q2
            await pilot.press("enter")
            await pilot.pause()
            qw1 = menu._question_widgets[1]
            assert qw1.has_focus
            assert qw1.has_class("ask-user-question-active")
            assert qw0.has_class("ask-user-question-inactive")
            assert not future.done(), "Should not submit yet"

            # Navigate to "M" on Q2 and confirm
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            text_input = menu.query_one(".ask-user-text-input", Input)
            assert text_input.has_focus
            assert not future.done(), "Should not submit yet"

            # Type answer for Q3 and submit
            text_input.value = "Alice"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {
                "type": "answered",
                "answers": ["red", "M", "Alice"],
            }

    async def test_active_question_has_visual_indicator(self) -> None:
        """The active question should have the active CSS class."""
        app = _AskUserTestApp(
            [
                {"question": "Q1?", "type": "text"},
                {"question": "Q2?", "type": "text"},
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            qw0 = menu._question_widgets[0]
            qw1 = menu._question_widgets[1]
            assert qw0.has_class("ask-user-question-active")
            assert qw1.has_class("ask-user-question-inactive")
