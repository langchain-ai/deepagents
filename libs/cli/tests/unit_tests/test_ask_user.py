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
