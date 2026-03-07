"""Tests for ask_user tool integration in the CLI."""

import asyncio

from textual.app import App, ComposeResult
from textual.widgets import Input, Static

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

    async def test_tab_advances_to_next_question(self) -> None:
        """Tab moves active indicator forward without confirming."""
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

            await pilot.press("tab")
            await pilot.pause()

            assert qw1.has_class("ask-user-question-active")
            assert qw0.has_class("ask-user-question-inactive")
            # Tab should NOT confirm the answer
            assert not menu._confirmed[0]

    async def test_tab_clamps_at_last_question(self) -> None:
        """Tab at the last question is a no-op."""
        app = _AskUserTestApp(
            [
                {"question": "Q1?", "type": "text"},
                {"question": "Q2?", "type": "text"},
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)

            # Move to last question
            menu.action_next_question()
            await pilot.pause()
            assert menu._current_question == 1

            # Tab again — should stay at 1
            menu.action_next_question()
            await pilot.pause()
            assert menu._current_question == 1

    async def test_tab_noop_for_single_question(self) -> None:
        """Single question: tab does nothing."""
        app = _AskUserTestApp([{"question": "Q1?", "type": "text"}])

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            assert menu._current_question == 0

            menu.action_next_question()
            await pilot.pause()
            assert menu._current_question == 0

    async def test_previous_question_navigates_backward(self) -> None:
        """`action_previous_question` moves backward."""
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

            # Move forward first
            menu.action_next_question()
            await pilot.pause()
            assert qw1.has_class("ask-user-question-active")

            # Move backward
            menu.action_previous_question()
            await pilot.pause()
            assert qw0.has_class("ask-user-question-active")
            assert qw1.has_class("ask-user-question-inactive")

    async def test_previous_question_clamps_at_first(self) -> None:
        """At first question: previous is a no-op."""
        app = _AskUserTestApp(
            [
                {"question": "Q1?", "type": "text"},
                {"question": "Q2?", "type": "text"},
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            assert menu._current_question == 0

            menu.action_previous_question()
            await pilot.pause()
            assert menu._current_question == 0

    async def test_help_text_shows_tab_hint_for_multiple(self) -> None:
        """Footer mentions Tab for 2+ questions."""
        app = _AskUserTestApp(
            [
                {"question": "Q1?", "type": "text"},
                {"question": "Q2?", "type": "text"},
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            help_text = menu.query_one(".ask-user-help").render()
            assert "Tab" in str(help_text)

    async def test_help_text_omits_tab_hint_for_single(self) -> None:
        """Footer omits Tab for 1 question."""
        app = _AskUserTestApp([{"question": "Q1?", "type": "text"}])

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            help_text = menu.query_one(".ask-user-help").render()
            assert "Tab" not in str(help_text)

    async def test_required_label_shown_for_required_question(self) -> None:
        """Required questions display a (required) indicator."""
        app = _AskUserTestApp([{"question": "Name?", "type": "text", "required": True}])

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            qw = menu._question_widgets[0]
            rendered = str(qw.query_one(Static).render())
            assert "required" in rendered

    async def test_required_label_hidden_for_optional_question(self) -> None:
        """Optional questions do not display a (required) indicator."""
        app = _AskUserTestApp(
            [{"question": "Nickname?", "type": "text", "required": False}]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            qw = menu._question_widgets[0]
            rendered = str(qw.query_one(Static).render())
            assert "required" not in rendered

    async def test_required_is_true_by_default(self) -> None:
        """Questions without explicit required field default to required."""
        app = _AskUserTestApp([{"question": "Name?", "type": "text"}])

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            qw = menu._question_widgets[0]
            assert qw._required is True
            rendered = str(qw.query_one(Static).render())
            assert "required" in rendered

    async def test_optional_question_submits_with_empty_answer(self) -> None:
        """Non-required questions can be submitted with empty answers."""
        app = _AskUserTestApp(
            [{"question": "Nickname?", "type": "text", "required": False}]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            future: asyncio.Future[dict[str, object]] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.pause()
            # Press enter without typing anything
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {"type": "answered", "answers": [""]}

    async def test_required_question_blocks_empty_submit(self) -> None:
        """Required questions block submission when answer is empty."""
        app = _AskUserTestApp([{"question": "Name?", "type": "text", "required": True}])

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            future: asyncio.Future[dict[str, object]] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.pause()
            # Press enter without typing anything
            await pilot.press("enter")
            await pilot.pause()

            assert not future.done()

    async def test_up_from_other_input_selects_last_choice_directly(self) -> None:
        """Pressing up while Other input is focused jumps to last real choice."""
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

            # Navigate to Other and enter it
            await pilot.press("down")
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            other_input = menu.query_one(".ask-user-other-input", Input)
            assert other_input.has_focus

            # Single up press should select "blue" (last real choice)
            await pilot.press("up")
            await pilot.pause()
            assert qw._selected_choice == 1
            assert not qw._is_other_selected
            assert qw.has_focus

    async def test_return_to_mc_other_refocuses_input(self) -> None:
        """Tab away from Other input and Shift+Tab back refocuses it."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick one",
                    "type": "multiple_choice",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                },
                {"question": "Name?", "type": "text"},
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)

            # Navigate to Other and enter it
            await pilot.press("down")
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            other_input = menu.query_one(".ask-user-other-input", Input)
            assert other_input.has_focus

            # Tab to next question
            menu.action_next_question()
            await pilot.pause()
            assert menu._current_question == 1

            # Go back — Other input should regain focus
            menu.action_previous_question()
            await pilot.pause()
            assert menu._current_question == 0
            assert other_input.has_focus
