"""Tests for ask_user tool integration in the CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Markdown, Static

import deepagents_code
from deepagents_code.tool_display import format_tool_display
from deepagents_code.tui.widgets.ask_user import (
    _TRAILING_ANNOTATION_RE,
    AskUserMenu,
    AskUserTextArea,
    _QuestionWidget,
)

if TYPE_CHECKING:
    import pytest

    from deepagents_code._ask_user_types import AskUserWidgetResult, Question


class _AskUserTestApp(App[None]):
    CSS_PATH = Path(deepagents_code.__file__).resolve().parent / "app.tcss"

    def __init__(self, questions: list[Question]) -> None:
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


class TestTrailingAnnotationRegex:
    """Strips LLM-appended '(optional)'/'- required' annotations from question text."""

    def test_strips_dash_optional(self) -> None:
        assert _TRAILING_ANNOTATION_RE.sub("", "Your name? - optional") == "Your name?"

    def test_strips_parens_optional(self) -> None:
        assert _TRAILING_ANNOTATION_RE.sub("", "Your name? (optional)") == "Your name?"

    def test_strips_brackets_required(self) -> None:
        assert _TRAILING_ANNOTATION_RE.sub("", "Pick one [required]") == "Pick one"

    def test_strips_em_dash(self) -> None:
        text = "Your name? \u2014 optional"
        assert _TRAILING_ANNOTATION_RE.sub("", text) == "Your name?"

    def test_strips_en_dash(self) -> None:
        text = "Your name? \u2013 optional"
        assert _TRAILING_ANNOTATION_RE.sub("", text) == "Your name?"

    def test_strips_parens_required(self) -> None:
        assert _TRAILING_ANNOTATION_RE.sub("", "Pick one (required)") == "Pick one"

    def test_strips_dash_required(self) -> None:
        assert _TRAILING_ANNOTATION_RE.sub("", "Pick one - required") == "Pick one"

    def test_strips_with_trailing_whitespace(self) -> None:
        text = "Your name? (optional)   "
        assert _TRAILING_ANNOTATION_RE.sub("", text) == "Your name?"

    def test_strips_with_trailing_newline(self) -> None:
        text = "Your name? (optional)\n"
        assert _TRAILING_ANNOTATION_RE.sub("", text) == "Your name?"

    def test_strips_with_trailing_punctuation(self) -> None:
        text = "Your name? \u2014 Optional."
        assert _TRAILING_ANNOTATION_RE.sub("", text) == "Your name?"
        assert _TRAILING_ANNOTATION_RE.sub("", "Pick one (OPTIONAL!)") == "Pick one"

    def test_case_insensitive(self) -> None:
        assert _TRAILING_ANNOTATION_RE.sub("", "Your name? (Optional)") == "Your name?"

    def test_preserves_trailing_word_optional_without_delimiter(self) -> None:
        """Bare trailing 'optional' with no delimiter is not an annotation."""
        text = "Which field is optional"
        assert _TRAILING_ANNOTATION_RE.sub("", text) == text

    def test_leaves_question_without_annotation(self) -> None:
        text = "What is your name?"
        assert _TRAILING_ANNOTATION_RE.sub("", text) == text


class TestAskUserTextAreaBindings:
    """Ensures the ask-user text area matches chat input editing shortcuts."""

    def test_modified_backspace_deletes_word_left(self) -> None:
        """Modified Backspace aliases should delete the previous word."""
        word_delete_keys = {
            key.strip()
            for binding in AskUserTextArea.BINDINGS
            if isinstance(binding, Binding) and binding.action == "delete_word_left"
            for key in binding.key.split(",")
        }

        assert "ctrl+backspace" in word_delete_keys
        assert "alt+backspace" in word_delete_keys


class TestAskUserMenu:
    def test_find_menu_logs_when_hierarchy_is_missing(
        self,
        caplog,
    ) -> None:
        """`_find_menu` should warn when no AskUserMenu ancestor exists."""
        question_widget = _QuestionWidget({"question": "Name?", "type": "text"}, 0)
        with caplog.at_level("WARNING", logger="deepagents_code.tui.widgets.ask_user"):
            assert question_widget._find_menu() is None
        assert "Failed to find AskUserMenu ancestor" in caplog.text

    async def test_text_input_receives_focus_on_mount(self) -> None:
        """The text area must have focus after mount so the user can type."""
        app = _AskUserTestApp([{"question": "What is your name?", "type": "text"}])

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            text_input = menu.query_one(".ask-user-text-input", AskUserTextArea)
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

    async def test_multiple_choice_option_wraps_in_narrow_menu(self) -> None:
        """Long choice labels should wrap instead of being clipped to one row."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick one",
                    "type": "multiple_choice",
                    "choices": [
                        {
                            "value": (
                                "this option label is intentionally long enough "
                                "to wrap in a narrow ask user menu"
                            )
                        }
                    ],
                }
            ]
        )

        async with app.run_test(size=(36, 24)) as pilot:
            await pilot.pause()
            choice = app.query_one(".ask-user-choice", Static)
            assert choice.size.height > 1

    async def test_text_question_submits_typed_answer(self) -> None:
        app = _AskUserTestApp([{"question": "What is your name?", "type": "text"}])

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            future: asyncio.Future[AskUserWidgetResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.pause()
            text_input = menu.query_one(".ask-user-text-input", AskUserTextArea)
            text_input.text = "Alice"
            text_input.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {"type": "answered", "answers": ["Alice"]}

    async def test_text_answer_expands_collapsed_paste(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A collapsed paste in a text answer expands in the submitted value."""
        from deepagents_code.tui.widgets import _paste_textarea as paste_textarea_module

        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: True
        )
        app = _AskUserTestApp([{"question": "Paste config?", "type": "text"}])

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            future: asyncio.Future[AskUserWidgetResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.pause()
            text_input = menu.query_one(".ask-user-text-input", AskUserTextArea)
            text_input.focus()
            big = "key=value\n" * 5
            await text_input._on_paste(events.Paste(big))
            await pilot.pause()
            assert text_input.text == "[Pasted text #1 +5 lines]"

            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {"type": "answered", "answers": [big]}

    async def test_text_input_soft_wraps_long_answers(self) -> None:
        """Soft-wrap is enabled so long answers wrap visually without newlines."""
        app = _AskUserTestApp([{"question": "Describe?", "type": "text"}])

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            text_input = menu.query_one(".ask-user-text-input", AskUserTextArea)
            assert text_input.soft_wrap is True

    async def test_enter_submits_without_inserting_newline(self) -> None:
        """Enter submits the answer instead of inserting a newline."""
        app = _AskUserTestApp([{"question": "Describe?", "type": "text"}])

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            future: asyncio.Future[AskUserWidgetResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.pause()
            text_input = menu.query_one(".ask-user-text-input", AskUserTextArea)
            text_input.text = "hi"
            text_input.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {"type": "answered", "answers": ["hi"]}

    async def test_shift_enter_inserts_newline_for_multiline_answer(self) -> None:
        """Shift+Enter inserts a literal newline for multi-paragraph answers."""
        app = _AskUserTestApp([{"question": "Describe?", "type": "text"}])

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            future: asyncio.Future[AskUserWidgetResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.pause()
            text_input = menu.query_one(".ask-user-text-input", AskUserTextArea)
            text_input.focus()
            await pilot.pause()

            await pilot.press("a")
            await pilot.press("shift+enter")
            await pilot.press("b")
            await pilot.pause()
            assert text_input.text == "a\nb"

            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {"type": "answered", "answers": ["a\nb"]}

    async def test_escape_cancels_and_resolves_future(self) -> None:
        app = _AskUserTestApp([{"question": "Name?", "type": "text"}])

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            future: asyncio.Future[AskUserWidgetResult] = (
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
            future: asyncio.Future[AskUserWidgetResult] = (
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
            future: asyncio.Future[AskUserWidgetResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.pause()
            await pilot.press("down")
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            other_input = menu.query_one(".ask-user-other-input", AskUserTextArea)
            other_input.text = "green"
            other_input.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {"type": "answered", "answers": ["green"]}

    async def test_multiple_choice_other_expands_collapsed_paste(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A collapsed paste in the "other" free-text answer expands on submit."""
        from deepagents_code.tui.widgets import _paste_textarea as paste_textarea_module

        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: True
        )
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
            future: asyncio.Future[AskUserWidgetResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.pause()
            await pilot.press("down")
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            other_input = menu.query_one(".ask-user-other-input", AskUserTextArea)
            other_input.focus()
            big = "detail\n" * 5
            await other_input._on_paste(events.Paste(big))
            await pilot.pause()
            assert other_input.text == "[Pasted text #1 +5 lines]"

            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {"type": "answered", "answers": [big]}

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
            future: asyncio.Future[AskUserWidgetResult] = (
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
            text_input = menu.query_one(".ask-user-text-input", AskUserTextArea)
            assert text_input.has_focus
            assert not future.done(), "Should not submit yet"

            # Type answer for Q3 and submit
            text_input.text = "Alice"
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

    async def test_clicking_text_question_moves_active_highlight(self) -> None:
        """Clicking a dimmed text question makes it the active/highlighted one."""
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

            second_input = qw1.query_one(".ask-user-text-input", AskUserTextArea)
            await pilot.click(second_input)
            await pilot.pause()

            assert menu._current_question == 1
            assert qw1.has_class("ask-user-question-active")
            assert qw0.has_class("ask-user-question-inactive")
            assert not qw0.has_class("ask-user-question-active")
            assert second_input.has_focus

    async def test_clicking_multiple_choice_question_moves_active_highlight(
        self,
    ) -> None:
        """Clicking a dimmed multiple-choice question highlights it."""
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
                    "choices": [{"value": "S"}, {"value": "M"}],
                },
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            qw0 = menu._question_widgets[0]
            qw1 = menu._question_widgets[1]
            assert qw0.has_class("ask-user-question-active")

            await pilot.click(qw1)
            await pilot.pause()

            assert menu._current_question == 1
            assert qw1.has_focus
            assert qw1.has_class("ask-user-question-active")
            assert qw0.has_class("ask-user-question-inactive")

    async def test_focus_sync_does_not_steal_focus_from_clicked_widget(self) -> None:
        """Syncing the highlight must not move focus off the clicked question."""
        app = _AskUserTestApp(
            [
                {"question": "Q1?", "type": "text"},
                {"question": "Q2?", "type": "text"},
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            qw1 = menu._question_widgets[1]
            second_input = qw1.query_one(".ask-user-text-input", AskUserTextArea)

            await pilot.click(second_input)
            await pilot.pause()

            # Typing lands in the clicked question's input, not the first one.
            await pilot.press("h", "i")
            await pilot.pause()
            assert second_input.text == "hi"
            first_input = menu._question_widgets[0].query_one(
                ".ask-user-text-input", AskUserTextArea
            )
            assert first_input.text == ""

    async def test_focus_sync_does_not_steal_focus_from_clicked_mc_question(
        self,
    ) -> None:
        """Clicking a dimmed MC question keeps focus there for choice nav.

        Multiple-choice questions take focus at the container level (their
        `_ChoiceOption`s are not focusable), so this exercises the widget-level
        focus path that the text-input focus-steal test does not.
        """
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
                    "choices": [{"value": "S"}, {"value": "M"}],
                },
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            qw1 = menu._question_widgets[1]

            await pilot.click(qw1)
            await pilot.pause()
            assert qw1.has_focus

            # Arrow keys drive the clicked question's choice list, not Q1's.
            await pilot.press("down")
            await pilot.pause()
            assert qw1._selected_choice == 1
            assert menu._question_widgets[0]._selected_choice == 0

    async def test_clicking_already_active_question_is_noop(self) -> None:
        """Clicking the currently active question leaves it active and focused."""
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
            first_input = qw0.query_one(".ask-user-text-input", AskUserTextArea)
            assert qw0.has_class("ask-user-question-active")

            # `node._index == _current_question`, so the handler short-circuits.
            await pilot.click(first_input)
            await pilot.pause()

            assert menu._current_question == 0
            assert qw0.has_class("ask-user-question-active")
            assert not qw0.has_class("ask-user-question-inactive")
            assert first_input.has_focus

    async def test_clicking_title_refocuses_active_question(self) -> None:
        """Clicking non-question chrome must not strand focus on the bare menu.

        Textual's focus-on-click lands focus on the menu container when the
        click hits the title, help text, or padding, and the container has no
        navigation bindings. The menu re-routes focus to the active question so
        arrow keys keep paging through choices.
        """
        app = _AskUserTestApp(
            [
                {
                    "question": "Color?",
                    "type": "multiple_choice",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                }
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            qw0 = menu._question_widgets[0]

            await pilot.click(".ask-user-title")
            await pilot.pause()
            assert qw0.has_focus
            assert app.focused is not menu

            # Arrow keys still drive the active question's choice list.
            await pilot.press("down")
            await pilot.pause()
            assert qw0._selected_choice == 1

    async def test_border_reflects_focus(self) -> None:
        """The border brightens while focus is within and dims when it leaves."""
        app = _AskUserTestApp([{"question": "Name?", "type": "text"}])

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            assert menu.has_focus_within
            focused_border = menu.styles.border_top

            app.set_focus(None)
            await pilot.pause()
            assert not menu.has_focus_within
            blurred_border = menu.styles.border_top

            # Focus only changes the border color, not its type (solid/ascii).
            assert focused_border != blurred_border
            assert focused_border[0] == blurred_border[0]
            # And it brightens on focus, not the other way around.
            assert blurred_border[1].brightness < focused_border[1].brightness

    async def test_border_uses_ascii_variant_in_ascii_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In ASCII mode the border keeps the ascii type across both focus states.

        The refactor moved the border into CSS so `:focus-within` can restyle it
        even on terminals that cannot draw box characters. Only the color should
        change on focus; the ascii border type must survive.
        """
        monkeypatch.setattr(
            "deepagents_code.tui.widgets._inline_prompt.is_ascii_mode",
            lambda: True,
        )
        app = _AskUserTestApp([{"question": "Name?", "type": "text"}])

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            assert menu.has_class("-ascii")
            assert menu.has_focus_within
            focused_border = menu.styles.border_top

            app.set_focus(None)
            await pilot.pause()
            blurred_border = menu.styles.border_top

            assert focused_border[0] == "ascii"
            assert blurred_border[0] == "ascii"
            # Focus brightens the color but leaves the ascii type intact.
            assert focused_border != blurred_border
            assert blurred_border[1].brightness < focused_border[1].brightness

    async def test_clicking_choice_restores_active_other_input(self) -> None:
        """Clicking a non-focusable choice restores the active Other input.

        With "Other" selected, the question's focus target is its free-text
        input, but a click on a plain choice option (not itself focusable) lands
        focus on the bare question container. `on_click` routes focus back to the
        Other input so typing keeps landing there. This is the switch branch that
        a click landing directly on a question's own focus target skips.
        """
        app = _AskUserTestApp(
            [
                {
                    "question": "Color?",
                    "type": "multiple_choice",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                }
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            qw0 = menu._question_widgets[0]

            # Select "Other" (red -> blue -> Other), revealing its input.
            await pilot.press("down", "down", "enter")
            await pilot.pause()
            other_input = qw0.query_one(".ask-user-other-input", AskUserTextArea)
            assert other_input.has_focus

            # Clicking a plain choice focuses the container; on_click restores it.
            await pilot.click(qw0._choice_widgets[0])
            await pilot.pause()
            assert other_input.has_focus
            assert menu._current_question == 0

    async def test_clicking_chrome_with_focus_in_question_is_noop(self) -> None:
        """Clicking chrome while a question holds focus leaves focus untouched."""
        app = _AskUserTestApp([{"question": "Name?", "type": "text"}])

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            text_input = menu._question_widgets[0].query_one(
                ".ask-user-text-input", AskUserTextArea
            )
            assert text_input.has_focus

            await pilot.click(".ask-user-title")
            await pilot.pause()

            # Focus was already inside a question, so the click did not move it.
            assert text_input.has_focus

    async def test_click_highlight_preserves_confirmed_and_answers(self) -> None:
        """Following focus on click must not confirm or clear existing answers."""
        app = _AskUserTestApp(
            [
                {"question": "Q1?", "type": "text"},
                {"question": "Q2?", "type": "text"},
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            first_input = menu._question_widgets[0].query_one(
                ".ask-user-text-input", AskUserTextArea
            )
            await pilot.press("t", "y", "p", "e", "d")
            await pilot.pause()
            assert first_input.text == "typed"

            second_input = menu._question_widgets[1].query_one(
                ".ask-user-text-input", AskUserTextArea
            )
            await pilot.click(second_input)
            await pilot.pause()

            # The click only moved the highlight: no confirmation, no lost answer.
            assert menu._current_question == 1
            assert menu._confirmed == [False, False]
            assert first_input.text == "typed"

    async def test_other_input_focus_syncs_highlight(self) -> None:
        """Focusing a question's Other input syncs the highlight to that question.

        Covers the free-text `_other_input` focus path (a distinct descendant
        from the plain text input) both when it stays within the active
        question and when the user then clicks a different question.
        """
        app = _AskUserTestApp(
            [
                {
                    "question": "Color?",
                    "type": "multiple_choice",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                },
                {"question": "Name?", "type": "text"},
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            qw0 = menu._question_widgets[0]

            # Select "Other" in Q1 (red -> blue -> Other), revealing its input.
            await pilot.press("down", "down")
            await pilot.press("enter")
            await pilot.pause()

            other_input = qw0.query_one(".ask-user-other-input", AskUserTextArea)
            assert other_input.has_focus
            # Focus is inside Q1, so the highlight stays on Q1 (no-op sync).
            assert menu._current_question == 0
            assert qw0.has_class("ask-user-question-active")

            # Clicking Q2 moves the highlight off the active Other input.
            second_input = menu._question_widgets[1].query_one(
                ".ask-user-text-input", AskUserTextArea
            )
            await pilot.click(second_input)
            await pilot.pause()

            assert menu._current_question == 1
            assert menu._question_widgets[1].has_class("ask-user-question-active")
            assert qw0.has_class("ask-user-question-inactive")
            assert second_input.has_focus

    async def test_focus_outside_any_question_leaves_highlight_unchanged(self) -> None:
        """Focus with no `_QuestionWidget` ancestor is a no-op for the highlight.

        The walk-up in `on_descendant_focus` terminates at `None` when the
        focused widget has no enclosing question. No focusable widget outside a
        question exists in the normal layout, so drive the handler directly to
        guard the walk-up termination and the `node is not None` check against
        regression.
        """
        app = _AskUserTestApp(
            [
                {"question": "Q1?", "type": "text"},
                {"question": "Q2?", "type": "text"},
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            menu._set_active_question(1)
            await pilot.pause()

            # `event.widget` is the menu itself; the walk-up climbs its
            # ancestors (none are `_QuestionWidget`s) and terminates at `None`.
            menu.on_descendant_focus(events.DescendantFocus(menu))

            assert menu._current_question == 1
            assert menu._question_widgets[1].has_class("ask-user-question-active")
            assert menu._question_widgets[0].has_class("ask-user-question-inactive")

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

    async def test_help_text_advertises_newline_shortcut(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Footer advertises the terminal-aware newline shortcut."""
        from deepagents_code import config as config_module

        # `newline_hint` resolves `newline_shortcut` via a call-time import from
        # config, so patch the name on the config module it looks up.
        monkeypatch.setattr(config_module, "newline_shortcut", lambda: "Ctrl+J")
        app = _AskUserTestApp([{"question": "Q1?", "type": "text"}])

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            help_text = menu.query_one(".ask-user-help").render()
            assert "Ctrl+J newline" in str(help_text)

    async def test_required_label_shown_for_required_question(self) -> None:
        """Required questions display a (required) indicator."""
        app = _AskUserTestApp([{"question": "Name?", "type": "text", "required": True}])

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            qw = menu._question_widgets[0]
            md = qw.query_one(Markdown)
            assert "required" in md.source

    async def test_required_label_hidden_for_optional_question(self) -> None:
        """Optional questions do not display a (required) indicator."""
        app = _AskUserTestApp(
            [{"question": "Nickname?", "type": "text", "required": False}]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            qw = menu._question_widgets[0]
            md = qw.query_one(Markdown)
            assert "required" not in md.source

    async def test_required_is_true_by_default(self) -> None:
        """Questions without explicit required field default to required."""
        app = _AskUserTestApp([{"question": "Name?", "type": "text"}])

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            qw = menu._question_widgets[0]
            assert qw._required is True
            md = qw.query_one(Markdown)
            assert "required" in md.source

    async def test_optional_question_submits_with_empty_answer(self) -> None:
        """Non-required questions can be submitted with empty answers."""
        app = _AskUserTestApp(
            [{"question": "Nickname?", "type": "text", "required": False}]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            future: asyncio.Future[AskUserWidgetResult] = (
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
            future: asyncio.Future[AskUserWidgetResult] = (
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
            other_input = menu.query_one(".ask-user-other-input", AskUserTextArea)
            assert other_input.has_focus

            # Single up press should select "blue" (last real choice)
            await pilot.press("up")
            await pilot.pause()
            assert qw._selected_choice == 1
            assert not qw._is_other_selected
            assert qw.has_focus

    async def test_down_in_wrapped_other_input_moves_cursor(self) -> None:
        """Down inside a wrapped Other answer should not leave the text input."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick one",
                    "type": "multiple_choice",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                }
            ]
        )

        async with app.run_test(size=(50, 24)) as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            qw = menu._question_widgets[0]

            await pilot.press("down")
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            other_input = menu.query_one(".ask-user-other-input", AskUserTextArea)
            other_input.text = " ".join(["wrapped"] * 20)
            other_input.move_cursor((0, 0))
            other_input.focus()
            await pilot.pause()

            await pilot.press("down")
            await pilot.pause()

            assert other_input.has_focus
            assert qw._is_other_selected
            assert other_input.cursor_location != (0, 0)

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
            other_input = menu.query_one(".ask-user-other-input", AskUserTextArea)
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

    async def test_cancel_after_submit_does_not_override_answer(self) -> None:
        """Cancel after submit is ignored by the resolve-once completion guard."""
        app = _AskUserTestApp([{"question": "Name?", "type": "text"}])

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            future: asyncio.Future[AskUserWidgetResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.pause()
            text_input = menu.query_one(".ask-user-text-input", AskUserTextArea)
            text_input.text = "Alice"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            menu.action_cancel()
            await pilot.pause()

            assert future.done()
            assert future.result() == {"type": "answered", "answers": ["Alice"]}

    async def test_submit_after_cancel_does_not_override_cancel(self) -> None:
        """Submit after cancel is ignored by the resolve-once completion guard."""
        app = _AskUserTestApp([{"question": "Name?", "type": "text"}])

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            future: asyncio.Future[AskUserWidgetResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.pause()
            menu.action_cancel()
            await pilot.pause()

            menu._submit()
            await pilot.pause()

            assert future.done()
            assert future.result() == {"type": "cancelled"}
