"""Tests for ask_user tool integration in the CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Markdown, Static

import deepagents_code
from deepagents_code._ask_user_types import (
    CHOICE_QUESTION_TYPES,
    QUESTION_TYPES,
    decode_multi_select_answer,
    encode_multi_select_answer,
)
from deepagents_code.config import ASCII_GLYPHS, get_glyphs
from deepagents_code.tui.widgets.ask_user import (
    _TRAILING_ANNOTATION_RE,
    ADD_ANOTHER_OTHER_LABEL,
    MAX_MULTI_SELECT_OTHER_ENTRIES,
    MAX_OTHER_ENTRIES_TOAST,
    MISSING_ANSWER_TOAST,
    MISSING_OTHER_TEXT_TOAST,
    OTHER_CHOICE_LABEL,
    AskUserMenu,
    AskUserTextArea,
    _ChoiceOption,
    _MultiSelectOption,
    _MultiSelectOtherEntry,
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
    async def test_multi_select_other_uses_an_inline_text_field(self) -> None:
        """A checked Other keeps its custom-answer field beside the checkbox."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick a color",
                    "type": "multi_select",
                    "choices": [{"value": "red"}],
                }
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            slot = menu.query_one(".ask-user-other-slot", _MultiSelectOtherEntry)
            other_input = menu.query_one(".ask-user-other-input", AskUserTextArea)

            assert isinstance(slot, Horizontal)

            await pilot.press("down", "space")
            await pilot.pause()

            other_option = list(menu.query(_MultiSelectOption))[1]
            assert other_option.has_class("ask-user-inline-other-choice")
            assert other_input.parent is slot
            assert other_input.display is True
            assert other_option._label_widget is not None
            assert str(other_option._label_widget.render()) == (
                get_glyphs().checkbox_checked
            )
            # Same row, field to the right of the collapsed checkbox. Deliberately
            # not an exact-pixel assertion: the label collapsing to just the box is
            # already pinned above, and padding changes should not break this.
            assert other_input.region.y == other_option.region.y
            assert other_input.region.x >= (
                other_option.region.x + other_option.region.width
            )
            assert other_input.region.height == 1

    def test_find_menu_logs_when_hierarchy_is_missing(
        self,
        caplog,
    ) -> None:
        """`_find_menu` should warn when no AskUserMenu ancestor exists."""
        question_widget = _QuestionWidget({"question": "Name?", "type": "text"}, 0)
        with caplog.at_level("WARNING", logger="deepagents_code.tui.widgets.ask_user"):
            assert question_widget._find_menu() is None
        assert "Failed to find AskUserMenu ancestor" in caplog.text

    def test_unrecognized_type_warns_and_degrades_to_text(self, caplog) -> None:
        """An unknown type must not silently become a free-text box."""
        with caplog.at_level("WARNING", logger="deepagents_code.tui.widgets.ask_user"):
            question_widget = _QuestionWidget(
                cast("Any", {"question": "Pick?", "type": "ranked_select"}), 0
            )

        assert question_widget.question_type == "text"
        assert "unrecognized type" in caplog.text

    def test_choice_question_without_choices_warns_and_degrades_to_text(
        self, caplog
    ) -> None:
        """A choice question with no options degrades its *type*, not just its render.

        The interrupt adapter accepts this payload (`choices` is `NotRequired`),
        so leaving `question_type` as `multi_select` would make the help footer
        advertise "Space toggle" for what is actually a text box.
        """
        with caplog.at_level("WARNING", logger="deepagents_code.tui.widgets.ask_user"):
            question_widget = _QuestionWidget(
                {"question": "Pick?", "type": "multi_select", "choices": []}, 0
            )

        assert question_widget.question_type == "text"
        assert "has no choices" in caplog.text

    async def test_no_question_type_silently_renders_as_bare_text(self) -> None:
        """No `QuestionType` member may render as a text box without warning.

        `_QuestionWidget.compose` dispatches on type literals, so a new member
        added to `QuestionType` would otherwise fall through to the text branch
        while the agent believes it constrained the answer to a choice set. The
        `assert_never` in `compose` catches this at type-check time; this catches
        it at runtime.
        """
        for question_type in sorted(QUESTION_TYPES):
            question: Any = {"question": "Q?", "type": question_type}
            if question_type in CHOICE_QUESTION_TYPES:
                question["choices"] = [{"value": "a"}, {"value": "b"}]
            app = _AskUserTestApp([question])

            async with app.run_test() as pilot:
                menu = app.query_one("#ask-user-menu", AskUserMenu)
                await pilot.pause()
                widget = menu._question_widgets[0]
                renders_choices = bool(widget._choice_widgets)
                assert renders_choices == (question_type in CHOICE_QUESTION_TYPES), (
                    f"{question_type!r} silently rendered as a bare text input"
                )

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
            choice = app.query_one(".ask-user-choice", _ChoiceOption)
            cursor = choice.query_one(".inline-prompt-option-cursor", Static)
            label = choice.query_one(".inline-prompt-option-label", Static)
            assert label.size.height > 1
            assert label.region.x == cursor.region.x + cursor.region.width

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
            # Post through the App so Textual's MRO dispatch reaches the
            # base handlers that perform the insert.
            pilot.app.post_message(events.Paste(big))
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

    async def test_escape_in_custom_multi_select_answer_unchecks_option(self) -> None:
        """Escape deselects the focused custom option before it cancels the prompt."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick a color",
                    "type": "multi_select",
                    "choices": [{"value": "red"}],
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
            await pilot.press("down", "space")
            await pilot.pause()

            question = menu.query_one(_QuestionWidget)
            entry = question._other_entries[0]
            assert entry.checked
            assert entry.text_input.has_focus
            assert "Esc to deselect" in str(menu.query_one(".ask-user-help").render())

            await pilot.press("escape")

            assert not future.done()
            assert not entry.checked
            assert entry.text_input.display is False
            assert question.has_focus
            assert "Esc to cancel" in str(menu.query_one(".ask-user-help").render())

    async def test_custom_multi_select_answer_grows_for_newlines(self) -> None:
        """A custom answer expands after Shift+Enter instead of scrolling one row."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick a color",
                    "type": "multi_select",
                    "choices": [{"value": "red"}],
                }
            ]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            await pilot.pause()
            await pilot.press("down", "space")
            await pilot.pause()

            entry = menu.query_one(_QuestionWidget)._other_entries[0]
            assert entry.text_input.region.height == 1

            await pilot.press("shift+enter")
            await pilot.pause()

            assert entry.text_input.text == "\n"
            assert entry.text_input.region.height == 2

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
            # Post through the App so Textual's MRO dispatch reaches the
            # base handlers that perform the insert.
            pilot.app.post_message(events.Paste(big))
            await pilot.pause()
            assert other_input.text == "[Pasted text #1 +5 lines]"

            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {"type": "answered", "answers": [big]}

    async def test_multi_select_toggles_and_collects_answers(self) -> None:
        """The answer is a JSON array of the selected values, in choice order."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [
                        {"value": "red"},
                        {"value": "blue"},
                        {"value": "green"},
                    ],
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
            # Toggle "red" (index 0), move to "blue" (index 1) and toggle it.
            await pilot.press("space")
            await pilot.press("down")
            await pilot.press("space")
            await pilot.pause()
            # Confirm the selection with Enter.
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {
                "type": "answered",
                "answers": ['["red", "blue"]'],
            }

    async def test_multi_select_required_enter_without_selection_does_not_submit(
        self,
    ) -> None:
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                    "required": True,
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
            # Enter with nothing selected on a required question blocks, and
            # explains why rather than looking like a frozen UI.
            await pilot.press("enter")
            await pilot.pause()
            assert not future.done()
            assert MISSING_ANSWER_TOAST in [n.message for n in app._notifications]

            # Selecting one option then Enter submits.
            await pilot.press("space")
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {"type": "answered", "answers": ['["red"]']}

    async def test_multi_select_get_answer_encodes_a_json_array(self) -> None:
        """`get_answer` returns the JSON encoding, decodable back to the values."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [
                        {"value": "push-to-main — no PR label, always strict"},
                        {"value": "blue"},
                    ],
                }
            ]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            question = menu.query_one(_QuestionWidget)
            await pilot.pause()

            await pilot.press("space")
            await pilot.pause()

            answer = question.get_answer()
            assert answer == '["push-to-main — no PR label, always strict"]'
            assert decode_multi_select_answer(answer) == [
                "push-to-main — no PR label, always strict"
            ]

    async def test_empty_multi_select_answer_is_empty_for_validation(self) -> None:
        """`"[]"` is truthy, so a raw `.strip()` would wrongly pass required.

        `answer_is_empty` decodes the answer instead, so an untouched required
        multi-select still fails the submit check.
        """
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}],
                    "required": True,
                }
            ]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            question = menu.query_one(_QuestionWidget)
            await pilot.pause()

            assert question.get_answer() == "[]"
            assert question.answer_is_empty()
            assert question.validate_for_submit() == MISSING_ANSWER_TOAST

    async def test_a_text_answer_of_literal_brackets_is_not_empty(self) -> None:
        """Emptiness is routed by question type, not by matching the encoding.

        `[]` is only the empty marker for `multi_select`. Typed into a text
        question it is a real answer, so a required question must accept it.
        """
        app = _AskUserTestApp(
            [{"question": "Paste it", "type": "text", "required": True}]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            question = menu.query_one(_QuestionWidget)
            await pilot.pause()
            await pilot.press("[", "]")
            await pilot.pause()

            assert question.get_answer() == "[]"
            assert not question.answer_is_empty()
            assert question.validate_for_submit() is None

    async def test_multi_select_untoggle_clears_choice(self) -> None:
        """Space is a toggle: pressing it twice deselects the choice again."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
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
            # Check both, then un-check "blue" again.
            await pilot.press("space")
            await pilot.press("down")
            await pilot.press("space")
            await pilot.press("space")
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {"type": "answered", "answers": ['["red"]']}

    async def test_multi_select_required_blocks_after_untoggling_all(self) -> None:
        """Un-toggling the last choice re-blocks a required question."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                    "required": True,
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
            await pilot.press("space")
            await pilot.press("space")
            await pilot.press("enter")
            await pilot.pause()

            assert not future.done()
            assert MISSING_ANSWER_TOAST in [n.message for n in app._notifications]

    async def test_multi_select_answer_uses_choice_order_not_toggle_order(self) -> None:
        """Answers list values in choice-list order regardless of toggle order."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [
                        {"value": "red"},
                        {"value": "blue"},
                        {"value": "green"},
                    ],
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
            # Toggle "green" (index 2) first, then "red" (index 0).
            await pilot.press("down")
            await pilot.press("down")
            await pilot.press("space")
            await pilot.press("up")
            await pilot.press("up")
            await pilot.press("space")
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {
                "type": "answered",
                "answers": ['["red", "green"]'],
            }

    async def test_multi_select_optional_submits_empty_answer(self) -> None:
        """An optional multi-select with nothing toggled submits an empty array."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                    "required": False,
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
            assert future.result() == {"type": "answered", "answers": ["[]"]}
            assert MISSING_ANSWER_TOAST not in [n.message for n in app._notifications]

    async def test_multi_select_toggle_glyphs_survive_cursor_move(self) -> None:
        """The toggle glyph tracks `checked`, independent of the highlight cursor."""
        glyphs = get_glyphs()
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                }
            ]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            await pilot.pause()

            options = list(menu.query(_MultiSelectOption))
            # Predefined choices plus the automatic Other row.
            assert [o.checked for o in options] == [False, False, False]
            assert options[2]._text == OTHER_CHOICE_LABEL

            await pilot.press("space")
            await pilot.pause()
            assert [o.checked for o in options] == [True, False, False]
            assert glyphs.checkbox_checked in str(
                options[0].query_one(".inline-prompt-option-label", Static).render()
            )
            assert glyphs.checkbox_empty in str(
                options[1].query_one(".inline-prompt-option-label", Static).render()
            )

            # Moving the cursor off a checked option must not clear its glyph.
            await pilot.press("down")
            await pilot.pause()
            assert [o.checked for o in options] == [True, False, False]
            assert glyphs.checkbox_checked in str(
                options[0].query_one(".inline-prompt-option-label", Static).render()
            )

    async def test_multi_select_other_combines_with_predefined_choices(self) -> None:
        """Other free-text is appended after toggled predefined values."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
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
            question = menu.query_one(_QuestionWidget)

            await pilot.pause()
            # Toggle "red", move to Other, toggle it, type custom text, submit.
            await pilot.press("space")
            await pilot.press("down")
            await pilot.press("down")
            await pilot.press("space")
            await pilot.pause()

            other_input = menu.query_one(".ask-user-other-input", AskUserTextArea)
            assert other_input.display is True
            other_input.text = "teal"
            question.sync_other_slots()
            other_input.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {
                "type": "answered",
                "answers": ['["red", "teal"]'],
            }

    async def test_multi_select_other_alone_submits_custom_text(self) -> None:
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
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
            question = menu.query_one(_QuestionWidget)

            await pilot.pause()
            await pilot.press("down")
            await pilot.press("down")
            await pilot.press("space")
            await pilot.pause()

            other_input = menu.query_one(".ask-user-other-input", AskUserTextArea)
            other_input.text = "purple"
            question.sync_other_slots()
            other_input.focus()
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {"type": "answered", "answers": ['["purple"]']}

    async def test_typing_on_highlighted_multi_select_other_activates_it(self) -> None:
        """The first typed character toggles and populates a highlighted Other row."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                }
            ]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            await pilot.pause()
            await pilot.press("down")
            await pilot.press("down")
            await pilot.press("p")
            await pilot.pause()

            other_entry = menu._question_widgets[0]._other_entries[0]
            assert other_entry.checked
            assert other_entry.text_input.has_focus
            assert other_entry.text_input.text == "p"

    async def test_multi_select_multiple_others_grow_and_collect(self) -> None:
        """Filling one Other reveals an Add-another slot for more custom values."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
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
            question = menu.query_one(_QuestionWidget)

            await pilot.pause()
            options = list(menu.query(_MultiSelectOption))
            assert len(options) == 3
            assert options[2]._text == OTHER_CHOICE_LABEL

            # Toggle first Other and fill its free-text value.
            await pilot.press("down")
            await pilot.press("down")
            await pilot.press("space")
            await pilot.pause()
            first_other = menu.query_one(".ask-user-other-input", AskUserTextArea)
            first_other.text = "teal"
            question.sync_other_slots()
            await pilot.pause()

            options = list(menu.query(_MultiSelectOption))
            assert len(options) == 4
            assert options[3]._text == ADD_ANOTHER_OTHER_LABEL
            other_inputs = list(
                menu.query(AskUserTextArea).filter(".ask-user-other-input")
            )
            assert len(other_inputs) == 2

            # Leave the first free-text with Up, then toggle/fill Add another.
            await pilot.press("up")
            await pilot.press("down")
            await pilot.press("space")
            await pilot.pause()
            other_inputs = list(
                menu.query(AskUserTextArea).filter(".ask-user-other-input")
            )
            second_other = other_inputs[1]
            assert second_other.display is True
            second_other.text = "cyan"
            question.sync_other_slots()
            second_other.focus()
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {
                "type": "answered",
                "answers": ['["teal", "cyan"]'],
            }
            # Exactly one spare empty Add-another row remains mounted after the
            # second custom is filled (2 choices + 3 Other rows); it must not
            # contribute to the answer.
            assert len(list(menu.query(_MultiSelectOption))) == 5
            assert len(question._other_entries) == 3

    async def test_multi_select_other_pruning_removes_empty_slot(self) -> None:
        """Clearing Other text removes its spare slot and wrapper."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}],
                }
            ]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            question = menu.query_one(_QuestionWidget)
            await pilot.pause()

            first_entry = question._other_entries[0]
            first_entry.option.set_checked(True)
            for custom_text in ("teal", "cyan"):
                first_entry.text_input.text = custom_text
                question.sync_other_slots()
                await pilot.pause()
                assert len(list(menu.query(".ask-user-other-slot"))) == 2

                first_entry.text_input.text = ""
                question.sync_other_slots()
                await pilot.pause()
                assert len(list(menu.query(".ask-user-other-slot"))) == 1

    async def test_multi_select_other_requires_text_when_checked(self) -> None:
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                    "required": False,
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
            # Checked Other with no custom text is incomplete, even when optional.
            await pilot.press("down")
            await pilot.press("down")
            await pilot.press("space")
            await pilot.press("enter")
            await pilot.pause()

            assert not future.done()
            assert MISSING_OTHER_TEXT_TOAST in [n.message for n in app._notifications]

    async def test_multi_select_other_accepts_comma_in_custom_text(self) -> None:
        """A comma in a custom answer is one value, not a fake second selection."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
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
            await pilot.press("space")
            await pilot.pause()

            other_input = menu.query_one(".ask-user-other-input", AskUserTextArea)
            other_input.text = "teal, cyan"
            other_input.focus()
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {
                "type": "answered",
                "answers": ['["teal, cyan"]'],
            }

    async def test_untoggling_confirmed_multi_select_reopens_with_toast(self) -> None:
        """Clearing an already-confirmed required question explains the bounce."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                    "required": True,
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
            # Confirm "red", advancing to the text question.
            await pilot.press("space")
            await pilot.press("enter")
            await pilot.pause()

            # Navigate back and un-toggle the only selection.
            menu.action_previous_question()
            await pilot.pause()
            await pilot.press("space")
            await pilot.pause()

            # Answer the text question and submit; the empty required
            # multi-select must re-open with an explanation.
            menu.action_next_question()
            await pilot.pause()
            text_input = menu.query_one(".ask-user-text-input", AskUserTextArea)
            text_input.text = "Alice"
            await pilot.press("enter")
            await pilot.pause()

            assert not future.done()
            assert MISSING_ANSWER_TOAST in [n.message for n in app._notifications]

            # The bounce must be recoverable, not a dead end: re-toggle and
            # confirm, and the prompt submits.
            await pilot.press("space")
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {
                "type": "answered",
                "answers": ['["red"]', "Alice"],
            }

    async def test_comma_edited_into_a_confirmed_custom_answer_submits_verbatim(
        self,
    ) -> None:
        """A comma edited into a confirmed custom answer stays one value.

        Answers are read live, so confirming question 1 is not a promise that it
        still holds. `focus_input` drops the cursor straight back into a checked
        Other field, so adding a comma afterwards takes no unusual input — and
        with the JSON encoding the agent receives one `"teal, cyan"` value, not
        two selections.
        """
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}, {"value": "blue"}],
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

            # Check Other on Q1 and confirm it with a clean custom value.
            await pilot.pause()
            await pilot.press("down", "down", "space")
            await pilot.pause()
            other_input = menu.query_one(".ask-user-other-input", AskUserTextArea)
            other_input.text = "teal"
            other_input.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert menu._current_question == 1

            # Back into the confirmed question; add a comma without re-confirming.
            menu.action_previous_question()
            await pilot.pause()
            assert other_input.has_focus
            other_input.text = "teal, cyan"

            # Answer Q2 and submit; the comma must survive verbatim.
            menu.action_next_question()
            await pilot.pause()
            text_input = menu.query_one(".ask-user-text-input", AskUserTextArea)
            text_input.text = "Alice"
            text_input.focus()
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {
                "type": "answered",
                "answers": ['["teal, cyan"]', "Alice"],
            }

    async def test_clearing_a_confirmed_custom_answer_blocks_submit(self) -> None:
        """A checked-but-emptied Other must bounce, not vanish from the answer.

        `get_answer` skips a checked Other with no text, so without the final
        re-validation the selection the user explicitly checked would silently
        disappear — and the sibling "red" selection keeps the answer non-empty,
        so the empty-required check would not catch it either.
        """
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}, {"value": "blue"}],
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

            # Check "red" and Other, give Other a value, and confirm.
            await pilot.pause()
            await pilot.press("space")
            await pilot.press("down", "down", "space")
            await pilot.pause()
            other_input = menu.query_one(".ask-user-other-input", AskUserTextArea)
            other_input.text = "teal"
            other_input.focus()
            await pilot.press("enter")
            await pilot.pause()

            # Empty the custom text again, leaving its box checked.
            menu.action_previous_question()
            await pilot.pause()
            other_input.text = ""

            menu.action_next_question()
            await pilot.pause()
            text_input = menu.query_one(".ask-user-text-input", AskUserTextArea)
            text_input.text = "Alice"
            text_input.focus()
            await pilot.press("enter")
            await pilot.pause()

            assert not future.done()
            assert MISSING_OTHER_TEXT_TOAST in [n.message for n in app._notifications]
            assert menu._current_question == 0

    async def test_unconfirmed_comma_answer_submits_from_another_question(
        self,
    ) -> None:
        """Tabbing past a valid question keeps its answer in the submit.

        `action_next_question` deliberately does no validation, and confirming
        the *last* question re-collects every answer — including one the user
        never confirmed. A comma in that unconfirmed custom answer is now just
        content, so the submit goes through with it intact.
        """
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}, {"value": "blue"}],
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
            await pilot.press("down", "down", "space")
            await pilot.pause()
            other_input = menu.query_one(".ask-user-other-input", AskUserTextArea)
            other_input.text = "teal, cyan"

            # Never confirm Q1 — Tab straight past it.
            menu.action_next_question()
            await pilot.pause()
            text_input = menu.query_one(".ask-user-text-input", AskUserTextArea)
            text_input.text = "Alice"
            text_input.focus()
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {
                "type": "answered",
                "answers": ['["teal, cyan"]', "Alice"],
            }

    async def test_typing_a_custom_answer_reveals_the_next_slot(self) -> None:
        """Real keystrokes must grow the slots, not only an explicit sync call.

        Pins the `AskUserMenu.on_text_area_changed` -> `sync_other_slots` wiring:
        with the handler removed, slots would grow only on Enter, and every test
        that calls `sync_other_slots()` by hand would still pass.
        """
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                }
            ]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            question = menu.query_one(_QuestionWidget)

            await pilot.pause()
            await pilot.press("down", "down", "space")
            await pilot.pause()
            assert len(question._other_entries) == 1
            assert question._other_entries[0].text_input.has_focus

            await pilot.press("t", "e", "a", "l")
            await pilot.pause()

            assert question._other_entries[0].text_input.text == "teal"
            assert len(question._other_entries) == 2
            options = list(menu.query(_MultiSelectOption))
            assert options[-1]._text == ADD_ANOTHER_OTHER_LABEL

    async def test_custom_answers_stop_growing_at_the_cap_and_say_so(self) -> None:
        """The cap is explained once, rather than silently withholding the row.

        The prompt advertises that filling one custom answer reveals another, so
        a row that stops appearing with no message reads as a broken UI.
        """
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}],
                }
            ]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            question = menu.query_one(_QuestionWidget)
            await pilot.pause()

            for i in range(MAX_MULTI_SELECT_OTHER_ENTRIES):
                entry = question._other_entries[i]
                entry.set_checked(True)
                entry.text_input.text = f"custom {i}"
                question.sync_other_slots()
                await pilot.pause()

            assert len(question._other_entries) == MAX_MULTI_SELECT_OTHER_ENTRIES
            messages = [n.message for n in app._notifications]
            assert MAX_OTHER_ENTRIES_TOAST in messages

            # Notified once: this runs on every keystroke in the last slot.
            question.sync_other_slots()
            question.sync_other_slots()
            await pilot.pause()
            assert [n.message for n in app._notifications].count(
                MAX_OTHER_ENTRIES_TOAST
            ) == 1
            assert len(question._other_entries) == MAX_MULTI_SELECT_OTHER_ENTRIES

    async def test_multi_select_other_expands_collapsed_paste(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A collapsed paste in a custom multi-select answer expands on submit.

        `get_answer` and `validate_for_submit` must agree on reading
        `submitted_value`: if one read raw `.text` instead, the placeholder
        token would land in the answer instead of the pasted content.
        """
        from deepagents_code.tui.widgets import _paste_textarea as paste_textarea_module

        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: True
        )
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}],
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
            await pilot.press("down", "space")
            await pilot.pause()
            other_input = menu.query_one(".ask-user-other-input", AskUserTextArea)
            other_input.focus()
            big = "detail\n" * 5
            pilot.app.post_message(events.Paste(big))
            await pilot.pause()
            assert other_input.text == "[Pasted text #1 +5 lines]"

            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            # Stripped, unlike the `multiple_choice` Other path: a custom value is
            # one item in the JSON array, and the same strip is what makes
            # "this slot is filled" decidable for slot growth.
            assert future.result() == {
                "type": "answered",
                "answers": [encode_multi_select_answer([big.strip()])],
            }

    async def test_pasted_comma_in_a_custom_answer_passes_validation(self) -> None:
        """Validation reads the expanded paste and finds nothing to reject."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}],
                }
            ]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            future: asyncio.Future[AskUserWidgetResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)
            question = menu.query_one(_QuestionWidget)

            await pilot.pause()
            await pilot.press("down", "space")
            await pilot.pause()
            other_input = menu.query_one(".ask-user-other-input", AskUserTextArea)
            other_input.text = "teal, cyan"

            assert question.validate_for_submit() is None
            other_input.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert future.done()
            assert future.result() == {
                "type": "answered",
                "answers": ['["teal, cyan"]'],
            }

    async def test_return_to_multi_select_other_refocuses_input(self) -> None:
        """Tab away from a checked custom answer and Shift+Tab back refocuses it."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}],
                },
                {"question": "Name?", "type": "text"},
            ]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            question = menu.query_one(_QuestionWidget)

            await pilot.pause()
            await pilot.press("down", "space")
            await pilot.pause()
            other_input = question._other_entries[0].text_input
            assert other_input.has_focus

            menu.action_next_question()
            await pilot.pause()
            assert menu._current_question == 1
            assert not other_input.has_focus

            menu.action_previous_question()
            await pilot.pause()
            assert menu._current_question == 0
            assert other_input.has_focus

    async def test_custom_answer_duplicating_a_selection_is_dropped(self) -> None:
        """The agent must not see the same selection twice."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
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
            # Check "red", then type "red" again as a custom answer.
            await pilot.press("space")
            await pilot.press("down", "down", "space")
            await pilot.pause()
            other_input = menu.query_one(".ask-user-other-input", AskUserTextArea)
            other_input.text = "red"
            other_input.focus()
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {"type": "answered", "answers": ['["red"]']}

    async def test_multiline_custom_answer_keeps_its_newline(self) -> None:
        """A newline inside a custom value is preserved, not flattened.

        Pins the contract: no punctuation is forbidden, because the JSON array
        is self-delimiting. A newline survives inside its JSON string escape in
        the `A:` transcript block, and decodes back to the literal value.
        """
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}],
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
            await pilot.press("down", "space")
            await pilot.pause()
            other_input = menu.query_one(".ask-user-other-input", AskUserTextArea)
            other_input.text = "line one\nline two"
            other_input.focus()
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {
                "type": "answered",
                "answers": ['["line one\\nline two"]'],
            }

    async def test_out_of_range_cursor_refuses_to_toggle(self, caplog) -> None:
        """A corrupt cursor must not flip a checkbox the user never highlighted."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                }
            ]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            question = menu.query_one(_QuestionWidget)
            await pilot.pause()

            before = [w.checked for w in question._multi_select_widgets]
            question._selected_choice = 99
            with caplog.at_level(
                "ERROR", logger="deepagents_code.tui.widgets.ask_user"
            ):
                question.action_toggle_choice()

            assert [w.checked for w in question._multi_select_widgets] == before
            assert not any(e.checked for e in question._other_entries)
            assert "outside" in caplog.text

    async def test_help_text_includes_newline_hint_for_multi_select_only(self) -> None:
        """Multi-select owns an Other free-text input, so the newline hint stays."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                }
            ]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            await pilot.pause()
            help_text = str(menu.query_one(".ask-user-help").render())
            assert "Space toggle" in help_text
            assert "Enter to continue" in help_text
            assert "newline" in help_text

    async def test_help_text_keeps_newline_hint_when_text_question_present(
        self,
    ) -> None:
        """A mixed prompt advertises the newline shortcut and keeps Enter's hint.

        "Space toggle" is additive rather than replacing "Enter to continue":
        the text question in this prompt still continues on Enter, so dropping
        that hint would describe the wrong keys while it is active.
        """
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                },
                {"question": "Name?", "type": "text"},
            ]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            await pilot.pause()
            help_text = str(menu.query_one(".ask-user-help").render())
            assert "Space toggle" in help_text
            assert "Enter to continue" in help_text
            assert "newline" in help_text

    async def test_help_text_omits_space_toggle_without_multi_select(self) -> None:
        """A prompt with no multi-select must not advertise Space as a toggle."""
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
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            await pilot.pause()
            help_text = str(menu.query_one(".ask-user-help").render())
            assert "Space toggle" not in help_text
            assert "Enter to continue" in help_text

    async def test_space_still_types_in_text_question(self) -> None:
        """The multi-select `space` binding must not swallow spaces in text input."""
        app = _AskUserTestApp(
            [
                {"question": "Name?", "type": "text"},
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}],
                },
            ]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            await pilot.pause()
            await pilot.press("a")
            await pilot.press("space")
            await pilot.press("b")
            await pilot.pause()

            text_input = menu.query_one(".ask-user-text-input", AskUserTextArea)
            assert text_input.text == "a b"

    async def test_space_is_not_consumed_on_multiple_choice(self) -> None:
        """`check_action` must leave `space` unbound where nothing can toggle.

        Otherwise the multi-select binding silently swallows the key on
        single-choice questions, whose container holds focus directly.
        """
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
            await pilot.pause()
            question = menu._question_widgets[0]
            assert question.check_action("toggle_choice", ()) is None
            # Still bound where it does something.
            assert question.check_action("select_or_submit", ()) is not None

    async def test_multi_select_renders_ascii_checkboxes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ASCII `[x]`/`[ ]` boxes must survive Rich-markup rendering.

        `[x]` is valid Textual markup, so passing it as a `Content.from_markup`
        substitution (rather than interpolating it into the template) is what
        keeps the box visible. CI resolves the Unicode glyphs, so without this
        the ASCII branch is never exercised.
        """
        # The checkbox comes from ask_user's `_label_content`, but the cursor
        # gutter renders through `_inline_prompt`'s own `get_glyphs` import.
        monkeypatch.setattr(
            "deepagents_code.tui.widgets.ask_user.get_glyphs",
            lambda: ASCII_GLYPHS,
        )
        monkeypatch.setattr(
            "deepagents_code.tui.widgets._inline_prompt.get_glyphs",
            lambda: ASCII_GLYPHS,
        )
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                }
            ]
        )

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            await pilot.pause()
            options = list(menu.query(_MultiSelectOption))

            await pilot.press("space")
            await pilot.pause()

            def rendered(option: _MultiSelectOption, css_class: str) -> str:
                return str(option.query_one(f".{css_class}", Static).render())

            cursor_cls = "inline-prompt-option-cursor"
            label_cls = "inline-prompt-option-label"
            assert rendered(options[0], cursor_cls) == f"{ASCII_GLYPHS.cursor} "
            assert rendered(options[0], label_cls) == "[x] red"
            assert rendered(options[1], cursor_cls) != ASCII_GLYPHS.cursor
            assert rendered(options[1], label_cls) == "[ ] blue"

    async def test_multi_select_mixed_with_other_question_types(self) -> None:
        app = _AskUserTestApp(
            [
                {
                    "question": "Toppings?",
                    "type": "multi_select",
                    "choices": [{"value": "cheese"}, {"value": "olives"}],
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
            # Q1: toggle both choices, confirm, advancing to the text question.
            await pilot.press("space")
            await pilot.press("down")
            await pilot.press("space")
            await pilot.press("enter")
            await pilot.pause()

            text_input = menu.query_one(".ask-user-text-input", AskUserTextArea)
            assert text_input.has_focus
            text_input.text = "Alice"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert future.done()
            assert future.result() == {
                "type": "answered",
                "answers": ['["cheese", "olives"]', "Alice"],
            }

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

    async def test_help_text_personalizes_editor_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Footer names the configured editor while a text field holds focus."""
        monkeypatch.setenv("VISUAL", "nvim")
        app = _AskUserTestApp([{"question": "Q1?", "type": "text"}])

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            assert isinstance(app.focused, AskUserTextArea)
            help_text = menu.query_one(".ask-user-help").render()
            assert "Ctrl+G edit in nvim" in str(help_text)

    async def test_help_text_uses_generic_editor_hint_without_configuration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VISUAL", raising=False)
        monkeypatch.delenv("EDITOR", raising=False)
        app = _AskUserTestApp([{"question": "Q1?", "type": "text"}])

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            help_text = menu.query_one(".ask-user-help").render()
            assert "Ctrl+G external editor" in str(help_text)

    async def test_help_text_shows_editor_hint_for_choiceless_multiple_choice(
        self,
    ) -> None:
        """A multiple_choice question with no choices renders a text field."""
        app = _AskUserTestApp(
            [{"question": "Pick one", "type": "multiple_choice", "choices": []}]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            help_text = menu.query_one(".ask-user-help").render()
            assert "Ctrl+G" in str(help_text)

    async def test_help_text_omits_editor_hint_for_multiple_choice(self) -> None:
        """Footer omits Ctrl+G when only choices (no free-text field) are shown."""
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
            help_text = menu.query_one(".ask-user-help").render()
            assert "Ctrl+G" not in str(help_text)

    async def test_help_text_shows_editor_hint_when_other_selected(self) -> None:
        """Landing on Other reveals and focuses its field, enabling Ctrl+G."""
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

            # Navigate down to the "Other" option, revealing its text field.
            await pilot.press("down")
            await pilot.press("down")
            await pilot.pause()

            other_input = menu._question_widgets[0]._other_input
            assert other_input is not None
            assert other_input.display is True
            assert app.focused is other_input
            help_text = menu.query_one(".ask-user-help").render()
            assert "Ctrl+G" in str(help_text)

    async def test_help_text_hides_editor_hint_when_leaving_other(self) -> None:
        """Moving off Other hides its field and retracts the Ctrl+G hint."""
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

            await pilot.press("down")
            await pilot.press("down")
            await pilot.pause()
            assert "Ctrl+G" in str(menu.query_one(".ask-user-help").render())

            # Back up to the last real choice: no free-text field is focused.
            await pilot.press("up")
            await pilot.pause()

            other_input = menu._question_widgets[0]._other_input
            assert other_input is not None
            assert other_input.display is False
            assert "Ctrl+G" not in str(menu.query_one(".ask-user-help").render())

    async def test_help_text_omits_editor_hint_when_other_field_unfocused(self) -> None:
        """A visible but unfocused Other field must not advertise Ctrl+G.

        `App.action_open_editor` routes to an ask-user text area only while one
        is focused, and otherwise opens the chat draft, so the hint has to
        track focus rather than field visibility.
        """
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

            await pilot.press("down")
            await pilot.press("down")
            await pilot.pause()
            assert "Ctrl+G" in str(menu.query_one(".ask-user-help").render())

            # Mirrors a click landing on the question container: the Other
            # field stays visible, but focus moves off it.
            menu.focus()
            await pilot.pause()

            other_input = menu._question_widgets[0]._other_input
            assert other_input is not None
            assert other_input.display is True
            assert app.focused is not other_input
            assert "Ctrl+G" not in str(menu.query_one(".ask-user-help").render())

    async def test_help_text_hides_editor_hint_when_focus_leaves_menu(self) -> None:
        """Focus moving off the menu entirely retracts the Ctrl+G hint."""
        app = _AskUserTestApp([{"question": "Q1?", "type": "text"}])

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            assert "Ctrl+G" in str(menu.query_one(".ask-user-help").render())

            app.set_focus(None)
            await pilot.pause()

            assert "Ctrl+G" not in str(menu.query_one(".ask-user-help").render())

    async def test_help_text_editor_hint_follows_clicked_question(self) -> None:
        """Clicking into another question's text field turns the hint on."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick one",
                    "type": "multiple_choice",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                },
                {"question": "Why?", "type": "text"},
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            assert "Ctrl+G" not in str(menu.query_one(".ask-user-help").render())

            text_input = menu._question_widgets[1]._text_input
            assert text_input is not None
            await pilot.click(text_input)
            await pilot.pause()

            assert app.focused is text_input
            assert "Ctrl+G" in str(menu.query_one(".ask-user-help").render())

    async def test_help_text_editor_hint_follows_active_question(self) -> None:
        """Mixed prompts only advertise Ctrl+G for the active free-text field."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick one",
                    "type": "multiple_choice",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                },
                {"question": "Why?", "type": "text"},
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)

            # Initial focus is the multiple-choice question: no free-text field.
            help_text = menu.query_one(".ask-user-help").render()
            assert "Ctrl+G" not in str(help_text)

            # Move to the text question: the focused field can use Ctrl+G.
            menu.action_next_question()
            await pilot.pause()
            help_text = menu.query_one(".ask-user-help").render()
            assert "Ctrl+G" in str(help_text)

            # Move back: stop advertising the shortcut for the choice list.
            menu.action_previous_question()
            await pilot.pause()
            help_text = menu.query_one(".ask-user-help").render()
            assert "Ctrl+G" not in str(help_text)

    async def test_single_question_hides_number_label(self) -> None:
        app = _AskUserTestApp([{"question": "Name?", "type": "text"}])

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            source = menu._question_widgets[0].query_one(Markdown).source
            assert source == "Name? *(required)*"

    async def test_multiple_questions_show_number_labels(self) -> None:
        app = _AskUserTestApp(
            [
                {"question": "Name?", "type": "text"},
                {"question": "Color?", "type": "text"},
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            sources = [qw.query_one(Markdown).source for qw in menu._question_widgets]
            assert sources == [
                "**1.** Name? *(required)*",
                "**2.** Color? *(required)*",
            ]

    async def test_multi_question_menu_marks_active_question_with_border(self) -> None:
        """Multi-question menus restore the side line on the active question.

        The `ask-user-menu-multi` class on the menu gates the border in CSS
        (`.ask-user-menu-multi .ask-user-question-active`), so the highlight
        appears only when there are 2+ questions.
        """
        app = _AskUserTestApp(
            [
                {"question": "Name?", "type": "text"},
                {"question": "Color?", "type": "text"},
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            assert menu.has_class("ask-user-menu-multi")

            active, inactive = menu._question_widgets
            assert active.has_class("ask-user-question-active")
            assert not active.has_class("ask-user-question-inactive")
            assert inactive.has_class("ask-user-question-inactive")
            assert not inactive.has_class("ask-user-question-active")

            # The border applies only to the active question; the inactive one
            # carries matching padding instead so the two stay left-aligned.
            border, _ = active.styles.border_left
            assert border == "thick"
            assert active.styles.padding.left != inactive.styles.padding.left

    async def test_single_question_menu_has_no_active_border(self) -> None:
        """Single-question prompts stay flat: no multi class, no side border."""
        app = _AskUserTestApp([{"question": "Name?", "type": "text"}])

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            assert not menu.has_class("ask-user-menu-multi")

            question = menu._question_widgets[0]
            assert question.has_class("ask-user-question-active")
            border, _ = question.styles.border_left
            assert not border

    async def test_required_label_shown_for_required_question(self) -> None:
        """Required questions display a (required) indicator."""
        app = _AskUserTestApp([{"question": "Name?", "type": "text", "required": True}])

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            qw = menu._question_widgets[0]
            md = qw.query_one(Markdown)
            assert "required" in md.source

    async def test_multi_select_label_says_select_all_that_apply(self) -> None:
        """Multi-select questions advertise that several options may be chosen."""
        app = _AskUserTestApp(
            [
                {
                    "question": "Pick some",
                    "type": "multi_select",
                    "choices": [{"value": "red"}, {"value": "blue"}],
                    "required": True,
                },
                {
                    "question": "Optional extras",
                    "type": "multi_select",
                    "choices": [{"value": "docs"}],
                    "required": False,
                },
            ]
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            required_md = menu._question_widgets[0].query_one(Markdown)
            optional_md = menu._question_widgets[1].query_one(Markdown)
            assert "required, select all that apply" in required_md.source
            assert "select all that apply" in optional_md.source
            assert "required" not in optional_md.source

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

    async def test_required_empty_submit_shows_toast(self) -> None:
        """Blocked empty submit surfaces a warning toast to the user."""
        app = _AskUserTestApp([{"question": "Name?", "type": "text", "required": True}])

        async with app.run_test() as pilot:
            menu = app.query_one("#ask-user-menu", AskUserMenu)
            future: asyncio.Future[AskUserWidgetResult] = (
                asyncio.get_running_loop().create_future()
            )
            menu.set_future(future)

            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert not future.done()
            messages = [n.message for n in app._notifications]
            assert MISSING_ANSWER_TOAST in messages

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
