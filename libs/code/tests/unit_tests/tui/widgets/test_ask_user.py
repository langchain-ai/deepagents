"""Tests for ask_user tool integration in the CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding

import deepagents_code
from deepagents_code.tui.widgets.ask_user import (
    MISSING_ANSWER_TOAST,
    MISSING_OTHER_TEXT_TOAST,
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


class TestTrailingAnnotationRegex:
    """Strips LLM-appended '(optional)'/'- required' annotations from question text."""


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
