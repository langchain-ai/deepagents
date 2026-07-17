"""Ask user widget for interactive questions during agent execution."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.message import Message
from textual.widgets import Markdown, Static

if TYPE_CHECKING:
    import asyncio

    from textual import events
    from textual.app import ComposeResult

    from deepagents_code._ask_user_types import (
        AskUserWidgetResult,
        Choice,
        Question,
    )

from deepagents_code.config import get_glyphs
from deepagents_code.tui.widgets._inline_prompt import (
    InlinePromptCompletion,
    InlinePromptOption,
    InlinePromptTextArea,
    apply_inline_prompt_border,
    stop_inline_prompt_blur,
)

OTHER_CHOICE_LABEL = "Other (type your answer)"
logger = logging.getLogger(__name__)

_TRAILING_ANNOTATION_RE = re.compile(
    # \u2013 = en-dash, \u2014 = em-dash.
    r"""
    \s*
    (?:
        [-\u2013\u2014]\s*(?:optional|required)
      | \((?:optional|required)[.!?]?\)
      | \[(?:optional|required)[.!?]?\]
    )
    [.!?]*
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
"""Strip LLM-appended trailing annotations like ' - optional', ' (optional)',
or ' [required]' from question text before rendering.

Defense-in-depth alongside the instruction in `ASK_USER_TOOL_DESCRIPTION`
(`ask_user.py`). The UI already renders a `*(required)*` marker based on the
`required` field, so any LLM-authored duplicate is redundant noise."""


class AskUserTextArea(InlinePromptTextArea):
    """Free-form answer input for ask-user questions.

    Adds one behavior over the shared base: when the cursor is on the first or
    last line of a `multiple_choice` question, Up/Down are handed back to the
    enclosing choice list instead of moving the text cursor.
    """

    class Submitted(InlinePromptTextArea.Submitted):
        """Posted when the user presses Enter to submit an ask-user answer."""

    async def _on_key(self, event: events.Key) -> None:
        if event.key in {"up", "down"}:
            cursor_location = self.cursor_location
            at_top = self.get_cursor_up_location() == cursor_location
            at_bottom = self.get_cursor_down_location() == cursor_location
            if (event.key == "up" and at_top) or (event.key == "down" and at_bottom):
                question = self._find_question_widget()
                if question is not None and question._q_type == "multiple_choice":
                    event.prevent_default()
                    event.stop()
                    if event.key == "up":
                        question.action_move_up()
                    else:
                        question.action_move_down()
                    return
        await super()._on_key(event)

    def _find_question_widget(self) -> _QuestionWidget | None:
        """Walk up to find the enclosing `_QuestionWidget`, if any.

        Returns:
            The enclosing `_QuestionWidget` ancestor, or `None` if not found.
        """
        node: Any = self.parent
        while node is not None:
            if isinstance(node, _QuestionWidget):
                return node
            node = node.parent
        return None


class AskUserMenu(Container):
    """Interactive widget for asking the user questions.

    Supports text input and multiple choice questions. Multiple choice
    questions always include an "Other" option for free-form input.
    """

    can_focus = True
    can_focus_children = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("tab", "next_question", "Next question", show=False, priority=True),
    ]

    class Answered(Message):
        """Message sent when user submits all answers."""

        def __init__(self, answers: list[str]) -> None:  # noqa: D107
            super().__init__()
            self.answers = answers

    class Cancelled(Message):
        """Message sent when user cancels the ask_user prompt."""

        def __init__(self) -> None:  # noqa: D107
            super().__init__()

    def __init__(  # noqa: D107
        self,
        questions: list[Question],
        id: str | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> None:
        super().__init__(
            id=id or "ask-user-menu",
            classes="inline-prompt ask-user-menu",
            **kwargs,
        )
        self._questions = questions
        self._answers: list[str] = [""] * len(questions)
        self._current_question = 0
        self._confirmed: list[bool] = [False] * len(questions)
        self._completion: InlinePromptCompletion[AskUserWidgetResult] = (
            InlinePromptCompletion()
        )
        self._question_widgets: list[_QuestionWidget] = []

    def set_future(self, future: asyncio.Future[AskUserWidgetResult]) -> None:
        """Set the future to resolve when user answers."""
        self._completion.set_future(future)

    def compose(self) -> ComposeResult:  # noqa: D102
        glyphs = get_glyphs()
        count = len(self._questions)
        if count == 1:
            title = "Agent has a question for you"
        else:
            title = f"Agent has {count} Questions for you"
        yield Static(
            f"{glyphs.cursor} {title}",
            classes="inline-prompt-title ask-user-title",
        )
        yield Static("")

        with Vertical(classes="ask-user-questions"):
            for i, q in enumerate(self._questions):
                qw = _QuestionWidget(q, index=i)
                self._question_widgets.append(qw)
                yield qw

        yield Static("")
        parts = [
            f"{glyphs.arrow_up}/{glyphs.arrow_down} Select",
            "Enter to continue",
        ]
        if len(self._questions) > 1:
            parts.append("Tab/Shift+Tab switch question")
        parts.append("Esc to cancel")
        yield Static(
            f" {glyphs.bullet} ".join(parts),
            classes="inline-prompt-help ask-user-help",
        )

    async def on_mount(self) -> None:  # noqa: D102
        apply_inline_prompt_border(self)
        self._set_active_question(0)

    def focus_active(self) -> None:
        """Focus the current active question's input."""
        self._set_active_question(self._current_question)

    def on_ask_user_text_area_submitted(self, event: AskUserTextArea.Submitted) -> None:
        """Confirm the question whose text area was submitted."""
        event.stop()
        for qw in self._question_widgets:
            if (qw._text_input and qw._text_input is event.text_area) or (
                qw._other_input and qw._other_input is event.text_area
            ):
                answer = qw.get_answer()
                if answer.strip() or not qw._required:
                    self.confirm_and_advance(qw._index)
                return

    def confirm_and_advance(self, index: int) -> None:
        """Confirm the answer at `index` and advance to the next question."""
        self._answers[index] = self._question_widgets[index].get_answer()
        self._confirmed[index] = True

        # Find next unconfirmed question.
        for i in range(index + 1, len(self._question_widgets)):
            if not self._confirmed[i]:
                self._set_active_question(i)
                return

        # All confirmed — collect final answers and submit.
        for i, qw in enumerate(self._question_widgets):
            self._answers[i] = qw.get_answer()
        if all(
            a.strip() or not self._question_widgets[i]._required
            for i, a in enumerate(self._answers)
        ):
            self._submit()
            return

        # Edge case: a confirmed required text field was left empty
        # (shouldn't happen normally). Re-open it.
        for i, a in enumerate(self._answers):
            if not a.strip() and self._question_widgets[i]._required:
                self._confirmed[i] = False
                self._set_active_question(i)
                return

    def _set_active_question(self, index: int) -> None:
        """Update the visual indicator and focus for the active question."""
        self._highlight_question(index)
        self._question_widgets[index].focus_input()

    def _highlight_question(self, index: int) -> None:
        """Highlight `index` and dim the rest without changing focus."""
        self._current_question = index
        for i, qw in enumerate(self._question_widgets):
            if i == index:
                qw.add_class("ask-user-question-active")
                qw.remove_class("ask-user-question-inactive")
            else:
                qw.remove_class("ask-user-question-active")
                qw.add_class("ask-user-question-inactive")

    def _submit(self) -> None:
        result: AskUserWidgetResult = {
            "type": "answered",
            "answers": self._answers,
        }
        if self._completion.resolve(result):
            self.post_message(self.Answered(self._answers))

    def action_next_question(self) -> None:
        """Navigate to the next question without confirming."""
        if self._current_question < len(self._question_widgets) - 1:
            self._set_active_question(self._current_question + 1)

    def action_previous_question(self) -> None:
        """Navigate to the previous question without confirming."""
        if self._current_question > 0:
            self._set_active_question(self._current_question - 1)

    def action_cancel(self) -> None:  # noqa: D102
        if self._completion.resolve({"type": "cancelled"}):
            self.post_message(self.Cancelled())

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        """Keep the active-question highlight in sync with focus.

        A mouse click moves focus into another question's text input, or onto
        the question container itself for multiple-choice (whose choices are
        not individually focusable), without going through
        `_set_active_question`, which would otherwise leave the highlight on
        the previously active question. Sync the highlight to the focused
        question so exactly one question is ever active. Focus is not moved
        here, so the widget the user clicked keeps focus.
        """
        node: Any = event.widget
        while node is not None and not isinstance(node, _QuestionWidget):
            node = node.parent
        if node is not None and node._index != self._current_question:
            self._highlight_question(node._index)

    def on_blur(self, event: events.Blur) -> None:  # noqa: PLR6301  # Textual event handler
        """Prevent blur from propagating and dismissing the menu."""
        stop_inline_prompt_blur(event)


class _ChoiceOption(InlinePromptOption):
    """A single selectable ask-user choice option."""

    def __init__(
        self, text: str, index: int, *, selected: bool = False, **kwargs: Any
    ) -> None:
        """Initialize an ask-user choice option."""
        super().__init__(
            text,
            index,
            selected=selected,
            classes="ask-user-choice",
            **kwargs,
        )


class _QuestionWidget(Vertical):
    """Widget for a single question (text or multiple choice)."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("k", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("j", "move_down", "Down", show=False),
        Binding("enter", "select_or_submit", "Select", show=False),
    ]

    can_focus = True
    can_focus_children = True

    def __init__(self, question: Question, index: int, **kwargs: Any) -> None:
        super().__init__(classes="ask-user-question", **kwargs)
        question_type = question.get("type", "text")
        self._question: Question = question
        self._index: int = index
        self._q_type: Literal["text", "multiple_choice"] = (
            "multiple_choice" if question_type == "multiple_choice" else "text"
        )
        self._choices: list[Choice] = question.get("choices", [])
        self._required: bool = question.get("required", True)
        self._choice_widgets: list[_ChoiceOption] = []
        self._selected_choice: int = 0
        self._text_input: AskUserTextArea | None = None
        self._other_input: AskUserTextArea | None = None
        self._is_other_selected: bool = False

    def compose(self) -> ComposeResult:
        q_text = _TRAILING_ANNOTATION_RE.sub("", self._question.get("question", ""))
        num = self._index + 1
        suffix = " *(required)*" if self._required else ""
        # q_text is agent-authored; rendered as markdown intentionally so
        # agents can use inline formatting, links, and code spans in questions.
        yield Markdown(f"**{num}.** {q_text}{suffix}", classes="ask-user-question-text")

        if self._q_type == "multiple_choice" and self._choices:
            for i, choice in enumerate(self._choices):
                label = choice.get("value", str(choice))
                cw = _ChoiceOption(label, index=i, selected=(i == 0))
                self._choice_widgets.append(cw)
                yield cw

            other_cw = _ChoiceOption(OTHER_CHOICE_LABEL, index=len(self._choices))
            self._choice_widgets.append(other_cw)
            yield other_cw

            self._other_input = AskUserTextArea(classes="ask-user-other-input")
            self._other_input.display = False
            yield self._other_input
        else:
            self._text_input = AskUserTextArea(classes="ask-user-text-input")
            yield self._text_input

    def focus_input(self) -> None:
        """Focus the appropriate input for this question."""
        if self._text_input:
            self._text_input.focus()
        elif self._is_other_selected and self._other_input:
            self._other_input.focus()
        elif self._choice_widgets:
            self.focus()

    def get_answer(self) -> str:
        """Return the current answer text for this question.

        Collapsed-paste placeholders are expanded so the agent receives the
        full pasted content, not the compact `[Pasted text #N]` token.
        """
        if self._q_type == "text" or not self._choices:
            return self._text_input.submitted_value if self._text_input else ""

        if self._is_other_selected and self._other_input:
            return self._other_input.submitted_value

        if self._choice_widgets and self._selected_choice < len(self._choices):
            return self._choices[self._selected_choice].get("value", "")

        return ""

    def action_move_up(self) -> None:
        """Move selection up in the choice list."""
        if self._q_type != "multiple_choice" or not self._choice_widgets:
            return
        if (
            self._is_other_selected
            and self._other_input
            and self._other_input.has_focus
        ):
            # Jump directly to the last real choice instead of requiring
            # two presses (one to defocus, one to navigate).
            self._selected_choice = max(0, len(self._choices) - 1)
            self._update_choice_selection()
            self.focus()
            return
        old = self._selected_choice
        self._selected_choice = max(0, self._selected_choice - 1)
        if old != self._selected_choice:
            self._update_choice_selection()

    def action_move_down(self) -> None:
        """Move selection down in the choice list."""
        if self._q_type != "multiple_choice" or not self._choice_widgets:
            return
        max_idx = len(self._choice_widgets) - 1
        old = self._selected_choice
        self._selected_choice = min(max_idx, self._selected_choice + 1)
        if old != self._selected_choice:
            self._update_choice_selection()

    def action_select_or_submit(self) -> None:
        """Confirm current choice or open the Other input."""
        if self._q_type == "multiple_choice" and self._choice_widgets:
            is_other = self._selected_choice == len(self._choices)
            if is_other:
                self._is_other_selected = True
                if self._other_input:
                    self._other_input.display = True
                    self._other_input.focus()
            else:
                self._is_other_selected = False
                if self._other_input:
                    self._other_input.display = False
                menu = self._find_menu()
                if menu is not None:
                    menu.confirm_and_advance(self._index)

    def _find_menu(self) -> AskUserMenu | None:
        node: Any = self.parent
        while node is not None:
            if isinstance(node, AskUserMenu):
                return node
            node = node.parent
        logger.warning(
            "Failed to find AskUserMenu ancestor for question index %d",
            self._index,
        )
        return None

    def _update_choice_selection(self) -> None:
        for i, cw in enumerate(self._choice_widgets):
            if i == self._selected_choice:
                cw.select()
            else:
                cw.deselect()

        is_other = self._selected_choice == len(self._choices)
        self._is_other_selected = is_other
        if self._other_input:
            self._other_input.display = is_other
            if is_other:
                self._other_input.focus()
