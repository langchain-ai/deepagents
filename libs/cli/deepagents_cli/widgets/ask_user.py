"""Ask user widget for interactive questions during agent execution."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.content import Content
from textual.message import Message
from textual.widgets import Input, Static

if TYPE_CHECKING:
    import asyncio

    from textual import events
    from textual.app import ComposeResult

    from deepagents_cli.ask_user import (
        AskUserWidgetResult,
        Choice,
        Question,
    )

from deepagents_cli.config import (
    CharsetMode,
    _detect_charset_mode,
    get_glyphs,
)

OTHER_CHOICE_LABEL = "Other (type your answer)"
logger = logging.getLogger(__name__)


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
        super().__init__(id=id or "ask-user-menu", classes="ask-user-menu", **kwargs)
        self._questions = questions
        self._answers: list[str] = [""] * len(questions)
        self._current_question = 0
        self._confirmed: list[bool] = [False] * len(questions)
        self._future: asyncio.Future[AskUserWidgetResult] | None = None
        self._question_widgets: list[_QuestionWidget] = []
        self._submitted = False

    def set_future(self, future: asyncio.Future[AskUserWidgetResult]) -> None:
        """Set the future to resolve when user answers."""
        self._future = future

    def compose(self) -> ComposeResult:  # noqa: D102
        glyphs = get_glyphs()
        count = len(self._questions)
        label = "Question" if count == 1 else "Questions"
        yield Static(
            f"{glyphs.cursor} Agent has {count} {label} for you",
            classes="ask-user-title",
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
            classes="ask-user-help",
        )

    async def on_mount(self) -> None:  # noqa: D102
        if _detect_charset_mode() == CharsetMode.ASCII:
            self.styles.border = ("ascii", "green")
        self._set_active_question(0)

    def focus_active(self) -> None:
        """Focus the current active question's input."""
        self._set_active_question(self._current_question)

    def on_input_submitted(self, event: Input.Submitted) -> None:  # noqa: D102
        event.stop()
        # Find which question owns this Input and confirm it.
        for qw in self._question_widgets:
            if (qw._text_input and qw._text_input is event.input) or (
                qw._other_input and qw._other_input is event.input
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
        self._current_question = index
        for i, qw in enumerate(self._question_widgets):
            if i == index:
                qw.add_class("ask-user-question-active")
                qw.remove_class("ask-user-question-inactive")
                qw.focus_input()
            else:
                qw.remove_class("ask-user-question-active")
                qw.add_class("ask-user-question-inactive")

    def _submit(self) -> None:
        if self._submitted:
            return
        self._submitted = True
        if self._future and not self._future.done():
            self._future.set_result({"type": "answered", "answers": self._answers})
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
        if self._submitted:
            return
        self._submitted = True
        if self._future and not self._future.done():
            self._future.set_result({"type": "cancelled"})
        self.post_message(self.Cancelled())

    def on_blur(self, event: events.Blur) -> None:  # noqa: PLR6301  # Textual event handler
        """Prevent blur from propagating and dismissing the menu."""
        event.stop()


class _ChoiceOption(Static):
    """A single selectable choice option."""

    def __init__(
        self, text: str, index: int, *, selected: bool = False, **kwargs: Any
    ) -> None:
        self.choice_index: int = index
        self.selected: bool = selected
        self._text: str = text
        super().__init__(self._render(), classes="ask-user-choice", **kwargs)

    def toggle(self) -> None:
        """Toggle the selected state."""
        self.selected = not self.selected
        self.update(self._render())

    def select(self) -> None:
        """Mark this choice as selected."""
        self.selected = True
        self.update(self._render())

    def deselect(self) -> None:
        """Mark this choice as deselected."""
        self.selected = False
        self.update(self._render())

    def _render(self) -> Content:
        """Build display content with cursor prefix.

        Returns:
            Styled Content with selection cursor and label text.
        """
        glyphs = get_glyphs()
        prefix = f"{glyphs.cursor} " if self.selected else "  "
        return Content.from_markup("$prefix$text", prefix=prefix, text=self._text)


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
        self._text_input: Input | None = None
        self._other_input: Input | None = None
        self._is_other_selected: bool = False

    def compose(self) -> ComposeResult:
        q_text = self._question.get("question", "")
        if self._required:
            markup = "[bold]$num. $text[/bold] [dim](required)[/dim]"
        else:
            markup = "[bold]$num. $text[/bold]"
        yield Static(Content.from_markup(markup, num=self._index + 1, text=q_text))

        if self._q_type == "multiple_choice" and self._choices:
            for i, choice in enumerate(self._choices):
                label = choice.get("value", str(choice))
                cw = _ChoiceOption(label, index=i, selected=(i == 0))
                self._choice_widgets.append(cw)
                yield cw

            other_cw = _ChoiceOption(OTHER_CHOICE_LABEL, index=len(self._choices))
            self._choice_widgets.append(other_cw)
            yield other_cw

            self._other_input = Input(
                placeholder="Type your answer...",
                classes="ask-user-other-input",
            )
            self._other_input.display = False
            yield self._other_input
        else:
            self._text_input = Input(
                placeholder="Type your answer...",
                classes="ask-user-text-input",
            )
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
        """Return the current answer text for this question."""
        if self._q_type == "text" or not self._choices:
            return self._text_input.value if self._text_input else ""

        if self._is_other_selected and self._other_input:
            return self._other_input.value

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
