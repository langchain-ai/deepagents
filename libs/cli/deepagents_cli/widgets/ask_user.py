"""Ask user widget for interactive questions during agent execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.message import Message
from textual.widgets import Input, Static

if TYPE_CHECKING:
    import asyncio

    from textual import events
    from textual.app import ComposeResult

from deepagents_cli.config import (
    CharsetMode,
    _detect_charset_mode,
    get_glyphs,
)

OTHER_CHOICE_LABEL = "Other (type your answer)"


class AskUserMenu(Container):
    """Interactive widget for asking the user questions.

    Supports text input and multiple choice questions. Multiple choice
    questions always include an "Other" option for free-form input.
    """

    can_focus = True
    can_focus_children = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
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
        questions: list[dict[str, Any]],
        id: str | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> None:
        super().__init__(id=id or "ask-user-menu", classes="ask-user-menu", **kwargs)
        self._questions = questions
        self._answers: list[str] = [""] * len(questions)
        self._current_question = 0
        self._future: asyncio.Future[dict[str, Any]] | None = None
        self._question_widgets: list[_QuestionWidget] = []

    def set_future(self, future: asyncio.Future[dict[str, Any]]) -> None:
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
        yield Static(
            f"Enter to submit {glyphs.bullet} Esc to cancel",
            classes="ask-user-help",
        )

    async def on_mount(self) -> None:  # noqa: D102
        if _detect_charset_mode() == CharsetMode.ASCII:
            self.styles.border = ("ascii", "green")
        if self._question_widgets:
            self._question_widgets[0].focus_input()

    def on_input_submitted(self, event: Input.Submitted) -> None:  # noqa: D102
        event.stop()
        for i, qw in enumerate(self._question_widgets):
            self._answers[i] = qw.get_answer()

        all_answered = all(a.strip() for a in self._answers)
        if all_answered:
            self._submit()
        else:
            for i, qw in enumerate(self._question_widgets):
                if not self._answers[i].strip():
                    qw.focus_input()
                    break

    def _submit(self) -> None:
        if self._future and not self._future.done():
            self._future.set_result({"type": "answered", "answers": self._answers})
        self.post_message(self.Answered(self._answers))

    def action_cancel(self) -> None:  # noqa: D102
        if self._future and not self._future.done():
            self._future.set_result({"type": "cancelled"})
        self.post_message(self.Cancelled())

    def on_blur(self, event: events.Blur) -> None:  # noqa: D102
        pass


class _ChoiceOption(Static):
    """A single selectable choice option."""

    def __init__(self, label: str, index: int, **kwargs: Any) -> None:
        super().__init__(label, classes="ask-user-choice", **kwargs)
        self.choice_index = index
        self.selected = False

    def toggle(self) -> None:
        self.selected = not self.selected
        self._update_display()

    def select(self) -> None:
        self.selected = True
        self._update_display()

    def deselect(self) -> None:
        self.selected = False
        self._update_display()

    def _update_display(self) -> None:
        glyphs = get_glyphs()
        label_text = self.renderable
        if isinstance(label_text, str):
            raw = label_text.lstrip()
            for prefix in (f"{glyphs.cursor} ", "  "):
                if raw.startswith(prefix):
                    raw = raw[len(prefix) :]
                    break
            prefix = f"{glyphs.cursor} " if self.selected else "  "
            self.update(f"{prefix}{raw}")


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

    def __init__(self, question: dict[str, Any], index: int, **kwargs: Any) -> None:
        super().__init__(classes="ask-user-question", **kwargs)
        self._question = question
        self._index = index
        self._q_type = question.get("type", "text")
        self._choices: list[dict[str, str]] = question.get("choices", [])
        self._choice_widgets: list[_ChoiceOption] = []
        self._selected_choice = 0
        self._text_input: Input | None = None
        self._other_input: Input | None = None
        self._is_other_selected = False

    def compose(self) -> ComposeResult:
        q_text = self._question.get("question", "")
        yield Static(f"[bold]{self._index + 1}. {q_text}[/bold]")

        if self._q_type == "multiple_choice" and self._choices:
            glyphs = get_glyphs()
            for i, choice in enumerate(self._choices):
                label = choice.get("value", str(choice))
                prefix = f"{glyphs.cursor} " if i == 0 else "  "
                cw = _ChoiceOption(f"{prefix}{label}", index=i)
                if i == 0:
                    cw.selected = True
                self._choice_widgets.append(cw)
                yield cw

            other_cw = _ChoiceOption(
                f"  {OTHER_CHOICE_LABEL}", index=len(self._choices)
            )
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
        if self._text_input:
            self._text_input.focus()
        elif self._choice_widgets:
            self.focus()

    def get_answer(self) -> str:
        if self._q_type == "text" or not self._choices:
            return self._text_input.value if self._text_input else ""

        if self._is_other_selected and self._other_input:
            return self._other_input.value

        if self._choice_widgets and self._selected_choice < len(self._choices):
            return self._choices[self._selected_choice].get("value", "")

        return ""

    def action_move_up(self) -> None:
        if self._q_type != "multiple_choice" or not self._choice_widgets:
            return
        if (
            self._is_other_selected
            and self._other_input
            and self._other_input.has_focus
        ):
            self.focus()
            return
        old = self._selected_choice
        self._selected_choice = max(0, self._selected_choice - 1)
        if old != self._selected_choice:
            self._update_choice_selection()

    def action_move_down(self) -> None:
        if self._q_type != "multiple_choice" or not self._choice_widgets:
            return
        max_idx = len(self._choice_widgets) - 1
        old = self._selected_choice
        self._selected_choice = min(max_idx, self._selected_choice + 1)
        if old != self._selected_choice:
            self._update_choice_selection()

    def action_select_or_submit(self) -> None:
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
