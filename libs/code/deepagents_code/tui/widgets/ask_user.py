"""Ask user widget for interactive questions during agent execution."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.content import Content
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
        QuestionType,
    )

from deepagents_code._ask_user_types import (
    CHOICE_QUESTION_TYPES,
    MULTI_SELECT_ANSWER_SEPARATOR,
    QUESTION_TYPES,
)
from deepagents_code.config import get_glyphs
from deepagents_code.editor import editor_display_name
from deepagents_code.tui.widgets._inline_prompt import (
    InlinePromptCompletion,
    InlinePromptOption,
    InlinePromptTextArea,
    apply_inline_prompt_border,
    newline_hint,
    stop_inline_prompt_blur,
)

OTHER_CHOICE_LABEL = "Other (type your answer)"
MISSING_ANSWER_TOAST = "Please provide an answer to all questions before continuing."
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

    Supports text input, multiple choice (pick exactly one), and multi-select
    (toggle one or more) questions. Multiple choice questions always include an
    "Other" option for free-form input; multi-select questions do not.
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
        classes = "inline-prompt ask-user-menu"
        if len(questions) > 1:
            # Gates the active question's left border in CSS; single-question
            # prompts stay borderless.
            classes += " ask-user-menu-multi"
        super().__init__(
            id=id or "ask-user-menu",
            classes=classes,
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
        self._help_widget: Static | None = None

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
                qw = _QuestionWidget(q, index=i, show_number=count > 1)
                self._question_widgets.append(qw)
                yield qw

        yield Static("")
        self._help_widget = Static(
            self._render_help(),
            classes="inline-prompt-help ask-user-help",
        )
        yield self._help_widget

    def _render_help(self) -> str:
        """Build the footer hint text for the current menu state.

        The `Ctrl+X` editor hint is included only while one of this menu's text
        areas holds focus, matching the routing in `App.action_open_editor`.

        Returns:
            The bullet-joined footer hint string.
        """
        glyphs = get_glyphs()
        parts = [
            f"{glyphs.arrow_up}/{glyphs.arrow_down} Select",
            "Enter to continue",
        ]
        if self._show_editor_hint():
            editor = editor_display_name()
            parts.append(
                f"Ctrl+X edit in {editor}"
                if editor is not None
                else "Ctrl+X external editor"
            )
        # Multi-select questions own no text area, so a prompt made up entirely
        # of them has nowhere to insert a newline.
        if any(qw.has_text_input for qw in self._question_widgets):
            parts.append(newline_hint())
        if any(qw.question_type == "multi_select" for qw in self._question_widgets):
            parts.append("Space to toggle")
        if len(self._questions) > 1:
            parts.append("Tab/Shift+Tab switch question")
        parts.append("Esc to cancel")
        return f" {glyphs.bullet} ".join(parts)

    def _show_editor_hint(self) -> bool:
        """Whether `ctrl+x` would currently open one of this menu's text areas.

        `App.action_open_editor` routes `ctrl+x` to an ask-user text area only
        when one is focused, and otherwise falls through to the chat input.
        A visible-but-unfocused field therefore must not advertise the
        shortcut: pressing it would open the user's chat draft instead. The
        conditions here mirror `App._focused_ask_user_editor`.

        Returns:
            `True` if a text area belonging to this menu holds focus.
        """
        focused = self.app.focused
        return (
            isinstance(focused, AskUserTextArea)
            and self in focused.ancestors
            and focused.is_attached
            and focused.display
            and focused.visible
        )

    def _update_help(self) -> None:
        """Refresh the footer hint after a focus or field-visibility change."""
        if self._help_widget is not None:
            self._help_widget.update(self._render_help())

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
                else:
                    self.app.notify(
                        MISSING_ANSWER_TOAST,
                        severity="warning",
                        markup=False,
                    )
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

        # A confirmed required question was left empty. Reachable normally: the
        # user can navigate back to a confirmed multi-select and un-toggle every
        # option. Re-open it and say why, so the jump is not unexplained.
        for i, a in enumerate(self._answers):
            if not a.strip() and self._question_widgets[i]._required:
                self._confirmed[i] = False
                self._set_active_question(i)
                self.app.notify(
                    MISSING_ANSWER_TOAST,
                    severity="warning",
                    markup=False,
                )
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
        the question container itself for choice-based questions (whose options
        are not individually focusable), without going through
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
        # Every focus change inside the menu can flip whether ctrl+x routes
        # here, including clicks that land on a question container rather than
        # its text area, so refresh regardless of which question is active.
        self._update_help()

    def on_descendant_blur(self, event: events.DescendantBlur) -> None:
        """Retract the `Ctrl+X` hint when focus leaves a text area."""
        del event  # Unused: the hint is recomputed from current focus.
        self._update_help()

    def on_blur(self, event: events.Blur) -> None:  # noqa: PLR6301  # Textual event handler
        """Prevent blur from propagating and dismissing the menu."""
        stop_inline_prompt_blur(event)


class _ChoiceOption(InlinePromptOption):
    """A single selectable ask-user choice option."""

    @property
    def _unselected_marker(self) -> str:
        return get_glyphs().bullet

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


class _MultiSelectOption(_ChoiceOption):
    """A toggleable ask-user choice option for `multi_select` questions.

    Renders a toggle glyph (`circle_filled`/`circle_empty`) before the label so
    the user can see which options are toggled on, independent of the highlight
    cursor. This is the source of truth for whether a choice is selected;
    `_QuestionWidget` reads it back rather than tracking selection separately.

    Overriding `_render` is what keeps the glyph correct: the base class's
    `select`/`deselect`/`set_state` all re-render through it, so cursor movement
    preserves the toggle without any extra bookkeeping.
    """

    def __init__(
        self, text: str, index: int, *, selected: bool = False, **kwargs: Any
    ) -> None:
        """Initialize a multi-select option with its toggle cleared.

        Args:
            text: Option label.
            index: Position in its owning question's choice list.
            selected: Whether the *highlight cursor* starts on this option. This
                is a separate axis from the toggle state — use `set_checked` for
                that.
            **kwargs: Additional `_ChoiceOption` arguments.
        """
        # Must precede `super().__init__()`: the base constructor calls
        # `self._render()`, which reads `self._checked`.
        self._checked = False
        super().__init__(text, index, selected=selected, **kwargs)

    @property
    def checked(self) -> bool:
        """Whether this option is currently toggled on."""
        return self._checked

    def set_checked(self, checked: bool) -> None:
        """Update the toggle state and re-render the option.

        Args:
            checked: Whether the option should be toggled on.
        """
        self._checked = checked
        self.update(self._render())

    def _render(self) -> Content:
        glyphs = get_glyphs()
        cursor = f"{glyphs.cursor} " if self._cursor_visible else "  "
        box = glyphs.circle_filled if self._checked else glyphs.circle_empty
        return Content.from_markup(
            "$cursor$box $text", cursor=cursor, box=box, text=self._text
        )


class _QuestionWidget(Vertical):
    """Widget for a single question (text, multiple choice, or multi-select)."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("k", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("j", "move_down", "Down", show=False),
        Binding("space", "toggle_choice", "Toggle", show=False),
        Binding("enter", "select_or_submit", "Select", show=False),
    ]

    can_focus = True
    can_focus_children = True

    def __init__(
        self,
        question: Question,
        index: int,
        *,
        show_number: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(classes="ask-user-question", **kwargs)
        question_type = question.get("type", "text")
        self._question: Question = question
        self._index: int = index
        self._show_number = show_number
        if question_type not in QUESTION_TYPES:
            # Runtime defense: the widget is also built from interrupt payloads
            # that never went through `_validate_questions`. Log rather than
            # silently render a choice question as a free-text box.
            logger.warning(
                "ask_user question %d has unrecognized type %r; rendering as text",
                index,
                question_type,
            )
            question_type = "text"
        self._q_type: QuestionType = question_type
        self._choices: list[Choice] = question.get("choices", [])
        if self._q_type in CHOICE_QUESTION_TYPES and not self._choices:
            logger.warning(
                "ask_user %s question %d has no choices; rendering as text",
                self._q_type,
                index,
            )
        self._required: bool = question.get("required", True)
        self._choice_widgets: list[_ChoiceOption] = []
        self._multi_select_widgets: list[_MultiSelectOption] = []
        self._selected_choice: int = 0
        self._text_input: AskUserTextArea | None = None
        self._other_input: AskUserTextArea | None = None
        self._is_other_selected: bool = False

    @property
    def question_type(self) -> QuestionType:
        """Resolved type of this question."""
        return self._q_type

    @property
    def has_text_input(self) -> bool:
        """Whether this question owns a text area.

        Mirrors `compose`: every type except a multi-select with choices yields
        either the main text input or the multiple-choice "Other" input.
        """
        return not (self._q_type == "multi_select" and bool(self._choices))

    def compose(self) -> ComposeResult:
        q_text = _TRAILING_ANNOTATION_RE.sub("", self._question.get("question", ""))
        prefix = f"**{self._index + 1}.** " if self._show_number else ""
        suffix = " *(required)*" if self._required else ""
        # q_text is agent-authored; rendered as markdown intentionally so
        # agents can use inline formatting, links, and code spans in questions.
        yield Markdown(f"{prefix}{q_text}{suffix}", classes="ask-user-question-text")

        if self._q_type == "multiple_choice" and self._choices:
            for i, choice in enumerate(self._choices):
                # Same fallback as `get_answer`, so the rendered label can never
                # disagree with the value submitted for it.
                label = choice.get("value", "")
                cw = _ChoiceOption(label, index=i, selected=(i == 0))
                self._choice_widgets.append(cw)
                yield cw

            other_cw = _ChoiceOption(OTHER_CHOICE_LABEL, index=len(self._choices))
            self._choice_widgets.append(other_cw)
            yield other_cw

            self._other_input = AskUserTextArea(classes="ask-user-other-input")
            self._other_input.display = False
            yield self._other_input
        elif self._q_type == "multi_select" and self._choices:
            for i, choice in enumerate(self._choices):
                label = choice.get("value", "")
                msw = _MultiSelectOption(label, index=i, selected=(i == 0))
                self._choice_widgets.append(msw)
                self._multi_select_widgets.append(msw)
                yield msw
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

        For text and "Other" answers, collapsed-paste placeholders are expanded
        so the agent receives the full pasted content, not the compact
        `[Pasted text #N]` token.

        A multi-select answer is the toggled values in choice-list order (not
        the order they were toggled) joined with
        `MULTI_SELECT_ANSWER_SEPARATOR`, and is empty when nothing is toggled.
        `_validate_choices` rejects values containing that separator, so the
        join stays unambiguous.
        """
        if self._q_type == "text" or not self._choices:
            return self._text_input.submitted_value if self._text_input else ""

        if self._q_type == "multi_select":
            return MULTI_SELECT_ANSWER_SEPARATOR.join(
                self._choices[widget.option_index].get("value", "")
                for widget in self._multi_select_widgets
                if widget.checked
            )

        if self._is_other_selected and self._other_input:
            return self._other_input.submitted_value

        if self._choice_widgets and self._selected_choice < len(self._choices):
            return self._choices[self._selected_choice].get("value", "")

        return ""

    def action_move_up(self) -> None:
        """Move the highlight cursor up in the choice list."""
        if self._q_type not in CHOICE_QUESTION_TYPES or not self._choice_widgets:
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
        """Move the highlight cursor down in the choice list."""
        if self._q_type not in CHOICE_QUESTION_TYPES or not self._choice_widgets:
            return
        max_idx = len(self._choice_widgets) - 1
        old = self._selected_choice
        self._selected_choice = min(max_idx, self._selected_choice + 1)
        if old != self._selected_choice:
            self._update_choice_selection()

    def action_toggle_choice(self) -> None:
        """Toggle the highlighted choice for `multi_select` questions."""
        if self._q_type != "multi_select" or not self._multi_select_widgets:
            return
        index = self._selected_choice
        if index >= len(self._multi_select_widgets):
            # Unreachable: `action_move_down` caps the cursor at the last choice
            # and multi-select adds no synthetic "Other" row. Log rather than
            # swallow, since reaching it means the cursor and choices skewed.
            logger.error(
                "multi_select question %d cursor %d exceeds %d toggleable choices",
                self._index,
                index,
                len(self._multi_select_widgets),
            )
            return
        widget = self._multi_select_widgets[index]
        widget.set_checked(not widget.checked)

    def action_select_or_submit(self) -> None:
        """Confirm the current answer, or open the Other input.

        For multi-select, warns and keeps the question open while a required
        question has nothing toggled.
        """
        if self._q_type == "multi_select" and self._choice_widgets:
            if not self.get_answer().strip() and self._required:
                # Keep the question open until at least one option is selected,
                # and say so — a bare no-op reads as a frozen UI.
                self.app.notify(
                    MISSING_ANSWER_TOAST,
                    severity="warning",
                    markup=False,
                )
                return
            menu = self._find_menu()
            if menu is not None:
                menu.confirm_and_advance(self._index)
            return
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
