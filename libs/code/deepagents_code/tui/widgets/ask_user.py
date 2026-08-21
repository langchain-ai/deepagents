"""Ask user widget for interactive questions during agent execution."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar, assert_never

from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal, Vertical
from textual.content import Content
from textual.message import Message
from textual.widgets import Markdown, Static, TextArea

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
    QUESTION_TYPES,
    ask_user_answer_is_empty,
    encode_multi_select_answer,
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
ADD_ANOTHER_OTHER_LABEL = "Add another custom answer"
MAX_MULTI_SELECT_OTHER_ENTRIES = 10
MISSING_ANSWER_TOAST = "Please provide an answer to all questions before continuing."
MISSING_OTHER_TEXT_TOAST = "Please type a custom answer for Other, or uncheck it."
MAX_OTHER_ENTRIES_TOAST = (
    f"At most {MAX_MULTI_SELECT_OTHER_ENTRIES} custom answers can be added to one "
    "question. Combine values into an existing field to add more."
)
UNSUBMITTABLE_ANSWER_TOAST = (
    "Could not submit this answer (internal UI error). Press Esc to cancel and retry."
)
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

    Adds two behaviors over the shared base:

    - When the cursor is on the first or last line of a choice question's Other
      free-text input, Up/Down are handed back to the enclosing choice list
      instead of moving the text cursor.
    - Escape deselects the checked custom multi-select option that owns this
      field (via the `uncheck_other` binding), instead of cancelling the whole
      prompt. `check_action` keeps the binding unmatched otherwise, so Escape
      still bubbles to `AskUserMenu.action_cancel`.
    """

    # TextArea defaults to `height: 1fr`, which lets open Other fields swallow
    # leftover vertical space when several are visible. Stay content-sized; app
    # CSS still sets min/max height per role (main text vs Other).
    DEFAULT_CSS = """
    AskUserTextArea {
        height: auto;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        *InlinePromptTextArea.BINDINGS,
        Binding("escape", "uncheck_other", show=False, priority=True),
    ]

    class Submitted(InlinePromptTextArea.Submitted):
        """Posted when the user presses Enter to submit an ask-user answer."""

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Bind Escape only while this field owns a checked custom option.

        Returns:
            Whether the action is available, or `None` to let it bubble.
        """
        if action == "uncheck_other":
            question = self._find_question_widget()
            return question is not None and question.can_uncheck_other_input(self)
        return super().check_action(action, parameters)

    def action_uncheck_other(self) -> None:
        """Deselect this field's custom multi-select option, when applicable."""
        question = self._find_question_widget()
        if question is not None:
            question.uncheck_other_input(self)

    async def _on_key(self, event: events.Key) -> None:
        if event.key in {"up", "down"}:
            cursor_location = self.cursor_location
            at_top = self.get_cursor_up_location() == cursor_location
            at_bottom = self.get_cursor_down_location() == cursor_location
            if (event.key == "up" and at_top) or (event.key == "down" and at_bottom):
                question = self._find_question_widget()
                # Only hand off from synthetic Other free-text input(s) under a
                # choice list. Plain text questions own no choice list to return
                # focus to.
                if (
                    question is not None
                    and question._q_type in CHOICE_QUESTION_TYPES
                    and question.owns_other_input(self)
                ):
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
    (toggle one or more) questions. Choice questions always include an "Other"
    option for free-form input. Multi-select can combine multiple custom Other
    values with predefined options: filling one Other reveals an "Add another"
    row.
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

        The `Ctrl+X external editor` hint is included only while one of this
        menu's text areas holds focus, matching the routing in
        `App.action_open_editor`. Multi-select prompts add a Space-toggle tip.

        Returns:
            The bullet-joined footer hint string.
        """
        glyphs = get_glyphs()
        parts = [f"{glyphs.arrow_up}/{glyphs.arrow_down} move"]
        # Additive rather than replacing "Enter to continue": in a mixed prompt
        # the other questions still continue on Enter, so swapping the hint out
        # wholesale would describe the wrong keys while they are active.
        if any(qw.question_type == "multi_select" for qw in self._question_widgets):
            parts.append("Space toggle")
        # The newline hint is unconditional: choice questions always own an Other
        # free-text input and text questions own a main text area, so every
        # question type has somewhere to insert a newline.
        parts.extend(("Enter to continue", newline_hint()))
        if self._show_editor_hint():
            editor = editor_display_name()
            parts.append(
                f"Ctrl+X edit in {editor}"
                if editor is not None
                else "Ctrl+X external editor"
            )
        if len(self._questions) > 1:
            parts.append("Tab/Shift+Tab switch question")
        parts.append(
            "Esc to deselect" if self._can_deselect_focused_other() else "Esc to cancel"
        )
        return f" {glyphs.bullet} ".join(parts)

    def _can_deselect_focused_other(self) -> bool:
        """Return whether Escape will deselect the currently focused custom option."""
        focused = self.app.focused
        if not isinstance(focused, AskUserTextArea):
            return False
        question = focused._find_question_widget()
        return question is not None and question.can_uncheck_other_input(focused)

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
                qw.owns_other_input(event.text_area)
            ):
                error = qw.validate_for_submit()
                if error is not None:
                    self.app.notify(error, severity="warning", markup=False)
                    return
                if not qw.answer_is_empty() or not qw._required:
                    self.confirm_and_advance(qw._index)
                else:
                    self.app.notify(
                        MISSING_ANSWER_TOAST,
                        severity="warning",
                        markup=False,
                    )
                return
        # The event is already stopped, so a text area this menu does not
        # recognize would make Enter do nothing at all. Only reachable if a slot
        # is removed while its field still holds focus.
        logger.warning(
            "ask_user menu received a submit from an unrecognized text area",
        )

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Grow multi-select Other slots as custom answers are filled in."""
        text_area = event.text_area
        if not isinstance(text_area, AskUserTextArea):
            return
        for qw in self._question_widgets:
            if qw.owns_other_input(text_area):
                qw.sync_other_slots()
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

        # Re-validate every question, not just the one being confirmed. Answers
        # are read live, so confirming one question is not a promise that the
        # others still hold: the user can Shift+Tab back into a confirmed
        # question and edit it (`focus_input` even drops the cursor straight into
        # a checked Other field), or leave a question unconfirmed entirely and
        # Tab past it. Without this, a custom multi-select answer cleared after
        # confirming would vanish from the answer the agent receives while its
        # checkbox stays on.
        #
        # This subsumes the empty-required check for `multi_select`, and reports
        # the more specific Other-row errors in preference to the generic
        # "answer all questions" toast. Emptiness goes through
        # `_QuestionWidget.answer_is_empty`, which knows the `[]` encoding.
        for i, qw in enumerate(self._question_widgets):
            error = qw.validate_for_submit()
            if error is None and qw._required and qw.answer_is_empty():
                error = MISSING_ANSWER_TOAST
            if error is not None:
                self._confirmed[i] = False
                self._set_active_question(i)
                self.app.notify(error, severity="warning", markup=False)
                return

        self._submit()

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
        if self._can_deselect_focused_other():
            focused = self.app.focused
            if isinstance(focused, AskUserTextArea):
                question = focused._find_question_widget()
                if question is not None and question.uncheck_other_input(focused):
                    return
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

    Renders a checkbox glyph (`checkbox_checked`/`checkbox_empty`) before the
    label so the user can see which options are toggled on, independent of the
    highlight cursor. The box is what distinguishes multi-select from a
    single-choice list, whose options carry a cursor and nothing else. This is
    the source of truth for whether a choice is toggled on; `_QuestionWidget`
    reads it back rather than tracking that separately.

    Note the inherited `selected` means "the highlight cursor is on this
    option", *not* "toggled on" — read `checked` for the latter.

    The box goes in the label, not the cursor gutter: the gutter renders either
    the cursor or nothing, so a box there would disappear whenever the
    highlight sits on a checked option. The plain-bullet `_unselected_marker`
    is suppressed here so the gutter never shows a bullet next to the box.
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
        # Set before mount: `compose` reads `_checked` for the initial box glyph.
        self._checked = False
        self._checked_label: str | None = None
        self._label_widget: Static | None = None
        super().__init__(text, index, selected=selected, **kwargs)

    @property
    def checked(self) -> bool:
        """Whether this option is currently toggled on."""
        return self._checked

    @property
    def _unselected_marker(self) -> str:
        # A bullet next to the checkbox would read as a second marker; keep the
        # gutter blank so only the cursor ever appears there.
        return " "

    def compose(self) -> ComposeResult:
        # Mirror the base `compose`, but route the label through
        # `_label_content` so the checkbox glyph is present from the start.
        # `self._cursor_widget` must be set exactly as the base does, because
        # `set_state` re-renders through it.
        self._cursor_widget = Static(
            self._cursor_content(),
            classes="inline-prompt-option-cursor",
        )
        yield self._cursor_widget
        self._label_widget = Static(
            self._label_content(),
            classes="inline-prompt-option-label",
        )
        yield self._label_widget

    def set_checked(self, checked: bool) -> None:
        """Update the toggle state and re-render the label's checkbox glyph.

        Args:
            checked: Whether the option should be toggled on.
        """
        self._checked = checked
        if self._label_widget is not None:
            self._label_widget.update(self._label_content())
            # A custom Other label shrinks to only its checkbox while checked.
            # Reflow the parent so its adjacent text field reclaims the old
            # placeholder's width.
            self.refresh(layout=True)

    def set_checked_label(self, label: str) -> None:
        """Use `label` in place of this option's text while it is checked.

        Args:
            label: Checked-state label. An empty label leaves only the checkbox,
                allowing an adjacent custom-answer field to take its place.
        """
        self._checked_label = label
        if self._checked and self._label_widget is not None:
            self._label_widget.update(self._label_content())
            self.refresh(layout=True)

    def _label_content(self) -> Content:
        glyphs = get_glyphs()
        box = glyphs.checkbox_checked if self._checked else glyphs.checkbox_empty
        label = (
            self._checked_label
            if self._checked and self._checked_label is not None
            else self._text
        )
        if not label:
            return Content.from_markup("$box", box=box)
        return Content.from_markup("$box $text", box=box, text=label)


class _MultiSelectOtherEntry(Horizontal):
    """One multi-select custom Other row: checkbox paired with its text field.

    Both widgets and the row containing them are one object, so there is no way
    for a caller to pair a checkbox with the wrong field. `set_checked` is the
    only way to toggle the row, which is what keeps the field's visibility from
    drifting out of sync with the checkbox.
    """

    DEFAULT_CSS = """
    _MultiSelectOtherEntry {
        height: auto;
        width: 1fr;
        margin: 0;
        padding: 0;
    }
    """

    def __init__(
        self,
        option: _MultiSelectOption,
        text_input: AskUserTextArea,
        **kwargs: Any,
    ) -> None:
        super().__init__(classes="ask-user-other-slot", **kwargs)
        self._option = option
        self._text_input = text_input

    @property
    def option(self) -> _MultiSelectOption:
        """The checkbox for this custom answer."""
        return self._option

    @property
    def text_input(self) -> AskUserTextArea:
        """The free-text field for this custom answer."""
        return self._text_input

    @property
    def checked(self) -> bool:
        """Whether this custom answer is toggled on."""
        return self._option.checked

    def set_checked(self, checked: bool) -> None:
        """Toggle this row, revealing or hiding its text field to match.

        Args:
            checked: Whether the custom answer is toggled on. The text field is
                shown exactly when it is; a typed draft is preserved either way,
                so re-checking restores the same text.
        """
        self._option.set_checked(checked)
        self.sync_display()

    def sync_display(self) -> None:
        """Re-assert that the text field is visible exactly when the box is checked."""
        self._text_input.display = self._option.checked

    def compose(self) -> ComposeResult:
        yield self._option
        yield self._text_input


class _QuestionWidget(Vertical):
    """Widget for a single question (text, multiple choice, or multi-select)."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("k", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("j", "move_down", "Down", show=False),
        # Safe despite the text-area children: Textual resolves the focused
        # widget's `check_consume_key` before ancestor bindings, and
        # `AskUserTextArea` claims printable keys, so a space still types into a
        # focused text area rather than being swallowed here. `check_action`
        # below keeps the binding from consuming space on non-multi-select
        # questions, where the question container itself holds focus.
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
            # Belt-and-braces: `TypeAdapter(AskUserRequest)` in `textual_adapter`
            # already rejects an unknown type before the payload reaches here, so
            # this only covers direct construction (tests, a future non-TUI
            # host). Log, rather than silently falling back to a text box.
            logger.warning(
                "ask_user question %d has unrecognized type %r; rendering as text",
                index,
                question_type,
            )
            question_type = "text"
        self._q_type: QuestionType = question_type
        self._choices: list[Choice] = question.get("choices", [])
        if self._q_type in CHOICE_QUESTION_TYPES and not self._choices:
            # A defensive degrade, not a live gap: `choices` is `NotRequired`,
            # but `AskUserRequest.questions` is `list[ValidatedQuestion]`, so
            # the adapter rejects a choice question with none before this runs.
            # Degrade `_q_type` too, not just the rendering, so it stays a true
            # discriminant — otherwise the help footer advertises "Space toggle"
            # for what is actually a text box.
            logger.warning(
                "ask_user %s question %d has no choices; rendering as text",
                self._q_type,
                index,
            )
            self._q_type = "text"
        self._required: bool = question.get("required", True)
        self._choice_widgets: list[_ChoiceOption] = []
        self._multi_select_widgets: list[_MultiSelectOption] = []
        self._other_entries: list[_MultiSelectOtherEntry] = []
        self._selected_choice: int = 0
        self._text_input: AskUserTextArea | None = None
        # `multiple_choice` keeps a single Other input here. Multi-select Others
        # live in `_other_entries` instead.
        self._other_input: AskUserTextArea | None = None
        # `multiple_choice` only: whether the cursor is on its Other row. Multi-select
        # reads `entry.checked` per row instead, since several can be on at once.
        self._is_other_selected: bool = False
        self._max_other_notified: bool = False

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Disable `toggle_choice` on questions that have nothing to toggle.

        Returning `None` (rather than `False`) leaves the binding unmatched, so
        `space` bubbles instead of being silently consumed on `text` and
        `multiple_choice` questions.

        Args:
            action: Name of the action being checked.
            parameters: Action parameters, unused.

        Returns:
            `None` to leave `toggle_choice` unbound here, otherwise the base
            class's answer.
        """
        if action == "toggle_choice" and self._q_type != "multi_select":
            return None
        return super().check_action(action, parameters)

    @property
    def question_type(self) -> QuestionType:
        """Resolved type of this question."""
        return self._q_type

    def owns_other_input(self, text_area: object) -> bool:
        """Return whether `text_area` is an Other free-text input of this question."""
        if self._other_input is text_area:
            return True
        return any(entry.text_input is text_area for entry in self._other_entries)

    def compose(self) -> ComposeResult:
        q_text = _TRAILING_ANNOTATION_RE.sub("", self._question.get("question", ""))
        prefix = f"**{self._index + 1}.** " if self._show_number else ""
        markers: list[str] = []
        if self._required:
            markers.append("required")
        if self._q_type == "multi_select":
            # Distinguish toggleable multi-select from single-choice lists.
            markers.append("select all that apply")
        suffix = f" *({', '.join(markers)})*" if markers else ""
        # q_text is agent-authored; rendered as markdown intentionally so
        # agents can use inline formatting, links, and code spans in questions.
        yield Markdown(f"{prefix}{q_text}{suffix}", classes="ask-user-question-text")

        # `__init__` guarantees a choice type here has a non-empty `_choices`, so
        # these branches need no `and self._choices` co-guard. The `assert_never`
        # makes the dispatch exhaustive: a new `QuestionType` member fails type
        # checking here instead of silently falling through to a text box.
        if self._q_type == "multiple_choice":
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
        elif self._q_type == "multi_select":
            for i, choice in enumerate(self._choices):
                label = choice.get("value", "")
                msw = _MultiSelectOption(label, index=i, selected=(i == 0))
                self._choice_widgets.append(msw)
                self._multi_select_widgets.append(msw)
                yield msw
            # Never highlighted initially: `__init__` guarantees a non-empty
            # `_choices` here, so choice 0 owns the cursor.
            yield from self._compose_other_entry(selected=False)
        elif self._q_type == "text":
            self._text_input = AskUserTextArea(classes="ask-user-text-input")
            yield self._text_input
        else:
            assert_never(self._q_type)

    @staticmethod
    def _other_label(entry_index: int) -> str:
        return OTHER_CHOICE_LABEL if entry_index == 0 else ADD_ANOTHER_OTHER_LABEL

    def _make_other_entry(self, *, selected: bool) -> _MultiSelectOtherEntry:
        """Build one multi-select Other checkbox + free-text pair.

        Args:
            selected: Whether the new checkbox row starts with the highlight.

        Returns:
            The entry tracking widgets for toggle/answer collection.
        """
        entry_index = len(self._other_entries)
        option = _MultiSelectOption(
            self._other_label(entry_index),
            index=len(self._choices) + entry_index,
            selected=selected,
        )
        text_input = AskUserTextArea(classes="ask-user-other-input")
        text_input.display = False
        option.set_checked_label("")
        option.add_class("ask-user-inline-other-choice")
        self._choice_widgets.append(option)
        entry = _MultiSelectOtherEntry(option, text_input)
        self._other_entries.append(entry)
        return entry

    def _compose_other_entry(self, *, selected: bool) -> ComposeResult:
        """Yield one multi-select Other slot (checkbox + free-text)."""
        yield self._make_other_entry(selected=selected)

    def _mount_other_entry(self) -> None:
        """Dynamically append another multi-select Other slot."""
        if self._q_type != "multi_select":
            return
        # Defensive: `sync_other_slots` owns the cap decision (and explains it to
        # the user); this keeps any other caller from exceeding it.
        if len(self._other_entries) >= MAX_MULTI_SELECT_OTHER_ENTRIES:
            return
        self.mount(self._make_other_entry(selected=False))

    def _other_entry_at(self, cursor: int) -> _MultiSelectOtherEntry | None:
        """Return the Other entry a cursor row maps to, if any.

        The choice list is the predefined choices followed by the Other entries,
        so this is the one place that translates between the two index spaces.

        Args:
            cursor: Index into `_choice_widgets`.

        Returns:
            The entry at `cursor`, or `None` when `cursor` is a predefined choice
            or out of range.
        """
        index = cursor - len(self._choices)
        if 0 <= index < len(self._other_entries):
            return self._other_entries[index]
        return None

    @staticmethod
    def _entry_custom_text(entry: _MultiSelectOtherEntry) -> str:
        # `submitted_value` (not `.text`) so a collapsed paste counts as its
        # expanded content rather than the `[Pasted text #N]` placeholder,
        # matching what `get_answer` and `validate_for_submit` read.
        return entry.text_input.submitted_value.strip()

    def _entry_is_filled(self, entry: _MultiSelectOtherEntry) -> bool:
        return bool(entry.checked and self._entry_custom_text(entry))

    def sync_other_slots(self) -> None:
        """Keep exactly one empty trailing Other slot after filled customs.

        Filling a checked Other (non-empty text) reveals an "Add another" row.
        Trailing empty unchecked slots beyond that spare are removed when the
        previous custom is cleared.

        The spare stops appearing at `MAX_MULTI_SELECT_OTHER_ENTRIES`, which is
        explained once via `MAX_OTHER_ENTRIES_TOAST` rather than left to look
        like a broken UI.
        """
        if self._q_type != "multi_select":
            return
        if not self._other_entries:
            # `compose` has not run yet, so there is no slot layout to sync.
            return

        # Drop extra trailing empty slots while the last filled boundary allows.
        # Entry 0 is never removed, so `_other_entries` stays non-empty.
        while len(self._other_entries) > 1:
            last = self._other_entries[-1]
            if last.checked or last.text_input.text.strip():
                break
            # Keep a single spare empty slot only when the prior entry is filled.
            if self._entry_is_filled(self._other_entries[-2]):
                break
            self._remove_last_other_entry()

        if not self._entry_is_filled(self._other_entries[-1]):
            return
        if len(self._other_entries) < MAX_MULTI_SELECT_OTHER_ENTRIES:
            self._mount_other_entry()
            return
        # At the cap. Notify once: this runs on every keystroke, so an
        # unconditional toast would spam a user who keeps typing in the last slot.
        if not self._max_other_notified:
            self._max_other_notified = True
            self.app.notify(
                MAX_OTHER_ENTRIES_TOAST,
                severity="warning",
                markup=False,
            )

    def _remove_last_other_entry(self) -> None:
        entry = self._other_entries.pop()
        # `_make_other_entry` always appends the option, so it is present.
        self._choice_widgets.remove(entry.option)
        entry.remove()
        if self._selected_choice >= len(self._choice_widgets):
            self._selected_choice = max(0, len(self._choice_widgets) - 1)
            self._update_choice_selection(focus_other_input=False)

    def focus_input(self) -> None:
        """Focus the appropriate input for this question."""
        if self._text_input:
            self._text_input.focus()
            return
        if (
            self._q_type == "multiple_choice"
            and self._is_other_selected
            and self._other_input
        ):
            self._other_input.focus()
            return
        if self._q_type == "multi_select":
            entry = self._other_entry_at(self._selected_choice)
            if entry is not None and entry.checked:
                # Only pull focus into that row's free-text when the cursor is
                # already on it and it is checked.
                entry.text_input.focus()
                return
        if self._choice_widgets:
            self.focus()
            return
        # Unreachable: `__init__` degrades a choice question with no choices to
        # `text`, so one of the branches above always fires. Log rather than
        # leaving a prompt the user silently cannot interact with.
        logger.warning(
            "ask_user question %d has no focusable input (type %r)",
            self._index,
            self._q_type,
        )

    def get_answer(self) -> str:
        """Return the current answer text for this question.

        For text and "Other" answers, collapsed-paste placeholders are expanded
        so the agent receives the full pasted content, not the compact
        `[Pasted text #N]` token.

        A multi-select answer is a JSON array (via `encode_multi_select_answer`)
        of the toggled predefined values in choice-list order, then each filled
        custom Other value in slot order — `[]` when nothing is toggled. JSON
        is self-delimiting, so values carrying commas, quotes, or newlines
        round-trip exactly through `decode_multi_select_answer`. A custom value
        duplicating an already-selected one is dropped, so the agent never sees
        the same selection twice.
        """
        if self._q_type == "text":
            return self._text_input.submitted_value if self._text_input else ""

        if self._q_type == "multi_select":
            selected: list[str] = [
                self._choices[widget.option_index].get("value", "")
                for widget in self._multi_select_widgets
                if widget.checked
            ]
            for entry in self._other_entries:
                if not entry.checked:
                    continue
                other_text = self._entry_custom_text(entry)
                if other_text and other_text not in selected:
                    selected.append(other_text)
            return encode_multi_select_answer(selected)

        if self._is_other_selected and self._other_input:
            return self._other_input.submitted_value

        if self._choice_widgets and self._selected_choice < len(self._choices):
            return self._choices[self._selected_choice].get("value", "")

        return ""

    def answer_is_empty(self) -> bool:
        """Return whether the current answer counts as "no answer".

        Defers to `ask_user_answer_is_empty` so this agrees with the
        authorization path, which applies the same rule to the answer once it
        is over the wire.

        Returns:
            `True` when the question has no substantive answer — an empty
            multi-select selection, or a blank answer of any other type.
        """
        return ask_user_answer_is_empty(self.get_answer(), self._q_type)

    def validate_for_submit(self) -> str | None:
        """Return a user-facing error if this question cannot be submitted yet.

        Called both when this question is confirmed and again for every question
        on the final submit (`AskUserMenu.confirm_and_advance`), because answers
        are read live and a confirmed question can still be edited afterwards.

        Returns:
            A toast message when the current draft is incomplete or ambiguous,
            otherwise `None`.
        """
        if self._q_type != "multi_select":
            return None
        for entry in self._other_entries:
            if not entry.checked:
                continue
            other_text = entry.text_input.submitted_value
            if not other_text.strip():
                return MISSING_OTHER_TEXT_TOAST
        if self._required and self.answer_is_empty():
            return MISSING_ANSWER_TOAST
        return None

    def _focused_other_entry_index(self) -> int | None:
        """Return the index in `_other_entries` of the focused custom field, if any.

        Multi-select only. Kept separate from `_multiple_choice_other_focused`
        because that type's single Other field lives in `_other_input` and has no
        index — folding both into one `int` return would conflate two different
        index spaces.
        """
        for i, entry in enumerate(self._other_entries):
            if entry.text_input.has_focus:
                return i
        return None

    def _multiple_choice_other_focused(self) -> bool:
        """Return whether this `multiple_choice` question's Other field has focus."""
        return self._other_input is not None and self._other_input.has_focus

    def uncheck_other_input(self, text_input: AskUserTextArea) -> bool:
        """Uncheck the custom multi-select option that owns `text_input`.

        This is the Escape shortcut for a focused custom answer. It returns focus
        to the question container (which owns the choice bindings, since options
        are not individually focusable), so a second Escape cancels the prompt.

        Args:
            text_input: Focused custom-answer field requesting deselection.

        Returns:
            `True` when a checked custom option was deselected, otherwise
            `False`.
        """
        if self._q_type != "multi_select":
            return False
        if not self.can_uncheck_other_input(text_input):
            return False
        for entry_index, entry in enumerate(self._other_entries):
            if entry.text_input is text_input:
                self._selected_choice = len(self._choices) + entry_index
                self.action_toggle_choice()
                # Focusing the checkbox posts a blur event, but that arrives on
                # the next message-loop turn. Refresh now so the Esc hint never
                # lags behind the newly deselected state.
                menu = self._find_menu()
                if menu is not None:
                    menu._update_help()
                return True
        return False

    def can_uncheck_other_input(self, text_input: AskUserTextArea) -> bool:
        """Return whether `text_input` belongs to a checked custom option."""
        return self._q_type == "multi_select" and any(
            entry.text_input is text_input and entry.checked
            for entry in self._other_entries
        )

    def action_move_up(self) -> None:
        """Move the highlight cursor up in the choice list."""
        if self._q_type not in CHOICE_QUESTION_TYPES or not self._choice_widgets:
            return
        if self._q_type == "multi_select":
            focused_other = self._focused_other_entry_index()
            if focused_other is not None:
                # Land on that Other checkbox row so Space can uncheck it.
                self._selected_choice = len(self._choices) + focused_other
                self._update_choice_selection(focus_other_input=False)
                self.focus()
                return
        elif self._multiple_choice_other_focused():
            # Jump directly to the last real choice instead of requiring two
            # presses (one to defocus, one to navigate).
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
        if self._q_type == "multi_select":
            focused_other = self._focused_other_entry_index()
            if focused_other is not None:
                # From an Other free-text box, Down jumps to the next Other
                # checkbox (the "Add another" row once the current slot is
                # filled), or no-ops at the final free-text span.
                next_row = len(self._choices) + focused_other + 1
                if next_row < len(self._choice_widgets):
                    self._selected_choice = next_row
                    self._update_choice_selection(focus_other_input=False)
                    self.focus()
                return
        elif self._multiple_choice_other_focused():
            # Other is the last row, so there is nowhere below it to move.
            return
        max_idx = len(self._choice_widgets) - 1
        old = self._selected_choice
        self._selected_choice = min(max_idx, self._selected_choice + 1)
        if old != self._selected_choice:
            self._update_choice_selection()

    def action_toggle_choice(self) -> None:
        """Toggle the highlighted choice for `multi_select` questions."""
        if self._q_type != "multi_select" or not self._choice_widgets:
            return
        index = self._selected_choice
        if not 0 <= index < len(self._choice_widgets):
            # Do not clamp into range and toggle anyway: that would flip a
            # checkbox the user never highlighted, and leave the cursor pointing
            # somewhere else. Refuse the keypress instead — the answer stays as
            # the user last saw it, and pressing Space again is harmless.
            logger.error(
                "multi_select question %d cursor %d outside 0..%d toggleable choices",
                self._index,
                index,
                len(self._choice_widgets) - 1,
            )
            return

        if index >= len(self._choices):
            entry = self._other_entry_at(index)
            if entry is None:
                # `_choice_widgets` and `_choices + _other_entries` have diverged;
                # same invariant break as the cursor check above, so say so rather
                # than presenting as "Space does nothing".
                logger.error(
                    "multi_select question %d cursor %d has no Other entry "
                    "(%d choices, %d entries)",
                    self._index,
                    index,
                    len(self._choices),
                    len(self._other_entries),
                )
                return
            checked = not entry.checked
            entry.set_checked(checked)
            if checked:
                entry.text_input.focus()
            else:
                # Preserve typed draft so re-checking restores the same text;
                # only collapse spare slots shown below.
                self.focus()
            self.sync_other_slots()
            return

        widget = self._multi_select_widgets[index]
        widget.set_checked(not widget.checked)

    async def _on_key(self, event: events.Key) -> None:
        """Start a highlighted custom answer when its first character is typed.

        A multi-select Other row is normally a checkbox, so it needs Space to
        reveal its adjacent text field. Treating the first printable character
        as that toggle makes the row behave like a focused text field without
        changing Space's established checkbox behavior.
        """
        if (
            event.is_printable
            and event.character is not None
            and event.key != "space"
            and self._q_type == "multi_select"
        ):
            entry = self._other_entry_at(self._selected_choice)
            if entry is not None and not entry.checked:
                self.action_toggle_choice()
                entry.text_input.insert(event.character)
                event.prevent_default()
                event.stop()
                return
        await super()._on_key(event)

    def action_select_or_submit(self) -> None:
        """Confirm the current answer, or open the Other input.

        For multi-select, warns and keeps the question open while a required
        question has nothing toggled, an Other row is checked without text, or
        custom Other text would make the joined answer ambiguous.
        """
        if self._q_type == "multi_select":
            self.sync_other_slots()
            error = self.validate_for_submit()
            if error is not None:
                self.app.notify(error, severity="warning", markup=False)
                return
            menu = self._find_menu()
            if menu is None:
                # `_find_menu` logs, but a bare return would leave the user
                # pressing Enter with no advance and no explanation while the
                # agent stays blocked on the interrupt.
                self.app.notify(
                    UNSUBMITTABLE_ANSWER_TOAST,
                    severity="error",
                    markup=False,
                )
                return
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

    def _update_choice_selection(self, *, focus_other_input: bool = True) -> None:
        """Sync highlight cursors and Other-input visibility after a move.

        Args:
            focus_other_input: When true, focusing an Other free-text input is
                allowed (multiple_choice when the cursor is on Other; multi-select
                when that Other slot is checked and the cursor lands on it).
                Callers that are deliberately leaving a free-text box pass False
                so Up does not immediately re-enter it.
        """
        for i, cw in enumerate(self._choice_widgets):
            if i == self._selected_choice:
                cw.select()
            else:
                cw.deselect()

        if self._q_type == "multiple_choice":
            is_other = self._selected_choice == len(self._choices)
            self._is_other_selected = is_other
            if self._other_input:
                self._other_input.display = is_other
                if is_other and focus_other_input:
                    self._other_input.focus()
            return

        if self._q_type == "multi_select":
            # `_is_other_selected` is deliberately not maintained here: it is a
            # `multiple_choice` concept (one cursor row), and every reader is
            # gated on that type. Multi-select rows carry their own `checked`.
            for i, entry in enumerate(self._other_entries):
                entry.sync_display()
                if (
                    focus_other_input
                    and entry.checked
                    and self._selected_choice == len(self._choices) + i
                ):
                    entry.text_input.focus()
