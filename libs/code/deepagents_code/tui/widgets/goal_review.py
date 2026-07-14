"""Goal acceptance-criteria review widget."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict

from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.widgets import Markdown, Static

if TYPE_CHECKING:
    import asyncio

    from textual import events
    from textual.app import ComposeResult

from deepagents_code.config import get_glyphs
from deepagents_code.tui.widgets._inline_prompt import (
    InlinePromptCompletion,
    InlinePromptOption,
    InlinePromptTextArea,
    apply_inline_prompt_border,
    stop_inline_prompt_blur,
)

# Menu options in display order: (label, `action_*` suffix). The list index is
# the cursor position, so labels and dispatch stay aligned from one source.
_OPTIONS: tuple[tuple[str, str], ...] = (
    ("1. Accept proposed criteria (y)", "accept"),
    ("2. Edit criteria (e)", "edit"),
    ("3. Reject with message (r)", "reject_with_message"),
    ("4. Cancel (n)", "cancel"),
)


class GoalReviewAccepted(TypedDict):
    """Widget result when the generated criteria are accepted unchanged."""

    type: Literal["accepted"]
    """Discriminator tag for accepting generated criteria unchanged."""


class GoalReviewEdited(TypedDict):
    """Widget result when the user submits revised criteria."""

    type: Literal["edited"]
    """Discriminator tag for submitting revised criteria."""

    criteria: str
    """User-edited acceptance criteria to activate for the goal."""


class GoalReviewRejected(TypedDict):
    """Widget result when the user rejects criteria with feedback."""

    type: Literal["rejected"]
    """Discriminator tag for regenerating criteria from user feedback."""

    message: str
    """User feedback explaining how the criteria should be regenerated."""


class GoalReviewCancelled(TypedDict):
    """Widget result when the user cancels the proposal."""

    type: Literal["cancelled"]
    """Discriminator tag for cancelling the pending goal proposal."""


GoalReviewResult = (
    GoalReviewAccepted | GoalReviewEdited | GoalReviewRejected | GoalReviewCancelled
)


class GoalReviewTextArea(InlinePromptTextArea):
    """Text input that keeps goal-review edit keystrokes inside the editor."""

    class Submitted(InlinePromptTextArea.Submitted):
        """Posted when the user presses Enter to submit goal-review text."""

    class CancelEdit(Message):
        """Posted when Escape should leave goal criteria edit mode."""

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.prevent_default()
            event.stop()
            self.post_message(self.CancelEdit())
            return

        await super()._on_key(event)


class GoalReviewMenu(Container):
    """Inline review widget for generated goal acceptance criteria."""

    can_focus = True
    """Allow the menu itself to receive navigation and quick-key focus."""

    can_focus_children = True
    """Allow the inline criteria editor to receive text input focus."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("k", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("j", "move_down", "Down", show=False),
        Binding("enter", "select", "Select", show=False),
        Binding("1", "accept", "Accept", show=False),
        Binding("y", "accept", "Accept", show=False),
        Binding("2", "edit", "Edit", show=False),
        Binding("e", "edit", "Edit", show=False),
        Binding("3", "reject_with_message", "Reject with message", show=False),
        Binding("r", "reject_with_message", "Reject with message", show=False),
        Binding("4", "cancel", "Cancel", show=False),
        Binding("n", "cancel", "Cancel", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]
    """Keyboard bindings for navigation, accepting, editing, and cancelling."""

    class Decided(Message):
        """Message sent when the user accepts, edits, or cancels."""

        def __init__(
            self,
            result: GoalReviewResult,
            widget: GoalReviewMenu | None = None,
        ) -> None:
            """Initialize a decision message."""
            super().__init__()
            self.result = result
            """Decision payload emitted by the review widget."""

            self.widget = widget
            """Review widget that emitted the decision."""

    def __init__(
        self,
        objective: str,
        criteria: str,
        *,
        amendment: bool = False,
        id: str | None = None,  # noqa: A002
    ) -> None:
        """Initialize the goal review menu."""
        super().__init__(
            id=id or "goal-review-menu",
            classes="inline-prompt goal-review-menu",
        )
        self._objective = objective
        """Goal objective whose generated criteria are being reviewed."""

        self._criteria = criteria
        """Generated acceptance criteria proposed for the goal."""

        self._amendment = amendment
        """Whether this review updates an existing goal."""

        self._selected = 0
        """Index of the currently highlighted action option."""

        self._option_widgets: list[InlinePromptOption] = []
        """Mounted option widgets updated when selection changes."""

        self._help_widget: Static | None = None
        """Mounted keyboard-help widget, populated during composition."""

        self._edit_input: GoalReviewTextArea | None = None
        """Inline editor used for revised criteria or rejection feedback."""

        self._input_mode: Literal["edit", "reject"] | None = None
        """Whether an inline text input is currently active."""

        self._completion: InlinePromptCompletion[GoalReviewResult] = (
            InlinePromptCompletion()
        )
        """One-shot resolver for accepted, edited, rejected, or cancelled results."""

    def set_future(self, future: asyncio.Future[GoalReviewResult]) -> None:
        """Set the future to resolve when the user decides."""
        self._completion.set_future(future)

    def compose(self) -> ComposeResult:
        """Compose the review widget.

        Yields:
            Widgets for the title, criteria preview, actions, editor, and help text.
        """
        glyphs = get_glyphs()
        title = "Review goal amendment" if self._amendment else "Review goal criteria"
        yield Static(
            Content.from_markup("$cursor $title", cursor=glyphs.cursor, title=title),
            classes="inline-prompt-title goal-review-title",
        )
        with (
            VerticalScroll(classes="goal-review-content"),
            Vertical(classes="goal-review-body"),
        ):
            source = f"**Proposed criteria**\n\n{self._criteria}"
            if self._amendment:
                source = (
                    f"**Proposed objective**\n\n{self._objective}\n\n"
                    f"**Proposed criteria**\n\n{self._criteria}"
                )
            yield Markdown(source, classes="goal-review-markdown")
        with Container(classes="goal-review-options-container"):
            for i, (label, _) in enumerate(_OPTIONS):
                widget = InlinePromptOption(
                    label,
                    i,
                    selected=i == self._selected,
                    selected_class="goal-review-option-selected",
                    classes="goal-review-option",
                )
                self._option_widgets.append(widget)
                yield widget
        self._edit_input = GoalReviewTextArea(classes="goal-review-edit-input")
        self._edit_input.text = self._criteria
        self._edit_input.display = False
        yield self._edit_input
        self._help_widget = Static(
            "",
            classes="inline-prompt-help goal-review-help",
        )
        yield self._help_widget

    async def on_mount(self) -> None:
        """Focus the menu and render options after mount."""
        apply_inline_prompt_border(self)
        self._update_options()
        self.focus()

    def focus_active(self) -> None:
        """Focus the active control."""
        if self._input_mode is not None and self._edit_input is not None:
            self._edit_input.focus()
            return
        self.focus()

    def action_move_up(self) -> None:
        """Move selection up."""
        if self._input_mode is not None:
            return
        self._selected = (self._selected - 1) % len(_OPTIONS)
        self._update_options()

    def action_move_down(self) -> None:
        """Move selection down."""
        if self._input_mode is not None:
            return
        self._selected = (self._selected + 1) % len(_OPTIONS)
        self._update_options()

    def action_select(self) -> None:
        """Select the highlighted option."""
        if self._input_mode is not None:
            return
        action_name = _OPTIONS[self._selected][1]
        getattr(self, f"action_{action_name}")()

    def action_accept(self) -> None:
        """Accept the proposed criteria unchanged."""
        if self._input_mode is not None:
            return
        self._submit({"type": "accepted"})

    def action_edit(self) -> None:
        """Open the inline editor for revised criteria."""
        if self._completion.resolved or self._input_mode is not None:
            return
        self._input_mode = "edit"
        if self._edit_input is not None:
            self._edit_input.text = self._criteria
            self._edit_input.display = True
            self._edit_input.focus()
        self._update_options()

    def action_reject_with_message(self) -> None:
        """Open the inline feedback input for regenerating criteria."""
        if self._completion.resolved or self._input_mode is not None:
            return
        self._input_mode = "reject"
        if self._edit_input is not None:
            self._edit_input.text = ""
            self._edit_input.display = True
            self._edit_input.focus()
        self._update_options()

    def action_cancel(self) -> None:
        """Cancel editing or cancel the whole proposal."""
        if self._completion.resolved:
            return
        if self._input_mode is not None:
            self._input_mode = None
            if self._edit_input is not None:
                self._edit_input.display = False
            self._update_options()
            self.focus()
            return
        self._submit({"type": "cancelled"})

    def on_goal_review_text_area_submitted(
        self,
        event: GoalReviewTextArea.Submitted,
    ) -> None:
        """Submit edited criteria when Enter is pressed in the editor."""
        if event.text_area is not self._edit_input:
            return
        event.stop()
        if self._input_mode == "edit":
            self._submit_edit()
            return
        if self._input_mode == "reject":
            self._submit_rejection()

    def on_goal_review_text_area_cancel_edit(
        self,
        event: GoalReviewTextArea.CancelEdit,
    ) -> None:
        """Return from edit mode when Escape is pressed in the editor."""
        event.stop()
        self.action_cancel()

    def on_blur(self, event: events.Blur) -> None:  # noqa: PLR6301  # Textual event handler
        """Prevent blur from dismissing the review prompt."""
        stop_inline_prompt_blur(event)

    def _submit_edit(self) -> None:
        """Submit the current editor text as revised criteria."""
        if self._edit_input is None:
            return
        criteria = self._edit_input.text.strip()
        if not criteria:
            self._hint_empty_submission("criteria")
            return
        self._submit({"type": "edited", "criteria": criteria})

    def _submit_rejection(self) -> None:
        """Submit the current editor text as regeneration feedback."""
        if self._edit_input is None:
            return
        message = self._edit_input.text.strip()
        if not message:
            self._hint_empty_submission("feedback")
            return
        self._submit({"type": "rejected", "message": message})

    def _hint_empty_submission(self, what: str) -> None:
        """Explain why an empty editor submission did nothing.

        Without this the editor silently no-ops on an empty Enter, leaving the
        user unsure whether the keypress registered.

        Args:
            what: Noun for the missing content (e.g. `criteria`, `feedback`).
        """
        if self._help_widget is None:
            return
        glyphs = get_glyphs()
        self._help_widget.update(
            f"Enter some {what}, or press Esc to go back {glyphs.bullet} "
            "Shift+Enter newline"
        )

    def _submit(self, result: GoalReviewResult) -> None:
        """Resolve the result future and post the decision message."""
        if self._completion.resolved:
            return
        self.display = False
        if self._completion.resolve(result):
            self.post_message(self.Decided(result, self))

    def _update_options(self) -> None:
        """Render option labels and help text."""
        for i, widget in enumerate(self._option_widgets):
            widget.set_state(
                cursor=i == self._selected,
                highlighted=i == self._selected and self._input_mode is None,
            )

        if self._help_widget is None:
            return
        glyphs = get_glyphs()
        if self._input_mode == "edit":
            self._help_widget.update(
                f"Enter save edits {glyphs.bullet} "
                f"Shift+Enter newline {glyphs.bullet} Esc back"
            )
            return
        if self._input_mode == "reject":
            self._help_widget.update(
                f"Enter regenerate {glyphs.bullet} "
                f"Shift+Enter newline {glyphs.bullet} Esc back"
            )
            return
        self._help_widget.update(
            f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate {glyphs.bullet} "
            f"Enter select {glyphs.bullet} y/e/r/n quick keys {glyphs.bullet} "
            "Esc cancel"
        )
