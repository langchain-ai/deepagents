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

from deepagents_code import theme
from deepagents_code.config import get_glyphs, is_ascii_mode
from deepagents_code.widgets.ask_user import AskUserTextArea

_ACCEPT_OPTION_INDEX = 0
_EDIT_OPTION_INDEX = 1
_REJECT_OPTION_INDEX = 2
_CANCEL_OPTION_INDEX = 3
_OPTION_COUNT = 4


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


class GoalReviewTextArea(AskUserTextArea):
    """Text input that keeps goal-review edit keystrokes inside the editor."""

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

        def __init__(self, result: GoalReviewResult) -> None:
            """Initialize a decision message."""
            super().__init__()
            self.result = result
            """Decision payload emitted by the review widget."""

    def __init__(
        self,
        objective: str,
        criteria: str,
        id: str | None = None,  # noqa: A002
    ) -> None:
        """Initialize the goal review menu."""
        super().__init__(id=id or "goal-review-menu", classes="goal-review-menu")
        self._objective = objective
        """Goal objective whose generated criteria are being reviewed."""

        self._criteria = criteria
        """Generated acceptance criteria proposed for the goal."""

        self._selected = _ACCEPT_OPTION_INDEX
        """Index of the currently highlighted action option."""

        self._option_widgets: list[Static] = []
        """Mounted option widgets updated when selection changes."""

        self._help_widget: Static | None = None
        """Mounted keyboard-help widget, populated during composition."""

        self._edit_input: GoalReviewTextArea | None = None
        """Inline editor used for revised criteria or rejection feedback."""

        self._input_mode: Literal["edit", "reject"] | None = None
        """Whether an inline text input is currently active."""

        self._future: asyncio.Future[GoalReviewResult] | None = None
        """Future resolved when the user accepts, edits, or cancels."""

        self._submitted = False
        """Whether a terminal decision has already been emitted."""

    def set_future(self, future: asyncio.Future[GoalReviewResult]) -> None:
        """Set the future to resolve when the user decides."""
        self._future = future

    def compose(self) -> ComposeResult:
        """Compose the review widget.

        Yields:
            Widgets for the title, criteria preview, actions, editor, and help text.
        """
        glyphs = get_glyphs()
        yield Static(
            Content.from_markup("$cursor Review goal criteria", cursor=glyphs.cursor),
            classes="goal-review-title",
        )
        with (
            VerticalScroll(classes="goal-review-content"),
            Vertical(classes="goal-review-body"),
        ):
            yield Markdown(
                f"**Proposed criteria**\n\n{self._criteria}",
                classes="goal-review-markdown",
            )
        with Container(classes="goal-review-options-container"):
            for _ in range(_OPTION_COUNT):
                widget = Static("", classes="goal-review-option")
                self._option_widgets.append(widget)
                yield widget
        self._edit_input = GoalReviewTextArea(classes="goal-review-edit-input")
        self._edit_input.text = self._criteria
        self._edit_input.display = False
        yield self._edit_input
        self._help_widget = Static("", classes="goal-review-help")
        yield self._help_widget

    async def on_mount(self) -> None:
        """Focus the menu and render options after mount."""
        if is_ascii_mode():
            colors = theme.get_theme_colors(self)
            self.styles.border = ("ascii", colors.success)
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
        self._selected = (self._selected - 1) % _OPTION_COUNT
        self._update_options()

    def action_move_down(self) -> None:
        """Move selection down."""
        if self._input_mode is not None:
            return
        self._selected = (self._selected + 1) % _OPTION_COUNT
        self._update_options()

    def action_select(self) -> None:
        """Select the highlighted option."""
        if self._input_mode is not None:
            return
        if self._selected == _ACCEPT_OPTION_INDEX:
            self.action_accept()
        elif self._selected == _EDIT_OPTION_INDEX:
            self.action_edit()
        elif self._selected == _REJECT_OPTION_INDEX:
            self.action_reject_with_message()
        elif self._selected == _CANCEL_OPTION_INDEX:
            self.action_cancel()

    def action_accept(self) -> None:
        """Accept the proposed criteria unchanged."""
        if self._input_mode is not None:
            return
        self._submit({"type": "accepted"})

    def action_edit(self) -> None:
        """Open the inline editor for revised criteria."""
        if self._submitted or self._input_mode is not None:
            return
        self._input_mode = "edit"
        if self._edit_input is not None:
            self._edit_input.text = self._criteria
            self._edit_input.display = True
            self._edit_input.focus()
        self._update_options()

    def action_reject_with_message(self) -> None:
        """Open the inline feedback input for regenerating criteria."""
        if self._submitted or self._input_mode is not None:
            return
        self._input_mode = "reject"
        if self._edit_input is not None:
            self._edit_input.text = ""
            self._edit_input.display = True
            self._edit_input.focus()
        self._update_options()

    def action_cancel(self) -> None:
        """Cancel editing or cancel the whole proposal."""
        if self._submitted:
            return
        if self._input_mode is not None:
            self._input_mode = None
            if self._edit_input is not None:
                self._edit_input.display = False
            self._update_options()
            self.focus()
            return
        self._submit({"type": "cancelled"})

    def on_ask_user_text_area_submitted(
        self,
        event: AskUserTextArea.Submitted,
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
        event.stop()

    def _submit_edit(self) -> None:
        """Submit the current editor text as revised criteria."""
        if self._edit_input is None:
            return
        criteria = self._edit_input.text.strip()
        if not criteria:
            return
        self._submit({"type": "edited", "criteria": criteria})

    def _submit_rejection(self) -> None:
        """Submit the current editor text as regeneration feedback."""
        if self._edit_input is None:
            return
        message = self._edit_input.text.strip()
        if not message:
            return
        self._submit({"type": "rejected", "message": message})

    def _submit(self, result: GoalReviewResult) -> None:
        """Resolve the result future and post the decision message."""
        if self._submitted:
            return
        self._submitted = True
        self.display = False
        if self._future is not None and not self._future.done():
            self._future.set_result(result)
        self.post_message(self.Decided(result))

    def _update_options(self) -> None:
        """Render option labels and help text."""
        options = [
            "1. Accept proposed criteria (y)",
            "2. Edit criteria (e)",
            "3. Reject with message (r)",
            "4. Cancel (n)",
        ]
        for i, (text, widget) in enumerate(
            zip(options, self._option_widgets, strict=True)
        ):
            cursor = f"{get_glyphs().cursor} " if i == self._selected else "  "
            widget.update(f"{cursor}{text}")
            widget.remove_class("goal-review-option-selected")
            if i == self._selected and self._input_mode is None:
                widget.add_class("goal-review-option-selected")

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
