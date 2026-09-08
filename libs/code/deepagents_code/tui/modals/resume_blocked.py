"""Action modal shown when policy blocks a launch-time thread resume."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_code.config import get_glyphs
from deepagents_code.unicode_security import sanitize_control_chars

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Click


ResumeBlockedChoice = Literal["new", "exit"]
"""Outcome of the blocked-resume action modal."""


class _ChoiceOption(Static):
    """One selectable action row."""

    def __init__(self, choice: ResumeBlockedChoice, label: str) -> None:
        super().__init__(markup=False, classes="resume-blocked-choice")
        self.choice = choice
        self._label = label
        self._selected = False

    def set_selected(self, selected: bool) -> None:
        """Update cursor styling for this row."""
        self._selected = selected
        self.set_class(selected, "-selected")
        self.update(self._render())

    def _render(self) -> Content:
        """Render the row with the active terminal's cursor glyph.

        Returns:
            The plain-text action row.
        """
        cursor = get_glyphs().cursor if self._selected else " "
        return Content(f"{cursor} {self._label}")

    def on_click(self, event: Click) -> None:  # noqa: PLR6301
        """Keep action activation keyboard-only."""
        event.stop()


class ResumeBlockedScreen(ModalScreen[ResumeBlockedChoice]):
    """Ask whether to start a new session or exit after a blocked resume."""

    can_focus = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "exit_app", "Exit", show=False, priority=True),
        Binding("up", "move_up", "Up", show=False, priority=True),
        Binding("k", "move_up", "Up", show=False, priority=True),
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("j", "move_down", "Down", show=False, priority=True),
        Binding("tab", "move_down", "Next", show=False, priority=True),
        Binding("shift+tab", "move_up", "Previous", show=False, priority=True),
        Binding("enter", "activate", "Select", show=False, priority=True),
    ]

    CSS = """
    ResumeBlockedScreen {
        align: center middle;
    }

    ResumeBlockedScreen > Vertical {
        width: 72;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $warning;
        padding: 1 2;
    }

    ResumeBlockedScreen .resume-blocked-title {
        text-style: bold;
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }

    ResumeBlockedScreen .resume-blocked-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    ResumeBlockedScreen .resume-blocked-choice {
        height: auto;
        padding: 0 1;
        color: $text;
    }

    ResumeBlockedScreen .resume-blocked-choice.-selected {
        background: $surface-lighten-1;
    }

    ResumeBlockedScreen .resume-blocked-help {
        height: auto;
        color: $text-muted;
        text-style: italic;
        text-align: center;
        margin-top: 1;
    }
    """

    def __init__(self, reason: str) -> None:
        """Initialize the prompt with the policy-safe resume failure reason."""
        super().__init__()
        self._reason = sanitize_control_chars(reason)
        self._options: list[_ChoiceOption] = []
        self._selected = 0

    def compose(self) -> ComposeResult:
        """Compose the blocked-resume prompt.

        Yields:
            Title, explanation, and keyboard help.
        """
        glyphs = get_glyphs()
        with Vertical():
            yield Static(
                "Thread can't be resumed",
                classes="resume-blocked-title",
                markup=False,
            )
            yield Static(
                f"{self._reason}\n\nChoose what to do next.",
                classes="resume-blocked-body",
                markup=False,
            )
            for choice, label in (
                ("new", "Start a new session"),
                ("exit", "Exit"),
            ):
                option = _ChoiceOption(choice, label)
                self._options.append(option)
                yield option
            yield Static(
                f"{glyphs.arrow_up}/{glyphs.arrow_down}: navigate "
                f"{glyphs.separator} Enter: select {glyphs.separator} Esc: exit",
                classes="resume-blocked-help",
                markup=False,
            )

    def on_mount(self) -> None:
        """Focus the modal and select the fresh-session action."""
        self.focus()
        self._options[0].set_selected(selected=True)

    def _move(self, offset: int) -> None:
        """Move the selection by the requested offset."""
        self._options[self._selected].set_selected(selected=False)
        self._selected = (self._selected + offset) % len(self._options)
        self._options[self._selected].set_selected(selected=True)

    def action_move_up(self) -> None:
        """Move the selection to the previous action."""
        self._move(-1)

    def action_move_down(self) -> None:
        """Move the selection to the next action."""
        self._move(1)

    def action_activate(self) -> None:
        """Resolve with the selected action."""
        self.dismiss(self._options[self._selected].choice)

    def action_exit_app(self) -> None:
        """Stop the launch."""
        self.dismiss("exit")

    def action_cancel(self) -> None:
        """Treat app-level cancellation as an exit choice."""
        self.action_exit_app()
