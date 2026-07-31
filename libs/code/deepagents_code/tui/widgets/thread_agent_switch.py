"""Confirmation prompt for resuming a thread owned by another agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


ThreadAgentSwitchChoice = Literal["switch", "cancel"]
"""Outcome of the cross-agent thread resume prompt."""


class ThreadAgentSwitchPromptScreen(ModalScreen[ThreadAgentSwitchChoice]):
    """Ask before changing agents to resume a thread."""

    can_focus = True
    can_focus_children = False

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "switch", "Switch", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        Binding(
            "ctrl+c",
            "quit_or_interrupt",
            "Quit/Interrupt",
            show=False,
            priority=True,
        ),
        Binding("ctrl+d", "quit_app", "Quit", show=False, priority=True),
    ]

    CSS = """
    ThreadAgentSwitchPromptScreen {
        align: center middle;
    }

    ThreadAgentSwitchPromptScreen > Vertical {
        width: 72;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $warning;
        padding: 1 2;
    }

    ThreadAgentSwitchPromptScreen .thread-agent-switch-title {
        text-style: bold;
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }

    ThreadAgentSwitchPromptScreen .thread-agent-switch-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    ThreadAgentSwitchPromptScreen .thread-agent-switch-help {
        /* `auto`, not `1`: the help line is 38 cells, so it wraps once the
           dialog is narrower than ~48 columns, and a fixed single row clips
           the wrapped remainder -- which is the `Esc: cancel` half. The
           cancel affordance must not disappear in narrow terminals. */
        height: auto;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(
        self,
        *,
        thread_id: str,
        current_agent: str,
        thread_agent: str,
    ) -> None:
        """Initialize the prompt.

        Args:
            thread_id: Thread the user asked to resume.
            current_agent: Agent active in the current session.
            thread_agent: Agent that owns the requested thread.
        """
        super().__init__()
        self._resume_thread_id = thread_id
        self._current_agent = current_agent
        self._thread_agent = thread_agent

    def _body_text(self) -> str:
        """Explain the agent transition and its persistence scope.

        Returns:
            Plain-text prompt body.
        """
        return (
            f"Thread {self._resume_thread_id} belongs to agent "
            f"{self._thread_agent}, but {self._current_agent} is active.\n\n"
            f"Switch to {self._thread_agent} and resume this thread? The local "
            "agent server will restart. Your saved default agent will not change."
        )

    def compose(self) -> ComposeResult:
        """Compose the confirmation dialog.

        Yields:
            Title, explanation, and keyboard help.
        """
        with Vertical():
            yield Static(
                "Switch agents to resume?",
                classes="thread-agent-switch-title",
                markup=False,
            )
            yield Static(
                self._body_text(),
                classes="thread-agent-switch-body",
                markup=False,
            )
            yield Static(
                "Enter: switch and resume · Esc: cancel",
                classes="thread-agent-switch-help",
                markup=False,
            )

    def on_mount(self) -> None:
        """Focus the modal so its bindings receive keyboard input."""
        self.focus()

    def action_switch(self) -> None:
        """Confirm the agent switch and resume."""
        self.dismiss("switch")

    def action_cancel(self) -> None:
        """Keep the current agent and thread."""
        self.dismiss("cancel")
