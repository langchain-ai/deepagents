"""Prompt for confirming or aborting a `-r` thread resume at launch."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_code.sessions import format_path

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from deepagents_code.app import DeepAgentsApp


class ResumeConfirmPromptScreen(ModalScreen[bool]):
    """Modal asking whether to resume a thread or start a new session."""

    can_focus = True
    can_focus_children = False

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "resume", "Resume", show=False, priority=True),
        Binding("escape", "abort", "Start new", show=False, priority=True),
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
    ResumeConfirmPromptScreen {
        align: center middle;
    }

    ResumeConfirmPromptScreen > Vertical {
        width: 72;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $accent;
        padding: 1 2;
    }

    ResumeConfirmPromptScreen .resume-confirm-title {
        text-style: bold;
        color: $accent;
        text-align: center;
        margin-bottom: 1;
    }

    ResumeConfirmPromptScreen .resume-confirm-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    ResumeConfirmPromptScreen .resume-confirm-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(
        self,
        *,
        thread_id: str,
        agent_name: str | None = None,
        thread_cwd: str | None = None,
    ) -> None:
        """Initialize the prompt."""
        super().__init__()
        self._resume_thread_id = thread_id
        self._agent_name = agent_name
        self._thread_cwd = thread_cwd

    def _body_text(self) -> str:
        """Return the prompt body text."""
        lines = [f"  Thread:    {self._resume_thread_id}"]
        if self._agent_name:
            lines.append(f"  Agent:     {self._agent_name}")
        if self._thread_cwd:
            lines.append(f"  Directory: {format_path(self._thread_cwd)}")
        details = "\n".join(lines)
        return f"Resume this thread, or start a new session instead?\n\n{details}"

    def compose(self) -> ComposeResult:
        """Compose the confirmation dialog.

        Yields:
            Widgets for the resume confirmation prompt.
        """
        with Vertical():
            yield Static(
                "Resume previous thread?",
                classes="resume-confirm-title",
                markup=False,
            )
            yield Static(
                self._body_text(),
                classes="resume-confirm-body",
                markup=False,
            )
            yield Static(
                "Enter: resume \u00b7 Esc: start new session",
                classes="resume-confirm-help",
                markup=False,
            )

    def on_mount(self) -> None:
        """Focus the modal so screen bindings work at launch."""
        self.focus()

    def action_resume(self) -> None:
        """Dismiss with `True` to resume the thread."""
        self.dismiss(True)

    def action_abort(self) -> None:
        """Dismiss with `False` to start a new session."""
        self.dismiss(False)

    def action_cancel(self) -> None:
        """Treat cancellation as aborting the resume."""
        self.action_abort()

    def action_quit_or_interrupt(self) -> None:
        """Delegate Ctrl+C to the app-level quit/interrupt handler."""
        cast("DeepAgentsApp", self.app).action_quit_or_interrupt()

    def action_quit_app(self) -> None:
        """Delegate Ctrl+D to the app-level quit handler."""
        cast("DeepAgentsApp", self.app).action_quit_app()
