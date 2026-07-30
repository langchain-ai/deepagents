"""Prompt for compaction when resuming a thread with a large context."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, cast

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_code._session_stats import format_token_count

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from deepagents_code.app import DeepAgentsApp


ResumeCompactionChoice = Literal["compact", "continue", "cancel"]
"""Outcome of the large-context resume prompt."""


class ResumeCompactionPromptScreen(ModalScreen[ResumeCompactionChoice]):
    """Ask how to resume a thread whose saved context exceeds the threshold."""

    can_focus = True
    can_focus_children = False

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter,c", "compact", "Compact", show=False, priority=True),
        Binding("w", "continue_without_compaction", "Without compaction", show=False),
        Binding("escape", "cancel_resume", "Cancel", show=False, priority=True),
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
    ResumeCompactionPromptScreen {
        align: center middle;
    }

    ResumeCompactionPromptScreen > Vertical {
        width: 72;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $warning;
        padding: 1 2;
    }

    ResumeCompactionPromptScreen .resume-compaction-title {
        text-style: bold;
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }

    ResumeCompactionPromptScreen .resume-compaction-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    ResumeCompactionPromptScreen .resume-compaction-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(self, *, context_tokens: int, threshold: int) -> None:
        """Initialize the prompt with the saved context size and threshold."""
        super().__init__()
        self._context_tokens = context_tokens
        self._threshold = threshold

    def _body_text(self) -> str:
        """Return the prompt body."""
        context = format_token_count(self._context_tokens)
        threshold = format_token_count(self._threshold)
        return (
            f"This thread has {context} context tokens, above your {threshold}-token "
            "resume threshold.\n\n"
            "Enter/C — Continue conversation with compaction\n"
            "W — Continue without compaction\n"
            "Esc — Cancel"
        )

    def compose(self) -> ComposeResult:
        """Compose the prompt.

        Yields:
            The prompt title, choices, and keyboard help.
        """
        with Vertical():
            yield Static(
                "Large conversation context",
                classes="resume-compaction-title",
                markup=False,
            )
            yield Static(
                self._body_text(),
                classes="resume-compaction-body",
                markup=False,
            )
            yield Static(
                "Enter/C: compact · W: without compaction · Esc: cancel",
                classes="resume-compaction-help",
                markup=False,
            )

    def on_mount(self) -> None:
        """Focus the modal so its bindings receive input."""
        self.focus()

    def action_compact(self) -> None:
        """Resume after compacting the thread."""
        self.dismiss("compact")

    def action_continue_without_compaction(self) -> None:
        """Resume without changing the saved context."""
        self.dismiss("continue")

    def action_cancel_resume(self) -> None:
        """Cancel the resume operation."""
        self.dismiss("cancel")

    def action_cancel(self) -> None:
        """Treat the app-level cancel action as cancelling resume."""
        self.action_cancel_resume()

    def action_quit_or_interrupt(self) -> None:
        """Delegate Ctrl+C to the app-level quit handler."""
        cast("DeepAgentsApp", self.app).action_quit_or_interrupt()

    def action_quit_app(self) -> None:
        """Delegate Ctrl+D to the app-level quit handler."""
        cast("DeepAgentsApp", self.app).action_quit_app()
