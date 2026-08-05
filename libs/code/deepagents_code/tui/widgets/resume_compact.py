"""Prompt to compact a large-context thread before resuming it."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_code._session_stats import format_token_count
from deepagents_code.tui.widgets.cwd_switch import CwdSwitchPromptScreen

if TYPE_CHECKING:
    from textual.app import ComposeResult


class ResumeCompactPromptScreen(ModalScreen[Literal["compact", "continue", "cancel"]]):
    """Ask whether to compact a large thread before resuming it."""

    can_focus = True
    can_focus_children = False

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "compact", "Compact", show=False, priority=True),
        Binding("c", "continue", "Continue", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    CSS = (
        CwdSwitchPromptScreen.CSS.replace(
            "CwdSwitchPromptScreen", "ResumeCompactPromptScreen"
        )
        .replace("cwd-switch", "resume-compact")
        .replace("height: 1;", "height: auto;")
    )

    def __init__(self, *, context_tokens: int, threshold: int) -> None:
        """Initialize the prompt.

        Args:
            context_tokens: Latest model-reported context size for the thread.
            threshold: Configured token count that triggered the suggestion.
        """
        super().__init__()
        self._context_tokens = context_tokens
        self._threshold = threshold

    def compose(self) -> ComposeResult:
        """Compose the confirmation dialog.

        Yields:
            Title, explanation, and keyboard help.
        """
        with Vertical():
            yield Static("Compact before resuming?", classes="resume-compact-title")
            yield Static(
                f"This thread uses {format_token_count(self._context_tokens)} context "
                "tokens, above the configured "
                f"{format_token_count(self._threshold)} token threshold. Compacting "
                "summarizes older messages to reduce context usage and cost.",
                classes="resume-compact-body",
            )
            yield Static(
                "Enter: compact and resume · C: resume without compact · Esc: cancel",
                classes="resume-compact-help",
            )

    def on_mount(self) -> None:
        """Focus the modal so its bindings receive keyboard input."""
        self.focus()

    def action_compact(self) -> None:
        """Resume after compacting the thread."""
        self.dismiss("compact")

    def action_continue(self) -> None:
        """Resume without compacting the thread."""
        self.dismiss("continue")

    def action_cancel(self) -> None:
        """Cancel the resume."""
        self.dismiss("cancel")
