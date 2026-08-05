"""Prompt for compacting a large resumed thread."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_code._session_stats import format_token_count
from deepagents_code.config import get_glyphs

if TYPE_CHECKING:
    from textual.app import ComposeResult


class ResumeCompactPromptScreen(ModalScreen[bool]):
    """Ask whether to compact a just-resumed thread before the next turn."""

    can_focus = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "compact", "Compact", show=False, priority=True),
        Binding("escape", "skip", "Skip", show=False, priority=True),
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
    ResumeCompactPromptScreen {
        align: center middle;
    }

    ResumeCompactPromptScreen > Vertical {
        width: 72;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $warning;
        padding: 1 2;
    }

    ResumeCompactPromptScreen .resume-compact-title {
        text-style: bold;
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }

    ResumeCompactPromptScreen .resume-compact-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    ResumeCompactPromptScreen .resume-compact-help {
        height: auto;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

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
            yield Static(
                "Compact this thread?",
                classes="resume-compact-title",
                markup=False,
            )
            yield Static(
                f"This thread uses {format_token_count(self._context_tokens)} context "
                "tokens, above the configured "
                f"{format_token_count(self._threshold)} token threshold. Compacting "
                "summarizes older messages so later turns cost less.",
                classes="resume-compact-body",
                markup=False,
            )
            yield Static(
                f" {get_glyphs().bullet} ".join(
                    ("Enter: compact now", "Esc: keep full context")
                ),
                classes="resume-compact-help",
                markup=False,
            )

    def on_mount(self) -> None:
        """Focus the modal so its bindings receive keyboard input."""
        self.focus()

    def action_compact(self) -> None:
        """Compact the thread before the next turn."""
        self.dismiss(True)

    def action_skip(self) -> None:
        """Leave the thread's context untouched."""
        self.dismiss(False)
