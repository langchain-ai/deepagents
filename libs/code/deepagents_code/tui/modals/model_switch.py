"""Confirmation modal for switching models with a large context."""

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


class ModelSwitchWarningScreen(ModalScreen[bool]):
    """Confirm a model switch that preserves a large conversation context."""

    can_focus = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "confirm", "Switch", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    CSS = """
    ModelSwitchWarningScreen {
        align: center middle;
    }

    ModelSwitchWarningScreen > Vertical {
        width: 76;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $warning;
        padding: 1 2;
    }

    ModelSwitchWarningScreen .model-switch-warning-title {
        text-style: bold;
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }

    ModelSwitchWarningScreen .model-switch-warning-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    ModelSwitchWarningScreen .model-switch-warning-help {
        height: auto;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(
        self,
        *,
        current_model: str,
        target_model: str,
        context_tokens: int,
        threshold: int,
        approximate: bool,
    ) -> None:
        """Initialize the model-switch warning."""
        super().__init__()
        self._current_model = current_model
        self._target_model = target_model
        self._context_tokens = context_tokens
        self._threshold = threshold
        self._approximate = approximate

    def compose(self) -> ComposeResult:
        """Compose warning copy and keyboard help.

        Yields:
            The modal's static content.
        """
        qualifier = "approximately " if self._approximate else ""
        body = (
            f"This thread currently uses {qualifier}"
            f"{format_token_count(self._context_tokens)} context tokens, above your "
            f"{format_token_count(self._threshold)} warning threshold.\n\n"
            f"Switching from {self._current_model} to {self._target_model} keeps "
            "the conversation, but may discard prompt-cache savings and the new "
            "model may have a different context limit. Run /compact first if you "
            "want to reduce the context."
        )
        glyphs = get_glyphs()
        with Vertical():
            yield Static(
                "Switch models with a large context?",
                classes="model-switch-warning-title",
                markup=False,
            )
            yield Static(
                body,
                classes="model-switch-warning-body",
                markup=False,
            )
            yield Static(
                f"Enter: switch model {glyphs.bullet} Esc: cancel",
                classes="model-switch-warning-help",
                markup=False,
            )

    def on_mount(self) -> None:
        """Focus the modal so its bindings receive keyboard input."""
        self.focus()

    def action_confirm(self) -> None:
        """Authorize the model switch."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Leave the current model active."""
        self.dismiss(False)
