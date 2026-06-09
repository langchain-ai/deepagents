"""Prompt for switching cwd when resuming threads."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_code.sessions import format_path

if TYPE_CHECKING:
    from textual.app import ComposeResult


CwdSwitchChoice = Literal["switch", "stay"]
"""Outcome of the cwd switch prompt."""


class CwdSwitchPromptScreen(ModalScreen[CwdSwitchChoice]):
    """Modal asking whether to switch cwd before resuming a thread."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "switch", "Switch", show=False, priority=True),
        Binding("escape", "stay", "Stay", show=False, priority=True),
    ]

    CSS = """
    CwdSwitchPromptScreen {
        align: center middle;
    }

    CwdSwitchPromptScreen > Vertical {
        width: 72;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $warning;
        padding: 1 2;
    }

    CwdSwitchPromptScreen .cwd-switch-title {
        text-style: bold;
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }

    CwdSwitchPromptScreen .cwd-switch-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    CwdSwitchPromptScreen .cwd-switch-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(
        self,
        *,
        current_cwd: str,
        thread_cwd: str,
        project_settings_change_detected: bool = False,
    ) -> None:
        """Initialize the prompt."""
        super().__init__()
        self._current_cwd = current_cwd
        self._thread_cwd = thread_cwd
        self._project_settings_change_detected = project_settings_change_detected

    def _body_text(self) -> str:
        """Return the prompt body text."""
        current = format_path(self._current_cwd)
        target = format_path(self._thread_cwd)
        settings_note = (
            " Project settings will refresh too."
            if self._project_settings_change_detected
            else ""
        )
        return (
            "This thread was last used in:\n"
            f"  {target}\n\n"
            "You are currently in:\n"
            f"  {current}\n\n"
            "Switching uses the thread's original project for tools and "
            f"commands.{settings_note} Stay only if you want to keep working "
            "in the current project."
        )

    def compose(self) -> ComposeResult:
        """Compose the confirmation dialog.

        Yields:
            Widgets for the cwd switch prompt.
        """
        with Vertical():
            yield Static(
                "Resume from a different working directory?",
                classes="cwd-switch-title",
                markup=False,
            )
            yield Static(
                self._body_text(),
                classes="cwd-switch-body",
                markup=False,
            )
            yield Static(
                "Enter to switch cwd, Esc to stay here",
                classes="cwd-switch-help",
                markup=False,
            )

    def action_switch(self) -> None:
        """Dismiss with `switch`."""
        self.dismiss("switch")

    def action_stay(self) -> None:
        """Dismiss with `stay`."""
        self.dismiss("stay")

    def action_cancel(self) -> None:
        """Treat cancellation as staying in the current cwd."""
        self.action_stay()
