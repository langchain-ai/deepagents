"""First-enable education modal for Auto mode.

Shown at most once per install after Auto successfully becomes active. Auto is
already on when this appears; the modal is informational, not a gate that can
block or undo enablement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


class AutoModeNoticeScreen(ModalScreen[None]):
    """In-TUI first-run notice describing what Auto mode does.

    Dismisses with `None` on Enter or Esc. Auto remains active either way.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "dismiss_notice", "Continue", show=False, priority=True),
        Binding("escape", "dismiss_notice", "Continue", show=False, priority=True),
    ]

    CSS = """
    AutoModeNoticeScreen {
        align: center middle;
    }

    AutoModeNoticeScreen > Vertical {
        width: 64;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $warning;
        padding: 1 2;
    }

    AutoModeNoticeScreen .auto-mode-notice-title {
        text-style: bold;
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }

    AutoModeNoticeScreen .auto-mode-notice-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    AutoModeNoticeScreen .auto-mode-notice-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    # The screen must be the focus target for its own priority Enter/Esc
    # bindings to fire (see `on_mount`); without this the keys reach no handler.
    can_focus = True

    def __init__(self, body: str) -> None:
        """Initialize the notice.

        Args:
            body: Explanatory copy under the title.
        """
        super().__init__()
        self._body = body

    def on_mount(self) -> None:
        """Take focus so priority bindings receive Enter/Esc."""
        self.focus()

    def compose(self) -> ComposeResult:
        """Compose the Auto first-enable notice.

        Yields:
            Title, body, and help-row widgets parented inside a `Vertical`.
        """
        with Vertical():
            yield Static(
                "Auto is active",
                classes="auto-mode-notice-title",
                markup=False,
            )
            yield Static(
                self._body,
                classes="auto-mode-notice-body",
                markup=False,
            )
            yield Static(
                "Enter or Esc to continue",
                classes="auto-mode-notice-help",
                markup=False,
            )

    def action_dismiss_notice(self) -> None:
        """Dismiss the notice."""
        self.dismiss(None)

    def action_cancel(self) -> None:
        """Alias for Esc so app-level interrupt still dismisses the notice.

        Defensive fallback: the screen's own priority Esc binding normally
        handles dismissal. But the app's priority Escape handler
        (`DeepAgentsApp.action_interrupt`) prefers `action_cancel` on modal
        screens before falling through to a bare `dismiss(None)`, so this alias
        keeps Esc working if that handler wins the binding race.
        """
        self.action_dismiss_notice()
