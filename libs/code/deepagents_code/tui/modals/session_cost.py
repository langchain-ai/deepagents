"""Persistent warning when a session exceeds its configured cost threshold."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_code._session_stats import format_cost

if TYPE_CHECKING:
    from textual.app import ComposeResult


class SessionCostWarningScreen(ModalScreen[None]):
    """Keep the session cost warning visible until acknowledged."""

    can_focus = True
    CSS_PATH = "session_cost.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter,escape", "cancel", "Dismiss", show=False, priority=True),
    ]

    def __init__(self, *, cost_usd: float, threshold: float) -> None:
        """Initialize the session cost warning.

        Args:
            cost_usd: Current estimated session cost in US dollars.
            threshold: Configured warning threshold in US dollars.
        """
        super().__init__()
        self._cost_usd = cost_usd
        self._threshold = threshold

    def compose(self) -> ComposeResult:
        """Compose the warning and dismissal hint.

        Yields:
            The modal's static content.
        """
        with Vertical():
            yield Static(
                "Session cost warning",
                classes="session-cost-warning-title",
                markup=False,
            )
            yield Static(
                f"Estimated session cost is {format_cost(self._cost_usd)}, "
                f"above the configured {format_cost(self._threshold)} threshold. "
                "Consider /offload to reduce context usage or /clear to start fresh.",
                classes="session-cost-warning-body",
                markup=False,
            )
            yield Static(
                "Enter or Esc: dismiss",
                classes="session-cost-warning-help",
                markup=False,
            )

    def on_mount(self) -> None:
        """Focus the modal for keyboard dismissal."""
        self.focus()

    def action_cancel(self) -> None:
        """Acknowledge the warning without changing the session."""
        self.dismiss(None)
