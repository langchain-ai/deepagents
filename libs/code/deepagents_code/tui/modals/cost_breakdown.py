"""Detailed token and estimated-cost breakdown modal."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_code.clipboard import copy_text_to_clipboard
from deepagents_code.config import get_glyphs
from deepagents_code.unicode_security import sanitize_control_chars

if TYPE_CHECKING:
    from collections.abc import Callable

    from textual.app import ComposeResult

logger = logging.getLogger(__name__)


class CostBreakdownScreen(ModalScreen[None]):
    """Modal showing the copyable entire-thread token and cost breakdown."""

    can_focus = True
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close", show=False),
        Binding("c", "copy", "Copy", show=False),
    ]
    CSS = """
    CostBreakdownScreen { align: center middle; }
    CostBreakdownScreen > VerticalScroll {
        width: 78; max-width: 92%; height: auto; max-height: 85%;
        background: $surface; border: solid $primary; padding: 1 2;
    }
    CostBreakdownScreen .cost-breakdown-title {
        height: auto; text-style: bold; color: $primary;
        text-align: center; margin-bottom: 1;
    }
    CostBreakdownScreen .cost-breakdown-body {
        height: auto; color: $text;
    }
    CostBreakdownScreen .cost-breakdown-help {
        height: auto; color: $text-muted; text-style: italic;
        text-align: center; margin-top: 1;
    }
    """

    def __init__(self, breakdown: str, provider: Callable[[], str]) -> None:
        """Initialize with a plain-text breakdown and its live provider."""
        super().__init__()
        self._provider = provider
        self._breakdown = sanitize_control_chars(
            breakdown, keep_newlines=True, collapse_whitespace=False
        )

    def compose(self) -> ComposeResult:
        """Compose the breakdown and keyboard help.

        Yields:
            Widgets that make up the modal.
        """
        with VerticalScroll():
            yield Static("Token & Cost Breakdown", classes="cost-breakdown-title")
            yield Static(
                Content(self._breakdown), classes="cost-breakdown-body", markup=False
            )
            separator = f" {get_glyphs().separator} "
            yield Static(
                separator.join(("c copy", "Esc close")),
                classes="cost-breakdown-help",
                markup=False,
            )

    def on_mount(self) -> None:
        """Focus the modal and refresh its breakdown while open."""
        self.focus()
        self.set_interval(0.5, self._refresh_breakdown)

    def _refresh_breakdown(self) -> None:
        """Update the displayed and copyable breakdown when costs change."""
        try:
            breakdown = self._provider()
        except Exception:
            logger.debug("Cost breakdown refresh failed", exc_info=True)
            return
        if not breakdown:
            return
        breakdown = sanitize_control_chars(
            breakdown, keep_newlines=True, collapse_whitespace=False
        )
        if breakdown != self._breakdown:
            self._breakdown = breakdown
            self.query_one(".cost-breakdown-body", Static).update(Content(breakdown))

    def action_copy(self) -> None:
        """Copy the complete breakdown to the clipboard."""
        success, error = copy_text_to_clipboard(self.app, self._breakdown)
        if success:
            self.app.notify(
                "Token and cost breakdown copied",
                severity="information",
                timeout=2,
                markup=False,
            )
            return
        suffix = f": {error}" if error else ""
        self.app.notify(
            f"Failed to copy{suffix}", severity="warning", timeout=3, markup=False
        )

    def action_close(self) -> None:
        """Dismiss the breakdown modal."""
        self.dismiss(None)
