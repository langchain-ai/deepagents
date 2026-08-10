"""Modal shell for the context-usage visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_code import theme
from deepagents_code.config import is_ascii_mode
from deepagents_code.tui.widgets.context_usage._models import _Snapshot
from deepagents_code.tui.widgets.context_usage._widgets import (
    _ContextBar,
    _ContextHeader,
    _ContextLegend,
)

if TYPE_CHECKING:
    from textual.app import ComposeResult


class ContextUsageScreen(ModalScreen[None]):
    """Modal visualization of the current model context window."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close", show=False)
    ]

    CSS = """
    ContextUsageScreen { align: center middle; }

    ContextUsageScreen > Vertical {
        width: 94%;
        max-width: 120;
        height: auto;
        max-height: 90%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    ContextUsageScreen _ContextHeader { height: auto; min-height: 2; }
    ContextUsageScreen _ContextBar { height: 2; margin: 1 0; }
    ContextUsageScreen _ContextLegend { height: auto; margin-top: 1; }

    ContextUsageScreen .context-usage-help {
        height: 1;
        color: $text-muted;
        margin-top: 2;
    }
    """

    def __init__(
        self,
        *,
        context_tokens: int | None,
        conversation_tokens: int | None,
        context_limit: int | None,
        model_spec: str | None,
        approximate: bool,
    ) -> None:
        """Initialize the modal from the latest usage measurements.

        Args:
            context_tokens: Reliable total usage, or `None` when unavailable.
            conversation_tokens: Estimated effective conversation usage.
            context_limit: Configured model context limit.
            model_spec: Active model identifier.
            approximate: Whether the displayed usage is approximate.
        """
        super().__init__()
        self._snapshot = _Snapshot.from_usage(
            context_tokens=context_tokens,
            conversation_tokens=conversation_tokens,
            context_limit=context_limit,
            model_spec=model_spec,
            approximate=approximate,
        )

    def compose(self) -> ComposeResult:
        """Compose the context header, bar, legend, and close hint.

        Yields:
            Widgets that make up the visualization.
        """
        with Vertical():
            yield _ContextHeader(self._snapshot)
            yield _ContextBar(self._snapshot)
            yield _ContextLegend(self._snapshot)
            yield Static("Esc to close", classes="context-usage-help")

    def on_mount(self) -> None:
        """Use an ASCII border when the terminal cannot render Unicode."""
        if is_ascii_mode():
            panel = self.query_one(Vertical)
            panel.styles.border = ("ascii", theme.get_theme_colors(self).primary)

    def action_close(self) -> None:
        """Dismiss the context visualization."""
        self.dismiss(None)
