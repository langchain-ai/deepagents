"""Blocking modal shown when the session time limit expires."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

from deepagents_cli import theme
from deepagents_cli.config import is_ascii_mode

TIME_UP_INPUT_DISABLED_MESSAGE = "Time is up. Input is disabled for this session."
"""Shared warning shown when input is attempted after the time limit."""


class TimesUpScreen(ModalScreen[None]):
    """Non-dismissible modal for a terminal time-limit state."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Blocked", show=False, priority=True),
    ]

    CSS = """
    TimesUpScreen {
        align: center middle;
    }

    TimesUpScreen > Vertical {
        width: 56;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $warning;
        padding: 1 2;
    }

    TimesUpScreen .times-up-title {
        text-style: bold;
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }

    TimesUpScreen .times-up-copy {
        height: auto;
        color: $text;
        text-align: center;
    }
    """

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual override
        """Compose the blocking time-limit modal.

        Yields:
            Widgets for the modal content.
        """
        with Vertical():
            yield Static("Time's up", classes="times-up-title")
            yield Static(
                Content.assemble("Refer to the screen for results."),
                classes="times-up-copy",
            )

    def on_mount(self) -> None:
        """Apply ASCII border styling when requested."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.warning)

    def action_cancel(self) -> None:
        """Keep the modal open and indicate that input is blocked."""
        self.notify(
            TIME_UP_INPUT_DISABLED_MESSAGE,
            severity="warning",
            timeout=3,
            markup=False,
        )
