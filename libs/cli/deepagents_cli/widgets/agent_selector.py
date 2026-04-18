"""Interactive agent selector screen for /agents command."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

if TYPE_CHECKING:
    from textual.app import ComposeResult

from deepagents_cli import theme
from deepagents_cli.config import get_glyphs, is_ascii_mode

logger = logging.getLogger(__name__)


class AgentSelectorScreen(ModalScreen[str | None]):
    """Modal dialog for switching between available agents.

    Displays agents found in `~/.deepagents/` in an `OptionList`. Returns the
    selected agent name on Enter, or `None` on Esc (no change).
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    CSS = """
    AgentSelectorScreen {
        align: center middle;
        background: transparent;
    }

    AgentSelectorScreen > Vertical {
        width: 50;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    AgentSelectorScreen .agent-selector-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    AgentSelectorScreen OptionList {
        height: auto;
        max-height: 16;
        background: $background;
    }

    AgentSelectorScreen .agent-selector-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }
    """

    def __init__(self, current_agent: str | None, agent_names: list[str]) -> None:
        """Initialize the AgentSelectorScreen.

        Args:
            current_agent: The name of the currently active agent (to
                highlight). May be `None` when no agent is active.
            agent_names: Sorted list of available agent names to display.
        """
        super().__init__()
        self._current_agent = current_agent
        self._agent_names = agent_names

    def compose(self) -> ComposeResult:
        """Compose the screen layout.

        Yields:
            Widgets for the agent selector UI.
        """
        glyphs = get_glyphs()
        options: list[Option] = []
        highlight_index = 0

        # Render labels via `Content.from_markup` so agent directory names
        # containing Rich markup characters (e.g. `[`) don't break rendering
        # or corrupt the option prompt.
        for i, name in enumerate(self._agent_names):
            if name == self._current_agent:
                label = Content.from_markup("$name [dim](current)[/dim]", name=name)
                highlight_index = i
            else:
                label = Content.from_markup("$name", name=name)
            options.append(Option(label, id=name))

        with Vertical():
            yield Static("Select Agent", classes="agent-selector-title")
            if options:
                option_list = OptionList(*options, id="agent-options")
                option_list.highlighted = highlight_index
                yield option_list
                help_text = (
                    f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate"
                    f" {glyphs.bullet} Enter select"
                    f" {glyphs.bullet} Esc cancel"
                )
            else:
                yield Static(
                    "No agents found in ~/.deepagents/.\n"
                    "Run the CLI with -a <name> to create one.",
                    classes="agent-selector-help",
                )
                help_text = f"{glyphs.bullet} Esc close"
            yield Static(help_text, classes="agent-selector-help")

    def on_mount(self) -> None:
        """Apply ASCII border if needed."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Dismiss with the selected agent name.

        Args:
            event: The option selected event.
        """
        name = event.option.id
        self.dismiss(name)

    def action_cancel(self) -> None:
        """Cancel without switching agents."""
        self.dismiss(None)
