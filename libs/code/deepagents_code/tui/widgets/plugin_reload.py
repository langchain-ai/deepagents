"""Confirmation modal offered after reload-relevant plugin changes."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


PluginReloadChoice = Literal["reload", "later"]
"""Outcome of the prompt: apply plugin changes now or defer."""


class PluginReloadPromptScreen(ModalScreen[PluginReloadChoice]):
    """Ask whether to reload after leaving the plugin manager."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "reload", "Reload", show=False, priority=True),
        Binding("escape", "later", "Later", show=False, priority=True),
    ]

    CSS = """
    PluginReloadPromptScreen {
        align: center middle;
    }

    PluginReloadPromptScreen > Vertical {
        width: 64;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    PluginReloadPromptScreen .plugin-reload-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    PluginReloadPromptScreen .plugin-reload-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    PluginReloadPromptScreen .plugin-reload-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual requires an instance method
        """Compose the title, explanation, and keyboard help.

        Yields:
            Widgets for the plugin reload prompt.
        """
        with Vertical():
            yield Static(
                "Reload plugins?",
                classes="plugin-reload-title",
                markup=False,
            )
            yield Static(
                "Reload to apply changes to plugin skills and MCP tools.",
                classes="plugin-reload-body",
                markup=False,
            )
            yield Static(
                "Enter to reload, Esc for later",
                classes="plugin-reload-help",
                markup=False,
            )

    def action_reload(self) -> None:
        """Choose to apply plugin changes now."""
        self.dismiss("reload")

    def action_later(self) -> None:
        """Defer the reload."""
        self.dismiss("later")

    def action_cancel(self) -> None:
        """Treat the app-level Esc action as an explicit deferral."""
        self.action_later()
