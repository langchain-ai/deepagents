"""Clickable tab labels for the plugin manager header."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.message import Message
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.events import Click

    from deepagents_code.tui.modals.plugin_manager.models import PluginTab

TAB_LABELS: dict[PluginTab, str] = {
    "discover": "Plugins",
    "installed": "Installed",
    "marketplaces": "Marketplaces",
    "errors": "Errors",
}


class PluginTabSelected(Message):
    """Posted when a plugin manager tab label is clicked."""

    def __init__(self, tab: PluginTab) -> None:
        """Initialize with the selected tab id.

        Args:
            tab: Tab to activate.
        """
        super().__init__()
        self.tab = tab


class PluginTabLabel(Static):
    """Mouse-clickable tab label in the plugin manager header."""

    def __init__(self, tab: PluginTab, label: str) -> None:
        """Create a tab label.

        Args:
            tab: Tab id this label activates.
            label: Display text for the tab.
        """
        super().__init__(
            f"  {label}  ",
            id=f"plugin-tab-{tab}",
            classes="plugin-manager-tab",
            markup=False,
        )
        self._tab = tab
        self._label = label
        self.can_focus = False

    def set_active(self, active: bool) -> None:
        """Update the active marker and style.

        Args:
            active: Whether this tab is the current tab.
        """
        self.update(f"> {self._label} <" if active else f"  {self._label}  ")
        self.set_class(active, "active")

    def on_click(self, event: Click) -> None:
        """Select this tab on click.

        Args:
            event: The click event.
        """
        event.stop()
        self.post_message(PluginTabSelected(self._tab))
