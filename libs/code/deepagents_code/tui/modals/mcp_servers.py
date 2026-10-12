"""Read-only MCP server details for the debug console."""

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
    from collections.abc import Callable, Sequence

    from textual.app import ComposeResult

    from deepagents_code.mcp_tools import MCPServerInfo

logger = logging.getLogger(__name__)


class MCPServersScreen(ModalScreen[None]):
    """Show a live, scrollable, copyable list of MCP server statuses."""

    CSS_PATH = "mcp_servers.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close", show=False),
        Binding("c", "copy", "Copy", show=False),
    ]

    def __init__(self, provider: Callable[[], Sequence[MCPServerInfo]]) -> None:
        """Initialize with an in-memory server metadata provider."""
        super().__init__()
        self._provider = provider
        self._details = "MCP server details unavailable"

    def compose(self) -> ComposeResult:
        """Compose the server list and keyboard help.

        Yields:
            The modal's server list and keyboard help widgets.
        """
        with VerticalScroll():
            yield Static("MCP Servers", classes="mcp-servers-title")
            yield Static(Content(self._details), id="mcp-servers-body", markup=False)
            yield Static(
                f" {get_glyphs().separator} ".join(("c copy", "Esc close")),
                classes="mcp-servers-help",
                markup=False,
            )

    def on_mount(self) -> None:
        """Focus the scroll container and keep server statuses current."""
        self.query_one(VerticalScroll).focus()
        self._refresh_details()
        self.set_interval(0.5, self._refresh_details)

    def _refresh_details(self) -> None:
        """Refresh sanitized names and statuses without exposing configuration."""
        try:
            details = (
                "\n".join(
                    f"{sanitize_control_chars(server.name)} ({server.status})"
                    for server in self._provider()
                )
                or "No MCP servers configured"
            )
        except Exception:
            logger.debug("MCP server details refresh failed", exc_info=True)
            return
        if details != self._details:
            self._details = details
            self.query_one("#mcp-servers-body", Static).update(Content(details))

    def action_copy(self) -> None:
        """Copy the displayed server list."""
        success, error = copy_text_to_clipboard(self.app, self._details)
        if success:
            self.app.notify("MCP server details copied", timeout=2)
        else:
            suffix = f": {error}" if error else ""
            self.app.notify(
                f"Failed to copy{suffix}", severity="warning", timeout=3, markup=False
            )

    def action_close(self) -> None:
        """Return to the debug console."""
        self.dismiss(None)
