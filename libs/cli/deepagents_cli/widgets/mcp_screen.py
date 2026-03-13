"""MCP server display screen."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static, Tree

if TYPE_CHECKING:
    from textual.app import ComposeResult

from deepagents_cli.config import get_glyphs


class MCPScreen(ModalScreen[None]):
    """Modal screen showing configured MCP servers and their tools.

    Displays a tree view of MCP servers from .mcp.json with
    server name, type, and URL. Expects config schema:
    ``{"mcpServers": {"name": {"type": "http"|"sse"|"stdio", "url": "..."}}}``
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Close", show=False, priority=True),
    ]

    DEFAULT_CSS = """
    MCPScreen {
        align: center middle;
    }

    MCPScreen > Vertical {
        width: 80;
        max-width: 90%;
        height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    MCPScreen .mcp-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    MCPScreen Tree {
        height: 1fr;
        background: $background;
    }

    MCPScreen .mcp-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize MCP screen.

        Args:
            config_path: Path to .mcp.json config file. Defaults to
                .mcp.json in the current working directory.
        """
        super().__init__()
        self._config_path = config_path or Path(".mcp.json")

    def compose(self) -> ComposeResult:
        """Build the MCP screen layout.

        Yields:
            Widgets for the MCP screen.
        """
        glyphs = get_glyphs()
        with Vertical():
            yield Static("MCP Servers", classes="mcp-title")
            yield Tree("Servers", id="mcp-tree")
            help_text = (
                f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate "
                f"{glyphs.bullet} Esc close"
            )
            yield Static(help_text, classes="mcp-help")

    async def on_mount(self) -> None:
        """Load MCP server config and populate tree."""
        tree: Tree[dict[str, Any]] = self.query_one("#mcp-tree", Tree)
        tree.root.expand()
        tree.show_root = False

        servers = self._load_mcp_config()
        glyphs = get_glyphs()

        if not servers:
            tree.root.add_leaf("No MCP servers configured")
            return

        for name, config in servers.items():
            server_type = MCPScreen._derive_server_type(config)
            label = f"{glyphs.circle_filled} {name:<24} {server_type}"
            tree.root.add(label)

    @staticmethod
    def _derive_server_type(config: dict[str, Any]) -> str:
        """Derive server type from config keys.

        Args:
            config: Server configuration dict.

        Returns:
            Server type string: "http", "sse", "stdio", or "unknown".
        """
        if "type" in config:
            return str(config["type"])
        if "command" in config:
            return "stdio"
        if "url" in config:
            return "http"
        return "unknown"

    def _load_mcp_config(self) -> dict[str, dict[str, Any]]:
        """Load MCP server configuration from .mcp.json.

        Returns:
            Dict of server name to server config.
        """
        if not self._config_path.exists():
            return {}
        try:
            data = json.loads(self._config_path.read_text())
            return data.get("mcpServers", {})
        except (json.JSONDecodeError, OSError):
            return {}

    def action_cancel(self) -> None:
        """Close the MCP screen."""
        self.dismiss(None)
