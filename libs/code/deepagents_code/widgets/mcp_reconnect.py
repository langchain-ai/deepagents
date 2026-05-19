"""Confirmation modal shown after a successful MCP login.

Restarting the LangGraph server is required for newly minted MCP tokens
to take effect, but auto-restarting interrupts users who want to
authenticate against several MCP servers back-to-back. This modal lets
the user choose between restarting now and deferring until later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


ReconnectChoice = Literal["reconnect", "later"]
"""Outcome of the prompt: restart the server now or keep the current one."""


class MCPReconnectPromptScreen(ModalScreen[ReconnectChoice]):
    """Modal asking whether to restart the server after an MCP login.

    Dismisses with `"reconnect"` when the user accepts the restart and
    `"later"` when the user defers. Esc is treated as "later" so the
    user is never forced into a reconnect they did not explicitly choose.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "reconnect", "Reconnect", show=False, priority=True),
        Binding("escape", "later", "Later", show=False, priority=True),
        Binding("l", "later", "Later", show=False, priority=True),
    ]

    CSS = """
    MCPReconnectPromptScreen {
        align: center middle;
    }

    MCPReconnectPromptScreen > Vertical {
        width: 64;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    MCPReconnectPromptScreen .mcp-reconnect-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    MCPReconnectPromptScreen .mcp-reconnect-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    MCPReconnectPromptScreen .mcp-reconnect-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(self, server_name: str) -> None:
        """Initialize the prompt.

        Args:
            server_name: Server whose login just succeeded. Surfaced in the
                modal title so the user knows which credential is queued.
        """
        super().__init__()
        self._server_name = server_name

    def compose(self) -> ComposeResult:
        """Compose the confirmation dialog.

        Yields:
            The vertical container holding the title, body, and help row.
        """
        with Vertical():
            yield Static(
                Content.from_markup(
                    "Connected to [bold]$name[/bold]",
                    name=self._server_name,
                ),
                classes="mcp-reconnect-title",
            )
            yield Static(
                "Reconnect now to load the new MCP tools, or stay connected "
                "and authenticate with additional servers first. You can "
                "trigger the reconnect later from `/mcp`.",
                classes="mcp-reconnect-body",
            )
            yield Static(
                "Enter to reconnect now, Esc/L for later",
                classes="mcp-reconnect-help",
            )

    def action_reconnect(self) -> None:
        """Confirm — caller will restart the server."""
        self.dismiss("reconnect")

    def action_later(self) -> None:
        """Defer — caller leaves the running server in place."""
        self.dismiss("later")
