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

from deepagents_code.config import get_glyphs

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
            server_name: Server whose login just succeeded.
        """
        super().__init__()
        self._server_name = server_name

    def compose(self) -> ComposeResult:
        """Compose the confirmation dialog.

        Yields:
            Title, body, and help-row widgets parented inside a `Vertical`.
        """
        glyphs = get_glyphs()
        with Vertical():
            yield Static(
                Content.from_markup(
                    "$check Connected to [bold]$name[/bold]",
                    check=glyphs.checkmark,
                    name=self._server_name,
                ),
                classes="mcp-reconnect-title",
                markup=False,
            )
            yield Static(
                "Reconnect to load new tools, or defer with `/mcp reconnect`.",
                classes="mcp-reconnect-body",
                markup=False,
            )
            yield Static(
                "Enter to reconnect, Esc to defer",
                classes="mcp-reconnect-help",
                markup=False,
            )

    def action_reconnect(self) -> None:
        """Dismiss with `"reconnect"`."""
        self.dismiss("reconnect")

    def action_later(self) -> None:
        """Dismiss with `"later"`."""
        self.dismiss("later")

    def action_cancel(self) -> None:
        """Alias for `action_later` so the app-level Esc handler defers.

        The app's `action_interrupt` (`escape` binding, `priority=True`)
        fires before this screen's own `escape` binding. When the active
        screen is a `ModalScreen`, it dispatches to `action_cancel` if
        present, else falls through to `dismiss(None)`. Without this
        alias, Esc would dismiss with `None`, which the caller treats as
        a programmatic dismiss (no toast, no reopen) instead of an
        explicit defer.
        """
        self.action_later()


class MCPReconnectForceConfirmScreen(ModalScreen[bool]):
    """Confirmation overlay for `/mcp reconnect --force` with no pending login.

    Guards a fat-fingered force-restart when nothing is actually queued.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "confirm", "Confirm", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    CSS = """
    MCPReconnectForceConfirmScreen {
        align: center middle;
    }

    MCPReconnectForceConfirmScreen > Vertical {
        width: 64;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $warning;
        padding: 1 2;
    }

    MCPReconnectForceConfirmScreen .mcp-reconnect-title {
        text-style: bold;
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }

    MCPReconnectForceConfirmScreen .mcp-reconnect-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    MCPReconnectForceConfirmScreen .mcp-reconnect-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual requires an instance method
        """Compose the force-reconnect confirmation dialog.

        Yields:
            Title, body, and help-row widgets parented inside a `Vertical`.
        """
        with Vertical():
            yield Static(
                "Force reconnect?",
                classes="mcp-reconnect-title",
                markup=False,
            )
            yield Static(
                "No MCP login is queued. Restart will drop the current "
                "session and reload all servers.",
                classes="mcp-reconnect-body",
                markup=False,
            )
            yield Static(
                "Enter to restart, Esc to cancel",
                classes="mcp-reconnect-help",
                markup=False,
            )

    def action_confirm(self) -> None:
        """Dismiss with `True`."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Dismiss with `False`."""
        self.dismiss(False)
