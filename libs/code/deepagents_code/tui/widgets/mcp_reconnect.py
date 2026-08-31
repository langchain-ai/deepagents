"""Confirmation modals for MCP changes that need a server restart.

Restarting the LangGraph server is required for newly minted MCP tokens
and for `/mcp` disable/enable toggles to take effect, but auto-restarting
interrupts users who want to make several MCP changes back-to-back. The
two `_ReconnectPromptScreen` subclasses let the user choose between
restarting now and deferring until later.

`MCPReconnectForceConfirmScreen` is the exception: it guards
`/mcp reconnect --force` when nothing is queued, so its Esc cancels the
restart outright rather than deferring it.
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
    from collections.abc import Callable, Sequence

    from textual.app import ComposeResult


ReconnectChoice = Literal["reconnect", "later"]
"""Outcome of the prompt: restart the server now or keep the current one.

Callers must also handle `None`, which Textual passes when a screen is
dismissed programmatically rather than by a user keypress. That is not a
choice, and callers deliberately stay quiet for it rather than narrating
an action the user did not take.
"""


class _ReconnectPromptScreen(ModalScreen[ReconnectChoice]):
    """Shared base for the reconnect-or-defer MCP modals.

    Subclasses supply only their title and body copy; the base owns the
    bindings, layout, styling, and the `"reconnect"`/`"later"` dismissal
    contract. The `DEFAULT_CSS` type selector matches subclasses because
    Textual resolves type selectors against every class name in the MRO.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "reconnect", "Reconnect", show=False, priority=True),
        Binding("escape", "later", "Later", show=False, priority=True),
    ]

    DEFAULT_CSS = """
    _ReconnectPromptScreen {
        align: center middle;
    }

    _ReconnectPromptScreen > Vertical {
        width: 64;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    _ReconnectPromptScreen .mcp-reconnect-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    _ReconnectPromptScreen .mcp-reconnect-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    _ReconnectPromptScreen .mcp-reconnect-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(self, *, title: str | Content, body: str | Content) -> None:
        """Store the dialog copy for `compose`.

        Args:
            title: Bold heading shown at the top of the dialog.
            body: Explanatory paragraph beneath the title.
        """
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        """Compose the confirmation dialog.

        Yields:
            Title, body, and help-row widgets parented inside a `Vertical`.
        """
        with Vertical():
            yield Static(
                self._title,
                classes="mcp-reconnect-title",
                markup=False,
            )
            yield Static(
                self._body,
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


class MCPReconnectPromptScreen(_ReconnectPromptScreen):
    """Modal asking whether to restart the server after an MCP login.

    Dismisses with `"reconnect"` when the user accepts the restart and
    `"later"` when the user defers. Esc is treated as "later" so the
    user is never forced into a reconnect they did not explicitly choose.
    """

    def __init__(self, server_name: str) -> None:
        """Initialize the prompt.

        Args:
            server_name: Server whose login just succeeded.
        """
        super().__init__(
            title=Content.from_markup(
                "$check Connected to [bold]$name[/bold]",
                check=get_glyphs().checkmark,
                name=server_name,
            ),
            body="Reconnect to load new tools.",
        )


class MCPDisableReconnectPromptScreen(_ReconnectPromptScreen):
    """Modal asking whether to reconnect after `/mcp` disable/enable toggles.

    Shown when the user closes the `/mcp` viewer with pending `F2`
    disable-state changes but without pressing `Ctrl+R`, so the toggles
    do not silently sit unapplied. Dismisses with `"reconnect"` when the
    user accepts the restart and `"later"` when the user defers; Esc is
    treated as "later".
    """

    def __init__(
        self,
        server_names: Sequence[str],
        *,
        on_choice: Callable[[ReconnectChoice], None] | None = None,
    ) -> None:
        """Initialize the prompt.

        Args:
            server_names: Servers whose disabled state changed and are
                waiting on a reconnect. Must be non-empty — the caller
                only opens this modal when at least one toggle is
                pending, and the body would otherwise name no server.
            on_choice: Optional callback invoked for an explicit reconnect
                or defer choice before the screen dismisses. This supports
                an atomic `switch_screen` transition from the MCP viewer,
                whose original result callback is removed by the switch.
        """
        super().__init__(
            title="Apply MCP server changes?",
            body=Content.from_markup(
                "Reconnect to apply the changes to $names.",
                names=", ".join(server_names),
            ),
        )
        self._on_choice = on_choice

    def action_reconnect(self) -> None:
        """Report and dismiss with `"reconnect"`."""
        if self._on_choice is not None:
            self._on_choice("reconnect")
        super().action_reconnect()

    def action_later(self) -> None:
        """Report and dismiss with `"later"`."""
        if self._on_choice is not None:
            self._on_choice("later")
        super().action_later()


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
