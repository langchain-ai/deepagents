"""Persistent banner shown when inside a stepped-into subagent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Static

from deepagents_cli.config import COLORS

if TYPE_CHECKING:
    from textual.app import ComposeResult


class SubagentBanner(Horizontal):
    """Persistent banner indicating the user is inside a subagent session.

    Displays the subagent type, nesting depth, and available commands.
    Hidden by default; call :meth:`show` / :meth:`hide` to toggle.
    """

    DEFAULT_CSS = f"""
    SubagentBanner {{
        height: auto;
        dock: top;
        display: none;
        background: {COLORS.get("mode_shell", "#3b82f6")};
        color: white;
        padding: 0 1;
    }}

    SubagentBanner.visible {{
        display: block;
    }}

    SubagentBanner .banner-label {{
        width: auto;
        text-style: bold;
        padding: 0 1 0 0;
    }}

    SubagentBanner .banner-commands {{
        width: 1fr;
        color: #cbd5e1;
    }}
    """

    subagent_type: reactive[str] = reactive("", init=False)
    depth: reactive[int] = reactive(0, init=False)
    breadcrumbs: reactive[str] = reactive("", init=False)

    def compose(self) -> ComposeResult:  # noqa: PLR6301 — Textual widget method
        """Compose the banner layout.

        Yields:
            Label and commands widgets.
        """
        yield Static("", classes="banner-label", id="subagent-label")
        yield Static("", classes="banner-commands", id="subagent-commands")

    def show(
        self,
        *,
        subagent_type: str,
        depth: int,
        context_stack: list[str] | None = None,
    ) -> None:
        """Show the banner for the given subagent context.

        Args:
            subagent_type: Name of the subagent (e.g. "general-purpose").
            depth: Current nesting depth (1 = first subagent).
            context_stack: List of subagent type names from root to current,
                used to render breadcrumbs (e.g. ["root", "researcher", "coder"]).
        """
        self.subagent_type = subagent_type
        self.depth = depth
        if context_stack and len(context_stack) > 2:  # noqa: PLR2004
            # Show breadcrumbs for nested subagents: root > researcher > coder
            self.breadcrumbs = " > ".join(context_stack)
        else:
            self.breadcrumbs = ""
        self._refresh_content()
        self.add_class("visible")

    def hide(self) -> None:
        """Hide the banner (back to root context)."""
        self.remove_class("visible")

    def _refresh_content(self) -> None:
        """Update label and command hint text."""
        try:
            label = self.query_one("#subagent-label", Static)
            commands = self.query_one("#subagent-commands", Static)
        except Exception:  # noqa: BLE001
            return

        if self.breadcrumbs:
            label.update(self.breadcrumbs)
        else:
            label.update(f"[{self.subagent_type}:{self.depth}]")
        commands.update("/return  /summary  /context")
