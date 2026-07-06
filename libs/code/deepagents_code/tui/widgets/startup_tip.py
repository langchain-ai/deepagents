"""Startup tip widget shown above the chat input."""

from __future__ import annotations

import random
from typing import Any

from textual.content import Content
from textual.widgets import Static

from deepagents_code._env_vars import HIDE_SPLASH_TIPS, is_env_truthy

_TIPS: dict[str, int] = {
    "Use @ to reference files and / for commands": 3,
    "Try /threads to resume a previous conversation": 2,
    "Use /offload when your conversation gets long": 2,
    "Use /copy to copy the latest assistant message": 3,
    "Use /mcp to search your MCP servers and inspect tool parameters": 1,
    "Use /mcp login <server> to authenticate MCP OAuth servers without leaving the TUI": 1,  # noqa: E501
    "Use /remember to save learnings from this conversation": 1,
    "Use /model to switch models mid-conversation": 2,
    "Use /effort high to change the current model's reasoning effort": 1,
    "Press ctrl+x to compose prompts in your external editor": 1,
    "Press ctrl+u to delete to the start of the line in the chat input": 1,
    "Use /skill:<name> to invoke a skill directly": 1,
    "Type /update to check for and install updates": 1,
    "Use /install <extra> to add optional dependencies (e.g. /install daytona)": 1,
    "Use /theme to customize the TUI's colors": 1,
    "Use /skill-creator to build reusable agent skills": 1,
    "Ask for a workflow to fan work out to subagents in parallel": 3,
    "Use /auto-update to toggle automatic updates": 1,
    "Use /timestamps to show or hide message timestamp footers": 1,
    "Use /agents to browse and switch between your available agents": 2,
    "In /agents, press Ctrl+S to set the highlighted agent as your default": 1,
    "Press Shift+Tab to toggle auto-approve mode": 2,
    "Use --startup-cmd to run a shell command before the first prompt": 1,
    "Use !! for incognito shell commands that stay out of model context": 1,
    "Deep Agents can explain its own features and look up its docs. Ask it how to use.": 3,  # noqa: E501
}
"""Rotating tips shown above the chat input, with relative selection weights."""


def _pick_tip() -> str:
    """Pick one startup tip using the configured relative weights.

    Returns:
        Tip text selected from `_TIPS`.
    """
    tips = list(_TIPS.keys())
    weights = list(_TIPS.values())
    return random.choices(tips, weights=weights, k=1)[0]  # noqa: S311


def show_startup_tip() -> bool:
    """Return whether startup tips should be shown.

    Returns:
        `True` when startup tips are enabled for the current process.
    """
    return not is_env_truthy(HIDE_SPLASH_TIPS)


class StartupTip(Static):
    """One startup tip displayed above the chat input."""

    DEFAULT_CSS = """
    StartupTip {
        height: auto;
        color: $text-muted;
        text-style: dim italic;
        padding: 0 1;
    }
    """

    def __init__(self, tip: str | None = None, **kwargs: Any) -> None:
        """Initialize the startup tip widget.

        Args:
            tip: Tip text to display. When omitted, one weighted tip is selected.
            **kwargs: Additional arguments passed to `Static`.
        """
        self.tip = tip if tip is not None else _pick_tip()
        super().__init__(
            Content.assemble(("Tip: ", "dim italic"), (self.tip, "dim italic")),
            **kwargs,
        )
