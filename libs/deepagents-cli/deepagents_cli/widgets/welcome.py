"""Welcome banner widget for deepagents-cli."""

from __future__ import annotations

from typing import Any

from textual.widgets import Static

from deepagents_cli._version import __version__
from deepagents_cli.config import DEEP_AGENTS_ASCII


class WelcomeBanner(Static):
    """Welcome banner displayed at startup."""

    DEFAULT_CSS = """
    WelcomeBanner {
        height: auto;
        padding: 1;
        margin-bottom: 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the welcome banner."""
        # Use the same green color as the original UI (#10b981)
        banner_text = f"[bold #10b981]{DEEP_AGENTS_ASCII}[/bold #10b981]"
        banner_text += f"[dim]v{__version__}[/dim]\n"
        banner_text += "[#10b981]Ready to code! What would you like to build?[/#10b981]\n"
        banner_text += "[dim]Enter send • Ctrl+J newline • @ files • / commands[/dim]"
        super().__init__(banner_text, **kwargs)
