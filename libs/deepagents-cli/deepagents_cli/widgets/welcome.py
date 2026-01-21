"""Welcome banner widget for deepagents-cli."""

from __future__ import annotations

import os
from typing import Any

from textual.widgets import Static

from deepagents_cli.config import DEEP_AGENTS_ASCII, settings


class WelcomeBanner(Static):
    """Welcome banner displayed at startup."""

    DEFAULT_CSS = """
    WelcomeBanner {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        color: $primary;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the welcome banner."""
        banner_text = f"[bold]{DEEP_AGENTS_ASCII}[/bold]\n"

        # Show LangSmith status if tracing is enabled
        langsmith_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
        langsmith_tracing = os.environ.get("LANGSMITH_TRACING") or os.environ.get(
            "LANGCHAIN_TRACING_V2"
        )

        if langsmith_key and langsmith_tracing:
            project = (
                settings.deepagents_langchain_project
                or os.environ.get("LANGSMITH_PROJECT")
                or "default"
            )
            banner_text += f"[green]✓[/green] LangSmith tracing: '{project}'\n"

        banner_text += "Ready to code! What would you like to build?\n"
        banner_text += "[dim]Enter send • Ctrl+J newline • @ files • / commands[/dim]"
        super().__init__(banner_text, **kwargs)
