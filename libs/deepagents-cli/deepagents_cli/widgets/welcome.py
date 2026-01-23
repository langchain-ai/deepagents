"""Welcome banner widget for deepagents-cli."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from rich.style import Style
from rich.text import Text
from textual.widgets import Static

from deepagents_cli.config import DEEP_AGENTS_ASCII, settings


def _fetch_project_url(project_name: str) -> str | None:
    """Fetch the LangSmith project URL (blocking, run in a thread)."""
    try:
        from langsmith import Client

        project = Client().read_project(project_name=project_name)
    except (OSError, ValueError, RuntimeError):
        return None
    else:
        return project.url if project.url else None


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
        self._project_name: str | None = None

        langsmith_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
        langsmith_tracing = os.environ.get("LANGSMITH_TRACING") or os.environ.get(
            "LANGCHAIN_TRACING_V2"
        )

        if langsmith_key and langsmith_tracing:
            self._project_name = (
                settings.deepagents_langchain_project
                or os.environ.get("LANGSMITH_PROJECT")
                or "default"
            )

        super().__init__(self._build_banner(), **kwargs)

    def on_mount(self) -> None:
        """Kick off background fetch for LangSmith project URL."""
        if self._project_name:
            self.run_worker(self._fetch_and_update, exclusive=True)

    async def _fetch_and_update(self) -> None:
        """Fetch the LangSmith URL in a thread and update the banner."""
        try:
            project_url = await asyncio.wait_for(
                asyncio.to_thread(_fetch_project_url, self._project_name),
                timeout=2.0,
            )
        except (TimeoutError, OSError):
            project_url = None
        if project_url:
            self.update(self._build_banner(project_url))

    def _build_banner(self, project_url: str | None = None) -> Text:
        """Build the banner rich text."""
        banner = Text()
        banner.append(DEEP_AGENTS_ASCII + "\n", style=Style(bold=True, color="#10b981"))

        if self._project_name:
            banner.append("✓ ", style="green")
            banner.append("LangSmith tracing: ")
            if project_url:
                banner.append(
                    f"'{self._project_name}'",
                    style=Style(color="cyan", link=project_url),
                )
            else:
                banner.append(f"'{self._project_name}'", style="cyan")
            banner.append("\n")

        banner.append("Ready to code! What would you like to build?\n", style="#10b981")
        banner.append("Enter send • Ctrl+J newline • @ files • / commands", style="dim")
        return banner
