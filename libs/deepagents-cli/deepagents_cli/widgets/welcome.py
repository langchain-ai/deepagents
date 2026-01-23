"""Welcome banner widget for deepagents-cli."""

from __future__ import annotations

import concurrent.futures
import os
from typing import Any

from rich.style import Style
from rich.text import Text
from textual.widgets import Static

from deepagents_cli.config import DEEP_AGENTS_ASCII, settings


def _get_langsmith_project_url(project_name: str) -> str | None:
    """Get the LangSmith project URL for a given project name.

    Uses a thread with a timeout to avoid blocking the UI.
    """

    def _fetch_url() -> str | None:
        import contextlib
        import io

        from langsmith import Client

        # Suppress stderr to hide API validation errors (e.g. project not yet created)
        with contextlib.redirect_stderr(io.StringIO()):
            client = Client()
            project = client.read_project(project_name=project_name)
            return project.url

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_fetch_url)
            return future.result(timeout=2.0)
    except Exception:
        return None


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
        banner = Text()
        banner.append_text(Text.from_markup(f"[bold #10b981]{DEEP_AGENTS_ASCII}[/bold #10b981]\n"))

        # Show LangSmith status if tracing is enabled
        langsmith_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
        langsmith_tracing = os.environ.get("LANGSMITH_TRACING") or os.environ.get(
            "LANGCHAIN_TRACING_V2"
        )

        if langsmith_key and langsmith_tracing:
            project_name = (
                settings.deepagents_langchain_project
                or os.environ.get("LANGSMITH_PROJECT")
                or "default"
            )
            project_url = _get_langsmith_project_url(project_name)
            banner.append("✓ ", style="green")
            banner.append("LangSmith tracing: ")
            if project_url:
                banner.append(
                    f"'{project_name}'",
                    style=Style(color="cyan", link=project_url),
                )
            else:
                banner.append(f"'{project_name}'", style="cyan")
            banner.append("\n")

        banner.append_text(
            Text.from_markup("[#10b981]Ready to code! What would you like to build?[/#10b981]\n")
        )
        banner.append_text(
            Text.from_markup("[dim]Enter send • Ctrl+J newline • @ files • / commands[/dim]")
        )
        super().__init__(banner, **kwargs)
