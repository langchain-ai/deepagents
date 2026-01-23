"""Welcome banner widget for deepagents-cli."""

from __future__ import annotations

import os
from typing import Any

from rich.markup import escape
from textual.widgets import Static

from deepagents_cli.config import DEEP_AGENTS_ASCII, settings


def _get_langsmith_project_url(project_name: str) -> str | None:
    """Get the LangSmith project URL for a given project name.

    Args:
        project_name: The name of the LangSmith project.

    Returns:
        The full URL to the project, or None if it couldn't be fetched.
    """
    import concurrent.futures

    def _fetch_url() -> str | None:
        from langsmith import Client

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
        banner_text = f"[bold #10b981]{DEEP_AGENTS_ASCII}[/bold #10b981]\n"

        langsmith_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get(
            "LANGCHAIN_API_KEY"
        )
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
            if project_url:
                escaped_url = escape(project_url)
                banner_text += (
                    f"[green]✓[/green] LangSmith tracing: [link={project_url}]{escaped_url}[/link]\n"
                )
            else:
                banner_text += (
                    f"[green]✓[/green] LangSmith tracing: [cyan]'{project_name}'[/cyan]\n"
                )

        banner_text += "[#10b981]Ready to code! What would you like to build?[/#10b981]\n"
        banner_text += "[dim]Enter send • Ctrl+J newline • @ files • / commands[/dim]"
        super().__init__(banner_text, **kwargs)
