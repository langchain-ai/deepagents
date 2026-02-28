"""Welcome banner widget for deepagents-cli."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from rich.style import Style
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.events import Click

from deepagents_cli.config import (
    COLORS,
    _is_editable_install,
    fetch_langsmith_project_url,
    get_banner,
    get_glyphs,
    get_langsmith_project_name,
)
from deepagents_cli.widgets._links import open_style_link


class WelcomeBanner(Static):
    """Welcome banner displayed at startup."""

    # Disable Textual's auto_links to prevent a flicker cycle: Style.__add__
    # calls .copy() for linked styles, generating a fresh random _link_id on
    # each render. This means highlight_link_id never stabilizes, causing an
    # infinite hover-refresh loop.
    auto_links = False

    DEFAULT_CSS = """
    WelcomeBanner {
        height: auto;
        padding: 1;
        margin-bottom: 1;
    }
    """

    def __init__(self, thread_id: str | None = None, **kwargs: Any) -> None:
        """Initialize the welcome banner.

        Args:
            thread_id: Optional thread ID to display in the banner.
            **kwargs: Additional arguments passed to parent.
        """
        # Avoid collision with Widget._thread_id (Textual internal int)
        self._cli_thread_id: str | None = thread_id
        self._project_name: str | None = get_langsmith_project_name()
        self._project_url: str | None = None

        super().__init__(self._build_banner(), **kwargs)

    def on_mount(self) -> None:
        """Kick off background fetch for LangSmith project URL."""
        if self._project_name:
            self.run_worker(self._fetch_and_update, exclusive=True)

    async def _fetch_and_update(self) -> None:
        """Fetch the LangSmith URL in a thread and update the banner."""
        if not self._project_name:
            return
        try:
            project_url = await asyncio.wait_for(
                asyncio.to_thread(fetch_langsmith_project_url, self._project_name),
                timeout=2.0,
            )
        except (TimeoutError, OSError):
            project_url = None
        if project_url:
            self._project_url = project_url
            self.update(self._build_banner(project_url))

    def update_thread_id(self, thread_id: str) -> None:
        """Update the displayed thread ID and re-render the banner.

        Args:
            thread_id: The new thread ID to display.
        """
        self._cli_thread_id = thread_id
        self.update(self._build_banner(self._project_url))

    def on_click(self, event: Click) -> None:  # noqa: PLR6301  # Textual event handler
        """Open Rich-style hyperlinks on single click."""
        open_style_link(event)

    def _build_banner(self, project_url: str | None = None) -> Text:
        """Build the banner rich text.

        When a `project_url` is provided and a thread ID is set, the thread ID
        is rendered as a clickable hyperlink to the LangSmith thread view.

        Args:
            project_url: LangSmith project URL used for linking the project
                name and thread ID. When `None`, text is rendered without links.

        Returns:
            Rich Text object containing the formatted banner.
        """
        banner = Text()
        # Use orange for local, green for production
        banner_color = (
            COLORS["primary_dev"] if _is_editable_install() else COLORS["primary"]
        )
        banner.append(get_banner() + "\n", style=Style(bold=True, color=banner_color))

        if self._project_name:
            banner.append(f"{get_glyphs().checkmark} ", style="green")
            banner.append("LangSmith tracing: ")
            if project_url:
                banner.append(
                    f"'{self._project_name}'",
                    style=Style(
                        color="cyan",
                        link=f"{project_url}?utm_source=deepagents-cli",
                    ),
                )
            else:
                banner.append(f"'{self._project_name}'", style="cyan")
            banner.append("\n")

        if self._cli_thread_id:
            if project_url:
                thread_url = (
                    f"{project_url.rstrip('/')}/t/{self._cli_thread_id}"
                    "?utm_source=deepagents-cli"
                )
                thread_line = Text.assemble(
                    ("Thread: ", "dim"),
                    (self._cli_thread_id, Style(dim=True, link=thread_url)),
                    ("\n", "dim"),
                )
                banner.append_text(thread_line)
            else:
                banner.append(f"Thread: {self._cli_thread_id}\n", style="dim")

        banner.append(
            "Ready to code! What would you like to build?\n", style=COLORS["primary"]
        )
        bullet = get_glyphs().bullet
        banner.append(
            f"Enter send {bullet} Ctrl+J newline {bullet} @ files {bullet} / commands",
            style="dim",
        )
        return banner
