"""Welcome banner widget for deepagents-cli."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from textual.color import Color as TColor
from textual.content import Content
from textual.style import Style as TStyle
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
    newline_shortcut,
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

    def __init__(
        self,
        thread_id: str | None = None,
        mcp_tool_count: int = 0,
        *,
        connecting: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the welcome banner.

        Args:
            thread_id: Optional thread ID to display in the banner.
            mcp_tool_count: Number of MCP tools loaded at startup.
            connecting: When `True`, show a "Connecting..." footer instead of
                the normal ready prompt. Call `set_connected` to transition.
            **kwargs: Additional arguments passed to parent.
        """
        # Avoid collision with Widget._thread_id (Textual internal int)
        self._cli_thread_id: str | None = thread_id
        self._mcp_tool_count = mcp_tool_count
        self._connecting = connecting
        self._failed = False
        self._failure_error: str = ""
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

    def set_connected(self, mcp_tool_count: int = 0) -> None:
        """Transition from "connecting" to "ready" state.

        Args:
            mcp_tool_count: Number of MCP tools loaded during connection.
        """
        self._connecting = False
        self._failed = False
        self._mcp_tool_count = mcp_tool_count
        self.update(self._build_banner(self._project_url))

    def set_failed(self, error: str) -> None:
        """Transition from "connecting" to a persistent failure state.

        Args:
            error: Error message describing the server startup failure.
        """
        self._connecting = False
        self._failed = True
        self._failure_error = error
        self.update(self._build_banner(self._project_url))

    def on_click(self, event: Click) -> None:  # noqa: PLR6301  # Textual event handler
        """Open style-embedded hyperlinks on single click."""
        open_style_link(event)

    def _build_banner(self, project_url: str | None = None) -> Content:
        """Build the banner content.

        When a `project_url` is provided and a thread ID is set, the thread ID
        is rendered as a clickable hyperlink to the LangSmith thread view.

        Args:
            project_url: LangSmith project URL used for linking the project
                name and thread ID. When `None`, text is rendered without links.

        Returns:
            Content object containing the formatted banner.
        """
        parts: list[str | tuple[str, str | TStyle] | Content] = []
        # Use orange for local, green for production
        banner_color = (
            COLORS["primary_dev"] if _is_editable_install() else COLORS["primary"]
        )
        parts.append(
            (
                get_banner() + "\n",
                TStyle(foreground=TColor.parse(banner_color), bold=True),
            )
        )

        if self._project_name:
            parts.extend(
                [
                    (f"{get_glyphs().checkmark} ", "green"),
                    "LangSmith tracing: ",
                ]
            )
            if project_url:
                parts.append(
                    (
                        f"'{self._project_name}'",
                        TStyle(
                            foreground=TColor.parse("cyan"),
                            link=f"{project_url}?utm_source=deepagents-cli",
                        ),
                    )
                )
            else:
                parts.append((f"'{self._project_name}'", "cyan"))
            parts.append("\n")

        if self._cli_thread_id:
            if project_url:
                thread_url = (
                    f"{project_url.rstrip('/')}/t/{self._cli_thread_id}"
                    "?utm_source=deepagents-cli"
                )
                parts.extend(
                    [
                        ("Thread: ", "dim"),
                        (self._cli_thread_id, TStyle(dim=True, link=thread_url)),
                        ("\n", "dim"),
                    ]
                )
            else:
                parts.append((f"Thread: {self._cli_thread_id}\n", "dim"))

        if self._mcp_tool_count > 0:
            parts.append((f"{get_glyphs().checkmark} ", "green"))
            label = "MCP tool" if self._mcp_tool_count == 1 else "MCP tools"
            parts.append(f"Loaded {self._mcp_tool_count} {label}\n")

        if self._failed:
            parts.append(build_failure_footer(self._failure_error))
        elif self._connecting:
            parts.append(build_connecting_footer())
        else:
            parts.append(build_welcome_footer())
        return Content.assemble(*parts)


def build_failure_footer(error: str) -> Content:
    """Build a footer shown when the server failed to start.

    Args:
        error: Error message describing the failure.

    Returns:
        Content with a persistent failure message.
    """
    return Content.assemble(
        ("\nServer failed to start: ", "bold red"),
        (error, "red"),
        ("\n", "red"),
    )


def build_connecting_footer() -> Content:
    """Build a footer shown while waiting for the server to connect.

    Returns:
        Content with a connecting status message.
    """
    return Content.styled("\nConnecting to server...\n", "dim")


def build_welcome_footer() -> Content:
    """Build the two-line footer shown at the bottom of the welcome banner.

    Returns:
        Content with the ready prompt and keyboard shortcut help line.
    """
    bullet = get_glyphs().bullet
    return Content.assemble(
        ("\nReady to code! What would you like to build?\n", COLORS["primary"]),
        (
            (
                f"Enter send {bullet} {newline_shortcut()} newline "
                f"{bullet} @ files {bullet} / commands"
            ),
            "dim",
        ),
    )
