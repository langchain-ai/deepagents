"""Welcome banner widget for deepagents-cli."""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, Any

from textual.color import Color as TColor
from textual.content import Content
from textual.style import Style as TStyle
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.events import Click

from deepagents_cli import theme
from deepagents_cli._version import __version__
from deepagents_cli.config import (
    _get_editable_install_path,
    _is_editable_install,
    fetch_langsmith_project_url,
    get_banner,
    get_glyphs,
    get_langsmith_project_name,
)
from deepagents_cli.widgets._links import open_style_link

_TIPS: list[str] = [
    "Use @ to reference files and / for commands",
    "Try /threads to resume a previous conversation",
    "Use /offload when your conversation gets long",
    "Use /mcp to see your loaded tools and servers",
    "Use /remember to save learnings from this conversation",
    "Use /model to switch models mid-conversation",
    "Press ctrl+x to compose prompts in your external editor",
    "Press ctrl+u to delete to the start of the line in the chat input",
    "Use /skill:<name> to invoke a skill directly",
    "Type /update to check for and install updates",
    "Use /theme to customize the CLI colors and style",
    "Use /skill-creator to build reusable agent skills",
]
"""Rotating tips shown in the welcome footer.

One is picked per session.
"""


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
        resuming: bool = False,
        local_server: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the welcome banner.

        Args:
            thread_id: Optional thread ID to display in the banner.
            mcp_tool_count: Number of MCP tools loaded at startup.
            connecting: When `True`, show a "Connecting..." footer instead of
                the normal ready prompt. Call `set_connected` to transition.
            resuming: When `True`, the connecting footer says "Resuming..."
                instead of any `'Connecting...'` variant.
            local_server: When `True`, the connecting footer qualifies the
                server as "local" (i.e. a server process managed by the
                CLI).

                Ignored when `resuming` is `True`.
            **kwargs: Additional arguments passed to parent.
        """
        # Avoid collision with Widget._thread_id (Textual internal int)
        self._cli_thread_id: str | None = thread_id
        self._mcp_tool_count = mcp_tool_count
        self._connecting = connecting
        self._resuming = resuming
        self._local_server = local_server
        self._failed = False
        self._failure_error: str = ""
        self._project_name: str | None = get_langsmith_project_name()
        self._project_url: str | None = None
        self._tip: str = random.choice(_TIPS)  # noqa: S311

        super().__init__(self._build_banner(), **kwargs)

    def on_mount(self) -> None:
        """Kick off background fetch for LangSmith project URL."""
        self.watch(self.app, "theme", self._on_theme_change, init=False)
        if self._project_name:
            self.run_worker(self._fetch_and_update, exclusive=True)

    def _on_theme_change(self) -> None:
        """Re-render the banner when the app theme changes."""
        self.update(self._build_banner(self._project_url))

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
        colors = theme.get_theme_colors(self)
        ansi = self.app.theme == "textual-ansi"

        banner = get_banner()
        primary_style: str | TStyle = (
            "bold"
            if ansi
            else TStyle(foreground=TColor.parse(colors.primary), bold=True)
        )

        if not ansi and _is_editable_install():
            # Highlight local-install version tag with tool accent; art stays primary.
            dev_style = TStyle(foreground=TColor.parse(colors.tool), bold=True)
            version_tag = f"v{__version__} (local)"
            idx = banner.rfind(version_tag)
            if idx >= 0:
                parts.extend(
                    [
                        (banner[:idx], primary_style),
                        (version_tag, dev_style),
                        (banner[idx + len(version_tag) :] + "\n", primary_style),
                    ]
                )
            else:
                parts.append((banner + "\n", primary_style))
        else:
            parts.append((banner + "\n", primary_style))

        # For ANSI theme, use "bold" (terminal foreground) instead of hex
        accent: str | TStyle = "bold" if ansi else colors.primary
        success_color: str = "bold green" if ansi else colors.success

        editable_path = _get_editable_install_path()
        if editable_path:
            parts.extend([("Installed from: ", "dim"), (editable_path, "dim"), "\n"])

        if self._project_name:
            parts.extend(
                [
                    (f"{get_glyphs().checkmark} ", success_color),
                    "LangSmith tracing: ",
                ]
            )
            if project_url:
                link_style: str | TStyle
                if ansi:
                    url = f"{project_url}?utm_source=deepagents-cli"
                    link_style = TStyle(bold=True, link=url)
                else:
                    link_style = TStyle(
                        foreground=TColor.parse(colors.primary),
                        link=f"{project_url}?utm_source=deepagents-cli",
                    )
                parts.append((f"'{self._project_name}'", link_style))
            else:
                parts.append((f"'{self._project_name}'", accent))
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
            parts.append((f"{get_glyphs().checkmark} ", success_color))
            label = "MCP tool" if self._mcp_tool_count == 1 else "MCP tools"
            parts.append(f"Loaded {self._mcp_tool_count} {label}\n")

        if self._failed:
            parts.append(build_failure_footer(self._failure_error))
        elif self._connecting:
            parts.append(
                build_connecting_footer(
                    resuming=self._resuming,
                    local_server=self._local_server,
                )
            )
        else:
            ready_color = "bold" if ansi else colors.primary
            parts.append(build_welcome_footer(primary_color=ready_color, tip=self._tip))
        return Content.assemble(*parts)


def build_failure_footer(error: str) -> Content:
    """Build a footer shown when the server failed to start.

    Args:
        error: Error message describing the failure.

    Returns:
        Content with a persistent failure message.
    """
    colors = theme.get_theme_colors()
    return Content.assemble(
        ("\nServer failed to start: ", f"bold {colors.error}"),
        (error, colors.error),
        ("\n", colors.error),
    )


def build_connecting_footer(
    *, resuming: bool = False, local_server: bool = False
) -> Content:
    """Build a footer shown while waiting for the server to connect.

    Args:
        resuming: Show `'Resuming...'` instead of any `'Connecting...'` variant.
        local_server: Qualify the server as "local" in the connecting message.

            Ignored when `resuming` is `True`.

    Returns:
        Content with a connecting status message.
    """
    if resuming:
        text = "\nResuming...\n"
    elif local_server:
        text = "\nConnecting to local server...\n"
    else:
        text = "\nConnecting to server...\n"
    return Content.styled(text, "dim")


def build_welcome_footer(
    *, primary_color: str = theme.PRIMARY, tip: str | None = None
) -> Content:
    """Build the footer shown at the bottom of the welcome banner.

    Includes a tip to help users discover features.

    Args:
        primary_color: Color string for the ready prompt.

            Defaults to the module-level ANSI `PRIMARY` constant; widget callers
            should pass the active theme's hex value.
        tip: Tip text to display. When `None`, a random tip is selected.

            Pass an explicit value to keep the tip stable across re-renders.

    Returns:
        Content with the ready prompt and a tip.
    """
    if tip is None:
        tip = random.choice(_TIPS)  # noqa: S311
    return Content.assemble(
        ("\nReady to code! What would you like to build?\n", primary_color),
        (f"Tip: {tip}", "dim italic"),
    )
