"""Welcome banner widget."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual.color import Color as TColor
from textual.content import Content
from textual.style import Style as TStyle
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.events import Click, MouseMove

from deepagents_code import theme
from deepagents_code._env_vars import (
    HIDE_CWD,
    HIDE_SPLASH_VERSION,
    is_env_truthy,
)
from deepagents_code._version import __version__
from deepagents_code.config import _is_editable_install, get_glyphs
from deepagents_code.widgets._links import open_style_link

_ANSI_THEMES = {"ansi-dark", "ansi-light"}


def _home_prefixed(cwd: str) -> str:
    """Format a directory path, using `~` for the home directory when possible.

    Args:
        cwd: Absolute working directory path.

    Returns:
        The path with the home prefix collapsed to `~` when applicable.
    """
    path = Path(cwd)
    try:
        home = Path.home()
        if path.is_relative_to(home):
            return "~/" + path.relative_to(home).as_posix()
    except (ValueError, RuntimeError):
        pass
    return str(path)


class WelcomeBanner(Static):
    """Compact welcome banner shown at startup.

    Renders a bordered box with the product title, the active model, and the
    working directory. Connection, MCP, and thread state are tracked for API
    compatibility but not displayed here — the status bar owns that surface.
    """

    # Disable Textual's auto_links to prevent a flicker cycle: Style.__add__
    # calls .copy() for linked styles, generating a fresh random _link_id on
    # each render. This means highlight_link_id never stabilizes, causing an
    # infinite hover-refresh loop.
    auto_links = False

    DEFAULT_CSS = """
    WelcomeBanner {
        height: auto;
        border: round $primary;
        padding: 0 2;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        model_provider: str = "",
        model_name: str = "",
        cwd: str | None = None,
        thread_id: str | None = None,
        mcp_tool_count: int = 0,
        *,
        mcp_unauthenticated: int = 0,
        mcp_errored: int = 0,
        mcp_awaiting_reconnect: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the welcome banner.

        Args:
            model_provider: Active model provider (e.g. `anthropic`).
            model_name: Active model name.
            cwd: Working directory to display. Defaults to the process cwd.
            thread_id: Tracked for API compatibility; not displayed.
            mcp_tool_count: Tracked for API compatibility; not displayed.
            mcp_unauthenticated: Tracked for API compatibility; not displayed.
            mcp_errored: Tracked for API compatibility; not displayed.
            mcp_awaiting_reconnect: Tracked for API compatibility; not displayed.
            **kwargs: Additional arguments passed to parent.
        """
        self._model_provider = model_provider
        self._model_name = model_name
        self._cwd = cwd if cwd is not None else str(Path.cwd())
        self._hide_version = is_env_truthy(HIDE_SPLASH_VERSION)
        self._hide_cwd = is_env_truthy(HIDE_CWD)
        # Avoid collision with Widget._thread_id (Textual internal int)
        self._cli_thread_id = thread_id
        self._mcp_tool_count = mcp_tool_count
        self._mcp_unauthenticated = mcp_unauthenticated
        self._mcp_errored = mcp_errored
        self._mcp_awaiting_reconnect = mcp_awaiting_reconnect
        super().__init__(self._build_banner(), **kwargs)

    def on_mount(self) -> None:
        """Re-render the banner when the app theme changes."""
        self.watch(self.app, "theme", self._on_theme_change, init=False)

    def _on_theme_change(self) -> None:
        """Re-render the banner when the app theme changes."""
        self.update(self._build_banner())

    def update_model(self, *, provider: str, model: str) -> None:
        """Update the displayed model and re-render.

        Args:
            provider: Active model provider.
            model: Active model name.
        """
        self._model_provider = provider
        self._model_name = model
        self.update(self._build_banner())

    def update_thread_id(self, thread_id: str) -> None:
        """Track a new thread ID (not displayed in the minimal banner).

        Args:
            thread_id: The new thread ID.
        """
        self._cli_thread_id = thread_id

    def set_connected(
        self,
        mcp_tool_count: int = 0,
        *,
        mcp_unauthenticated: int = 0,
        mcp_errored: int = 0,
        mcp_awaiting_reconnect: int = 0,
    ) -> None:
        """Track MCP/connection counts (not displayed; status bar owns this).

        Args:
            mcp_tool_count: Number of MCP tools loaded during connection.
            mcp_unauthenticated: Number of MCP servers awaiting login.
            mcp_errored: Number of MCP servers that failed to load.
            mcp_awaiting_reconnect: Number of MCP servers awaiting reconnect.
        """
        self._mcp_tool_count = mcp_tool_count
        self._mcp_unauthenticated = mcp_unauthenticated
        self._mcp_errored = mcp_errored
        self._mcp_awaiting_reconnect = mcp_awaiting_reconnect

    def set_connecting(self) -> None:
        """No-op retained for API compatibility; status bar owns progress."""

    def set_idle(self) -> None:
        """No-op retained for API compatibility; status bar owns failures."""

    def on_click(self, event: Click) -> None:  # noqa: PLR6301  # Textual event handler
        """Open style-embedded hyperlinks on single click."""
        open_style_link(event)

    def on_mouse_move(self, event: MouseMove) -> None:
        """Show a hand pointer over link spans and reset it elsewhere."""
        self.styles.pointer = "pointer" if event.style.link else "default"

    def on_leave(self) -> None:
        """Reset the pointer shape when the mouse leaves the banner."""
        self.styles.pointer = "default"

    def _build_banner(self) -> Content:
        """Build the minimal banner content.

        Returns:
            Content with the title, model, and directory rows.
        """
        colors = theme.get_theme_colors(self)
        ansi = self.app.theme in _ANSI_THEMES
        accent: str | TStyle = "bold" if ansi else colors.primary
        title_style: str | TStyle = (
            "bold"
            if ansi
            else TStyle(foreground=TColor.parse(colors.primary), bold=True)
        )

        parts: list[str | tuple[str, str | TStyle]] = [
            (f"{get_glyphs().cursor} ", title_style),
            ("dcode", "bold"),
        ]
        if not self._hide_version:
            parts.append((f"  v{__version__}", "dim"))
            if not ansi and _is_editable_install():
                parts.append(
                    (
                        " (local)",
                        TStyle(foreground=TColor.parse(colors.tool), bold=True),
                    )
                )

        rows: list[list[tuple[str, str | TStyle]]] = []
        if self._model_name:
            model_value = (
                f"{self._model_provider}:{self._model_name}"
                if self._model_provider
                else self._model_name
            )
            rows.append([("model:     ", "dim"), (model_value, accent)])
        if not self._hide_cwd and self._cwd:
            rows.append([("directory: ", "dim"), (_home_prefixed(self._cwd), accent)])

        for index, row in enumerate(rows):
            parts.append("\n\n" if index == 0 else "\n")
            parts.extend(row)

        return Content.assemble(*parts)
