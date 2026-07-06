"""Welcome banner widget."""

from __future__ import annotations

import asyncio
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
    DEBUG,
    HIDE_LANGSMITH_TRACING,
    HIDE_SPLASH_VERSION,
    SPLASH_SHOW_CWD,
    SPLASH_SHOW_MODEL,
    is_env_truthy,
)
from deepagents_code._version import __version__
from deepagents_code.config import (
    _get_editable_install_path,
    _is_editable_install,
    fetch_langsmith_project_url,
    get_glyphs,
    get_langsmith_project_name,
)
from deepagents_code.widgets._links import open_style_link

_ANSI_THEMES = {"ansi-dark", "ansi-light"}

_LANGSMITH_UTM_SOURCE = "deepagents-code"


def _langsmith_project_link(project_url: str) -> str:
    """Append the Deep Agents source tag to a LangSmith project URL.

    Args:
        project_url: LangSmith project URL.

    Returns:
        Project URL with the Deep Agents source tag.
    """
    return f"{project_url}?utm_source={_LANGSMITH_UTM_SOURCE}"


def _langsmith_project_link_style(
    project_url: str,
    *,
    ansi: bool,
    colors: theme.ThemeColors,
) -> TStyle:
    """Build the clickable style for a LangSmith project name.

    Args:
        project_url: LangSmith project URL.
        ansi: Whether the active theme is an ANSI terminal theme.
        colors: Active Deep Agents theme colors.

    Returns:
        Link style for a LangSmith project name.
    """
    link = _langsmith_project_link(project_url)
    if ansi:
        return TStyle(bold=True, link=link)
    return TStyle(foreground=TColor.parse(colors.primary), link=link)


def _home_prefixed(cwd: str) -> str:
    """Format a directory path, using `~` for the home directory when possible.

    Args:
        cwd: Working directory path.

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

    Renders a bordered box with the product title and version. Additional rows
    appear only when their data is present: the LangSmith tracing project
    (clickable once the URL resolves), the MCP tool count, MCP server warnings,
    and the editable-install path. The active model (`SPLASH_SHOW_MODEL`) and working
    directory (`SPLASH_SHOW_CWD`) rows are opt-in, and the thread ID appears only in
    debug mode.
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
        thread_id: str | None = None,
        mcp_tool_count: int = 0,
        *,
        model_provider: str = "",
        model_name: str = "",
        cwd: str | None = None,
        mcp_unauthenticated: int = 0,
        mcp_errored: int = 0,
        mcp_awaiting_reconnect: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the welcome banner.

        Args:
            thread_id: Displayed only when debug mode is enabled.
            mcp_tool_count: Number of MCP tools loaded at startup.
            model_provider: Active model provider (e.g. `anthropic`). Displayed
                only when `SPLASH_SHOW_MODEL` is enabled.
            model_name: Active model name. Displayed only when `SPLASH_SHOW_MODEL` is
                enabled.
            cwd: Working directory. Defaults to the process cwd. Displayed only
                when `SPLASH_SHOW_CWD` is enabled.
            mcp_unauthenticated: Number of MCP servers awaiting login.
            mcp_errored: Number of MCP servers that failed to load.
            mcp_awaiting_reconnect: Number of MCP servers awaiting reconnect.
            **kwargs: Additional arguments passed to parent.
        """
        self._model_provider = model_provider
        self._model_name = model_name
        self._cwd = cwd if cwd is not None else str(Path.cwd())
        self._show_model = is_env_truthy(SPLASH_SHOW_MODEL)
        self._show_cwd = is_env_truthy(SPLASH_SHOW_CWD)
        self._hide_version = is_env_truthy(HIDE_SPLASH_VERSION)
        # Avoid collision with Widget._thread_id (Textual internal int)
        self._cli_thread_id = thread_id
        self._mcp_tool_count = mcp_tool_count
        self._mcp_unauthenticated = mcp_unauthenticated
        self._mcp_errored = mcp_errored
        self._mcp_awaiting_reconnect = mcp_awaiting_reconnect
        self._hide_langsmith_tracing = is_env_truthy(HIDE_LANGSMITH_TRACING)
        self._project_name: str | None = (
            None if self._hide_langsmith_tracing else get_langsmith_project_name()
        )
        self._project_url: str | None = None
        self._show_thread_id = is_env_truthy(DEBUG)
        super().__init__(self._build_banner(), **kwargs)

    def on_mount(self) -> None:
        """Re-render the banner when the app theme changes."""
        self.watch(self.app, "theme", self._on_theme_change, init=False)
        if self._project_name:
            self.run_worker(self._fetch_and_update, exclusive=True)

    def _on_theme_change(self) -> None:
        """Re-render the banner when the app theme changes."""
        self.update(self._build_banner())

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

    def update_cwd(self, cwd: str) -> None:
        """Track a new working directory and re-render when it is displayed.

        Args:
            cwd: New working directory path.
        """
        self._cwd = cwd
        if self._show_cwd:
            self.update(self._build_banner())

    def update_thread_id(self, thread_id: str) -> None:
        """Track a new thread ID and re-render when debug mode is active.

        Args:
            thread_id: The new thread ID.
        """
        self._cli_thread_id = thread_id
        if self._show_thread_id:
            self.update(self._build_banner())

    def set_connected(
        self,
        mcp_tool_count: int = 0,
        *,
        mcp_unauthenticated: int = 0,
        mcp_errored: int = 0,
        mcp_awaiting_reconnect: int = 0,
    ) -> None:
        """Update MCP tool counts and re-render the banner.

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
        self.update(self._build_banner())

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
        """Build the banner content.

        Returns:
            Content with the title and version, followed by any applicable rows
            in order: model (when `SPLASH_SHOW_MODEL`), directory (when
            `SPLASH_SHOW_CWD`), tracing (clickable once the URL resolves), thread
            ID (debug only), MCP tool count, MCP server warnings, and the
            editable-install path.
        """
        colors = theme.get_theme_colors(self)
        ansi = self.app.theme in _ANSI_THEMES
        accent: str | TStyle = "bold" if ansi else colors.primary
        title_style: str | TStyle = (
            "bold"
            if ansi
            else TStyle(foreground=TColor.parse(colors.primary), bold=True)
        )
        warn_color: str = "bold yellow" if ansi else colors.warning

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

        # Row labels share a common column width so values stay aligned; the
        # longest label ("directory:") needs 11 columns including its space.
        rows: list[list[tuple[str, str | TStyle]]] = []
        if self._show_model and self._model_name:
            model_value = (
                f"{self._model_provider}:{self._model_name}"
                if self._model_provider
                else self._model_name
            )
            rows.append([("model:     ", "dim"), (model_value, accent)])
        if self._show_cwd and self._cwd:
            rows.append([("directory: ", "dim"), (_home_prefixed(self._cwd), accent)])
        if self._project_name:
            if self._project_url:
                rows.append(
                    [
                        ("tracing:   ", "dim"),
                        (
                            f"'{self._project_name}'",
                            _langsmith_project_link_style(
                                self._project_url,
                                ansi=ansi,
                                colors=colors,
                            ),
                        ),
                    ]
                )
            else:
                rows.append(
                    [("tracing:   ", "dim"), (f"'{self._project_name}'", accent)]
                )
        if self._show_thread_id and self._cli_thread_id:
            rows.append([("thread:    ", "dim"), (self._cli_thread_id, "dim")])
        if self._mcp_tool_count > 0:
            label = "tool" if self._mcp_tool_count == 1 else "tools"
            rows.append(
                [("mcp:       ", "dim"), (f"{self._mcp_tool_count} {label}", accent)]
            )

        for index, row in enumerate(rows):
            parts.append("\n\n" if index == 0 else "\n")
            parts.extend(row)

        # MCP server warnings — actionable alerts not shown elsewhere in the banner.
        warning_lines: list[list[tuple[str, str | TStyle]]] = []
        if self._mcp_unauthenticated > 0:
            server_label = "server" if self._mcp_unauthenticated == 1 else "servers"
            verb = "needs" if self._mcp_unauthenticated == 1 else "need"
            warning_lines.append(
                [
                    (f"{get_glyphs().warning} ", warn_color),
                    (
                        (
                            f"{self._mcp_unauthenticated} MCP {server_label} {verb}"
                            " login — open /mcp"
                        ),
                        "dim",
                    ),
                ]
            )
        if self._mcp_errored > 0:
            server_label = "server" if self._mcp_errored == 1 else "servers"
            warning_lines.append(
                [
                    (f"{get_glyphs().warning} ", warn_color),
                    (
                        (
                            f"{self._mcp_errored} MCP {server_label} failed to load"
                            " — open /mcp for details"
                        ),
                        "dim",
                    ),
                ]
            )
        if self._mcp_awaiting_reconnect > 0:
            server_label = "server" if self._mcp_awaiting_reconnect == 1 else "servers"
            warning_lines.append(
                [
                    (f"{get_glyphs().warning} ", warn_color),
                    (
                        (
                            f"{self._mcp_awaiting_reconnect} MCP {server_label}"
                            " ready to load — run `/mcp reconnect`"
                        ),
                        "dim",
                    ),
                ]
            )

        for line in warning_lines:
            parts.append("\n")
            parts.extend(line)

        # Editable-install path for local development visibility.
        if not self._hide_version:
            editable_path = _get_editable_install_path()
            if editable_path:
                parts.append("\n")
                parts.extend([("installed: ", "dim"), (editable_path, "dim")])

        return Content.assemble(*parts)
