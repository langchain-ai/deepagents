"""Welcome banner widget."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from textual.color import Color as TColor
from textual.content import Content
from textual.style import Style as TStyle
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.events import Click, MouseMove

from deepagents_code import theme
from deepagents_code._env_vars import (
    DEBUG,
    HIDE_CWD,
    HIDE_LANGSMITH_TRACING,
    HIDE_SPLASH_VERSION,
    SHOW_LANGSMITH_REPLICA_TRACING,
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
    get_langsmith_replica_project,
)
from deepagents_code.tui.widgets._links import open_style_link

logger = logging.getLogger(__name__)

_ANSI_THEMES: Final[frozenset[str]] = frozenset({"ansi-dark", "ansi-light"})
"""Theme names whose color palette is determined by the terminal emulator
rather than by the app, so link styles use bold instead of a parsed color."""

_LANGSMITH_UTM_SOURCE: Final[str] = "deepagents-code"
"""UTM source tag appended to LangSmith project URLs in the welcome banner."""


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


def _local_tag_style(*, ansi: bool, colors: theme.ThemeColors) -> str | TStyle:
    """Build the style for the editable-install `(local)` tag.

    Args:
        ansi: Whether the active theme is an ANSI terminal theme.
        colors: Active Deep Agents theme colors.

    Returns:
        A bold markup style under ANSI themes (whose palette the terminal owns,
            so a parsed color could be invisible) or a bold themed
            color otherwise.
    """
    if ansi:
        return "bold"
    return TStyle(foreground=TColor.parse(colors.tool), bold=True)


def _debug_tag_style(*, ansi: bool, colors: theme.ThemeColors) -> str | TStyle:
    """Build the style for the `(debug enabled)` tag.

    Args:
        ansi: Whether the active theme is an ANSI terminal theme.
        colors: Active Deep Agents theme colors.

    Returns:
        A bold yellow markup style under ANSI themes (whose palette the terminal
            owns, so a parsed color could be invisible) or a bold themed warning
            color otherwise.
    """
    if ansi:
        return "bold yellow"
    return TStyle(foreground=TColor.parse(colors.warning), bold=True)


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
        if path == home:
            return "~"
        if path.is_relative_to(home):
            return "~/" + path.relative_to(home).as_posix()
    # `Path.home()` raises `RuntimeError` when the home dir can't be resolved
    # (no HOME/passwd entry); `ValueError` guards odd paths (e.g. embedded NUL).
    # Either way, fall back to the accurate absolute path.
    except (ValueError, RuntimeError):
        pass
    return str(path)


class WelcomeBanner(Static):
    """Compact welcome banner shown at startup.

    Renders a bordered box with the product title and version. The version
    carries a `(debug enabled)` tag when `DEEPAGENTS_CODE_DEBUG` is set and a
    `(local)` tag for editable installs; both appear only when the version is
    shown. Rows follow that appear only when their data (and any env gate) is
    present. In render order: the active model (`SPLASH_SHOW_MODEL`, opt-in),
    working directory (`SPLASH_SHOW_CWD`, opt-in), LangSmith tracing project and
    its replica (each clickable once its URL resolves), thread ID (debug mode
    only), and the MCP tool count. MCP server warnings and the editable-install
    path follow.
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
        # `_show_cwd` and `_hide_cwd` are deliberately orthogonal: `_show_cwd`
        # gates the working-directory row (opt-in), while `_hide_cwd` gates the
        # editable-install path below. They govern different surfaces, so both
        # can be set without contradiction.
        self._show_cwd = is_env_truthy(SPLASH_SHOW_CWD)
        self._hide_cwd = is_env_truthy(HIDE_CWD)
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
        show_replica_tracing = is_env_truthy(
            SHOW_LANGSMITH_REPLICA_TRACING,
            default=True,
        )
        self._replica_project: str | None = (
            get_langsmith_replica_project()
            if self._project_name and show_replica_tracing
            else None
        )
        self._project_urls: dict[str, str] = {}
        self._debug_enabled = is_env_truthy(DEBUG)
        self._show_thread_id = self._debug_enabled
        super().__init__(self._build_banner(), **kwargs)

    def on_mount(self) -> None:
        """Watch for theme changes and start the LangSmith project-URL fetch."""
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
        projects = dict.fromkeys(
            project
            for project in (self._project_name, self._replica_project)
            if project
        )
        for project in projects:
            try:
                project_url = await asyncio.wait_for(
                    asyncio.to_thread(fetch_langsmith_project_url, project),
                    timeout=2.0,
                )
            except (TimeoutError, OSError):
                logger.debug(
                    "LangSmith project URL fetch failed for %r", project, exc_info=True
                )
                project_url = None
            if project_url:
                self._project_urls[project] = project_url
                self.update(self._build_banner())

    def _project_url(self, project: str | None) -> str | None:
        """Return the resolved LangSmith URL for a project.

        Args:
            project: Project name to look up.

        Returns:
            Resolved project URL, or `None` when missing or not yet fetched.
        """
        if project is None:
            return None
        return self._project_urls.get(project)

    def update_model(self, *, provider: str, model: str) -> None:
        """Track a new model and re-render when it is displayed.

        Args:
            provider: Active model provider.
            model: Active model name.
        """
        self._model_provider = provider
        self._model_name = model
        if self._show_model:
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
            `SPLASH_SHOW_CWD`), tracing and replica (each clickable once its URL
            resolves), thread ID (debug only), MCP tool count, MCP server
            warnings, and the editable-install path.
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
            if self._debug_enabled:
                parts.append(
                    (
                        " (debug enabled)",
                        _debug_tag_style(ansi=ansi, colors=colors),
                    )
                )
            if _is_editable_install():
                parts.append((" (local)", _local_tag_style(ansi=ansi, colors=colors)))

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
            project_url = self._project_url(self._project_name)
            if project_url:
                rows.append(
                    [
                        ("tracing:   ", "dim"),
                        (
                            f"'{self._project_name}'",
                            _langsmith_project_link_style(
                                project_url,
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
        if self._replica_project:
            replica_url = self._project_url(self._replica_project)
            if replica_url:
                rows.append(
                    [
                        ("replica:   ", "dim"),
                        (
                            f"'{self._replica_project}'",
                            _langsmith_project_link_style(
                                replica_url,
                                ansi=ansi,
                                colors=colors,
                            ),
                        ),
                    ]
                )
            else:
                rows.append(
                    [("replica:   ", "dim"), (f"'{self._replica_project}'", accent)]
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
        if not self._hide_version and not self._hide_cwd:
            editable_path = _get_editable_install_path()
            if editable_path:
                parts.append("\n")
                parts.extend([("installed: ", "dim"), (editable_path, "dim")])

        return Content.assemble(*parts)
