"""Read-only MCP server and tool viewer modal."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.events import (
    Click,  # noqa: TC002 - needed at runtime for Textual event dispatch
)
from textual.screen import ModalScreen
from textual.widgets import Input, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

from deepagents_cli import theme
from deepagents_cli.config import Glyphs, get_glyphs, is_ascii_mode
from deepagents_cli.mcp_tools import MCPServerInfo, MCPServerStatus, MCPToolInfo


def _status_glyph(status: MCPServerStatus, glyphs: Glyphs) -> str:
    """Return the glyph character for a server `status`.

    Maps onto the existing `Glyphs` set so ASCII fallback is automatic
    (`✓ ⚠ ✗` -> `[OK] [!] [X]`). No new glyph definitions needed.
    """
    if status == "ok":
        return glyphs.checkmark
    if status == "unauthenticated":
        return glyphs.warning
    return glyphs.error


def _status_color(status: MCPServerStatus, colors: theme.ThemeColors) -> str:
    """Map a server `status` onto a semantic theme color.

    `ok` -> success (green); `unauthenticated` -> warning (yellow);
    `error` -> error (red). Returning the theme's hex string lets callers
    pass the value to `Content.styled()` or `Content.assemble()` so a
    theme switch recolors the indicator without code changes.
    """
    if status == "ok":
        return colors.success
    if status == "unauthenticated":
        return colors.warning
    return colors.error


def _tool_haystack(tool: MCPToolInfo) -> str:
    """Return the searchable text for a single tool (lower-cased).

    Includes the tool name, description, and any parameter names from
    `input_schema.properties`. Param names are a small bonus signal so
    `"path"` finds tools that accept `path: string` even when neither the
    name nor description mentions it.
    """
    parts = [tool.name, tool.description or ""]
    schema = tool.input_schema
    if schema:
        properties = schema.get("properties") or {}
        if isinstance(properties, dict):
            parts.extend(str(name) for name in properties)
    return " ".join(parts).lower()


def _server_haystack(server: MCPServerInfo) -> str:
    """Return the searchable text for a server header (lower-cased)."""
    return f"{server.name} {server.transport}".lower()


def _visible_tools_for(
    server: MCPServerInfo, tokens: list[str]
) -> tuple[MCPToolInfo, ...] | None:
    """Return the tools to render for `server` under the active filter.

    - `tokens=[]` (empty filter) -> all tools (or `()` if the server has none).
    - Server-name match across all tokens -> all tools.
    - Otherwise -> only tools whose haystack matches every token.
    - Returns `None` when nothing about the server matches and no tools
      survive — the caller should skip rendering the header entirely.
    """
    if not tokens:
        return server.tools

    if all(token in _server_haystack(server) for token in tokens):
        return server.tools

    matching = tuple(
        tool
        for tool in server.tools
        if all(token in _tool_haystack(tool) for token in tokens)
    )
    if matching:
        return matching
    return None


class MCPToolItem(Static):
    """A selectable tool item in the MCP viewer."""

    def __init__(
        self,
        name: str,
        description: str,
        index: int,
        *,
        classes: str = "",
        input_schema: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a tool item.

        Args:
            name: Tool name.
            description: Full tool description.
            index: Flat index of this tool in the list.
            classes: CSS classes.
            input_schema: Raw MCP `inputSchema` dict; rendered as parameters
                when the tool is expanded. `None` is treated as "no schema".
        """
        self.tool_name = name
        self.tool_description = description
        self.index = index
        self._input_schema = input_schema
        self._expanded = False
        self._selected = "mcp-tool-selected" in classes
        # Pass a placeholder label — `_format_collapsed` reads `self.size`,
        # which is only valid after the widget is attached to a screen.
        # `on_mount` re-renders with width-aware truncation.
        super().__init__(classes=classes)

    def _desc_style(self) -> str:
        """Return the markup style tag for the description span.

        Dim text on the `$primary` selection background is unreadable, so
        selected rows drop the dim and use bold for tool names only.
        """
        return "" if self._selected else "dim"

    def _format_collapsed(self, name: str, description: str) -> Content:
        """Build the collapsed (single-line) label.

        Truncates the description with `(...)` if it would overflow
        the widget width.

        Args:
            name: Tool name.
            description: Tool description.

        Returns:
            Styled Content label.
        """
        if not description:
            return Content.from_markup("  $name", name=name)
        prefix_len = 2 + len(name) + 1
        avail = self.size.width - prefix_len - 1 if self.size.width else 0
        ellipsis = " (...)"
        if avail > 0 and len(description) > avail:
            cut = max(0, avail - len(ellipsis))
            desc_text = description[:cut] + ellipsis
        else:
            desc_text = description
        style = self._desc_style()
        template = "  $name [" + style + "]$desc[/]" if style else "  $name $desc"
        return Content.from_markup(template, name=name, desc=desc_text)

    def _format_expanded(self, name: str, description: str) -> Content:
        """Build the expanded (multi-line) label.

        When `input_schema` carries a non-empty `properties` dict, append
        a `Parameters:` block listing each parameter as `name: type` with
        `*` for required.

        Args:
            name: Tool name.
            description: Tool description.

        Returns:
            Styled Content label with description and parameters on
            following lines.
        """
        style = self._desc_style()
        if description:
            template = (
                "  [bold]$name[/bold]\n    [" + style + "]$desc[/]"
                if style
                else "  [bold]$name[/bold]\n    $desc"
            )
            base = Content.from_markup(template, name=name, desc=description)
        else:
            base = Content.from_markup("  [bold]$name[/bold]", name=name)

        params = self._format_parameters()
        return base.append(params) if params is not None else base

    def _format_parameters(self) -> Content | None:
        """Build the parameter list rendered below the description.

        Returns `None` when there is no `input_schema`, the schema is not
        an object with non-empty `properties`, or `properties` is malformed.
        """
        schema = self._input_schema
        if not schema:
            return None
        properties = schema.get("properties")
        if not isinstance(properties, dict) or not properties:
            return None
        required = schema.get("required") or []
        if not isinstance(required, list):
            required = []
        required_set = {str(item) for item in required}

        style = self._desc_style()
        line_style = style or "dim"
        # Header line — always dim/muted to keep visual hierarchy below the
        # tool name and description.
        result = Content.from_markup(
            "\n    [" + line_style + "]Parameters:[/]"
        )
        for prop_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict):
                prop_type = str(prop_schema.get("type") or "any")
            else:
                prop_type = "any"
            star = " *" if str(prop_name) in required_set else ""
            # `Content.from_markup` substitution escapes user-supplied
            # text, so a parameter named `[bold]foo[/]` cannot inject
            # markup tags into the output.
            line = Content.from_markup(
                "\n      [" + line_style + "]$name: $ptype$star[/]",
                name=str(prop_name),
                ptype=prop_type,
                star=star,
            )
            result = result.append(line)
        return result

    def _rerender(self) -> None:
        """Re-render the label with the current selected/expanded state."""
        if self._expanded:
            self.update(self._format_expanded(self.tool_name, self.tool_description))
        else:
            self.update(self._format_collapsed(self.tool_name, self.tool_description))

    def set_selected(self, selected: bool) -> None:
        """Apply or remove the selected-row styling and re-render the label."""
        if self._selected == selected:
            return
        self._selected = selected
        if selected:
            self.add_class("mcp-tool-selected")
        else:
            self.remove_class("mcp-tool-selected")
        self._rerender()

    def toggle_expand(self) -> None:
        """Toggle between collapsed and expanded view."""
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded: bool) -> None:
        """Set expansion state explicitly; no-op when already in that state.

        Provides a single seam through which expansion changes flow, so the
        screen-level `Ctrl+E` toggle-all action and the per-row `toggle_expand`
        can share the same render path.
        """
        if expanded == self._expanded:
            return
        self._expanded = expanded
        self.styles.height = "auto" if expanded else 1
        self._rerender()

    def on_mount(self) -> None:
        """Re-render with correct truncation once width is known."""
        self._rerender()

    def on_resize(self) -> None:
        """Re-truncate when widget width changes."""
        if not self._expanded:
            self.update(self._format_collapsed(self.tool_name, self.tool_description))

    def on_click(self, event: Click) -> None:
        """Handle click — select and toggle expand via parent screen.

        Args:
            event: The click event.
        """
        event.stop()
        screen = self.screen
        if isinstance(screen, MCPViewerScreen):
            screen._move_to(self.index)
            self.toggle_expand()


class MCPViewerScreen(ModalScreen[None]):
    """Modal viewer for active MCP servers and their tools.

    Displays servers grouped by name with transport type and tool count.
    Navigate with arrow keys, Enter to expand/collapse tool descriptions,
    Escape to close.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False, priority=True),
        Binding("shift+tab", "move_up", "Up", show=False, priority=True),
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("tab", "move_down", "Down", show=False, priority=True),
        Binding("enter", "toggle_expand", "Expand", show=False, priority=True),
        # Use a non-letter chord so it does not steal text input from the
        # filter Input. PR #2949 originally proposed `a` for the same
        # action; we rebound to `ctrl+e` for that reason.
        Binding("ctrl+e", "toggle_all", "Toggle all", show=False, priority=True),
        Binding("pageup", "page_up", "Page up", show=False, priority=True),
        Binding("pagedown", "page_down", "Page down", show=False, priority=True),
        Binding("escape", "cancel", "Close", show=False, priority=True),
    ]
    """Key bindings for navigation, expansion, and cancel.

    All bindings use `priority=True` so they take precedence over the
    embedded filter `Input`. Vim-style `j`/`k` bindings are deliberately
    omitted because they would prevent typing those letters into the
    always-focused filter input — same rationale as `model_selector.py`.
    """

    CSS = """
    MCPViewerScreen {
        align: center middle;
    }

    MCPViewerScreen > Vertical {
        width: 80;
        max-width: 90%;
        height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    MCPViewerScreen .mcp-viewer-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    MCPViewerScreen #mcp-filter {
        margin-bottom: 1;
        border: solid $primary-lighten-2;
    }

    MCPViewerScreen #mcp-filter:focus {
        border: solid $primary;
    }

    MCPViewerScreen .mcp-list {
        height: 1fr;
        min-height: 5;
        scrollbar-gutter: stable;
        background: $background;
    }

    MCPViewerScreen .mcp-server-header {
        color: $primary;
        margin-top: 1;
    }

    MCPViewerScreen .mcp-list > .mcp-server-header:first-child {
        margin-top: 0;
    }

    MCPViewerScreen .mcp-tool-item {
        height: 1;
        padding: 0 1;
    }

    MCPViewerScreen .mcp-tool-item:hover {
        background: $surface-lighten-1;
    }

    MCPViewerScreen .mcp-tool-selected {
        background: $primary;
        color: $text;
        text-style: bold;
    }

    MCPViewerScreen .mcp-tool-selected:hover {
        background: $primary-lighten-1;
        color: $text;
    }

    MCPViewerScreen .mcp-empty {
        color: $text-muted;
        text-style: italic;
        text-align: center;
        margin-top: 2;
    }

    MCPViewerScreen .mcp-viewer-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }
    """

    def __init__(
        self,
        server_info: list[MCPServerInfo],
        *,
        connecting: bool = False,
    ) -> None:
        """Initialize the MCP viewer screen.

        Args:
            server_info: List of MCP server metadata to display.
            connecting: When `True` and `server_info` is empty, show a
                "connecting..." placeholder instead of the "no servers"
                message; the screen refreshes when `refresh_server_info`
                is called after the server startup completes.
        """
        super().__init__()
        self._server_info = server_info
        self._connecting = connecting
        self._tool_widgets: list[MCPToolItem] = []
        self._selected_index = 0
        self._query: str = ""

    def refresh_server_info(self, server_info: list[MCPServerInfo]) -> None:
        """Replace the displayed server list; typically after server startup.

        Rebuilds the modal body in place so a user who opened `/mcp` before
        tools finished loading sees them appear without closing/reopening.
        """
        self._server_info = server_info
        self._connecting = False
        body = self.query_one(Vertical)
        body.remove_children()
        self._tool_widgets = []
        self._selected_index = 0
        self._mount_body(body)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Rebuild the visible tool list whenever the filter input changes.

        Only the scroll's children are torn down — the title, filter Input,
        and help footer stay mounted so focus is preserved across keystrokes.
        """
        if event.input.id != "mcp-filter":
            return
        self._query = event.value
        scroll = self.query_one(".mcp-list", VerticalScroll)
        scroll.remove_children()
        self._tool_widgets = []
        self._selected_index = 0
        self._populate_scroll(scroll, self._query)

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual requires an instance method
        """Compose the screen layout.

        Yields:
            Empty `Vertical` — `_mount_body` fills it on mount so the same
            builder can also refresh the screen in place after server-ready.
        """
        yield Vertical()

    def on_mount(self) -> None:
        """Build the body once the screen is mounted."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)
        self._mount_body(self.query_one(Vertical))

    def _mount_body(self, container: Vertical) -> None:
        """Populate `container` with the title, filter input, list, and help footer.

        The filter Input and scroll container are mounted once. Subsequent
        filter rebuilds replace only the scroll's children via
        `_populate_scroll`, keeping the Input focused across keystrokes.
        """
        glyphs = get_glyphs()
        total_servers = len(self._server_info)
        total_tools = sum(len(s.tools) for s in self._server_info)

        if total_servers:
            server_label = "server" if total_servers == 1 else "servers"
            tool_label = "tool" if total_tools == 1 else "tools"
            title = (
                f"MCP Servers ({total_servers} {server_label},"
                f" {total_tools} {tool_label})"
            )
        else:
            title = "MCP Servers"
        container.mount(Static(title, classes="mcp-viewer-title"))

        # Suppress the filter Input while the connecting placeholder is
        # showing — there's nothing to filter yet.
        if self._server_info:
            container.mount(
                Input(
                    id="mcp-filter",
                    placeholder="Filter tools...",
                    value=self._query,
                )
            )

        scroll = VerticalScroll(classes="mcp-list")
        container.mount(scroll)
        self._populate_scroll(scroll, self._query)

        help_text = (
            f"{glyphs.arrow_up}/{glyphs.arrow_down} or Tab navigate"
            f" {glyphs.bullet} Enter expand/collapse"
            f" {glyphs.bullet} Type to filter"
            f" {glyphs.bullet} Esc close"
        )
        container.mount(Static(help_text, classes="mcp-viewer-help"))

    def _populate_scroll(self, scroll: VerticalScroll, query: str) -> None:
        """Mount filtered server headers + tool items into `scroll`.

        Empty `query` shows everything; otherwise multi-token AND matching
        across server names, transport, tool names, descriptions, and
        parameter names.
        """
        glyphs = get_glyphs()

        if not self._server_info:
            placeholder = (
                "Loading MCP tools — server is starting up..."
                if self._connecting
                else ("No MCP servers configured.\nUse `--mcp-config` to load servers.")
            )
            scroll.mount(Static(placeholder, classes="mcp-empty"))
            return

        tokens = [tok for tok in query.lower().split() if tok]
        colors = theme.get_theme_colors(self)
        flat_index = 0
        rendered_any_tool = False

        for server in self._server_info:
            visible_tools = _visible_tools_for(server, tokens)
            if visible_tools is None:
                # Server filtered out entirely.
                continue

            tool_count = len(visible_tools)
            t_label = "tool" if tool_count == 1 else "tools"
            indicator_color = _status_color(server.status, colors)
            indicator_glyph = _status_glyph(server.status, glyphs)
            if server.status == "ok":
                header_content = Content.assemble(
                    (f"{indicator_glyph} ", indicator_color),
                    (server.name, "bold"),
                    (
                        f" {server.transport} {glyphs.bullet}"
                        f" {tool_count} {t_label}",
                        "dim",
                    ),
                )
            else:
                error_text = server.error or ""
                header_content = Content.assemble(
                    (f"{indicator_glyph} ", indicator_color),
                    (server.name, "bold"),
                    (f" {server.transport}", "dim"),
                    (f" {glyphs.bullet} {server.status}", indicator_color),
                    (f" — {error_text}", "dim") if error_text else "",
                )
            scroll.mount(Static(header_content, classes="mcp-server-header"))
            for tool in visible_tools:
                classes = "mcp-tool-item"
                if flat_index == 0:
                    classes += " mcp-tool-selected"
                widget = MCPToolItem(
                    name=tool.name,
                    description=tool.description,
                    index=flat_index,
                    classes=classes,
                    input_schema=tool.input_schema,
                )
                self._tool_widgets.append(widget)
                scroll.mount(widget)
                flat_index += 1
                rendered_any_tool = True

        if tokens and not self._tool_widgets and not rendered_any_tool:
            scroll.mount(Static("No matching tools.", classes="mcp-empty"))

    def _move_to(self, index: int) -> None:
        """Move selection to the given index.

        Args:
            index: Target tool index.
        """
        if not self._tool_widgets:
            return
        old = self._selected_index
        self._selected_index = index

        if old != index:
            self._tool_widgets[old].set_selected(False)
            self._tool_widgets[index].set_selected(True)
            self._tool_widgets[index].scroll_visible()

    def _move_selection(self, delta: int) -> None:
        """Move selection by delta positions.

        Args:
            delta: Number of positions to move.
        """
        if not self._tool_widgets:
            return
        count = len(self._tool_widgets)
        target = (self._selected_index + delta) % count
        self._move_to(target)

    def action_move_up(self) -> None:
        """Move selection up."""
        self._move_selection(-1)

    def action_move_down(self) -> None:
        """Move selection down."""
        self._move_selection(1)

    def action_toggle_expand(self) -> None:
        """Toggle expand/collapse on the selected tool."""
        if self._tool_widgets:
            self._tool_widgets[self._selected_index].toggle_expand()

    def action_toggle_all(self) -> None:
        """Expand or collapse every visible tool at once.

        If any visible tool is collapsed, expand all; otherwise collapse all.
        Operates on `_tool_widgets`, so a filtered view affects only the
        currently visible subset — hidden tools keep their state.
        """
        if not self._tool_widgets:
            return
        any_collapsed = any(not w._expanded for w in self._tool_widgets)
        for widget in self._tool_widgets:
            widget.set_expanded(any_collapsed)

    def action_page_up(self) -> None:
        """Scroll up by one page."""
        scroll = self.query_one(".mcp-list", VerticalScroll)
        scroll.scroll_page_up()

    def action_page_down(self) -> None:
        """Scroll down by one page."""
        scroll = self.query_one(".mcp-list", VerticalScroll)
        scroll.scroll_page_down()

    def action_cancel(self) -> None:
        """Close the viewer."""
        self.dismiss(None)
