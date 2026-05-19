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

    from deepagents_code.mcp_tools import MCPServerInfo, MCPServerStatus, MCPToolInfo

from deepagents_code import theme
from deepagents_code.config import Glyphs, get_glyphs, is_ascii_mode


def _status_glyph(status: MCPServerStatus, glyphs: Glyphs) -> str:
    """Return the glyph character for a server `status`.

    Maps onto the existing `Glyphs` set so ASCII fallback is automatic
    (`✓ ⚠ ✗` -> `[OK] [!] [X]`). No new glyph definitions needed.

    Args:
        status: One of `ok` / `unauthenticated` / `error`.
        glyphs: Active `Glyphs` table (Unicode or ASCII).

    Returns:
        The unicode or ASCII glyph character matching `status`.
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

    Args:
        status: One of `ok` / `unauthenticated` / `error`.
        colors: Active theme palette (typically from `theme.get_theme_colors`).

    Returns:
        Hex color string from the active theme.
    """
    if status == "ok":
        return colors.success
    if status == "unauthenticated":
        return colors.warning
    return colors.error


def _styled(inner: str, style: str) -> str:
    """Wrap a `Content.from_markup` template fragment in `[style]…[/]` if needed.

    Centralizes the `'[' + style + ']…[/]' if style else …` pattern that the
    three formatter methods would otherwise repeat five times.

    Args:
        inner: Template fragment (may contain `$var` substitutions).
        style: Active style string; empty string means render unstyled.

    Returns:
        `inner` wrapped in `[style]…[/]` when `style` is truthy, otherwise
        `inner` unchanged.
    """
    return f"[{style}]{inner}[/]" if style else inner


def _format_prop_type(prop_type: Any) -> str:  # noqa: ANN401 - JSON Schema field is intentionally untyped
    """Render a JSON Schema `type` field for parameter display.

    JSON Schema allows `type` to be a string (`"string"`) or a list of
    strings (`["string", "null"]` for nullable types). Plain `str()` on a
    list produces an ugly Python repr; we join with `|` instead.

    Args:
        prop_type: The raw value of the schema's `type` field.

    Returns:
        Display-friendly type string. `"any"` when `prop_type` is missing
        or not coercible to a meaningful string.
    """
    if prop_type is None:
        return "any"
    if isinstance(prop_type, list):
        parts = [str(t) for t in prop_type if t]
        return "|".join(parts) if parts else "any"
    return str(prop_type) or "any"


def _sanitize_inline(text: str, *, max_length: int = 200) -> str:
    """Strip control characters and truncate text rendered inline in a header.

    External-origin strings (e.g. `MCPServerInfo.error` returned by an MCP
    server) might contain newlines, ANSI escape sequences, or other control
    characters that would corrupt the modal layout. Replace control chars
    with spaces and truncate to `max_length` so a hostile or buggy server
    cannot break the screen.

    Args:
        text: Untrusted inline text.
        max_length: Maximum characters to retain (truncated with `…`).

    Returns:
        Sanitized single-line text safe to embed inside a `Content.assemble`
        tuple or `from_markup` substitution.
    """
    cleaned = "".join(ch if ch == " " or ch.isprintable() else " " for ch in text)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > max_length:
        cleaned = cleaned[: max_length - 1] + "…"
    return cleaned


def _visible_tools_for(
    server: MCPServerInfo, tokens: list[str]
) -> tuple[MCPToolInfo, ...] | None:
    """Return the tools to render for `server` under the active filter.

    Filter matches tool and server *names* only — descriptions, parameter
    names, and the transport are deliberately not in the haystack so long
    MCP docstrings don't produce spurious matches. A server with zero tools
    that matches by name returns `None` so the caller can skip rendering a
    stub header followed by the global "No matching tools" empty-state.

    Args:
        server: The server whose tools are candidates for display.
        tokens: Lower-cased filter tokens — empty means "no filter".

    Returns:
        - `server.tools` when the filter is empty or matches the server name
          and the server actually has tools.
        - A subset tuple when individual tool names match.
        - `None` when nothing matches, including the server-name-match case
          on a server with zero tools — caller skips the header entirely.
    """
    if not tokens:
        return server.tools

    if all(token in server.name.lower() for token in tokens):
        return server.tools or None

    matching = tuple(
        tool
        for tool in server.tools
        if all(token in tool.name.lower() for token in tokens)
    )
    return matching or None


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
        template = f"  $name {_styled('$desc', self._desc_style())}"
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
        if description:
            style = self._desc_style()
            template = f"  [bold]$name[/bold]\n    {_styled('$desc', style)}"
            base = Content.from_markup(template, name=name, desc=description)
        else:
            base = Content.from_markup("  [bold]$name[/bold]", name=name)

        params = self._format_parameters()
        return base.append(params) if params is not None else base

    def _format_parameters(self) -> Content | None:
        """Build the parameter list rendered below the description.

        Returns:
            A `Content` block with one line per parameter, or `None` when
            there is no `input_schema`, the schema is not an object with
            non-empty `properties`, or `properties` is malformed.
        """
        schema = self._input_schema
        if not schema or not isinstance(schema, dict):
            return None
        properties = schema.get("properties")
        if not isinstance(properties, dict) or not properties:
            return None
        required = schema.get("required") or []
        if not isinstance(required, list):
            required = []
        required_set = {str(item) for item in required}

        # Mirror `_desc_style`: empty when this row is selected, so the
        # parameter block stays readable on the `$primary` selection
        # background (CSS recolors text via `.mcp-tool-selected`). When
        # not selected, render dim so the params sit visually below the
        # description.
        style = self._desc_style()
        result = Content.from_markup("\n    " + _styled("Parameters:", style))
        line_template = "\n      " + _styled("$name: $ptype$star", style)
        for prop_name, prop_schema in properties.items():
            prop_type = _format_prop_type(
                prop_schema.get("type") if isinstance(prop_schema, dict) else None
            )
            star = " *" if str(prop_name) in required_set else ""
            # `Content.from_markup` substitution escapes user-supplied
            # text, so a parameter named `[bold]foo[/]` cannot inject
            # markup tags into the output. Newlines are stripped to
            # protect viewport-row math (smart-scroll relies on
            # `widget.region.height` matching the rendered row count).
            safe_name = str(prop_name).replace("\n", " ").replace("\r", " ")[:80]
            line = Content.from_markup(
                line_template,
                name=safe_name,
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
        """Set expansion state explicitly and re-render.

        Single seam through which expansion changes flow, so the screen-level
        `Ctrl+E` toggle-all action and the per-row `toggle_expand` share the
        same render path. Always re-applies `styles.height` and re-renders so
        callers do not need to know whether the state changed — the redundant
        write is cheap and avoids drift if `styles.height` was changed
        externally (CSS reload, theme switch, programmatic edit).

        Args:
            expanded: `True` for expanded multi-line view, `False` for
                collapsed single-line view.
        """
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


def _render_server_header(
    server: MCPServerInfo,
    indicator_glyph: str,
    indicator_color: str,
    visible_tools: tuple[MCPToolInfo, ...],
    glyphs: Glyphs,
    *,
    selected: bool = False,
) -> Content:
    """Build the styled header line for one server.

    Uses `Content.assemble`'s `(text, style)` tuple form so the per-span
    color is applied dynamically from the theme palette — `from_markup`
    does NOT substitute into bracket tags (`[$icolor]…[/]` would render
    as a literal/unknown tag, not a hex color). The tuple form is also
    injection-safe: each span's `text` is rendered verbatim, never
    markup-parsed, so a server name like `[bold]foo[/]` shows literally
    rather than getting styled. `server.error` additionally goes through
    `_sanitize_inline` because MCP servers can return arbitrary error
    text including newlines or terminal escapes.

    Args:
        server: The server whose header is being rendered.
        indicator_glyph: Status glyph (already chosen from `_status_glyph`).
        indicator_color: Status color (already chosen from `_status_color`).
        visible_tools: Tools that survived the active filter — used only
            for the count label.
        glyphs: Active `Glyphs` table for the bullet separator.
        selected: When `True`, suppresses `dim` styling on secondary spans
            so the text stays readable on the `$primary` selection background.

    Returns:
        Styled `Content` ready to mount inside a `Static`.
    """
    dim_style = "" if selected else "dim"
    tool_count = len(visible_tools)
    t_label = "tool" if tool_count == 1 else "tools"
    if server.status == "ok":
        summary = f" {server.transport} {glyphs.bullet} {tool_count} {t_label}"
        return Content.assemble(
            (f"{indicator_glyph} ", indicator_color),
            (server.name, "bold"),
            (summary, dim_style),
        )
    error_text = _sanitize_inline(server.error or "")
    return Content.assemble(
        (f"{indicator_glyph} ", indicator_color),
        (server.name, "bold"),
        (f" {server.transport}", dim_style),
        (f" {glyphs.bullet} {server.status}", indicator_color),
        (f" — {error_text}", dim_style) if error_text else "",
    )


class MCPServerHeaderItem(Static):
    """A selectable server-header row in the MCP viewer.

    Cursor-selectable so users can navigate to every server — even those
    in `unauthenticated` or `error` states which have no tool rows by the
    `MCPServerInfo` invariant — and read the full status / error text on
    the line. Not expandable: `Enter` and `Ctrl+E` are no-ops here.
    """

    def __init__(
        self,
        server: MCPServerInfo,
        indicator_glyph: str,
        indicator_color: str,
        visible_tools: tuple[MCPToolInfo, ...],
        glyphs: Glyphs,
        index: int,
        *,
        classes: str = "",
    ) -> None:
        """Initialize a server-header row.

        Args:
            server: Server metadata used to re-render content on selection
                state changes.
            indicator_glyph: Pre-computed status glyph character.
            indicator_color: Pre-computed status hex color string.
            visible_tools: Filtered tool tuple — used for the count label.
            glyphs: Active glyph table.
            index: Flat row index inside `MCPViewerScreen._row_widgets`.
            classes: CSS classes — should include `mcp-server-header`, and
                optionally `mcp-header-selected` for the initial selection.
        """
        self._server = server
        self._indicator_glyph = indicator_glyph
        self._indicator_color = indicator_color
        self._visible_tools = visible_tools
        self._glyphs = glyphs
        self.index = index
        self._selected = "mcp-header-selected" in classes
        content = _render_server_header(
            server,
            indicator_glyph,
            indicator_color,
            visible_tools,
            glyphs,
            selected=self._selected,
        )
        super().__init__(content, classes=classes)

    def set_selected(self, selected: bool) -> None:
        """Apply or remove the selected-row styling and re-render the label.

        Re-renders content on selection change so `dim` secondary spans
        are suppressed on the `$primary` selection background — the same
        approach `MCPToolItem` uses via `_desc_style`.
        """
        if self._selected == selected:
            return
        self._selected = selected
        if selected:
            self.add_class("mcp-header-selected")
        else:
            self.remove_class("mcp-header-selected")
        self.update(
            _render_server_header(
                self._server,
                self._indicator_glyph,
                self._indicator_color,
                self._visible_tools,
                self._glyphs,
                selected=selected,
            )
        )

    def on_click(self, event: Click) -> None:
        """Handle click — select the header row via parent screen.

        Headers are not expandable; clicking only moves the cursor.

        Args:
            event: The click event.
        """
        event.stop()
        screen = self.screen
        if isinstance(screen, MCPViewerScreen):
            screen._move_to(self.index)


class MCPViewerScreen(ModalScreen[None]):
    """Modal viewer for active MCP servers and their tools.

    Displays servers grouped by name with transport type and tool count.
    Navigate with arrow keys, Enter to expand/collapse tool descriptions,
    Escape to close.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False, priority=True),
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("shift+tab", "jump_up", "Up", show=False, priority=True),
        Binding("tab", "jump_down", "Down", show=False, priority=True),
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

    MCPViewerScreen .mcp-server-header:hover {
        background: $surface-lighten-1;
    }

    MCPViewerScreen .mcp-list > .mcp-server-header:first-child {
        margin-top: 0;
    }

    MCPViewerScreen .mcp-header-selected {
        background: $primary;
        color: $text;
        text-style: bold;
    }

    MCPViewerScreen .mcp-header-selected:hover {
        background: $primary-lighten-1;
        color: $text;
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
        # All cursor-navigable rows in render order: server headers + tool
        # items intermixed. `_selected_index` indexes into this list.
        self._row_widgets: list[MCPToolItem | MCPServerHeaderItem] = []
        self._selected_index = 0
        self._query: str = ""

    @property
    def _tool_widgets(self) -> list[MCPToolItem]:
        """Tool rows only — excludes server headers.

        Convenience view used by `Ctrl+E` toggle-all and by tests that
        only care about tool-level state. The authoritative storage is
        `_row_widgets`.
        """
        return [w for w in self._row_widgets if isinstance(w, MCPToolItem)]

    def refresh_server_info(self, server_info: list[MCPServerInfo]) -> None:
        """Replace the displayed server list; typically after server startup.

        Rebuilds the modal body in place so a user who opened `/mcp` before
        tools finished loading sees them appear without closing/reopening.

        The active filter is cleared on refresh: the connecting placeholder
        suppresses the filter input, so `_query` cannot be non-empty when
        this is called. Resetting it also prevents the programmatic
        `Input(value=...)` mount in `_mount_body` from triggering a
        redundant `Input.Changed` repopulation.
        """
        self._server_info = server_info
        self._connecting = False
        self._query = ""
        body = self.query_one(Vertical)
        body.remove_children()
        self._row_widgets = []
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
        self._row_widgets = []
        self._selected_index = 0
        self._populate_scroll(scroll, self._query)
        self._selected_index = min(
            self._selected_index, max(0, len(self._row_widgets) - 1)
        )

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
            f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate"
            f" {glyphs.bullet} Enter expand"
            f" {glyphs.bullet} Ctrl+E expand all"
            f" {glyphs.bullet} type to filter"
            f" {glyphs.bullet} Esc close"
        )
        container.mount(Static(help_text, classes="mcp-viewer-help"))

    def _populate_scroll(self, scroll: VerticalScroll, query: str) -> None:
        """Mount filtered server headers + tool items into `scroll`.

        Empty `query` shows everything; otherwise multi-token AND matching
        on server names and tool names only — descriptions, parameter
        names, and transport are not in the haystack (see
        `_visible_tools_for`).
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

        for server in self._server_info:
            visible_tools = _visible_tools_for(server, tokens)
            if visible_tools is None:
                # Server filtered out entirely.
                continue

            indicator_color = _status_color(server.status, colors)
            indicator_glyph = _status_glyph(server.status, glyphs)
            header_classes = "mcp-server-header"
            if flat_index == 0:
                header_classes += " mcp-header-selected"
            header = MCPServerHeaderItem(
                server=server,
                indicator_glyph=indicator_glyph,
                indicator_color=indicator_color,
                visible_tools=visible_tools,
                glyphs=glyphs,
                index=flat_index,
                classes=header_classes,
            )
            self._row_widgets.append(header)
            scroll.mount(header)
            flat_index += 1

            for tool in visible_tools:
                classes = "mcp-tool-item"
                widget = MCPToolItem(
                    name=tool.name,
                    description=tool.description,
                    index=flat_index,
                    classes=classes,
                    input_schema=tool.input_schema,
                )
                self._row_widgets.append(widget)
                scroll.mount(widget)
                flat_index += 1

        if not self._row_widgets:
            msg = "No matching tools." if tokens else "No tools available."
            scroll.mount(Static(msg, classes="mcp-empty"))

    def _move_to(self, index: int) -> None:
        """Move selection to the given row index.

        Args:
            index: Target row index inside `_row_widgets` (header or tool).
        """
        count = len(self._row_widgets)
        if not count:
            return
        if not (0 <= index < count):
            # Stale index from a widget that survived a filter rebuild.
            return
        old = self._selected_index
        if not (0 <= old < count):
            old = 0
        self._selected_index = index

        if old != index:
            self._row_widgets[old].set_selected(False)
            self._row_widgets[index].set_selected(True)
            # Caller (action) is responsible for any viewport pin — different
            # navigation directions want different anchors (top for down,
            # bottom for up).

    def _move_selection(self, delta: int) -> None:
        """Move selection by delta row positions, clamped at the list ends.

        No wrap-around — pressing `Down` past the last row stays put rather
        than jumping to the first. Walks every row (headers + tools).

        Args:
            delta: Number of row positions to move.
        """
        if not self._row_widgets:
            return
        target = self._selected_index + delta
        if 0 <= target < len(self._row_widgets):
            self._move_to(target)

    def _next_tool_row(self, start: int, step: int) -> int | None:
        """Return the index of the next `MCPToolItem` row in `step` direction.

        Used by `Tab` / `Shift+Tab` to skip server-header rows during
        cross-tool navigation. Returns `None` when there is no tool row in
        the requested direction.

        Args:
            start: Index to start searching from (exclusive).
            step: `+1` (forward) or `-1` (backward).
        """
        idx = start + step
        while 0 <= idx < len(self._row_widgets):
            if isinstance(self._row_widgets[idx], MCPToolItem):
                return idx
            idx += step
        return None

    def _scroll_widget_bottom_to_view(
        self, widget: MCPToolItem | MCPServerHeaderItem
    ) -> None:
        """Scroll so `widget.region.bottom` aligns with the viewport bottom.

        Used when jumping upward to the previous row: lands the user at
        the bottom of that row so the next `Up` press immediately
        line-scrolls upward through its content rather than jumping again.
        """
        scroll = self.query_one(".mcp-list", VerticalScroll)
        delta = (widget.region.y + widget.region.height) - (
            scroll.region.y + scroll.region.height
        )
        if delta:
            scroll.scroll_relative(y=delta, animate=False)

    def action_move_up(self) -> None:
        """Smart up: scroll one row inside a tall expanded row, else jump.

        If the selected row's top edge is already inside the viewport, jump
        to the previous row (header or tool) and pin its **bottom** to the
        viewport so the next `Up` resumes line-stepping through that row.
        Otherwise scroll the viewport up by one row. `Tab` / `Shift+Tab`
        skip the smart check AND skip header rows (see `action_jump_up`).
        """
        if not self._row_widgets:
            return
        scroll = self.query_one(".mcp-list", VerticalScroll)
        selected = self._row_widgets[self._selected_index]
        if selected.region.y >= scroll.region.y:
            old = self._selected_index
            self._move_selection(-1)
            if self._selected_index != old:
                self._scroll_widget_bottom_to_view(
                    self._row_widgets[self._selected_index]
                )
        else:
            scroll.scroll_relative(y=-1, animate=False)

    def action_move_down(self) -> None:
        """Smart down: scroll one row inside a tall expanded row, else jump.

        If the selected row's bottom edge is already inside the viewport,
        jump to the next row (header or tool) and pin its top to the
        viewport. Otherwise scroll the viewport down by one row. `Tab` /
        `Shift+Tab` skip the smart check AND skip header rows (see
        `action_jump_down`).
        """
        if not self._row_widgets:
            return
        scroll = self.query_one(".mcp-list", VerticalScroll)
        selected = self._row_widgets[self._selected_index]
        selected_bottom = selected.region.y + selected.region.height
        viewport_bottom = scroll.region.y + scroll.region.height
        if selected_bottom <= viewport_bottom:
            old = self._selected_index
            self._move_selection(1)
            if self._selected_index != old:
                self._row_widgets[self._selected_index].scroll_visible(top=True)
        else:
            scroll.scroll_relative(y=1, animate=False)

    def action_jump_up(self) -> None:
        """Jump to the previous tool (Shift+Tab); skips headers; pin bottom."""
        target = self._next_tool_row(self._selected_index, -1)
        if target is None:
            return
        self._move_to(target)
        self._scroll_widget_bottom_to_view(self._row_widgets[target])

    def action_jump_down(self) -> None:
        """Jump to the next tool (Tab); skips headers; pin top."""
        target = self._next_tool_row(self._selected_index, +1)
        if target is None:
            return
        self._move_to(target)
        self._row_widgets[target].scroll_visible(top=True)

    def action_toggle_expand(self) -> None:
        """Toggle expand/collapse on the selected tool (no-op for headers)."""
        if not self._row_widgets:
            return
        row = self._row_widgets[self._selected_index]
        if isinstance(row, MCPToolItem):
            row.toggle_expand()

    def action_toggle_all(self) -> None:
        """Expand or collapse every visible tool at once.

        If any visible tool is collapsed, expand all; otherwise collapse all.
        Operates on tool rows only — server headers are not expandable.
        Hidden tools (filtered out) keep their state.
        """
        tools = self._tool_widgets
        if not tools:
            return
        any_collapsed = any(not w._expanded for w in tools)
        for widget in tools:
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
