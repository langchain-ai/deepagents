"""Read-only MCP server and tool viewer modal."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, assert_never

from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.events import (
    Click,  # noqa: TC002 - needed at runtime for Textual event dispatch
)
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from deepagents_code.clipboard import copy_text_to_clipboard

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from textual.app import ComposeResult

    from deepagents_code.mcp_tools import MCPServerInfo, MCPServerStatus, MCPToolInfo

from deepagents_code import theme
from deepagents_code.config import Glyphs, get_glyphs, is_ascii_mode
from deepagents_code.unicode_security import sanitize_control_chars

logger = logging.getLogger(__name__)

MCP_VIEWER_RECONNECT_REQUEST = "\x00__mcp_reconnect__"
"""Sentinel returned by `MCPViewerScreen.dismiss` to request a reconnect.

The null-byte prefix makes the value un-collidable with any valid MCP
server name returned by `MCPServerInfo.name`, so callers can branch on
this exact string without weakening the existing server-name dispatch.
"""

MCP_RECONNECT_KEY = "ctrl+r"
"""Textual `Binding` key for the in-viewer reconnect action.

Kept as a module constant so the footer hint and the help-text rendered
in server headers stay in sync with the bound chord.
"""

MCP_RECONNECT_KEY_LABEL = "Ctrl+R"
"""Display label for `MCP_RECONNECT_KEY`.

Shown in the footer hint chip and inline header prompts so the user
sees the same chord text the binding will fire on.
"""


def _status_glyph(status: MCPServerStatus, glyphs: Glyphs) -> str:
    """Return the glyph character for a server `status`.

    Maps onto the existing `Glyphs` set so ASCII fallback is automatic
    (`✓ ⚠ ✗` -> `[OK] [!] [X]`). No new glyph definitions needed.

    Args:
        status: One of `ok` / `unauthenticated` / `awaiting_reconnect` /
            `error` / `disabled`.
        glyphs: Active `Glyphs` table (Unicode or ASCII).

    Returns:
        The unicode or ASCII glyph character matching `status`.
    """
    if status == "ok":
        return glyphs.checkmark
    if status == "unauthenticated":
        return glyphs.warning
    if status == "awaiting_reconnect":
        return glyphs.circle_empty
    if status == "disabled":
        return glyphs.pause
    if status == "error":
        return glyphs.error
    assert_never(status)


def _status_color(status: MCPServerStatus, colors: theme.ThemeColors) -> str:
    """Map a server `status` onto a semantic theme color.

    `ok` -> success (green); `unauthenticated` -> warning (yellow);
    `error` -> error (red); `disabled` -> muted. Returning the theme's
    hex string lets callers pass the value to `Content.styled()` or
    `Content.assemble()` so a theme switch recolors the indicator
    without code changes.

    Args:
        status: One of `ok` / `unauthenticated` / `awaiting_reconnect` /
            `error` / `disabled`.
        colors: Active theme palette (typically from `theme.get_theme_colors`).

    Returns:
        Hex color string from the active theme.
    """
    if status == "ok":
        return colors.success
    if status == "unauthenticated":
        return colors.warning
    if status == "awaiting_reconnect":
        return colors.warning
    if status == "disabled":
        return colors.muted
    if status == "error":
        return colors.error
    assert_never(status)


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


_INLINE_TEXT_LIMIT = 200
"""Max characters for untrusted text rendered inline in a server header.

Bounds a hostile or buggy server's error/name so it cannot overflow the
single-line header; full text is available in the error-detail modal.
"""


def _sort_servers_for_display(
    server_info: list[MCPServerInfo],
) -> list[MCPServerInfo]:
    """Return `server_info` with attention-needed servers floated to the top.

    Stable sort so the user's config order is preserved within each group.
    Surfacing unauthenticated and awaiting-reconnect servers first makes
    the next action visible without scrolling on configs with many `ok`
    servers.
    """
    priority = {"unauthenticated": 0, "awaiting_reconnect": 1}
    return sorted(server_info, key=lambda s: priority.get(s.status, 2))


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
        """Re-render with correct truncation once width is known.

        Defers via `call_after_refresh` so the first paint happens AFTER
        the layout pass. At `on_mount` time `self.size.width` is still
        0, which short-circuits `_format_collapsed`'s `avail > 0` guard
        and emits the full description un-truncated for one frame. The
        subsequent resize re-render then snaps an ellipsis in,
        producing a visible overflow flicker on every mount (initial
        open, filter rebuild, F2 toggle rebuild).
        """
        self.call_after_refresh(self._rerender)

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
    `sanitize_control_chars` because MCP servers can return arbitrary error
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
    if server.status == "unauthenticated":
        login_hint = " — Enter to log in"
        return Content.assemble(
            (f"{indicator_glyph} ", indicator_color),
            (server.name, "bold"),
            (f" {server.transport}", dim_style),
            (f" {glyphs.bullet} {server.status}", indicator_color),
            (login_hint, dim_style),
        )
    if server.status == "awaiting_reconnect":
        return Content.assemble(
            (f"{indicator_glyph} ", indicator_color),
            (server.name, "bold"),
            (f" {server.transport}", dim_style),
            (f" {glyphs.bullet} ready to load", indicator_color),
            (f" — {MCP_RECONNECT_KEY_LABEL} to load tools", dim_style),
        )
    if server.status == "error":
        return Content.assemble(
            (f"{indicator_glyph} ", indicator_color),
            (server.name, "bold"),
            (f" {server.transport}", dim_style),
            (f" {glyphs.bullet} {server.status}", indicator_color),
            (" — Enter for details", dim_style),
        )
    if server.status == "disabled":
        error_text = sanitize_control_chars(
            server.error or "", max_length=_INLINE_TEXT_LIMIT
        )
        return Content.assemble(
            (f"{indicator_glyph} ", indicator_color),
            (server.name, "bold"),
            (f" {server.transport}", dim_style),
            (f" {glyphs.bullet} {server.status}", indicator_color),
            (f" — {error_text}", dim_style) if error_text else "",
        )
    assert_never(server.status)


class MCPServerErrorScreen(ModalScreen[None]):
    """Read-only modal for a failed MCP server's error details."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("c", "copy_error", "Copy", show=False, priority=True),
        Binding("escape", "cancel", "Close", show=False, priority=True),
    ]

    CSS = """
    MCPServerErrorScreen {
        align: center middle;
    }

    MCPServerErrorScreen > Vertical {
        width: 100;
        max-width: 90%;
        height: 80%;
        background: $surface;
        border: solid $error;
        padding: 1 2;
    }

    MCPServerErrorScreen .mcp-error-title {
        text-style: bold;
        color: $error;
        text-align: center;
        margin-bottom: 1;
    }

    MCPServerErrorScreen .mcp-error-body {
        height: 1fr;
        background: $background;
        scrollbar-gutter: stable;
        padding: 0 1;
    }

    MCPServerErrorScreen .mcp-error-text {
        color: $text;
    }

    MCPServerErrorScreen .mcp-error-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }
    """

    def __init__(self, server: MCPServerInfo) -> None:
        """Initialize the error-detail modal.

        Args:
            server: Failed MCP server whose error text should be displayed.
        """
        super().__init__()
        self._server = server
        self._error = sanitize_control_chars(
            server.error or "No error details were reported.",
            keep_newlines=True,
            collapse_whitespace=False,
        )

    def compose(self) -> ComposeResult:
        """Compose the modal layout.

        Yields:
            Modal shell with title, scrollable error text, and help footer.
        """
        glyphs = get_glyphs()
        yield Vertical(
            Static(
                Content.from_markup(
                    "MCP Server Error: $server",
                    server=sanitize_control_chars(
                        self._server.name, max_length=_INLINE_TEXT_LIMIT
                    ),
                ),
                classes="mcp-error-title",
            ),
            VerticalScroll(
                Static(
                    Content.from_markup("$error", error=self._error),
                    classes="mcp-error-text",
                ),
                classes="mcp-error-body",
            ),
            Static(
                f"c copy error {glyphs.bullet} Esc close",
                classes="mcp-error-help",
            ),
        )

    def action_copy_error(self) -> None:
        """Copy the server error details to the clipboard."""
        success, error = copy_text_to_clipboard(self.app, self._error)
        if success:
            self.app.notify(
                "MCP error copied",
                severity="information",
                timeout=2,
                markup=False,
            )
            return
        suffix = f": {error}" if error else ""
        self.app.notify(
            f"Failed to copy MCP error{suffix}",
            severity="warning",
            timeout=3,
            markup=False,
        )

    def action_cancel(self) -> None:
        """Close the error details modal."""
        self.dismiss(None)


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

    @property
    def server(self) -> MCPServerInfo:
        """Server metadata this header row is rendering."""
        return self._server

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

    def refresh_from_server(
        self,
        server: MCPServerInfo,
        indicator_glyph: str,
        indicator_color: str,
        visible_tools: tuple[MCPToolInfo, ...],
        glyphs: Glyphs,
    ) -> None:
        """Replace the underlying server data and re-render in place.

        Used by `apply_server_disable_toggle` so an F2 toggle updates this
        header without tearing down and re-mounting the widget. Preserves
        the row's selected state so the cursor stays put visually.

        Args:
            server: Updated server metadata.
            indicator_glyph: New status glyph character.
            indicator_color: New status hex color.
            visible_tools: New filtered tool tuple (drives the count label).
            glyphs: Active glyph table.
        """
        self._server = server
        self._indicator_glyph = indicator_glyph
        self._indicator_color = indicator_color
        self._visible_tools = visible_tools
        self._glyphs = glyphs
        self.update(
            _render_server_header(
                server,
                indicator_glyph,
                indicator_color,
                visible_tools,
                glyphs,
                selected=self._selected,
            )
        )

    def on_click(self, event: Click) -> None:
        """Handle click — select the header, or start login on unauth re-click.

        Headers are not expandable. Clicking once moves the cursor;
        clicking the already-selected header either starts login for
        an `unauthenticated` server or opens details for an `error`
        server.

        Args:
            event: The click event.
        """
        event.stop()
        screen = self.screen
        if not isinstance(screen, MCPViewerScreen):
            return
        if self._selected and self._server.needs_attention():
            screen.dismiss(self._server.name)
            return
        if self._selected and self._server.status == "error":
            screen.show_server_error(self._server)
            return
        screen._move_to(self.index)


class MCPViewerScreen(ModalScreen[str | None]):
    """Modal viewer for active MCP servers and their tools.

    Displays servers grouped by name with transport type and tool count.
    Navigate with arrow keys, Enter to expand/collapse tool descriptions,
    start in-app OAuth login for an unauthenticated server, or inspect a
    failed server. Ctrl+R requests a reconnect, F2 on a server header
    toggles its disabled state, and Escape closes the modal.

    Dismisses with `None` when closed without action, the server name to
    drive an in-TUI OAuth login when the user activates an
    `unauthenticated` server header, or `MCP_VIEWER_RECONNECT_REQUEST`
    for a reconnect. The disable/enable toggle (`F2`) is handled in-place
    via the `on_toggle_disable` callback so the screen never tears down
    — see the constructor.
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
        Binding(MCP_RECONNECT_KEY, "reconnect", "Reconnect", show=False, priority=True),
        Binding("f2", "toggle_disable", "Toggle disable", show=False, priority=True),
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
        pending_reconnect: bool = False,
        on_toggle_disable: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        """Initialize the MCP viewer screen.

        Args:
            server_info: List of MCP server metadata to display.
            connecting: When `True` and `server_info` is empty, show a
                "connecting..." placeholder instead of the "no servers"
                message; the screen refreshes when `refresh_server_info`
                is called after the server startup completes.
            pending_reconnect: `True` when a deferred MCP login is queued
                and a restart will pick it up. Surfaces the `Ctrl+R`
                reconnect hint in the footer; the keybind itself is a
                no-op when this is `False`.
            on_toggle_disable: Async callback invoked with the selected
                server's name when the user presses `F2` on a header row.
                The callback persists the new disabled state and is
                expected to call `refresh_server_info` on this screen so
                the user sees the updated status without a screen swap.
                When `None`, `F2` is a no-op.
        """
        super().__init__()
        self._server_info = server_info
        self._connecting = connecting
        self._pending_reconnect = pending_reconnect
        self._on_toggle_disable = on_toggle_disable
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

    async def refresh_server_info(
        self,
        server_info: list[MCPServerInfo],
        *,
        pending_reconnect: bool | None = None,
        select_server: str | None = None,
    ) -> None:
        """Replace the displayed server list; typically after server startup.

        Rebuilds the modal body in place so a user who opened `/mcp` before
        tools finished loading sees them appear without closing/reopening.
        Also used by the in-place disable toggle so the cursor lands back
        on the same server header after F2.

        The active filter is cleared on refresh: the connecting placeholder
        suppresses the filter input, so `_query` cannot be non-empty when
        this is called. Resetting it also prevents the programmatic
        `Input(value=...)` mount in `_mount_body` from triggering a
        redundant `Input.Changed` repopulation.

        This is async because `body.remove_children()` must complete before
        `_mount_body` re-inserts an `Input(id="mcp-filter")` — Textual
        defers child removal, so a non-awaited remove leaves the old
        widget attached and the mount raises `DuplicateIds`.

        Args:
            server_info: Refreshed server metadata.
            pending_reconnect: When provided, updates the footer's
                reconnect hint. `None` preserves the existing value.
            select_server: When provided, after the rebuild, move the
                cursor to the header row whose `server.name` matches.
                Unmatched names are silently ignored (the rebuild keeps
                the default index-0 selection).
        """
        self._server_info = server_info
        self._connecting = False
        self._query = ""
        if pending_reconnect is not None:
            self._pending_reconnect = pending_reconnect
        body = self.query_one(Vertical)
        await body.remove_children()
        self._row_widgets = []
        self._selected_index = 0
        self._mount_body(body)
        if select_server is not None:
            for idx, widget in enumerate(self._row_widgets):
                if (
                    isinstance(widget, MCPServerHeaderItem)
                    and widget.server.name == select_server
                ):
                    self._move_to(idx)
                    self._reveal_selection(widget, direction=1)
                    break
        self._focus_filter_input()

    async def apply_server_disable_toggle(
        self,
        server_info: list[MCPServerInfo],
        *,
        toggled_server: str,
        pending_reconnect: bool | None = None,
    ) -> None:
        """Patch a single server's row in place after an F2 toggle.

        Surgically updates only the affected server's header and tool
        rows so unchanged widgets keep their identity — no full
        `body.remove_children()` + remount, which would re-create every
        `MCPToolItem` and reintroduce the `on_mount` truncation flicker
        across the entire list.

        Falls back to `refresh_server_info` when the toggled server
        cannot be patched in place: server missing from the new info
        list, no existing header widget (e.g., the active filter
        currently hides it), or the new state would filter the server
        out entirely.

        Args:
            server_info: Refreshed server metadata (full list).
            toggled_server: Name of the server whose disabled state just
                changed; identifies which row to patch.
            pending_reconnect: When provided, updates the footer's
                reconnect hint. `None` preserves the existing value.
        """
        self._server_info = server_info
        if pending_reconnect is not None:
            self._pending_reconnect = pending_reconnect

        new_server = next((s for s in server_info if s.name == toggled_server), None)
        header_idx = next(
            (
                i
                for i, w in enumerate(self._row_widgets)
                if isinstance(w, MCPServerHeaderItem)
                and w.server.name == toggled_server
            ),
            None,
        )
        tokens = [tok for tok in self._query.lower().split() if tok]
        visible_tools = (
            _visible_tools_for(new_server, tokens) if new_server is not None else None
        )

        if new_server is None or header_idx is None or visible_tools is None:
            logger.debug(
                "apply_server_disable_toggle fallback for %r: "
                "new_server=%s header_idx=%s visible_tools=%s",
                toggled_server,
                new_server is not None,
                header_idx,
                visible_tools is not None,
            )
            await self.refresh_server_info(
                server_info,
                pending_reconnect=pending_reconnect,
                select_server=toggled_server,
            )
            return

        header = self._row_widgets[header_idx]
        if not isinstance(header, MCPServerHeaderItem):
            # The lookup above filters by isinstance, so this branch
            # should be unreachable. Log loudly rather than silently
            # returning so a future invariant break is visible.
            logger.warning(
                "apply_server_disable_toggle: expected header at index %d, got %r",
                header_idx,
                type(header).__name__,
            )
            return
        next_header_idx = next(
            (
                i
                for i in range(header_idx + 1, len(self._row_widgets))
                if isinstance(self._row_widgets[i], MCPServerHeaderItem)
            ),
            len(self._row_widgets),
        )

        # `remove_children(to_remove)` removes the listed widgets
        # atomically in one refresh; awaiting `widget.remove()` per
        # row would yield to Textual between each, animating the
        # tool list shrinking one entry at a time.
        scroll = self.query_one(".mcp-list", VerticalScroll)
        to_remove = self._row_widgets[header_idx + 1 : next_header_idx]
        if to_remove:
            await scroll.remove_children(to_remove)
        del self._row_widgets[header_idx + 1 : next_header_idx]

        colors = theme.get_theme_colors(self)
        glyphs = get_glyphs()
        header.refresh_from_server(
            new_server,
            _status_glyph(new_server.status, glyphs),
            _status_color(new_server.status, colors),
            visible_tools,
            glyphs,
        )

        if visible_tools:
            new_widgets: list[MCPToolItem] = [
                MCPToolItem(
                    name=tool.name,
                    description=tool.description,
                    index=0,  # renumbered below
                    classes="mcp-tool-item",
                    input_schema=tool.input_schema,
                )
                for tool in visible_tools
            ]
            await scroll.mount(*new_widgets, after=header)
            self._row_widgets[header_idx + 1 : header_idx + 1] = new_widgets

        # `MCPToolItem.on_click` calls `screen._move_to(self.index)`,
        # so every row's stored index must match its position after
        # the splice — otherwise clicks land on the wrong row.
        for idx, widget in enumerate(self._row_widgets):
            widget.index = idx
        if self._selected_index >= len(self._row_widgets):
            self._selected_index = max(0, len(self._row_widgets) - 1)

        # `_build_help_text` is cheap and reads `_pending_reconnect`,
        # so re-render the footer whenever the caller supplied a new
        # value — saves comparing against the prior state.
        if pending_reconnect is not None:
            help_static = self.query_one(".mcp-viewer-help", Static)
            help_static.update(self._build_help_text(glyphs))

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

        container.mount(
            Static(self._build_help_text(glyphs), classes="mcp-viewer-help")
        )

    def _focus_filter_input(self) -> None:
        """Refocus the filter `Input` after an in-place body rebuild.

        `refresh_server_info` clears the body via `remove_children`, which
        blurs the screen (Textual resets focus to `None` when the focused
        widget is pruned). The newly mounted filter `Input` is not
        auto-focused on a re-mount — Textual auto-focuses only on the first
        mount — so a viewer opened while the server is still connecting
        would leave the rebuilt input unfocused once tools load, and
        keystrokes would never reach it. Restore focus explicitly here,
        deferred via `call_after_refresh` because `_mount_body` mounts
        without awaiting.

        The `Input` exists only when there are servers to filter (see
        `_mount_body`); when the list is empty there is nothing to focus, so
        return early. Gating on `_server_info` rather than swallowing a
        missing-widget error keeps a genuinely-absent input (id drift, a
        failed mount) visible instead of silently re-introducing the
        keystroke-swallow this method exists to prevent.
        """

        def _focus() -> None:
            if not self._server_info:
                return
            self.query_one("#mcp-filter", Input).focus()

        self.call_after_refresh(_focus)

    def _build_help_text(self, glyphs: Glyphs) -> str:
        """Compose the help-footer string from the current `_pending_reconnect`.

        Single source of truth so `_mount_body` (initial) and
        `apply_server_disable_toggle` (incremental) stay in sync — F2
        flips the reconnect-pending state, and the footer must update
        without a full re-mount.

        Returns:
            The rendered help line for the modal footer.
        """
        help_parts = [
            f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate",
            "Enter expand/login/details",
            "F2 disable/enable",
            "Ctrl+E expand all",
        ]
        if self._pending_reconnect:
            help_parts.append(f"{MCP_RECONNECT_KEY_LABEL} reconnect")
        help_parts.extend(["type to filter", "Esc close"])
        return f" {glyphs.bullet} ".join(help_parts)

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
                "Loading MCP tools..."
                if self._connecting
                else ("No MCP servers configured.\nUse `--mcp-config` to load servers.")
            )
            scroll.mount(Static(placeholder, classes="mcp-empty"))
            return

        tokens = [tok for tok in query.lower().split() if tok]
        colors = theme.get_theme_colors(self)
        flat_index = 0

        for server in _sort_servers_for_display(self._server_info):
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
        """Move selection by delta row positions within the list bounds.

        Walks every row (headers + tools). Navigation actions handle wrapping
        before calling this helper at a list boundary.

        Args:
            delta: Number of row positions to move.
        """
        if not self._row_widgets:
            return
        target = self._selected_index + delta
        if 0 <= target < len(self._row_widgets):
            self._move_to(target)

    def _next_server_header(self, start: int, step: int) -> int | None:
        """Return the next server-header index in the requested direction.

        Args:
            start: Index to start searching from (exclusive).
            step: `+1` (forward) or `-1` (backward).

        Returns:
            The index of the nearest `MCPServerHeaderItem` in that direction,
            or `None` when no server header exists there.
        """
        index = start + step
        while 0 <= index < len(self._row_widgets):
            if isinstance(self._row_widgets[index], MCPServerHeaderItem):
                return index
            index += step
        return None

    def _scroll_widget_bottom_to_view(
        self, widget: MCPToolItem | MCPServerHeaderItem
    ) -> None:
        """Scroll so `widget.region.bottom` aligns with the viewport bottom.

        Used when jumping upward into a row taller than the viewport: lands
        the user at the bottom of that row so the next `Up` press immediately
        line-scrolls upward through its content rather than jumping again.
        """
        scroll = self.query_one(".mcp-list", VerticalScroll)
        delta = (widget.region.y + widget.region.height) - (
            scroll.region.y + scroll.region.height
        )
        if delta:
            scroll.scroll_relative(y=delta, animate=False)

    def _reveal_selection(
        self,
        widget: MCPToolItem | MCPServerHeaderItem,
        *,
        direction: int,
    ) -> None:
        """Bring `widget` into view after a selection change.

        Only force-anchors rows taller than the viewport — these need a
        deliberate edge alignment so subsequent arrow presses can line-scroll
        through the row's body. For normal rows, defers to `scroll_visible`,
        which is a no-op when the row is already fully visible. Matches
        `/model` switcher behavior where short, in-view rows don't tug the
        viewport on every keypress.

        Args:
            widget: The newly selected row.
            direction: `+1` when moving down (anchor top for tall rows),
                `-1` when moving up (anchor bottom for tall rows).
        """
        scroll = self.query_one(".mcp-list", VerticalScroll)
        if widget.region.height > scroll.region.height:
            if direction > 0:
                widget.scroll_visible(top=True)
            else:
                self._scroll_widget_bottom_to_view(widget)
        else:
            widget.scroll_visible()

    def action_move_up(self) -> None:
        """Smart up: scroll one row inside a tall expanded row, else jump.

        If the selected row's top edge is already inside the viewport, jump
        to the previous row (header or tool), wrapping to the final row from
        the first. For rows taller than the viewport, pin the new selection's
        **bottom** to the viewport so the next `Up` resumes line-stepping
        through that row; otherwise just ensure the row is visible. `Tab` /
        `Shift+Tab` jump between server headers (see `action_jump_up`).
        """
        if not self._row_widgets:
            return
        scroll = self.query_one(".mcp-list", VerticalScroll)
        selected = self._row_widgets[self._selected_index]
        if selected.region.y >= scroll.region.y:
            old = self._selected_index
            if old == 0:
                self._move_to(len(self._row_widgets) - 1)
            else:
                self._move_selection(-1)
            if self._selected_index != old:
                self._reveal_selection(
                    self._row_widgets[self._selected_index], direction=-1
                )
        else:
            scroll.scroll_relative(y=-1, animate=False)

    def action_move_down(self) -> None:
        """Smart down: scroll one row inside a tall expanded row, else jump.

        If the selected row's bottom edge is already inside the viewport,
        jump to the next row (header or tool), wrapping to the first row from
        the final one. For rows taller than the viewport, pin the new
        selection's top to the viewport; otherwise just ensure the row is
        visible. `Tab` / `Shift+Tab` jump between server headers (see
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
            if old == len(self._row_widgets) - 1:
                self._move_to(0)
            else:
                self._move_selection(1)
            if self._selected_index != old:
                self._reveal_selection(
                    self._row_widgets[self._selected_index], direction=1
                )
        else:
            scroll.scroll_relative(y=1, animate=False)

    def action_jump_up(self) -> None:
        """Jump backward to the nearest server header (Shift+Tab), wrapping.

        From a tool row this lands on the current server's own header; from a
        header it moves to the previous server. Wraps to the final header from
        the top.
        """
        target = self._next_server_header(self._selected_index, -1)
        if target is None:
            target = self._next_server_header(len(self._row_widgets), -1)
        if target is None or target == self._selected_index:
            return
        self._move_to(target)
        self._reveal_selection(self._row_widgets[target], direction=-1)

    def action_jump_down(self) -> None:
        """Jump to the next server (Tab), wrapping at the end."""
        target = self._next_server_header(self._selected_index, +1)
        if target is None:
            target = self._next_server_header(-1, +1)
        if target is None or target == self._selected_index:
            return
        self._move_to(target)
        self._reveal_selection(self._row_widgets[target], direction=1)

    def show_server_error(self, server: MCPServerInfo) -> None:
        """Open the read-only error detail modal for `server`.

        Args:
            server: Failed MCP server to inspect.
        """
        self.app.push_screen(MCPServerErrorScreen(server))

    def action_toggle_expand(self) -> None:
        """Toggle expand on a tool row, log in, or show error details.

        Tool rows expand/collapse as before; activating a header row for
        a server in `unauthenticated` state dismisses the viewer with the
        server name so the app can drive in-TUI OAuth login. Activating an
        `error` header opens a read-only detail modal. Headers for other
        states (ok, awaiting reconnect, disabled) remain no-ops.
        """
        if not self._row_widgets:
            return
        row = self._row_widgets[self._selected_index]
        if isinstance(row, MCPToolItem):
            row.toggle_expand()
            # The new height isn't reflected until after the next layout
            # pass, so defer the visibility scroll. Without this, expanding
            # a row near the viewport bottom leaves its new body off-screen.
            self.call_after_refresh(row.scroll_visible)
            return
        server = row.server
        if server.needs_attention():
            self.dismiss(server.name)
            return
        if server.status == "error":
            self.show_server_error(server)

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
        """Scroll up by one page and snap selection to the topmost visible row.

        Without the selection snap, `_selected_index` would still point at
        the now-offscreen row, and a subsequent `Up`/`Down` press would
        yank the viewport back to it (see `action_move_up` / `_move_down`,
        which scroll the offscreen selection back into view).
        """
        if not self._row_widgets:
            return
        scroll = self.query_one(".mcp-list", VerticalScroll)
        scroll.scroll_page_up()
        self.call_after_refresh(self._snap_selection_to_topmost_visible)

    def action_page_down(self) -> None:
        """Scroll down by one page and snap selection to the bottommost visible row.

        Mirror of `action_page_up`: prevents a subsequent arrow key from
        scrolling the viewport back to a now-offscreen selection.
        """
        if not self._row_widgets:
            return
        scroll = self.query_one(".mcp-list", VerticalScroll)
        scroll.scroll_page_down()
        self.call_after_refresh(self._snap_selection_to_bottommost_visible)

    def _snap_selection_to_topmost_visible(self) -> None:
        """Move selection to the first row whose top is at or below the viewport top."""
        if not self._row_widgets:
            return
        scroll = self.query_one(".mcp-list", VerticalScroll)
        top = scroll.region.y
        for idx, widget in enumerate(self._row_widgets):
            if widget.region.y >= top:
                self._move_to(idx)
                return

    def _snap_selection_to_bottommost_visible(self) -> None:
        """Move selection to the last row whose bottom fits inside the viewport."""
        if not self._row_widgets:
            return
        scroll = self.query_one(".mcp-list", VerticalScroll)
        bottom = scroll.region.y + scroll.region.height
        target: int | None = None
        for idx, widget in enumerate(self._row_widgets):
            if widget.region.y + widget.region.height <= bottom:
                target = idx
        if target is not None:
            self._move_to(target)

    def action_cancel(self) -> None:
        """Close the viewer without selecting a server to log into."""
        self.dismiss(None)

    def action_reconnect(self) -> None:
        """Dismiss with the reconnect sentinel when a login is pending.

        Bindings are static, so the keybind is always bound; this guard
        is what makes it a no-op when nothing is queued.
        """
        if not self._pending_reconnect:
            return
        self.dismiss(MCP_VIEWER_RECONNECT_REQUEST)

    def action_toggle_disable(self) -> None:
        """Hand off a toggle-disable request to the app without dismissing.

        Only fires when a server header is selected — pressing F2 on a
        tool row is a no-op. The app's callback persists the new state
        and is expected to call `refresh_server_info(..., select_server=)`
        on this screen, so the user sees the new status without the
        screen tearing down (which would flicker and reset selection).
        """
        if not self._row_widgets:
            return
        row = self._row_widgets[self._selected_index]
        if isinstance(row, MCPToolItem):
            return
        if self._on_toggle_disable is None:
            return
        self.app.call_later(self._on_toggle_disable, row.server.name)
