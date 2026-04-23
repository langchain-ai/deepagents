"""Read-only MCP server and tool viewer modal."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.events import (
    Click,  # noqa: TC002 - needed at runtime for Textual event dispatch
)
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from deepagents_cli.mcp_tools import MCPServerInfo

from deepagents_cli import theme
from deepagents_cli.config import get_glyphs, is_ascii_mode


class MCPToolItem(Static):
    """A selectable tool item in the MCP viewer."""

    def __init__(
        self,
        name: str,
        description: str,
        index: int,
        *,
        classes: str = "",
    ) -> None:
        """Initialize a tool item.

        Args:
            name: Tool name.
            description: Full tool description.
            index: Flat index of this tool in the list.
            classes: CSS classes.
        """
        self.tool_name = name
        self.tool_description = description
        self.index = index
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

        Args:
            name: Tool name.
            description: Tool description.

        Returns:
            Styled Content label with full description on next line.
        """
        if description:
            style = self._desc_style()
            template = (
                "  [bold]$name[/bold]\n    [" + style + "]$desc[/]"
                if style
                else "  [bold]$name[/bold]\n    $desc"
            )
            return Content.from_markup(template, name=name, desc=description)
        return Content.from_markup("  [bold]$name[/bold]", name=name)

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
        self._expanded = not self._expanded
        self.styles.height = "auto" if self._expanded else 1
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
        Binding("k", "move_up", "Up", show=False, priority=True),
        Binding("shift+tab", "move_up", "Up", show=False, priority=True),
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("j", "move_down", "Down", show=False, priority=True),
        Binding("tab", "move_down", "Down", show=False, priority=True),
        Binding("enter", "toggle_expand", "Expand", show=False, priority=True),
        Binding("pageup", "page_up", "Page up", show=False, priority=True),
        Binding("pagedown", "page_down", "Page down", show=False, priority=True),
        Binding("escape", "cancel", "Close", show=False, priority=True),
    ]

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
        """Populate `container` with the title, list, and help footer."""
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

        scroll = VerticalScroll(classes="mcp-list")
        container.mount(scroll)

        if not self._server_info:
            placeholder = (
                "Loading MCP tools — server is starting up..."
                if self._connecting
                else ("No MCP servers configured.\nUse `--mcp-config` to load servers.")
            )
            scroll.mount(Static(placeholder, classes="mcp-empty"))
        else:
            flat_index = 0
            for server in self._server_info:
                tool_count = len(server.tools)
                t_label = "tool" if tool_count == 1 else "tools"
                if server.status == "ok":
                    header_markup = (
                        "[bold]$name[/bold]"
                        f" [dim]$transport {glyphs.bullet}"
                        f" {tool_count} {t_label}[/dim]"
                    )
                else:
                    header_markup = (
                        "[bold]$name[/bold]"
                        " [dim]$transport[/dim]"
                        f" [yellow]{glyphs.bullet} $status[/yellow]"
                        " [dim] — $error[/dim]"
                    )
                scroll.mount(
                    Static(
                        Content.from_markup(
                            header_markup,
                            name=server.name,
                            transport=server.transport,
                            status=server.status,
                            error=server.error or "",
                        ),
                        classes="mcp-server-header",
                    )
                )
                for tool in server.tools:
                    classes = "mcp-tool-item"
                    if flat_index == 0:
                        classes += " mcp-tool-selected"
                    widget = MCPToolItem(
                        name=tool.name,
                        description=tool.description,
                        index=flat_index,
                        classes=classes,
                    )
                    self._tool_widgets.append(widget)
                    scroll.mount(widget)
                    flat_index += 1

        help_text = (
            f"{glyphs.arrow_up}/{glyphs.arrow_down} or Tab navigate"
            f" {glyphs.bullet} Enter expand/collapse"
            f" {glyphs.bullet} Esc close"
        )
        container.mount(Static(help_text, classes="mcp-viewer-help"))

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
