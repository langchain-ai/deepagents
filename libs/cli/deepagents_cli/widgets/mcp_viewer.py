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
        if description:
            label = Content.from_markup(
                "  $name [dim]$desc[/dim]", name=name, desc=description
            )
        else:
            label = Content.from_markup("  $name", name=name)
        super().__init__(label, classes=classes)
        self.tool_name = name
        self.tool_description = description
        self.index = index
        self._expanded = False

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
        return Content.from_markup(
            "  $name [dim]$desc[/dim]", name=name, desc=desc_text
        )

    @staticmethod
    def _format_expanded(name: str, description: str) -> Content:
        """Build the expanded (multi-line) label.

        Args:
            name: Tool name.
            description: Tool description.

        Returns:
            Styled Content label with full description on next line.
        """
        if description:
            return Content.from_markup(
                "  [bold]$name[/bold]\n    [dim]$desc[/dim]",
                name=name,
                desc=description,
            )
        return Content.from_markup("  [bold]$name[/bold]", name=name)

    def toggle_expand(self) -> None:
        """Toggle between collapsed and expanded view."""
        self._expanded = not self._expanded
        if self._expanded:
            label = self._format_expanded(self.tool_name, self.tool_description)
            self.styles.height = "auto"
        else:
            label = self._format_collapsed(self.tool_name, self.tool_description)
            self.styles.height = 1
        self.update(label)

    def on_mount(self) -> None:
        """Re-render with correct truncation once width is known."""
        if not self._expanded:
            self.update(self._format_collapsed(self.tool_name, self.tool_description))

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
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("j", "move_down", "Down", show=False, priority=True),
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
        text-style: bold;
    }

    MCPViewerScreen .mcp-tool-selected:hover {
        background: $primary-lighten-1;
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

    def __init__(self, server_info: list[MCPServerInfo]) -> None:
        """Initialize the MCP viewer screen.

        Args:
            server_info: List of MCP server metadata to display.
        """
        super().__init__()
        self._server_info = server_info
        self._tool_widgets: list[MCPToolItem] = []
        self._selected_index = 0

    def compose(self) -> ComposeResult:
        """Compose the screen layout.

        Yields:
            Widgets for the MCP viewer UI.
        """
        glyphs = get_glyphs()
        total_servers = len(self._server_info)
        total_tools = sum(len(s.tools) for s in self._server_info)

        with Vertical():
            if total_servers:
                server_label = "server" if total_servers == 1 else "servers"
                tool_label = "tool" if total_tools == 1 else "tools"
                title = (
                    f"MCP Servers ({total_servers} {server_label},"
                    f" {total_tools} {tool_label})"
                )
            else:
                title = "MCP Servers"
            yield Static(title, classes="mcp-viewer-title")

            with VerticalScroll(classes="mcp-list"):
                if not self._server_info:
                    yield Static(
                        "No MCP servers configured.\n"
                        "Use `--mcp-config` to load servers.",
                        classes="mcp-empty",
                    )
                else:
                    flat_index = 0
                    for server in self._server_info:
                        tool_count = len(server.tools)
                        t_label = "tool" if tool_count == 1 else "tools"
                        yield Static(
                            Content.from_markup(
                                "[bold]$name[/bold]"
                                f" [dim]$transport {glyphs.bullet}"
                                f" {tool_count} {t_label}[/dim]",
                                name=server.name,
                                transport=server.transport,
                            ),
                            classes="mcp-server-header",
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
                            yield widget
                            flat_index += 1

            help_text = (
                f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate"
                f" {glyphs.bullet} Enter expand/collapse"
                f" {glyphs.bullet} Esc close"
            )
            yield Static(help_text, classes="mcp-viewer-help")

    async def on_mount(self) -> None:
        """Apply ASCII border fallback if needed."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)

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
            self._tool_widgets[old].remove_class("mcp-tool-selected")
            self._tool_widgets[index].add_class("mcp-tool-selected")
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
