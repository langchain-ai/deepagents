"""Tests for the MCP viewer modal screen."""

from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.widgets import Static

from deepagents_cli.mcp_tools import MCPServerInfo, MCPToolInfo
from deepagents_cli.widgets.mcp_viewer import (
    MCPToolItem,
    MCPViewerScreen,
)


def _widget_text(widget: Widget) -> str:
    """Extract plain text content from a Static widget."""
    content = widget._Static__content  # type: ignore[attr-defined]
    return str(content)


class MCPViewerTestApp(App[None]):
    """Minimal app wrapper for testing MCPViewerScreen."""

    def compose(self) -> ComposeResult:
        yield Static("base")


def _sample_info() -> list[MCPServerInfo]:
    return [
        MCPServerInfo(
            name="filesystem",
            transport="stdio",
            tools=[
                MCPToolInfo(name="read_file", description="Read a file"),
                MCPToolInfo(name="write_file", description="Write a file"),
            ],
        ),
        MCPServerInfo(
            name="remote-api",
            transport="sse",
            tools=[
                MCPToolInfo(name="search", description="Search the web"),
            ],
        ),
    ]


class TestMCPViewerScreen:
    """Tests for the MCP viewer screen widget."""

    async def test_render_with_servers(self) -> None:
        """Viewer displays server names, transports, and tool info."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            title = screen.query_one(".mcp-viewer-title", Static)
            assert "2 servers" in _widget_text(title)
            assert "3 tools" in _widget_text(title)

            headers = screen.query(".mcp-server-header")
            assert len(headers) == 2
            assert "filesystem" in _widget_text(headers[0])
            assert "remote-api" in _widget_text(headers[1])

            tools = screen.query(".mcp-tool-item")
            assert len(tools) == 3

    async def test_render_empty_state(self) -> None:
        """Viewer shows empty message when no servers configured."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=[])
            app.push_screen(screen)
            await pilot.pause()

            title = screen.query_one(".mcp-viewer-title", Static)
            assert "MCP Servers" in _widget_text(title)

            empty = screen.query_one(".mcp-empty", Static)
            assert "--mcp-config" in _widget_text(empty)

    async def test_escape_dismisses(self) -> None:
        """Pressing Escape closes the viewer."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            dismissed = False

            def on_dismiss(result: None) -> None:  # noqa: ARG001
                nonlocal dismissed
                dismissed = True

            screen = MCPViewerScreen(server_info=[])
            app.push_screen(screen, on_dismiss)
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()
            assert dismissed

    async def test_single_server_singular_labels(self) -> None:
        """Title uses singular forms for 1 server and 1 tool."""
        info = [
            MCPServerInfo(
                name="only",
                transport="http",
                tools=[MCPToolInfo(name="do_thing", description="")],
            ),
        ]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=info)
            app.push_screen(screen)
            await pilot.pause()

            title = screen.query_one(".mcp-viewer-title", Static)
            text = _widget_text(title)
            assert "1 server," in text
            assert "1 tool)" in text

    async def test_keyboard_navigation(self) -> None:
        """Up/down keys move selection between tools."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            # First tool starts selected
            assert screen._selected_index == 0
            assert screen._tool_widgets[0].has_class("mcp-tool-selected")

            # Move down
            await pilot.press("down")
            await pilot.pause()
            assert screen._selected_index == 1
            assert screen._tool_widgets[1].has_class("mcp-tool-selected")
            assert not screen._tool_widgets[0].has_class("mcp-tool-selected")

            # Move down again
            await pilot.press("j")
            await pilot.pause()
            assert screen._selected_index == 2

            # Wrap around
            await pilot.press("down")
            await pilot.pause()
            assert screen._selected_index == 0

    async def test_enter_toggles_expand(self) -> None:
        """Enter key expands and collapses tool description."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            widget = screen._tool_widgets[0]
            assert isinstance(widget, MCPToolItem)
            assert not widget._expanded

            # Expand
            await pilot.press("enter")
            await pilot.pause()
            assert widget._expanded
            rendered = _widget_text(widget)
            assert "read_file" in rendered
            assert "Read a file" in rendered

            # Collapse
            await pilot.press("enter")
            await pilot.pause()
            assert not widget._expanded

    async def test_click_expands_tool(self) -> None:
        """Clicking a tool selects it and toggles expand."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            widget = screen._tool_widgets[0]
            assert not widget._expanded

            await pilot.click(MCPToolItem)
            await pilot.pause()
            assert widget._expanded

    async def test_toggle_all_expands_then_collapses(self) -> None:
        """Pressing `a` expands every tool, pressing again collapses all."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            assert all(not w._expanded for w in screen._tool_widgets)

            # First press expands every tool
            await pilot.press("a")
            await pilot.pause()
            assert all(w._expanded for w in screen._tool_widgets)

            # Second press collapses every tool
            await pilot.press("a")
            await pilot.pause()
            assert all(not w._expanded for w in screen._tool_widgets)

    async def test_toggle_all_with_partial_state(self) -> None:
        """If any tool is collapsed, `a` expands all; otherwise collapses all."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            # Expand only the first tool
            screen._tool_widgets[0].set_expanded(True)
            await pilot.pause()
            assert screen._tool_widgets[0]._expanded
            assert not screen._tool_widgets[1]._expanded

            # `a` should expand the rest (because some are still collapsed)
            await pilot.press("a")
            await pilot.pause()
            assert all(w._expanded for w in screen._tool_widgets)

    async def test_toggle_all_no_op_when_empty(self) -> None:
        """Pressing `a` with no tools is a no-op."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=[])
            app.push_screen(screen)
            await pilot.pause()

            # Should not raise
            await pilot.press("a")
            await pilot.pause()
            assert screen._tool_widgets == []
