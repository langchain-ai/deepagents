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
            tools=(
                MCPToolInfo(name="read_file", description="Read a file"),
                MCPToolInfo(name="write_file", description="Write a file"),
            ),
        ),
        MCPServerInfo(
            name="remote-api",
            transport="sse",
            tools=(MCPToolInfo(name="search", description="Search the web"),),
        ),
    ]


def _mixed_status_info() -> list[MCPServerInfo]:
    """Three servers covering all `MCPServerStatus` values."""
    return [
        MCPServerInfo(
            name="filesystem",
            transport="stdio",
            tools=(MCPToolInfo(name="read_file", description="Read a file"),),
        ),
        MCPServerInfo(
            name="github",
            transport="http",
            status="unauthenticated",
            error="Run: deepagents mcp login github",
        ),
        MCPServerInfo(
            name="broken",
            transport="sse",
            status="error",
            error="Connection refused",
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
                tools=(MCPToolInfo(name="do_thing", description=""),),
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

            # Tab advances selection
            await pilot.press("tab")
            await pilot.pause()
            assert screen._selected_index == 1

            # Shift+Tab moves selection back
            await pilot.press("shift+tab")
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

    async def test_three_state_status_indicators_render(self) -> None:
        """Each `MCPServerStatus` produces a visually distinct header line."""
        from deepagents_cli import theme

        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_mixed_status_info())
            app.push_screen(screen)
            await pilot.pause()

            colors = theme.get_theme_colors(screen)
            headers = screen.query(".mcp-server-header")
            assert len(headers) == 3

            # Headers are ordered: filesystem (ok), github (unauth), broken (err).
            ok_text = _widget_text(headers[0])
            unauth_text = _widget_text(headers[1])
            err_text = _widget_text(headers[2])

            assert "filesystem" in ok_text
            assert "stdio" in ok_text

            assert "github" in unauth_text
            assert "unauthenticated" in unauth_text
            assert "Run: deepagents mcp login github" in unauth_text

            assert "broken" in err_text
            assert "error" in err_text
            assert "Connection refused" in err_text

            # Each header carries the matching theme color in its content spans.
            ok_spans = repr(headers[0]._Static__content)  # type: ignore[attr-defined]
            unauth_spans = repr(headers[1]._Static__content)  # type: ignore[attr-defined]
            err_spans = repr(headers[2]._Static__content)  # type: ignore[attr-defined]
            assert colors.success in ok_spans
            assert colors.warning in unauth_spans
            assert colors.error in err_spans

    async def test_status_indicator_glyphs_use_glyph_set(self) -> None:
        """Status icons reuse existing `Glyphs` (unicode by default)."""
        from deepagents_cli.config import get_glyphs

        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_mixed_status_info())
            app.push_screen(screen)
            await pilot.pause()

            glyphs = get_glyphs()
            headers = screen.query(".mcp-server-header")
            assert glyphs.checkmark in _widget_text(headers[0])
            assert glyphs.warning in _widget_text(headers[1])
            assert glyphs.error in _widget_text(headers[2])

    async def test_synthetic_config_error_entry_renders(self) -> None:
        """A `<config:foo>` entry from a malformed config file does not crash."""
        info = [
            MCPServerInfo(
                name="<config:bad.json>",
                transport="config",
                status="error",
                error="JSON decode failed at line 3",
            ),
        ]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=info)
            app.push_screen(screen)
            await pilot.pause()

            headers = screen.query(".mcp-server-header")
            assert len(headers) == 1
            text = _widget_text(headers[0])
            assert "<config:bad.json>" in text
            assert "JSON decode failed" in text
            # No tools to render — only the header line and the help footer.
            assert len(screen.query(".mcp-tool-item")) == 0

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
