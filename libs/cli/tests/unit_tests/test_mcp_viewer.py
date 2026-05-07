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
        """Arrow / Tab navigation moves selection between tools.

        Vim-style `j`/`k` bindings are intentionally absent so they can be
        typed into the filter Input — see `test_letter_keys_type_into_filter`.
        """
        from textual.widgets import Input

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

            # Down again still moves selection (priority binding wins
            # over the filter Input even when it has focus)
            await pilot.press("down")
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

            # The filter Input exists and is the focused widget
            filter_input = screen.query_one("#mcp-filter", Input)
            assert filter_input.has_focus

    async def test_letter_keys_type_into_filter(self) -> None:
        """`j` and `k` are accepted by the filter Input, not navigation."""
        from textual.widgets import Input

        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            filter_input = screen.query_one("#mcp-filter", Input)
            assert filter_input.has_focus

            # Selection starts at 0; j/k must not move it.
            assert screen._selected_index == 0

            await pilot.press("j")
            await pilot.pause()
            assert "j" in filter_input.value

            await pilot.press("k")
            await pilot.pause()
            assert "k" in filter_input.value

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

    async def test_filter_narrows_tool_list(self) -> None:
        """Typing into the filter Input reduces the visible tool set."""
        from textual.widgets import Input

        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            assert len(screen._tool_widgets) == 3

            filter_input = screen.query_one("#mcp-filter", Input)
            filter_input.value = "search"
            await pilot.pause()

            visible = [w.tool_name for w in screen._tool_widgets]
            assert visible == ["search"]

    async def test_filter_clearing_restores_all(self) -> None:
        """Clearing the filter restores the full tool list."""
        from textual.widgets import Input

        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            filter_input = screen.query_one("#mcp-filter", Input)
            filter_input.value = "search"
            await pilot.pause()
            assert len(screen._tool_widgets) == 1

            filter_input.value = ""
            await pilot.pause()
            assert len(screen._tool_widgets) == 3

    async def test_filter_server_name_match_shows_all_tools(self) -> None:
        """Matching the server name surfaces every tool on that server."""
        from textual.widgets import Input

        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            filter_input = screen.query_one("#mcp-filter", Input)
            filter_input.value = "filesystem"
            await pilot.pause()

            visible = sorted(w.tool_name for w in screen._tool_widgets)
            assert visible == ["read_file", "write_file"]

    async def test_filter_multi_token_and(self) -> None:
        """Multi-token filter requires every token to match."""
        from textual.widgets import Input

        info = [
            MCPServerInfo(
                name="store",
                transport="stdio",
                tools=(
                    MCPToolInfo(name="search_orders", description="Search orders"),
                    MCPToolInfo(name="search_users", description="Search users"),
                    MCPToolInfo(name="list_orders", description="List orders"),
                ),
            ),
        ]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=info)
            app.push_screen(screen)
            await pilot.pause()

            filter_input = screen.query_one("#mcp-filter", Input)
            filter_input.value = "search orders"
            await pilot.pause()

            visible = [w.tool_name for w in screen._tool_widgets]
            assert visible == ["search_orders"]

    async def test_filter_no_matches_renders_empty_state(self) -> None:
        """An unmatched filter renders the 'No matching tools.' message."""
        from textual.widgets import Input

        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            filter_input = screen.query_one("#mcp-filter", Input)
            filter_input.value = "asdfghjkl"
            await pilot.pause()

            assert screen._tool_widgets == []
            empty_states = list(screen.query(".mcp-empty"))
            assert len(empty_states) == 1
            assert "No matching tools" in _widget_text(empty_states[0])

    async def test_filter_input_suppressed_while_connecting(self) -> None:
        """The filter Input is not mounted while the connecting placeholder shows."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=[], connecting=True)
            app.push_screen(screen)
            await pilot.pause()

            assert len(screen.query("#mcp-filter")) == 0

    async def test_expanded_tool_renders_parameters(self) -> None:
        """Expanding a tool with `input_schema` renders Parameters block."""
        info = [
            MCPServerInfo(
                name="srv",
                transport="stdio",
                tools=(
                    MCPToolInfo(
                        name="read_file",
                        description="Read a file",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "encoding": {"type": "string"},
                            },
                            "required": ["path"],
                        },
                    ),
                ),
            ),
        ]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=info)
            app.push_screen(screen)
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            tool_widget = screen._tool_widgets[0]
            text = _widget_text(tool_widget)
            assert "Parameters:" in text
            assert "path: string *" in text
            assert "encoding: string" in text
            # Optional param has no asterisk on its own line.
            assert "encoding: string *" not in text

    async def test_expanded_tool_without_schema_has_no_parameters(self) -> None:
        """Tool with `input_schema=None` shows only name + description."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            text = _widget_text(screen._tool_widgets[0])
            assert "Parameters:" not in text

    async def test_expanded_tool_with_empty_properties(self) -> None:
        """Empty `properties` dict means no Parameters block."""
        info = [
            MCPServerInfo(
                name="srv",
                transport="stdio",
                tools=(
                    MCPToolInfo(
                        name="ping",
                        description="No-op",
                        input_schema={"type": "object", "properties": {}},
                    ),
                ),
            ),
        ]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=info)
            app.push_screen(screen)
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()
            assert "Parameters:" not in _widget_text(screen._tool_widgets[0])

    async def test_expanded_tool_param_missing_type_renders_any(self) -> None:
        """Property without `type` renders as `:any`."""
        info = [
            MCPServerInfo(
                name="srv",
                transport="stdio",
                tools=(
                    MCPToolInfo(
                        name="run",
                        description="Run",
                        input_schema={
                            "type": "object",
                            "properties": {"opts": {}},
                        },
                    ),
                ),
            ),
        ]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=info)
            app.push_screen(screen)
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()
            assert "opts: any" in _widget_text(screen._tool_widgets[0])

    async def test_expanded_param_name_with_markup_is_safe(self) -> None:
        """A parameter name containing markup metachars renders literally."""
        info = [
            MCPServerInfo(
                name="srv",
                transport="stdio",
                tools=(
                    MCPToolInfo(
                        name="weird",
                        description="Has weird args",
                        input_schema={
                            "type": "object",
                            "properties": {"[bold]hax[/]": {"type": "string"}},
                        },
                    ),
                ),
            ),
        ]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=info)
            app.push_screen(screen)
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()
            text = _widget_text(screen._tool_widgets[0])
            # The literal characters should be present, not consumed as markup.
            assert "[bold]hax[/]" in text

    async def test_filter_param_name_match(self) -> None:
        """Param-name filter matches when input_schema.properties has the key."""
        from textual.widgets import Input

        info = [
            MCPServerInfo(
                name="srv",
                transport="stdio",
                tools=(
                    MCPToolInfo(
                        name="run",
                        description="Run something",
                        input_schema={
                            "type": "object",
                            "properties": {"target_path": {"type": "string"}},
                        },
                    ),
                    MCPToolInfo(name="reset", description="Reset state"),
                ),
            ),
        ]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=info)
            app.push_screen(screen)
            await pilot.pause()

            filter_input = screen.query_one("#mcp-filter", Input)
            filter_input.value = "target_path"
            await pilot.pause()

            visible = [w.tool_name for w in screen._tool_widgets]
            assert visible == ["run"]

    async def test_toggle_all_expands_then_collapses(self) -> None:
        """`Ctrl+E` expands every tool, pressing again collapses all."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            assert all(not w._expanded for w in screen._tool_widgets)

            await pilot.press("ctrl+e")
            await pilot.pause()
            assert all(w._expanded for w in screen._tool_widgets)

            await pilot.press("ctrl+e")
            await pilot.pause()
            assert all(not w._expanded for w in screen._tool_widgets)

    async def test_toggle_all_with_partial_state(self) -> None:
        """When some tools are collapsed, `Ctrl+E` expands the rest."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            screen._tool_widgets[0].set_expanded(True)
            await pilot.pause()
            assert screen._tool_widgets[0]._expanded
            assert not screen._tool_widgets[1]._expanded

            await pilot.press("ctrl+e")
            await pilot.pause()
            assert all(w._expanded for w in screen._tool_widgets)

    async def test_toggle_all_no_op_when_empty(self) -> None:
        """`Ctrl+E` with no tools is a no-op and does not raise."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=[])
            app.push_screen(screen)
            await pilot.pause()

            await pilot.press("ctrl+e")
            await pilot.pause()
            assert screen._tool_widgets == []

    async def test_toggle_all_only_affects_visible_after_filter(self) -> None:
        """A filter hides some tools; `Ctrl+E` must not change hidden ones."""
        from textual.widgets import Input

        info = [
            MCPServerInfo(
                name="srv",
                transport="stdio",
                tools=(
                    MCPToolInfo(name="alpha_one", description=""),
                    MCPToolInfo(name="alpha_two", description=""),
                    MCPToolInfo(name="beta_one", description=""),
                ),
            ),
        ]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=info)
            app.push_screen(screen)
            await pilot.pause()

            filter_input = screen.query_one("#mcp-filter", Input)
            filter_input.value = "alpha"
            await pilot.pause()

            visible = [w.tool_name for w in screen._tool_widgets]
            assert visible == ["alpha_one", "alpha_two"]

            await pilot.press("ctrl+e")
            await pilot.pause()

            # The two visible alpha tools became expanded; the filter
            # rebuild created widgets that were not part of the previous
            # toggle, so we assert against the post-press visible set.
            assert all(w._expanded for w in screen._tool_widgets)

    async def test_pressing_a_does_not_toggle_all(self) -> None:
        """`a` is no longer the toggle-all binding — it types into the filter."""
        from textual.widgets import Input

        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            filter_input = screen.query_one("#mcp-filter", Input)
            await pilot.press("a")
            await pilot.pause()
            assert "a" in filter_input.value
            # No expansion changed because nothing matched "a" filtering;
            # the rebuilt widget list starts collapsed again.
            assert all(not w._expanded for w in screen._tool_widgets)

    async def test_help_text_lists_all_keybindings(self) -> None:
        """Footer mentions navigate, expand, expand all, filter, and close."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            help_widgets = list(screen.query(".mcp-viewer-help"))
            assert len(help_widgets) == 1
            text = _widget_text(help_widgets[0]).lower()
            assert "navigate" in text
            assert "enter" in text
            assert "ctrl+e" in text
            assert "filter" in text
            assert "esc" in text

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
