"""Tests for the MCP viewer modal screen."""

import asyncio

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.notifications import Notification
from textual.widget import Widget
from textual.widgets import Static

from deepagents_code.mcp_tools import MCPServerInfo, MCPToolInfo
from deepagents_code.tui.widgets.mcp_viewer import (
    MCP_VIEWER_RECONNECT_REQUEST,
    MCPServerErrorScreen,
    MCPServerHeaderItem,
    MCPToolItem,
    MCPViewerScreen,
)


def _widget_text(widget: Widget) -> str:
    """Extract plain text content from a Static widget."""
    content = widget._Static__content  # ty: ignore
    return str(content)


def _latest_notification(app: App[None]) -> Notification | None:
    """Return the most recently raised toast, or `None` if there are none."""
    notifications = list(app._notifications)
    return notifications[-1] if notifications else None


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
            uses_oauth=True,
        ),
    ]


def _mixed_status_info() -> list[MCPServerInfo]:
    """Servers covering all `MCPServerStatus` values."""
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
            error="Run: dcode mcp login github",
        ),
        MCPServerInfo(
            name="notion",
            transport="http",
            status="awaiting_reconnect",
            error="Authenticated — run `/mcp reconnect` to load tools.",
        ),
        MCPServerInfo(
            name="broken",
            transport="sse",
            status="error",
            error="Connection refused",
        ),
        MCPServerInfo(
            name="paused",
            transport="stdio",
            status="disabled",
            error="Disabled in this session",
        ),
    ]


class TestMCPViewerScreen:
    """Tests for the MCP viewer screen widget."""

    async def test_refresh_focuses_filter_input(self) -> None:
        """Refreshing after server-ready must refocus the filter input.

        Regression: a viewer opened while the server is still connecting
        shows a placeholder with no filter input. When tools load,
        `refresh_server_info` rebuilds the body and mounts the filter
        `Input`, but Textual only auto-focuses on the first mount — so the
        input was left unfocused and keystrokes never reached it. The final
        `press` + `value` assertion verifies the typed text actually lands
        in the input, not just that focus was restored.
        """
        from textual.widgets import Input

        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=[], connecting=True)
            app.push_screen(screen)
            await pilot.pause()

            # No filter input exists during the connecting placeholder.
            assert not screen.query("#mcp-filter")

            await screen.refresh_server_info(_sample_info())
            await pilot.pause()

            filter_input = screen.query_one("#mcp-filter", Input)
            assert app.focused is filter_input

            await pilot.press("r", "e", "a", "d")
            await pilot.pause()
            assert filter_input.value == "read"

    @pytest.mark.parametrize("size", [(80, 24), (80, 14), (50, 20)])
    async def test_footer_hints_stay_on_screen(self, size: tuple[int, int]) -> None:
        """Every hint renders, however narrow or short the window is.

        Asserts the footer's region sits inside the modal, not just that
        its text is set: the bug being guarded here is that the string is
        complete but laid out past the modal's bottom edge, where the
        compositor never paints it. `_widget_text` cannot observe that.
        `(80, 14)` and `(50, 20)` are both sizes where the footer used to
        be pushed clean out of the modal.
        """
        app = MCPViewerTestApp()
        async with app.run_test(size=size) as pilot:
            screen = MCPViewerScreen(
                server_info=_sample_info(),
                pending_reconnect=True,
            )
            app.push_screen(screen)
            await pilot.pause()

            help_widget = screen.query_one(".mcp-viewer-help", Static)
            modal = screen.query_one(Vertical)
            assert help_widget in app.screen._compositor.visible_widgets
            assert modal.content_region.contains_region(help_widget.region)
            assert "Ctrl+R reconnect" in _widget_text(help_widget)
            assert "Esc close" in _widget_text(help_widget)

    async def test_footer_wraps_rather_than_truncating(self) -> None:
        """A footer too long for one line grows instead of losing hints."""
        app = MCPViewerTestApp()
        async with app.run_test(size=(80, 24)) as pilot:
            screen = MCPViewerScreen(
                server_info=_sample_info(),
                pending_reconnect=True,
            )
            app.push_screen(screen)
            await pilot.pause()

            help_widget = screen.query_one(".mcp-viewer-help", Static)
            # The modal caps at `width: 80`, so the full hint string never
            # fits on one line — it must occupy two.
            assert help_widget.size.height == 2

    async def test_escape_dismisses(self) -> None:
        """Pressing Escape closes the viewer."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            dismissed = False

            def on_dismiss(result: str | None) -> None:  # noqa: ARG001
                nonlocal dismissed
                dismissed = True

            screen = MCPViewerScreen(server_info=[])
            app.push_screen(screen, on_dismiss)
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()
            assert dismissed

    async def test_f2_falls_back_when_server_missing_from_refreshed_info(
        self,
    ) -> None:
        """Toggle falls back to full rebuild when the server vanishes.

        Guards the `new_server is None` branch in
        `apply_server_disable_toggle` — a regression that skipped the
        fallback would leave the viewer showing stale rows for the
        missing server. Also pins focus: the fallback routes through
        `refresh_server_info`, which must refocus the filter `Input` (this
        is a second live caller of `_focus_filter_input` alongside the
        server-ready path).
        """
        from textual.widgets import Input

        app = MCPViewerTestApp()
        async with app.run_test() as pilot:

            async def on_toggle(server_name: str) -> None:
                # Drop the toggled server entirely from the new list,
                # forcing the in-place patch into its fallback path.
                updated = [info for info in _sample_info() if info.name != server_name]
                await screen.apply_server_disable_toggle(
                    updated,
                    toggled_server=server_name,
                    pending_reconnect=True,
                )

            screen = MCPViewerScreen(
                server_info=_sample_info(), on_toggle_disable=on_toggle
            )
            app.push_screen(screen)
            await pilot.pause()

            await pilot.press("f2")
            await pilot.pause()

            headers = screen.query(".mcp-server-header")
            # `filesystem` vanished, `remote-api` remains.
            assert len(headers) == 1
            remaining = screen._row_widgets[0]
            assert isinstance(remaining, MCPServerHeaderItem)
            assert remaining.server.name == "remote-api"

            # Fallback re-mounts the body; focus must land on the filter input.
            assert app.focused is screen.query_one("#mcp-filter", Input)

    async def test_tool_description_truncated_on_first_paint(self) -> None:
        """First paint has ellipsis truncation, not the full description.

        `MCPToolItem.on_mount` defers via `call_after_refresh` so this
        passes; a regression to a synchronous `_rerender()` in
        `on_mount` would render the un-truncated description for one
        frame because `self.size.width == 0` short-circuits the
        truncation guard.
        """
        long_desc = "x" * 500
        info = [
            MCPServerInfo(
                name="filesystem",
                transport="stdio",
                tools=(MCPToolInfo(name="tool", description=long_desc),),
            )
        ]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=info)
            app.push_screen(screen)
            await pilot.pause()

            tool_widget = next(
                w for w in screen._row_widgets if isinstance(w, MCPToolItem)
            )
            rendered = _widget_text(tool_widget)
            assert "(...)" in rendered
            assert long_desc not in rendered

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

            # Step past the server header onto the tool row, then expand.
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            tool_widget = screen._tool_widgets[0]
            text = _widget_text(tool_widget)
            assert "Parameters:" in text
            assert "path: string *" in text
            assert "encoding: string" in text
            # Optional param has no asterisk on its own line.
            assert "encoding: string *" not in text

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

            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            assert "Parameters:" not in _widget_text(screen._tool_widgets[0])

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

            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            text = _widget_text(screen._tool_widgets[0])
            # The literal characters should be present, not consumed as markup.
            assert "[bold]hax[/]" in text

    async def test_arrow_down_scrolls_inside_tall_tool_then_jumps(self) -> None:
        """Down scrolls inside an over-tall expanded tool; then jumps to next."""
        from textual.containers import VerticalScroll

        long_desc = "\n".join(f"line {i}" for i in range(40))
        info = [
            MCPServerInfo(
                name="srv",
                transport="stdio",
                tools=(
                    MCPToolInfo(name="big", description=long_desc),
                    MCPToolInfo(name="next", description="short"),
                ),
            ),
        ]
        # Rows: [0: srv header, 1: big, 2: next]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=info)
            app.push_screen(screen)
            await pilot.pause()

            # Step to big (row 1) and expand.
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            assert screen._selected_index == 1
            assert screen._row_widgets[1]._expanded  # ty: ignore

            scroll = screen.query_one(".mcp-list", VerticalScroll)
            initial_offset = scroll.scroll_offset.y

            # First Down must scroll, not jump.
            await pilot.press("down")
            await pilot.pause()
            assert screen._selected_index == 1
            assert scroll.scroll_offset.y > initial_offset

            # Put the bottom edge in view, then the next Down should jump.
            scroll.scroll_relative(y=1000, animate=False)
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            assert screen._selected_index == 2, (
                "expected to jump to next tool once the bottom is visible"
            )

    async def test_arrow_up_scrolls_inside_tall_tool_then_jumps(self) -> None:
        """Up scrolls back through an over-tall expanded tool; then jumps."""
        from textual.containers import VerticalScroll

        long_desc = "\n".join(f"line {i}" for i in range(40))
        info = [
            MCPServerInfo(
                name="srv",
                transport="stdio",
                tools=(
                    MCPToolInfo(name="prev", description="short"),
                    MCPToolInfo(name="big", description=long_desc),
                ),
            ),
        ]
        # Rows: [0: srv header, 1: prev, 2: big]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=info)
            app.push_screen(screen)
            await pilot.pause()

            # Walk to "big" (row 2) and expand it; scroll past its top.
            await pilot.press("down")
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            assert screen._selected_index == 2

            scroll = screen.query_one(".mcp-list", VerticalScroll)
            scroll.scroll_relative(y=30, animate=False)
            await pilot.pause()
            offset_before = scroll.scroll_offset.y

            # First Up must scroll back, not jump.
            await pilot.press("up")
            await pilot.pause()
            assert screen._selected_index == 2
            assert scroll.scroll_offset.y < offset_before

            # Put the top edge in view, then the next Up should jump to "prev".
            scroll.scroll_relative(y=-1000, animate=False)
            await pilot.pause()
            await pilot.press("up")
            await pilot.pause()
            assert screen._selected_index == 1

    async def test_up_jump_pins_previous_tool_to_viewport_bottom(self) -> None:
        """After jumping up, the new tool's bottom is at the viewport bottom.

        This means the next `Up` immediately line-scrolls within that tool
        (does not re-jump), so the user can keep reading upward.
        """
        from textual.containers import VerticalScroll

        long_desc = "\n".join(f"line {i}" for i in range(40))
        info = [
            MCPServerInfo(
                name="srv",
                transport="stdio",
                tools=(
                    MCPToolInfo(name="big", description=long_desc),
                    MCPToolInfo(name="next", description="short"),
                ),
            ),
        ]
        # Rows: [0: srv header, 1: big, 2: next]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=info)
            app.push_screen(screen)
            await pilot.pause()

            # Step to big (row 1) and expand.
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            scroll = screen.query_one(".mcp-list", VerticalScroll)
            big = screen._tool_widgets[0]

            # Put the bottom edge in view, then press Down to jump to "next" (row 2).
            scroll.scroll_relative(y=1000, animate=False)
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            assert screen._selected_index == 2

            # Press Up — should jump back to "big" (row 1) and pin its
            # bottom near the viewport bottom (within 1 row, allowing for
            # layout-tick rounding).
            await pilot.press("up")
            await pilot.pause()
            assert screen._selected_index == 1
            big_bottom = big.region.y + big.region.height
            viewport_bottom = scroll.region.y + scroll.region.height
            assert abs(big_bottom - viewport_bottom) <= 1

            # The next Up must line-scroll inside "big" (row 1), not jump
            # to the server header above. Smart-scroll keeps the cursor in
            # place and just shifts the viewport.
            offset_before = scroll.scroll_offset.y
            await pilot.press("up")
            await pilot.pause()
            assert screen._selected_index == 1
            assert scroll.scroll_offset.y < offset_before

    async def test_navigation_wraps_at_list_ends(self) -> None:
        """Arrow and Tab navigation wrap in both directions."""
        # Rows: [0: filesystem header, 1: read_file, 2: write_file,
        #        3: remote-api header, 4: search]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            assert screen._selected_index == 0

            # Up from the first row wraps to the final row; Down wraps back.
            await pilot.press("up")
            await pilot.pause()
            assert screen._selected_index == 4
            await pilot.press("down")
            await pilot.pause()
            assert screen._selected_index == 0

            # Shift+Tab from the first server wraps to the final server; Tab
            # from there wraps back to the first server.
            await pilot.press("shift+tab")
            await pilot.pause()
            assert screen._selected_index == 3
            await pilot.press("tab")
            await pilot.pause()
            assert screen._selected_index == 0

    async def test_tab_always_jumps_even_inside_tall_tool(self) -> None:
        """Tab / Shift+Tab jump to headers despite tall expanded content."""
        from textual.containers import VerticalScroll

        long_desc = "\n".join(f"line {i}" for i in range(40))
        info = [
            MCPServerInfo(
                name="srv",
                transport="stdio",
                tools=(
                    MCPToolInfo(name="big", description=long_desc),
                    MCPToolInfo(name="next", description="short"),
                ),
            ),
            MCPServerInfo(
                name="other",
                transport="stdio",
                tools=(MCPToolInfo(name="last", description="short"),),
            ),
        ]
        # Rows: [0: srv header, 1: big, 2: next, 3: other header, 4: last]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=info)
            app.push_screen(screen)
            await pilot.pause()

            # Step from the header to big and expand it.
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            assert screen._selected_index == 1

            # Precondition: big is now taller than the viewport, so a plain
            # Down would only line-scroll — the jump below must ignore that.
            scroll = screen.query_one(".mcp-list", VerticalScroll)
            assert screen._row_widgets[1].region.height > scroll.region.height

            # Tab jumps past the expanded body to the next server.
            await pilot.press("tab")
            await pilot.pause()
            assert screen._selected_index == 3

            # Return to big, then Shift+Tab jumps to its server header.
            await pilot.press("shift+tab")
            await pilot.press("down")
            await pilot.pause()
            assert screen._selected_index == 1
            await pilot.press("shift+tab")
            await pilot.pause()
            assert screen._selected_index == 0

    async def test_tab_single_server_returns_to_header(self) -> None:
        """Tab/Shift+Tab return a tool to its sole header, else no-op.

        With one server, jumping from the tool lands on that server's header.
        Jumping again from the lone header has nowhere to go, so it must be a
        true no-op that leaves the scroll offset untouched — even when an
        expanded tool has scrolled the viewport.
        """
        from textual.containers import VerticalScroll

        long_desc = "\n".join(f"line {i}" for i in range(40))
        info = [
            MCPServerInfo(
                name="srv",
                transport="stdio",
                tools=(MCPToolInfo(name="only", description=long_desc),),
            ),
        ]
        # Rows: [0: srv header, 1: only]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=info)
            app.push_screen(screen)
            await pilot.pause()

            # Tab and Shift+Tab from the tool both return to the sole header.
            await pilot.press("down")
            await pilot.press("tab")
            await pilot.pause()
            assert screen._selected_index == 0

            await pilot.press("down")
            await pilot.press("shift+tab")
            await pilot.pause()
            assert screen._selected_index == 0

            # Expand the tool (taller than the viewport) and Tab back up to the
            # header so a later scroll has room to move.
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.press("tab")
            await pilot.pause()
            assert screen._selected_index == 0

            scroll = screen.query_one(".mcp-list", VerticalScroll)
            scroll.scroll_relative(y=10, animate=False)
            await pilot.pause()
            offset = scroll.scroll_offset

            # On the lone header, Tab and Shift+Tab are no-ops: neither the
            # selection nor the viewport moves.
            await pilot.press("tab")
            await pilot.pause()
            assert screen._selected_index == 0
            assert scroll.scroll_offset == offset

            await pilot.press("shift+tab")
            await pilot.pause()
            assert screen._selected_index == 0
            assert scroll.scroll_offset == offset

    async def test_error_only_server_list_is_navigable(self) -> None:
        """A list with no `ok` servers (only error/unauth) is still navigable.

        Before R10 this was the original pain point — `_tool_widgets` was
        empty by `MCPServerInfo` invariant, so the cursor had nowhere to go
        and the user could not read or interact with the failed-server info.
        """
        info = [
            MCPServerInfo(
                name="broken-a",
                transport="http",
                status="error",
                error="Connection refused",
            ),
            MCPServerInfo(
                name="broken-b",
                transport="sse",
                status="error",
                error="Timed out",
            ),
        ]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=info)
            app.push_screen(screen)
            await pilot.pause()

            assert len(screen._row_widgets) == 2
            assert all(isinstance(w, MCPServerHeaderItem) for w in screen._row_widgets)

            assert screen._selected_index == 0
            await pilot.press("down")
            await pilot.pause()
            assert screen._selected_index == 1

            # Down wraps across header-only lists. Tab and Shift+Tab traverse
            # those same rows even though there are no tools.
            await pilot.press("down")
            await pilot.pause()
            assert screen._selected_index == 0
            await pilot.press("tab")
            await pilot.pause()
            assert screen._selected_index == 1
            await pilot.press("shift+tab")
            await pilot.pause()
            assert screen._selected_index == 0

            # Up wraps from the first header back to the last.
            await pilot.press("up")
            await pilot.pause()
            assert screen._selected_index == 1

    async def test_enter_on_unauth_header_dismisses_with_server_name(self) -> None:
        """Activating an unauthenticated header returns the server name.

        The app uses the dismiss value to drive in-TUI OAuth login.
        """
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            dismissed_with: list[str | None] = []

            def on_dismiss(result: str | None) -> None:
                dismissed_with.append(result)

            screen = MCPViewerScreen(server_info=_mixed_status_info())
            app.push_screen(screen, on_dismiss)
            await pilot.pause()

            # `unauthenticated` servers are floated to the top, so github
            # is now the first row and starts selected.
            assert screen._row_widgets[0]._server.name == "github"  # ty: ignore
            help_widget = screen.query_one(".mcp-viewer-help", Static)
            assert "Enter log in" in _widget_text(help_widget)

            await pilot.press("enter")
            await pilot.pause()

            assert dismissed_with == ["github"]

    @pytest.mark.parametrize("transport", ["http", "sse"])
    async def test_enter_on_oauth_remote_reauthenticates(self, transport: str) -> None:
        """Activating a healthy OAuth-backed remote server starts OAuth again.

        Both remote transports are covered because `_can_start_login` gates on
        the normalized `transport` string, which `mcp_tools` derives two modules
        away — `http` alone would not pin `sse`.
        """
        info = [
            MCPServerInfo(
                name="slack",
                transport=transport,
                tools=(MCPToolInfo(name="search", description="Search Slack"),),
                uses_oauth=True,
            )
        ]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            dismissed_with: list[str | None] = []

            def on_dismiss(result: str | None) -> None:
                dismissed_with.append(result)

            screen = MCPViewerScreen(server_info=info)
            app.push_screen(screen, on_dismiss)
            await pilot.pause()

            header = screen._row_widgets[0]
            assert isinstance(header, MCPServerHeaderItem)
            assert "re-auth" not in _widget_text(header)
            help_widget = screen.query_one(".mcp-viewer-help", Static)
            assert "Enter re-auth" in _widget_text(help_widget)

            await pilot.press("enter")
            await pilot.pause()

            assert dismissed_with == ["slack"]

    async def test_enter_on_non_oauth_remote_is_noop(self) -> None:
        """A healthy remote server not backed by OAuth offers no re-auth.

        A static `Authorization` header takes precedence over stored OAuth
        tokens, and a public server has no flow to run, so the affordance
        would be a lie.
        """
        info = [
            MCPServerInfo(
                name="public-api",
                transport="http",
                tools=(MCPToolInfo(name="search", description="Search"),),
                uses_oauth=False,
            )
        ]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            dismissed_with: list[str | None] = []

            def on_dismiss(result: str | None) -> None:
                dismissed_with.append(result)

            screen = MCPViewerScreen(server_info=info)
            app.push_screen(screen, on_dismiss)
            await pilot.pause()

            header = screen._row_widgets[0]
            assert isinstance(header, MCPServerHeaderItem)
            assert "re-auth" not in _widget_text(header)
            help_widget = screen.query_one(".mcp-viewer-help", Static)
            assert "Enter" not in _widget_text(help_widget)

            await pilot.press("enter")
            await pilot.pause()

            assert dismissed_with == []

    async def test_click_on_selected_oauth_header_reauthenticates(self) -> None:
        """Re-clicking a selected OAuth remote header starts login.

        `on_click` shares the `_can_start_login` gate with `action_toggle_expand`
        but is a separate activation path, so it needs its own coverage.
        """
        info = [
            MCPServerInfo(
                name="slack",
                transport="http",
                tools=(MCPToolInfo(name="search", description="Search Slack"),),
                uses_oauth=True,
            )
        ]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            dismissed_with: list[str | None] = []

            def on_dismiss(result: str | None) -> None:
                dismissed_with.append(result)

            screen = MCPViewerScreen(server_info=info)
            app.push_screen(screen, on_dismiss)
            await pilot.pause()

            header = screen._row_widgets[0]
            assert isinstance(header, MCPServerHeaderItem)
            assert screen._selected_index == 0

            await pilot.click(header)
            await pilot.pause()

            assert dismissed_with == ["slack"]

    async def test_enter_on_error_header_opens_detail_modal(self) -> None:
        """Activating an error-status header opens a detail modal."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            dismissed_with: list[str | None] = []

            def on_dismiss(result: str | None) -> None:
                dismissed_with.append(result)

            screen = MCPViewerScreen(server_info=_mixed_status_info())
            app.push_screen(screen, on_dismiss)
            await pilot.pause()

            # Attention-needed states are floated to the top: github(0),
            # notion(1), filesystem(2), read_file tool(3), broken(4).
            for _ in range(4):
                await pilot.press("down")
                await pilot.pause()
            assert screen._row_widgets[4]._server.name == "broken"  # ty: ignore

            await pilot.press("enter")
            await pilot.pause()

            assert dismissed_with == []
            assert isinstance(app.screen, MCPServerErrorScreen)
            assert "Connection refused" in _widget_text(
                app.screen.query_one(".mcp-error-text", Static)
            )

            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is screen

    async def test_enter_on_stdio_server_header_is_noop(self) -> None:
        """Enter on a healthy stdio server does not start OAuth or expand."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            dismissed_with: list[str | None] = []

            def on_dismiss(result: str | None) -> None:
                dismissed_with.append(result)

            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen, on_dismiss)
            await pilot.pause()

            assert isinstance(screen._row_widgets[0], MCPServerHeaderItem)
            assert screen._selected_index == 0
            help_widget = screen.query_one(".mcp-viewer-help", Static)
            assert "Enter" not in _widget_text(help_widget)

            # Press Enter on the header — must be a no-op and must not
            # affect any tool's expansion state.
            await pilot.press("enter")
            await pilot.pause()
            assert dismissed_with == []
            assert screen._selected_index == 0
            assert all(not tool._expanded for tool in screen._tool_widgets)

    async def test_help_text_matches_selected_row(self) -> None:
        """Footer describes the Enter action for the selected row."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            help_widgets = list(screen.query(".mcp-viewer-help"))
            assert len(help_widgets) == 1
            help_widget = help_widgets[0]
            text = _widget_text(help_widget).lower()
            assert "navigate" in text
            assert "tab/shift+tab servers" in text
            assert "enter" not in text
            assert "f2" in text
            assert "ctrl+e" in text
            assert "filter" in text
            assert "esc" in text

            await pilot.press("down")
            await pilot.pause()
            assert "enter expand/collapse" in _widget_text(help_widget).lower()

            await pilot.press("tab")
            await pilot.pause()
            assert "enter re-auth" in _widget_text(help_widget).lower()

    async def test_status_indicators_render(self) -> None:
        """Each `MCPServerStatus` produces a visually distinct header line.

        We assert on rendered text + glyph (the user-visible signal); the
        per-state theme color is verified separately by the unit-level
        `_status_color` test, not by inspecting `Content` internal repr.
        """
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_mixed_status_info())
            app.push_screen(screen)
            await pilot.pause()

            headers = screen.query(".mcp-server-header")
            assert len(headers) == 5

            # `unauthenticated` servers float to the top, so the order is:
            # github (unauth), notion (ready to load), filesystem (ok),
            # broken (err), paused (disabled).
            unauth_text = _widget_text(headers[0])
            pending_text = _widget_text(headers[1])
            ok_text = _widget_text(headers[2])
            err_text = _widget_text(headers[3])
            disabled_text = _widget_text(headers[4])

            assert "filesystem" in ok_text
            assert "stdio" in ok_text

            assert "github" in unauth_text
            assert "unauthenticated" in unauth_text

            assert "notion" in pending_text
            assert "ready to load" in pending_text
            assert "Ctrl+R to load tools" in pending_text

            assert "broken" in err_text
            assert "error" in err_text
            assert "Connection refused" not in err_text

            assert "paused" in disabled_text
            assert "disabled" in disabled_text

    async def test_status_indicator_glyphs_use_glyph_set(self) -> None:
        """Status icons reuse existing `Glyphs` (unicode by default)."""
        from deepagents_code.config import get_glyphs

        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_mixed_status_info())
            app.push_screen(screen)
            await pilot.pause()

            glyphs = get_glyphs()
            headers = screen.query(".mcp-server-header")
            # Attention-needed states float to the top: unauth (warning),
            # awaiting_reconnect (empty circle), ok, error, disabled.
            assert glyphs.warning in _widget_text(headers[0])
            assert glyphs.circle_empty in _widget_text(headers[1])
            assert glyphs.checkmark in _widget_text(headers[2])
            assert glyphs.error in _widget_text(headers[3])
            assert glyphs.pause in _widget_text(headers[4])

    async def test_error_modal_escape_dismisses(self) -> None:
        """Escape closes the error modal without disturbing the parent."""
        screen = MCPViewerScreen(server_info=_mixed_status_info())
        server = MCPServerInfo(
            name="broken",
            transport="sse",
            status="error",
            error="Connection refused",
        )
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            app.push_screen(screen)
            await pilot.pause()
            screen.show_server_error(server)
            await pilot.pause()
            assert isinstance(app.screen, MCPServerErrorScreen)

            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is screen

    async def test_error_modal_footer_wraps_in_narrow_window(self) -> None:
        """The error modal's footer wraps rather than truncating."""
        server = MCPServerInfo(
            name="broken",
            transport="stdio",
            status="error",
            error="Server exited with code 1.",
        )
        app = MCPViewerTestApp()
        # 30 columns leaves 25 for the hints, two short of the 27 they
        # need — the narrowest realistic window that forces a wrap.
        async with app.run_test(size=(30, 12)) as pilot:
            app.push_screen(MCPServerErrorScreen(server))
            await pilot.pause()

            help_widget = app.screen.query_one(".mcp-error-help", Static)
            modal = app.screen.query_one(Vertical)
            assert help_widget.size.height == 2
            assert modal.content_region.contains_region(help_widget.region)
            assert "Esc close" in _widget_text(help_widget)

    async def test_f2_hint_only_shows_on_server_row(self) -> None:
        """The footer only advertises F2 when it can toggle a server."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()
            help_text = screen.query_one(".mcp-viewer-help", Static)

            assert "F2 disable/enable" in _widget_text(help_text)

            await pilot.press("down")
            assert "F2 disable/enable" not in _widget_text(help_text)

            await pilot.press("up")
            assert "F2 disable/enable" in _widget_text(help_text)


class TestModuleLevelHelpers:
    """Unit tests for module-level helper functions in mcp_viewer."""

    # --- _format_prop_type ---

    # --- _sort_servers_for_display ---

    def test_sort_servers_is_stable_within_groups(self) -> None:
        """Original config order is preserved among same-priority servers."""
        from deepagents_code.tui.widgets.mcp_viewer import _sort_servers_for_display

        info = [
            MCPServerInfo(name="ok-a", transport="stdio"),
            MCPServerInfo(
                name="unauth-a",
                transport="http",
                status="unauthenticated",
                error="login required",
            ),
            MCPServerInfo(name="ok-b", transport="stdio"),
            MCPServerInfo(
                name="unauth-b",
                transport="http",
                status="unauthenticated",
                error="login required",
            ),
            MCPServerInfo(
                name="err-a",
                transport="sse",
                status="error",
                error="boom",
            ),
        ]
        ordered = _sort_servers_for_display(info)
        assert [s.name for s in ordered] == [
            "unauth-a",
            "unauth-b",
            "ok-a",
            "ok-b",
            "err-a",
        ]

    def test_sort_servers_no_unauthenticated_preserves_order(self) -> None:
        """When no server is unauthenticated, the order is identical."""
        from deepagents_code.tui.widgets.mcp_viewer import _sort_servers_for_display

        info = _sample_info()
        ordered = _sort_servers_for_display(info)
        assert [s.name for s in ordered] == [s.name for s in info]

    def test_sort_servers_all_unauthenticated_preserves_order(self) -> None:
        """When every server is unauthenticated, config order is preserved."""
        from deepagents_code.tui.widgets.mcp_viewer import _sort_servers_for_display

        info = [
            MCPServerInfo(
                name="alpha",
                transport="http",
                status="unauthenticated",
                error="login required",
            ),
            MCPServerInfo(
                name="bravo",
                transport="http",
                status="unauthenticated",
                error="login required",
            ),
        ]
        ordered = _sort_servers_for_display(info)
        assert [s.name for s in ordered] == ["alpha", "bravo"]

    # --- _visible_tools_for ---

    def test_visible_tools_for_zero_tool_server_name_match_returns_none(self) -> None:
        """Server name match on a zero-tool server returns None (no stub header)."""
        from deepagents_code.tui.widgets.mcp_viewer import _visible_tools_for

        info = MCPServerInfo(
            name="github",
            transport="http",
            status="unauthenticated",
            error="Run: dcode mcp login github",
        )
        # Server name matches but tools=() → or None collapses to None
        assert _visible_tools_for(info, ["github"]) is None

    def test_visible_tools_for_zero_tool_server_no_tokens_returns_empty_tuple(
        self,
    ) -> None:
        """Without a filter, empty-tool servers return () so their header renders."""
        from deepagents_code.tui.widgets.mcp_viewer import _visible_tools_for

        info = MCPServerInfo(
            name="github",
            transport="http",
            status="unauthenticated",
            error="Run: dcode mcp login github",
        )
        assert _visible_tools_for(info, []) == ()
