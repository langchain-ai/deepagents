"""Tests for the MCP viewer modal screen."""

import asyncio

import pytest
from textual.app import App, ComposeResult
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

    async def test_reconnect_hint_hidden_when_no_pending(self) -> None:
        """Footer hint omits the `Ctrl+R` chip when nothing is queued."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(
                server_info=_sample_info(),
                pending_reconnect=False,
            )
            app.push_screen(screen)
            await pilot.pause()

            help_widget = screen.query_one(".mcp-viewer-help", Static)
            assert "Ctrl+R" not in _widget_text(help_widget)

    async def test_reconnect_hint_shown_when_pending(self) -> None:
        """Footer hint surfaces `Ctrl+R` when a reconnect is queued."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(
                server_info=_sample_info(),
                pending_reconnect=True,
            )
            app.push_screen(screen)
            await pilot.pause()

            help_widget = screen.query_one(".mcp-viewer-help", Static)
            assert "Ctrl+R reconnect" in _widget_text(help_widget)

    async def test_ctrl_r_dismisses_with_reconnect_sentinel_when_pending(
        self,
    ) -> None:
        """`Ctrl+R` dismisses with the reconnect sentinel when pending."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            outcomes: list[str | None] = []

            def on_dismiss(result: str | None) -> None:
                outcomes.append(result)

            screen = MCPViewerScreen(
                server_info=_sample_info(),
                pending_reconnect=True,
            )
            app.push_screen(screen, on_dismiss)
            await pilot.pause()

            await pilot.press("ctrl+r")
            await pilot.pause()

            assert outcomes == [MCP_VIEWER_RECONNECT_REQUEST]

    async def test_ctrl_r_is_noop_when_not_pending(self) -> None:
        """`Ctrl+R` does nothing when no reconnect is queued."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            outcomes: list[str | None] = []

            def on_dismiss(result: str | None) -> None:
                outcomes.append(result)

            screen = MCPViewerScreen(
                server_info=_sample_info(),
                pending_reconnect=False,
            )
            app.push_screen(screen, on_dismiss)
            await pilot.pause()

            await pilot.press("ctrl+r")
            await pilot.pause()

            assert outcomes == []

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

    async def test_f2_invokes_toggle_callback_without_dismissing(self) -> None:
        """F2 on a server header fires the callback in place; no screen swap."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            outcomes: list[str | None] = []
            toggled: list[str] = []

            def on_dismiss(result: str | None) -> None:
                outcomes.append(result)

            async def on_toggle(server_name: str) -> None:
                toggled.append(server_name)
                await asyncio.sleep(0)  # async signature required by callback protocol

            screen = MCPViewerScreen(
                server_info=_sample_info(), on_toggle_disable=on_toggle
            )
            app.push_screen(screen, on_dismiss)
            await pilot.pause()

            await pilot.press("f2")
            await pilot.pause()

            assert toggled == ["filesystem"]
            # The viewer must stay mounted — no dismiss callback firing
            # means no pop/push flicker.
            assert outcomes == []
            assert app.screen is screen

    async def test_f2_refresh_preserves_list_and_selection(self) -> None:
        """After F2 + in-place refresh, headers and tools still render."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:

            async def on_toggle(server_name: str) -> None:
                # Simulate the app's "disable" path: rebuild the server
                # entry with status="disabled" and patch in place.
                updated = [
                    MCPServerInfo(
                        name=info.name,
                        transport=info.transport,
                        status="disabled" if info.name == server_name else info.status,
                        tools=() if info.name == server_name else info.tools,
                        error=(
                            "Disabled by user (pending reconnect)."
                            if info.name == server_name
                            else None
                        ),
                    )
                    for info in _sample_info()
                ]
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
            assert len(headers) == 2
            # Selection lands on the toggled server.
            selected_widget = screen._row_widgets[screen._selected_index]
            assert isinstance(selected_widget, MCPServerHeaderItem)
            assert selected_widget.server.name == "filesystem"
            # The other server's tools must still render.
            tools = screen.query(".mcp-tool-item")
            tool_text = " ".join(_widget_text(t) for t in tools)
            assert "search" in tool_text

    async def test_f2_patch_preserves_unrelated_widget_identity(self) -> None:
        """In-place patch must NOT re-create widgets for unrelated servers.

        Guards against a regression to the full-rebuild path: the other
        server's header and tool rows must be the same Python instances
        before and after the toggle. A full rebuild would replace them.
        """
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:

            async def on_toggle(server_name: str) -> None:
                updated = [
                    MCPServerInfo(
                        name=info.name,
                        transport=info.transport,
                        status="disabled" if info.name == server_name else info.status,
                        tools=() if info.name == server_name else info.tools,
                        error=(
                            "Disabled by user (pending reconnect)."
                            if info.name == server_name
                            else None
                        ),
                    )
                    for info in _sample_info()
                ]
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

            other_header_before = next(
                w
                for w in screen._row_widgets
                if isinstance(w, MCPServerHeaderItem) and w.server.name == "remote-api"
            )
            other_tool_before = next(
                w
                for w in screen._row_widgets
                if isinstance(w, MCPToolItem) and w.tool_name == "search"
            )

            await pilot.press("f2")
            await pilot.pause()

            other_header_after = next(
                w
                for w in screen._row_widgets
                if isinstance(w, MCPServerHeaderItem) and w.server.name == "remote-api"
            )
            other_tool_after = next(
                w
                for w in screen._row_widgets
                if isinstance(w, MCPToolItem) and w.tool_name == "search"
            )

            # Same Python objects -> no re-mount -> no flicker on this row.
            assert other_header_after is other_header_before
            assert other_tool_after is other_tool_before

    async def test_f2_patch_updates_footer_when_pending_reconnect_changes(
        self,
    ) -> None:
        """Toggling a server on must show `Ctrl+R reconnect` in the footer.

        The footer is mounted once in `_mount_body`; the in-place patch
        must keep it in sync when `pending_reconnect` flips, otherwise
        the user wouldn't see the reconnect hint until the next viewer
        open.
        """
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen_holder: dict[str, MCPViewerScreen] = {}

            async def on_toggle(server_name: str) -> None:
                updated = [
                    MCPServerInfo(
                        name=info.name,
                        transport=info.transport,
                        status="disabled" if info.name == server_name else info.status,
                        tools=() if info.name == server_name else info.tools,
                        error=(
                            "Disabled by user (pending reconnect)."
                            if info.name == server_name
                            else None
                        ),
                    )
                    for info in _sample_info()
                ]
                await screen_holder["screen"].apply_server_disable_toggle(
                    updated,
                    toggled_server=server_name,
                    pending_reconnect=True,
                )

            screen = MCPViewerScreen(
                server_info=_sample_info(),
                pending_reconnect=False,
                on_toggle_disable=on_toggle,
            )
            screen_holder["screen"] = screen
            app.push_screen(screen)
            await pilot.pause()

            help_static = screen.query_one(".mcp-viewer-help", Static)
            assert "Ctrl+R reconnect" not in _widget_text(help_static)

            await pilot.press("f2")
            await pilot.pause()

            assert "Ctrl+R reconnect" in _widget_text(help_static)

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

    async def test_f2_renumbers_indices_so_clicks_resolve_correctly(self) -> None:
        """Every widget's `index` matches its `_row_widgets` position post-F2.

        Click handlers call `screen._move_to(self.index)`; a stale
        index would land selection on the wrong row (or out of bounds).
        """
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:

            async def on_toggle(server_name: str) -> None:
                # Disable: drops the server's tool rows, shrinking the list.
                updated = [
                    MCPServerInfo(
                        name=info.name,
                        transport=info.transport,
                        status="disabled" if info.name == server_name else info.status,
                        tools=() if info.name == server_name else info.tools,
                        error=(
                            "Disabled by user (pending reconnect)."
                            if info.name == server_name
                            else None
                        ),
                    )
                    for info in _sample_info()
                ]
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

            for position, widget in enumerate(screen._row_widgets):
                assert widget.index == position, (
                    f"stale index at position {position}: widget.index={widget.index}"
                )

    async def test_f2_clamps_selected_index_when_list_shrinks(self) -> None:
        """`_selected_index` must clamp into bounds after a shrink.

        Defensive guard: if `_selected_index` ever points past the new
        list end, the next arrow keypress would `IndexError`. Exercises
        the `max(0, len(self._row_widgets) - 1)` clamp directly because
        the F2 happy path leaves selection on the (still-present)
        header.
        """
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            updated = [
                MCPServerInfo(
                    name=info.name,
                    transport=info.transport,
                    status="disabled" if info.name == "filesystem" else info.status,
                    tools=() if info.name == "filesystem" else info.tools,
                    error=(
                        "Disabled by user (pending reconnect)."
                        if info.name == "filesystem"
                        else None
                    ),
                )
                for info in _sample_info()
            ]
            screen._selected_index = 9999  # well past any plausible row count
            await screen.apply_server_disable_toggle(
                updated, toggled_server="filesystem", pending_reconnect=True
            )

            assert 0 <= screen._selected_index < len(screen._row_widgets)

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

    async def test_refresh_from_server_preserves_selected_state(self) -> None:
        """`refresh_from_server` must keep the header's selected class.

        The in-place patch invokes this on the toggled header; if it
        dropped `_selected` (or rendered without `selected=True`), the
        cursor would visually deselect on every F2 even though the
        user's intent was to stay put.
        """
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            header = screen._row_widgets[0]
            assert isinstance(header, MCPServerHeaderItem)
            header.set_selected(True)

            from deepagents_code import theme
            from deepagents_code.config import get_glyphs
            from deepagents_code.tui.widgets.mcp_viewer import (
                _status_color,
                _status_glyph,
            )

            new_server = MCPServerInfo(
                name=header.server.name,
                transport=header.server.transport,
                status="disabled",
                tools=(),
                error="Disabled by user (pending reconnect).",
            )
            glyphs = get_glyphs()
            colors = theme.get_theme_colors(screen)
            header.refresh_from_server(
                new_server,
                _status_glyph(new_server.status, glyphs),
                _status_color(new_server.status, colors),
                (),
                glyphs,
            )

            assert header._selected is True
            assert header.has_class("mcp-header-selected")

    async def test_f2_is_noop_without_callback(self) -> None:
        """Without a callback the screen stays mounted and nothing fires."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            outcomes: list[str | None] = []

            def on_dismiss(result: str | None) -> None:
                outcomes.append(result)

            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen, on_dismiss)
            await pilot.pause()

            await pilot.press("f2")
            await pilot.pause()

            assert outcomes == []

    async def test_f2_is_noop_on_tool_row(self) -> None:
        """F2 only toggles server headers, not individual tools."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            toggled: list[str] = []

            async def on_toggle(server_name: str) -> None:
                toggled.append(server_name)
                await asyncio.sleep(0)

            screen = MCPViewerScreen(
                server_info=_sample_info(), on_toggle_disable=on_toggle
            )
            app.push_screen(screen)
            await pilot.pause()

            await pilot.press("down")
            await pilot.press("f2")
            await pilot.pause()

            assert isinstance(screen._row_widgets[screen._selected_index], MCPToolItem)
            assert toggled == []

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
        """Arrow / Tab navigation walks rows (headers + tools).

        Up/Down step through every row including server headers. Tab /
        Shift+Tab skip headers as a faster cross-tool nav. Vim-style
        `j`/`k` bindings are absent so they can be typed into the filter
        Input — see `test_letter_keys_type_into_filter`.
        """
        from textual.widgets import Input

        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            # `_sample_info()` rows: [filesystem header, read_file, write_file,
            # remote-api header, search] — 5 rows.
            assert len(screen._row_widgets) == 5
            assert isinstance(screen._row_widgets[0], MCPServerHeaderItem)
            assert isinstance(screen._row_widgets[1], MCPToolItem)
            assert isinstance(screen._row_widgets[3], MCPServerHeaderItem)

            # First row (filesystem header) starts selected.
            assert screen._selected_index == 0
            assert screen._row_widgets[0].has_class("mcp-header-selected")

            # Down walks every row in order.
            for expected in (1, 2, 3, 4):
                await pilot.press("down")
                await pilot.pause()
                assert screen._selected_index == expected

            # No wrap-around at the end of the list.
            await pilot.press("down")
            await pilot.pause()
            assert screen._selected_index == 4

            # Up walks back row by row.
            await pilot.press("up")
            await pilot.pause()
            assert screen._selected_index == 3

            # Tab skips headers — from row 3 (remote-api header), next tool
            # is row 4 (search).
            await pilot.press("tab")
            await pilot.pause()
            assert screen._selected_index == 4

            # Tab from search: no further tool → no-op.
            await pilot.press("tab")
            await pilot.pause()
            assert screen._selected_index == 4

            # Shift+Tab skip-header: search (4) → write_file (2).
            await pilot.press("shift+tab")
            await pilot.pause()
            assert screen._selected_index == 2

            # Shift+Tab again: write_file (2) → read_file (1).
            await pilot.press("shift+tab")
            await pilot.pause()
            assert screen._selected_index == 1

            # Shift+Tab from read_file (1): no prior tool (only header
            # before) → no-op.
            await pilot.press("shift+tab")
            await pilot.pause()
            assert screen._selected_index == 1

            # The filter Input exists and is the focused widget.
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

            # Step from initial header (row 0) to the first tool row.
            await pilot.press("down")
            await pilot.pause()
            widget = screen._row_widgets[screen._selected_index]
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

    async def test_expanded_tool_without_schema_has_no_parameters(self) -> None:
        """Tool with `input_schema=None` shows only name + description."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            await pilot.press("down")
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

            await pilot.press("down")
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

            await pilot.press("down")
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

    async def test_no_wrap_around_at_list_ends(self) -> None:
        """Down past the last row, or Up past the first, are no-ops."""
        # Rows: [0: filesystem header, 1: read_file, 2: write_file,
        #        3: remote-api header, 4: search]
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            assert screen._selected_index == 0
            # Up from the first row (header) stays put.
            await pilot.press("up")
            await pilot.pause()
            assert screen._selected_index == 0
            # Shift+Tab from a header with no prior tool is also a no-op.
            await pilot.press("shift+tab")
            await pilot.pause()
            assert screen._selected_index == 0

            # Walk to the last row (search, row 4) via Down.
            for _ in range(4):
                await pilot.press("down")
                await pilot.pause()
            assert screen._selected_index == 4

            # Down past the last row stays put.
            await pilot.press("down")
            await pilot.pause()
            assert screen._selected_index == 4

            # Tab past the last tool also stays put.
            await pilot.press("tab")
            await pilot.pause()
            assert screen._selected_index == 4

    async def test_tab_always_jumps_even_inside_tall_tool(self) -> None:
        """Tab / Shift+Tab unconditionally jump, ignoring viewport visibility."""
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

            # Step from header to big and expand it.
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            assert screen._selected_index == 1

            # One Tab must jump to "next" (row 2) even though Down would
            # only scroll one row.
            await pilot.press("tab")
            await pilot.pause()
            assert screen._selected_index == 2

            # Shift+Tab back to "big" (row 1).
            await pilot.press("shift+tab")
            await pilot.pause()
            assert screen._selected_index == 1

    async def test_server_header_rows_are_selectable(self) -> None:
        """Up/Down lands the cursor on server header rows too (R10)."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            # First row is the filesystem header.
            assert isinstance(screen._row_widgets[0], MCPServerHeaderItem)
            assert screen._row_widgets[0].has_class("mcp-header-selected")

            # Step Down to the remote-api header (row 3 — past 2 tool rows).
            for _ in range(3):
                await pilot.press("down")
                await pilot.pause()
            assert screen._selected_index == 3
            assert isinstance(screen._row_widgets[3], MCPServerHeaderItem)
            assert screen._row_widgets[3].has_class("mcp-header-selected")
            assert not screen._row_widgets[0].has_class("mcp-header-selected")

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

            # No wrap; no tools to Tab to.
            await pilot.press("down")
            await pilot.pause()
            assert screen._selected_index == 1
            await pilot.press("tab")
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

            await pilot.press("enter")
            await pilot.pause()

            assert dismissed_with == ["github"]

    async def test_error_header_hides_error_until_enter(self) -> None:
        """Error headers show an affordance, not the raw exception text."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_mixed_status_info())
            app.push_screen(screen)
            await pilot.pause()

            # Attention-needed states are floated to the top: github(0),
            # notion(1), filesystem(2), read_file tool(3), broken(4).
            for _ in range(4):
                await pilot.press("down")
                await pilot.pause()
            header = screen._row_widgets[4]
            assert isinstance(header, MCPServerHeaderItem)
            assert header.server.name == "broken"

            text = _widget_text(header)
            assert "Enter for details" in text
            assert "Connection refused" not in text

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

    async def test_enter_on_awaiting_reconnect_header_is_noop(self) -> None:
        """Activating a pending-reconnect header does not restart login."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            dismissed_with: list[str | None] = []

            def on_dismiss(result: str | None) -> None:
                dismissed_with.append(result)

            screen = MCPViewerScreen(server_info=_mixed_status_info())
            app.push_screen(screen, on_dismiss)
            await pilot.pause()

            await pilot.press("down")
            await pilot.pause()
            assert screen._row_widgets[1]._server.name == "notion"  # ty: ignore

            await pilot.press("enter")
            await pilot.pause()

            assert dismissed_with == []

    async def test_enter_on_server_header_is_noop(self) -> None:
        """Enter on a server header does not expand or crash."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            assert isinstance(screen._row_widgets[0], MCPServerHeaderItem)
            assert screen._selected_index == 0

            # Press Enter on the header — must be a no-op and must not
            # affect any tool's expansion state.
            await pilot.press("enter")
            await pilot.pause()
            assert screen._selected_index == 0
            assert all(not tool._expanded for tool in screen._tool_widgets)

    async def test_ctrl_e_toggles_only_tool_rows(self) -> None:
        """`Ctrl+E` expands tool rows but never affects server headers."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_sample_info())
            app.push_screen(screen)
            await pilot.pause()

            # Initial cursor is on a header — Ctrl+E should still expand
            # every visible tool, regardless of what is selected.
            assert isinstance(screen._row_widgets[0], MCPServerHeaderItem)
            await pilot.press("ctrl+e")
            await pilot.pause()
            assert all(tool._expanded for tool in screen._tool_widgets)

            # Header rows do not have an _expanded attribute; verify they
            # were left untouched (no AttributeError raised).
            for w in screen._row_widgets:
                if isinstance(w, MCPServerHeaderItem):
                    assert not hasattr(w, "_expanded") or not getattr(
                        w, "_expanded", False
                    )

    async def test_filter_matches_only_tool_and_server_names(self) -> None:
        """Filter ignores descriptions, param names, and transport."""
        from textual.widgets import Input

        info = [
            MCPServerInfo(
                name="srv",
                transport="stdio",
                tools=(
                    MCPToolInfo(
                        name="run",
                        description="Run something with a target_path argument",
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
            for query in ("target_path", "argument", "stdio"):
                filter_input.value = query
                await pilot.pause()
                assert screen._tool_widgets == [], f"{query!r} unexpectedly matched"

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
            assert "f2" in text
            assert "ctrl+e" in text
            assert "filter" in text
            assert "esc" in text

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
            # The header now invites in-TUI login instead of telling the
            # user to leave the app and run `dcode mcp login`.
            assert "Enter to log in" in unauth_text

            assert "notion" in pending_text
            assert "ready to load" in pending_text
            assert "Ctrl+R to load tools" in pending_text

            assert "broken" in err_text
            assert "error" in err_text
            assert "Enter for details" in err_text
            assert "Connection refused" not in err_text

            assert "paused" in disabled_text
            assert "disabled" in disabled_text

    def test_status_color_maps_all_states(self) -> None:
        """Unit-level: each status maps to the correct theme color attribute."""
        from deepagents_code import theme
        from deepagents_code.tui.widgets.mcp_viewer import _status_color

        colors = theme.get_theme_colors()
        assert _status_color("ok", colors) == colors.success
        assert _status_color("unauthenticated", colors) == colors.warning
        assert _status_color("awaiting_reconnect", colors) == colors.warning
        assert _status_color("error", colors) == colors.error
        assert _status_color("disabled", colors) == colors.muted

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
            assert "Enter for details" in text
            assert "JSON decode failed" not in text

            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, MCPServerErrorScreen)
            assert "JSON decode failed" in _widget_text(
                app.screen.query_one(".mcp-error-text", Static)
            )
            # No tools to render — only the header line and the help footer.
            assert len(screen.query(".mcp-tool-item")) == 0

    async def test_click_on_selected_error_header_opens_modal(self) -> None:
        """Clicking an already-selected error header opens the detail modal."""
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            screen = MCPViewerScreen(server_info=_mixed_status_info())
            app.push_screen(screen)
            await pilot.pause()

            # broken (error status) floats to index 4; see ordering note above.
            # Move the cursor onto it so the click hits an already-selected row,
            # the precondition the `on_click` error branch guards on.
            for _ in range(4):
                await pilot.press("down")
                await pilot.pause()
            header = screen._row_widgets[4]
            assert isinstance(header, MCPServerHeaderItem)
            assert header.server.name == "broken"
            assert header._selected

            await pilot.click(header)
            await pilot.pause()
            assert isinstance(app.screen, MCPServerErrorScreen)
            assert "Connection refused" in _widget_text(
                app.screen.query_one(".mcp-error-text", Static)
            )

    async def test_error_modal_copy_success_notifies(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The copy action writes the error and raises an info toast."""
        copied: list[str] = []

        def _fake_copy(_app: App[None], text: str) -> tuple[bool, str | None]:
            copied.append(text)
            return True, None

        monkeypatch.setattr(
            "deepagents_code.tui.widgets.mcp_viewer.copy_text_to_clipboard",
            _fake_copy,
        )

        server = MCPServerInfo(
            name="broken",
            transport="sse",
            status="error",
            error="Connection refused",
        )
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            app.push_screen(MCPServerErrorScreen(server))
            await pilot.pause()

            await pilot.press("c")
            await pilot.pause()

            assert copied == ["Connection refused"]
            toast = _latest_notification(app)
            assert toast is not None
            assert toast.message == "MCP error copied"
            assert toast.severity == "information"

    async def test_error_modal_copy_failure_notifies(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed copy surfaces a warning toast with the backend reason."""

        def _fake_copy(_app: App[None], _text: str) -> tuple[bool, str | None]:
            return False, "no clipboard backend"

        monkeypatch.setattr(
            "deepagents_code.tui.widgets.mcp_viewer.copy_text_to_clipboard",
            _fake_copy,
        )

        server = MCPServerInfo(
            name="broken",
            transport="sse",
            status="error",
            error="Connection refused",
        )
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            app.push_screen(MCPServerErrorScreen(server))
            await pilot.pause()

            await pilot.press("c")
            await pilot.pause()

            toast = _latest_notification(app)
            assert toast is not None
            assert toast.message == "Failed to copy MCP error: no clipboard backend"
            assert toast.severity == "warning"

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

    async def test_error_modal_falls_back_when_error_missing(self) -> None:
        """A server without error text renders the placeholder message."""
        # An `ok` server is the only way to obtain `error=None` — the
        # `MCPServerInfo` invariant forbids a non-`ok` status without an
        # error — so this exercises the modal's defensive fallback.
        server = MCPServerInfo(name="filesystem", transport="stdio")
        assert server.error is None
        app = MCPViewerTestApp()
        async with app.run_test() as pilot:
            app.push_screen(MCPServerErrorScreen(server))
            await pilot.pause()

            body = app.screen.query_one(".mcp-error-text", Static)
            assert "No error details were reported." in _widget_text(body)

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


class TestModuleLevelHelpers:
    """Unit tests for module-level helper functions in mcp_viewer."""

    # --- _format_prop_type ---

    def test_format_prop_type_string(self) -> None:
        """Plain string type renders as-is."""
        from deepagents_code.tui.widgets.mcp_viewer import _format_prop_type

        assert _format_prop_type("string") == "string"

    def test_format_prop_type_none_returns_any(self) -> None:
        """None type renders as 'any'."""
        from deepagents_code.tui.widgets.mcp_viewer import _format_prop_type

        assert _format_prop_type(None) == "any"

    def test_format_prop_type_list_nullable_joins_with_pipe(self) -> None:
        """List type (nullable pattern) joins with '|'."""
        from deepagents_code.tui.widgets.mcp_viewer import _format_prop_type

        assert _format_prop_type(["string", "null"]) == "string|null"

    def test_format_prop_type_single_item_list(self) -> None:
        """Single-element list renders without pipe."""
        from deepagents_code.tui.widgets.mcp_viewer import _format_prop_type

        assert _format_prop_type(["integer"]) == "integer"

    def test_format_prop_type_empty_list_returns_any(self) -> None:
        """Empty list renders as 'any'."""
        from deepagents_code.tui.widgets.mcp_viewer import _format_prop_type

        assert _format_prop_type([]) == "any"

    def test_format_prop_type_empty_string_returns_any(self) -> None:
        """Empty string renders as 'any'."""
        from deepagents_code.tui.widgets.mcp_viewer import _format_prop_type

        assert _format_prop_type("") == "any"

    # --- _sort_servers_for_display ---

    def test_sort_servers_floats_attention_needed_to_top(self) -> None:
        """Actionable servers move ahead of `ok` and `error` servers."""
        from deepagents_code.tui.widgets.mcp_viewer import _sort_servers_for_display

        info = _mixed_status_info()
        ordered = _sort_servers_for_display(info)
        assert [s.name for s in ordered] == [
            "github",
            "notion",
            "filesystem",
            "broken",
            "paused",
        ]

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

    def test_sort_servers_empty_list(self) -> None:
        """Empty input yields an empty list."""
        from deepagents_code.tui.widgets.mcp_viewer import _sort_servers_for_display

        assert _sort_servers_for_display([]) == []

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

    def test_visible_tools_for_no_tokens_returns_all(self) -> None:
        """Empty token list returns all tools (no filter)."""
        from deepagents_code.tui.widgets.mcp_viewer import _visible_tools_for

        info = MCPServerInfo(
            name="fs",
            transport="stdio",
            tools=(MCPToolInfo(name="read", description=""),),
        )
        assert _visible_tools_for(info, []) == info.tools

    def test_visible_tools_for_server_name_match_returns_all_tools(self) -> None:
        """Token matching server name returns all its tools."""
        from deepagents_code.tui.widgets.mcp_viewer import _visible_tools_for

        info = MCPServerInfo(
            name="filesystem",
            transport="stdio",
            tools=(
                MCPToolInfo(name="read", description=""),
                MCPToolInfo(name="write", description=""),
            ),
        )
        assert _visible_tools_for(info, ["filesystem"]) == info.tools

    def test_visible_tools_for_tool_name_match_returns_subset(self) -> None:
        """Token matching tool names returns only matching tools."""
        from deepagents_code.tui.widgets.mcp_viewer import _visible_tools_for

        read_tool = MCPToolInfo(name="read_file", description="")
        write_tool = MCPToolInfo(name="write_file", description="")
        info = MCPServerInfo(
            name="fs", transport="stdio", tools=(read_tool, write_tool)
        )
        result = _visible_tools_for(info, ["read"])
        assert result == (read_tool,)

    def test_visible_tools_for_no_match_returns_none(self) -> None:
        """No matching server or tool name returns None."""
        from deepagents_code.tui.widgets.mcp_viewer import _visible_tools_for

        info = MCPServerInfo(
            name="fs",
            transport="stdio",
            tools=(MCPToolInfo(name="read", description=""),),
        )
        assert _visible_tools_for(info, ["zzz"]) is None

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
