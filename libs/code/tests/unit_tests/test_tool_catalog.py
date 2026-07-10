"""Tests for tool enumeration behind `dcode tools list`."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, PropertyMock, patch

from deepagents_code.config import Settings
from deepagents_code.mcp_tools import MCPServerInfo, MCPToolInfo
from deepagents_code.tool_catalog import (
    BUILT_IN_GROUP,
    ToolEntry,
    ToolGroup,
    UnavailableServer,
    _first_line,
    _load_mcp_server_info,
    build_catalog_from_server_info,
    collect_built_in_tools,
    collect_catalog,
    collect_mcp_catalog,
    split_mcp_server_info,
)

# Core tools the agent always binds, independent of optional integrations.
_CORE_BUILT_IN = {
    "write_todos",
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "delete",
    "glob",
    "grep",
    "execute",
    "task",
    "ask_user",
    "fetch_url",
    "get_current_thread_id",
}


class TestFirstLine:
    """Tests for `_first_line` description normalization."""

    def test_returns_first_non_empty_line_collapsed(self) -> None:
        assert _first_line("\n  Hello   world \n\nmore") == "Hello world"

    def test_empty_input(self) -> None:
        assert _first_line(None) == ""
        assert _first_line("   \n  ") == ""


class TestCollectBuiltInTools:
    """Tests for enumerating built-in tools from the compiled agent."""

    def test_includes_core_tools(self) -> None:
        tools = collect_built_in_tools()
        names = {tool.name for tool in tools}
        assert names >= _CORE_BUILT_IN
        # Every entry carries a non-empty, single-line description.
        for tool in tools:
            assert tool.description
            assert "\n" not in tool.description

    def test_web_search_present_with_tavily(self) -> None:
        with patch.object(
            Settings, "has_tavily", new_callable=PropertyMock, return_value=True
        ):
            names = {tool.name for tool in collect_built_in_tools()}
        assert "web_search" in names

    def test_web_search_absent_without_tavily(self) -> None:
        with patch.object(
            Settings, "has_tavily", new_callable=PropertyMock, return_value=False
        ):
            names = {tool.name for tool in collect_built_in_tools()}
        assert "web_search" not in names

    def test_interpreter_toggles_js_eval(self) -> None:
        without = {tool.name for tool in collect_built_in_tools()}
        assert "js_eval" not in without
        with_interp = {
            tool.name for tool in collect_built_in_tools(enable_interpreter=True)
        }
        assert "js_eval" in with_interp

    def test_forwards_assistant_id_to_agent_compilation(self) -> None:
        tool_node = SimpleNamespace(
            tools_by_name={"task": SimpleNamespace(description="Run a subagent")}
        )
        agent = SimpleNamespace(nodes={"tools": SimpleNamespace(bound=tool_node)})
        with patch(
            "deepagents_code.agent.create_cli_agent",
            return_value=(agent, None),
        ) as create:
            tools = collect_built_in_tools(assistant_id="custom-agent")
        assert tools == [ToolEntry(name="task", description="Run a subagent")]
        create.assert_called_once()
        assert create.call_args.kwargs["assistant_id"] == "custom-agent"


class TestCollectMcpCatalog:
    """Tests for MCP discovery: grouping tools and surfacing broken servers."""

    def test_groups_ok_servers_and_surfaces_unavailable(self) -> None:
        servers = [
            MCPServerInfo(
                name="docs",
                transport="http",
                tools=(
                    MCPToolInfo(name="search_docs", description="Search the docs\nX"),
                ),
                status="ok",
            ),
            MCPServerInfo(
                name="broken",
                transport="http",
                status="error",
                error="boom",
            ),
            MCPServerInfo(
                name="needslogin",
                transport="http",
                status="unauthenticated",
                error="run login",
            ),
        ]
        loader = AsyncMock(return_value=servers)
        with patch("deepagents_code.tool_catalog._load_mcp_server_info", new=loader):
            groups, unavailable, mcp_error = collect_mcp_catalog(
                mcp_config_path="/tmp/mcp.json",
                trust_project_mcp=True,
            )
        assert mcp_error is None
        assert len(groups) == 1
        group = groups[0]
        assert group.label == "docs"
        assert group.source == "mcp"
        assert group.tools == (
            ToolEntry(name="search_docs", description="Search the docs"),
        )
        # Non-ok servers are reported, not dropped, so the omission is explained.
        assert unavailable == [
            UnavailableServer(name="broken", status="error", detail="boom"),
            UnavailableServer(
                name="needslogin", status="unauthenticated", detail="run login"
            ),
        ]
        # MCP options are forwarded to discovery unchanged.
        loader.assert_awaited_once_with(
            mcp_config_path="/tmp/mcp.json", trust_project_mcp=True
        )

    def test_ok_server_without_tools_is_neither_group_nor_unavailable(self) -> None:
        servers = [MCPServerInfo(name="empty", transport="http", status="ok")]
        with patch(
            "deepagents_code.tool_catalog._load_mcp_server_info",
            new=AsyncMock(return_value=servers),
        ):
            groups, unavailable, mcp_error = collect_mcp_catalog()
        assert groups == []
        assert unavailable == []
        assert mcp_error is None

    def test_disabled_and_awaiting_reconnect_servers_are_surfaced(self) -> None:
        # `disabled` and `awaiting_reconnect` share the `!= "ok"` branch with
        # error/unauthenticated; lock the contract for the full non-ok set.
        # (`MCPServerInfo` requires a reason for any non-ok status.)
        servers = [
            MCPServerInfo(
                name="off",
                transport="unknown",
                status="disabled",
                error="turned off via /mcp",
            ),
            MCPServerInfo(
                name="pending",
                transport="http",
                status="awaiting_reconnect",
                error="reconnecting after login",
            ),
        ]
        with patch(
            "deepagents_code.tool_catalog._load_mcp_server_info",
            new=AsyncMock(return_value=servers),
        ):
            groups, unavailable, mcp_error = collect_mcp_catalog()
        assert groups == []
        assert mcp_error is None
        assert unavailable == [
            UnavailableServer(
                name="off", status="disabled", detail="turned off via /mcp"
            ),
            UnavailableServer(
                name="pending",
                status="awaiting_reconnect",
                detail="reconnecting after login",
            ),
        ]

    def test_discovery_failure_returns_generic_error_without_leaking(self) -> None:
        with patch(
            "deepagents_code.tool_catalog._load_mcp_server_info",
            new=AsyncMock(side_effect=RuntimeError("secret /path/mcp.json boom")),
        ):
            groups, unavailable, mcp_error = collect_mcp_catalog()
        assert groups == []
        assert unavailable == []
        # Generic message only — raw exception text must not leak to output.
        assert mcp_error == "MCP discovery failed; showing built-in tools only."
        assert "secret" not in mcp_error
        assert "/path/mcp.json" not in mcp_error


class TestSplitMcpServerInfo:
    """Tests for the pure server-info splitter shared by CLI and `/tools`."""

    def test_ok_server_becomes_group(self) -> None:
        servers = [
            MCPServerInfo(
                name="docs",
                transport="http",
                tools=(
                    MCPToolInfo(name="search_docs", description="Search the docs\nX"),
                ),
                status="ok",
            ),
        ]
        groups, unavailable = split_mcp_server_info(servers)
        assert groups == [
            ToolGroup(
                label="docs",
                source="mcp",
                tools=(ToolEntry(name="search_docs", description="Search the docs"),),
            )
        ]
        assert unavailable == []

    def test_non_ok_server_becomes_unavailable(self) -> None:
        servers = [
            MCPServerInfo(
                name="broken", transport="http", status="error", error="boom"
            ),
        ]
        groups, unavailable = split_mcp_server_info(servers)
        assert groups == []
        assert unavailable == [
            UnavailableServer(name="broken", status="error", detail="boom")
        ]

    def test_ok_server_without_tools_is_dropped(self) -> None:
        servers = [MCPServerInfo(name="empty", transport="http", status="ok")]
        groups, unavailable = split_mcp_server_info(servers)
        assert groups == []
        assert unavailable == []


class TestBuildCatalogFromServerInfo:
    """Tests for the TUI entry point that avoids `asyncio.run` discovery."""

    def test_built_in_first_then_live_mcp_no_error(self) -> None:
        built_in = [ToolEntry(name="read_file", description="Read a file")]
        servers = [
            MCPServerInfo(
                name="docs",
                transport="http",
                tools=(MCPToolInfo(name="search_docs", description="Search"),),
                status="ok",
            ),
            MCPServerInfo(
                name="broken", transport="http", status="error", error="boom"
            ),
        ]
        catalog = build_catalog_from_server_info(built_in, servers)
        assert catalog.groups[0].label == BUILT_IN_GROUP
        assert catalog.groups[0].source == "built-in"
        assert catalog.groups[0].tools == (
            ToolEntry(name="read_file", description="Read a file"),
        )
        assert catalog.groups[-1].label == "docs"
        assert catalog.unavailable == (
            UnavailableServer(name="broken", status="error", detail="boom"),
        )
        # Discovery is never attempted here, so there is no discovery error.
        assert catalog.mcp_error is None

    def test_does_not_run_mcp_discovery(self) -> None:
        # The whole point of this path is to avoid `asyncio.run`-based discovery,
        # which cannot run inside a live event loop.
        with patch("deepagents_code.tool_catalog._load_mcp_server_info") as loader:
            build_catalog_from_server_info([], [])
        loader.assert_not_called()

    def test_empty_inputs_yield_only_built_in_group(self) -> None:
        catalog = build_catalog_from_server_info([], [])
        assert len(catalog.groups) == 1
        assert catalog.groups[0].label == BUILT_IN_GROUP
        assert catalog.groups[0].tools == ()
        assert catalog.unavailable == ()


class TestLoadMcpServerInfo:
    """Tests for `_load_mcp_server_info` session lifecycle and cwd handling."""

    def test_cleans_up_session_manager(self) -> None:
        session_manager = AsyncMock()
        server_info = [
            MCPServerInfo(
                name="docs",
                transport="http",
                tools=(MCPToolInfo(name="t", description="d"),),
            )
        ]
        loader = AsyncMock(return_value=([], session_manager, server_info))
        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                return_value=None,
            ),
            patch("deepagents_code.mcp_tools.resolve_and_load_mcp_tools", new=loader),
        ):
            result = asyncio.run(
                _load_mcp_server_info(mcp_config_path=None, trust_project_mcp=None)
            )
        assert result == server_info
        session_manager.cleanup.assert_awaited_once()

    def test_cleanup_failure_is_swallowed(self) -> None:
        session_manager = AsyncMock()
        session_manager.cleanup.side_effect = RuntimeError("cleanup boom")
        server_info = [
            MCPServerInfo(
                name="docs",
                transport="http",
                tools=(MCPToolInfo(name="t", description="d"),),
            )
        ]
        loader = AsyncMock(return_value=([], session_manager, server_info))
        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                return_value=None,
            ),
            patch("deepagents_code.mcp_tools.resolve_and_load_mcp_tools", new=loader),
        ):
            # A cleanup failure must not mask the return value or propagate.
            result = asyncio.run(
                _load_mcp_server_info(mcp_config_path=None, trust_project_mcp=None)
            )
        assert result == server_info

    def test_cwd_oserror_forwards_none_project_context(self) -> None:
        loader = AsyncMock(return_value=([], None, []))
        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                side_effect=OSError("no cwd"),
            ),
            patch("deepagents_code.mcp_tools.resolve_and_load_mcp_tools", new=loader),
        ):
            result = asyncio.run(
                _load_mcp_server_info(mcp_config_path=None, trust_project_mcp=None)
            )
        assert result == []
        await_args = loader.await_args
        assert await_args is not None
        assert await_args.kwargs["project_context"] is None


class TestCollectCatalog:
    """Tests for the combined built-in + MCP assembly."""

    def test_built_in_group_first_and_mcp_optional(self) -> None:
        with (
            patch(
                "deepagents_code.tool_catalog.collect_built_in_tools",
                return_value=[],
            ) as built_in,
            patch("deepagents_code.tool_catalog.collect_mcp_catalog") as mock_mcp,
        ):
            catalog = collect_catalog(include_mcp=False)
        mock_mcp.assert_not_called()
        built_in.assert_called_once_with(assistant_id="agent", enable_interpreter=False)
        assert len(catalog.groups) == 1
        assert catalog.groups[0].label == BUILT_IN_GROUP
        assert catalog.groups[0].source == "built-in"
        assert catalog.unavailable == ()
        assert catalog.mcp_error is None

    def test_appends_mcp_groups_and_carries_unavailable(self) -> None:
        mcp_groups = [
            ToolGroup(
                label="docs",
                source="mcp",
                tools=(ToolEntry(name="search_docs", description="Search"),),
            )
        ]
        unavailable = [UnavailableServer(name="broken", status="error", detail="boom")]
        with (
            patch(
                "deepagents_code.tool_catalog.collect_mcp_catalog",
                return_value=(mcp_groups, unavailable, None),
            ),
            patch(
                "deepagents_code.tool_catalog.collect_built_in_tools",
                return_value=[],
            ) as built_in,
        ):
            catalog = collect_catalog(
                assistant_id="custom-agent",
                include_mcp=True,
                mcp_config_path="/tmp/mcp.json",
            )
        built_in.assert_called_once_with(
            assistant_id="custom-agent", enable_interpreter=False
        )
        # Built-in group stays first; MCP groups follow.
        assert catalog.groups[0].label == BUILT_IN_GROUP
        assert catalog.groups[-1].label == "docs"
        assert catalog.unavailable == (
            UnavailableServer(name="broken", status="error", detail="boom"),
        )
