"""Tests for tool enumeration behind `dcode tools list`."""

from __future__ import annotations

from unittest.mock import AsyncMock, PropertyMock, patch

from deepagents_code.config import Settings
from deepagents_code.mcp_tools import MCPServerInfo, MCPToolInfo
from deepagents_code.tool_catalog import (
    BUILT_IN_GROUP,
    ToolEntry,
    _first_line,
    collect_built_in_tools,
    collect_mcp_tool_groups,
    collect_tool_groups,
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


class TestCollectMcpToolGroups:
    """Tests for MCP tool grouping."""

    def test_groups_per_server_and_skips_toolless(self) -> None:
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
        ]
        with patch(
            "deepagents_code.tool_catalog._load_mcp_server_info",
            new=AsyncMock(return_value=servers),
        ):
            groups = collect_mcp_tool_groups()
        assert len(groups) == 1
        group = groups[0]
        assert group.label == "docs"
        assert group.source == "mcp"
        assert group.tools == (
            ToolEntry(name="search_docs", description="Search the docs"),
        )

    def test_discovery_failure_returns_no_groups(self) -> None:
        with patch(
            "deepagents_code.tool_catalog._load_mcp_server_info",
            new=AsyncMock(side_effect=RuntimeError("no config")),
        ):
            assert collect_mcp_tool_groups() == []


class TestCollectToolGroups:
    """Tests for the combined built-in + MCP assembly."""

    def test_built_in_group_first_and_mcp_optional(self) -> None:
        with patch("deepagents_code.tool_catalog.collect_mcp_tool_groups") as mock_mcp:
            groups = collect_tool_groups(include_mcp=False)
        mock_mcp.assert_not_called()
        assert groups[0].label == BUILT_IN_GROUP
        assert groups[0].source == "built-in"
        assert len(groups) == 1
