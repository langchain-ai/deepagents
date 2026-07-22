"""Tests for tool enumeration behind `dcode tools list`."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest

from deepagents_code._env_vars import EXPERIMENTAL
from deepagents_code.config import Settings
from deepagents_code.mcp_tools import MCPServerInfo, MCPToolInfo
from deepagents_code.tool_catalog import (
    BUILT_IN_GROUP,
    ToolEntry,
    ToolGroup,
    UnavailableServer,
    _CatalogModel,
    _first_line,
    _load_mcp_server_info,
    build_catalog_from_server_info,
    collect_built_in_tools,
    collect_catalog,
    collect_mcp_catalog,
    collect_tools_from_agent,
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

    def test_respects_filesystem_allowlist(self) -> None:
        """The catalog listing is narrowed to an explicit allowlist.

        Scope: this validates the `/tools` display contract for
        `collect_built_in_tools`, NOT runtime `FilesystemMiddleware` enforcement
        (covered in `test_agent.py`). The narrowing is produced by the SDK
        middleware, which omits disallowed tools from the node entirely; the
        `collect_built_in_tools` post-filter is a defensive backstop over the
        same result. Either way the listing must exclude the disallowed names.
        """
        names = {
            tool.name for tool in collect_built_in_tools(fs_tools=["ls", "read_file"])
        }
        assert {"ls", "read_file", "task"} <= names
        assert (
            not {
                "write_file",
                "edit_file",
                "delete",
                "glob",
                "grep",
                "execute",
            }
            & names
        )

    def test_all_lists_every_filesystem_tool(self) -> None:
        """`fs_tools="all"` skips filtering, so every filesystem tool is listed."""
        names = {tool.name for tool in collect_built_in_tools(fs_tools="all")}
        assert {
            "ls",
            "read_file",
            "write_file",
            "edit_file",
            "delete",
            "glob",
            "grep",
            "execute",
        } <= names

    def test_backstop_strips_and_logs_when_disallowed_tool_leaks_through(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """If the SDK ever stops narrowing, the post-filter must not silently lie.

        Normally the SDK's `FilesystemMiddleware` omits disallowed tools from the
        node, so the post-filter is a no-op. Here we simulate that guarantee
        breaking (a disallowed `write_file` reaches enumeration) and assert the
        backstop both (a) removes it from the listing and (b) logs an error,
        rather than silently reshaping the display over an unrestricted agent.
        """
        leaked = [
            ToolEntry(name="read_file", description="read"),
            ToolEntry(name="write_file", description="write"),
            ToolEntry(name="task", description="delegate"),
        ]
        with (
            patch(
                "deepagents_code.agent.create_cli_agent",
                return_value=(SimpleNamespace(), None),
            ),
            patch(
                "deepagents_code.tool_catalog.collect_tools_from_agent",
                return_value=leaked,
            ),
            caplog.at_level("ERROR", logger="deepagents_code.tool_catalog"),
        ):
            names = {
                tool.name for tool in collect_built_in_tools(fs_tools=["read_file"])
            }

        assert "write_file" not in names
        assert {"read_file", "task"} <= names
        assert any(
            "allowlist backstop removed" in record.getMessage()
            and "write_file" in record.getMessage()
            for record in caplog.records
        )

    def test_backstop_silent_when_allowlist_already_applied(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The backstop must stay quiet when enumeration already respects it."""
        applied = [
            ToolEntry(name="read_file", description="read"),
            ToolEntry(name="task", description="delegate"),
        ]
        with (
            patch(
                "deepagents_code.agent.create_cli_agent",
                return_value=(SimpleNamespace(), None),
            ),
            patch(
                "deepagents_code.tool_catalog.collect_tools_from_agent",
                return_value=applied,
            ),
            caplog.at_level("ERROR", logger="deepagents_code.tool_catalog"),
        ):
            collect_built_in_tools(fs_tools=["read_file"])

        assert not caplog.records

    def test_filesystem_tool_names_match_sdk(self) -> None:
        """`_FILESYSTEM_TOOL_NAMES` must not drift from the SDK's `FsToolName`."""
        from typing import get_args

        from deepagents import FsToolName

        from deepagents_code.tool_catalog import _FILESYSTEM_TOOL_NAMES

        assert set(get_args(FsToolName)) == _FILESYSTEM_TOOL_NAMES

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
            tools = collect_built_in_tools(
                assistant_id="custom-agent", fs_tools=["ls", "read_file"]
            )
        assert tools == [ToolEntry(name="task", description="Run a subagent")]
        create.assert_called_once()
        assert create.call_args.kwargs["assistant_id"] == "custom-agent"
        assert create.call_args.kwargs["fs_tools"] == ["ls", "read_file"]

    def test_raises_when_compiled_agent_not_inspectable(self) -> None:
        # A compiled agent whose graph does not expose the conventional tool
        # node must fail loudly (documented `Raises: RuntimeError`) rather than
        # silently returning an empty list — `collect_tools_from_agent` returns
        # `None`, which this function turns into the raise.
        agent = SimpleNamespace()
        with (
            patch(
                "deepagents_code.agent.create_cli_agent",
                return_value=(agent, None),
            ),
            pytest.raises(RuntimeError, match="does not expose"),
        ):
            collect_built_in_tools()


class TestExperimentalTodoRemoval:
    """`DEEPAGENTS_CODE_EXPERIMENTAL` drops the SDK `write_todos` tool."""

    def test_write_todos_bound_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(EXPERIMENTAL, raising=False)
        names = {tool.name for tool in collect_built_in_tools()}
        assert "write_todos" in names

    def test_write_todos_removed_when_experimental(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(EXPERIMENTAL, "1")
        names = {tool.name for tool in collect_built_in_tools()}
        assert "write_todos" not in names
        # Only the todo tool is dropped; the rest of the core set stays bound.
        assert names >= _CORE_BUILT_IN - {"write_todos"}


class TestCollectToolsFromAgent:
    """Tests for inspecting the tool node of an already-running local graph."""

    def test_reads_bound_tools(self) -> None:
        tool_node = SimpleNamespace(
            tools_by_name={
                "custom_search": SimpleNamespace(
                    description="Search custom data\nAdditional details"
                )
            }
        )
        agent = SimpleNamespace(nodes={"tools": SimpleNamespace(bound=tool_node)})

        assert collect_tools_from_agent(agent) == [
            ToolEntry(name="custom_search", description="Search custom data")
        ]

    def test_returns_empty_for_local_agent_without_tool_node(self) -> None:
        from langchain.agents import create_agent

        agent = create_agent(model=_CatalogModel(), tools=[])

        assert collect_tools_from_agent(agent) == []

    def test_returns_none_for_remote_agent(self) -> None:
        agent = SimpleNamespace(url="https://example.test")

        assert collect_tools_from_agent(agent) is None

    def test_returns_none_when_tool_node_shape_unexpected(self) -> None:
        # A "tools" node exists but its `bound` object lacks a `tools_by_name`
        # mapping — a LangGraph internal-shape drift. Reported as uninspectable
        # (`None`), not as a validly-empty tool set (`[]`).
        agent = SimpleNamespace(
            nodes={"tools": SimpleNamespace(bound=SimpleNamespace())}
        )

        assert collect_tools_from_agent(agent) is None

    def test_skips_non_string_names_and_defaults_missing_description(self) -> None:
        tool_node = SimpleNamespace(
            tools_by_name={
                "ok": SimpleNamespace(description="Fine"),
                123: SimpleNamespace(description="dropped: non-str name"),
                "no_desc": SimpleNamespace(description=None),
            }
        )
        agent = SimpleNamespace(nodes={"tools": SimpleNamespace(bound=tool_node)})

        assert collect_tools_from_agent(agent) == [
            ToolEntry(name="ok", description="Fine"),
            ToolEntry(name="no_desc", description=""),
        ]


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
            UnavailableServer(name="off", status="disabled", detail=""),
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

    def test_pending_reenable_guidance_is_preserved(self) -> None:
        # `pending_reconnect` (not the guidance text) is what keeps the detail:
        # a plainly-disabled server with the same text would be blanked.
        servers = [
            MCPServerInfo(
                name="notion",
                transport="http",
                status="disabled",
                error="Re-enabled — press Ctrl+R to load.",
                pending_reconnect=True,
            ),
        ]

        groups, unavailable = split_mcp_server_info(servers)

        assert groups == []
        assert unavailable == [
            UnavailableServer(
                name="notion",
                status="disabled",
                detail="Re-enabled — press Ctrl+R to load.",
            )
        ]

    def test_disabled_server_detail_blanked_without_pending_reconnect(self) -> None:
        # Same guidance text, but no `pending_reconnect`: the detail is dropped
        # so the renderers fall back to the generic "disabled by user" label.
        servers = [
            MCPServerInfo(
                name="notion",
                transport="http",
                status="disabled",
                error="Re-enabled — press Ctrl+R to load.",
            ),
        ]

        _, unavailable = split_mcp_server_info(servers)

        assert unavailable == [
            UnavailableServer(name="notion", status="disabled", detail="")
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

    def test_passes_plugin_configs_and_project_dir(self) -> None:
        """Tool catalog discovery should include project-scoped plugin MCP."""
        project_root = object()
        project_context = SimpleNamespace(
            project_root=project_root,
            user_cwd=object(),
        )
        plugin_configs = ({"mcpServers": {"plugin": {}}},)
        loader = AsyncMock(return_value=([], None, []))
        with (
            patch(
                "deepagents_code.project_utils.ProjectContext.from_user_cwd",
                return_value=project_context,
            ),
            patch(
                "deepagents_code.plugins.adapters.mcp.discover_plugin_mcp_configs",
                return_value=plugin_configs,
            ) as discover_plugin_mcp,
            patch("deepagents_code.mcp_tools.resolve_and_load_mcp_tools", new=loader),
        ):
            result = asyncio.run(
                _load_mcp_server_info(mcp_config_path=None, trust_project_mcp=None)
            )

        assert result == []
        loader.assert_awaited_once_with(
            explicit_config_path=None,
            no_mcp=False,
            trust_project_mcp=None,
            project_context=project_context,
            additional_configs=plugin_configs,
        )
        discover_plugin_mcp.assert_called_once_with(project_dir=project_root)

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
        built_in.assert_called_once_with(
            assistant_id="agent", enable_interpreter=False, fs_tools=None
        )
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
                fs_tools=["ls", "read_file"],
                include_mcp=True,
                mcp_config_path="/tmp/mcp.json",
            )
        built_in.assert_called_once_with(
            assistant_id="custom-agent",
            enable_interpreter=False,
            fs_tools=["ls", "read_file"],
        )
        # Built-in group stays first; MCP groups follow.
        assert catalog.groups[0].label == BUILT_IN_GROUP
        assert catalog.groups[-1].label == "docs"
        assert catalog.unavailable == (
            UnavailableServer(name="broken", status="error", detail="boom"),
        )
