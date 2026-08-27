"""Tests for tool enumeration behind `dcode tools list`."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, PropertyMock, patch

from deepagents_code.config import Credentials
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

if TYPE_CHECKING:
    import pytest

# Core tools the agent always binds, independent of optional integrations.
_CORE_BUILT_IN = {
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


class TestCollectBuiltInTools:
    """Tests for enumerating built-in tools from the compiled agent."""

    def test_enumeration_survives_a_policy_blocked_subagent_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Listing tools must not raise because a subagent names a blocked model.

        `collect_built_in_tools` compiles a graph with a placeholder model purely
        to read its bound tool node -- nothing is ever invoked. Enforcing
        `models.allowed` here turned `dcode tools list` into a crash and made
        `/tools` report a false reason.

        The subagent's own model is still resolved by the SDK during assembly,
        which is pre-existing behavior and needs a credential for the provider
        it names. A placeholder key isolates this test to the policy question
        rather than the environment's credentials. Dcode's runtime model factory
        must not eagerly reconstruct it because that factory enforces policy.
        """
        from deepagents_code.model_config import ModelConfig

        monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder-not-a-real-key")

        policy = ModelConfig(
            allowed_models=("anthropic:allowed",),
            allowed_models_source="managed config",
        )
        monkeypatch.setattr(
            ModelConfig,
            "load",
            classmethod(lambda _cls, _path=None: policy),
        )
        monkeypatch.setattr(
            "deepagents_code.agent.list_subagents",
            lambda **_kwargs: [
                {
                    "name": "blocked",
                    "description": "Names a model the policy forbids",
                    "system_prompt": "Help.",
                    "model": "openai:blocked",
                    "path": "/agents/blocked/AGENTS.md",
                }
            ],
        )

        with patch("deepagents_code.agent._resolve_retry_owned_model") as resolve:
            names = {tool.name for tool in collect_built_in_tools()}

        assert names >= _CORE_BUILT_IN
        resolve.assert_not_called()

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

    def test_backstop_surfaces_and_logs_when_disallowed_tool_leaks_through(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """If the SDK ever stops narrowing, the listing must not silently lie.

        Normally the SDK's `FilesystemMiddleware` omits disallowed tools from the
        node, so the check is a no-op. Here we simulate that guarantee breaking
        (a disallowed `write_file` reaches enumeration) and assert the backstop
        (a) keeps it in the listing — because the agent really does expose it, so
        hiding it would misreport a restricted surface over an unrestricted
        agent — and (b) logs an error so the discrepancy is visible.
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

        # The leaked tool is surfaced, not scrubbed: the listing reflects the
        # agent's real (unrestricted) tools rather than a false restricted view.
        assert "write_file" in names
        assert {"read_file", "task"} <= names
        assert any(
            "allowlist backstop detected" in record.getMessage()
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

    def test_web_search_present_with_tavily(self) -> None:
        with patch.object(
            Credentials, "has_tavily", new_callable=PropertyMock, return_value=True
        ):
            names = {tool.name for tool in collect_built_in_tools()}
        assert "web_search" in names

    def test_web_search_absent_without_tavily(self) -> None:
        with patch.object(
            Credentials, "has_tavily", new_callable=PropertyMock, return_value=False
        ):
            names = {tool.name for tool in collect_built_in_tools()}
        assert "web_search" not in names


class TestTodoToolNotBound:
    """Todos are opt-in in the SDK, so dcode binds no `write_todos` by default."""


class TestCollectToolsFromAgent:
    """Tests for inspecting the tool node of an already-running local graph."""


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


class TestBuildCatalogFromServerInfo:
    """Tests for the TUI entry point that avoids `asyncio.run` discovery."""


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
