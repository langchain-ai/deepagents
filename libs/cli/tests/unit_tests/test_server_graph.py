"""Tests for server graph MCP loading behavior."""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from deepagents_cli._env_vars import SERVER_ENV_PREFIX
from deepagents_cli._server_config import ServerConfig
from deepagents_cli.mcp_tools import MCPSessionManager


def _import_fresh_server_graph() -> ModuleType:
    """Import `deepagents_cli.server_graph` from a clean module state."""
    sys.modules.pop("deepagents_cli.server_graph", None)
    return importlib.import_module("deepagents_cli.server_graph")


def _module_with_attrs(name: str, **attrs: object) -> ModuleType:
    """Create a module stub with dynamically assigned attributes."""
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class TestServerGraph:
    """Tests for server-mode graph bootstrap."""

    def test_auto_discovery_loads_mcp_without_explicit_config(self) -> None:
        """Server mode should auto-discover MCP configs when no path is passed."""
        graph_obj = object()
        model_obj = object()
        fetch_tool = object()
        mcp_tool = object()
        mcp_server_info = [SimpleNamespace(name="docs")]
        create_cli_agent = MagicMock(return_value=(graph_obj, object()))
        agent_module = _module_with_attrs(
            "deepagents_cli.agent",
            DEFAULT_AGENT_NAME="agent",
            create_cli_agent=create_cli_agent,
            load_async_subagents=MagicMock(return_value=None),
        )

        model_result = SimpleNamespace(
            model=model_obj,
            apply_to_settings=MagicMock(),
        )
        config_module = _module_with_attrs(
            "deepagents_cli.config",
            create_model=MagicMock(return_value=model_result),
            settings=SimpleNamespace(
                has_tavily=False,
                reload_from_environment=MagicMock(),
            ),
        )

        tools_module = _module_with_attrs(
            "deepagents_cli.tools",
            fetch_url=fetch_tool,
            web_search=object(),
        )

        resolve_mcp_tools = AsyncMock(return_value=([mcp_tool], None, mcp_server_info))
        mcp_module = _module_with_attrs(
            "deepagents_cli.mcp_tools",
            resolve_and_load_mcp_tools=resolve_mcp_tools,
        )

        # Build env from ServerConfig to exercise the same serialization
        # path the real CLI uses.
        config = ServerConfig(no_mcp=False)
        env_overrides = {}
        for suffix, value in config.to_env().items():
            if value is not None:
                env_overrides[f"{SERVER_ENV_PREFIX}{suffix}"] = value

        with (
            patch.dict(os.environ, env_overrides, clear=False),
            patch.dict(
                sys.modules,
                {
                    "deepagents_cli.agent": agent_module,
                    "deepagents_cli.config": config_module,
                    "deepagents_cli.tools": tools_module,
                    "deepagents_cli.mcp_tools": mcp_module,
                },
            ),
            patch(
                "deepagents_cli.project_utils.get_server_project_context",
                return_value=None,
            ),
        ):
            for suffix in (
                "MCP_CONFIG_PATH",
                "TRUST_PROJECT_MCP",
                "CWD",
                "PROJECT_ROOT",
            ):
                os.environ.pop(f"{SERVER_ENV_PREFIX}{suffix}", None)

            module = _import_fresh_server_graph()

        resolve_mcp_tools.assert_awaited_once_with(
            explicit_config_path=None,
            no_mcp=False,
            trust_project_mcp=None,
            project_context=None,
        )
        create_cli_agent.assert_called_once_with(
            model=model_obj,
            assistant_id="agent",
            tools=[fetch_tool, mcp_tool],
            sandbox=None,
            sandbox_type=None,
            system_prompt=None,
            interactive=True,
            auto_approve=False,
            interrupt_shell_only=False,
            shell_allow_list=None,
            enable_ask_user=False,
            enable_memory=True,
            enable_skills=True,
            enable_shell=True,
            mcp_server_info=mcp_server_info,
            cwd=None,
            project_context=None,
            async_subagents=None,
        )
        assert module.graph is graph_obj


class TestBuildToolsMCPSessionLifetime:
    """Regression tests for the ClosedResourceError bug in MCP tool loading.

    When `_build_tools` used `asyncio.run()` to load MCP tools, the temporary
    event loop was torn down after loading, closing any anyio streams bound to
    it. Stdio MCP sessions (which hold persistent subprocess streams) were
    closed before tool invocation, causing `ClosedResourceError`.

    The fix has two parts:
    1. Remote (HTTP/SSE) tools now use per-call sessions (no persistent stream).
    2. Stdio session managers are stored at module level so they outlive
       `asyncio.run()`.
    """

    def test_stdio_session_manager_stored_at_module_level(self) -> None:
        """Stdio session manager must be held at module level after _build_tools.

        Stdio MCP servers require a persistent subprocess session. If the
        MCPSessionManager is garbage-collected when `asyncio.run()` exits,
        the subprocess streams are closed and subsequent tool calls fail.
        Storing the manager at module level keeps it alive for the server
        process lifetime.
        """
        cleanup_called = False

        async def fake_cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True

        manager = MCPSessionManager()
        manager.cleanup = fake_cleanup  # type: ignore[method-assign]

        mcp_tool = MagicMock(name="stdio-tool")

        resolve_mcp = AsyncMock(return_value=([mcp_tool], manager, []))

        from deepagents_cli._server_config import ServerConfig
        from deepagents_cli.server_graph import _build_tools

        config = ServerConfig(no_mcp=False)

        import deepagents_cli.server_graph as sg

        with patch("deepagents_cli.mcp_tools.resolve_and_load_mcp_tools", resolve_mcp):
            tools, _info = _build_tools(config, project_context=None)

        assert mcp_tool in tools
        assert not cleanup_called, (
            "Session manager must NOT be cleaned up inside _build_tools"
        )
        assert sg._mcp_session_manager is manager, (
            "Stdio session manager must be stored at module level"
        )
