"""Unit tests for ACP mode behavior in `cli_main`."""

from __future__ import annotations

import argparse
import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_code.main import _preload_session_mcp_server_info, cli_main


def _make_acp_args(**overrides: object) -> argparse.Namespace:
    args = argparse.Namespace(
        acp=True,
        model=None,
        model_params=None,
        profile_override=None,
        agent="agent",
        mcp_config=None,
        no_mcp=False,
        trust_project_mcp=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_acp_mode_loads_tools_and_mcp_and_runs_server() -> None:
    """`--acp` should build the ACP agent with web tools and MCP tools."""
    args = _make_acp_args(
        model_params='{"temperature": 0.2}',
        profile_override='{"max_input_tokens": 4096}',
    )
    model_obj = object()
    model_result = SimpleNamespace(
        model=model_obj,
        provider="anthropic",
        model_name="claude-sonnet-4-6",
        apply_to_settings=MagicMock(),
    )
    server = object()
    mcp_loop = None

    def _run_agent_with_bound_loop(agent_server: object) -> None:
        assert agent_server is server
        assert asyncio.get_running_loop() is mcp_loop

    run_agent = AsyncMock(side_effect=_run_agent_with_bound_loop)
    mcp_manager = SimpleNamespace(cleanup=AsyncMock(return_value=None))
    mcp_tool = object()
    mcp_server_info = [SimpleNamespace(name="docs")]
    fetch_tool = object()
    thread_tool = object()
    search_tool = object()
    acp_project_root = object()
    acp_project_context = SimpleNamespace(
        project_root=acp_project_root,
        user_cwd=object(),
    )
    plugin_configs = ({"mcpServers": {"plugin": {}}},)

    def _resolve_mcp_tools_with_bound_loop(
        *,
        explicit_config_path: str | None,
        no_mcp: bool,
        trust_project_mcp: bool | None,
        project_context: object | None,
        additional_configs: tuple[dict[str, object], ...],
    ) -> tuple[list[object], object, list[SimpleNamespace]]:
        assert explicit_config_path is None
        assert not no_mcp
        assert trust_project_mcp is False
        assert project_context is acp_project_context
        assert additional_configs == plugin_configs
        nonlocal mcp_loop
        mcp_loop = asyncio.get_running_loop()
        return [mcp_tool], mcp_manager, mcp_server_info

    resolve_mcp_tools = AsyncMock(side_effect=_resolve_mcp_tools_with_bound_loop)

    with (
        patch.object(sys, "argv", ["deepagents", "--acp"]),
        patch(
            "deepagents_code.main.check_cli_dependencies",
            side_effect=AssertionError("check_cli_dependencies should be skipped"),
        ),
        patch("deepagents_code.main.parse_args", return_value=args),
        patch("deepagents_code.config.settings", new=SimpleNamespace(has_tavily=True)),
        patch(
            "deepagents_code.config.is_memory_auto_save_enabled", return_value=False
        ) as mock_memory_auto_save,
        patch("deepagents_code.model_config.save_recent_model", return_value=True),
        patch(
            "deepagents_code.config.create_model", return_value=model_result
        ) as mock_create_model,
        patch(
            "deepagents_code.project_utils.ProjectContext.from_user_cwd",
            return_value=acp_project_context,
        ),
        patch(
            "deepagents_code.plugins.adapters.mcp.discover_plugin_mcp_configs",
            return_value=plugin_configs,
        ) as discover_plugin_mcp,
        patch(
            "deepagents_code.mcp_tools.resolve_and_load_mcp_tools", resolve_mcp_tools
        ),
        patch("deepagents_code.tools.fetch_url", new=fetch_tool),
        patch("deepagents_code.tools.get_current_thread_id", new=thread_tool),
        patch("deepagents_code.tools.web_search", new=search_tool),
        patch(
            "deepagents_code.agent.create_cli_agent", return_value=("graph", object())
        ) as mock_create_agent,
        patch(
            "deepagents_acp.server.AgentServerACP", return_value=server
        ) as mock_server_cls,
        patch("acp.run_agent", run_agent),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli_main()

    assert exc_info.value.code == 0
    mock_create_model.assert_called_once_with(
        None,
        extra_kwargs={"temperature": 0.2},
        profile_overrides={"max_input_tokens": 4096},
    )
    resolve_mcp_tools.assert_awaited_once_with(
        explicit_config_path=None,
        no_mcp=False,
        trust_project_mcp=False,
        project_context=acp_project_context,
        additional_configs=plugin_configs,
    )
    discover_plugin_mcp.assert_called_once_with(project_dir=acp_project_root)
    model_result.apply_to_settings.assert_called_once_with()
    mock_create_agent.assert_called_once()
    call_kwargs = mock_create_agent.call_args.kwargs
    assert call_kwargs["model"] is model_obj
    assert call_kwargs["assistant_id"] == "agent"
    assert call_kwargs["tools"] == [fetch_tool, thread_tool, search_tool, mcp_tool]
    assert call_kwargs["mcp_server_info"] is mcp_server_info
    assert call_kwargs["checkpointer"] is not None
    assert call_kwargs["memory_auto_save"] is False
    mock_memory_auto_save.assert_called_once_with()
    mock_server_cls.assert_called_once_with("graph")
    run_agent.assert_awaited_once_with(server)
    mcp_manager.cleanup.assert_awaited_once_with()


def test_acp_mode_omits_web_search_without_tavily() -> None:
    """`--acp` should skip `web_search` when Tavily is not configured."""
    args = _make_acp_args()
    model_obj = object()
    model_result = SimpleNamespace(
        model=model_obj,
        provider="anthropic",
        model_name="claude-sonnet-4-6",
        apply_to_settings=MagicMock(),
    )
    server = object()
    run_agent = AsyncMock(return_value=None)
    fetch_tool = object()
    thread_tool = object()
    search_tool = object()
    resolve_mcp_tools = AsyncMock(return_value=([], None, []))

    with (
        patch.object(sys, "argv", ["deepagents", "--acp"]),
        patch(
            "deepagents_code.main.check_cli_dependencies",
            side_effect=AssertionError("check_cli_dependencies should be skipped"),
        ),
        patch("deepagents_code.main.parse_args", return_value=args),
        patch("deepagents_code.config.settings", new=SimpleNamespace(has_tavily=False)),
        patch("deepagents_code.model_config.save_recent_model", return_value=True),
        patch("deepagents_code.config.create_model", return_value=model_result),
        patch(
            "deepagents_code.mcp_tools.resolve_and_load_mcp_tools", resolve_mcp_tools
        ),
        patch("deepagents_code.tools.fetch_url", new=fetch_tool),
        patch("deepagents_code.tools.get_current_thread_id", new=thread_tool),
        patch("deepagents_code.tools.web_search", new=search_tool),
        patch(
            "deepagents_code.agent.create_cli_agent", return_value=("graph", object())
        ) as mock_create_agent,
        patch("deepagents_acp.server.AgentServerACP", return_value=server),
        patch("acp.run_agent", run_agent),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli_main()

    assert exc_info.value.code == 0
    mock_create_agent.assert_called_once()
    call_kwargs = mock_create_agent.call_args.kwargs
    assert call_kwargs["model"] is model_obj
    assert call_kwargs["assistant_id"] == "agent"
    assert call_kwargs["tools"] == [fetch_tool, thread_tool]
    assert call_kwargs["mcp_server_info"] == []
    assert call_kwargs["checkpointer"] is not None


def test_acp_mode_forwards_allow_fs_tools() -> None:
    """`--acp --allow-fs-tools` forwards the parsed allowlist as `fs_tools`."""
    args = _make_acp_args(allow_fs_tools="ls,read_file")
    model_obj = object()
    model_result = SimpleNamespace(
        model=model_obj,
        provider="anthropic",
        model_name="claude-sonnet-4-6",
        apply_to_settings=MagicMock(),
    )
    server = object()
    run_agent = AsyncMock(return_value=None)
    resolve_mcp_tools = AsyncMock(return_value=([], None, []))

    with (
        patch.object(sys, "argv", ["deepagents", "--acp"]),
        patch(
            "deepagents_code.main.check_cli_dependencies",
            side_effect=AssertionError("check_cli_dependencies should be skipped"),
        ),
        patch("deepagents_code.main.parse_args", return_value=args),
        patch("deepagents_code.config.settings", new=SimpleNamespace(has_tavily=False)),
        patch("deepagents_code.model_config.save_recent_model", return_value=True),
        patch("deepagents_code.config.create_model", return_value=model_result),
        patch(
            "deepagents_code.mcp_tools.resolve_and_load_mcp_tools", resolve_mcp_tools
        ),
        patch("deepagents_code.tools.fetch_url", new=object()),
        patch("deepagents_code.tools.get_current_thread_id", new=object()),
        patch("deepagents_code.tools.web_search", new=object()),
        patch(
            "deepagents_code.agent.create_cli_agent", return_value=("graph", object())
        ) as mock_create_agent,
        patch("deepagents_acp.server.AgentServerACP", return_value=server),
        patch("acp.run_agent", run_agent),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli_main()

    assert exc_info.value.code == 0
    mock_create_agent.assert_called_once()
    assert mock_create_agent.call_args.kwargs["fs_tools"] == ["ls", "read_file"]


def test_acp_mode_forwards_none_allow_fs_tools_by_default() -> None:
    """`--acp` without `--allow-fs-tools` forwards `fs_tools=None` (unrestricted)."""
    args = _make_acp_args()  # no allow_fs_tools override
    model_result = SimpleNamespace(
        model=object(),
        provider="anthropic",
        model_name="claude-sonnet-4-6",
        apply_to_settings=MagicMock(),
    )
    run_agent = AsyncMock(return_value=None)
    resolve_mcp_tools = AsyncMock(return_value=([], None, []))

    with (
        patch.object(sys, "argv", ["deepagents", "--acp"]),
        patch(
            "deepagents_code.main.check_cli_dependencies",
            side_effect=AssertionError("check_cli_dependencies should be skipped"),
        ),
        patch("deepagents_code.main.parse_args", return_value=args),
        patch("deepagents_code.config.settings", new=SimpleNamespace(has_tavily=False)),
        patch("deepagents_code.model_config.save_recent_model", return_value=True),
        patch("deepagents_code.config.create_model", return_value=model_result),
        patch(
            "deepagents_code.mcp_tools.resolve_and_load_mcp_tools", resolve_mcp_tools
        ),
        patch("deepagents_code.tools.fetch_url", new=object()),
        patch("deepagents_code.tools.get_current_thread_id", new=object()),
        patch("deepagents_code.tools.web_search", new=object()),
        patch(
            "deepagents_code.agent.create_cli_agent", return_value=("graph", object())
        ) as mock_create_agent,
        patch("deepagents_acp.server.AgentServerACP", return_value=object()),
        patch("acp.run_agent", run_agent),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli_main()

    assert exc_info.value.code == 0
    mock_create_agent.assert_called_once()
    assert mock_create_agent.call_args.kwargs["fs_tools"] is None


def test_mcp_preload_includes_plugin_configs() -> None:
    """The TUI metadata preload should include enabled plugin MCP servers."""
    project_root = object()
    project_context = SimpleNamespace(project_root=project_root, user_cwd=object())
    plugin_configs = ({"mcpServers": {"plugin": {}}},)
    session_manager = SimpleNamespace(cleanup=AsyncMock(return_value=None))
    server_info = [SimpleNamespace(name="plugin")]
    resolver = AsyncMock(return_value=([], session_manager, server_info))

    with (
        patch(
            "deepagents_code.project_utils.ProjectContext.from_user_cwd",
            return_value=project_context,
        ),
        patch(
            "deepagents_code.plugins.adapters.mcp.discover_plugin_mcp_configs",
            return_value=plugin_configs,
        ) as discover_plugin_mcp,
        patch("deepagents_code.mcp_tools.resolve_and_load_mcp_tools", resolver),
    ):
        result = asyncio.run(
            _preload_session_mcp_server_info(
                mcp_config_path=None,
                no_mcp=False,
                trust_project_mcp=None,
            )
        )

    assert result == server_info
    resolver.assert_awaited_once_with(
        explicit_config_path=None,
        no_mcp=False,
        trust_project_mcp=None,
        project_context=project_context,
        additional_configs=plugin_configs,
    )
    discover_plugin_mcp.assert_called_once_with(project_dir=project_root)
    session_manager.cleanup.assert_awaited_once_with()


def test_non_acp_mode_checks_dependencies_before_parsing() -> None:
    """Non-ACP invocations should still run dependency checks first."""
    with (
        patch.object(sys, "argv", ["deepagents"]),
        patch(
            "deepagents_code.main.check_cli_dependencies", side_effect=SystemExit(7)
        ) as mock_check,
        patch("deepagents_code.main.parse_args") as mock_parse,
        pytest.raises(SystemExit) as exc_info,
    ):
        cli_main()

    assert exc_info.value.code == 7
    mock_check.assert_called_once_with()
    mock_parse.assert_not_called()
