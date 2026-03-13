"""Unit tests for ACP mode behavior in `cli_main`."""

from __future__ import annotations

import argparse
import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_cli.main import cli_main


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
    http_tool = object()
    fetch_tool = object()
    search_tool = object()

    def _resolve_mcp_tools_with_bound_loop(
        *,
        explicit_config_path: str | None,
        no_mcp: bool,
        trust_project_mcp: bool | None,
    ) -> tuple[list[object], object, list[SimpleNamespace]]:
        assert explicit_config_path is None
        assert not no_mcp
        assert trust_project_mcp is False
        nonlocal mcp_loop
        mcp_loop = asyncio.get_running_loop()
        return [mcp_tool], mcp_manager, mcp_server_info

    resolve_mcp_tools = AsyncMock(side_effect=_resolve_mcp_tools_with_bound_loop)

    with (
        patch.object(sys, "argv", ["deepagents", "--acp"]),
        patch(
            "deepagents_cli.main.check_cli_dependencies",
            side_effect=AssertionError("check_cli_dependencies should be skipped"),
        ),
        patch("deepagents_cli.main.parse_args", return_value=args),
        patch("deepagents_cli.config.settings", new=SimpleNamespace(has_tavily=True)),
        patch("deepagents_cli.model_config.save_recent_model", return_value=True),
        patch(
            "deepagents_cli.config.create_model", return_value=model_result
        ) as mock_create_model,
        patch("deepagents_cli.mcp_tools.resolve_and_load_mcp_tools", resolve_mcp_tools),
        patch("deepagents_cli.tools.http_request", new=http_tool),
        patch("deepagents_cli.tools.fetch_url", new=fetch_tool),
        patch("deepagents_cli.tools.web_search", new=search_tool),
        patch(
            "deepagents_cli.agent.create_cli_agent", return_value=("graph", object())
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
    )
    model_result.apply_to_settings.assert_called_once_with()
    mock_create_agent.assert_called_once()
    call_kwargs = mock_create_agent.call_args.kwargs
    assert call_kwargs["model"] is model_obj
    assert call_kwargs["assistant_id"] == "agent"
    assert call_kwargs["tools"] == [http_tool, fetch_tool, search_tool, mcp_tool]
    assert call_kwargs["mcp_server_info"] is mcp_server_info
    assert call_kwargs["checkpointer"] is not None
    assert call_kwargs["enable_memory"] is False
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
    http_tool = object()
    fetch_tool = object()
    search_tool = object()
    resolve_mcp_tools = AsyncMock(return_value=([], None, []))

    with (
        patch.object(sys, "argv", ["deepagents", "--acp"]),
        patch(
            "deepagents_cli.main.check_cli_dependencies",
            side_effect=AssertionError("check_cli_dependencies should be skipped"),
        ),
        patch("deepagents_cli.main.parse_args", return_value=args),
        patch("deepagents_cli.config.settings", new=SimpleNamespace(has_tavily=False)),
        patch("deepagents_cli.model_config.save_recent_model", return_value=True),
        patch("deepagents_cli.config.create_model", return_value=model_result),
        patch("deepagents_cli.mcp_tools.resolve_and_load_mcp_tools", resolve_mcp_tools),
        patch("deepagents_cli.tools.http_request", new=http_tool),
        patch("deepagents_cli.tools.fetch_url", new=fetch_tool),
        patch("deepagents_cli.tools.web_search", new=search_tool),
        patch(
            "deepagents_cli.agent.create_cli_agent", return_value=("graph", object())
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
    assert call_kwargs["tools"] == [http_tool, fetch_tool]
    assert call_kwargs["mcp_server_info"] == []
    assert call_kwargs["checkpointer"] is not None
    assert call_kwargs["enable_memory"] is False


def test_non_acp_mode_checks_dependencies_before_parsing() -> None:
    """Non-ACP invocations should still run dependency checks first."""
    with (
        patch.object(sys, "argv", ["deepagents"]),
        patch(
            "deepagents_cli.main.check_cli_dependencies", side_effect=SystemExit(7)
        ) as mock_check,
        patch("deepagents_cli.main.parse_args") as mock_parse,
        pytest.raises(SystemExit) as exc_info,
    ):
        cli_main()

    assert exc_info.value.code == 7
    mock_check.assert_called_once_with()
    mock_parse.assert_not_called()
