"""Unit tests for ACP mode behavior in `cli_main`."""

from __future__ import annotations

import argparse
import asyncio
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_code.main import _build_acp_model_options, cli_main

if TYPE_CHECKING:
    from collections.abc import Callable


class _Mode(Protocol):
    """Small protocol for ACP mode entries in tests."""

    id: str


class _ModeState(Protocol):
    """Small protocol for ACP mode state assertions in tests."""

    current_mode_id: str
    available_modes: list[_Mode]


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


def test_acp_model_options_filter_missing_credential_providers() -> None:
    """ACP model selector should not offer providers that cannot start."""
    from deepagents_code.model_config import (
        ProviderAuthSource,
        ProviderAuthState,
        ProviderAuthStatus,
    )

    def _auth_status(provider: str) -> ProviderAuthStatus:
        state = (
            ProviderAuthState.MISSING
            if provider == "anthropic"
            else ProviderAuthState.CONFIGURED
        )
        source = (
            ProviderAuthSource.ENV if state is ProviderAuthState.CONFIGURED else None
        )
        return ProviderAuthStatus(
            state=state,
            provider=provider,
            env_var=f"{provider.upper()}_API_KEY",
            source=source,
        )

    with (
        patch("deepagents_code.model_config.load_recent_models", return_value=[]),
        patch(
            "deepagents_code.model_config.get_available_models",
            return_value={
                "openai": ["gpt-5.5"],
                "anthropic": ["claude-sonnet-4-6"],
            },
        ),
        patch("deepagents_code.model_config.get_provider_auth_status", _auth_status),
    ):
        options = _build_acp_model_options("openai:gpt-5.5")

    assert [option["value"] for option in options] == ["openai:gpt-5.5"]
    assert [option["name"] for option in options] == ["gpt-5.5"]


def test_acp_mode_loads_tools_and_mcp_and_runs_server() -> None:
    """`--acp` should expose selectors and build a factory-backed agent."""
    from deepagents_code.model_config import (
        ProviderAuthSource,
        ProviderAuthState,
        ProviderAuthStatus,
    )

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
    captured_factory: Callable[[object], object] | None = None

    def _auth_status(provider: str) -> ProviderAuthStatus:
        return ProviderAuthStatus(
            state=ProviderAuthState.CONFIGURED,
            provider=provider,
            source=ProviderAuthSource.ENV,
        )

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

    def _build_acp_server(
        agent_factory: Callable[[object], object],
        *,
        modes: _ModeState,
        models: list[dict[str, str]],
    ) -> object:
        nonlocal captured_factory
        assert callable(agent_factory)
        captured_factory = agent_factory
        assert modes.current_mode_id == "agent"
        assert [mode.id for mode in modes.available_modes] == ["agent", "plan", "ask"]
        assert [model["value"] for model in models] == [
            "anthropic:claude-sonnet-4-6",
            "anthropic:claude-haiku-4",
            "openai:gpt-5.2",
        ]
        assert [model["name"] for model in models] == [
            "claude-sonnet-4-6",
            "claude-haiku-4",
            "gpt-5.2",
        ]
        return server

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
            "deepagents_code.model_config.get_available_models",
            return_value={
                "anthropic": ["claude-sonnet-4-6", "claude-haiku-4"],
                "openai": ["gpt-5.2"],
            },
        ),
        patch(
            "deepagents_code.model_config.load_recent_models",
            return_value=["anthropic:claude-haiku-4"],
        ),
        patch("deepagents_code.model_config.get_provider_auth_status", _auth_status),
        patch("deepagents_code.model_config.save_recent_model", return_value=True),
        patch("deepagents_code.model_config.touch_recent_model", return_value=True),
        patch(
            "deepagents_code.config.create_model", return_value=model_result
        ) as mock_create_model,
        patch(
            "deepagents_code.mcp_tools.resolve_and_load_mcp_tools", resolve_mcp_tools
        ),
        patch("deepagents_code.tools.fetch_url", new=fetch_tool),
        patch("deepagents_code.tools.get_current_thread_id", new=thread_tool),
        patch("deepagents_code.tools.web_search", new=search_tool),
        patch(
            "deepagents_code.agent.create_cli_agent", return_value=("graph", object())
        ) as mock_create_agent,
        patch("deepagents_code.agent.get_system_prompt", return_value="base prompt"),
        patch(
            "deepagents_acp.server.AgentServerACP", side_effect=_build_acp_server
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

    assert captured_factory is not None
    graph = captured_factory(
        SimpleNamespace(cwd="/tmp/acp", mode="plan", model="openai:gpt-5.2")
    )
    assert graph == "graph"
    assert mock_create_model.call_args_list[-1].args == ("openai:gpt-5.2",)
    call_kwargs = mock_create_agent.call_args.kwargs
    assert call_kwargs["model"] is model_obj
    assert call_kwargs["assistant_id"] == "agent"
    assert call_kwargs["tools"] == [fetch_tool, thread_tool, search_tool, mcp_tool]
    assert "ACP Planning Mode" in call_kwargs["system_prompt"]
    assert call_kwargs["mcp_server_info"] is mcp_server_info
    assert call_kwargs["checkpointer"] is not None
    assert call_kwargs["cwd"] == "/tmp/acp"
    assert call_kwargs["enable_shell"] is True
    assert call_kwargs["permissions"] is None

    ask_graph = captured_factory(
        SimpleNamespace(cwd="/tmp/acp", mode="ask", model="openai:gpt-5.2")
    )
    assert ask_graph == "graph"
    ask_call_kwargs = mock_create_agent.call_args.kwargs
    assert "ACP Ask Mode" in ask_call_kwargs["system_prompt"]
    assert ask_call_kwargs["enable_shell"] is False
    [rule] = ask_call_kwargs["permissions"]
    assert rule.operations == ["write"]
    assert rule.paths == ["/**"]
    assert rule.mode == "deny"
    assert mock_create_agent.call_count == 2
    mock_server_cls.assert_called_once()
    run_agent.assert_awaited_once_with(server)
    mcp_manager.cleanup.assert_awaited_once_with()


def test_acp_mode_omits_web_search_without_tavily() -> None:
    """`--acp` should skip `web_search` when Tavily is not configured."""
    from deepagents_code.model_config import (
        ProviderAuthSource,
        ProviderAuthState,
        ProviderAuthStatus,
    )

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
    captured_factory: Callable[[object], object] | None = None

    def _auth_status(provider: str) -> ProviderAuthStatus:
        return ProviderAuthStatus(
            state=ProviderAuthState.CONFIGURED,
            provider=provider,
            source=ProviderAuthSource.ENV,
        )

    def _build_acp_server(
        agent_factory: Callable[[object], object],
        *,
        modes: _ModeState,
        models: list[dict[str, str]],
    ) -> object:
        nonlocal captured_factory
        assert callable(agent_factory)
        captured_factory = agent_factory
        assert modes.current_mode_id == "agent"
        assert models[0]["value"] == "anthropic:claude-sonnet-4-6"
        return server

    with (
        patch.object(sys, "argv", ["deepagents", "--acp"]),
        patch(
            "deepagents_code.main.check_cli_dependencies",
            side_effect=AssertionError("check_cli_dependencies should be skipped"),
        ),
        patch("deepagents_code.main.parse_args", return_value=args),
        patch("deepagents_code.config.settings", new=SimpleNamespace(has_tavily=False)),
        patch(
            "deepagents_code.model_config.get_available_models",
            return_value={"anthropic": ["claude-sonnet-4-6"]},
        ),
        patch("deepagents_code.model_config.load_recent_models", return_value=[]),
        patch("deepagents_code.model_config.get_provider_auth_status", _auth_status),
        patch("deepagents_code.model_config.save_recent_model", return_value=True),
        patch("deepagents_code.model_config.touch_recent_model", return_value=True),
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
        patch("deepagents_acp.server.AgentServerACP", side_effect=_build_acp_server),
        patch("acp.run_agent", run_agent),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli_main()

    assert exc_info.value.code == 0
    assert captured_factory is not None
    graph = captured_factory(SimpleNamespace(cwd="/tmp/acp", mode="agent", model=None))
    assert graph == "graph"
    mock_create_agent.assert_called_once()
    call_kwargs = mock_create_agent.call_args.kwargs
    assert call_kwargs["model"] is model_obj
    assert call_kwargs["assistant_id"] == "agent"
    assert call_kwargs["tools"] == [fetch_tool, thread_tool]
    assert call_kwargs["system_prompt"] is None
    assert call_kwargs["mcp_server_info"] == []
    assert call_kwargs["checkpointer"] is not None


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
