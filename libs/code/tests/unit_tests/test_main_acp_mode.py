"""Unit tests for ACP mode behavior in `cli_main`."""

from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import asynccontextmanager
from inspect import signature
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents_acp.server import AgentServerACP

from deepagents_code.main import _preload_session_mcp_server_info, cli_main
from unit_tests.conftest import redirect_managed_config

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Generator
    from pathlib import Path


@pytest.fixture(autouse=True)
def test_acp_checkpointer() -> Generator[SimpleNamespace]:
    checkpointer = SimpleNamespace(setup=AsyncMock(return_value=None))

    @asynccontextmanager
    async def get_checkpointer() -> AsyncIterator[SimpleNamespace]:
        yield checkpointer

    with patch("deepagents_code.sessions.get_checkpointer", get_checkpointer):
        yield checkpointer


def _build_agent_server(server: object) -> Callable[..., object]:
    """Stand in for `AgentServerACP`, exercising the agent factory it is handed."""

    def build(agent_factory: Callable[..., object], **kwargs: object) -> object:
        signature(AgentServerACP).bind(agent_factory, **kwargs)
        agent_factory(SimpleNamespace(cwd="/tmp", model=None))
        return server

    return build


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
        auto_classifier_model=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_acp_mode_rejects_auto_classifier_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ACP must reject an authorization setting it cannot apply."""
    args = _make_acp_args(auto_classifier_model="openai:gpt-5.5-mini")

    with (
        patch.object(
            sys,
            "argv",
            ["deepagents", "--acp", "--auto-classifier-model", "openai:gpt-5.5-mini"],
        ),
        patch("deepagents_code.main.parse_args", return_value=args),
        patch("deepagents_code.main._resolve_agent_arg") as resolve_agent,
        pytest.raises(SystemExit) as exc_info,
    ):
        cli_main()

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "--auto-classifier-model requires Auto mode in ACP mode" in err
    resolve_agent.assert_not_called()


def test_acp_mode_rejects_auto_classifier_model_in_yolo(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """YOLO must not silently drop the classifier model.

    `create_cli_agent` only installs `AutoModeHITLMiddleware` when
    `auto_mode_enabled` is true; in ACP that flag follows `auto`, not `yolo`,
    so a classifier supplied alongside `--yolo` would never be consulted.
    """
    args = _make_acp_args(yolo=True, auto_classifier_model="openai:gpt-5.5-mini")

    with (
        patch.object(
            sys,
            "argv",
            [
                "deepagents",
                "--acp",
                "--yolo",
                "--auto-classifier-model",
                "openai:gpt-5.5-mini",
            ],
        ),
        patch("deepagents_code.main.parse_args", return_value=args),
        patch(
            "deepagents_code.approval_mode.has_yolo_acknowledgement",
            return_value=True,
        ),
        patch("deepagents_code.main._resolve_agent_arg") as resolve_agent,
        pytest.raises(SystemExit) as exc_info,
    ):
        cli_main()

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "--auto-classifier-model requires Auto mode in ACP mode" in err
    resolve_agent.assert_not_called()


def test_acp_mode_rejects_unacknowledged_yolo_without_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _make_acp_args(yolo=True)

    with (
        patch.object(sys, "argv", ["deepagents", "--acp", "--yolo"]),
        patch("deepagents_code.main.parse_args", return_value=args),
        patch(
            "deepagents_code.approval_mode.has_yolo_acknowledgement",
            return_value=False,
        ),
        patch("deepagents_code.main._resolve_agent_arg") as resolve_agent,
        pytest.raises(SystemExit) as exc_info,
    ):
        cli_main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "acknowledge YOLO" in captured.err
    resolve_agent.assert_not_called()


def test_acp_mode_loads_tools_and_mcp_and_runs_server(
    test_acp_checkpointer: SimpleNamespace,
) -> None:
    """`--acp --yolo` should build the persistent ACP agent unrestricted."""
    args = _make_acp_args(
        model_params='{"temperature": 0.2}',
        profile_override='{"max_input_tokens": 4096}',
        yolo=True,
    )
    model_obj = object()
    model_result = SimpleNamespace(
        model=model_obj,
        provider="anthropic",
        model_name="claude-sonnet-4-6",
        apply_to_settings=MagicMock(),
        model_retries=5,
        cli_max_retries=None,
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
        patch.object(sys, "argv", ["deepagents", "--acp", "--yolo"]),
        patch(
            "deepagents_code.approval_mode.has_yolo_acknowledgement", return_value=True
        ),
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
            "deepagents_acp.server.AgentServerACP",
            side_effect=_build_agent_server(server),
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
        cli_max_retries=None,
    )
    resolve_mcp_tools.assert_awaited_once_with(
        explicit_config_path=None,
        no_mcp=False,
        trust_project_mcp=False,
        project_context=acp_project_context,
        additional_configs=plugin_configs,
    )
    discover_plugin_mcp.assert_called_once_with(project_dir=acp_project_root)
    assert model_result.apply_to_settings.call_count == 2
    mock_create_agent.assert_called_once()
    call_kwargs = mock_create_agent.call_args.kwargs
    assert call_kwargs["model"] is model_obj
    assert call_kwargs["assistant_id"] == "agent"
    assert call_kwargs["tools"] == [fetch_tool, thread_tool, search_tool, mcp_tool]
    assert call_kwargs["mcp_server_info"] is mcp_server_info
    assert call_kwargs["checkpointer"] is test_acp_checkpointer
    assert call_kwargs["cwd"] == "/tmp"
    assert call_kwargs["project_context"] is acp_project_context
    assert call_kwargs["auto_approve"] is True
    assert call_kwargs["auto_mode_enabled"] is False
    assert call_kwargs["memory_auto_save"] is False
    mock_memory_auto_save.assert_called_once_with()
    test_acp_checkpointer.setup.assert_awaited_once_with()
    assert mock_server_cls.call_args.kwargs["models"][0] == {
        "value": "anthropic:claude-sonnet-4-6",
        "name": "anthropic:claude-sonnet-4-6",
    }
    assert mock_server_cls.call_args.kwargs["load_sessions"] is True
    run_agent.assert_awaited_once_with(server)
    mcp_manager.cleanup.assert_awaited_once_with()


def test_acp_mode_auto_forwards_classifier_and_store() -> None:
    args = _make_acp_args(
        auto_approve=True,
        auto_classifier_model="openai:gpt-5.5-mini",
    )
    model_result = SimpleNamespace(
        model=object(),
        provider="openai",
        model_name="gpt-5.5",
        apply_to_settings=MagicMock(),
        model_retries=5,
        cli_max_retries=None,
    )
    server = object()
    auto_server = MagicMock(return_value=server)
    run_agent = AsyncMock(return_value=None)

    def build_auto_server(
        agent_factory: Callable[..., object], **kwargs: object
    ) -> object:
        auto_server(agent_factory, **kwargs)
        agent_factory(SimpleNamespace(cwd="/tmp", model=None))
        return server

    with (
        patch.object(sys, "argv", ["deepagents", "--acp", "--auto-approve"]),
        patch("deepagents_code.main.parse_args", return_value=args),
        patch("deepagents_code.config.settings", new=SimpleNamespace(has_tavily=False)),
        patch("deepagents_code.model_config.save_recent_model", return_value=True),
        patch("deepagents_code.config.create_model", return_value=model_result),
        patch(
            "deepagents_code.mcp_tools.resolve_and_load_mcp_tools",
            new=AsyncMock(return_value=([], None, [])),
        ),
        patch("deepagents_code.tools.fetch_url", new=object()),
        patch("deepagents_code.tools.get_current_thread_id", new=object()),
        patch("deepagents_code.tools.web_search", new=object()),
        patch(
            "deepagents_code.agent.create_cli_agent", return_value=("graph", object())
        ) as create_agent,
        patch("deepagents_code.acp.AgentServerACP", side_effect=build_auto_server),
        patch("acp.run_agent", run_agent),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli_main()

    assert exc_info.value.code == 0
    kwargs = create_agent.call_args.kwargs
    assert kwargs["auto_mode_enabled"] is True
    assert kwargs["auto_approve"] is False
    assert kwargs["auto_classifier_model"] == "openai:gpt-5.5-mini"
    assert kwargs["store"] is not None
    auto_server.assert_called_once()
    assert auto_server.call_args.kwargs["store"] is kwargs["store"]
    assert auto_server.call_args.kwargs["load_sessions"] is True


def test_acp_mode_omits_web_search_without_tavily() -> None:
    """`--acp` should skip `web_search` when Tavily is not configured."""
    args = _make_acp_args()
    model_obj = object()
    model_result = SimpleNamespace(
        model=model_obj,
        provider="anthropic",
        model_name="claude-sonnet-4-6",
        apply_to_settings=MagicMock(),
        model_retries=5,
        cli_max_retries=None,
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
        patch(
            "deepagents_acp.server.AgentServerACP",
            side_effect=_build_agent_server(server),
        ),
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
        model_retries=5,
        cli_max_retries=None,
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
        patch(
            "deepagents_acp.server.AgentServerACP",
            side_effect=_build_agent_server(server),
        ),
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
        model_retries=5,
        cli_max_retries=None,
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
        patch(
            "deepagents_acp.server.AgentServerACP",
            side_effect=_build_agent_server(object()),
        ),
        patch("acp.run_agent", run_agent),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli_main()

    assert exc_info.value.code == 0
    mock_create_agent.assert_called_once()
    assert mock_create_agent.call_args.kwargs["fs_tools"] is None
    assert mock_create_agent.call_args.kwargs["recursion_limit"] is None


def test_acp_mode_forwards_recursion_limit() -> None:
    """`--acp --recursion-limit` forwards the effective CLI value.

    ACP builds in the parent, but uses the same boundary helper as TUI and
    headless launches, so this pins the serialized value too.
    """
    args = _make_acp_args(recursion_limit=3000)
    model_result = SimpleNamespace(
        model=object(),
        provider="anthropic",
        model_name="claude-sonnet-4-6",
        apply_to_settings=MagicMock(),
        model_retries=5,
        cli_max_retries=None,
    )
    run_agent = AsyncMock(return_value=None)
    resolve_mcp_tools = AsyncMock(return_value=([], None, []))

    with (
        patch.object(sys, "argv", ["deepagents", "--acp", "--recursion-limit", "3000"]),
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
        patch(
            "deepagents_acp.server.AgentServerACP",
            side_effect=_build_agent_server(object()),
        ),
        patch("acp.run_agent", run_agent),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli_main()

    assert exc_info.value.code == 0
    mock_create_agent.assert_called_once()
    assert mock_create_agent.call_args.kwargs["recursion_limit"] == 3000


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


def test_acp_managed_manual_revokes_the_yolo_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed `manual` revokes `--yolo` in the ACP launch, not just the resolver.

    The ACP branch derives every approval decision from `_resolve_approval_mode`
    rather than raw flags. Nothing pinned the wiring: reverting the forwarding
    to `getattr(args, "yolo", False)` left the whole suite green while
    `dcode --acp --yolo` under a managed `manual` launched with approvals
    disabled and no acknowledgement gate -- an administrator-policy bypass on a
    privilege-granting key.
    """
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text('[startup]\nmode = "manual"\n', encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()

    args = _make_acp_args(yolo=True, auto_approve=False)
    run_acp = AsyncMock(return_value=0)

    try:
        with (
            patch.object(sys, "argv", ["deepagents", "--acp", "--yolo"]),
            patch("deepagents_code.main.parse_args", return_value=args),
            # Would abort with exit 2 if the gate still read `args.yolo`.
            patch(
                "deepagents_code.approval_mode.has_yolo_acknowledgement",
                return_value=False,
            ),
            patch("deepagents_code.main._resolve_agent_arg", return_value="agent"),
            patch("deepagents_code.main._run_acp_cli_async", run_acp),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
    finally:
        service.invalidate_config_sources()

    assert exc_info.value.code == 0
    kwargs = run_acp.call_args.kwargs
    assert kwargs["yolo"] is False
    assert kwargs["auto"] is False
