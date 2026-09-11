"""Unit tests for ACP mode behavior in `cli_main`."""

from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from deepagents_code.main import _run_acp_cli_async, cli_main


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
        summarization_model=None,
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


async def test_acp_defaults_classifier_after_provider_resolution(tmp_path) -> None:
    """ACP passes the resolved main provider's default into agent construction."""
    classifier_models: list[str | None] = []
    model_result = SimpleNamespace(
        model=object(),
        provider="openai",
        model_name="gpt-5.6-sol",
        model_retries=3,
        cli_max_retries=None,
        apply_to_runtime_state=MagicMock(),
    )

    class Checkpointer:
        async def setup(self) -> None:
            return None

    @asynccontextmanager
    async def get_checkpointer() -> AsyncIterator[Checkpointer]:
        yield Checkpointer()

    class AgentServer:
        def __init__(self, build_agent, **_kwargs: object) -> None:
            self.build_agent = build_agent

    async def run_agent(server: AgentServer) -> None:
        server.build_agent(SimpleNamespace(model=None, cwd=str(tmp_path)))
        await asyncio.sleep(0)

    def create_cli_agent(**kwargs: object) -> tuple[object, object]:
        classifier_model = kwargs["auto_classifier_model"]
        assert isinstance(classifier_model, str) or classifier_model is None
        classifier_models.append(classifier_model)
        return object(), object()

    with (
        patch("deepagents_code.config.create_model", return_value=model_result),
        patch("deepagents_code.config.is_memory_auto_save_enabled", return_value=True),
        patch("deepagents_code.config.credentials") as credentials,
        patch("deepagents_code.agent.create_cli_agent", side_effect=create_cli_agent),
        patch("deepagents_code.agent.load_async_subagents", return_value=None),
        patch("deepagents_code.model_config.get_available_models", return_value={}),
        patch("deepagents_code.model_config.save_recent_model"),
        patch("deepagents_code.model_config.touch_recent_model"),
        patch(
            "deepagents_code.mcp_tools.resolve_and_load_mcp_tools",
            new=AsyncMock(return_value=([], None, None)),
        ),
        patch("deepagents_code.sessions.get_checkpointer", new=get_checkpointer),
        patch("deepagents_code.acp.AgentServerACP", new=AgentServer),
    ):
        credentials.has_tavily = False
        exit_code = await _run_acp_cli_async(
            "agent",
            run_acp_agent=run_agent,
            agent_server_cls=AgentServer,
            model_name="openai:gpt-5.6-sol",
            no_mcp=True,
            auto=True,
        )

    assert exit_code == 0
    assert classifier_models == ["openai:gpt-5.6-luna"]
