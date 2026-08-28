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
