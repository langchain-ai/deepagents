"""Unit tests for ACP mode behavior in `cli_main`."""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_cli.main import cli_main


def _make_acp_args() -> argparse.Namespace:
    return argparse.Namespace(
        acp=True,
        model=None,
        model_params=None,
        profile_override=None,
        agent="agent",
    )


def test_acp_mode_skips_dependency_check_and_runs_server() -> None:
    """`--acp` should bypass UI dependency checks and start ACP server path."""
    args = _make_acp_args()
    model_obj = object()
    model_result = SimpleNamespace(
        model=model_obj,
        apply_to_settings=MagicMock(),
    )
    server = object()
    run_agent = AsyncMock(return_value=None)

    with (
        patch.object(sys, "argv", ["deepagents", "--acp"]),
        patch(
            "deepagents_cli.main.check_cli_dependencies",
            side_effect=AssertionError("check_cli_dependencies should be skipped"),
        ),
        patch("deepagents_cli.main.parse_args", return_value=args),
        patch(
            "deepagents_cli.config.create_model", return_value=model_result
        ) as mock_create_model,
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
    mock_create_model.assert_called_once_with(None, extra_kwargs=None)
    model_result.apply_to_settings.assert_called_once_with()
    mock_create_agent.assert_called_once_with(model=model_obj, assistant_id="agent")
    mock_server_cls.assert_called_once_with("graph")
    run_agent.assert_awaited_once_with(server)


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
