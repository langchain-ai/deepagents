"""Unit tests for ACP mode behavior in `cli_main`."""

from __future__ import annotations

import argparse
import sys
from unittest.mock import patch

import pytest

from deepagents_code.main import cli_main


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
