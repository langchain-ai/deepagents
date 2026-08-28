"""Unit tests for rubric (`RubricMiddleware`) CLI wiring."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from rich.console import Console

if TYPE_CHECKING:
    from pathlib import Path

from deepagents_code._env_vars import SERVER_ENV_PREFIX
from deepagents_code._server_config import ServerConfig
from deepagents_code.client.non_interactive import (
    StreamState,
    _build_non_interactive_header,
    _process_rubric_event,
)
from deepagents_code.goal_state_limits import RUBRIC_CHAR_LIMIT
from deepagents_code.main import _resolve_rubric_text


class TestResolveRubricText:
    """`_resolve_rubric_text` literal/file/@path resolution."""


def _run_cli_main_devnull_stdin(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run `cli_main` in a subprocess with empty (non-piped) stdin.

    `stdin=DEVNULL` makes `apply_stdin_pipe` read an empty string and return
    early, so `non_interactive_message` stays unset — the deterministic way to
    reach the interactive-only argument guards without a TTY. `parse_args`
    handles `--non-interactive`/`-m`, and `check_cli_dependencies` is patched
    purely for environment portability (it only calls `importlib.util.find_spec`).
    """
    code = """
        import sys
        from unittest.mock import patch

        from deepagents_code.main import cli_main

        with (
            patch.object(sys, "argv", sys.argv[1:]),
            patch("deepagents_code.main.check_cli_dependencies"),
        ):
            cli_main()
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), "deepagents", *argv],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )


class TestRubricGating:
    """Rubric flags require `-n`/piped stdin; the guard lives in `cli_main`."""


class TestServerConfigRubric:
    """Rubric grader settings round-trip through env serialization."""

    def test_from_cli_args_forwards_rubric_settings(self) -> None:
        config = ServerConfig.from_cli_args(
            project_context=None,
            model_name=None,
            model_params=None,
            assistant_id="agent",
            auto_approve=False,
            sandbox_type="none",
            sandbox_id=None,
            sandbox_snapshot_name=None,
            sandbox_setup=None,
            enable_shell=True,
            enable_ask_user=False,
            rubric_model="openai:gpt-5.1",
            rubric_max_iterations=7,
            mcp_config_path=None,
            no_mcp=False,
            trust_project_mcp=None,
            interactive=True,
        )
        assert config.rubric_model == "openai:gpt-5.1"
        assert config.rubric_max_iterations == 7


def _render_event(data: dict, *, show_rubric_iterations: bool = False) -> str:
    state = StreamState(show_rubric_iterations=show_rubric_iterations)
    buf = io.StringIO()
    console = Console(file=buf, width=200, highlight=False)
    _process_rubric_event(data, state, console)
    return buf.getvalue()
