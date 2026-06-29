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
from deepagents_code.main import _resolve_rubric_text
from deepagents_code.non_interactive import (
    StreamState,
    _build_non_interactive_header,
    _process_rubric_event,
)


class TestResolveRubricText:
    """`_resolve_rubric_text` literal/file/@path resolution."""

    def test_none_when_unset(self) -> None:
        assert _resolve_rubric_text(None) is None

    def test_literal(self) -> None:
        assert _resolve_rubric_text("tests pass; minimal") == "tests pass; minimal"

    def test_literal_is_stripped(self) -> None:
        assert _resolve_rubric_text("  do X  ") == "do X"

    def test_empty_literal_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _resolve_rubric_text("   ")

    def test_at_path_in_rubric(self, tmp_path: Path) -> None:
        f = tmp_path / "rubric.md"
        f.write_text("from at-path", encoding="utf-8")
        assert _resolve_rubric_text(f"@{f}") == "from at-path"

    def test_at_prefix_always_treated_as_path(self) -> None:
        # Documents the one-way ambiguity: any `@`-prefixed value is read as a
        # file path, so a literal rubric beginning with `@` is unreachable and
        # surfaces a read error rather than being used verbatim.
        with pytest.raises(ValueError, match="Could not read rubric file"):
            _resolve_rubric_text("@tests pass; minimal diff")

    def test_bare_at_sign_rejected(self) -> None:
        # `@` with no path (e.g. an empty shell glob) must still error rather
        # than silently resolving to the current directory.
        with pytest.raises(ValueError, match="Could not read rubric file"):
            _resolve_rubric_text("@")

    def test_at_path_expands_tilde(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Exercises the `.expanduser()` call, otherwise uncovered.
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "rubric.md").write_text("tilde criteria", encoding="utf-8")
        assert _resolve_rubric_text("@~/rubric.md") == "tilde criteria"

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Could not read rubric file"):
            _resolve_rubric_text(f"@{tmp_path / 'nope.md'}")

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "rubric.md"
        f.write_text("   \n", encoding="utf-8")
        with pytest.raises(ValueError, match="is empty"):
            _resolve_rubric_text(f"@{f}")


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

    def test_rubric_without_non_interactive_errors(self) -> None:
        result = _run_cli_main_devnull_stdin(["--rubric", "tests pass"])
        assert result.returncode == 2, result.stderr
        assert "--non-interactive" in result.stderr
        assert "--rubric" in result.stderr
        # The removed flag must not resurface in the guidance.
        assert "--rubric-file" not in result.stderr


class TestServerConfigRubric:
    """Rubric grader settings round-trip through env serialization."""

    def test_defaults(self) -> None:
        config = ServerConfig()
        assert config.rubric_model is None
        assert config.rubric_max_iterations == 3

    def test_round_trip(self) -> None:
        original = ServerConfig(
            rubric_model="anthropic:claude-sonnet-4-6",
            rubric_max_iterations=5,
        )
        env = {
            f"{SERVER_ENV_PREFIX}{k}": v
            for k, v in original.to_env().items()
            if v is not None
        }
        with patch.dict(os.environ, env, clear=False):
            restored = ServerConfig.from_env()
        assert restored.rubric_model == "anthropic:claude-sonnet-4-6"
        assert restored.rubric_max_iterations == 5

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


class TestHeaderIndicator:
    def test_rubric_active_marker(self) -> None:
        header = _build_non_interactive_header("agent", "thread-1", rubric_active=True)
        assert "Rubric: active" in header.plain

    def test_no_marker_when_inactive(self) -> None:
        header = _build_non_interactive_header("agent", "thread-1", rubric_active=False)
        assert "Rubric" not in header.plain


def _render_event(data: dict) -> str:
    state = StreamState()
    buf = io.StringIO()
    console = Console(file=buf, width=200, highlight=False)
    _process_rubric_event(data, state, console)
    return buf.getvalue()


class TestProcessRubricEvent:
    def test_ignores_non_rubric_payload(self) -> None:
        assert _render_event({"type": "something_else"}) == ""

    def test_start_event(self) -> None:
        out = _render_event({"type": "rubric_evaluation_start", "iteration": 0})
        assert "Grading against rubric" in out
        assert "iteration 1" in out

    def test_satisfied(self) -> None:
        out = _render_event(
            {"type": "rubric_evaluation_end", "result": "satisfied", "criteria": []}
        )
        assert "Rubric satisfied" in out

    def test_needs_revision_with_criteria(self) -> None:
        out = _render_event(
            {
                "type": "rubric_evaluation_end",
                "result": "needs_revision",
                "explanation": "tests missing",
                "criteria": [
                    {"name": "tests", "passed": False, "gap": "no coverage"},
                    {"name": "style", "passed": True},
                ],
            }
        )
        assert "needs revision" in out
        assert "tests missing" in out
        assert "no coverage" in out
        assert "style" not in out

    def test_max_iterations(self) -> None:
        out = _render_event(
            {"type": "rubric_evaluation_end", "result": "max_iterations_reached"}
        )
        assert "max iterations reached" in out

    def test_failed(self) -> None:
        out = _render_event(
            {
                "type": "rubric_evaluation_end",
                "result": "failed",
                "explanation": "bad rubric",
            }
        )
        assert "grader failed" in out
        assert "bad rubric" in out
