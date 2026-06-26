"""Unit tests for rubric (`RubricMiddleware`) CLI wiring."""

from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from rich.console import Console

if TYPE_CHECKING:
    from pathlib import Path

from deepagents_code._env_vars import SERVER_ENV_PREFIX
from deepagents_code._server_config import ServerConfig, _read_env_int
from deepagents_code.agent import _build_rubric_middleware
from deepagents_code.main import _resolve_rubric_text
from deepagents_code.non_interactive import (
    StreamState,
    _build_non_interactive_header,
    _process_rubric_event,
)


class TestResolveRubricText:
    """`_resolve_rubric_text` literal/file/@path resolution."""

    def test_none_when_unset(self) -> None:
        assert _resolve_rubric_text(None, None) is None

    def test_literal(self) -> None:
        assert (
            _resolve_rubric_text("tests pass; minimal", None) == "tests pass; minimal"
        )

    def test_literal_is_stripped(self) -> None:
        assert _resolve_rubric_text("  do X  ", None) == "do X"

    def test_empty_literal_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _resolve_rubric_text("   ", None)

    def test_rubric_file(self, tmp_path: Path) -> None:
        f = tmp_path / "rubric.md"
        f.write_text("criteria here\n", encoding="utf-8")
        assert _resolve_rubric_text(None, str(f)) == "criteria here"

    def test_at_path_in_rubric(self, tmp_path: Path) -> None:
        f = tmp_path / "rubric.md"
        f.write_text("from at-path", encoding="utf-8")
        assert _resolve_rubric_text(f"@{f}", None) == "from at-path"

    def test_mutually_exclusive(self, tmp_path: Path) -> None:
        f = tmp_path / "rubric.md"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="mutually exclusive"):
            _resolve_rubric_text("literal", str(f))

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Could not read rubric file"):
            _resolve_rubric_text(None, str(tmp_path / "nope.md"))

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "rubric.md"
        f.write_text("   \n", encoding="utf-8")
        with pytest.raises(ValueError, match="is empty"):
            _resolve_rubric_text(None, str(f))


class TestServerConfigRubric:
    """Rubric fields round-trip through the env serialization."""

    def test_defaults(self) -> None:
        config = ServerConfig()
        assert config.enable_rubric is False
        assert config.rubric_model is None
        assert config.rubric_max_iterations == 3

    def test_round_trip(self) -> None:
        original = ServerConfig(
            enable_rubric=True,
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
        assert restored.enable_rubric is True
        assert restored.rubric_model == "anthropic:claude-sonnet-4-6"
        assert restored.rubric_max_iterations == 5

    def test_from_cli_args_forwards_rubric(self) -> None:
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
            enable_rubric=True,
            rubric_model="openai:gpt-5.1",
            rubric_max_iterations=7,
            mcp_config_path=None,
            no_mcp=False,
            trust_project_mcp=None,
            interactive=True,
        )
        assert config.enable_rubric is True
        assert config.rubric_model == "openai:gpt-5.1"
        assert config.rubric_max_iterations == 7


class TestReadEnvInt:
    def test_missing_uses_default(self) -> None:
        assert _read_env_int("RUBRIC_MAX_ITERATIONS_X", default=3) == 3

    def test_parses_int(self) -> None:
        with patch.dict(os.environ, {f"{SERVER_ENV_PREFIX}FOO_INT": "9"}):
            assert _read_env_int("FOO_INT", default=3) == 9

    def test_invalid_falls_back(self) -> None:
        with patch.dict(os.environ, {f"{SERVER_ENV_PREFIX}FOO_INT": "nope"}):
            assert _read_env_int("FOO_INT", default=3) == 3


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


class TestBuildRubricMiddleware:
    def test_returns_none_when_sdk_lacks_middleware(self) -> None:
        """The pinned SDK predates `RubricMiddleware`, so this degrades to None.

        When a future SDK ships `RubricMiddleware`, this test should be updated
        to assert a middleware instance is returned instead.
        """
        import importlib.util

        if importlib.util.find_spec("deepagents.middleware.outcomes") is not None:
            pytest.skip("Installed SDK provides RubricMiddleware")
        assert (
            _build_rubric_middleware(grader_model="anthropic:x", max_iterations=3)
            is None
        )
