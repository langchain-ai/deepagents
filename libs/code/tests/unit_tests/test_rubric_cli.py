"""Unit tests for rubric (`RubricMiddleware`) CLI wiring."""

from __future__ import annotations

from deepagents_code._server_config import ServerConfig


class TestResolveRubricText:
    """`_resolve_rubric_text` literal/file/@path resolution."""


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
