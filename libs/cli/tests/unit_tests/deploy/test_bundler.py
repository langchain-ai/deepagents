"""Tests for deploy bundler."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from deepagents_cli.deploy.bundler import (
    _MODEL_PROVIDER_DEPS,
    _build_seed,
    _render_deploy_graph,
    _render_langgraph_json,
    _render_pyproject,
    bundle,
    print_bundle_summary,
)
from deepagents_cli.deploy.config import (
    _MODEL_PROVIDER_ENV,
    AGENTS_MD_FILENAME,
    MCP_FILENAME,
    SKILLS_DIRNAME,
    AgentConfig,
    DeployConfig,
    SandboxConfig,
)

if TYPE_CHECKING:
    from pathlib import Path


def _minimal_project(tmp_path: Path, *, mcp: bool = False) -> Path:
    """Create a minimal project directory and return its path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / AGENTS_MD_FILENAME).write_text("# Agent prompt", encoding="utf-8")
    if mcp:
        data = {"mcpServers": {"s": {"type": "http", "url": "http://x"}}}
        (tmp_path / MCP_FILENAME).write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def _minimal_config(
    *,
    provider: str = "none",
    model: str = "anthropic:claude-sonnet-4-6",
) -> DeployConfig:
    return DeployConfig(
        agent=AgentConfig(name="test-agent", model=model),
        sandbox=SandboxConfig(provider=provider),  # type: ignore[arg-type]
    )


class TestBuildSeed:
    def test_memories_contain_agents_md(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        assert "/AGENTS.md" in seed["memories"]
        assert seed["memories"]["/AGENTS.md"] == "# prompt"

    def test_skills_empty_when_no_dir(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        assert seed["skills"] == {}

    def test_skills_populated_from_dir(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        skills = project / SKILLS_DIRNAME / "review"
        skills.mkdir(parents=True)
        (skills / "SKILL.md").write_text("skill content", encoding="utf-8")
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        assert "/review/SKILL.md" in seed["skills"]
        assert seed["skills"]["/review/SKILL.md"] == "skill content"

    def test_dotfiles_excluded(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        skills = project / SKILLS_DIRNAME
        skills.mkdir()
        (skills / ".hidden").write_text("secret", encoding="utf-8")
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        assert seed["skills"] == {}


class TestRenderLanggraphJson:
    def test_without_env(self) -> None:
        result = json.loads(_render_langgraph_json(env_present=False))
        assert "env" not in result
        assert result["python_version"] == "3.12"

    def test_with_env(self) -> None:
        result = json.loads(_render_langgraph_json(env_present=True))
        assert result["env"] == ".env"


class TestRenderPyproject:
    def test_no_extra_deps(self) -> None:
        # Use a model without a provider prefix so no provider dep is inferred.
        config = _minimal_config(model="bare-model")
        result = _render_pyproject(config, mcp_present=False)
        assert "test-agent" in result
        assert "langchain-mcp-adapters" not in result
        assert "langchain-openai" not in result

    def test_mcp_dep_added(self) -> None:
        config = _minimal_config()
        result = _render_pyproject(config, mcp_present=True)
        assert "langchain-mcp-adapters" in result

    def test_provider_dep_inferred(self) -> None:
        config = _minimal_config(provider="daytona")
        result = _render_pyproject(config, mcp_present=False)
        assert "langchain-daytona" in result

    def test_model_provider_dep(self) -> None:
        config = _minimal_config(model="openai:gpt-5.3-codex")
        result = _render_pyproject(config, mcp_present=False)
        assert "langchain-openai" in result

    def test_deps_cover_all_validated_providers(self) -> None:
        """Every validated provider must have a bundler dep."""
        no_partner_pkg: set[str] = set()
        missing = set(_MODEL_PROVIDER_ENV) - set(_MODEL_PROVIDER_DEPS) - no_partner_pkg
        assert not missing, (
            f"Providers validated but missing from bundler deps: {missing}"
        )

    @pytest.mark.parametrize(
        "provider",
        sorted(_MODEL_PROVIDER_DEPS),
    )
    def test_each_model_provider_dep_rendered(self, provider: str) -> None:
        config = _minimal_config(model=f"{provider}:some-model")
        result = _render_pyproject(config, mcp_present=False)
        assert _MODEL_PROVIDER_DEPS[provider] in result


class TestRenderDeployGraph:
    def test_output_is_valid_python(self) -> None:
        config = _minimal_config()
        result = _render_deploy_graph(config, mcp_present=False)
        compile(result, "<deploy_graph>", "exec")

    def test_mcp_block_included_when_present(self) -> None:
        config = _minimal_config()
        result = _render_deploy_graph(config, mcp_present=True)
        assert "_load_mcp_tools" in result
        assert "tools.extend(await _load_mcp_tools())" in result

    def test_mcp_block_absent_when_not_present(self) -> None:
        config = _minimal_config()
        result = _render_deploy_graph(config, mcp_present=False)
        assert "_load_mcp_tools" not in result
        assert "pass  # no MCP servers configured" in result

    def test_no_system_prompt_in_output(self) -> None:
        """AGENTS.md should not be baked into the deploy graph as a system prompt."""
        config = _minimal_config()
        result = _render_deploy_graph(config, mcp_present=False)
        compile(result, "<deploy_graph>", "exec")
        assert "SYSTEM_PROMPT" not in result
        assert "system_prompt=" not in result

    def test_each_provider_renders(self) -> None:
        """Every valid provider should produce compilable output."""
        from deepagents_cli.deploy.config import VALID_SANDBOX_PROVIDERS

        for provider in VALID_SANDBOX_PROVIDERS:
            config = _minimal_config(provider=provider)
            result = _render_deploy_graph(config, mcp_present=False)
            compile(result, f"<deploy_graph_{provider}>", "exec")


class TestBundle:
    def test_produces_expected_files(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path / "project")
        build = tmp_path / "build"
        config = _minimal_config()
        bundle(config, project, build)

        assert (build / "_seed.json").exists()
        assert (build / "deploy_graph.py").exists()
        assert (build / "langgraph.json").exists()
        assert (build / "pyproject.toml").exists()

    def test_mcp_copied(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path / "project", mcp=True)
        build = tmp_path / "build"
        config = _minimal_config()
        bundle(config, project, build)
        assert (build / "_mcp.json").exists()

    def test_env_copied(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path / "project")
        (project / ".env").write_text("KEY=val", encoding="utf-8")
        build = tmp_path / "build"
        config = _minimal_config()
        bundle(config, project, build)
        assert (build / ".env").exists()
        assert (build / ".env").read_text(encoding="utf-8") == "KEY=val"

    def test_unknown_provider_raises(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path / "project")
        build = tmp_path / "build"
        # Bypass Literal typing to test runtime guard in bundler.
        config = DeployConfig(
            agent=AgentConfig(name="x"),
            sandbox=SandboxConfig(provider="bogus"),  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="Unknown sandbox provider"):
            bundle(config, project, build)


class TestPrintBundleSummary:
    def test_handles_valid_seed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        seed = {"memories": {"/AGENTS.md": "x"}, "skills": {}}
        (tmp_path / "_seed.json").write_text(json.dumps(seed), encoding="utf-8")
        config = _minimal_config()
        print_bundle_summary(config, tmp_path)
        out = capsys.readouterr().out
        assert "test-agent" in out
        assert "1 file(s)" in out

    def test_handles_missing_seed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = _minimal_config()
        print_bundle_summary(config, tmp_path)
        out = capsys.readouterr().out
        assert "test-agent" in out

    def test_handles_corrupt_seed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should log warning and continue when _seed.json is invalid JSON."""
        (tmp_path / "_seed.json").write_text("{bad", encoding="utf-8")
        config = _minimal_config()
        print_bundle_summary(config, tmp_path)
        out = capsys.readouterr().out
        assert "test-agent" in out
