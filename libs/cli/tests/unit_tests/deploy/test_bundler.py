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
    SUBAGENTS_DIRNAME,
    USER_DIRNAME,
    AgentConfig,
    AsyncSubAgentConfig,
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

    def test_user_memories_empty_when_no_dir(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        assert seed["user_memories"] == {}

    def test_user_memories_seeds_empty_agents_md_when_empty_dir(
        self,
        tmp_path: Path,
    ) -> None:
        project = _minimal_project(tmp_path)
        (project / USER_DIRNAME).mkdir()
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        assert seed["user_memories"] == {"/AGENTS.md": ""}

    def test_user_memories_reads_agents_md(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        user_dir = project / USER_DIRNAME
        user_dir.mkdir(parents=True)
        content = "# User Prefs\n"
        (user_dir / "AGENTS.md").write_text(content, encoding="utf-8")
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        assert seed["user_memories"] == {"/AGENTS.md": content}

    def test_user_memories_ignores_non_agents_md_files(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        user_dir = project / USER_DIRNAME
        user_dir.mkdir(parents=True)
        (user_dir / "AGENTS.md").write_text("# user mem", encoding="utf-8")
        (user_dir / "other.md").write_text("ignored", encoding="utf-8")
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        assert list(seed["user_memories"].keys()) == ["/AGENTS.md"]


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
        no_partner_pkg = {"together"}
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

    def test_subagent_model_dep_inferred(self) -> None:
        config = _minimal_config()
        result = _render_pyproject(
            config,
            mcp_present=False,
            subagent_model_providers=["openai"],
        )
        assert "langchain-openai" in result

    def test_subagent_mcp_adds_dep(self) -> None:
        config = _minimal_config()
        result = _render_pyproject(
            config,
            mcp_present=False,
            has_subagent_mcp=True,
        )
        assert "langchain-mcp-adapters" in result


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

    def test_skills_prefix_under_memories(self) -> None:
        config = _minimal_config()
        result = _render_deploy_graph(config, mcp_present=False)
        assert 'SKILLS_PREFIX = "/memories/skills/"' in result

    def test_user_memories_disabled_by_default(self) -> None:
        config = _minimal_config()
        result = _render_deploy_graph(config, mcp_present=False)
        assert "HAS_USER_MEMORIES = False" in result

    def test_user_memories_enabled(self) -> None:
        config = _minimal_config()
        result = _render_deploy_graph(
            config,
            mcp_present=False,
            has_user_memories=True,
        )
        compile(result, "<deploy_graph_user_mem>", "exec")
        assert "HAS_USER_MEMORIES = True" in result
        assert 'USER_PREFIX = "/memories/user/"' in result
        assert "_seed_user_memories_if_needed" in result
        # Single user AGENTS.md path preloaded into memory sources
        assert "USER_PREFIX" in result
        assert "AGENTS.md" in result

    def test_no_default_user_id_fallback(self) -> None:
        """User namespace factory raises instead of falling back to 'default'."""
        config = _minimal_config()
        result = _render_deploy_graph(
            config,
            mcp_present=False,
            has_user_memories=True,
        )
        # The user namespace factory should raise, not fall back
        fn = "_make_user_namespace_factory"
        assert fn in result
        assert '"default"' not in result.split(fn)[1].split("\n\n")[0]

    def test_agents_md_and_skills_denied_writes(self) -> None:
        config = _minimal_config()
        result = _render_deploy_graph(config, mcp_present=False)
        assert "AGENTS.md" in result
        assert 'mode="deny"' in result


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

    def test_empty_user_dir_enables_user_memories(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path / "project")
        (project / USER_DIRNAME).mkdir()
        build = tmp_path / "build"
        config = _minimal_config()
        bundle(config, project, build)
        graph_py = (build / "deploy_graph.py").read_text(encoding="utf-8")
        assert "HAS_USER_MEMORIES = True" in graph_py

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

    def test_bundle_with_subagents(self, tmp_path: Path) -> None:
        """Full bundle with sync and async subagents produces valid artifacts."""
        project = _minimal_project(tmp_path / "project")
        _add_subagent(
            project,
            "researcher",
            description="Research agent",
            skills={"search/SKILL.md": "# Search"},
        )
        _add_subagent(project, "coder", description="Coding agent")
        build = tmp_path / "build"
        config = DeployConfig(
            agent=AgentConfig(name="test-agent"),
            sandbox=SandboxConfig(),
            async_subagents=[
                AsyncSubAgentConfig(
                    name="writer",
                    description="Content writer",
                    graph_id="writer-graph",
                    url="https://example.com",
                ),
            ],
        )
        bundle(config, project, build)

        # Verify seed has subagents.
        seed = json.loads((build / "_seed.json").read_text(encoding="utf-8"))
        assert "subagents" in seed
        assert "researcher" in seed["subagents"]
        assert "coder" in seed["subagents"]
        assert seed["subagents"]["researcher"]["skills"]["/search/SKILL.md"] == "# Search"
        assert "async_subagents" in seed
        assert len(seed["async_subagents"]) == 1
        assert seed["async_subagents"][0]["name"] == "writer"

        # Verify generated deploy_graph.py is valid Python.
        graph_py = (build / "deploy_graph.py").read_text(encoding="utf-8")
        compile(graph_py, "<deploy_graph_subagents>", "exec")
        assert "SubAgent" in graph_py
        assert "AsyncSubAgent" in graph_py
        assert "_build_sync_subagents" in graph_py
        assert "_build_async_subagents" in graph_py


class TestPrintBundleSummary:
    def test_handles_valid_seed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        seed = {"memories": {"/AGENTS.md": "x"}, "skills": {}, "user_memories": {}}
        (tmp_path / "_seed.json").write_text(json.dumps(seed), encoding="utf-8")
        config = _minimal_config()
        print_bundle_summary(config, tmp_path)
        out = capsys.readouterr().out
        assert "test-agent" in out
        assert "1 file(s)" in out

    def test_user_memory_summary(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        seed = {
            "memories": {"/AGENTS.md": "x"},
            "skills": {},
            "user_memories": {"/AGENTS.md": ""},
        }
        (tmp_path / "_seed.json").write_text(json.dumps(seed), encoding="utf-8")
        config = _minimal_config()
        print_bundle_summary(config, tmp_path)
        out = capsys.readouterr().out
        assert "User memory seed" in out
        assert "/AGENTS.md" in out

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

    def test_sync_subagent_summary(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        seed = {
            "memories": {"/AGENTS.md": "x"},
            "skills": {},
            "subagents": {
                "researcher": {
                    "config": {"name": "researcher", "description": "Research agent", "model": "anthropic:claude-sonnet-4-6"},
                    "memories": {"/AGENTS.md": "y"},
                    "skills": {"/search/SKILL.md": "z"},
                    "mcp": None,
                },
            },
        }
        (tmp_path / "_seed.json").write_text(json.dumps(seed), encoding="utf-8")
        config = _minimal_config()
        print_bundle_summary(config, tmp_path)
        out = capsys.readouterr().out
        assert "Subagents (1 sync" in out
        assert "researcher" in out

    def test_async_subagent_summary(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        seed = {
            "memories": {"/AGENTS.md": "x"},
            "skills": {},
            "async_subagents": [
                {"name": "writer", "description": "Content writer", "graph_id": "g", "url": ""},
            ],
        }
        (tmp_path / "_seed.json").write_text(json.dumps(seed), encoding="utf-8")
        config = _minimal_config()
        print_bundle_summary(config, tmp_path)
        out = capsys.readouterr().out
        assert "1 async" in out
        assert "writer" in out


def _add_subagent(
    project: Path,
    name: str,
    *,
    description: str = "A subagent",
    skills: dict[str, str] | None = None,
    mcp: dict | None = None,
) -> Path:
    """Add a subagent directory to an existing project."""
    sa_dir = project / SUBAGENTS_DIRNAME / name
    sa_dir.mkdir(parents=True, exist_ok=True)
    toml = f'[agent]\nname = "{name}"\ndescription = "{description}"\n'
    (sa_dir / "deepagents.toml").write_text(toml, encoding="utf-8")
    (sa_dir / "AGENTS.md").write_text(f"# {name} prompt", encoding="utf-8")
    if skills:
        for skill_path, content in skills.items():
            skill_file = sa_dir / "skills" / skill_path
            skill_file.parent.mkdir(parents=True, exist_ok=True)
            skill_file.write_text(content, encoding="utf-8")
    if mcp is not None:
        (sa_dir / "mcp.json").write_text(json.dumps(mcp), encoding="utf-8")
    return sa_dir


class TestBuildSeedSubagents:
    def test_no_subagents_key_when_none(self, tmp_path: Path) -> None:
        """No subagents dir means no 'subagents' key in seed."""
        project = _minimal_project(tmp_path)
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        assert "subagents" not in seed

    def test_sync_subagent_included(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        _add_subagent(project, "helper", description="A helper")
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        assert "subagents" in seed
        assert "helper" in seed["subagents"]
        sa = seed["subagents"]["helper"]
        assert sa["config"]["name"] == "helper"
        assert sa["config"]["description"] == "A helper"
        assert "/AGENTS.md" in sa["memories"]
        assert sa["memories"]["/AGENTS.md"] == "# helper prompt"

    def test_sync_subagent_with_skills(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        _add_subagent(
            project,
            "coder",
            description="A coder",
            skills={"review/SKILL.md": "review skill"},
        )
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        sa = seed["subagents"]["coder"]
        assert "/review/SKILL.md" in sa["skills"]
        assert sa["skills"]["/review/SKILL.md"] == "review skill"

    def test_sync_subagent_with_mcp(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        mcp_data = {"mcpServers": {"s": {"type": "http", "url": "http://x"}}}
        _add_subagent(project, "mcp-agent", description="MCP agent", mcp=mcp_data)
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        sa = seed["subagents"]["mcp-agent"]
        assert sa["mcp"] == mcp_data

    def test_sync_subagent_no_mcp(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        _add_subagent(project, "plain", description="Plain agent")
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        sa = seed["subagents"]["plain"]
        assert sa["mcp"] is None

    def test_async_subagents_included(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        config = DeployConfig(
            agent=AgentConfig(name="test-agent"),
            sandbox=SandboxConfig(),
            async_subagents=[
                AsyncSubAgentConfig(
                    name="writer",
                    description="Content writer",
                    graph_id="writer-graph",
                    url="https://example.com",
                ),
            ],
        )
        seed = _build_seed(config, project, "# prompt")
        assert "async_subagents" in seed
        assert len(seed["async_subagents"]) == 1
        asa = seed["async_subagents"][0]
        assert asa["name"] == "writer"
        assert asa["description"] == "Content writer"
        assert asa["graph_id"] == "writer-graph"
        assert asa["url"] == "https://example.com"

    def test_no_async_subagents_key_when_none(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        assert "async_subagents" not in seed

    def test_multiple_sync_subagents(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        _add_subagent(project, "alpha", description="Alpha agent")
        _add_subagent(project, "beta", description="Beta agent")
        config = _minimal_config()
        seed = _build_seed(config, project, "# prompt")
        assert "alpha" in seed["subagents"]
        assert "beta" in seed["subagents"]
        assert len(seed["subagents"]) == 2


class TestRenderDeployGraphSubagents:
    def test_subagent_imports_when_sync(self) -> None:
        config = _minimal_config()
        result = _render_deploy_graph(
            config, mcp_present=False, has_sync_subagents=True,
        )
        compile(result, "<deploy_graph_sync_sa>", "exec")
        assert "from deepagents.middleware.subagents import SubAgent" in result
        assert "_build_sync_subagents" in result

    def test_subagent_imports_when_async(self) -> None:
        config = _minimal_config()
        result = _render_deploy_graph(
            config, mcp_present=False, has_async_subagents=True,
        )
        compile(result, "<deploy_graph_async_sa>", "exec")
        assert "from deepagents.middleware.async_subagents import AsyncSubAgent" in result
        assert "_build_async_subagents" in result

    def test_no_subagent_imports_when_none(self) -> None:
        config = _minimal_config()
        result = _render_deploy_graph(config, mcp_present=False)
        assert "SubAgent" not in result
        assert "AsyncSubAgent" not in result

    def test_subagents_passed_to_create_deep_agent(self) -> None:
        config = _minimal_config()
        result = _render_deploy_graph(
            config, mcp_present=False, has_sync_subagents=True, has_async_subagents=True,
        )
        compile(result, "<deploy_graph_both_sa>", "exec")
        assert "subagents=" in result

    def test_subagent_seeding(self) -> None:
        config = _minimal_config()
        result = _render_deploy_graph(
            config, mcp_present=False, has_sync_subagents=True,
        )
        assert "_build_sync_subagents" in result
        assert '"subagents"' in result
