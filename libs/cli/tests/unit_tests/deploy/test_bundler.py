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
    AuthConfig,
    DeployConfig,
    MemoriesConfig,
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
    auth: AuthConfig | None = None,
) -> DeployConfig:
    return DeployConfig(
        agent=AgentConfig(name="test-agent", model=model),
        sandbox=SandboxConfig(provider=provider),  # type: ignore[arg-type]
        auth=auth,
    )


class TestBuildSeed:
    def test_memories_contain_agents_md(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        seed = _build_seed(project, "# prompt")
        assert "/AGENTS.md" in seed["memories"]
        assert seed["memories"]["/AGENTS.md"] == "# prompt"

    def test_skills_empty_when_no_dir(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        seed = _build_seed(project, "# prompt")
        assert seed["skills"] == {}

    def test_skills_populated_from_dir(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        skills = project / SKILLS_DIRNAME / "review"
        skills.mkdir(parents=True)
        (skills / "SKILL.md").write_text("skill content", encoding="utf-8")
        seed = _build_seed(project, "# prompt")
        assert "/review/SKILL.md" in seed["skills"]
        assert seed["skills"]["/review/SKILL.md"] == "skill content"

    def test_dotfiles_excluded(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        skills = project / SKILLS_DIRNAME
        skills.mkdir()
        (skills / ".hidden").write_text("secret", encoding="utf-8")
        seed = _build_seed(project, "# prompt")
        assert seed["skills"] == {}

    def test_user_memories_empty_when_no_dir(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        seed = _build_seed(project, "# prompt")
        assert seed["user_memories"] == {}

    def test_user_memories_seeds_empty_agents_md_when_empty_dir(
        self,
        tmp_path: Path,
    ) -> None:
        project = _minimal_project(tmp_path)
        (project / USER_DIRNAME).mkdir()
        seed = _build_seed(project, "# prompt")
        assert seed["user_memories"] == {"/AGENTS.md": ""}

    def test_user_memories_reads_agents_md(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        user_dir = project / USER_DIRNAME
        user_dir.mkdir(parents=True)
        content = "# User Prefs\n"
        (user_dir / "AGENTS.md").write_text(content, encoding="utf-8")
        seed = _build_seed(project, "# prompt")
        assert seed["user_memories"] == {"/AGENTS.md": content}

    def test_user_memories_ignores_non_agents_md_files(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        user_dir = project / USER_DIRNAME
        user_dir.mkdir(parents=True)
        (user_dir / "AGENTS.md").write_text("# user mem", encoding="utf-8")
        (user_dir / "other.md").write_text("ignored", encoding="utf-8")
        seed = _build_seed(project, "# prompt")
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

    def test_langsmith_block_uses_snapshot_api(self) -> None:
        """Generated langsmith block must reference the snapshot API surface.

        Catches drift between the `sandbox_snapshot` bundler variable and the
        `SANDBOX_SNAPSHOT` reference inside the generated block, as well as
        silent removal of the snapshot env vars.
        """
        config = _minimal_config(provider="langsmith")
        result = _render_deploy_graph(config, mcp_present=False)

        # Snapshot API symbols must appear (not the old template API).
        assert "SANDBOX_SNAPSHOT" in result
        assert "list_snapshots()" in result
        assert "create_snapshot(" in result
        assert "snapshot_id=snapshot_id" in result
        assert "LANGSMITH_SANDBOX_SNAPSHOT_ID" in result
        assert "LANGSMITH_SANDBOX_SNAPSHOT_NAME" in result

        # Legacy template API must be gone.
        assert "SANDBOX_TEMPLATE" not in result
        assert "template_name=" not in result
        assert "get_template(" not in result

    def test_langsmith_block_wraps_errors_with_runtime_error(self) -> None:
        """Generated langsmith block must wrap SDK calls in RuntimeError.

        Mirrors `_LangSmithProvider._ensure_snapshot` so deployed agents
        surface actionable errors instead of raw SDK tracebacks.
        """
        config = _minimal_config(provider="langsmith")
        result = _render_deploy_graph(config, mcp_present=False)

        assert "Failed to list snapshots" in result
        assert "Failed to build snapshot" in result
        assert "Failed to create sandbox from snapshot" in result
        # API-key fallback must not raise KeyError on missing env vars.
        assert 'os.environ["LANGCHAIN_API_KEY"]' not in result
        assert "No LangSmith sandbox API key found" in result

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

    def test_default_memories_backend_is_store(self) -> None:
        result = _render_deploy_graph(_minimal_config(), mcp_present=False)
        assert "MEMORIES_BACKEND = 'store'" in result

    def test_hub_backend_wires_context_hub_route(self) -> None:
        config = DeployConfig(
            agent=AgentConfig(name="hubtest"),
            memories=MemoriesConfig(backend="hub"),
        )
        result = _render_deploy_graph(config, mcp_present=False)
        assert "MEMORIES_BACKEND = 'hub'" in result
        assert "MEMORIES_HUB_IDENTIFIER = '-/hubtest'" in result
        assert "from _context_hub import ContextHubBackend" in result
        assert "_seed_hub_if_needed" in result

    def test_hub_backend_honors_identifier_override(self) -> None:
        config = DeployConfig(
            agent=AgentConfig(name="hubtest"),
            memories=MemoriesConfig(backend="hub", identifier="org-ns/custom"),
        )
        result = _render_deploy_graph(config, mcp_present=False)
        assert "MEMORIES_HUB_IDENTIFIER = 'org-ns/custom'" in result


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

    def test_hub_bundle_vendors_context_hub(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path / "project")
        build = tmp_path / "build"
        config = DeployConfig(
            agent=AgentConfig(name="hubtest"),
            memories=MemoriesConfig(backend="hub"),
        )
        bundle(config, project, build)
        vendored = build / "_context_hub.py"
        assert vendored.exists()
        # The vendored file must contain the class the graph imports.
        assert "class ContextHubBackend" in vendored.read_text(encoding="utf-8")
        # Generated graph should syntactically compile.
        graph_py = (build / "deploy_graph.py").read_text(encoding="utf-8")
        compile(graph_py, "<hub_deploy_graph>", "exec")

    def test_store_bundle_does_not_vendor_context_hub(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path / "project")
        build = tmp_path / "build"
        config = _minimal_config()
        bundle(config, project, build)
        assert not (build / "_context_hub.py").exists()

    def test_hub_bundle_adds_langsmith_dep(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path / "project")
        build = tmp_path / "build"
        config = DeployConfig(
            agent=AgentConfig(name="hubtest"),
            memories=MemoriesConfig(backend="hub"),
        )
        bundle(config, project, build)
        pyproject = (build / "pyproject.toml").read_text(encoding="utf-8")
        assert "langsmith>=0.7.35" in pyproject

    def test_bundle_with_subagents(self, tmp_path: Path) -> None:
        """Full bundle with sync subagents produces valid artifacts."""
        project = _minimal_project(tmp_path / "project")
        _add_subagent(
            project,
            "researcher",
            description="Research agent",
            skills={"search/SKILL.md": "# Search"},
        )
        _add_subagent(project, "coder", description="Coding agent")
        build = tmp_path / "build"
        config = _minimal_config()
        bundle(config, project, build)

        # Verify seed has subagents.
        seed = json.loads((build / "_seed.json").read_text(encoding="utf-8"))
        assert "subagents" in seed
        assert "researcher" in seed["subagents"]
        assert "coder" in seed["subagents"]
        assert (
            seed["subagents"]["researcher"]["skills"]["/search/SKILL.md"] == "# Search"
        )
        assert "async_subagents" not in seed

        # Verify generated deploy_graph.py is valid Python.
        graph_py = (build / "deploy_graph.py").read_text(encoding="utf-8")
        compile(graph_py, "<deploy_graph_subagents>", "exec")
        assert "SubAgent" in graph_py
        assert "_build_sync_subagents" in graph_py


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
                    "config": {
                        "name": "researcher",
                        "description": "Research agent",
                        "model": "anthropic:claude-sonnet-4-6",
                    },
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
        assert "Subagents (1)" in out
        assert "researcher" in out


class TestRenderLanggraphJsonAuth:
    def test_without_auth(self) -> None:
        result = json.loads(
            _render_langgraph_json(env_present=False, auth_present=False)
        )
        assert "auth" not in result

    def test_with_auth(self) -> None:
        result = json.loads(_render_langgraph_json(env_present=True, auth_present=True))
        assert result["auth"] == {"path": "./auth.py:auth"}


class TestRenderPyprojectAuth:
    def test_no_auth_dep(self) -> None:
        config = _minimal_config(model="bare-model")
        result = _render_pyproject(config, mcp_present=False)
        assert "pyjwt" not in result

    def test_clerk_adds_pyjwt(self) -> None:
        config = _minimal_config(model="bare-model", auth=AuthConfig(provider="clerk"))
        result = _render_pyproject(config, mcp_present=False)
        assert "pyjwt" in result

    def test_supabase_no_extra_dep(self) -> None:
        config = _minimal_config(
            model="bare-model", auth=AuthConfig(provider="supabase")
        )
        result = _render_pyproject(config, mcp_present=False)
        assert "pyjwt" not in result


class TestBundleAuth:
    def test_auth_py_generated(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path / "project")
        build = tmp_path / "build"
        config = _minimal_config(auth=AuthConfig(provider="supabase"))
        bundle(config, project, build)
        assert (build / "auth.py").exists()
        content = (build / "auth.py").read_text(encoding="utf-8")
        assert "auth = Auth()" in content
        assert "SUPABASE_URL" in content
        assert "add_owner" in content

    def test_auth_py_not_generated_without_auth(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path / "project")
        build = tmp_path / "build"
        config = _minimal_config()
        bundle(config, project, build)
        assert not (build / "auth.py").exists()

    def test_langgraph_json_includes_auth(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path / "project")
        build = tmp_path / "build"
        config = _minimal_config(auth=AuthConfig(provider="supabase"))
        bundle(config, project, build)
        lg = json.loads((build / "langgraph.json").read_text(encoding="utf-8"))
        assert lg["auth"] == {"path": "./auth.py:auth"}

    def test_langgraph_json_no_auth_without_config(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path / "project")
        build = tmp_path / "build"
        config = _minimal_config()
        bundle(config, project, build)
        lg = json.loads((build / "langgraph.json").read_text(encoding="utf-8"))
        assert "auth" not in lg

    def test_clerk_auth_py_valid_python(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path / "project")
        build = tmp_path / "build"
        config = _minimal_config(auth=AuthConfig(provider="clerk"))
        bundle(config, project, build)
        content = (build / "auth.py").read_text(encoding="utf-8")
        compile(content, "<auth.py>", "exec")
        assert "CLERK_SECRET_KEY" in content
        assert "add_owner" in content

    def test_supabase_auth_py_valid_python(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path / "project")
        build = tmp_path / "build"
        config = _minimal_config(auth=AuthConfig(provider="supabase"))
        bundle(config, project, build)
        content = (build / "auth.py").read_text(encoding="utf-8")
        compile(content, "<auth.py>", "exec")


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
        seed = _build_seed(project, "# prompt")
        assert "subagents" not in seed

    def test_sync_subagent_included(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        _add_subagent(project, "helper", description="A helper")
        seed = _build_seed(project, "# prompt")
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
        seed = _build_seed(project, "# prompt")
        sa = seed["subagents"]["coder"]
        assert "/review/SKILL.md" in sa["skills"]
        assert sa["skills"]["/review/SKILL.md"] == "review skill"

    def test_sync_subagent_with_mcp(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        mcp_data = {"mcpServers": {"s": {"type": "http", "url": "http://x"}}}
        _add_subagent(project, "mcp-agent", description="MCP agent", mcp=mcp_data)
        seed = _build_seed(project, "# prompt")
        sa = seed["subagents"]["mcp-agent"]
        assert sa["mcp"] == mcp_data

    def test_sync_subagent_no_mcp(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        _add_subagent(project, "plain", description="Plain agent")
        seed = _build_seed(project, "# prompt")
        sa = seed["subagents"]["plain"]
        assert sa["mcp"] is None

    def test_no_async_subagents_key(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        seed = _build_seed(project, "# prompt")
        assert "async_subagents" not in seed

    def test_multiple_sync_subagents(self, tmp_path: Path) -> None:
        project = _minimal_project(tmp_path)
        _add_subagent(project, "alpha", description="Alpha agent")
        _add_subagent(project, "beta", description="Beta agent")
        seed = _build_seed(project, "# prompt")
        assert "alpha" in seed["subagents"]
        assert "beta" in seed["subagents"]
        assert len(seed["subagents"]) == 2


class TestRenderDeployGraphSubagents:
    def test_subagent_imports_when_sync(self) -> None:
        config = _minimal_config()
        result = _render_deploy_graph(
            config,
            mcp_present=False,
            has_sync_subagents=True,
        )
        compile(result, "<deploy_graph_sync_sa>", "exec")
        assert "from deepagents.middleware.subagents import SubAgent" in result
        assert "_build_sync_subagents" in result

    def test_no_subagent_imports_when_none(self) -> None:
        config = _minimal_config()
        result = _render_deploy_graph(config, mcp_present=False)
        assert "SubAgent" not in result

    def test_subagents_passed_to_create_deep_agent(self) -> None:
        config = _minimal_config()
        result = _render_deploy_graph(
            config,
            mcp_present=False,
            has_sync_subagents=True,
        )
        compile(result, "<deploy_graph_sync_sa>", "exec")
        assert "subagents=" in result

    def test_subagent_seeding(self) -> None:
        config = _minimal_config()
        result = _render_deploy_graph(
            config,
            mcp_present=False,
            has_sync_subagents=True,
        )
        assert "_build_sync_subagents" in result
        assert '"subagents"' in result
