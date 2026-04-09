"""Unit tests for deploy bundler with subagent support."""

import json
from pathlib import Path

from deepagents_cli.deploy.bundler import _build_seed, bundle
from deepagents_cli.deploy.config import (
    AgentConfig,
    DeployConfig,
    SandboxConfig,
    SubagentConfig,
)


def _make_project(tmp_path: Path, *, subagents: list[dict] | None = None) -> Path:
    """Scaffold a minimal deploy project."""
    (tmp_path / "AGENTS.md").write_text("# Main Agent\nYou are the main agent.\n")
    (tmp_path / "deepagents.toml").write_text(
        '[agent]\nname = "test-agent"\nmodel = "anthropic:claude-sonnet-4-6"\n'
    )
    for sa in subagents or []:
        sa_dir = tmp_path / "agents" / sa["name"]
        sa_dir.mkdir(parents=True)
        (sa_dir / "AGENTS.md").write_text(
            f"---\nname: {sa['name']}\ndescription: {sa.get('description', 'Test')}\n---\n\n"
            f"{sa.get('system_prompt', 'You are a test subagent.')}\n"
        )
        if sa.get("skills"):
            skill_dir = sa_dir / "skills" / "s1"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("---\nname: s1\ndescription: Skill\n---\n\nDo it.\n")
        if sa.get("mcp"):
            (sa_dir / "mcp.json").write_text(
                '{"mcpServers": {"test": {"type": "http", "url": "http://localhost:8080"}}}'
            )
    return tmp_path


class TestBuildSeedWithSubagents:
    def test_seed_without_subagents(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        config = DeployConfig(agent=AgentConfig(name="test-agent"))
        seed = _build_seed(config, project, "# Main Agent\n")
        assert "subagents" in seed
        assert seed["subagents"] == {}

    def test_seed_with_subagent(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path, subagents=[{"name": "researcher", "description": "Research stuff"}])
        sa_config = SubagentConfig(
            agent=AgentConfig(name="researcher", model="anthropic:claude-haiku-4-5-20251001"),
            sandbox=None, system_prompt="You are a test subagent.", description="Research stuff",
            skills_dir=None, mcp_path=None,
        )
        config = DeployConfig(agent=AgentConfig(name="test-agent"), subagents=[sa_config])
        seed = _build_seed(config, project, "# Main Agent\n")
        assert "researcher" in seed["subagents"]
        sa_seed = seed["subagents"]["researcher"]
        assert sa_seed["system_prompt"] == "You are a test subagent."
        assert sa_seed["description"] == "Research stuff"
        assert "/AGENTS.md" in sa_seed["memories"]

    def test_seed_with_subagent_skills(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path, subagents=[{"name": "researcher", "skills": True}])
        skills_dir = project / "agents" / "researcher" / "skills"
        sa_config = SubagentConfig(
            agent=AgentConfig(name="researcher"), sandbox=None,
            system_prompt="Test.", description="Test",
            skills_dir=skills_dir, mcp_path=None,
        )
        config = DeployConfig(agent=AgentConfig(name="test-agent"), subagents=[sa_config])
        seed = _build_seed(config, project, "# Main Agent\n")
        sa_seed = seed["subagents"]["researcher"]
        assert len(sa_seed["skills"]) > 0
        assert any("SKILL.md" in k for k in sa_seed["skills"])


class TestBundleWithSubagents:
    def test_bundle_copies_subagent_mcp(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path, subagents=[{"name": "researcher", "mcp": True}])
        mcp_path = project / "agents" / "researcher" / "mcp.json"
        sa_config = SubagentConfig(
            agent=AgentConfig(name="researcher"), sandbox=None,
            system_prompt="Test.", description="Test",
            skills_dir=None, mcp_path=mcp_path,
        )
        config = DeployConfig(agent=AgentConfig(name="test-agent"), subagents=[sa_config])
        build_dir = tmp_path / "build"
        bundle(config, project, build_dir)
        assert (build_dir / "_mcp_researcher.json").is_file()

    def test_bundle_seed_has_subagents_key(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path, subagents=[{"name": "researcher"}])
        sa_config = SubagentConfig(
            agent=AgentConfig(name="researcher"), sandbox=None,
            system_prompt="Test.", description="Test",
            skills_dir=None, mcp_path=None,
        )
        config = DeployConfig(agent=AgentConfig(name="test-agent"), subagents=[sa_config])
        build_dir = tmp_path / "build"
        bundle(config, project, build_dir)
        seed = json.loads((build_dir / "_seed.json").read_text())
        assert "subagents" in seed
        assert "researcher" in seed["subagents"]
