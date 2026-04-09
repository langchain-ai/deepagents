"""End-to-end test for deploy bundling with subagents."""

import ast
import json
from pathlib import Path

from deepagents_cli.deploy.bundler import bundle, print_bundle_summary
from deepagents_cli.deploy.config import (
    AgentConfig,
    DeployConfig,
    SandboxConfig,
    load_subagents,
)


def _scaffold_project(root: Path) -> None:
    """Create a full project with main agent + 2 subagents."""
    (root / "AGENTS.md").write_text("# Main Agent\nYou are the main agent.\n")
    (root / "deepagents.toml").write_text(
        '[agent]\nname = "my-agent"\nmodel = "anthropic:claude-sonnet-4-6"\n\n'
        '[sandbox]\nprovider = "none"\n'
    )

    # Skill for main agent.
    skill_dir = root / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review code\n---\n\nReview the code.\n"
    )

    # Subagent: researcher (with skills, no mcp)
    researcher = root / "agents" / "researcher"
    researcher.mkdir(parents=True)
    (researcher / "AGENTS.md").write_text(
        "---\nname: researcher\ndescription: Research topics\n---\n\n"
        "You are a research assistant.\n"
    )
    (researcher / "deepagents.toml").write_text(
        '[agent]\nname = "researcher"\nmodel = "anthropic:claude-haiku-4-5-20251001"\n'
    )
    r_skill = researcher / "skills" / "summarize"
    r_skill.mkdir(parents=True)
    (r_skill / "SKILL.md").write_text(
        "---\nname: summarize\ndescription: Summarize findings\n---\n\nSummarize.\n"
    )

    # Subagent: reviewer (no skills, with mcp)
    reviewer = root / "agents" / "reviewer"
    reviewer.mkdir(parents=True)
    (reviewer / "AGENTS.md").write_text(
        "---\nname: reviewer\ndescription: Review code changes\n---\n\n"
        "You are a code reviewer.\n"
    )
    (reviewer / "deepagents.toml").write_text(
        '[agent]\nname = "reviewer"\nmodel = "anthropic:claude-sonnet-4-6"\n'
    )
    (reviewer / "mcp.json").write_text(
        '{"mcpServers": {"gh": {"type": "http", "url": "http://localhost:3000/github"}}}'
    )


class TestDeployEndToEnd:
    def test_full_bundle_with_subagents(self, tmp_path: Path) -> None:
        """Bundle a project with two subagents and verify all outputs."""
        project = tmp_path / "project"
        project.mkdir()
        _scaffold_project(project)

        # Load subagents.
        subagents = load_subagents(project)
        assert len(subagents) == 2

        config = DeployConfig(
            agent=AgentConfig(name="my-agent", model="anthropic:claude-sonnet-4-6"),
            subagents=subagents,
        )

        build_dir = tmp_path / "build"
        bundle(config, project, build_dir)

        # Check _seed.json.
        seed = json.loads((build_dir / "_seed.json").read_text())
        assert "researcher" in seed["subagents"]
        assert "reviewer" in seed["subagents"]
        assert len(seed["subagents"]["researcher"]["skills"]) > 0
        assert "/AGENTS.md" in seed["subagents"]["researcher"]["memories"]
        assert "/AGENTS.md" in seed["subagents"]["reviewer"]["memories"]

        # Check subagent MCP file copied.
        assert (build_dir / "_mcp_reviewer.json").is_file()
        assert not (build_dir / "_mcp_researcher.json").exists()

        # Check deploy_graph.py is valid Python and mentions subagents.
        graph_py = (build_dir / "deploy_graph.py").read_text()
        ast.parse(graph_py)
        assert "SUBAGENT_CONFIGS" in graph_py
        assert "researcher" in graph_py
        assert "reviewer" in graph_py
        assert "_load_mcp_tools_reviewer" in graph_py

        # Check pyproject.toml includes MCP dep (reviewer has mcp).
        pyproject = (build_dir / "pyproject.toml").read_text()
        assert "langchain-mcp-adapters" in pyproject

        # Smoke-test print_bundle_summary (should not raise).
        print_bundle_summary(config, build_dir)

    def test_bundle_without_subagents_still_works(self, tmp_path: Path) -> None:
        """Existing projects without agents/ dir still bundle correctly."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "AGENTS.md").write_text("# Agent\nYou are an agent.\n")

        config = DeployConfig(
            agent=AgentConfig(name="simple-agent"),
        )

        build_dir = tmp_path / "build"
        bundle(config, project, build_dir)

        seed = json.loads((build_dir / "_seed.json").read_text())
        assert seed["subagents"] == {}

        graph_py = (build_dir / "deploy_graph.py").read_text()
        ast.parse(graph_py)
        assert "SUBAGENT_CONFIGS = []" in graph_py
