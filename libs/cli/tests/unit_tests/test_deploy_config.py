"""Unit tests for deploy subagent config loading."""

from pathlib import Path

import pytest

from deepagents_cli.deploy.config import (
    AGENTS_MD_FILENAME,
    DeployConfig,
    SubagentConfig,
    AgentConfig,
    load_subagents,
)


def _write_subagent(
    agents_dir: Path,
    name: str,
    *,
    description: str = "A test subagent",
    model: str = "anthropic:claude-haiku-4-5-20251001",
    system_prompt: str = "You are a test assistant.",
    include_toml: bool = True,
    include_skills: bool = False,
    include_mcp: bool = False,
    sandbox_provider: str | None = None,
) -> Path:
    sa_dir = agents_dir / name
    sa_dir.mkdir(parents=True, exist_ok=True)
    (sa_dir / "AGENTS.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{system_prompt}\n"
    )
    if include_toml:
        sandbox_section = ""
        if sandbox_provider:
            sandbox_section = f'\n[sandbox]\nprovider = "{sandbox_provider}"\n'
        (sa_dir / "deepagents.toml").write_text(
            f'[agent]\nname = "{name}"\nmodel = "{model}"\n{sandbox_section}'
        )
    if include_skills:
        skill_dir = sa_dir / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A test skill\n---\n\nDo the thing.\n"
        )
    if include_mcp:
        (sa_dir / "mcp.json").write_text(
            '{"mcpServers": {"test": {"type": "http", "url": "http://localhost:8080"}}}'
        )
    return sa_dir


class TestLoadSubagents:
    def test_no_agents_dir(self, tmp_path: Path) -> None:
        result = load_subagents(tmp_path)
        assert result == []

    def test_empty_agents_dir(self, tmp_path: Path) -> None:
        (tmp_path / "agents").mkdir()
        result = load_subagents(tmp_path)
        assert result == []

    def test_single_subagent(self, tmp_path: Path) -> None:
        _write_subagent(tmp_path / "agents", "researcher")
        result = load_subagents(tmp_path)
        assert len(result) == 1
        assert result[0].agent.name == "researcher"
        assert result[0].agent.model == "anthropic:claude-haiku-4-5-20251001"
        assert result[0].description == "A test subagent"
        assert "test assistant" in result[0].system_prompt

    def test_subagent_without_toml_uses_defaults(self, tmp_path: Path) -> None:
        _write_subagent(tmp_path / "agents", "researcher", include_toml=False)
        result = load_subagents(tmp_path)
        assert result[0].agent.model == "anthropic:claude-sonnet-4-6"

    def test_subagent_sandbox_inheritance(self, tmp_path: Path) -> None:
        _write_subagent(tmp_path / "agents", "researcher")
        result = load_subagents(tmp_path)
        assert result[0].sandbox is None

    def test_subagent_sandbox_override(self, tmp_path: Path) -> None:
        _write_subagent(tmp_path / "agents", "coder", sandbox_provider="modal")
        result = load_subagents(tmp_path)
        assert result[0].sandbox is not None
        assert result[0].sandbox.provider == "modal"

    def test_subagent_with_skills(self, tmp_path: Path) -> None:
        _write_subagent(tmp_path / "agents", "researcher", include_skills=True)
        result = load_subagents(tmp_path)
        assert result[0].skills_dir is not None
        assert result[0].skills_dir.is_dir()

    def test_subagent_with_mcp(self, tmp_path: Path) -> None:
        _write_subagent(tmp_path / "agents", "researcher", include_mcp=True)
        result = load_subagents(tmp_path)
        assert result[0].mcp_path is not None
        assert result[0].mcp_path.is_file()

    def test_multiple_subagents_sorted(self, tmp_path: Path) -> None:
        _write_subagent(tmp_path / "agents", "writer")
        _write_subagent(tmp_path / "agents", "researcher")
        result = load_subagents(tmp_path)
        assert len(result) == 2
        assert result[0].agent.name == "researcher"
        assert result[1].agent.name == "writer"

    def test_missing_agents_md_raises(self, tmp_path: Path) -> None:
        sa_dir = tmp_path / "agents" / "broken"
        sa_dir.mkdir(parents=True)
        (sa_dir / "deepagents.toml").write_text('[agent]\nname = "broken"\n')
        with pytest.raises(ValueError, match="AGENTS.md not found"):
            load_subagents(tmp_path)

    def test_missing_frontmatter_description_raises(self, tmp_path: Path) -> None:
        sa_dir = tmp_path / "agents" / "bad"
        sa_dir.mkdir(parents=True)
        (sa_dir / "AGENTS.md").write_text("---\nname: bad\n---\n\nNo description.\n")
        with pytest.raises(ValueError, match="description"):
            load_subagents(tmp_path)

    def test_name_mismatch_toml_vs_frontmatter_raises(self, tmp_path: Path) -> None:
        sa_dir = tmp_path / "agents" / "researcher"
        sa_dir.mkdir(parents=True)
        (sa_dir / "AGENTS.md").write_text("---\nname: researcher\ndescription: Research\n---\n\nPrompt.\n")
        (sa_dir / "deepagents.toml").write_text('[agent]\nname = "wrong-name"\n')
        with pytest.raises(ValueError, match="mismatched name"):
            load_subagents(tmp_path)

    def test_reserved_name_general_purpose_raises(self, tmp_path: Path) -> None:
        _write_subagent(tmp_path / "agents", "general-purpose")
        with pytest.raises(ValueError, match="reserved"):
            load_subagents(tmp_path)

    def test_duplicate_names_raises(self, tmp_path: Path) -> None:
        _write_subagent(tmp_path / "agents", "researcher")
        sa_dir2 = tmp_path / "agents" / "researcher-v2"
        sa_dir2.mkdir(parents=True)
        (sa_dir2 / "AGENTS.md").write_text("---\nname: researcher\ndescription: Dup\n---\n\nPrompt.\n")
        with pytest.raises(ValueError, match="Duplicate subagent name"):
            load_subagents(tmp_path)

    def test_skips_non_directories(self, tmp_path: Path) -> None:
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "stray-file.md").write_text("not a subagent")
        _write_subagent(agents_dir, "researcher")
        result = load_subagents(tmp_path)
        assert len(result) == 1

    def test_subagent_mcp_stdio_fails_validation(self, tmp_path: Path) -> None:
        sa_dir = _write_subagent(tmp_path / "agents", "researcher", include_mcp=False)
        (sa_dir / "mcp.json").write_text('{"mcpServers": {"local": {"type": "stdio", "command": "node"}}}')
        result = load_subagents(tmp_path)
        (tmp_path / "AGENTS.md").write_text("# Main agent\n")
        config = DeployConfig(agent=AgentConfig(name="test"), subagents=result)
        errors = config.validate(tmp_path)
        assert any("stdio" in e for e in errors)
