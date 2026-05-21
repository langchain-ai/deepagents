"""Tests for Project.load() (parsing agent.json/AGENTS.md/tools.json/skills/subagents)."""

from __future__ import annotations

from pathlib import Path

import pytest

from deepagents_cli.deploy.project import Project, ProjectError

_FIXTURES = Path(__file__).parent / "fixtures" / "projects"


def test_load_bare_project_reads_agent_json_and_agents_md() -> None:
    proj = Project.load(_FIXTURES / "bare")
    assert proj.name == "research-assistant"
    assert proj.description == "Researches a topic and returns a summary."
    assert "careful research assistant" in proj.system_prompt
    assert proj.tools is None
    assert proj.skills == []
    assert proj.subagents == []
    assert proj.runtime is None
    assert proj.permissions is None


def test_load_missing_agent_json_raises(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match="agent.json"):
        Project.load(tmp_path)


def test_load_missing_agents_md_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    with pytest.raises(ProjectError, match="AGENTS.md"):
        Project.load(tmp_path)


def test_load_invalid_agent_json_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text("{not json")
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match="agent.json"):
        Project.load(tmp_path)


def test_load_missing_name_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"description": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match="name"):
        Project.load(tmp_path)


def test_runtime_and_permissions_round_trip(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        """
        {
          "name": "x",
          "runtime": {
            "model": {"model_id": "anthropic:claude-sonnet-4-6"},
            "backend_type": "thread_scoped_sandbox"
          },
          "permissions": {
            "identity": "personal",
            "visibility": "tenant",
            "tenant_access_level": "read"
          }
        }
        """
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    proj = Project.load(tmp_path)
    assert proj.runtime == {
        "model": {"model_id": "anthropic:claude-sonnet-4-6"},
        "backend_type": "thread_scoped_sandbox",
    }
    assert proj.permissions == {
        "identity": "personal",
        "visibility": "tenant",
        "tenant_access_level": "read",
    }


def test_invalid_runtime_backend_type_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        '{"name": "x", "runtime": {"backend_type": "lol_unknown"}}'
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match="backend_type"):
        Project.load(tmp_path)


def test_load_with_tools_reads_tools_json() -> None:
    proj = Project.load(_FIXTURES / "with_tools")
    assert proj.tools is not None
    assert proj.tools["tools"][0]["name"] == "tavily_web_search"
    assert proj.tools["tools"][0]["mcp_server_url"] == "https://tools.langchain.com"
    assert proj.tools["interrupt_config"][
        "https://tools.langchain.com::tavily_web_search::Fleet"
    ] is True


def test_invalid_tools_json_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    (tmp_path / "tools.json").write_text("[]")  # array, not object
    with pytest.raises(ProjectError, match="tools.json"):
        Project.load(tmp_path)


def test_tools_missing_mcp_server_url_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    (tmp_path / "tools.json").write_text(
        '{"tools": [{"name": "search"}], "interrupt_config": {}}'
    )
    with pytest.raises(ProjectError, match="mcp_server_url"):
        Project.load(tmp_path)


def test_load_with_skills_parses_frontmatter_and_files() -> None:
    proj = Project.load(_FIXTURES / "with_skills")
    assert len(proj.skills) == 1
    skill = proj.skills[0]
    assert skill.name == "summarize"
    assert skill.description == "Summarise text into a one-paragraph summary."
    assert "one-paragraph summary" in skill.instructions
    assert "examples.md" in skill.files
    assert "Example 1" in skill.files["examples.md"]


def test_skill_missing_frontmatter_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    skill_dir = tmp_path / "skills" / "bad"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# No frontmatter here\n")
    with pytest.raises(ProjectError, match="frontmatter"):
        Project.load(tmp_path)


def test_skill_duplicate_names_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    for dirname in ("a", "b"):
        d = tmp_path / "skills" / dirname
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            "---\nname: same\ndescription: x\n---\nhi\n"
        )
    with pytest.raises(ProjectError, match="duplicate"):
        Project.load(tmp_path)


def test_load_with_subagents() -> None:
    proj = Project.load(_FIXTURES / "with_subagents")
    assert len(proj.subagents) == 1
    sa = proj.subagents[0]
    assert sa.name == "researcher"
    assert sa.description == "Researches a topic."
    assert sa.model_id == "anthropic:claude-sonnet-4-6"
    assert "research a topic" in sa.instructions
    assert sa.tools is not None
    assert sa.tools["tools"][0]["name"] == "search"
    assert sa.extra_files == {}


def test_subagent_local_skills_go_into_extra_files() -> None:
    proj = Project.load(_FIXTURES / "subagent_with_local_skills")
    sa = proj.subagents[0]
    assert "skills/note/SKILL.md" in sa.extra_files
    assert "Take a note." in sa.extra_files["skills/note/SKILL.md"]


def test_subagent_missing_agent_json_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    sa = tmp_path / "subagents" / "broken"
    sa.mkdir(parents=True)
    (sa / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match="agent.json"):
        Project.load(tmp_path)


def test_subagent_duplicate_names_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    from deepagents_cli.deploy.project import _read_subagents  # noqa: PLC0415
    sa1 = tmp_path / "subagents" / "x"
    sa1.mkdir(parents=True)
    (sa1 / "agent.json").write_text("{}")
    (sa1 / "AGENTS.md").write_text("hi")
    sa2 = tmp_path / "subagents" / "X"
    if sa1.resolve() == sa2.resolve():  # case-insensitive FS — synthesize
        pytest.skip("case-insensitive FS")
    try:
        sa2.mkdir(parents=True)
    except FileExistsError:
        pytest.skip("case-insensitive FS")
    (sa2 / "agent.json").write_text("{}")
    (sa2 / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match="duplicate"):
        _read_subagents(tmp_path)
