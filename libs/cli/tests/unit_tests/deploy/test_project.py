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
