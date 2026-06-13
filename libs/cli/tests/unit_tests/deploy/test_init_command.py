"""Tests for `deepagents init`."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

import pytest

from deepagents_cli.deploy.commands import execute_init_command

if TYPE_CHECKING:
    from pathlib import Path


def _ns(name: str | None, *, force: bool = False) -> argparse.Namespace:
    return argparse.Namespace(name=name, force=force)


def test_init_scaffolds_new_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    execute_init_command(_ns("my-agent"))
    project = tmp_path / "my-agent"
    assert (project / "agent.json").is_file()
    agent = json.loads((project / "agent.json").read_text())
    assert agent["name"] == "my-agent"
    assert agent["model"] == "openai:gpt-5.5"
    assert agent["backend"] == {"type": "state"}
    assert "runtime" not in agent
    assert (project / "AGENTS.md").is_file()
    assert (project / ".gitignore").is_file()
    assert not (project / ".env").exists()
    gitignore = (project / ".gitignore").read_text()
    assert ".env" in gitignore
    assert ".deepagents/" not in gitignore
    assert (project / "skills").is_dir()
    assert (project / "skills" / "example-skill" / "SKILL.md").is_file()
    tools = json.loads((project / "tools.json").read_text())
    assert tools["tools"] == []
    subagent = project / "subagents" / "researcher"
    assert (subagent / "agent.json").is_file()
    assert (subagent / "AGENTS.md").is_file()
    subagent_cfg = json.loads((subagent / "agent.json").read_text())
    assert subagent_cfg["model"] == "openai:gpt-5.5"


def test_init_scaffold_loads_as_valid_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_cli.deploy.project import Project

    monkeypatch.chdir(tmp_path)
    execute_init_command(_ns("my-agent"))
    project = Project.load(tmp_path / "my-agent")
    assert project.backend == {"type": "state"}
    assert project.tools is not None
    assert len(project.skills) == 1
    assert project.skills[0].name == "example-skill"
    assert len(project.subagents) == 1
    assert project.subagents[0].name == "researcher"
    assert project.subagents[0].model_id == "openai:gpt-5.5"


def test_init_prints_welcome_walkthrough(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    execute_init_command(_ns("my-agent"))
    out = capsys.readouterr().out
    # Walkthrough covers editing, the API key prereq + MCP servers
    # (API key and OAuth auth), and deploy.
    assert "Next steps" in out
    assert "AGENTS.md" in out
    assert "LANGSMITH_API_KEY" in out
    assert "mcp-servers" in out
    assert "OAuth" in out
    assert "deepagents deploy" in out


def test_init_refuses_existing_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x").mkdir()
    with pytest.raises(SystemExit):
        execute_init_command(_ns("x"))


def test_init_force_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "agent.json").write_text("{}")
    execute_init_command(_ns("x", force=True))
    agent = json.loads((tmp_path / "x" / "agent.json").read_text())
    assert agent["name"] == "x"
