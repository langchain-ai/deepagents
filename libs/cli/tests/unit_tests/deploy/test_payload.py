"""Snapshot tests for build_payload over fixture projects."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepagents_cli.deploy.payload import (
    build_directory_delta,
    build_directory_files,
    build_metadata_payload,
    build_payload,
)
from deepagents_cli.deploy.project import Project

_FIXTURES = Path(__file__).parent / "fixtures" / "projects"

_FIXTURE_NAMES = [
    "bare",
    "with_tools",
    "with_skills",
    "with_subagents",
    "subagent_with_local_skills",
]


@pytest.mark.parametrize("name", _FIXTURE_NAMES)
def test_create_payload_matches_expected(name: str) -> None:
    project = Project.load(_FIXTURES / name)
    payload = build_payload(project, mode="create")
    expected = json.loads(
        (_FIXTURES / name / "expected_payload.json").read_text(encoding="utf-8")
    )
    assert payload == expected


def test_metadata_payload_omits_directory_backed_fields() -> None:
    project = Project.load(_FIXTURES / "with_subagents")
    payload = build_metadata_payload(project)
    assert payload["name"] == "parent"
    assert "system_prompt" not in payload
    assert "tools" not in payload
    assert "skills" not in payload
    assert "subagents" not in payload
    assert "files" not in payload


def test_create_payload_normalizes_legacy_sandbox_backend(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        '{"name": "x", "backend": {"type": "sandbox"}}'
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    project = Project.load(tmp_path)
    payload = build_payload(project, mode="create")
    assert payload["backend"] == {
        "type": "sandbox",
        "sandbox_config": {"scope": "thread"},
    }


def test_create_payload_normalizes_legacy_default_backend(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        '{"name": "x", "backend": {"type": "default"}}'
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    project = Project.load(tmp_path)
    payload = build_payload(project, mode="create")
    assert payload["backend"] == {"type": "state"}


def test_create_payload_allows_state_backend(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x", "backend": {"type": "state"}}')
    (tmp_path / "AGENTS.md").write_text("hi")
    project = Project.load(tmp_path)
    payload = build_payload(project, mode="create")
    assert payload["backend"] == {"type": "state"}


def test_create_payload_allows_sandbox_config(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        """
        {
          "name": "x",
          "backend": {
            "type": "sandbox",
            "sandbox_config": {
              "scope": "thread",
              "policy_ids": ["p-1"]
            }
          }
        }
        """
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    project = Project.load(tmp_path)
    payload = build_payload(project, mode="create")
    assert payload["backend"] == {
        "type": "sandbox",
        "sandbox_config": {"scope": "thread", "policy_ids": ["p-1"]},
    }


def test_create_payload_compiles_model_shorthand(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        '{"name": "x", "model": "anthropic:claude-sonnet-4-6"}'
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    project = Project.load(tmp_path)
    payload = build_payload(project, mode="create")
    assert payload["runtime"] == {"model": {"model_id": "anthropic:claude-sonnet-4-6"}}


def test_metadata_payload_compiles_model_shorthand(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        '{"name": "x", "model": "anthropic:claude-sonnet-4-6"}'
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    project = Project.load(tmp_path)
    payload = build_metadata_payload(project)
    assert payload["runtime"] == {"model": {"model_id": "anthropic:claude-sonnet-4-6"}}


def test_directory_files_include_project_and_subagent_sources() -> None:
    project = Project.load(_FIXTURES / "subagent_with_local_skills")
    files = build_directory_files(project)
    assert "AGENTS.md" in files
    assert "subagents/researcher/AGENTS.md" in files
    assert (
        'model_id: "anthropic:claude-sonnet-4-6"'
        in files["subagents/researcher/AGENTS.md"]
    )
    assert "subagents/researcher/tools.json" in files
    assert "subagents/researcher/skills/note/SKILL.md" in files


def test_directory_delta_deletes_removed_managed_files_only() -> None:
    remote = {
        "AGENTS.md": {"type": "file", "content": "old"},
        "skills/old/SKILL.md": {"type": "file", "content": "delete me"},
        "README.md": {"type": "file", "content": "keep me"},
    }
    delta = build_directory_delta(remote, {"AGENTS.md": "new"})
    assert delta == {
        "AGENTS.md": {"type": "file", "content": "new"},
        "skills/old/SKILL.md": None,
    }
