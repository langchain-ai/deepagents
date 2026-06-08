"""Tests for Project.load()."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepagents_cli.deploy.project import Project, ProjectError

_FIXTURES = Path(__file__).parent / "fixtures" / "projects"


def _write_minimal_project(root: Path) -> None:
    (root / "agent.json").write_text('{"name": "x"}')
    (root / "AGENTS.md").write_text("hi")


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


def test_load_bare_project_reads_agent_json_and_agents_md() -> None:
    proj = Project.load(_FIXTURES / "bare")
    assert proj.name == "research-assistant"
    assert proj.description == "Researches a topic and returns a summary."
    assert "careful research assistant" in proj.system_prompt
    assert proj.tools is None
    assert proj.skills == []
    assert proj.subagents == []
    assert proj.runtime is None
    assert proj.backend is None
    assert proj.permissions is None
    assert proj.tools_text is None
    assert proj.agent_id is None
    assert proj.model is None


def test_load_missing_agent_json_raises(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match=r"agent\.json"):
        Project.load(tmp_path)


def test_load_missing_agents_md_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    with pytest.raises(ProjectError, match=r"AGENTS\.md"):
        Project.load(tmp_path)


def test_load_invalid_agent_json_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text("{not json")
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match=r"agent\.json"):
        Project.load(tmp_path)


def test_load_missing_name_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"description": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match="name"):
        Project.load(tmp_path)


def test_top_level_symlink_file_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    target = tmp_path.parent / "prompt.txt"
    target.write_text("outside project")
    _symlink_or_skip(tmp_path / "AGENTS.md", target)

    with pytest.raises(ProjectError, match="symlinks are not allowed"):
        Project.load(tmp_path)


def test_runtime_and_permissions_round_trip(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        """
        {
          "name": "x",
          "backend": {
            "type": "thread_scoped_sandbox",
            "sandbox": {"policy_ids": ["p-1"]}
          },
          "runtime": {
            "model": {"model_id": "anthropic:claude-sonnet-4-6"}
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
    assert proj.runtime == {"model": {"model_id": "anthropic:claude-sonnet-4-6"}}
    assert proj.backend == {
        "type": "sandbox",
        "sandbox_config": {"scope": "thread", "policy_ids": ["p-1"]},
    }
    assert proj.permissions == {
        "identity": "personal",
        "visibility": "tenant",
        "tenant_access_level": "read",
    }
    assert proj.model is None


def test_model_round_trip(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        '{"name": "x", "model": "  anthropic:claude-sonnet-4-6  "}'
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    proj = Project.load(tmp_path)
    assert proj.model == "anthropic:claude-sonnet-4-6"
    assert proj.runtime is None


@pytest.mark.parametrize("model", ["", 123])
def test_model_must_be_non_empty_string(tmp_path: Path, model: object) -> None:
    (tmp_path / "agent.json").write_text(
        '{"name": "x", "model": ' + json.dumps(model) + "}"
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match="model"):
        Project.load(tmp_path)


def test_model_and_runtime_conflict(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        """
        {
          "name": "x",
          "model": "anthropic:claude-sonnet-4-6",
          "runtime": {
            "model": {"model_id": "anthropic:claude-sonnet-4-6"}
          }
        }
        """
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match="either top-level `model` or `runtime`"):
        Project.load(tmp_path)


def test_agent_id_round_trip(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x", "agent_id": "  a-1  "}')
    (tmp_path / "AGENTS.md").write_text("hi")
    proj = Project.load(tmp_path)
    assert proj.agent_id == "a-1"


def test_agent_id_must_be_non_empty_string(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x", "agent_id": ""}')
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match="agent_id"):
        Project.load(tmp_path)


def test_runtime_backend_type_raises_migration_error(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        '{"name": "x", "runtime": {"backend_type": "sandbox"}}'
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match=r"runtime\.backend_type") as excinfo:
        Project.load(tmp_path)
    assert "sandbox_config" in str(excinfo.value)


def test_sandbox_backend_type_defaults_to_thread_scope(
    tmp_path: Path,
) -> None:
    (tmp_path / "agent.json").write_text(
        """
        {
          "name": "x",
          "backend": {
            "type": "sandbox",
            "sandbox": {"policy_ids": ["p-1"]}
          }
        }
        """
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    proj = Project.load(tmp_path)
    assert proj.backend == {
        "type": "sandbox",
        "sandbox_config": {"scope": "thread", "policy_ids": ["p-1"]},
    }


def test_legacy_default_backend_type_normalizes_to_state(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        '{"name": "x", "backend": {"type": "default"}}'
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    proj = Project.load(tmp_path)
    assert proj.backend == {"type": "state"}


def test_state_backend_type_is_allowed(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x", "backend": {"type": "state"}}')
    (tmp_path / "AGENTS.md").write_text("hi")
    proj = Project.load(tmp_path)
    assert proj.backend == {"type": "state"}


def test_agent_scoped_sandbox_backend_type_is_allowed(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        '{"name": "x", "backend": {"type": "agent_scoped_sandbox"}}'
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    proj = Project.load(tmp_path)
    assert proj.backend == {"type": "sandbox", "sandbox_config": {"scope": "agent"}}


def test_sandbox_backend_config_is_allowed(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        """
        {
          "name": "x",
          "backend": {
            "type": "sandbox",
            "sandbox_config": {
              "scope": "agent",
              "policy_ids": ["p-1"],
              "idle_ttl_seconds": 900,
              "delete_after_stop_seconds": 300
            }
          }
        }
        """
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    proj = Project.load(tmp_path)
    assert proj.backend == {
        "type": "sandbox",
        "sandbox_config": {
            "scope": "agent",
            "policy_ids": ["p-1"],
            "idle_ttl_seconds": 900,
            "delete_after_stop_seconds": 300,
        },
    }


def test_invalid_backend_type_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        '{"name": "x", "backend": {"type": "unknown_sandbox"}}'
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match=r"backend\.type"):
        Project.load(tmp_path)


def test_sandbox_settings_with_default_backend_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        '{"name": "x", "backend": {"type": "default", "sandbox": {}}}'
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match=r"sandbox settings"):
        Project.load(tmp_path)


def test_sandbox_policy_ids_must_be_strings(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        """
        {
          "name": "x",
          "backend": {
            "type": "thread_scoped_sandbox",
            "sandbox": {"policy_ids": ["p-1", 2]}
          }
        }
        """
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match=r"policy_ids"):
        Project.load(tmp_path)


def test_sandbox_ttl_fields_must_be_integers(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        """
        {
          "name": "x",
          "backend": {
            "type": "thread_scoped_sandbox",
            "sandbox": {"idle_ttl_seconds": true}
          }
        }
        """
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match=r"idle_ttl_seconds"):
        Project.load(tmp_path)


def test_sandbox_config_scope_must_be_valid(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        """
        {
          "name": "x",
          "backend": {
            "type": "sandbox",
            "sandbox_config": {"scope": "workspace"}
          }
        }
        """
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match=r"scope"):
        Project.load(tmp_path)


def test_sandbox_config_overrides_legacy_sandbox_config(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        """
        {
          "name": "x",
          "backend": {
            "type": "thread_scoped_sandbox",
            "sandbox": {"policy_ids": ["old"], "idle_ttl_seconds": 900},
            "sandbox_config": {"policy_ids": ["new"]}
          }
        }
        """
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    proj = Project.load(tmp_path)
    assert proj.backend == {
        "type": "sandbox",
        "sandbox_config": {
            "scope": "thread",
            "policy_ids": ["new"],
            "idle_ttl_seconds": 900,
        },
    }


def test_load_with_tools_reads_tools_json() -> None:
    proj = Project.load(_FIXTURES / "with_tools")
    assert proj.tools is not None
    assert proj.tools_text is not None
    assert proj.tools["tools"][0]["name"] == "tavily_web_search"
    assert proj.tools["tools"][0]["mcp_server_url"] == "https://tools.langchain.com"
    assert (
        proj.tools["interrupt_config"][
            "https://tools.langchain.com::tavily_web_search::Fleet"
        ]
        is True
    )


def test_invalid_tools_json_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    (tmp_path / "tools.json").write_text("[]")  # array, not object
    with pytest.raises(ProjectError, match=r"tools\.json"):
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
    assert skill.skill_file.startswith("---")


def test_top_level_skill_files_are_recursive(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    skill_dir = tmp_path / "skills" / "guide"
    (skill_dir / "data").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: guide\ndescription: read nested data\n---\nUse data.\n"
    )
    (skill_dir / "data" / "facts.md").write_text("nested")

    project = Project.load(tmp_path)

    assert project.skills[0].files["data/facts.md"] == "nested"


def test_skill_extra_file_symlink_raises(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path)
    skill_dir = tmp_path / "skills" / "guide"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: guide\ndescription: read nested data\n---\nUse data.\n"
    )
    target = tmp_path.parent / "secret.txt"
    target.write_text("outside project")
    _symlink_or_skip(skill_dir / "secret.txt", target)

    with pytest.raises(ProjectError, match="symlinks are not allowed"):
        Project.load(tmp_path)


def test_skill_directory_symlink_raises(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path)
    target = tmp_path.parent / "external-skill"
    target.mkdir()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _symlink_or_skip(skills_dir / "external", target)

    with pytest.raises(ProjectError, match="symlinks are not allowed"):
        Project.load(tmp_path)


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
        (d / "SKILL.md").write_text("---\nname: same\ndescription: x\n---\nhi\n")
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


def _write_subagent(root: Path, agent_json: str) -> None:
    _write_minimal_project(root)
    sa = root / "subagents" / "researcher"
    sa.mkdir(parents=True)
    (sa / "agent.json").write_text(agent_json)
    (sa / "AGENTS.md").write_text("Research.")


def test_subagent_model_round_trip(tmp_path: Path) -> None:
    _write_subagent(tmp_path, '{"model": "  anthropic:claude-sonnet-4-6  "}')
    proj = Project.load(tmp_path)
    assert proj.subagents[0].model_id == "anthropic:claude-sonnet-4-6"


def test_subagent_legacy_model_id_still_supported(tmp_path: Path) -> None:
    _write_subagent(tmp_path, '{"model_id": "anthropic:claude-sonnet-4-6"}')
    proj = Project.load(tmp_path)
    assert proj.subagents[0].model_id == "anthropic:claude-sonnet-4-6"


def test_subagent_model_and_model_id_conflict_raises(tmp_path: Path) -> None:
    _write_subagent(tmp_path, '{"model": "a:b", "model_id": "a:b"}')
    with pytest.raises(ProjectError, match="either `model` or `model_id`"):
        Project.load(tmp_path)


@pytest.mark.parametrize("model", ["", 123])
def test_subagent_model_must_be_non_empty_string(tmp_path: Path, model: object) -> None:
    _write_subagent(tmp_path, '{"model": ' + json.dumps(model) + "}")
    with pytest.raises(ProjectError, match="model"):
        Project.load(tmp_path)


def test_subagent_local_skills_go_into_extra_files() -> None:
    proj = Project.load(_FIXTURES / "subagent_with_local_skills")
    sa = proj.subagents[0]
    assert "skills/note/SKILL.md" in sa.extra_files
    assert "Take a note." in sa.extra_files["skills/note/SKILL.md"]


def test_subagent_local_skill_file_symlink_raises(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path)
    sa = tmp_path / "subagents" / "researcher"
    skill_dir = sa / "skills" / "note"
    skill_dir.mkdir(parents=True)
    (sa / "agent.json").write_text('{"description": "Researches."}')
    (sa / "AGENTS.md").write_text("Research.")
    target = tmp_path.parent / "secret.txt"
    target.write_text("outside project")
    _symlink_or_skip(skill_dir / "secret.txt", target)

    with pytest.raises(ProjectError, match="symlinks are not allowed"):
        Project.load(tmp_path)


def test_subagent_missing_agent_json_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    sa = tmp_path / "subagents" / "broken"
    sa.mkdir(parents=True)
    (sa / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match=r"agent\.json"):
        Project.load(tmp_path)


def test_subagent_duplicate_names_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    from deepagents_cli.deploy.project import _read_subagents

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


def test_legacy_deepagents_toml_raises_migration_hint(tmp_path: Path) -> None:
    (tmp_path / "deepagents.toml").write_text('[agent]\nname = "x"\n')
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match=r"legacy deepagents\.toml"):
        Project.load(tmp_path)


def test_legacy_mcp_json_raises_migration_hint(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    (tmp_path / "mcp.json").write_text('{"mcpServers": {}}')
    with pytest.raises(ProjectError, match=r"mcp\.json"):
        Project.load(tmp_path)
