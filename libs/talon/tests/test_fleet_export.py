from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.config import TalonConfig
from deepagents_talon.fleet import FleetAgentComponents
from deepagents_talon.fleet_export import (
    FleetExportValidationError,
    fleet_tool_entries,
    validate_fleet_export,
)
from deepagents_talon.fleet_manifest import build_fleet_run_manifest

if TYPE_CHECKING:
    from pathlib import Path


def test_validate_fleet_export_rejects_missing_required_files(tmp_path: Path) -> None:
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()

    with pytest.raises(FleetExportValidationError, match=r"AGENTS\.md"):
        validate_fleet_export(fleet_dir)

    (fleet_dir / "AGENTS.md").write_text("Fleet prompt", encoding="utf-8")

    with pytest.raises(FleetExportValidationError, match=r"config\.json"):
        validate_fleet_export(fleet_dir)

    (fleet_dir / "config.json").write_text(
        json.dumps({"model": "openai:gpt-5-mini"}), encoding="utf-8"
    )

    with pytest.raises(FleetExportValidationError, match=r"tools\.json"):
        validate_fleet_export(fleet_dir)


def test_validate_fleet_export_rejects_malformed_tools_json(tmp_path: Path) -> None:
    fleet_dir = _fleet_export(tmp_path)
    (fleet_dir / "tools.json").write_text(json.dumps({"tools": [{}]}), encoding="utf-8")

    with pytest.raises(FleetExportValidationError, match=r"tools\[0\]\.name"):
        validate_fleet_export(fleet_dir)


def test_fleet_tool_entries_discovers_subagents_and_strips_url_secrets(tmp_path: Path) -> None:
    fleet_dir = _fleet_export(tmp_path)
    _write_tools(
        fleet_dir / "tools.json",
        [
            {
                "name": "search",
                "mcp_server_url": "https://tools.example/mcp?token=secret#fragment",
                "auth_type": "oauth",
            }
        ],
    )
    subagent_dir = fleet_dir / "subagents" / "researcher"
    subagent_dir.mkdir(parents=True)
    _write_tools(
        subagent_dir / "tools.json",
        [
            {
                "name": "calendar",
                "mcp_server_url": "https://calendar.example/mcp?token=secret",
                "headers": {"Authorization": "Bearer secret"},
            }
        ],
    )

    entries = fleet_tool_entries(fleet_dir)

    assert [(entry.scope, entry.tool_name) for entry in entries] == [
        ("root", "search"),
        ("subagent:researcher", "calendar"),
    ]
    assert entries[0].mcp_server_url == "https://tools.example/mcp"
    assert entries[1].mcp_server_url == "https://calendar.example/mcp"
    assert entries[1].auth_path == "headers"


def test_manifest_summarizes_tool_and_approval_requirements(tmp_path: Path) -> None:
    fleet_dir = _fleet_export(tmp_path)
    _write_tools(
        fleet_dir / "tools.json",
        [
            {
                "name": "search",
                "mcp_server_url": "https://tools.example/mcp?token=one",
            },
            {
                "name": "crawl",
                "mcp_server_url": "https://tools.example/mcp?token=two",
            },
        ],
    )
    subagent_dir = fleet_dir / "subagents" / "researcher"
    subagent_dir.mkdir(parents=True)
    _write_tools(
        subagent_dir / "tools.json",
        [{"name": "search", "mcp_server_url": "https://tools.example/mcp?token=three"}],
    )
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "DEEPAGENTS_TALON_FLEET_DIR": str(fleet_dir),
        },
        base_home=tmp_path,
    )

    manifest = build_fleet_run_manifest(
        config,
        FleetAgentComponents(
            model="openai:gpt-5-mini",
            system_prompt="fleet prompt",
            tools=(SimpleNamespace(name="search"),),
            subagents=(),
            interrupt_on={"crawl": True, "send_email": True},
        ),
    )

    requirements = {(item.scope, item.tool_name): item for item in manifest.tool_requirements}
    assert requirements[("root", "search")].loaded is True
    assert requirements[("root", "crawl")].loaded is False
    assert requirements[("subagent:researcher", "search")].loaded is True
    assert {item.mcp_server_url for item in manifest.tool_requirements} == {
        "https://tools.example/mcp"
    }
    assert manifest.approval_tool_names == ("crawl", "send_email")
    assert len(manifest.setup_tasks) == 3
    assert {task.target_path for task in manifest.setup_tasks} == {str(config.home / ".mcp.json")}


def test_validate_fleet_export_allows_missing_subagent_tool_content(tmp_path: Path) -> None:
    fleet_dir = _fleet_export(tmp_path)
    (fleet_dir / "subagents" / "researcher").mkdir(parents=True)

    validate_fleet_export(fleet_dir)

    assert fleet_tool_entries(fleet_dir) == []


def _fleet_export(tmp_path: Path) -> Path:
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / "AGENTS.md").write_text("Fleet prompt", encoding="utf-8")
    (fleet_dir / "config.json").write_text(
        json.dumps({"model": "openai:gpt-5-mini"}),
        encoding="utf-8",
    )
    _write_tools(fleet_dir / "tools.json", [])
    return fleet_dir


def _write_tools(path: Path, tools: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"tools": tools}), encoding="utf-8")
