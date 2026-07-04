from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.config import TalonConfig
from deepagents_talon.fleet import FleetAgentComponents
from deepagents_talon.fleet_manifest import (
    ChannelSelection,
    FleetRunManifestValidationError,
    build_fleet_run_manifest,
    load_fleet_run_manifest,
    manifest_path,
    refresh_fleet_run_manifest,
    write_fleet_run_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_manifest_serializes_empty_tool_requirements(tmp_path: Path) -> None:
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
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
            tools=(),
            subagents=(),
            interrupt_on=None,
        ),
        selected_channel=ChannelSelection(provider="telegram", source="cli"),
        created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )

    path = manifest_path(config.home)
    write_fleet_run_manifest(path, manifest)
    loaded = load_fleet_run_manifest(path)

    assert path == tmp_path / "assistant" / "fleet-run-manifest.json"
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600
    assert loaded.assistant_id == "assistant"
    assert loaded.fleet_dir == str(fleet_dir)
    assert loaded.selected_channel == ChannelSelection(provider="telegram", source="cli")
    assert loaded.model_source == "fleet"
    assert loaded.model == "openai:gpt-5-mini"
    assert loaded.local_mcp_config_path == str(tmp_path / "assistant" / ".mcp.json")
    assert loaded.tool_requirements == ()
    assert loaded.approval_tool_names == ()
    assert loaded.setup_tasks == ()


def test_manifest_refresh_records_root_and_subagent_tool_requirements(tmp_path: Path) -> None:
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    _write_tools(
        fleet_dir / "tools.json",
        [
            {
                "name": "search",
                "mcp_server_url": "https://builtin.example/catalog?token=raw-secret",
                "auth_type": "builtin",
            },
            {
                "name": "missing",
                "mcp_server_url": "https://missing.example/mcp?token=missing-secret#frag",
                "auth_type": "headers",
                "headers": {"Authorization": "Bearer header-secret"},
            },
        ],
    )
    subagent_dir = fleet_dir / "subagents" / "researcher"
    subagent_dir.mkdir(parents=True)
    _write_tools(
        subagent_dir / "tools.json",
        [
            {
                "name": "calendar",
                "mcp_server_url": "https://calendar.example/mcp?token=oauth-secret",
                "auth_type": "oauth",
            }
        ],
    )
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "AGENT_MODEL": "override:model",
            "DEEPAGENTS_TALON_FLEET_DIR": str(fleet_dir),
            "BUILTIN_MCP_URL": "https://builtin.example/mcp?api_key=builtin-secret",
        },
        base_home=tmp_path,
    )
    components = FleetAgentComponents(
        model="fleet:model",
        system_prompt="fleet prompt",
        tools=(SimpleNamespace(name="search"),),
        subagents=({"tools": [SimpleNamespace(name="calendar")]},),
        interrupt_on={"search": True, "send_email": True},
    )

    first = refresh_fleet_run_manifest(
        config,
        components,
        selected_channel=ChannelSelection(
            provider="multiple",
            source="mixed",
            metadata={"providers": "whatsapp,telegram"},
        ),
        now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    second = refresh_fleet_run_manifest(
        config,
        components,
        selected_channel=first.selected_channel,
        now=datetime(2026, 1, 3, 3, 4, 5, tzinfo=UTC),
    )

    assert second.created_at == first.created_at
    assert second.model_source == "environment"
    assert second.model == "override:model"
    assert (fleet_dir / "fleet-run-manifest.json").exists() is False
    assert manifest_path(config.home).is_file()

    requirements = {requirement.tool_name: requirement for requirement in second.tool_requirements}
    assert set(requirements) == {"calendar", "missing", "search"}
    assert requirements["search"].scope == "root"
    assert requirements["search"].mcp_server_url == "https://builtin.example/catalog"
    assert requirements["search"].auth_path == "builtin"
    assert requirements["search"].loaded is True
    assert requirements["missing"].mcp_server_url == "https://missing.example/mcp"
    assert requirements["missing"].loaded is False
    assert requirements["calendar"].scope == "subagent:researcher"
    assert requirements["calendar"].mcp_server_url == "https://calendar.example/mcp"
    assert requirements["calendar"].loaded is True
    assert second.approval_tool_names == ("search", "send_email")

    first_ids = [requirement.id for requirement in first.tool_requirements]
    second_ids = [requirement.id for requirement in second.tool_requirements]
    assert second_ids == first_ids
    assert {task.target_path for task in second.setup_tasks} == {
        str(tmp_path / "assistant" / ".mcp.json")
    }
    assert {task.tool_requirement_ids[0] for task in second.setup_tasks} == {
        requirements["calendar"].id,
        requirements["missing"].id,
    }
    assert "secret" not in manifest_path(config.home).read_text(encoding="utf-8")


def test_manifest_load_rejects_invalid_schema(tmp_path: Path) -> None:
    path = tmp_path / "fleet-run-manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assistant_id": "assistant",
                "fleet_dir": str(tmp_path / "fleet"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FleetRunManifestValidationError, match="model_source"):
        load_fleet_run_manifest(path)


def _write_tools(path: Path, tools: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"tools": tools}), encoding="utf-8")
