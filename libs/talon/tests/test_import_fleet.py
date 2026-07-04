from __future__ import annotations

import json
import zipfile
from argparse import Namespace
from pathlib import Path
from typing import cast

import pytest

from deepagents_talon.__main__ import (
    _run_import_fleet_command,
)
from deepagents_talon.import_fleet import (
    FleetImportError,
    import_fleet_export,
)


def test_import_fleet_export_materializes_zip_and_writes_mcp_notes(tmp_path: Path) -> None:
    fleet = _fleet_export(tmp_path)
    export = _fleet_zip(tmp_path, fleet)
    home = tmp_path / "home"
    target = home / "agent-1" / ".mcp.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "local": {
                        "type": "stdio",
                        "command": "local-server",
                    },
                },
                "_fleet_import": {"stale": True},
            }
        ),
        encoding="utf-8",
    )

    summary = import_fleet_export(
        export,
        assistant_id="agent-1",
        env={"DEEPAGENTS_TALON_HOME": str(home), "AGENT_MODEL": "unused:model"},
    )
    import_fleet_export(
        export,
        assistant_id="agent-1",
        env={"DEEPAGENTS_TALON_HOME": str(home), "AGENT_MODEL": "unused:model"},
    )

    payload = json.loads(target.read_text(encoding="utf-8"))
    notes = cast("dict[str, object]", payload["_fleet_import"])
    servers = cast("list[dict[str, object]]", notes["servers"])
    agent_dir = home / "agent-1" / "agent"
    assert summary.fleet_source == export.resolve()
    assert summary.agent_dir == agent_dir
    assert summary.mcp_config_path == target
    assert summary.tool_count == 4
    assert summary.server_count == 3
    assert summary.interrupt_tool_count == 2
    assert payload["mcpServers"] == {
        "local": {
            "type": "stdio",
            "command": "local-server",
        },
    }
    assert notes["assistant_id"] == "agent-1"
    assert notes["fleet_export"] == str(export.resolve())
    assert notes["agent_dir"] == str(agent_dir)
    assert servers == [
        {
            "auth_path": "builtin",
            "endpoint": "https://builtin.example/mcp",
            "interrupt_tools": [],
            "scope": "root",
            "server_name": "builtin",
            "tool_names": ["builtin_search"],
        },
        {
            "auth_path": "headers",
            "endpoint": "https://missing.example/mcp",
            "interrupt_tools": ["search"],
            "scope": "root",
            "server_name": "missing",
            "tool_names": ["lookup", "search"],
        },
        {
            "auth_path": "oauth",
            "endpoint": "https://calendar.example/mcp",
            "interrupt_tools": ["calendar"],
            "scope": "subagent:researcher",
            "server_name": "calendar",
            "tool_names": ["calendar"],
        },
    ]
    assert (agent_dir / "AGENTS.md").read_text(encoding="utf-8") == "system prompt"
    assert (agent_dir / "skills" / "triage" / "SKILL.md").read_text(encoding="utf-8") == (
        "skill prompt"
    )
    assert not (agent_dir / "config.json").exists()
    assert not (agent_dir / "tools.json").exists()
    assert (agent_dir / "subagents" / "researcher" / "AGENTS.md").read_text(
        encoding="utf-8"
    ) == "research prompt"
    assert not (agent_dir / "subagents" / "researcher" / "config.json").exists()
    assert not (agent_dir / "subagents" / "researcher" / "tools.json").exists()
    assert target.stat().st_mode & 0o777 == 0o600
    assert "raw-secret" not in target.read_text(encoding="utf-8")
    assert "header-secret" not in target.read_text(encoding="utf-8")


def test_import_fleet_export_accepts_unzipped_export_directory(tmp_path: Path) -> None:
    fleet = _fleet_export(tmp_path)
    summary = import_fleet_export(
        fleet,
        assistant_id="agent-1",
        env={"DEEPAGENTS_TALON_HOME": str(tmp_path / "home")},
    )

    assert summary.fleet_source == fleet.resolve()
    assert summary.tool_count == 4
    assert (summary.agent_dir / "AGENTS.md").read_text(encoding="utf-8") == "system prompt"


def test_import_fleet_export_accepts_single_root_directory_zip(tmp_path: Path) -> None:
    fleet = _fleet_export(tmp_path)
    export = tmp_path / "nested.zip"
    with zipfile.ZipFile(export, "w") as archive:
        for path in sorted(fleet.rglob("*")):
            archive.write(path, Path("fleet-export") / path.relative_to(fleet))

    summary = import_fleet_export(
        export,
        assistant_id="agent-1",
        env={"DEEPAGENTS_TALON_HOME": str(tmp_path / "home")},
    )

    assert summary.tool_count == 4
    assert (summary.agent_dir / "subagents" / "researcher" / "AGENTS.md").is_file()


def test_import_fleet_export_fails_on_missing_prompt(tmp_path: Path) -> None:
    export = tmp_path / "fleet.zip"
    with zipfile.ZipFile(export, "w") as archive:
        archive.writestr("tools.json", json.dumps({"tools": []}))

    with pytest.raises(FleetImportError, match=r"AGENTS\.md"):
        import_fleet_export(
            export,
            assistant_id="agent-1",
            env={"DEEPAGENTS_TALON_HOME": str(tmp_path / "home")},
        )


def test_import_fleet_export_fails_on_unsafe_zip_path(tmp_path: Path) -> None:
    export = tmp_path / "fleet.zip"
    with zipfile.ZipFile(export, "w") as archive:
        archive.writestr("../AGENTS.md", "system prompt")

    with pytest.raises(FleetImportError, match="unsafe path"):
        import_fleet_export(
            export,
            assistant_id="agent-1",
            env={"DEEPAGENTS_TALON_HOME": str(tmp_path / "home")},
        )


def test_import_fleet_export_fails_on_malformed_tools_json(tmp_path: Path) -> None:
    fleet = _fleet_export(tmp_path)
    (fleet / "tools.json").write_text(json.dumps({"tools": [{}]}), encoding="utf-8")
    export = _fleet_zip(tmp_path, fleet)

    with pytest.raises(FleetImportError, match=r"tools\[0\]\.name"):
        import_fleet_export(
            export,
            assistant_id="agent-1",
            env={"DEEPAGENTS_TALON_HOME": str(tmp_path / "home")},
        )


def test_import_fleet_command_prints_secret_free_mcp_notes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fleet = _fleet_export(tmp_path)
    export = _fleet_zip(tmp_path, fleet)
    monkeypatch.setenv("DEEPAGENTS_TALON_HOME", str(tmp_path / "home"))

    code = _run_import_fleet_command(
        Namespace(
            fleet_dir=export,
            assistant_id="agent-1",
            channel="telegram",
            non_interactive=True,
        )
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "assistant_id: agent-1" in output
    assert "tools_summarized: 4" in output
    assert "mcp_servers: 3" in output
    assert "interrupt_tools: 2" in output
    assert "root_mcp_config:" in output
    assert "root: missing (https://missing.example/mcp)" in output
    assert "interrupt_on: search" in output
    assert "subagent:researcher: calendar (https://calendar.example/mcp)" in output
    assert "interrupt_on: calendar" in output
    assert "Recommended human-in-the-loop tools: calendar, search" in output
    assert "raw-secret" not in output
    assert "header-secret" not in output


def _fleet_export(tmp_path: Path) -> Path:
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "AGENTS.md").write_text("system prompt", encoding="utf-8")
    skill = fleet / "skills" / "triage"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("skill prompt", encoding="utf-8")
    (fleet / "tools.json").write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "search",
                        "mcp_server_url": "https://missing.example/mcp?token=raw-secret",
                        "mcp_server_name": "missing",
                        "auth_type": "headers",
                        "headers": {"Authorization": "Bearer header-secret"},
                    },
                    {
                        "name": "lookup",
                        "mcp_server_url": "https://missing.example/mcp?token=raw-secret",
                        "mcp_server_name": "missing",
                        "auth_type": "headers",
                    },
                    {
                        "name": "builtin_search",
                        "mcp_server_url": "https://builtin.example/mcp?token=builtin-secret",
                        "mcp_server_name": "builtin",
                        "auth_type": "builtin",
                    },
                ],
                "interrupt_config": {
                    "https://missing.example/mcp::search::missing": True,
                    "https://missing.example/mcp::lookup::missing": False,
                    "https://builtin.example/mcp::builtin_search::builtin": False,
                },
            }
        ),
        encoding="utf-8",
    )
    subagent = fleet / "subagents" / "researcher"
    subagent.mkdir(parents=True)
    (subagent / "AGENTS.md").write_text("research prompt", encoding="utf-8")
    (subagent / "config.json").write_text(json.dumps({"description": "unused"}), encoding="utf-8")
    (subagent / "tools.json").write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "calendar",
                        "mcp_server_url": "https://calendar.example/mcp?token=raw-secret",
                        "mcp_server_name": "calendar",
                        "auth_type": "oauth",
                        "interrupt_on": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return fleet


def _fleet_zip(tmp_path: Path, fleet: Path) -> Path:
    export = tmp_path / "fleet.zip"
    with zipfile.ZipFile(export, "w") as archive:
        for path in sorted(fleet.rglob("*")):
            archive.write(path, path.relative_to(fleet))
    return export
