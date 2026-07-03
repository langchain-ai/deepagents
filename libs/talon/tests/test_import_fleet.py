from __future__ import annotations

import json
from argparse import Namespace
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.__main__ import (
    _resolve_import_channel,
    _run_import_fleet_command,
)
from deepagents_talon.import_fleet import FleetImportError, import_fleet_manifest

if TYPE_CHECKING:
    from pathlib import Path


class InteractiveStdin:
    def isatty(self) -> bool:
        return True


def test_import_fleet_manifest_writes_and_refreshes_assistant_manifest(tmp_path: Path) -> None:
    fleet = _fleet_export(tmp_path)
    home = tmp_path / "home"
    target = home / "agent-1" / "agent" / "tools.json"
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
                "manifest": {"stale": True},
            }
        ),
        encoding="utf-8",
    )

    summary = import_fleet_manifest(
        fleet,
        assistant_id="agent-1",
        channel="telegram",
        env={"DEEPAGENTS_TALON_HOME": str(home), "AGENT_MODEL": "override:model"},
    )
    import_fleet_manifest(
        fleet,
        assistant_id="agent-1",
        channel="telegram",
        env={"DEEPAGENTS_TALON_HOME": str(home), "AGENT_MODEL": "override:model"},
    )

    manifest = json.loads(target.read_text(encoding="utf-8"))
    payload = manifest["manifest"]
    assert summary.manifest_path == target
    assert summary.model_source == "local_override"
    assert summary.replacement_tool_count == 4
    assert summary.setup_task_count == 3
    assert manifest["mcpServers"] == {
        "local": {
            "type": "stdio",
            "command": "local-server",
        },
    }
    assert payload["assistant_id"] == "agent-1"
    assert payload["channel"] == "telegram"
    assert payload["fleet_dir"] == str(fleet.resolve())
    assert payload["model"] == "override:model"
    assert payload["model_source"] == "local_override"
    assert len(payload["replacement_tools"]) == 4
    assert len(payload["setup_tasks"]) == 3
    assert target.stat().st_mode & 0o777 == 0o600
    assert "raw-secret" not in target.read_text(encoding="utf-8")


def test_import_fleet_manifest_records_builtin_mcp_as_resolved(tmp_path: Path) -> None:
    fleet = _fleet_export(tmp_path)
    summary = import_fleet_manifest(
        fleet,
        assistant_id="agent-1",
        channel="telegram",
        env={
            "DEEPAGENTS_TALON_HOME": str(tmp_path / "home"),
            "BUILTIN_MCP_URL": "https://builtin.example/mcp?api_key=local-secret",
        },
    )

    assert summary.model_source == "fleet_config"
    assert summary.replacement_tool_count == 3
    assert summary.setup_task_count == 2


def test_import_fleet_manifest_reads_nested_fleet_model_config(tmp_path: Path) -> None:
    fleet = _fleet_export(tmp_path)
    (fleet / "config.json").write_text(
        json.dumps(
            {
                "config": {
                    "configurable": {
                        "llm_model_config": {
                            "modelId": "openai:gpt-5.5",
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    summary = import_fleet_manifest(
        fleet,
        assistant_id="agent-1",
        channel="whatsapp",
        env={"DEEPAGENTS_TALON_HOME": str(tmp_path / "home")},
    )

    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert summary.model_source == "fleet_config"
    assert manifest["manifest"]["model"] == "openai:gpt-5.5"


def test_import_fleet_manifest_fails_on_malformed_required_config(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "AGENTS.md").write_text("system prompt", encoding="utf-8")
    (fleet / "config.json").write_text("[]", encoding="utf-8")

    with pytest.raises(FleetImportError, match=r"Fleet config\.json"):
        import_fleet_manifest(
            fleet,
            assistant_id="agent-1",
            channel="telegram",
            env={"DEEPAGENTS_TALON_HOME": str(tmp_path / "home")},
        )


def test_import_fleet_command_fails_without_channel_in_non_interactive_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fleet = _fleet_export(tmp_path)
    monkeypatch.delenv("DEEPAGENTS_TALON_TELEGRAM_ENABLED", raising=False)
    monkeypatch.delenv("DEEPAGENTS_TALON_WHATSAPP_ENABLED", raising=False)

    code = _run_import_fleet_command(
        Namespace(
            fleet_dir=fleet,
            assistant_id="agent-1",
            channel=None,
            non_interactive=True,
        )
    )

    assert code == 2
    assert "--channel is required" in capsys.readouterr().err


def test_resolve_import_channel_prompts_when_interactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPAGENTS_TALON_TELEGRAM_ENABLED", raising=False)
    monkeypatch.delenv("DEEPAGENTS_TALON_WHATSAPP_ENABLED", raising=False)
    monkeypatch.setattr("sys.stdin", InteractiveStdin())
    answers = iter(["bad", "whatsapp"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert _resolve_import_channel(None, non_interactive=False) == "whatsapp"


def test_import_fleet_command_prints_secret_free_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fleet = _fleet_export(tmp_path)
    monkeypatch.setenv("DEEPAGENTS_TALON_HOME", str(tmp_path / "home"))

    code = _run_import_fleet_command(
        Namespace(
            fleet_dir=fleet,
            assistant_id="agent-1",
            channel="telegram",
            non_interactive=True,
        )
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "channel: telegram" in output
    assert "assistant_id: agent-1" in output
    assert "replacement_tools: 4" in output
    assert "setup_tasks: 3" in output
    assert "local_mcp_config:" in output
    assert "raw-secret" not in output
    assert "header-secret" not in output


def _fleet_export(tmp_path: Path) -> Path:
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "AGENTS.md").write_text("system prompt", encoding="utf-8")
    (fleet / "config.json").write_text(
        json.dumps({"model": "fleet:model"}),
        encoding="utf-8",
    )
    (fleet / "tools.json").write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "search",
                        "mcp_server_url": "https://missing.example/mcp?token=raw-secret",
                        "auth_type": "headers",
                        "headers": {"Authorization": "Bearer header-secret"},
                    },
                    {
                        "name": "lookup",
                        "mcp_server_url": "https://missing.example/mcp?token=raw-secret",
                        "auth_type": "headers",
                    },
                    {
                        "name": "builtin_search",
                        "mcp_server_url": "https://builtin.example/mcp?token=builtin-secret",
                        "auth_type": "builtin",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    subagent = fleet / "subagents" / "researcher"
    subagent.mkdir(parents=True)
    (subagent / "tools.json").write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "calendar",
                        "mcp_server_url": "https://calendar.example/mcp?token=raw-secret",
                        "auth_type": "oauth",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return fleet
