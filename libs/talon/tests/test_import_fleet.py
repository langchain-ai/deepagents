from __future__ import annotations

import json
from argparse import Namespace
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.__main__ import (
    _config_from_fleet_run_manifest,
    _resolve_import_channel,
    _run_fleet_command,
    _run_import_fleet_command,
)
from deepagents_talon.import_fleet import (
    FleetImportError,
    import_fleet_manifest,
    load_fleet_run_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path


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


def test_import_fleet_command_fails_without_channel(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fleet = _fleet_export(tmp_path)
    monkeypatch.setenv("DEEPAGENTS_TALON_TELEGRAM_ENABLED", "true")
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


def test_resolve_import_channel_requires_explicit_channel() -> None:
    with pytest.raises(FleetImportError, match="--channel is required"):
        _resolve_import_channel(None, non_interactive=False)


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


def test_load_fleet_run_manifest_fails_when_manifest_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FleetImportError, match="Fleet run manifest not found"):
        load_fleet_run_manifest(
            assistant_id="agent-1",
            env={"DEEPAGENTS_TALON_HOME": str(tmp_path / "home")},
        )


def test_load_fleet_run_manifest_fails_when_fleet_dir_is_stale(tmp_path: Path) -> None:
    home = tmp_path / "home"
    target = home / "agent-1" / "agent" / "tools.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "manifest": {
                    "source": "fleet",
                    "assistant_id": "agent-1",
                    "channel": "telegram",
                    "fleet_dir": str(tmp_path / "missing-fleet"),
                    "replacement_tools": [],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FleetImportError, match="Fleet directory does not exist"):
        load_fleet_run_manifest(
            assistant_id="agent-1",
            env={"DEEPAGENTS_TALON_HOME": str(home)},
        )


def test_run_fleet_command_starts_selected_telegram_manifest(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fleet = _fleet_export(tmp_path)
    home = tmp_path / "home"
    import_fleet_manifest(
        fleet,
        assistant_id="agent-1",
        channel="telegram",
        env={"DEEPAGENTS_TALON_HOME": str(home)},
    )
    monkeypatch.setenv("DEEPAGENTS_TALON_HOME", str(home))
    captured: dict[str, object] = {}

    def run_host(config: object, **kwargs: object) -> None:
        captured["config"] = config
        captured.update(kwargs)

    monkeypatch.setattr("deepagents_talon.__main__._run_host", run_host)

    caplog.set_level("INFO", logger="deepagents_talon.__main__")
    code = _run_fleet_command(_run_fleet_args("agent-1"))

    config = captured["config"]
    assert code == 0
    assert captured["selected_channel"] == "telegram"
    assert captured["whatsapp"] is False
    assert captured["telegram"] is False
    assert config.assistant_id == "agent-1"
    assert config.fleet_dir == fleet.resolve()
    event = _talon_events(caplog, event="fleet.run_startup")[0]
    assert event["assistant_id"] == "agent-1"
    assert event["channel"] == "telegram"
    assert event["fleet_dir"] == str(fleet.resolve())
    assert event["replacement_tool_count"] == 4
    assert "raw-secret" not in caplog.text


def test_run_fleet_command_starts_selected_whatsapp_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fleet = _fleet_export(tmp_path)
    home = tmp_path / "home"
    import_fleet_manifest(
        fleet,
        assistant_id="agent-1",
        channel="whatsapp",
        env={"DEEPAGENTS_TALON_HOME": str(home)},
    )
    monkeypatch.setenv("DEEPAGENTS_TALON_HOME", str(home))
    captured: dict[str, object] = {}

    def run_host(config: object, **kwargs: object) -> None:
        captured["config"] = config
        captured.update(kwargs)

    monkeypatch.setattr("deepagents_talon.__main__._run_host", run_host)

    code = _run_fleet_command(_run_fleet_args("agent-1", telegram=True))

    assert code == 0
    assert captured["selected_channel"] == "whatsapp"
    assert captured["telegram"] is True


def test_run_fleet_config_keeps_environment_model_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fleet = _fleet_export(tmp_path)
    home = tmp_path / "home"
    import_fleet_manifest(
        fleet,
        assistant_id="agent-1",
        channel="telegram",
        env={"DEEPAGENTS_TALON_HOME": str(home)},
    )
    manifest = load_fleet_run_manifest(
        assistant_id="agent-1",
        env={"DEEPAGENTS_TALON_HOME": str(home)},
    )
    monkeypatch.setenv("AGENT_MODEL", "override:model")

    config = _config_from_fleet_run_manifest(
        manifest,
        env={"DEEPAGENTS_TALON_HOME": str(home), "AGENT_MODEL": "override:model"},
    )

    assert config.model == "override:model"
    assert config.fleet_dir == fleet.resolve()


def test_run_fleet_command_passes_once_to_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fleet = _fleet_export(tmp_path)
    home = tmp_path / "home"
    import_fleet_manifest(
        fleet,
        assistant_id="agent-1",
        channel="telegram",
        env={"DEEPAGENTS_TALON_HOME": str(home)},
    )
    monkeypatch.setenv("DEEPAGENTS_TALON_HOME", str(home))
    captured: dict[str, object] = {}

    def run_host(config: object, **kwargs: object) -> None:
        captured["config"] = config
        captured.update(kwargs)

    monkeypatch.setattr("deepagents_talon.__main__._run_host", run_host)

    code = _run_fleet_command(_run_fleet_args("agent-1", once=True))

    assert code == 0
    assert captured["once"] is True


def _run_fleet_args(
    assistant_id: str,
    *,
    once: bool = False,
    whatsapp: bool = False,
    telegram: bool = False,
) -> Namespace:
    return Namespace(
        assistant_id=assistant_id,
        once=once,
        whatsapp=whatsapp,
        telegram=telegram,
    )


def _talon_events(
    caplog: pytest.LogCaptureFixture,
    *,
    event: str,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for message in caplog.messages:
        if not message.startswith("talon_event "):
            continue
        payload = json.loads(message.removeprefix("talon_event "))
        if payload.get("event") == event:
            events.append(payload)
    return events


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
