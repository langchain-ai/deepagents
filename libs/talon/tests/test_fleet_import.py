from __future__ import annotations

import json
import sys
import zipfile
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.__main__ import main
from deepagents_talon.fleet_import import FleetImportError, format_import_stdout, import_fleet_zip
from deepagents_talon.runtime import INTERRUPT_ON_TOOLS_ENV_KEY

if TYPE_CHECKING:
    from pathlib import Path


def test_import_fleet_zip_materializes_agent_files_and_mcp_notes(tmp_path: Path) -> None:
    source = tmp_path / "fleet.zip"
    _write_zip(
        source,
        {
            "AGENTS.md": "root prompt",
            "config.json": "{}",
            "tools.json": json.dumps(_tools_json("root")),
            "skills/review/SKILL.md": "---\nname: review\n---\nReview things.",
            "subagents/researcher/AGENTS.md": (
                "---\ndescription: Research tasks\n---\nResearch carefully."
            ),
            "subagents/researcher/tools.json": json.dumps(_tools_json("researcher")),
        },
    )
    target = tmp_path / "agent"

    result = import_fleet_zip(source, target_dir=target)

    assert result.target_dir == target
    assert result.root_prompt_count == 1
    assert result.subagent_prompt_count == 1
    assert result.config_ignored is True
    assert result.interrupt_tools == ("write_remote",)
    assert (target / "AGENTS.md").read_text(encoding="utf-8") == "root prompt"
    assert (target / "skills" / "review" / "SKILL.md").is_file()
    assert (target / "subagents" / "researcher" / "AGENTS.md").is_file()
    assert not (target / "tools.json").exists()
    assert not (target / "config.json").exists()
    notes = (target / ".mcp.json.setup").read_text(encoding="utf-8")
    assert "Server: sample" in notes
    assert "Scopes: researcher, root" in notes
    assert "Interrupt-enabled tools: write_remote" in notes
    assert '"allowedTools": [' in notes


def test_import_fleet_cli_defaults_target_to_selected_assistant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "fleet.zip"
    _write_zip(source, {"AGENTS.md": "root prompt"})
    monkeypatch.setenv("DEEPAGENTS_TALON_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DEEPAGENTS_TALON_ASSISTANT_ID", "default")
    monkeypatch.setattr(sys, "argv", ["deepagents-talon", "import-fleet", str(source)])

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    target = tmp_path / "home" / "default" / "agent"
    assert (target / "AGENTS.md").read_text(encoding="utf-8") == "root prompt"
    assert f"Target directory: {target}" in capsys.readouterr().out


def test_import_fleet_cli_assistant_id_overrides_default_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "fleet.zip"
    _write_zip(source, {"AGENTS.md": "root prompt"})
    monkeypatch.setenv("DEEPAGENTS_TALON_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DEEPAGENTS_TALON_ASSISTANT_ID", "default")
    monkeypatch.setattr(
        sys,
        "argv",
        ["deepagents-talon", "import-fleet", str(source), "--assistant-id", "chosen"],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    assert (tmp_path / "home" / "chosen" / "agent" / "AGENTS.md").is_file()
    assert not (tmp_path / "home" / "default" / "agent" / "AGENTS.md").exists()


def test_import_fleet_help_documents_operator_workflow(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["deepagents-talon", "import-fleet", "--help"])

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "deepagents-talon import-fleet <fleet-export.zip>" in output
    assert "selected assistant manifest directory" in output
    assert ".mcp.json.setup is a human-readable setup handoff" in output
    assert ".mcp.json remains the runtime MCP config file" in output
    assert "Fleet config.json is ignored" in output
    assert "Fleet tools.json is import input only" in output
    assert "old Fleet direct-run environment variables are unsupported" in output


def test_import_fleet_zip_rejects_missing_root_prompt(tmp_path: Path) -> None:
    source = tmp_path / "fleet.zip"
    _write_zip(source, {"subagents/researcher/AGENTS.md": "prompt"})

    with pytest.raises(FleetImportError, match=r"AGENTS\.md: missing required root prompt"):
        import_fleet_zip(source, target_dir=tmp_path / "agent")


def test_import_fleet_zip_rejects_malformed_tools_json(tmp_path: Path) -> None:
    source = tmp_path / "fleet.zip"
    _write_zip(source, {"AGENTS.md": "root prompt", "tools.json": "{bad"})

    with pytest.raises(FleetImportError, match=r"tools\.json: malformed tools\.json"):
        import_fleet_zip(source, target_dir=tmp_path / "agent")


def test_import_fleet_zip_rejects_unsafe_paths(tmp_path: Path) -> None:
    source = tmp_path / "fleet.zip"
    _write_zip(source, {"AGENTS.md": "root prompt", "../escape": "bad"})

    with pytest.raises(FleetImportError, match=r"\.\./escape: unsafe zip path"):
        import_fleet_zip(source, target_dir=tmp_path / "agent")


def test_import_fleet_zip_rejects_unsafe_subagent_names(tmp_path: Path) -> None:
    source = tmp_path / "fleet.zip"
    _write_zip(source, {"AGENTS.md": "root prompt", "subagents/bad name/AGENTS.md": "bad"})

    with pytest.raises(FleetImportError, match="unsafe subagent name"):
        import_fleet_zip(source, target_dir=tmp_path / "agent")


def test_import_fleet_zip_stdout_recommends_interrupt_env(tmp_path: Path) -> None:
    source = tmp_path / "fleet.zip"
    _write_zip(source, {"AGENTS.md": "root prompt", "tools.json": json.dumps(_tools_json("root"))})

    result = import_fleet_zip(source, target_dir=tmp_path / "agent")

    output = format_import_stdout(result)
    assert f"{INTERRUPT_ON_TOOLS_ENV_KEY}=write_remote" in output


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def _tools_json(scope: str) -> dict[str, object]:
    return {
        "tools": [
            {
                "name": "read_remote",
                "mcp_server_url": "https://tools.example/mcp?token=secret",
                "mcp_server_name": "sample",
                "display_name": f"read_remote_{scope}",
            },
            {
                "name": "write_remote",
                "mcp_server_url": "https://tools.example/mcp?token=secret",
                "mcp_server_name": "sample",
                "display_name": f"write_remote_{scope}",
            },
        ],
        "interrupt_config": {
            "https://tools.example/mcp?token=secret::write_remote::sample": True,
        },
    }
