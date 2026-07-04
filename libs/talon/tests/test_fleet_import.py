from __future__ import annotations

import json
import stat
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
    assert (target / "skills" / "review" / "SKILL.md").read_text(encoding="utf-8") == (
        "---\nname: review\n---\nReview things."
    )
    assert (target / "subagents" / "researcher" / "AGENTS.md").is_file()
    assert not (target / "tools.json").exists()
    assert not (target / "config.json").exists()
    notes = (target / ".mcp.json.setup").read_text(encoding="utf-8")
    assert "Server: sample" in notes
    assert "Tool count: 2" in notes
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


def test_import_fleet_zip_rejects_missing_root_prompt(tmp_path: Path) -> None:
    source = tmp_path / "fleet.zip"
    _write_zip(source, {"subagents/researcher/AGENTS.md": "prompt"})

    with pytest.raises(FleetImportError, match=r"AGENTS\.md: missing required root prompt"):
        import_fleet_zip(source, target_dir=tmp_path / "agent")


def test_import_fleet_zip_rejects_malformed_tools_json(tmp_path: Path) -> None:
    source = tmp_path / "fleet.zip"
    _write_zip(source, {"AGENTS.md": "root prompt", "tools.json": "{bad"})

    with pytest.raises(
        FleetImportError,
        match=rf"{source}: tools\.json: malformed tools\.json: Expecting property name",
    ):
        import_fleet_zip(source, target_dir=tmp_path / "agent")


@pytest.mark.parametrize("path", ["../escape", "/escape", "C:/escape"])
def test_import_fleet_zip_rejects_unsafe_paths(tmp_path: Path, path: str) -> None:
    source = tmp_path / "fleet.zip"
    _write_zip(source, {"AGENTS.md": "root prompt", path: "bad"})

    with pytest.raises(FleetImportError, match="unsafe zip path"):
        import_fleet_zip(source, target_dir=tmp_path / "agent")


def test_import_fleet_zip_rejects_symlink_entries_before_writing_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fleet.zip"
    target = tmp_path / "agent"
    target.mkdir()
    (target / "AGENTS.md").write_text("existing prompt", encoding="utf-8")
    symlink = zipfile.ZipInfo("skills/review/SKILL.md")
    symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("AGENTS.md", "new prompt")
        archive.writestr(symlink, "../secret")

    with pytest.raises(FleetImportError, match=r"skills/review/SKILL\.md: symlink"):
        import_fleet_zip(source, target_dir=target)

    assert (target / "AGENTS.md").read_text(encoding="utf-8") == "existing prompt"
    assert not (target / "skills").exists()


def test_import_fleet_zip_rejects_unsafe_subagent_names(tmp_path: Path) -> None:
    source = tmp_path / "fleet.zip"
    _write_zip(source, {"AGENTS.md": "root prompt", "subagents/bad name/AGENTS.md": "bad"})

    with pytest.raises(FleetImportError, match="unsafe subagent name"):
        import_fleet_zip(source, target_dir=tmp_path / "agent")


def test_import_fleet_zip_stdout_recommends_interrupt_env(tmp_path: Path) -> None:
    source = tmp_path / "fleet.zip"
    tools = _tools_json("root")
    tools["tools"].append(
        {
            "name": "approve_remote",
            "mcp_server_url": "https://tools.example/mcp",
            "mcp_server_name": "sample",
            "interrupt_config": True,
        }
    )
    _write_zip(source, {"AGENTS.md": "root prompt", "tools.json": json.dumps(tools)})

    result = import_fleet_zip(source, target_dir=tmp_path / "agent")

    output = format_import_stdout(result)
    assert result.interrupt_tools == ("approve_remote", "write_remote")
    assert f"{INTERRUPT_ON_TOOLS_ENV_KEY}=approve_remote,write_remote" in output


def test_import_fleet_zip_removes_existing_mcp_notes_when_no_tools(tmp_path: Path) -> None:
    source = tmp_path / "fleet.zip"
    target = tmp_path / "agent"
    target.mkdir()
    (target / ".mcp.json.setup").write_text("stale notes", encoding="utf-8")
    _write_zip(source, {"AGENTS.md": "root prompt", "tools.json": json.dumps({"tools": []})})

    result = import_fleet_zip(source, target_dir=target)

    assert result.mcp_notes is None
    assert result.interrupt_tools == ()
    assert not (target / ".mcp.json.setup").exists()
    assert "MCP setup: no Fleet MCP tool requirements found" in format_import_stdout(result)


def test_import_fleet_zip_sanitizes_secret_bearing_mcp_urls(tmp_path: Path) -> None:
    source = tmp_path / "fleet.zip"
    _write_zip(
        source,
        {
            "AGENTS.md": "root prompt",
            "tools.json": json.dumps(
                {
                    "tools": [
                        {
                            "name": "secure_lookup",
                            "mcp_server_url": (
                                "https://operator:password@tools.example/"
                                "tenant/bearer-token/mcp?api_key=secret#oauth"
                            ),
                            "mcp_server_name": "secret server",
                        },
                    ],
                    "interrupt_config": {},
                }
            ),
        },
    )

    result = import_fleet_zip(source, target_dir=tmp_path / "agent")

    assert result.mcp_notes is not None
    assert "operator" not in result.mcp_notes
    assert "password" not in result.mcp_notes
    assert "api_key" not in result.mcp_notes
    assert "secret#oauth" not in result.mcp_notes
    assert "https://tools.example/tenant/<secret-redacted>/mcp" in result.mcp_notes
    assert '"auth": "oauth"' in result.mcp_notes


def test_import_fleet_zip_repeated_imports_refresh_generated_files(tmp_path: Path) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    target = tmp_path / "agent"
    _write_zip(
        first,
        {
            "AGENTS.md": "first root",
            "tools.json": json.dumps(_tools_json("root")),
            "skills/review/SKILL.md": "first skill",
            "subagents/researcher/AGENTS.md": "first subagent",
        },
    )
    _write_zip(
        second,
        {
            "AGENTS.md": "second root",
            "skills/write/SKILL.md": "second skill",
            "subagents/writer/AGENTS.md": "second subagent",
        },
    )

    import_fleet_zip(first, target_dir=target)
    import_fleet_zip(second, target_dir=target)

    assert (target / "AGENTS.md").read_text(encoding="utf-8") == "second root"
    assert not (target / ".mcp.json.setup").exists()
    assert not (target / "skills" / "review").exists()
    assert (target / "skills" / "write" / "SKILL.md").read_text(encoding="utf-8") == (
        "second skill"
    )
    assert not (target / "subagents" / "researcher").exists()
    assert (target / "subagents" / "writer" / "AGENTS.md").read_text(
        encoding="utf-8"
    ) == "second subagent"


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
