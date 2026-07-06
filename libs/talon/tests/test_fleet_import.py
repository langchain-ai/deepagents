from __future__ import annotations

import json
import stat
import sys
import zipfile
from typing import TYPE_CHECKING, cast

import pytest

from deepagents_code.mcp_tools import load_mcp_config
from deepagents_talon.__main__ import main
from deepagents_talon.fleet_import import FleetImportError, format_import_stdout, import_fleet_zip
from deepagents_talon.runtime import INTERRUPT_ON_TOOLS_ENV_KEY

if TYPE_CHECKING:
    from pathlib import Path


def test_import_fleet_zip_materializes_agent_files_and_mcp_config(tmp_path: Path) -> None:
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
    target = tmp_path / "agent-home" / "agent"

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
    assert (target / "agents" / "researcher" / "AGENTS.md").is_file()
    assert not (target / "subagents").exists()
    assert not (target / "tools.json").exists()
    assert not (target / "config.json").exists()
    notes = (target / ".mcp.json.setup").read_text(encoding="utf-8")
    assert "Server: sample" in notes
    assert "Tool count: 2" in notes
    assert "Scopes: researcher, root" in notes
    assert "Interrupt-enabled tools: write_remote" in notes
    assert '"allowedTools": [' in notes
    config = json.loads((target / ".mcp.json").read_text(encoding="utf-8"))
    assert config == {
        "mcpServers": {
            "sample": {
                "allowedTools": ["read_remote", "write_remote"],
                "auth": "oauth",
                "type": "http",
                "url": "https://tools.example/mcp",
            },
        },
    }


def test_import_fleet_zip_normalizes_mcp_server_names_for_loader(tmp_path: Path) -> None:
    source = tmp_path / "fleet.zip"
    _write_zip(
        source,
        {
            "AGENTS.md": "root prompt",
            "tools.json": json.dumps(
                {
                    "tools": [
                        {
                            "name": "read_remote",
                            "mcp_server_url": "https://tools.example/mcp",
                            "mcp_server_name": "foo.bar",
                        },
                    ],
                    "interrupt_config": {},
                }
            ),
        },
    )
    target = tmp_path / "agent"

    import_fleet_zip(source, target_dir=target)

    config_path = target / ".mcp.json"
    config = load_mcp_config(str(config_path))
    assert set(config["mcpServers"]) == {"foo-bar"}
    notes = (target / ".mcp.json.setup").read_text(encoding="utf-8")
    assert '"foo-bar": {' in notes


@pytest.mark.parametrize(
    ("extra_args", "expected_id", "unexpected_id"),
    [
        ((), "default", None),
        (("--assistant-id", "chosen"), "chosen", "default"),
    ],
)
def test_import_fleet_cli_resolves_target_assistant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_args: tuple[str, ...],
    expected_id: str,
    unexpected_id: str | None,
) -> None:
    source = tmp_path / "fleet.zip"
    _write_zip(
        source,
        {
            "AGENTS.md": "root prompt",
            "subagents/researcher/AGENTS.md": "Research carefully.",
        },
    )
    monkeypatch.setenv("DEEPAGENTS_TALON_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DEEPAGENTS_TALON_ASSISTANT_ID", "default")
    monkeypatch.setattr(
        sys,
        "argv",
        ["deepagents-talon", "import-fleet", str(source), *extra_args],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    target = tmp_path / "home" / expected_id / "agent"
    assert (target / "AGENTS.md").read_text(encoding="utf-8") == "root prompt"
    assert (tmp_path / "home" / expected_id / "agents" / "researcher" / "AGENTS.md").is_file()
    if unexpected_id is not None:
        assert not (tmp_path / "home" / unexpected_id / "agent" / "AGENTS.md").exists()


def test_import_fleet_cli_explicit_target_keeps_subagents_under_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "fleet.zip"
    _write_zip(
        source,
        {
            "AGENTS.md": "root prompt",
            "subagents/researcher/AGENTS.md": "Research carefully.",
        },
    )
    sibling_agents = tmp_path / "agents"
    sibling_agents.mkdir()
    (sibling_agents / "keep.txt").write_text("keep", encoding="utf-8")
    target = tmp_path / "imported-agent"
    monkeypatch.setenv("DEEPAGENTS_TALON_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["deepagents-talon", "import-fleet", str(source), "--target-dir", str(target)],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    assert (target / "AGENTS.md").read_text(encoding="utf-8") == "root prompt"
    assert (target / "agents" / "researcher" / "AGENTS.md").is_file()
    assert (sibling_agents / "keep.txt").read_text(encoding="utf-8") == "keep"


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
    target = tmp_path / "agent-home" / "agent"
    target.mkdir(parents=True)
    (target / "AGENTS.md").write_text("existing prompt", encoding="utf-8")
    (target / "subagents" / "stale" / "AGENTS.md").parent.mkdir(parents=True)
    (target / "subagents" / "stale" / "AGENTS.md").write_text("stale", encoding="utf-8")
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


def test_import_fleet_zip_rejects_high_compression_ratio(tmp_path: Path) -> None:
    source = tmp_path / "fleet.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("AGENTS.md", "root prompt")
        archive.writestr("skills/bomb/SKILL.md", "0" * 1_000_000)

    with pytest.raises(FleetImportError, match="compression ratio exceeds limit"):
        import_fleet_zip(source, target_dir=tmp_path / "agent")


def test_import_fleet_zip_rejects_too_many_entries(tmp_path: Path) -> None:
    source = tmp_path / "fleet.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("AGENTS.md", "root prompt")
        for index in range(10_000):
            archive.writestr(f"skills/skill-{index}/SKILL.md", "skill")

    with pytest.raises(FleetImportError, match="too many zip entries"):
        import_fleet_zip(source, target_dir=tmp_path / "agent")


def test_import_fleet_zip_stdout_recommends_interrupt_env(tmp_path: Path) -> None:
    source = tmp_path / "fleet.zip"
    tools = _tools_json("root")
    raw_tools = cast("list[dict[str, object]]", tools["tools"])
    raw_tools.append(
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
    assert (
        f"- Add HITL for sensitive tools with "
        f"{INTERRUPT_ON_TOOLS_ENV_KEY}=approve_remote,write_remote."
    ) in output


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
                        {
                            "name": "token_lookup",
                            "mcp_server_url": "https://tools.example/token/abcd1234/mcp",
                            "mcp_server_name": "token server",
                        },
                        {
                            "name": "key_lookup",
                            "mcp_server_url": "https://tools.example/api_key/live-secret/mcp",
                            "mcp_server_name": "key server",
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
    assert "abcd1234" not in result.mcp_notes
    assert "live-secret" not in result.mcp_notes
    assert "https://tools.example/tenant/<secret-redacted>/mcp" in result.mcp_notes
    assert "https://tools.example/<secret-redacted>/<secret-redacted>/mcp" in result.mcp_notes
    config_text = (tmp_path / "agent" / ".mcp.json").read_text(encoding="utf-8")
    assert "operator" not in config_text
    assert "password" not in config_text
    assert "api_key" not in config_text
    assert "secret#oauth" not in config_text
    assert "abcd1234" not in config_text
    assert "live-secret" not in config_text
    config = json.loads(config_text)
    server = config["mcpServers"]["secret-server"]
    assert server["url"] == "https://tools.example/tenant/<secret-redacted>/mcp"
    assert server["allowedTools"] == ["secure_lookup"]


def test_import_fleet_zip_repeated_imports_refresh_generated_files(tmp_path: Path) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    target = tmp_path / "agent-home" / "agent"
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
    assert not (target / ".mcp.json").exists()
    assert not (target / "skills" / "review").exists()
    assert (target / "skills" / "write" / "SKILL.md").read_text(encoding="utf-8") == (
        "second skill"
    )
    assert not (target / "subagents").exists()
    assert not (target / "agents" / "researcher").exists()
    assert (target / "agents" / "writer" / "AGENTS.md").read_text(
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
