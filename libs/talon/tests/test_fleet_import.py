from __future__ import annotations

import stat
import sys
import zipfile
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.__main__ import main
from deepagents_talon.fleet_import import FleetImportError, import_fleet_zip

if TYPE_CHECKING:
    from pathlib import Path


def test_import_fleet_zip_materializes_agent_files_and_ignores_config(tmp_path: Path) -> None:
    source = tmp_path / "fleet.zip"
    _write_zip(
        source,
        {
            "AGENTS.md": "root prompt",
            "config.json": "{}",
            "tools.json": "{ignored",
            "skills/review/SKILL.md": "---\nname: review\n---\nReview things.",
            "subagents/researcher/AGENTS.md": (
                "---\ndescription: Research tasks\n---\nResearch carefully."
            ),
            "subagents/researcher/tools.json": "{ignored",
        },
    )
    target = tmp_path / "agent-home" / "agent"

    result = import_fleet_zip(source, target_dir=target)

    assert result.target_dir == target
    assert result.root_prompt_count == 1
    assert result.subagent_prompt_count == 1
    assert result.config_ignored is True
    assert (target / "AGENTS.md").read_text(encoding="utf-8") == "root prompt"
    assert (target / "skills" / "review" / "SKILL.md").read_text(encoding="utf-8") == (
        "---\nname: review\n---\nReview things."
    )
    assert (target / "agents" / "researcher" / "AGENTS.md").is_file()
    assert not (target / "subagents").exists()
    assert not (target / "tools.json").exists()
    assert not (target / "config.json").exists()
    assert not (target / ".mcp.json.setup").exists()
    assert not (target / ".mcp.json").exists()


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
    target = tmp_path / "home" / expected_id
    assert (target / "AGENTS.md").read_text(encoding="utf-8") == "root prompt"
    assert (target / "agents" / "researcher" / "AGENTS.md").is_file()
    if unexpected_id is not None:
        assert not (tmp_path / "home" / unexpected_id / "AGENTS.md").exists()


def test_import_fleet_cli_defaults_assistant_to_export_stem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "crowbar.zip"
    _write_zip(
        source,
        {
            "AGENTS.md": "root prompt",
            "subagents/researcher/AGENTS.md": "Research carefully.",
        },
    )
    monkeypatch.setenv("DEEPAGENTS_TALON_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("DEEPAGENTS_TALON_ASSISTANT_ID", raising=False)
    monkeypatch.delenv("AGENT_ASSISTANT_ID", raising=False)
    monkeypatch.setattr(sys, "argv", ["deepagents-talon", "import-fleet", str(source)])

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    target = tmp_path / "home" / "crowbar"
    assert (target / "AGENTS.md").read_text(encoding="utf-8") == "root prompt"
    assert (target / "agents" / "researcher" / "AGENTS.md").is_file()
    assert not (tmp_path / "home" / "default" / "AGENTS.md").exists()


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


def test_import_fleet_zip_repeated_imports_refresh_generated_files(tmp_path: Path) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    target = tmp_path / "agent-home" / "agent"
    target.mkdir(parents=True)
    (target / ".mcp.json").write_text("operator config", encoding="utf-8")
    (target / ".mcp.json.setup").write_text("operator notes", encoding="utf-8")
    _write_zip(
        first,
        {
            "AGENTS.md": "first root",
            "tools.json": "{ignored",
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
    assert (target / ".mcp.json.setup").read_text(encoding="utf-8") == "operator notes"
    assert (target / ".mcp.json").read_text(encoding="utf-8") == "operator config"
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
