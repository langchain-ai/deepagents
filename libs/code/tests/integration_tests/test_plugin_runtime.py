"""No-network integration coverage for session plugin snapshots."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

from deepagents_code._env_vars import EXPERIMENTAL, PLUGIN_DIRS
from deepagents_code.plugins.runtime import (
    clear_plugin_snapshot,
    reload_plugin_snapshot,
)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_session_plugin_builds_complete_runtime_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = tmp_path / "plugin"
    _write_json(
        plugin / ".claude-plugin" / "plugin.json",
        {"name": "integration-plugin"},
    )
    skill = plugin / "skills" / "review" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: review\ndescription: Review code.\n---\nReview.",
        encoding="utf-8",
    )
    command = plugin / "commands" / "check.md"
    command.parent.mkdir()
    command.write_text(
        "---\ndescription: Check code.\n---\nCheck $ARGUMENTS.",
        encoding="utf-8",
    )
    agent = plugin / "agents" / "reviewer.md"
    agent.parent.mkdir()
    agent.write_text(
        "---\ndescription: Review code.\n---\nReview carefully.",
        encoding="utf-8",
    )
    _write_json(
        plugin / ".mcp.json",
        {"mcpServers": {"docs": {"command": "docs-server"}}},
    )
    _write_json(
        plugin / "hooks" / "hooks.json",
        {
            "hooks": {
                "PostToolUse": [
                    {"hooks": [{"type": "command", "command": "echo complete"}]}
                ]
            }
        },
    )
    monkeypatch.setenv(EXPERIMENTAL, "1")
    monkeypatch.setenv(PLUGIN_DIRS, str(plugin))
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    clear_plugin_snapshot()

    snapshot = reload_plugin_snapshot(project_dir=tmp_path)

    assert snapshot.discovery.plugins[0].plugin_id == "integration-plugin@inline"
    assert snapshot.skill_sources[0][2] == "integration-plugin:"
    assert snapshot.commands[0].name == "integration-plugin:check"
    assert snapshot.agents[0]["name"] == "integration-plugin:reviewer"
    assert snapshot.hooks[0].source_event == "PostToolUse"
    assert snapshot.mcp_configs
