"""Hooks contributed by installed plugins."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_code.hooks.models.domain import HookEvent
from deepagents_code.hooks.runtime import HooksRuntime
from deepagents_code.plugins import add_local_marketplace, install_plugin
from deepagents_code.plugins.adapters.hooks import discover_plugin_hook_sources
from deepagents_code.plugins.store import plugin_data_dir

if TYPE_CHECKING:
    import pytest

PLUGIN_ID = "quality-review-plugin@company-tools"


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _install_hooks_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Install an enabled plugin whose `hooks/hooks.json` runs a bundled script."""
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    root = tmp_path / "marketplace"
    _write_json(
        root / ".claude-plugin" / "marketplace.json",
        {
            "name": "company-tools",
            "owner": {"name": "Team"},
            "plugins": [
                {
                    "name": "quality-review-plugin",
                    "source": "./plugins/quality-review-plugin",
                    "description": "Quality review",
                }
            ],
        },
    )
    plugin = root / "plugins" / "quality-review-plugin"
    _write_json(
        plugin / ".claude-plugin" / "plugin.json",
        {"name": "quality-review-plugin", "version": "1.0.0"},
    )
    _write_json(
        plugin / "hooks" / "hooks.json",
        {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Bash",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "${CLAUDE_PLUGIN_ROOT}/scripts/check.sh",
                            }
                        ],
                    }
                ]
            }
        },
    )
    add_local_marketplace(root)
    install_plugin(PLUGIN_ID)


def test_plugin_hooks_resolve_plugin_root_in_command_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_hooks_plugin(tmp_path, monkeypatch)

    runtime = HooksRuntime.create(
        cwd=tmp_path / "workspace",
        config_dir=tmp_path / "config",
        transcript_root=tmp_path / "transcripts",
        plugin_sources=discover_plugin_hook_sources(),
    )

    (handler,) = runtime.snapshot.handlers[HookEvent.PRE_TOOL_USE]
    assert "${CLAUDE_PLUGIN_ROOT}" not in handler.command
    assert handler.command.endswith("/scripts/check.sh")
    assert handler.source.origin == PLUGIN_ID
    assert handler.source.env["CLAUDE_PLUGIN_ROOT"] == str(
        Path(handler.command).parent.parent
    )
    assert handler.source.env["CLAUDE_PLUGIN_DATA"] == str(plugin_data_dir(PLUGIN_ID))


def test_plugin_hooks_load_without_workspace_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enabling a plugin is its own consent gate, and is not project trust."""
    _install_hooks_plugin(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    _write_json(
        workspace / ".deepagents" / "hooks.json",
        {
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": "project-only"}]}
                ]
            }
        },
    )

    runtime = HooksRuntime.create(
        cwd=workspace,
        workspace_trusted=False,
        config_dir=tmp_path / "config",
        transcript_root=tmp_path / "transcripts",
        plugin_sources=discover_plugin_hook_sources(),
    )

    assert runtime.configured_events() == {HookEvent.PRE_TOOL_USE}
    assert not runtime.project_hooks_loaded
