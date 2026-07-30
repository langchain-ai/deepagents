"""Hooks contributed by installed plugins."""

from __future__ import annotations

import json
import sys
from types import MappingProxyType
from typing import TYPE_CHECKING

import pytest

from deepagents_code._env_vars import EXPERIMENTAL
from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks.loading import HooksSource
from deepagents_code.hooks.manager import HookSessionIdentity, HooksManager
from deepagents_code.hooks.models.domain import HookEvent, SessionStartCause
from deepagents_code.hooks.runtime import HooksRuntime
from deepagents_code.plugins import add_local_marketplace, install_plugin
from deepagents_code.plugins.adapters.hooks import discover_plugin_hook_sources

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_code.hooks.presenter import HookNoticeSeverity

PLUGIN_ID = "quality-review-plugin@company-tools"
OTHER_PLUGIN_ID = "broken-plugin@company-tools"

_HOOK_SCRIPT = """#!/bin/sh
printf '%s\\n%s\\n%s\\n' \\
  "$CLAUDE_PLUGIN_ROOT" "$CLAUDE_PLUGIN_DATA" "$CLAUDE_PROJECT_DIR" \\
  > "$CLAUDE_PLUGIN_DATA/observed.txt"
"""


@pytest.fixture(autouse=True)
def _enable_hooks_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hooks v2 only loads in experimental mode, which these tests exercise."""
    monkeypatch.setenv(EXPERIMENTAL, "1")


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _isolate_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point plugin storage and hook configuration at `tmp_path`.

    Returns:
        The isolated user configuration directory.
    """
    user_dir = tmp_path / "config"
    user_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_DIR", user_dir)
    monkeypatch.setattr("deepagents_code.hooks.loading.DEFAULT_CONFIG_DIR", user_dir)
    monkeypatch.setattr(
        "deepagents_code.hooks.runtime.DEFAULT_CONFIG_DIR", tmp_path / "transcripts"
    )
    return user_dir


def _marketplace(tmp_path: Path, plugin_names: tuple[str, ...]) -> Path:
    """Write a local marketplace manifest listing `plugin_names`.

    Returns:
        The marketplace root directory.
    """
    root = tmp_path / "marketplace"
    _write_json(
        root / ".claude-plugin" / "marketplace.json",
        {
            "name": "company-tools",
            "owner": {"name": "Team"},
            "plugins": [
                {
                    "name": name,
                    "source": f"./plugins/{name}",
                    "description": "Plugin",
                }
                for name in plugin_names
            ],
        },
    )
    return root


def _install_hooks_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install an enabled plugin whose `hooks/hooks.json` runs a bundled script.

    Returns:
        The installed plugin's cached root directory.
    """
    _isolate_config(tmp_path, monkeypatch)
    root = _marketplace(tmp_path, ("quality-review-plugin",))
    plugin = root / "plugins" / "quality-review-plugin"
    _write_json(
        plugin / ".claude-plugin" / "plugin.json",
        {"name": "quality-review-plugin", "version": "1.0.0"},
    )
    script = plugin / "scripts" / "check.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(_HOOK_SCRIPT, encoding="utf-8")
    script.chmod(0o755)
    _write_json(
        plugin / "hooks" / "hooks.json",
        {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": '"${CLAUDE_PLUGIN_ROOT}/scripts/check.sh"',
                            }
                        ]
                    }
                ]
            }
        },
    )
    add_local_marketplace(root)
    return install_plugin(PLUGIN_ID).root


@pytest.mark.skipif(
    sys.platform == "win32", reason="the bundled hook script needs a POSIX shell"
)
async def test_plugin_hook_runs_with_its_exported_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundled plugin hook runs through the manager and sees its own paths."""
    from deepagents_code.plugins.store import plugin_data_dir

    plugin_root = _install_hooks_plugin(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    (workspace / ".git").mkdir(parents=True)
    monkeypatch.chdir(workspace)

    manager = HooksManager.create(
        cwd=workspace,
        identity=lambda: HookSessionIdentity("thread", ApprovalMode.MANUAL),
    )
    outcome = await manager.on_session_start(SessionStartCause.STARTUP)

    assert outcome.ok
    observed = (plugin_data_dir(PLUGIN_ID) / "observed.txt").read_text(encoding="utf-8")
    assert observed.splitlines() == [
        str(plugin_root),
        str(plugin_data_dir(PLUGIN_ID)),
        str(workspace),
    ]


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
                "Stop": [{"hooks": [{"type": "command", "command": "project-only"}]}]
            }
        },
    )

    manager = HooksManager.create(
        cwd=workspace,
        identity=lambda: HookSessionIdentity("thread", ApprovalMode.MANUAL),
    )

    assert manager.has_handlers(HookEvent.SESSION_START)
    assert not manager.has_handlers(HookEvent.STOP)


def test_unreadable_plugin_document_is_isolated_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One plugin's undecodable hooks document must not disable other hooks."""
    user_dir = _isolate_config(tmp_path, monkeypatch)
    root = _marketplace(tmp_path, ("quality-review-plugin", "broken-plugin"))
    for name in ("quality-review-plugin", "broken-plugin"):
        _write_json(
            root / "plugins" / name / ".claude-plugin" / "plugin.json",
            {"name": name, "version": "1.0.0"},
        )
    _write_json(
        root / "plugins" / "quality-review-plugin" / "hooks" / "hooks.json",
        {
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": "plugin-hook"}]}
                ]
            }
        },
    )
    broken = root / "plugins" / "broken-plugin" / "hooks" / "hooks.json"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b'{"hooks": {"Stop": "\xff\xfe"}}')
    _write_json(
        user_dir / "hooks.json",
        {
            "hooks": {
                "UserPromptSubmit": [
                    {"hooks": [{"type": "command", "command": "user-hook"}]}
                ]
            }
        },
    )
    add_local_marketplace(root)
    install_plugin(PLUGIN_ID)
    install_plugin(OTHER_PLUGIN_ID)

    notices: list[tuple[str, HookNoticeSeverity]] = []
    manager = HooksManager.create(
        cwd=tmp_path / "workspace",
        identity=lambda: HookSessionIdentity("thread", ApprovalMode.MANUAL),
        notice=lambda message, severity: notices.append((message, severity)),
    )

    assert manager.has_handlers(HookEvent.SESSION_START)
    assert manager.has_handlers(HookEvent.USER_PROMPT_SUBMIT)
    assert any("broken-plugin" in message for message, _severity in notices)


def test_malformed_plugin_document_is_reported_not_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plugin document that is not a hooks object produces a diagnostic."""
    _isolate_config(tmp_path, monkeypatch)
    root = _marketplace(tmp_path, ("quality-review-plugin",))
    plugin = root / "plugins" / "quality-review-plugin"
    _write_json(
        plugin / ".claude-plugin" / "plugin.json",
        {"name": "quality-review-plugin", "version": "1.0.0"},
    )
    hooks_path = plugin / "hooks" / "hooks.json"
    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    hooks_path.write_text('["not", "an", "object"]', encoding="utf-8")
    add_local_marketplace(root)
    installed = install_plugin(PLUGIN_ID).root / "hooks" / "hooks.json"

    notices: list[tuple[str, HookNoticeSeverity]] = []
    HooksManager.create(
        cwd=tmp_path / "workspace",
        identity=lambda: HookSessionIdentity("thread", ApprovalMode.MANUAL),
        notice=lambda message, severity: notices.append((message, severity)),
    )

    assert any(str(installed) in message for message, _severity in notices)


def test_plugin_hooks_change_the_snapshot_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plugin provenance participates in the snapshot hash."""
    _install_hooks_plugin(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    with_plugin = HooksRuntime.create(
        cwd=workspace,
        transcript_root=tmp_path / "transcripts",
        plugin_sources=discover_plugin_hook_sources().documents,
    )
    without_plugin = HooksRuntime.create(
        cwd=workspace,
        transcript_root=tmp_path / "transcripts",
    )

    assert with_plugin.snapshot_id != without_plugin.snapshot_id


def test_hooks_source_rejects_an_environment_without_a_plugin() -> None:
    """Only a plugin source may carry an overlay, so hashing matches execution."""
    with pytest.raises(ValueError, match="environment overlay"):
        HooksSource(location="/tmp/hooks.json", env={"CLAUDE_PLUGIN_ROOT": "/tmp"})


def test_hooks_source_freezes_its_environment() -> None:
    """A frozen source must not share a caller's mutable environment mapping."""
    env = {"CLAUDE_PLUGIN_ROOT": "/tmp/plugin"}
    source = HooksSource(location="/tmp/hooks.json", plugin_id=PLUGIN_ID, env=env)

    env["CLAUDE_PLUGIN_ROOT"] = "/tmp/other"

    assert source.env["CLAUDE_PLUGIN_ROOT"] == "/tmp/plugin"
    assert isinstance(source.env, MappingProxyType)


def test_plugin_hook_paths_are_substituted_for_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`argv` never reaches a shell, so its variables resolve at load time."""
    _isolate_config(tmp_path, monkeypatch)
    root = _marketplace(tmp_path, ("quality-review-plugin",))
    plugin = root / "plugins" / "quality-review-plugin"
    _write_json(
        plugin / ".claude-plugin" / "plugin.json",
        {"name": "quality-review-plugin", "version": "1.0.0"},
    )
    _write_json(
        plugin / "hooks" / "hooks.json",
        {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "check",
                                "argv": ["${CLAUDE_PLUGIN_ROOT}/scripts/check.sh"],
                            }
                        ]
                    }
                ]
            }
        },
    )
    add_local_marketplace(root)
    installed_root = install_plugin(PLUGIN_ID).root

    runtime = HooksRuntime.create(
        cwd=tmp_path / "workspace",
        transcript_root=tmp_path / "transcripts",
        plugin_sources=discover_plugin_hook_sources().documents,
    )

    (handler,) = runtime.snapshot.handlers[HookEvent.SESSION_START]
    assert handler.argv == (str(installed_root / "scripts" / "check.sh"),)
    assert handler.source.plugin_id == PLUGIN_ID
    assert handler.source.env["CLAUDE_PLUGIN_ROOT"] == str(installed_root)
