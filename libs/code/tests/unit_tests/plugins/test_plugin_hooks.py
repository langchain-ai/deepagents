"""Hooks contributed by installed plugins."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from deepagents_code._env_vars import EXPERIMENTAL
from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks.manager import HookSessionIdentity, HooksManager
from deepagents_code.hooks.models.domain import HookEvent, SessionStartCause
from deepagents_code.plugins import add_local_marketplace, install_plugin

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
