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
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from deepagents_code.hooks.presenter import HookNoticeSeverity

PLUGIN_ID = "quality-review-plugin@company-tools"

_HOOK_SCRIPT = """#!/bin/sh
printf '%s\\n%s\\n%s\\n' \\
  "$CLAUDE_PLUGIN_ROOT" "$CLAUDE_PLUGIN_DATA" "$CLAUDE_PROJECT_DIR" \\
  > "$CLAUDE_PLUGIN_DATA/observed.txt"
"""


def _hooks_document(command: str) -> dict[str, object]:
    return {
        "hooks": {
            "SessionStart": [{"hooks": [{"type": "command", "command": command}]}]
        }
    }


@pytest.fixture(autouse=True)
def _enable_hooks_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hooks v2 only loads in experimental mode, which these tests exercise."""
    monkeypatch.setenv(EXPERIMENTAL, "1")


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _stage_plugins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    documents: Mapping[str, dict[str, object] | bytes],
) -> tuple[Path, Path]:
    """Stage a local marketplace whose plugins carry the given hooks documents.

    Nothing is installed yet, so a test may add further files to a plugin source
    before it is copied into the plugin cache.

    Args:
        tmp_path: Per-test temporary directory.
        monkeypatch: Fixture used to isolate plugin and hook storage.
        documents: Hooks document per plugin name, either as JSON-serializable
            data or as raw bytes for the malformed cases.

    Returns:
        The isolated user configuration directory and the marketplace root.
    """
    user_dir = tmp_path / "config"
    user_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_DIR", user_dir)
    monkeypatch.setattr("deepagents_code.hooks.loading.DEFAULT_CONFIG_DIR", user_dir)
    monkeypatch.setattr(
        "deepagents_code.hooks.runtime.DEFAULT_CONFIG_DIR", tmp_path / "transcripts"
    )

    root = tmp_path / "marketplace"
    _write_json(
        root / ".claude-plugin" / "marketplace.json",
        {
            "name": "company-tools",
            "owner": {"name": "Team"},
            "plugins": [
                {"name": name, "source": f"./plugins/{name}", "description": "Plugin"}
                for name in documents
            ],
        },
    )
    for name, document in documents.items():
        plugin = root / "plugins" / name
        _write_json(
            plugin / ".claude-plugin" / "plugin.json",
            {"name": name, "version": "1.0.0"},
        )
        hooks_path = plugin / "hooks" / "hooks.json"
        hooks_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(document, bytes):
            hooks_path.write_bytes(document)
        else:
            _write_json(hooks_path, document)
    return user_dir, root


def _install_all(root: Path, names: Iterable[str]) -> tuple[Path, ...]:
    """Register the staged marketplace and install every named plugin.

    Returns:
        The installed plugins' cached root directories, in `names` order.
    """
    add_local_marketplace(root)
    return tuple(install_plugin(f"{name}@company-tools").root for name in names)


def _notice_manager(
    cwd: Path, notices: list[tuple[str, HookNoticeSeverity]]
) -> HooksManager:
    return HooksManager.create(
        cwd=cwd,
        identity=lambda: HookSessionIdentity("thread", ApprovalMode.MANUAL),
        notice=lambda message, severity: notices.append((message, severity)),
    )


@pytest.mark.skipif(
    sys.platform == "win32", reason="the bundled hook script needs a POSIX shell"
)
async def test_plugin_hook_runs_with_its_exported_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundled plugin hook runs through the manager and sees its own paths."""
    from deepagents_code.plugins.store import plugin_data_dir

    _user_dir, root = _stage_plugins(
        tmp_path,
        monkeypatch,
        {
            "quality-review-plugin": _hooks_document(
                '"${CLAUDE_PLUGIN_ROOT}/scripts/check.sh"'
            )
        },
    )
    script = root / "plugins" / "quality-review-plugin" / "scripts" / "check.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(_HOOK_SCRIPT, encoding="utf-8")
    script.chmod(0o755)
    (plugin_root,) = _install_all(root, ("quality-review-plugin",))
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
    user_dir, root = _stage_plugins(
        tmp_path,
        monkeypatch,
        {
            "quality-review-plugin": _hooks_document("plugin-hook"),
            "broken-plugin": b'{"hooks": {"Stop": "\xff\xfe"}}',
        },
    )
    _install_all(root, ("quality-review-plugin", "broken-plugin"))
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

    notices: list[tuple[str, HookNoticeSeverity]] = []
    manager = _notice_manager(tmp_path / "workspace", notices)

    assert manager.has_handlers(HookEvent.SESSION_START)
    assert manager.has_handlers(HookEvent.USER_PROMPT_SUBMIT)
    assert any("broken-plugin" in message for message, _severity in notices)
