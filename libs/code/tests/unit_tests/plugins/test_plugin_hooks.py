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
    from collections.abc import Mapping
    from pathlib import Path

PLUGIN_ID = "quality-review-plugin@company-tools"


def _hooks_document(command: str) -> dict[str, object]:
    return {
        "hooks": {
            "SessionStart": [{"hooks": [{"type": "command", "command": command}]}]
        }
    }


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _stage_plugins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    documents: Mapping[str, dict[str, object] | bytes],
) -> tuple[Path, Path]:
    monkeypatch.setenv(EXPERIMENTAL, "1")
    user_dir = tmp_path / "config"
    user_dir.mkdir(parents=True, exist_ok=True)
    for module in ("model_config", "hooks.loading", "hooks.runtime"):
        monkeypatch.setattr(f"deepagents_code.{module}.DEFAULT_CONFIG_DIR", user_dir)

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


def _install_all(root: Path, names: tuple[str, ...]) -> tuple[Path, ...]:
    add_local_marketplace(root)
    return tuple(install_plugin(f"{name}@company-tools").root for name in names)


@pytest.mark.skipif(
    sys.platform == "win32", reason="the hook command needs a POSIX shell"
)
async def test_plugin_hook_runs_with_its_exported_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.plugins.store import plugin_data_dir

    _, root = _stage_plugins(
        tmp_path,
        monkeypatch,
        {
            "quality-review-plugin": _hooks_document(
                "printf '%s\\n%s\\n%s\\n' "
                '"${CLAUDE_PLUGIN_ROOT}" "${CLAUDE_PLUGIN_DATA}" '
                '"${CLAUDE_PROJECT_DIR}" > "${CLAUDE_PLUGIN_DATA}/observed.txt"'
            )
        },
    )
    (plugin_root,) = _install_all(root, ("quality-review-plugin",))
    workspace = tmp_path / "$(touch${IFS}PWNED)"
    (workspace / ".git").mkdir(parents=True)

    manager = HooksManager.create(
        cwd=workspace,
        identity=lambda: HookSessionIdentity("thread", ApprovalMode.MANUAL),
    )
    await manager.on_session_start(SessionStartCause.STARTUP)

    observed = (plugin_data_dir(PLUGIN_ID) / "observed.txt").read_text(encoding="utf-8")
    assert observed.splitlines() == [
        str(plugin_root),
        str(plugin_data_dir(PLUGIN_ID)),
        str(workspace),
    ]
    assert not (workspace / "PWNED").exists()


def test_malformed_plugin_documents_are_isolated_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_dir, root = _stage_plugins(
        tmp_path,
        monkeypatch,
        {
            "quality-review-plugin": _hooks_document("plugin-hook"),
            "broken-plugin": b'{"hooks": {"Stop": "\xff\xfe"}}',
            "null-plugin": b"null",
        },
    )
    _install_all(root, ("quality-review-plugin", "broken-plugin", "null-plugin"))
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

    notices: list[tuple[str, object]] = []
    manager = HooksManager.create(
        cwd=tmp_path / "workspace",
        identity=lambda: HookSessionIdentity("thread", ApprovalMode.MANUAL),
        notice=lambda message, severity: notices.append((message, severity)),
    )

    assert manager.has_handlers(HookEvent.SESSION_START)
    assert manager.has_handlers(HookEvent.USER_PROMPT_SUBMIT)
    assert any("broken-plugin" in message for message, _severity in notices)
    assert any("null-plugin" in message for message, _severity in notices)
