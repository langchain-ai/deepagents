"""Hooks contributed by installed plugins."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks.manager import HookSessionIdentity, HooksManager
from deepagents_code.hooks.models.domain import HookEvent, SessionStartCause
from deepagents_code.plugins import add_local_marketplace, install_plugin

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import pytest

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
