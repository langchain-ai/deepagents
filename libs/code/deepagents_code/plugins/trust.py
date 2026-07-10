"""Persistent trust decisions for plugin executable surfaces."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from deepagents_code.plugins.store import _atomic_write_json, _load_json

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_code.plugins.models import ComponentInventory, PluginManifest

_TRUST_VERSION = 1


def _trust_path() -> Path:
    from deepagents_code.model_config import DEFAULT_STATE_DIR

    return DEFAULT_STATE_DIR / "plugin_trust.json"


def _file_fingerprint(path: Path, root: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        return {"path": str(path), "sha256": "outside-plugin-root"}
    try:
        digest = sha256(resolved.read_bytes()).hexdigest()
    except OSError:
        digest = "unreadable"
    return {"path": str(resolved.relative_to(root.resolve())), "sha256": digest}


def plugin_surface_fingerprint(
    *,
    plugin_id: str,
    version: str,
    root: Path,
    manifest: PluginManifest | None,
    inventory: ComponentInventory,
) -> str:
    """Return a stable fingerprint of a plugin's executable surface."""
    payload: dict[str, Any] = {
        "plugin_id": plugin_id,
        "version": version,
        "mcp_files": [
            _file_fingerprint(path, root)
            for path in sorted(inventory.mcp_files, key=str)
        ],
        "inline_mcp": manifest.inline_mcp if manifest else {},
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode()
    return sha256(encoded).hexdigest()


def is_plugin_surface_trusted(
    plugin_id: str, *, version: str, fingerprint: str
) -> bool:
    """Return whether the exact plugin executable surface is trusted."""
    data = _load_json(_trust_path(), max_version=_TRUST_VERSION)
    entries = data.get("plugins")
    if not isinstance(entries, dict):
        return False
    entry = entries.get(plugin_id)
    return (
        isinstance(entry, dict)
        and entry.get("version") == version
        and entry.get("fingerprint") == fingerprint
    )


def trust_plugin_surface(plugin_id: str, *, version: str, fingerprint: str) -> None:
    """Trust an exact plugin executable surface."""
    data = _load_json(_trust_path(), max_version=_TRUST_VERSION)
    entries = data.get("plugins")
    if not isinstance(entries, dict):
        entries = {}
    entries[plugin_id] = {
        "fingerprint": fingerprint,
        "trustedAt": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "version": version,
    }
    _atomic_write_json(_trust_path(), {"version": _TRUST_VERSION, "plugins": entries})


def revoke_plugin_trust(plugin_id: str) -> None:
    """Remove a plugin trust decision."""
    data = _load_json(_trust_path(), max_version=_TRUST_VERSION)
    entries = data.get("plugins")
    if not isinstance(entries, dict) or plugin_id not in entries:
        return
    entries.pop(plugin_id, None)
    _atomic_write_json(_trust_path(), {"version": _TRUST_VERSION, "plugins": entries})
