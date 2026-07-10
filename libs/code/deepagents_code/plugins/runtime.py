"""Atomic runtime snapshots for active plugins."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from deepagents_code.plugins.adapters.agents import plugin_agents
from deepagents_code.plugins.adapters.commands import PluginCommand, plugin_commands
from deepagents_code.plugins.adapters.mcp import plugin_mcp_configs
from deepagents_code.plugins.adapters.skills import (
    SkillSourceTuple,
    plugin_skill_roots,
    plugin_skill_sources,
)
from deepagents_code.plugins.discovery import discover_plugins

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_code.plugins.models import PluginDiscoveryResult
    from deepagents_code.subagents import SubagentMetadata


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginRuntimeSnapshot:
    """Validated plugin state consumed by runtime integrations."""

    discovery: PluginDiscoveryResult
    project_dir: Path | None
    skill_sources: tuple[SkillSourceTuple, ...]
    skill_roots: tuple[Path, ...]
    commands: tuple[PluginCommand, ...]
    agents: tuple[SubagentMetadata, ...]
    mcp_configs: tuple[dict[str, Any], ...]
    fingerprint: str


_SNAPSHOT_LOCK = threading.RLock()


@dataclass(slots=True)
class _SnapshotState:
    active: PluginRuntimeSnapshot | None = None


_STATE = _SnapshotState()


def _path_stamp(path: Path) -> tuple[str, str]:
    try:
        if path.is_file():
            stat = path.stat()
            stamp = f"{stat.st_mtime_ns}:{stat.st_size}"
        elif path.is_dir():
            entries: list[str] = []
            for child in sorted(path.rglob("*")):
                if not child.is_file():
                    continue
                stat = child.stat()
                entries.append(
                    f"{child.relative_to(path)}:{stat.st_mtime_ns}:{stat.st_size}"
                )
            stamp = sha256("\n".join(entries).encode()).hexdigest()
        else:
            stamp = "missing"
    except OSError:
        stamp = "unreadable"
    return str(path), stamp


def _snapshot_fingerprint(result: PluginDiscoveryResult) -> str:
    payload: list[dict[str, object]] = []
    for plugin in result.plugins:
        component_paths = (
            *plugin.inventory.skills,
            *plugin.inventory.commands,
            *plugin.inventory.agents,
            *plugin.inventory.mcp_files,
        )
        payload.append(
            {
                "id": plugin.plugin_id,
                "version": plugin.version,
                "trusted": plugin.trusted,
                "components": [_path_stamp(path) for path in component_paths],
            }
        )
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode()
    return sha256(encoded).hexdigest()


def build_plugin_snapshot(*, project_dir: Path | None = None) -> PluginRuntimeSnapshot:
    """Build a complete plugin runtime snapshot without activating it.

    Returns:
        Complete candidate snapshot.
    """
    result = discover_plugins()
    resolved_project = project_dir.resolve() if project_dir is not None else None
    return PluginRuntimeSnapshot(
        discovery=result,
        project_dir=resolved_project,
        skill_sources=tuple(plugin_skill_sources(result.plugins)),
        skill_roots=tuple(plugin_skill_roots(result.plugins)),
        commands=plugin_commands(result.plugins),
        agents=plugin_agents(result.plugins),
        mcp_configs=tuple(plugin_mcp_configs(result.plugins, project_dir=project_dir)),
        fingerprint=_snapshot_fingerprint(result),
    )


def reload_plugin_snapshot(*, project_dir: Path | None = None) -> PluginRuntimeSnapshot:
    """Build and atomically activate a fresh plugin snapshot.

    Returns:
        Newly activated snapshot.
    """
    candidate = build_plugin_snapshot(project_dir=project_dir)
    with _SNAPSHOT_LOCK:
        _STATE.active = candidate
    return candidate


def get_plugin_snapshot(*, project_dir: Path | None = None) -> PluginRuntimeSnapshot:
    """Return the active snapshot, building it on first use."""
    resolved_project = project_dir.resolve() if project_dir is not None else None
    with _SNAPSHOT_LOCK:
        snapshot = _STATE.active
    if snapshot is not None and snapshot.project_dir == resolved_project:
        return snapshot
    return reload_plugin_snapshot(project_dir=project_dir)


def clear_plugin_snapshot() -> None:
    """Clear the active snapshot."""
    with _SNAPSHOT_LOCK:
        _STATE.active = None
