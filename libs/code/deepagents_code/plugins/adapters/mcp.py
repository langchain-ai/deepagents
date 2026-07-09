"""Adapter from plugin MCP declarations to dcode MCP config dictionaries."""

from __future__ import annotations

import json
import logging
import re
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_code.plugins.substitution import plugin_environment, substitute_json

if TYPE_CHECKING:
    from deepagents_code.plugins.models import PluginInstance

logger = logging.getLogger(__name__)
JsonObject = dict[str, object]
_MCP_NAME_PART_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MCP_NAME_PART_LENGTH = 48


def _safe_mcp_name_part(value: str) -> str:
    sanitized = _MCP_NAME_PART_RE.sub("_", value).strip("_")
    if sanitized == value and sanitized and len(sanitized) <= _MCP_NAME_PART_LENGTH:
        return sanitized
    digest = sha256(value.encode()).hexdigest()[:8]
    prefix = sanitized[:_MCP_NAME_PART_LENGTH] or "unnamed"
    return f"{prefix}_{digest}"


def scoped_mcp_server_name(plugin_name: str, server_name: str) -> str:
    """Return an MCP-loader-safe scoped server name for a plugin server.

    Plugin identifiers may contain characters rejected by dcode's MCP loader.
    Use `__` as the namespace separator so names stay unique and valid.

    Args:
        plugin_name: Logical plugin name.
        server_name: Unscoped server name from the plugin config.

    Returns:
        Scoped server name safe for `_SERVER_NAME_RE`.
    """
    plugin_part = _safe_mcp_name_part(plugin_name)
    server_part = _safe_mcp_name_part(server_name)
    return f"plugin__{plugin_part}__{server_part}"


def _string_key_map(raw: object) -> JsonObject:
    if not isinstance(raw, dict):
        return {}
    return {key: value for key, value in raw.items() if isinstance(key, str)}


def _server_map(raw: object) -> JsonObject:
    if not isinstance(raw, dict):
        return {}
    wrapped = raw.get("mcpServers")
    if isinstance(wrapped, dict):
        return _string_key_map(wrapped)
    codex_wrapped = raw.get("mcp_servers")
    if isinstance(codex_wrapped, dict):
        return _string_key_map(codex_wrapped)
    return _string_key_map(raw)


def _load_mcp_file(path: Path) -> JsonObject:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping plugin MCP config %s: %s", path, exc)
        return {}
    return _server_map(raw)


def _normalize_server(
    server: object, *, plugin: PluginInstance, project_dir: Path | None
) -> object | None:
    substituted = substitute_json(
        server,
        plugin_root=plugin.root,
        plugin_data=plugin.data_dir,
        project_dir=project_dir,
        warning_key=plugin.plugin_id,
    )
    if isinstance(substituted, dict):
        cwd = substituted.get("cwd")
        if isinstance(cwd, str) and cwd and not Path(cwd).is_absolute():
            resolved_cwd = (plugin.root / cwd).resolve()
            if not resolved_cwd.is_relative_to(plugin.root.resolve()):
                logger.warning(
                    "Skipping MCP server with cwd outside plugin root: %s", cwd
                )
                return None
            substituted = {**substituted, "cwd": str(resolved_cwd)}
        env = substituted.get("env")
        plugin_env = plugin_environment(
            plugin_root=plugin.root,
            plugin_data=plugin.data_dir,
            project_dir=project_dir,
        )
        if isinstance(env, dict):
            substituted = {**substituted, "env": {**plugin_env, **env}}
        else:
            substituted = {**substituted, "env": plugin_env}
    return substituted


def plugin_mcp_configs(
    plugins: tuple[PluginInstance, ...], *, project_dir: Path | None = None
) -> list[JsonObject]:
    """Build MCP config layers for enabled plugins.

    Default `.mcp.json` files are loaded before manifest `mcpServers`, so manifest
    entries win on server-name conflicts.

    Args:
        plugins: Enabled plugin instances.
        project_dir: Project directory for `${CLAUDE_PROJECT_DIR}` substitution.

    Returns:
        MCP config layers ready for dcode's merge path.
    """
    configs: list[JsonObject] = []
    for plugin in plugins:
        if not plugin.trusted:
            logger.warning(
                "Skipping MCP servers for untrusted plugin %s", plugin.plugin_id
            )
            continue
        servers: JsonObject = {}
        for path in plugin.inventory.mcp_files:
            if path.suffix in {".mcpb", ".dxt"}:
                logger.warning(
                    "Skipping unsupported MCP bundle for plugin %s: %s",
                    plugin.plugin_id,
                    path,
                )
                continue
            servers.update(_load_mcp_file(path))
        if plugin.manifest and plugin.manifest.inline_mcp:
            servers.update(_server_map(plugin.manifest.inline_mcp))
        scoped: JsonObject = {}
        for name, server in servers.items():
            if not isinstance(name, str):
                continue
            scoped_name = scoped_mcp_server_name(plugin.name, name)
            normalized = _normalize_server(
                server, plugin=plugin, project_dir=project_dir
            )
            if normalized is not None:
                scoped[scoped_name] = normalized
        if scoped:
            configs.append({"mcpServers": scoped})
    return configs
