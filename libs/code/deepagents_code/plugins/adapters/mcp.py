"""Adapter from plugin MCP declarations to dcode MCP config dictionaries."""

from __future__ import annotations

import json
import logging
import re
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_code.plugins._json import json_object, json_value
from deepagents_code.plugins.substitution import substitute_json

if TYPE_CHECKING:
    from deepagents_code.plugins.models import JsonObject, JsonValue, PluginInstance

logger = logging.getLogger(__name__)
_MCP_NAME_PART_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MCP_NAME_PART_LENGTH = 48


def _safe_mcp_name_part(value: str) -> str:
    sanitized = _MCP_NAME_PART_RE.sub("_", value).strip("_")
    if sanitized == value and sanitized and len(sanitized) <= _MCP_NAME_PART_LENGTH:
        return sanitized
    digest = sha256(value.encode()).hexdigest()[:8]
    prefix = sanitized[:_MCP_NAME_PART_LENGTH] or "unnamed"
    return f"{prefix}_{digest}"


def scoped_mcp_server_name(plugin_id: str, server_name: str) -> str:
    """Return an MCP-loader-safe scoped server name for a plugin server.

    Plugin identifiers may contain characters rejected by dcode's MCP loader.
    Use `__` as the namespace separator so names stay unique and valid.

    Args:
        plugin_id: Full plugin id in `name@marketplace` form.
        server_name: Unscoped server name from the plugin config.

    Returns:
        Scoped server name safe for `_SERVER_NAME_RE`.
    """
    plugin_part = _safe_mcp_name_part(plugin_id)
    server_part = _safe_mcp_name_part(server_name)
    return f"plugin__{plugin_part}__{server_part}"


def _server_map(raw: object) -> JsonObject:
    if not isinstance(raw, dict):
        return {}
    wrapped = raw.get("mcpServers")
    if isinstance(wrapped, dict):
        return json_object(wrapped)
    codex_wrapped = raw.get("mcp_servers")
    if isinstance(codex_wrapped, dict):
        return json_object(codex_wrapped)
    return json_object(raw)


def _load_mcp_file(path: Path) -> JsonObject:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping plugin MCP config %s: %s", path, exc)
        return {}
    return _server_map(raw)


def _normalize_server(
    server: object, *, plugin: PluginInstance, project_dir: Path | None
) -> JsonValue:
    normalized_server = json_value(server)
    substituted = substitute_json(
        normalized_server,
        plugin_root=plugin.root,
        plugin_data=plugin.data_dir,
        project_dir=project_dir,
    )
    if isinstance(substituted, dict):
        cwd = substituted.get("cwd")
        if isinstance(cwd, str) and cwd and not Path(cwd).is_absolute():
            substituted = {**substituted, "cwd": str((plugin.root / cwd).resolve())}
        env = substituted.get("env")
        plugin_env = {
            "CLAUDE_PLUGIN_ROOT": str(plugin.root),
            "CLAUDE_PLUGIN_DATA": str(plugin.data_dir),
            "PLUGIN_ROOT": str(plugin.root),
            "PLUGIN_DATA": str(plugin.data_dir),
            "DEEPAGENTS_PLUGIN_ROOT": str(plugin.root),
            "DEEPAGENTS_PLUGIN_DATA": str(plugin.data_dir),
        }
        if isinstance(env, dict):
            substituted = {**substituted, "env": {**plugin_env, **env}}
        else:
            substituted = {**substituted, "env": plugin_env}
    return json_value(substituted)


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
            scoped_name = scoped_mcp_server_name(plugin.plugin_id, name)
            scoped[scoped_name] = _normalize_server(
                server, plugin=plugin, project_dir=project_dir
            )
        if scoped:
            configs.append({"mcpServers": scoped})
    return configs
