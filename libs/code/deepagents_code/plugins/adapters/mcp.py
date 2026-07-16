"""Adapter from plugin MCP declarations to dcode MCP config dictionaries."""

from __future__ import annotations

import json
import logging
import re
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_code.plugins._json import json_object, json_value
from deepagents_code.plugins.substitution import plugin_environment, substitute_json

if TYPE_CHECKING:
    from deepagents_code.plugins.models import JsonObject, JsonValue, PluginInstance

logger = logging.getLogger(__name__)
# For example, `tools@example.com` becomes `tools_example_com_<hash>`.
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
    """Namespace a plugin-declared MCP server's name under its plugin id.

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


def _mcp_server_needs_login(server: object) -> bool:
    """Return whether an MCP server config typically requires interactive login."""
    if not isinstance(server, dict):
        return False
    server_type = server.get("type")
    if server_type in {"http", "sse"}:
        return True
    return isinstance(server.get("url"), str)


def plugin_mcp_server_entries(
    plugin: PluginInstance,
) -> tuple[tuple[str, str, bool], ...]:
    """List plugin MCP servers as `(label, scoped_name, needs_login)` tuples.

    `label` is the unscoped name from the plugin config (for UI). `scoped_name`
    is what dcode registers after namespacing.

    Args:
        plugin: Plugin whose MCP declarations should be listed.

    Returns:
        Deduplicated server entries in declaration order.
    """
    servers: dict[str, object] = {}
    for path in plugin.inventory.mcp_files:
        if path.suffix in {".mcpb", ".dxt"}:
            continue
        servers.update(_load_mcp_server_map(path))
    if plugin.manifest and plugin.manifest.inline_mcp:
        servers.update(_server_map(plugin.manifest.inline_mcp))
    entries: list[tuple[str, str, bool]] = []
    seen: set[str] = set()
    for name, server in servers.items():
        if not isinstance(name, str) or name in seen:
            continue
        seen.add(name)
        entries.append(
            (
                name,
                scoped_mcp_server_name(plugin.plugin_id, name),
                _mcp_server_needs_login(server),
            )
        )
    return tuple(entries)


def _server_map(raw: object) -> JsonObject:
    """Extract the server-name to config map from a decoded MCP document.

    Accepts Claude's `{"mcpServers": {...}}` wrapper, Codex's
    `{"mcp_servers": {...}}` wrapper, or a bare server map.

    Returns:
        The extracted server map, or an empty map for non-object input.
    """
    if not isinstance(raw, dict):
        return {}
    wrapped = raw.get("mcpServers")
    if isinstance(wrapped, dict):
        return json_object(wrapped)
    codex_wrapped = raw.get("mcp_servers")
    if isinstance(codex_wrapped, dict):
        return json_object(codex_wrapped)
    return json_object(raw)


def _load_mcp_server_map(path: Path) -> JsonObject:
    """Load an MCP config file and extract its server-name to config map.

    Returns:
        The extracted server map, or an empty map when the file cannot be read.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping plugin MCP config %s: %s", path, exc)
        return {}
    return _server_map(raw)


def _plugin_mcp_server_map(plugin: PluginInstance) -> JsonObject:
    """Load a plugin's declared MCP servers without creating runtime state.

    Returns:
        The unscoped server configuration keyed by declared server name.
    """
    servers: JsonObject = {}
    for path in plugin.inventory.mcp_files:
        if path.suffix in {".mcpb", ".dxt"}:
            logger.warning(
                "Skipping unsupported MCP bundle for plugin %s: %s",
                plugin.plugin_id,
                path,
            )
            continue
        servers.update(_load_mcp_server_map(path))
    if plugin.manifest and plugin.manifest.inline_mcp:
        servers.update(_server_map(plugin.manifest.inline_mcp))
    return servers


def plugin_mcp_server_names(plugin: PluginInstance) -> tuple[str, ...]:
    """Return scoped MCP server names without preparing plugin runtime state.

    Args:
        plugin: Plugin instance whose declarations should be inspected.

    Returns:
        Scoped MCP server names in declaration order.
    """
    return tuple(
        scoped_mcp_server_name(plugin.plugin_id, name)
        for name in _plugin_mcp_server_map(plugin)
        if isinstance(name, str)
    )


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
        plugin_env = plugin_environment(
            plugin_root=plugin.root,
            plugin_data=plugin.data_dir,
            project_dir=project_dir,
        )
        if isinstance(env, dict):
            substituted = {**substituted, "env": {**plugin_env, **env}}
        else:
            substituted = {**substituted, "env": plugin_env}
    return json_value(substituted)


def discover_plugin_mcp_configs(
    *, project_dir: Path | None = None
) -> tuple[JsonObject, ...]:
    """Discover enabled plugins and compose their MCP config layers.

    Args:
        project_dir: Project directory for variable substitution.

    Returns:
        Plugin MCP config layers, or an empty tuple when plugins are disabled or
        discovery fails.
    """
    from deepagents_code._env_vars import EXPERIMENTAL, is_env_truthy

    if not is_env_truthy(EXPERIMENTAL):
        return ()
    try:
        from deepagents_code.plugins import discover_plugins

        result = discover_plugins()
    except (OSError, RuntimeError):
        logger.warning("Could not discover plugin MCP configs", exc_info=True)
        return ()
    if result.warnings:
        logger.warning(
            "Plugin discovery warnings while loading MCP: %s", result.warnings
        )
    return tuple(plugin_mcp_configs(result.plugins, project_dir=project_dir))


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
        # Create the writable data dir when MCP configs need it. Discovery itself
        # only computes the path so it stays safe for blockbuster-guarded callers.
        try:
            plugin.data_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning(
                "Could not create plugin data dir for %s: %s",
                plugin.plugin_id,
                plugin.data_dir,
                exc_info=True,
            )
        servers = _plugin_mcp_server_map(plugin)
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
