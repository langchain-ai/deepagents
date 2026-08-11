"""Adapter from plugin MCP declarations to dcode MCP config dictionaries."""

from __future__ import annotations

import json
import logging
import re
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_code.plugins._json import json_object, json_value
from deepagents_code.plugins.layouts import accepts_mcp_schema, dialect_by_name
from deepagents_code.plugins.substitution import plugin_environment, substitute_json

if TYPE_CHECKING:
    from collections.abc import Mapping

    from deepagents_code.plugins.layouts import PluginDialect
    from deepagents_code.plugins.models import JsonObject, JsonValue, PluginInstance

logger = logging.getLogger(__name__)
# For example, `tools@example.com` becomes `tools_example_com_<hash>`.
_MCP_NAME_PART_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MCP_NAME_PART_LENGTH = 48
_REMOTE_TRANSPORTS = {"http", "sse", "streamable-http", "streamable_http"}


class _PluginMCPConfigError(ValueError):
    """Raised when a plugin-relative MCP path escapes its permitted root."""


def _resolved_within(path: Path, root: Path, field: str) -> Path:
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        msg = f"Could not resolve plugin MCP {field}: {exc}"
        raise _PluginMCPConfigError(msg) from exc
    if not resolved.is_relative_to(root):
        msg = f"Plugin MCP {field} escapes its permitted root"
        raise _PluginMCPConfigError(msg)
    return resolved


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
    servers = _plugin_mcp_server_map(plugin)
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
    if "mcpServers" in raw:
        wrapped = raw.get("mcpServers")
        return json_object(wrapped) if isinstance(wrapped, dict) else {}
    if "mcp_servers" in raw:
        wrapped = raw.get("mcp_servers")
        return json_object(wrapped) if isinstance(wrapped, dict) else {}
    return json_object(raw)


def _load_mcp_server_map(
    path: Path,
    *,
    dialect: PluginDialect | None = None,
) -> JsonObject:
    """Load an MCP config file and extract its server-name to config map.

    Args:
        path: MCP configuration path.
        dialect: Dialect policy for a conventional schema-bearing document.

    Returns:
        The extracted server map, or an empty map when the file cannot be read
        with the selected dialect.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping plugin MCP config %s: %s", path, exc)
        return {}
    schema = raw.get("$schema") if isinstance(raw, dict) else None
    if dialect is not None and not accepts_mcp_schema(dialect, schema):
        logger.warning(
            "Skipping plugin MCP config %s: unsupported schema %r",
            path,
            schema,
        )
        return {}
    return _server_map(raw)


def _plugin_mcp_server_map(plugin: PluginInstance) -> JsonObject:
    """Load a plugin's declared MCP servers without creating runtime state.

    Returns:
        The unscoped server configuration keyed by declared server name.
    """
    servers: JsonObject = {}
    dialect = (
        dialect_by_name(plugin.manifest.dialect)
        if plugin.manifest is not None
        else None
    )
    conventional_mcp = (
        (plugin.root / "mcp.json").resolve()
        if dialect is not None and dialect.mcp_schema is not None
        else None
    )
    for path in plugin.inventory.mcp_files:
        if path.suffix in {".mcpb", ".dxt"}:
            logger.warning(
                "Skipping unsupported MCP bundle for plugin %s: %s",
                plugin.plugin_id,
                path,
            )
            continue
        servers.update(
            _load_mcp_server_map(
                path,
                dialect=dialect if path == conventional_mcp else None,
            )
        )
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


def _is_stdio_server(server: Mapping[str, object]) -> bool:
    transport = server.get("type") or server.get("transport")
    if transport is not None and not isinstance(transport, str):
        return False
    return "url" not in server and transport not in _REMOTE_TRANSPORTS


def _normalize_server(
    server: object, *, plugin: PluginInstance, project_dir: Path | None
) -> JsonValue:
    plugin_root = plugin.root.resolve()
    plugin_data = plugin.data_dir.resolve()
    dialect = (
        dialect_by_name(plugin.manifest.dialect)
        if plugin.manifest is not None
        else None
    )
    enforce_containment = dialect is not None and dialect.mcp_schema is not None
    normalized_server = json_value(server)
    raw_cwd = (
        normalized_server.get("cwd") if isinstance(normalized_server, dict) else None
    )
    substituted = substitute_json(
        normalized_server,
        plugin_root=plugin_root,
        plugin_data=plugin_data,
        project_dir=project_dir,
    )
    if not isinstance(substituted, dict):
        return json_value(substituted)

    if _is_stdio_server(substituted):
        command = substituted.get("command")
        if isinstance(command, str) and command.startswith("./"):
            command_path = plugin_root / command[2:]
            resolved_command = (
                _resolved_within(command_path, plugin_root, "command")
                if enforce_containment
                else command_path.resolve()
            )
            substituted = {**substituted, "command": str(resolved_command)}
        cwd = substituted.get("cwd")
        if isinstance(cwd, str) and cwd and not Path(cwd).is_absolute():
            if "${" not in cwd:
                cwd_path = plugin_root / cwd
                resolved_cwd = (
                    _resolved_within(cwd_path, plugin_root, "cwd")
                    if enforce_containment
                    else cwd_path.resolve()
                )
                substituted = {**substituted, "cwd": str(resolved_cwd)}
        elif cwd is None:
            substituted = {**substituted, "cwd": str(plugin_root)}
        if enforce_containment and isinstance(raw_cwd, str):
            cwd_root = None
            if raw_cwd.startswith(("${PLUGIN_ROOT}", "${CLAUDE_PLUGIN_ROOT}")):
                cwd_root = plugin_root
            elif raw_cwd.startswith(("${PLUGIN_DATA}", "${CLAUDE_PLUGIN_DATA}")):
                cwd_root = plugin_data
            substituted_cwd = substituted.get("cwd")
            if cwd_root is not None and isinstance(substituted_cwd, str):
                resolved_cwd = _resolved_within(
                    Path(substituted_cwd),
                    cwd_root,
                    "cwd",
                )
                substituted = {**substituted, "cwd": str(resolved_cwd)}

    env = substituted.get("env")
    plugin_env = plugin_environment(
        plugin_root=plugin_root,
        plugin_data=plugin_data,
        project_dir=project_dir,
    )
    if isinstance(env, dict):
        substituted = {**substituted, "env": {**env, **plugin_env}}
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
        Plugin MCP config layers, or an empty tuple when discovery fails.
    """
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

    Conventional MCP files are loaded before manifest `mcpServers`, so manifest
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
            try:
                scoped[scoped_name] = _normalize_server(
                    server,
                    plugin=plugin,
                    project_dir=project_dir,
                )
            except _PluginMCPConfigError as exc:
                logger.warning(
                    "Skipping plugin MCP server %s from %s: %s",
                    name,
                    plugin.plugin_id,
                    exc,
                )
        if scoped:
            configs.append({"mcpServers": scoped})
    return configs
