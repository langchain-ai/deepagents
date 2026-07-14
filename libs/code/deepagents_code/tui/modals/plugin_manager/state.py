"""Plugin manager state loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepagents_code.mcp_tools import MCPServerInfo
    from deepagents_code.plugins.models import PluginInstance

from deepagents_code.plugins import discover_plugins
from deepagents_code.plugins.marketplace import (
    MarketplaceError,
    load_marketplace_location,
    redact_marketplace_source,
    redact_urls_in_text,
)
from deepagents_code.plugins.models import split_plugin_id
from deepagents_code.plugins.store import (
    get_primary_install_entry,
    load_enabled_plugin_ids,
    load_installed_plugins,
    load_marketplace_records,
)
from deepagents_code.tui.modals.plugin_manager.models import (
    _ManagerState,
    _MarketplaceRow,
    _PluginRow,
)

logger = logging.getLogger(__name__)


def _extract_name(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str):
            return name
    return None


def _list_plugin_skill_names(instance: PluginInstance) -> tuple[str, ...]:
    from deepagents.backends.filesystem import FilesystemBackend

    from deepagents_code.plugins.adapters.skills import plugin_skill_sources
    from deepagents_code.plugins.adapters.skills_middleware import (
        load_namespaced_skills,
    )

    names: list[str] = []
    for path, _label, namespace in plugin_skill_sources((instance,)):
        try:
            source = Path(path).resolve()
            backend = FilesystemBackend(root_dir=str(source), virtual_mode=False)
            names.extend(
                skill["name"]
                for skill in load_namespaced_skills(backend, str(source), namespace)
            )
        except (OSError, RuntimeError):
            logger.warning(
                "Could not list skills for plugin %s", instance.plugin_id, exc_info=True
            )
    return tuple(dict.fromkeys(names))


def _plugin_mcp_server_names(instance: PluginInstance) -> tuple[str, ...]:
    from deepagents_code.plugins.adapters.mcp import plugin_mcp_configs

    names: list[str] = []
    for config in plugin_mcp_configs((instance,)):
        servers = config.get("mcpServers")
        if isinstance(servers, dict):
            names.extend(key for key in servers if isinstance(key, str))
    return tuple(dict.fromkeys(names))


def _plugin_mcp_connected(
    instance: PluginInstance, mcp_server_info: Sequence[MCPServerInfo]
) -> bool | None:
    expected = frozenset(_plugin_mcp_server_names(instance))
    if not expected:
        return None
    connected = {info.name for info in mcp_server_info if info.status == "ok"}
    return expected <= connected


def _instance_for_manager_row(
    plugin_id: str,
    *,
    discovered: dict[str, PluginInstance],
    is_installed: bool,
    errors: list[str],
) -> PluginInstance | None:
    instance = discovered.get(plugin_id)
    if instance is not None:
        return instance
    if not is_installed:
        return None
    entry = get_primary_install_entry(plugin_id)
    if entry is None:
        return None
    root = Path(entry.install_path)
    try:
        installed = root.is_dir()
    except (OSError, RuntimeError) as exc:
        errors.append(f"{plugin_id}: could not inspect install path: {exc}")
        return None
    if not installed:
        return None
    from deepagents_code.plugins.discovery import _plugin_from_install_path

    try:
        plugin_name, marketplace_name = split_plugin_id(plugin_id)
    except ValueError:
        return None
    try:
        loaded, warnings = _plugin_from_install_path(
            plugin_id=plugin_id,
            root=root,
            marketplace_name=marketplace_name,
            fallback_name=plugin_name,
        )
    except (OSError, RuntimeError) as exc:
        errors.append(f"{plugin_id}: {exc}")
        return None
    errors.extend(warnings)
    return loaded


def _load_manager_state(mcp_server_info: Sequence[MCPServerInfo] = ()) -> _ManagerState:
    records = load_marketplace_records()
    enabled = load_enabled_plugin_ids()
    installed = load_installed_plugins()
    errors: list[str] = []
    plugin_result = discover_plugins()
    errors.extend(plugin_result.warnings)
    discovered = {instance.plugin_id: instance for instance in plugin_result.plugins}
    available_plugins: list[_PluginRow] = []
    installed_plugins: list[_PluginRow] = []
    marketplaces: list[_MarketplaceRow] = []
    for name, record in sorted(records.items()):
        try:
            marketplace = load_marketplace_location(Path(record.install_location))
        except MarketplaceError as exc:
            detail = redact_urls_in_text(str(exc))
            if record.source_type not in {"directory", "file"}:
                detail = detail.replace(record.install_location, "<managed cache>")
            errors.append(f"{name}: {detail}")
            marketplaces.append(
                _MarketplaceRow(
                    name,
                    redact_marketplace_source(record.source),
                    None,
                    sum(plugin_id.endswith(f"@{name}") for plugin_id in installed),
                    detail,
                )
            )
            continue
        marketplaces.append(
            _MarketplaceRow(
                marketplace.name,
                redact_marketplace_source(record.source),
                len(marketplace.plugins),
                sum(
                    plugin_id.endswith(f"@{marketplace.name}")
                    for plugin_id in installed
                ),
            )
        )
        errors.extend(
            f"{marketplace.name}: {warning}" for warning in marketplace.warnings
        )
        for plugin in marketplace.plugins:
            plugin_id = f"{plugin.name}@{marketplace.name}"
            is_enabled = plugin_id in enabled
            is_installed = plugin_id in installed
            instance = _instance_for_manager_row(
                plugin_id,
                discovered=discovered,
                is_installed=is_installed,
                errors=errors,
            )
            skill_names = _list_plugin_skill_names(instance) if instance else ()
            mcp_names = _plugin_mcp_server_names(instance) if instance else ()
            row = _PluginRow(
                plugin_id,
                plugin.description or "",
                is_enabled,
                instance.version if instance else None,
                _extract_name(plugin.author),
                len(skill_names) if instance else None,
                skill_names,
                _plugin_mcp_connected(instance, mcp_server_info)
                if instance and is_enabled
                else None,
                mcp_names,
            )
            (installed_plugins if is_installed else available_plugins).append(row)
    return _ManagerState(
        tuple(available_plugins),
        tuple(installed_plugins),
        tuple(marketplaces),
        tuple(dict.fromkeys(errors)),
    )
