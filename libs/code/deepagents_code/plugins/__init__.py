"""Plugin support for dcode."""

from deepagents_code.plugins.discovery import (
    add_local_marketplace,
    add_marketplace_source,
    discover_plugins,
    install_plugin,
    list_available_plugins,
    list_installed_plugin_ids,
    remove_marketplace,
    set_installed_plugin_enabled,
    uninstall_plugin,
)
from deepagents_code.plugins.models import PluginDiscoveryResult, PluginInstance

__all__ = [
    "PluginDiscoveryResult",
    "PluginInstance",
    "add_local_marketplace",
    "add_marketplace_source",
    "discover_plugins",
    "install_plugin",
    "list_available_plugins",
    "list_installed_plugin_ids",
    "remove_marketplace",
    "set_installed_plugin_enabled",
    "uninstall_plugin",
]
