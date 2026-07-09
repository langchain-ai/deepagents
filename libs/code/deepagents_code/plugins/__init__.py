"""Plugin support for dcode."""

from deepagents_code.plugins.discovery import (
    add_local_marketplace,
    add_marketplace_source,
    disable_plugin,
    discover_plugins,
    enable_plugin,
    enable_plugin_with_scope,
    install_plugin,
    list_available_plugins,
    list_installed_plugin_ids,
    uninstall_plugin,
)
from deepagents_code.plugins.models import PluginDiscoveryResult, PluginInstance

__all__ = [
    "PluginDiscoveryResult",
    "PluginInstance",
    "add_local_marketplace",
    "add_marketplace_source",
    "disable_plugin",
    "discover_plugins",
    "enable_plugin",
    "enable_plugin_with_scope",
    "install_plugin",
    "list_available_plugins",
    "list_installed_plugin_ids",
    "uninstall_plugin",
]
