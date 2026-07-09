"""Plugin support for dcode."""

from deepagents_code.plugins.discovery import (
    add_local_marketplace,
    add_marketplace_source,
    disable_plugin,
    discover_plugins,
    enable_plugin,
    enable_plugin_with_scope,
    get_plugin_info,
    install_plugin,
    list_available_plugins,
    list_installed_plugin_ids,
    remove_marketplace,
    trust_plugin,
    uninstall_plugin,
    update_marketplace,
)
from deepagents_code.plugins.models import (
    PluginDiscoveryResult,
    PluginInstance,
    PluginLoadError,
)

__all__ = [
    "PluginDiscoveryResult",
    "PluginInstance",
    "PluginLoadError",
    "add_local_marketplace",
    "add_marketplace_source",
    "disable_plugin",
    "discover_plugins",
    "enable_plugin",
    "enable_plugin_with_scope",
    "get_plugin_info",
    "install_plugin",
    "list_available_plugins",
    "list_installed_plugin_ids",
    "remove_marketplace",
    "trust_plugin",
    "uninstall_plugin",
    "update_marketplace",
]
