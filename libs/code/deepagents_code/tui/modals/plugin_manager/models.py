"""Plugin manager view models."""

from dataclasses import dataclass
from typing import Literal

PluginTab = Literal["discover", "installed", "marketplaces", "errors"]
PluginManagerView = Literal[
    "list",
    "add_marketplace",
    "plugin_details",
    "installed_details",
    "marketplace_details",
    "confirm_remove_marketplace",
]


@dataclass(frozen=True, slots=True)
class _PluginRow:
    plugin_id: str
    description: str
    enabled: bool
    version: str | None
    author: str | None
    skill_count: int | None = None
    skill_names: tuple[str, ...] = ()
    mcp_connected: bool | None = None
    mcp_server_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _MarketplaceRow:
    name: str
    source: str
    plugin_count: int | None
    installed_count: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _ManagerState:
    available_plugins: tuple[_PluginRow, ...]
    installed_plugins: tuple[_PluginRow, ...]
    marketplaces: tuple[_MarketplaceRow, ...]
    errors: tuple[str, ...]
