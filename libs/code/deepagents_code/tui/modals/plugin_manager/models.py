"""Plugin manager view models."""

from dataclasses import dataclass
from typing import Literal

from deepagents_code.plugins.models import UnsupportedComponent

PluginTab = Literal["discover", "installed", "marketplaces", "errors"]
PluginManagerView = Literal[
    "list",
    "add_marketplace",
    "plugin_details",
    "installed_details",
    "marketplace_details",
    "confirm_remove_marketplace",
]
PluginLoadState = Literal["disabled", "pending_reload", "enabled", "error"]


@dataclass(frozen=True, slots=True, kw_only=True)
class _PluginRow:
    plugin_id: str
    description: str
    enabled: bool
    version: str | None
    author: str | None
    display_name: str = ""
    skill_count: int | None = None
    skill_names: tuple[str, ...] = ()
    mcp_connected: bool | None = None
    mcp_server_names: tuple[str, ...] = ()
    mcp_login_servers: tuple[str, ...] = ()
    unsupported_components: tuple[UnsupportedComponent, ...] = ()
    session_loaded: bool = False
    load_error: str | None = None

    @property
    def load_state(self) -> PluginLoadState:
        """Session-aware plugin status for list and detail copy."""
        if self.load_error:
            return "error"
        if self.enabled != self.session_loaded:
            return "pending_reload"
        if self.enabled:
            return "enabled"
        return "disabled"

    @property
    def label(self) -> str:
        """Human-readable plugin name for UI copy."""
        if self.display_name:
            return self.display_name
        return self.plugin_id.partition("@")[0]


@dataclass(frozen=True, slots=True)
class _MarketplaceRow:
    name: str
    source: str
    plugin_count: int | None
    installed_count: int
    error: str | None = None

    @property
    def has_error(self) -> bool:
        """Whether the configured marketplace could not be loaded.

        Marketplace loading failures set `error` and produce an error status. Warnings
        from a marketplace that loaded successfully appear on the Errors tab without
        marking the marketplace itself as errored.
        """
        return self.error is not None


@dataclass(frozen=True, slots=True)
class _ManagerState:
    available_plugins: tuple[_PluginRow, ...]
    installed_plugins: tuple[_PluginRow, ...]
    marketplaces: tuple[_MarketplaceRow, ...]
    errors: tuple[str, ...]
