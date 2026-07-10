"""Data models for plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

MarketplaceSourceType = Literal["directory", "file", "github", "git", "url"]
InstallScope = Literal["user", "project", "local"]
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalMarketplaceSource:
    """Local directory or JSON file used as a marketplace source."""

    source_type: Literal["directory", "file"]
    value: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RepositoryMarketplaceSource:
    """GitHub or Git repository used as a marketplace source.

    `ref` selects an optional branch or tag. Commit SHA checkout is not part of
    the shallow-clone flow.
    """

    source_type: Literal["github", "git"]
    value: str
    ref: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class UrlMarketplaceSource:
    """Marketplace manifest downloaded from an HTTP URL."""

    source_type: Literal["url"]
    value: str


MarketplaceSource = (
    LocalMarketplaceSource | RepositoryMarketplaceSource | UrlMarketplaceSource
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginManifest:
    """Parsed plugin manifest.

    Attributes:
        name: Plugin name from the manifest, or `None` for manifest-less plugins.
        version: Version string from the plugin manifest.
        description: Optional user-facing description.
        author: Optional author metadata.
        default_enabled: Whether install should enable the plugin by default.
        component_paths: Validated skill and MCP paths keyed by component name.
        inline_mcp: Inline MCP servers declared in the manifest.
    """

    name: str | None
    version: str | None
    description: str | None
    author: str | JsonObject | None
    default_enabled: bool
    component_paths: dict[str, tuple[Path, ...]]
    inline_mcp: JsonObject


@dataclass(frozen=True, slots=True, kw_only=True)
class ComponentInventory:
    """Inventory of supported plugin components."""

    skills: tuple[Path, ...] = ()
    mcp_files: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginInstance:
    """A discovered plugin ready to feed dcode adapters.

    Attributes:
        plugin_id: Stable id in `{name}@{marketplace}` form.
        name: Plugin namespace name.
        marketplace: Parent marketplace used for identity and namespacing.
        version: Version declared by the plugin manifest, if any.
        root: Plugin root directory.
        data_dir: Writable data directory for this plugin.
        manifest: Parsed manifest, if any.
        inventory: Component inventory.
        in_place: Whether the plugin is loaded directly from its source directory.
        trusted: Whether executable surfaces are trusted.
    """

    plugin_id: str
    name: str
    marketplace: str
    version: str | None
    root: Path
    data_dir: Path
    manifest: PluginManifest | None
    inventory: ComponentInventory
    in_place: bool
    trusted: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketplacePluginEntry:
    """A catalog entry from a marketplace manifest."""

    name: str
    source: str | JsonObject
    description: str | None = None
    manifest_fields: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginMarketplace:
    """A parsed marketplace manifest."""

    name: str
    root: Path
    manifest_path: Path
    owner: JsonObject | None
    metadata: JsonObject
    plugins: tuple[MarketplacePluginEntry, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketplaceRecord:
    """Persisted marketplace source record."""

    name: str
    source_type: MarketplaceSourceType
    source: str
    install_location: str
    ref: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class InstalledPluginEntry:
    """One scope's install record for a plugin.

    `version` is the value declared by the plugin manifest, if any.
    """

    scope: InstallScope
    install_path: str
    version: str | None
    installed_at: str
    last_updated: str
    project_path: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginDiscoveryResult:
    """Result from plugin discovery."""

    plugins: tuple[PluginInstance, ...]
    warnings: tuple[str, ...] = ()
