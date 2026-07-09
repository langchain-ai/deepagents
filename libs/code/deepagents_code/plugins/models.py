"""Data models for plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pathlib import Path

PluginOrigin = Literal["dev-dir", "marketplace", "claude-install"]
MarketplaceSourceType = Literal["directory", "file", "github", "git", "url"]
InstallScope = Literal["user", "project", "local"]
PluginErrorCode = Literal[
    "cache-miss",
    "dependency-missing",
    "invalid-plugin-id",
    "load-failed",
    "namespace-collision",
    "not-installed",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketplaceSource:
    """Parsed source used to add or refresh a marketplace."""

    source_type: MarketplaceSourceType
    value: str
    ref: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginManifest:
    """Parsed plugin manifest.

    Attributes:
        name: Plugin name from the manifest, or `None` for manifest-less plugins.
        version: Optional plugin version.
        description: Optional user-facing description.
        author: Optional author metadata.
        display_name: Optional human-readable display name.
        default_enabled: Whether install should enable the plugin by default.
        dependencies: Normalized dependency references.
        component_paths: Validated component paths keyed by component name.
        inline_hooks: Inline hook objects declared in the manifest.
        inline_mcp: Inline MCP servers declared in the manifest.
        inline_commands: Inline command metadata records.
        raw: Original manifest dictionary.
    """

    name: str | None
    version: str | None
    description: str | None
    author: str | dict[str, Any] | None
    display_name: str | None
    default_enabled: bool
    dependencies: tuple[str, ...]
    component_paths: dict[str, tuple[Path, ...]]
    inline_hooks: tuple[dict[str, Any], ...]
    inline_mcp: dict[str, Any]
    inline_commands: dict[str, dict[str, Any]]
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class ComponentInventory:
    """Inventory of plugin components discovered from defaults and manifest paths."""

    skills: tuple[Path, ...] = ()
    commands: tuple[Path, ...] = ()
    agents: tuple[Path, ...] = ()
    hooks_files: tuple[Path, ...] = ()
    mcp_files: tuple[Path, ...] = ()
    unsupported: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginInstance:
    """A discovered plugin ready to feed dcode adapters.

    Attributes:
        plugin_id: Stable id in `{name}@{marketplace}` form.
        name: Plugin namespace name.
        marketplace: Marketplace name.
        version: Manifest version, git SHA, or `dev`.
        root: Plugin root directory.
        data_dir: Writable data directory for this plugin.
        manifest: Parsed manifest, if any.
        inventory: Component inventory.
        origin: Where this plugin came from.
        in_place: Whether the plugin is loaded directly from its source directory.
        trusted: Whether executable surfaces are trusted.
    """

    plugin_id: str
    name: str
    marketplace: str
    version: str
    root: Path
    data_dir: Path
    manifest: PluginManifest | None
    inventory: ComponentInventory
    origin: PluginOrigin
    in_place: bool
    trusted: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketplacePlugin:
    """A plugin entry from a marketplace manifest."""

    name: str
    source: str | dict[str, Any]
    description: str | None = None
    version: str | None = None
    category: str | None = None
    tags: tuple[str, ...] = ()
    strict: bool = True
    manifest_fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginMarketplace:
    """A parsed marketplace manifest."""

    name: str
    root: Path
    manifest_path: Path
    owner: dict[str, Any] | None
    metadata: dict[str, Any]
    plugins: tuple[MarketplacePlugin, ...]


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
    """One scope's install record for a plugin (Claude-like v2 shape)."""

    scope: InstallScope
    install_path: str
    version: str
    installed_at: str
    last_updated: str
    git_commit_sha: str | None = None
    project_path: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginLoadError:
    """Structured plugin load failure."""

    code: PluginErrorCode
    message: str
    plugin_id: str | None = None
    component: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginDiscoveryResult:
    """Result from plugin discovery."""

    plugins: tuple[PluginInstance, ...]
    warnings: tuple[str, ...] = ()
    errors: tuple[PluginLoadError, ...] = ()
