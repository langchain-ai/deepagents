"""Data models for plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

MarketplaceSourceType = Literal["directory", "file", "github", "git", "url"]
ExternalPluginRepositorySourceType = Literal["github", "git-subdir", "url"]
UnsupportedComponent = Literal["agents", "commands", "hooks"]
"""Plugin component directory that `deepagents-code` does not load."""
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
        component_paths: Validated skill and MCP paths keyed by component name.
        inline_mcp: Inline MCP servers declared in the manifest.
    """

    name: str | None
    version: str | None
    component_paths: dict[str, tuple[Path, ...]]
    inline_mcp: JsonObject


@dataclass(frozen=True, slots=True, kw_only=True)
class ComponentInventory:
    """Inventory of supported plugin components.

    `unsupported` lists plugin component directories that `deepagents-code` does
    not load (e.g. `agents/`, `commands/`, `hooks/`).
    """

    skills: tuple[Path, ...] = ()
    mcp_files: tuple[Path, ...] = ()
    unsupported: tuple[UnsupportedComponent, ...] = ()
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
    """

    plugin_id: str
    name: str
    marketplace: str
    version: str | None
    root: Path
    data_dir: Path
    manifest: PluginManifest | None
    inventory: ComponentInventory

    def __post_init__(self) -> None:
        """Validate the canonical plugin identity.

        Raises:
            ValueError: If `plugin_id` disagrees with `name` and `marketplace`.
        """
        expected = f"{self.name}@{self.marketplace}"
        if self.plugin_id != expected:
            msg = f"Plugin id {self.plugin_id!r} does not match {expected!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalPluginSource:
    """A plugin stored relative to its marketplace."""

    source_type: Literal["local"]
    path: str


@dataclass(frozen=True, slots=True, kw_only=True)
class GithubPluginSource:
    """A plugin sourced from a GitHub repository."""

    source_type: Literal["github"]
    repo: str
    ref: str | None = None
    path: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class GitSubdirectoryPluginSource:
    """A plugin sourced from a subdirectory in a Git repository."""

    source_type: Literal["git-subdir"]
    url: str
    ref: str | None = None
    path: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class UrlPluginSource:
    """A plugin sourced from a Git repository URL."""

    source_type: Literal["url"]
    url: str
    ref: str | None = None
    path: str | None = None


PluginSource = (
    LocalPluginSource
    | GithubPluginSource
    | GitSubdirectoryPluginSource
    | UrlPluginSource
)


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketplacePluginEntry:
    """A catalog entry from a marketplace manifest."""

    name: str
    source: PluginSource
    description: str | None = None
    author: str | JsonObject | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginMarketplace:
    """A parsed marketplace manifest."""

    name: str
    root: Path
    manifest_path: Path
    metadata: JsonObject
    plugins: tuple[MarketplacePluginEntry, ...]
    warnings: tuple[str, ...] = ()


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
    """Install record for a plugin.

    `version` is the value declared by the plugin manifest, if any.
    """

    install_path: str
    version: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginDiscoveryResult:
    """Result from plugin discovery."""

    plugins: tuple[PluginInstance, ...]
    warnings: tuple[str, ...] = ()


def split_plugin_id(plugin_id: str) -> tuple[str, str]:
    """Split a plugin id in `{plugin}@{marketplace}` form.

    Returns:
        Plugin and marketplace names.

    Raises:
        ValueError: If either part is missing.
    """
    if "@" not in plugin_id:
        msg = f"Invalid plugin id {plugin_id!r}; expected name@marketplace"
        raise ValueError(msg)
    plugin, marketplace = plugin_id.rsplit("@", 1)
    if not plugin or not marketplace:
        msg = f"Invalid plugin id {plugin_id!r}; expected name@marketplace"
        raise ValueError(msg)
    return plugin, marketplace
