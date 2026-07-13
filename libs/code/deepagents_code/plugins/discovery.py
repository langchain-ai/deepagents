"""Plugin discovery, install, and enablement helpers."""

from __future__ import annotations

import logging
import shutil
from functools import partial
from pathlib import Path

from deepagents_code.plugins.manifest import (
    PluginManifestError,
    build_inventory,
    load_manifest,
)
from deepagents_code.plugins.marketplace import (
    MarketplaceError,
    load_marketplace,
    load_marketplace_location,
    materialize_marketplace_source,
    materialize_plugin_source,
    parse_marketplace_source,
)
from deepagents_code.plugins.models import (
    MarketplacePluginEntry,
    MarketplaceRecord,
    PluginDiscoveryResult,
    PluginInstance,
    PluginMarketplace,
    RepositoryMarketplaceSource,
    split_plugin_id,
)
from deepagents_code.plugins.store import (
    cache_and_register_plugin,
    ensure_marketplace_cache_dir,
    ensure_plugin_data_dir,
    get_primary_install_entry,
    load_enabled_plugin_ids,
    load_installed_plugins,
    load_marketplace_records,
    remove_marketplace_record,
    save_marketplace_record,
    set_plugin_enabled,
    uninstall_plugin as uninstall_plugin_record,
)

logger = logging.getLogger(__name__)


def add_local_marketplace(path: str | Path) -> PluginMarketplace:
    """Add a local marketplace to dcode state.

    Args:
        path: Marketplace root directory.

    Returns:
        Parsed marketplace.
    """
    marketplace = load_marketplace(Path(path))
    save_marketplace_record(
        MarketplaceRecord(
            name=marketplace.name,
            source_type="directory",
            source=str(marketplace.root),
            install_location=str(marketplace.root),
        )
    )
    return marketplace


def add_marketplace_source(raw: str) -> PluginMarketplace:
    """Add a marketplace from a pasted source string.

    Args:
        raw: GitHub shorthand, Git URL, marketplace JSON URL, file, or directory.

    Returns:
        Parsed marketplace.
    """
    source = parse_marketplace_source(raw)
    marketplace, location = materialize_marketplace_source(source)
    save_marketplace_record(
        MarketplaceRecord(
            name=marketplace.name,
            source_type=source.source_type,
            source=source.value,
            install_location=str(location),
            ref=source.ref if isinstance(source, RepositoryMarketplaceSource) else None,
        )
    )
    return marketplace


def remove_marketplace(name: str) -> bool:
    """Remove a marketplace and every plugin installed from it.

    Local marketplace source directories are never deleted. Managed marketplace
    clones and installed plugin caches are removed.

    Args:
        name: Marketplace name.

    Returns:
        `True` when a configured marketplace was removed.
    """
    record = load_marketplace_records().get(name)
    if record is None:
        return False

    plugin_ids = set(load_installed_plugins()) | set(load_enabled_plugin_ids())
    for plugin_id in plugin_ids:
        try:
            _plugin_name, marketplace_name = split_plugin_id(plugin_id)
        except ValueError:
            continue
        if marketplace_name == name:
            uninstall_plugin(plugin_id)

    removed = remove_marketplace_record(name)
    location = Path(record.install_location)
    try:
        resolved = location.resolve()
        cache_root = ensure_marketplace_cache_dir().resolve()
    except OSError:
        return removed
    if record.source_type in {"github", "git", "url"} and resolved.is_relative_to(
        cache_root
    ):
        if resolved.is_dir():
            shutil.rmtree(resolved, ignore_errors=True)
        elif resolved.is_file():
            resolved.unlink(missing_ok=True)
    return removed


def _require_installed_plugin(plugin_id: str) -> None:
    """Raise when `plugin_id` does not identify an installed plugin.

    Raises:
        MarketplaceError: If the plugin is not installed.
    """
    if plugin_id not in load_installed_plugins():
        msg = f"Plugin {plugin_id!r} is not installed"
        raise MarketplaceError(msg)


def set_installed_plugin_enabled(plugin_id: str, *, enabled: bool) -> None:
    """Set the enabled state of an installed plugin.

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.
        enabled: Whether to enable the plugin.
    """
    _require_installed_plugin(plugin_id)
    set_plugin_enabled(plugin_id, enabled)


def uninstall_plugin(plugin_id: str) -> None:
    """Uninstall a plugin (disable, clear records, delete orphaned cache).

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.
    """
    uninstall_plugin_record(plugin_id)


def _resolve_marketplace_and_entry(
    plugin_id: str,
) -> tuple[PluginMarketplace, MarketplacePluginEntry]:
    try:
        plugin_name, marketplace_name = split_plugin_id(plugin_id)
    except ValueError as exc:
        raise MarketplaceError(str(exc)) from exc
    records = load_marketplace_records()
    record = records.get(marketplace_name)
    if record is None:
        msg = f"Marketplace {marketplace_name!r} is not configured"
        raise MarketplaceError(msg)
    marketplace = load_marketplace_location(Path(record.install_location))
    entry = next(
        (plugin for plugin in marketplace.plugins if plugin.name == plugin_name),
        None,
    )
    if entry is None:
        msg = f"Plugin {plugin_id!r} not found in marketplace {marketplace_name}"
        raise MarketplaceError(msg)
    return marketplace, entry


def install_plugin(plugin_id: str) -> PluginInstance:
    """Install a marketplace plugin into the versioned cache and enable it.

    Copies the plugin source into `plugins/cache/{marketplace}/{plugin}/{version}/`,
    writes `installed_plugins.json`, and enables the plugin.

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.

    Returns:
        Discovered plugin instance loaded from the cache path.

    Raises:
        MarketplaceError: If the marketplace/plugin cannot be resolved, the
            source is unsupported, or the cached plugin fails to load.
    """
    marketplace, entry = _resolve_marketplace_and_entry(plugin_id)
    source_root = materialize_plugin_source(marketplace, entry)
    if source_root is None:
        msg = (
            f"Plugin {plugin_id} has unsupported source {entry.source!r}; "
            "use a local path, GitHub repository, or Git repository source"
        )
        raise MarketplaceError(msg)

    try:
        manifest, _manifest_path, manifest_warnings = load_manifest(
            source_root, fallback_name=entry.name
        )
    except PluginManifestError as exc:
        msg = f"Cannot install {plugin_id}: {exc}"
        raise MarketplaceError(msg) from exc

    for warning in manifest_warnings:
        logger.debug("Plugin install warning for %s: %s", plugin_id, warning)

    version = manifest.version if manifest is not None else None
    cache_path = cache_and_register_plugin(
        plugin_id,
        source_root,
        version=version,
        validate=partial(
            _validate_plugin_copy,
            plugin_id=plugin_id,
            fallback_name=entry.name,
        ),
    )

    set_plugin_enabled(plugin_id, True)

    instance, warnings = _plugin_from_install_path(
        plugin_id=plugin_id,
        root=cache_path,
        marketplace_name=marketplace.name,
        fallback_name=entry.name,
    )
    if instance is None:
        detail = "; ".join(warnings)
        uninstall_plugin_record(plugin_id)
        msg = f"Installed {plugin_id} but failed to load from cache: {detail}"
        raise MarketplaceError(msg)
    return instance


def _validate_plugin_copy(
    root: Path,
    *,
    plugin_id: str,
    fallback_name: str,
) -> None:
    try:
        manifest, _manifest_path, warnings = load_manifest(
            root, fallback_name=fallback_name
        )
    except PluginManifestError as exc:
        msg = f"Cannot install {plugin_id}: {exc}"
        raise MarketplaceError(msg) from exc
    build_inventory(root, manifest, warnings)


def _plugin_from_install_path(
    *,
    plugin_id: str,
    root: Path,
    marketplace_name: str,
    fallback_name: str,
) -> tuple[PluginInstance | None, tuple[str, ...]]:
    warnings: list[str] = []
    try:
        manifest, _manifest_path, manifest_warnings = load_manifest(
            root, fallback_name=fallback_name
        )
    except PluginManifestError as exc:
        return None, (f"Skipping plugin {plugin_id}: {exc}",)
    warnings.extend(manifest_warnings)
    name = manifest.name if manifest and manifest.name else fallback_name
    inventory = build_inventory(root, manifest, tuple(warnings))
    instance = PluginInstance(
        plugin_id=plugin_id,
        name=name,
        marketplace=marketplace_name,
        version=manifest.version if manifest is not None else None,
        root=root,
        data_dir=ensure_plugin_data_dir(plugin_id),
        manifest=manifest,
        inventory=inventory,
    )
    return instance, inventory.warnings


def discover_plugins() -> PluginDiscoveryResult:
    """Discover enabled marketplace plugins from their install cache paths.

    Returns:
        Discovery result. Broken marketplaces/plugins are returned as warnings and
        never abort sibling plugin loading.
    """
    enabled = load_enabled_plugin_ids()
    plugins: list[PluginInstance] = []
    warnings: list[str] = []

    for plugin_id in sorted(enabled):
        try:
            plugin_name, marketplace_name = split_plugin_id(plugin_id)
        except ValueError:
            warnings.append(f"Ignoring invalid plugin id {plugin_id!r}")
            continue
        entry = get_primary_install_entry(plugin_id)
        if entry is None:
            warnings.append(
                f"Plugin {plugin_id} is enabled but not installed "
                "(missing installed_plugins.json entry); run install to fix this"
            )
            continue
        root = Path(entry.install_path)
        if not root.is_dir():
            warnings.append(
                f"Plugin {plugin_id} cache miss at {entry.install_path}; "
                "re-run install to refresh"
            )
            continue
        plugin, plugin_warnings = _plugin_from_install_path(
            plugin_id=plugin_id,
            root=root,
            marketplace_name=marketplace_name,
            fallback_name=plugin_name,
        )
        warnings.extend(plugin_warnings)
        if plugin is not None:
            plugins.append(plugin)

    return PluginDiscoveryResult(plugins=tuple(plugins), warnings=tuple(warnings))


def list_available_plugins() -> tuple[tuple[str, str, bool], ...]:
    """List plugins from configured marketplaces.

    Returns:
        Tuples of `(plugin_id, description, enabled)`.
    """
    records = load_marketplace_records()
    enabled = load_enabled_plugin_ids()
    rows: list[tuple[str, str, bool]] = []
    for name, record in sorted(records.items()):
        try:
            marketplace = load_marketplace_location(Path(record.install_location))
        except MarketplaceError as exc:
            rows.append((f"<marketplace:{name}>", str(exc), False))
            continue
        for plugin in marketplace.plugins:
            plugin_id = f"{plugin.name}@{marketplace.name}"
            rows.append((plugin_id, plugin.description or "", plugin_id in enabled))
    return tuple(rows)


def list_installed_plugin_ids() -> frozenset[str]:
    """Return plugin ids that have install records.

    Returns:
        Set of installed plugin ids.
    """
    return frozenset(load_installed_plugins())
