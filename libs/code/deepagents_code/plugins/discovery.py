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
    redact_urls_in_text,
)
from deepagents_code.plugins.models import (
    MarketplacePluginEntry,
    MarketplaceRecord,
    PluginDiscoveryResult,
    PluginInstance,
    PluginMarketplace,
    RepositoryMarketplaceSource,
    UrlMarketplaceSource,
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
    plugin_data_dir,
    plugin_mutation_lock,
    remove_marketplace_record,
    save_marketplace_record,
    set_plugin_enabled,
    uninstall_plugin as uninstall_plugin_record,
)

logger = logging.getLogger(__name__)


@plugin_mutation_lock()
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


@plugin_mutation_lock()
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


@plugin_mutation_lock()
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

    installed = load_installed_plugins(strict=True)
    enabled = load_enabled_plugin_ids(strict=True)
    plugin_ids = set(installed) | set(enabled)
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
    if plugin_id not in load_installed_plugins(strict=True):
        msg = f"Plugin {plugin_id!r} is not installed"
        raise MarketplaceError(msg)


@plugin_mutation_lock()
def set_installed_plugin_enabled(plugin_id: str, *, enabled: bool) -> None:
    """Set the enabled state of an installed plugin.

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.
        enabled: Whether to enable the plugin.
    """
    _require_installed_plugin(plugin_id)
    set_plugin_enabled(plugin_id, enabled)
    if enabled:
        ensure_plugin_data_dir(plugin_id)


@plugin_mutation_lock()
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


@plugin_mutation_lock()
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
    load_installed_plugins(strict=True)
    load_enabled_plugin_ids(strict=True)
    marketplace, entry = _resolve_marketplace_and_entry(plugin_id)
    source_root = materialize_plugin_source(marketplace, entry)
    if source_root is None:
        msg = (
            f"Plugin {plugin_id} has unsupported source "
            f"{redact_urls_in_text(repr(entry.source))}; "
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
    ensure_plugin_data_dir(plugin_id)

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
    try:
        instance = PluginInstance(
            plugin_id=plugin_id,
            name=name,
            marketplace=marketplace_name,
            version=manifest.version if manifest is not None else None,
            root=root,
            data_dir=plugin_data_dir(plugin_id),
            manifest=manifest,
            inventory=inventory,
        )
    except ValueError as exc:
        return None, (f"Skipping plugin {plugin_id}: {exc}",)
    return instance, inventory.warnings


def plugin_auto_update_setting() -> tuple[bool, str]:
    """Resolve whether plugin auto-updates are enabled and from which source.

    Returns:
        The enabled state and its configuration source.
    """
    from deepagents_code.config_manifest import (
        _emit_ranked_diagnostics,
        _ranked_source,
        get_option,
    )
    from deepagents_code.configuration.resolver import get_config_resolver

    option = get_option("plugins.auto_update")
    if option is None:
        return True, "default"
    resolved = get_config_resolver().get(option)
    _emit_ranked_diagnostics(option, resolved)
    return bool(resolved.value), _ranked_source(resolved)


def auto_update_plugins() -> tuple[str, ...]:
    """Stage updated versions of enabled remote marketplace plugins.

    Unversioned plugins are skipped so the running session's shared cache is not
    replaced.

    Returns:
        Plugin ids whose installed cache path changed.
    """  # noqa: DOC501  # Marketplace errors are isolated per source/plugin.
    from filelock import Timeout

    from deepagents_code._env_vars import OFFLINE, is_env_truthy

    if is_env_truthy(OFFLINE) or not plugin_auto_update_setting()[0]:
        return ()

    try:
        with plugin_mutation_lock(timeout=0):
            records = load_marketplace_records(strict=True)
            installed = load_installed_plugins(strict=True)
            enabled = load_enabled_plugin_ids(strict=True)
            updated: list[str] = []

            for marketplace_name, record in sorted(records.items()):
                match record.source_type:
                    case "github" | "git":
                        source = RepositoryMarketplaceSource(
                            source_type=record.source_type,
                            value=record.source,
                            ref=record.ref,
                        )
                    case "url":
                        source = UrlMarketplaceSource(
                            source_type="url", value=record.source
                        )
                    case _:
                        continue

                try:
                    marketplace, _ = materialize_marketplace_source(source)
                    if marketplace.name != record.name:
                        msg = (
                            f"Marketplace {record.name!r} now declares the name "
                            f"{marketplace.name!r}"
                        )
                        raise MarketplaceError(msg)
                except (OSError, RuntimeError, ValueError) as exc:
                    logger.warning(
                        "Could not refresh plugin marketplace %s: %s",
                        marketplace_name,
                        redact_urls_in_text(str(exc)),
                    )
                    continue

                for plugin_id, installed_entry in sorted(installed.items()):
                    if plugin_id not in enabled or installed_entry.version is None:
                        continue
                    try:
                        plugin_name, plugin_marketplace = split_plugin_id(plugin_id)
                    except ValueError:
                        continue
                    if plugin_marketplace != marketplace_name:
                        continue

                    try:
                        entry = next(
                            (
                                plugin
                                for plugin in marketplace.plugins
                                if plugin.name == plugin_name
                            ),
                            None,
                        )
                        if entry is None:
                            msg = (
                                f"Plugin {plugin_id!r} not found in marketplace "
                                f"{marketplace_name}"
                            )
                            raise MarketplaceError(msg)
                        source_root = materialize_plugin_source(marketplace, entry)
                        if source_root is None:
                            msg = f"Plugin {plugin_id} has an unsupported source"
                            raise MarketplaceError(msg)
                        manifest, _manifest_path, _warnings = load_manifest(
                            source_root, fallback_name=entry.name
                        )
                        if (
                            manifest is None
                            or manifest.name != plugin_name
                            or not manifest.auto_update
                            or not manifest.version
                            or manifest.version == installed_entry.version
                        ):
                            continue

                        cache_and_register_plugin(
                            plugin_id,
                            source_root,
                            version=manifest.version,
                            validate=partial(
                                _validate_plugin_copy,
                                plugin_id=plugin_id,
                                fallback_name=entry.name,
                            ),
                        )
                        updated.append(plugin_id)
                    except (OSError, RuntimeError, ValueError) as exc:
                        logger.warning(
                            "Could not update plugin %s: %s",
                            plugin_id,
                            redact_urls_in_text(str(exc)),
                        )

            return tuple(updated)
    except Timeout:
        logger.debug(
            "Skipping plugin auto-update because another mutation holds the lock"
        )
        return ()


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
        try:
            root_exists = root.is_dir()
        except (OSError, RuntimeError) as exc:
            warnings.append(f"Plugin {plugin_id} cache could not be inspected: {exc}")
            continue
        if not root_exists:
            warnings.append(
                f"Plugin {plugin_id} cache miss at {entry.install_path}; "
                "re-run install to refresh"
            )
            continue
        try:
            plugin, plugin_warnings = _plugin_from_install_path(
                plugin_id=plugin_id,
                root=root,
                marketplace_name=marketplace_name,
                fallback_name=plugin_name,
            )
        except (OSError, RuntimeError) as exc:
            warnings.append(f"Skipping plugin {plugin_id}: {exc}")
            continue
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
