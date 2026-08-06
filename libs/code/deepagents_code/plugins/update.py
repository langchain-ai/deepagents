"""Background updates for installed marketplace plugins."""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import Timeout

from deepagents_code.plugins.discovery import (
    _plugin_from_install_path,
    _resolve_marketplace_and_entry,
    _validate_plugin_copy,
)
from deepagents_code.plugins.manifest import PluginManifestError, load_manifest
from deepagents_code.plugins.marketplace import (
    MarketplaceError,
    materialize_marketplace_source,
    materialize_plugin_source,
    redact_urls_in_text,
)
from deepagents_code.plugins.models import (
    MarketplaceRecord,
    RepositoryMarketplaceSource,
    UrlMarketplaceSource,
    split_plugin_id,
)
from deepagents_code.plugins.store import (
    cache_and_register_plugin,
    load_enabled_plugin_ids,
    load_installed_plugins,
    load_marketplace_records,
    plugin_mutation_lock,
    save_marketplace_record,
    versioned_cache_path,
)

if TYPE_CHECKING:
    from deepagents_code.plugins.models import (
        InstalledPluginEntry,
        MarketplaceSource,
        MarketplaceSourceType,
    )

logger = logging.getLogger(__name__)
_REMOTE_MARKETPLACE_SOURCE_TYPES: frozenset[MarketplaceSourceType] = frozenset(
    {"github", "git", "url"}
)


def is_plugin_auto_update_enabled() -> bool:
    """Return whether background plugin updates are enabled."""
    from deepagents_code._env_vars import OFFLINE, is_env_truthy
    from deepagents_code.config_manifest import (
        get_option,
        load_config_toml,
        resolve_scalar,
    )

    if is_env_truthy(OFFLINE):
        return False
    option = get_option("plugins.auto_update")
    if option is None:
        return False
    value, _source = resolve_scalar(option, toml_data=load_config_toml())
    return bool(value)


def _source_for_record(record: MarketplaceRecord) -> MarketplaceSource | None:
    if record.source_type == "github":
        return RepositoryMarketplaceSource(
            source_type="github", value=record.source, ref=record.ref
        )
    if record.source_type == "git":
        return RepositoryMarketplaceSource(
            source_type="git", value=record.source, ref=record.ref
        )
    if record.source_type == "url":
        return UrlMarketplaceSource(source_type="url", value=record.source)
    return None


def _refresh_marketplace(record: MarketplaceRecord) -> None:
    source = _source_for_record(record)
    if source is None:
        return
    marketplace, location = materialize_marketplace_source(source)
    if marketplace.name != record.name:
        msg = f"Marketplace {record.name!r} now declares the name {marketplace.name!r}"
        raise MarketplaceError(msg)
    save_marketplace_record(
        MarketplaceRecord(
            name=record.name,
            source_type=record.source_type,
            source=record.source,
            install_location=str(location),
            ref=record.ref,
        )
    )


def _validate_update_copy(
    root: Path,
    *,
    plugin_id: str,
    marketplace_name: str,
    fallback_name: str,
) -> None:
    _validate_plugin_copy(root, plugin_id=plugin_id, fallback_name=fallback_name)
    instance, warnings = _plugin_from_install_path(
        plugin_id=plugin_id,
        root=root,
        marketplace_name=marketplace_name,
        fallback_name=fallback_name,
    )
    if instance is None:
        detail = "; ".join(warnings) or "cached plugin could not be loaded"
        msg = f"Cannot update {plugin_id}: {detail}"
        raise MarketplaceError(msg)


def _update_plugin(plugin_id: str, installed: InstalledPluginEntry) -> bool:
    marketplace, entry = _resolve_marketplace_and_entry(plugin_id)
    source_root = materialize_plugin_source(marketplace, entry)
    if source_root is None:
        msg = f"Plugin {plugin_id} has an unsupported source"
        raise MarketplaceError(msg)
    try:
        manifest, _manifest_path, _warnings = load_manifest(
            source_root, fallback_name=entry.name
        )
    except PluginManifestError as exc:
        msg = f"Cannot update {plugin_id}: {exc}"
        raise MarketplaceError(msg) from exc

    version = manifest.version if manifest is not None else None
    if version is None or version == installed.version:
        return False

    existing_cache = versioned_cache_path(plugin_id, version)
    if existing_cache.is_dir():
        _validate_update_copy(
            existing_cache,
            plugin_id=plugin_id,
            marketplace_name=marketplace.name,
            fallback_name=entry.name,
        )

    cache_path = cache_and_register_plugin(
        plugin_id,
        source_root,
        version=version,
        validate=partial(
            _validate_update_copy,
            plugin_id=plugin_id,
            marketplace_name=marketplace.name,
            fallback_name=entry.name,
        ),
    )
    return cache_path.resolve() != Path(installed.install_path).resolve()


def _update_plugins() -> tuple[str, ...]:
    records = load_marketplace_records(strict=True)
    installed = load_installed_plugins(strict=True)
    enabled = load_enabled_plugin_ids(strict=True)
    updated: list[str] = []

    for marketplace_name, record in sorted(records.items()):
        if record.source_type not in _REMOTE_MARKETPLACE_SOURCE_TYPES:
            continue
        try:
            _refresh_marketplace(record)
        except (MarketplaceError, OSError, ValueError) as exc:
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
                _plugin_name, plugin_marketplace = split_plugin_id(plugin_id)
            except ValueError:
                continue
            if plugin_marketplace != marketplace_name:
                continue
            try:
                if _update_plugin(plugin_id, installed_entry):
                    updated.append(plugin_id)
            except (MarketplaceError, OSError, ValueError) as exc:
                logger.warning(
                    "Could not update plugin %s: %s",
                    plugin_id,
                    redact_urls_in_text(str(exc)),
                )

    return tuple(updated)


def auto_update_plugins() -> tuple[str, ...]:
    """Update enabled plugins when their manifest version string changes.

    Manifest versions are opaque publisher revisions. Unversioned plugins are
    skipped so the running session's shared `unversioned` cache is never replaced.

    Returns:
        Plugin ids whose installed cache path changed.
    """
    if not is_plugin_auto_update_enabled():
        return ()

    try:
        with plugin_mutation_lock(timeout=0):
            return _update_plugins()
    except Timeout:
        logger.debug(
            "Skipping plugin auto-update because another mutation holds the lock"
        )
        return ()
