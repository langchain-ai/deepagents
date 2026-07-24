"""State storage for dcode plugin marketplaces, installs, and enablement."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Never

from deepagents_code.plugins.models import (
    InstalledPluginEntry,
    MarketplaceRecord,
    MarketplaceSourceType,
    split_plugin_id,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)
_STORAGE_VERSION = 1
_INSTALLED_STORAGE_VERSION = 2
_UNVERSIONED_CACHE_KEY = "unversioned"
_CACHE_SLUG_LENGTH = 48
_CACHE_DIGEST_LENGTH = 32
SUPPORTED_MARKETPLACE_SOURCE_TYPES: frozenset[MarketplaceSourceType] = frozenset(
    {"directory", "file", "github", "git", "url"}
)

DEFAULT_PLUGIN_DIRNAME = "plugins"
"""Default directory name for plugin storage under `~/.deepagents/`."""


class PluginStateError(OSError):
    """Raised when existing plugin state cannot be safely modified."""


def plugin_storage_root() -> Path:
    """Return the plugin storage root directory."""
    from deepagents_code._env_vars import PLUGIN_CACHE_DIR
    from deepagents_code.model_config import DEFAULT_CONFIG_DIR

    raw = os.environ.get(PLUGIN_CACHE_DIR)
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_CONFIG_DIR / DEFAULT_PLUGIN_DIRNAME


def plugin_data_dir(plugin_id: str) -> Path:
    """Return the data directory path for a plugin id without creating it.

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.

    Returns:
        Path under the plugin storage root's `data/` directory.
    """
    return plugin_storage_root() / "data" / sanitize_plugin_id(plugin_id)


def ensure_plugin_data_dir(plugin_id: str) -> Path:
    """Return the lazily-created data directory for a plugin id."""
    data_dir = plugin_data_dir(plugin_id)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def sanitize_plugin_id(value: str) -> str:
    """Return a bounded, collision-resistant filesystem key.

    Args:
        value: Identity string to encode.

    Returns:
        Filesystem-safe plugin id.
    """
    slug = "".join(
        ch if ch.isascii() and (ch.isalnum() or ch in {"_", "-"}) else "-"
        for ch in value
    )
    slug = slug.strip("-")[:_CACHE_SLUG_LENGTH] or "plugin"
    digest = sha256(value.encode()).hexdigest()[:_CACHE_DIGEST_LENGTH]
    return f"{slug}-{digest}"


def opaque_cache_key(value: str) -> str:
    """Return a cache key that cannot disclose source credentials."""
    return sha256(value.encode()).hexdigest()


def ensure_marketplace_cache_dir() -> Path:
    """Return the marketplace cache directory."""
    path = plugin_storage_root() / "marketplaces"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_plugin_install_cache_dir() -> Path:
    """Return the versioned plugin install cache root."""
    path = plugin_storage_root() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def versioned_cache_path(plugin_id: str, version: str | None) -> Path:
    """Return the versioned cache path for a plugin id.

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.
        version: Plugin version string, or `None` when unversioned.

    Returns:
        Cache directory `cache/{marketplace}/{plugin}/{version}/`, relative to
        the plugin storage root.

    """
    plugin_name, marketplace = split_plugin_id(plugin_id)
    safe_version = sanitize_plugin_id(version or _UNVERSIONED_CACHE_KEY)
    return (
        ensure_plugin_install_cache_dir()
        / sanitize_plugin_id(marketplace)
        / sanitize_plugin_id(plugin_name)
        / safe_version
    )


def _state_dir() -> Path:
    from deepagents_code.model_config import DEFAULT_STATE_DIR

    return DEFAULT_STATE_DIR


def _marketplaces_path() -> Path:
    return _state_dir() / "plugin_marketplaces.json"


def _plugin_state_path() -> Path:
    return _state_dir() / "plugin_state.json"


def _installed_plugins_path() -> Path:
    return _state_dir() / "installed_plugins.json"


def _invalid_state(
    path: Path, detail: str, *, strict: bool, cause: Exception | None = None
) -> dict[str, Any]:
    msg = f"Plugin state file {path} {detail}"
    if strict:
        raise PluginStateError(msg) from cause
    logger.warning("%s", msg)
    return {}


def _load_json(
    path: Path,
    *,
    max_version: int = _STORAGE_VERSION,
    strict: bool = False,
) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _invalid_state(
            path, f"could not be read: {exc}", strict=strict, cause=exc
        )
    if not isinstance(data, dict):
        return _invalid_state(path, "is not a JSON object", strict=strict)
    version = data.get("version")
    if version is not None and (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version > max_version
    ):
        return _invalid_state(
            path, f"has unsupported version {version!r}", strict=strict
        )
    return data


def _raise_state_shape(path: Path, detail: str) -> Never:
    msg = f"Plugin state file {path} {detail}"
    raise PluginStateError(msg)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        Path(tmp_name).replace(path)
    except Exception:
        with suppress(OSError):
            Path(tmp_name).unlink()
        raise


def load_marketplace_records(*, strict: bool = False) -> dict[str, MarketplaceRecord]:
    """Load persisted marketplace records.

    Returns:
        Marketplace records keyed by marketplace name.
    """
    data = _load_json(_marketplaces_path(), strict=strict)
    raw_records = data.get("marketplaces", {})
    if not isinstance(raw_records, dict):
        if strict:
            _raise_state_shape(_marketplaces_path(), "has invalid marketplaces data")
        return {}
    records: dict[str, MarketplaceRecord] = {}
    for name, record in raw_records.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            continue
        source_type = record.get("source_type")
        source = record.get("source")
        if (
            source_type not in SUPPORTED_MARKETPLACE_SOURCE_TYPES
            or not isinstance(source, str)
            or not isinstance(record.get("install_location", source), str)
        ):
            logger.debug("Skipping unsupported marketplace record %r", name)
            continue
        ref = record.get("ref")
        records[name] = MarketplaceRecord(
            name=name,
            source_type=source_type,
            source=source,
            install_location=record.get("install_location", source),
            ref=ref if isinstance(ref, str) else None,
        )
    return records


def save_marketplace_record(record: MarketplaceRecord) -> None:
    """Persist a marketplace record."""
    data = _load_json(_marketplaces_path(), strict=True)
    marketplaces = data.get("marketplaces")
    if marketplaces is None:
        marketplaces = {}
    elif not isinstance(marketplaces, dict):
        _raise_state_shape(_marketplaces_path(), "has invalid marketplaces data")
    marketplaces[record.name] = {
        "install_location": record.install_location,
        "source_type": record.source_type,
        "source": record.source,
    }
    if record.ref:
        marketplaces[record.name]["ref"] = record.ref
    _atomic_write_json(
        _marketplaces_path(),
        {"version": _STORAGE_VERSION, "marketplaces": marketplaces},
    )


def remove_marketplace_record(name: str) -> bool:
    """Remove a marketplace record.

    Returns:
        `True` when a record was removed.
    """
    data = _load_json(_marketplaces_path(), strict=True)
    marketplaces = data.get("marketplaces")
    if marketplaces is None:
        return False
    if not isinstance(marketplaces, dict):
        _raise_state_shape(_marketplaces_path(), "has invalid marketplaces data")
    if name not in marketplaces:
        return False
    marketplaces.pop(name, None)
    _atomic_write_json(
        _marketplaces_path(),
        {"version": _STORAGE_VERSION, "marketplaces": marketplaces},
    )
    return True


def load_enabled_plugin_ids(*, strict: bool = False) -> frozenset[str]:
    """Load enabled plugin ids.

    Returns:
        Enabled plugin ids.
    """
    data = _load_json(_plugin_state_path(), strict=strict)
    enabled = data.get("enabledPlugins", {})
    if not isinstance(enabled, dict):
        if strict:
            _raise_state_shape(_plugin_state_path(), "has invalid enabledPlugins data")
        return frozenset()
    if strict and any(
        not isinstance(key, str) or not isinstance(value, bool)
        for key, value in enabled.items()
    ):
        _raise_state_shape(_plugin_state_path(), "has malformed enabledPlugins entries")
    return frozenset(
        key for key, value in enabled.items() if isinstance(key, str) and value is True
    )


def _write_plugin_state(*, enabled_plugin_ids: set[str]) -> None:
    _atomic_write_json(
        _plugin_state_path(),
        {
            "version": _STORAGE_VERSION,
            "enabledPlugins": dict.fromkeys(sorted(enabled_plugin_ids), True),
        },
    )


def set_plugin_enabled(plugin_id: str, enabled: bool) -> None:
    """Persist a plugin enablement value."""
    enabled_plugin_ids = set(load_enabled_plugin_ids(strict=True))
    if enabled:
        enabled_plugin_ids.add(plugin_id)
    else:
        enabled_plugin_ids.discard(plugin_id)
    _write_plugin_state(enabled_plugin_ids=enabled_plugin_ids)


def _parse_installed_plugin_json_entry(
    persisted_entry: object,
) -> InstalledPluginEntry | None:
    if not isinstance(persisted_entry, dict):
        return None
    install_path = persisted_entry.get("installPath") or persisted_entry.get(
        "install_path"
    )
    version = persisted_entry.get("version")
    if (
        not isinstance(install_path, str)
        or not install_path
        or (version is not None and (not isinstance(version, str) or not version))
    ):
        return None
    return InstalledPluginEntry(
        install_path=install_path,
        version=version if isinstance(version, str) else None,
    )


def load_installed_plugins(*, strict: bool = False) -> dict[str, InstalledPluginEntry]:
    """Load installed plugin records.

    Returns:
        Map of plugin id to its install entry.
    """
    data = _load_json(
        _installed_plugins_path(),
        max_version=_INSTALLED_STORAGE_VERSION,
        strict=strict,
    )
    raw_plugins = data.get("plugins", {})
    if not isinstance(raw_plugins, dict):
        if strict:
            _raise_state_shape(_installed_plugins_path(), "has invalid plugins data")
        return {}
    result: dict[str, InstalledPluginEntry] = {}
    for plugin_id, entries in raw_plugins.items():
        if not isinstance(plugin_id, str) or not isinstance(entries, list):
            if strict:
                _raise_state_shape(
                    _installed_plugins_path(), "has malformed plugin entries"
                )
            continue
        parsed = next(
            (
                entry
                for item in entries
                if (entry := _parse_installed_plugin_json_entry(item))
            ),
            None,
        )
        if parsed is not None:
            result[plugin_id] = parsed
        elif strict:
            _raise_state_shape(
                _installed_plugins_path(), f"has malformed entry for {plugin_id!r}"
            )
    return result


def _entry_to_json(entry: InstalledPluginEntry) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "installPath": entry.install_path,
    }
    if entry.version is not None:
        payload["version"] = entry.version
    return payload


def _write_installed_plugins(
    plugins: dict[str, InstalledPluginEntry],
) -> None:
    _atomic_write_json(
        _installed_plugins_path(),
        {
            "version": _INSTALLED_STORAGE_VERSION,
            "plugins": {
                plugin_id: [_entry_to_json(entry)]
                for plugin_id, entry in sorted(plugins.items())
            },
        },
    )


def add_installed_plugin(
    plugin_id: str,
    *,
    install_path: str,
    version: str | None,
) -> InstalledPluginEntry:
    """Add or replace the record for `plugin_id` in `installed_plugins.json`.

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.
        install_path: Absolute path to the cached plugin root.
        version: Version declared by the plugin manifest, if any.

    Returns:
        The written install entry.

    """
    plugins = dict(load_installed_plugins(strict=True))
    entry = InstalledPluginEntry(
        install_path=install_path,
        version=version,
    )
    plugins[plugin_id] = entry
    _write_installed_plugins(plugins)
    return entry


def get_primary_install_entry(plugin_id: str) -> InstalledPluginEntry | None:
    """Return the install entry for a plugin id."""
    return load_installed_plugins().get(plugin_id)


def remove_installed_plugin(
    plugin_id: str,
) -> InstalledPluginEntry | None:
    """Remove the install record for a plugin.

    Args:
        plugin_id: Plugin id.

    Returns:
        Removed install entry, if present.
    """
    plugins = dict(load_installed_plugins(strict=True))
    removed = plugins.pop(plugin_id, None)
    _write_installed_plugins(plugins)
    return removed


def cache_and_register_plugin(
    plugin_id: str,
    source_dir: Path,
    *,
    version: str | None,
    validate: Callable[[Path], None] | None = None,
) -> Path:
    """Copy a plugin into the versioned cache and register the install.

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.
        source_dir: Source plugin root to copy from.
        version: Version declared by the plugin manifest, if any.
        validate: Optional validation to run against the temporary copy before
            replacing an existing cache.

    Returns:
        Absolute path to the cached plugin root.

    Raises:
        FileNotFoundError: If `source_dir` is not an existing directory.
        OSError: If the cache cannot be copied or atomically replaced.
    """
    source = source_dir.resolve()
    if not source.is_dir():
        msg = f"Plugin source directory not found: {source}"
        raise FileNotFoundError(msg)

    cache_path = versioned_cache_path(plugin_id, version)
    if cache_path.exists() and version is not None:
        try:
            if any(cache_path.iterdir()):
                add_installed_plugin(
                    plugin_id,
                    install_path=str(cache_path),
                    version=version,
                )
                return cache_path
        except OSError:
            pass

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = cache_path.parent / f".{cache_path.name}.tmp-{os.getpid()}"
    backup_dir = cache_path.parent / f".{cache_path.name}.backup-{os.getpid()}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)
    try:
        shutil.copytree(source, temp_dir, symlinks=True, dirs_exist_ok=False)
        git_dir = temp_dir / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir, ignore_errors=True)
        if validate is not None:
            validate(temp_dir)
        if cache_path.exists():
            cache_path.replace(backup_dir)
        try:
            temp_dir.replace(cache_path)
        except OSError:
            if backup_dir.exists() and not cache_path.exists():
                backup_dir.replace(cache_path)
            raise
        shutil.rmtree(backup_dir, ignore_errors=True)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    add_installed_plugin(
        plugin_id,
        install_path=str(cache_path.resolve()),
        version=version,
    )
    return cache_path.resolve()


def uninstall_plugin(
    plugin_id: str,
) -> None:
    """Disable a plugin, remove install records, and delete orphaned cache dirs.

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.
    """
    load_installed_plugins(strict=True)
    load_enabled_plugin_ids(strict=True)
    removed = remove_installed_plugin(plugin_id)

    set_plugin_enabled(plugin_id, False)

    if removed is not None:
        path = Path(removed.install_path)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
