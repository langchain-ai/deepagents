"""State storage for dcode plugin marketplaces, installs, and enablement."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from deepagents_code.plugins.models import (
    InstalledPluginEntry,
    InstallScope,
    MarketplaceRecord,
    MarketplaceSourceType,
)

logger = logging.getLogger(__name__)
_STORAGE_VERSION = 1
_INSTALLED_STORAGE_VERSION = 2
_CACHE_SLUG_LENGTH = 48
_CACHE_DIGEST_LENGTH = 32
_INSTALL_SCOPES: frozenset[str] = frozenset({"user", "project", "local"})


def plugin_root_dir() -> Path:
    """Return the plugin storage root directory."""
    from deepagents_code._env_vars import PLUGIN_CACHE_DIR
    from deepagents_code.model_config import DEFAULT_CONFIG_DIR

    raw = os.environ.get(PLUGIN_CACHE_DIR)
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_CONFIG_DIR / "plugins"


def plugin_data_dir(plugin_id: str) -> Path:
    """Return the data directory path for a plugin id."""
    return plugin_root_dir() / "data" / sanitize_plugin_id(plugin_id)


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


def marketplace_cache_dir() -> Path:
    """Return the marketplace cache directory."""
    path = plugin_root_dir() / "marketplaces"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plugin_install_cache_dir() -> Path:
    """Return the versioned plugin install cache root."""
    path = plugin_root_dir() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def versioned_cache_path(plugin_id: str, version: str) -> Path:
    """Return the versioned cache path for a plugin id.

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.
        version: Plugin version string.

    Returns:
        Cache directory `cache/{marketplace}/{plugin}/{version}/`.

    Raises:
        ValueError: If `plugin_id` is not in `{name}@{marketplace}` form.
    """
    if "@" not in plugin_id:
        msg = f"Invalid plugin id {plugin_id!r}"
        raise ValueError(msg)
    plugin_name, marketplace = plugin_id.rsplit("@", 1)
    safe_version = sanitize_plugin_id(version) or "dev"
    return (
        plugin_install_cache_dir()
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


def _default_project_path() -> str:
    from deepagents_code.project_utils import find_project_root

    cwd = Path.cwd().resolve()
    return str((find_project_root(cwd) or cwd).resolve())


def _project_plugin_state_path(project_path: str) -> Path:
    return Path(project_path).resolve() / ".deepagents" / "plugins.json"


def _load_project_enabled(project_path: str) -> dict[str, bool]:
    data = _load_json(_project_plugin_state_path(project_path))
    enabled = data.get("enabledPlugins")
    if not isinstance(enabled, dict):
        return {}
    return {
        key: value
        for key, value in enabled.items()
        if isinstance(key, str) and isinstance(value, bool)
    }


def _write_project_enabled(project_path: str, enabled: dict[str, bool]) -> None:
    _atomic_write_json(
        _project_plugin_state_path(project_path),
        {"version": _STORAGE_VERSION, "enabledPlugins": enabled},
    )


def _load_local_enabled(data: dict[str, Any]) -> dict[str, dict[str, bool]]:
    raw = data.get("localEnabledPlugins")
    if not isinstance(raw, dict):
        return {}
    return {
        path: {
            plugin_id: value
            for plugin_id, value in values.items()
            if isinstance(plugin_id, str) and isinstance(value, bool)
        }
        for path, values in raw.items()
        if isinstance(path, str) and isinstance(values, dict)
    }


def _load_json(path: Path, *, max_version: int = _STORAGE_VERSION) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read plugin state file %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("Plugin state file %s is not a JSON object", path)
        return {}
    version = data.get("version")
    if version is not None and (not isinstance(version, int) or version > max_version):
        logger.warning("Plugin state file %s has unsupported version %r", path, version)
        return {}
    return data


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


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_marketplace_records() -> dict[str, MarketplaceRecord]:
    """Load persisted marketplace records.

    Returns:
        Marketplace records keyed by marketplace name.
    """
    data = _load_json(_marketplaces_path())
    raw_records = data.get("marketplaces", {})
    if not isinstance(raw_records, dict):
        return {}
    records: dict[str, MarketplaceRecord] = {}
    supported_types: set[MarketplaceSourceType] = {
        "directory",
        "file",
        "github",
        "git",
        "url",
    }
    for name, record in raw_records.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            continue
        source_type = record.get("source_type")
        source = record.get("source")
        if source_type == "local":
            source_type = "directory"
        if (
            source_type not in supported_types
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
    data = _load_json(_marketplaces_path())
    marketplaces = data.get("marketplaces")
    if not isinstance(marketplaces, dict):
        marketplaces = {}
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
    data = _load_json(_marketplaces_path())
    marketplaces = data.get("marketplaces")
    if not isinstance(marketplaces, dict) or name not in marketplaces:
        return False
    marketplaces.pop(name, None)
    _atomic_write_json(
        _marketplaces_path(),
        {"version": _STORAGE_VERSION, "marketplaces": marketplaces},
    )
    return True


def load_enabled_plugins(*, project_path: str | None = None) -> dict[str, bool]:
    """Load enabled plugin state.

    Returns:
        Enabled-state map keyed by plugin id.
    """
    data = _load_json(_plugin_state_path())
    enabled_raw = data.get("enabledPlugins", {})
    enabled = (
        {
            key: value
            for key, value in enabled_raw.items()
            if isinstance(key, str) and isinstance(value, bool)
        }
        if isinstance(enabled_raw, dict)
        else {}
    )
    active_project = project_path or _default_project_path()
    enabled.update(_load_project_enabled(active_project))
    project_key = str(Path(active_project).resolve())
    enabled.update(_load_local_enabled(data).get(project_key, {}))
    return enabled


def load_plugin_scopes(*, project_path: str | None = None) -> dict[str, InstallScope]:
    """Load plugin install scopes.

    Returns:
        Scope map keyed by plugin id.
    """
    data = _load_json(_plugin_state_path())
    enabled = data.get("enabledPlugins")
    scopes: dict[str, InstallScope] = (
        {key: "user" for key in enabled if isinstance(key, str)}
        if isinstance(enabled, dict)
        else {}
    )
    active_project = project_path or _default_project_path()
    scopes.update(dict.fromkeys(_load_project_enabled(active_project), "project"))
    local = _load_local_enabled(data).get(str(Path(active_project).resolve()), {})
    scopes.update(dict.fromkeys(local, "local"))
    return scopes


def load_favorite_plugins() -> set[str]:
    """Load favorited plugin ids and MCP server names.

    Returns:
        Set of favorited ids (plugin ids or MCP server names).
    """
    data = _load_json(_plugin_state_path())
    favorites = data.get("favoritePlugins", [])
    if not isinstance(favorites, list):
        return set()
    return {item for item in favorites if isinstance(item, str) and item}


def _write_plugin_state(
    *,
    enabled_plugins: dict[str, Any],
    scopes: dict[str, Any],
    favorites: set[str],
    local_enabled: dict[str, dict[str, bool]] | None = None,
) -> None:
    _atomic_write_json(
        _plugin_state_path(),
        {
            "version": _STORAGE_VERSION,
            "enabledPlugins": enabled_plugins,
            "pluginScopes": scopes,
            "favoritePlugins": sorted(favorites),
            "localEnabledPlugins": local_enabled or {},
        },
    )


def set_plugin_enabled(
    plugin_id: str,
    enabled: bool,
    *,
    scope: InstallScope | None = None,
    project_path: str | None = None,
) -> None:
    """Persist a plugin enablement value.

    Favorites are preserved across enable/disable so a favorited plugin can
    return to the Favorites group after re-enable.

    Raises:
        ValueError: If `scope` is invalid.
    """
    if scope is not None and scope not in _INSTALL_SCOPES:
        msg = f"Invalid plugin install scope: {scope!r}"
        raise ValueError(msg)
    data = _load_json(_plugin_state_path())
    enabled_plugins = data.get("enabledPlugins")
    if not isinstance(enabled_plugins, dict):
        enabled_plugins = {}
    resolved_scope = scope or "user"
    active_project = project_path or _default_project_path()
    local_enabled = _load_local_enabled(data)
    if resolved_scope == "user":
        enabled_plugins[plugin_id] = enabled
    elif resolved_scope == "project":
        project_enabled = _load_project_enabled(active_project)
        project_enabled[plugin_id] = enabled
        _write_project_enabled(active_project, project_enabled)
    else:
        project_key = str(Path(active_project).resolve())
        local_enabled.setdefault(project_key, {})[plugin_id] = enabled
    scopes = data.get("pluginScopes")
    if not isinstance(scopes, dict):
        scopes = {}
    scopes[plugin_id] = resolved_scope
    favorites_raw = data.get("favoritePlugins", [])
    favorites = (
        {item for item in favorites_raw if isinstance(item, str) and item}
        if isinstance(favorites_raw, list)
        else set()
    )
    _write_plugin_state(
        enabled_plugins=enabled_plugins,
        scopes=scopes,
        favorites=favorites,
        local_enabled=local_enabled,
    )


def set_plugin_favorite(item_id: str, favorite: bool) -> None:
    """Add or remove a plugin id or MCP server name from favorites.

    Args:
        item_id: Plugin id (`name@marketplace`) or MCP server name.
        favorite: Whether the item should be favorited.
    """
    data = _load_json(_plugin_state_path())
    enabled_plugins = data.get("enabledPlugins")
    if not isinstance(enabled_plugins, dict):
        enabled_plugins = {}
    scopes = data.get("pluginScopes")
    if not isinstance(scopes, dict):
        scopes = {}
    favorites_raw = data.get("favoritePlugins", [])
    local_enabled = _load_local_enabled(data)
    favorites = (
        {item for item in favorites_raw if isinstance(item, str) and item}
        if isinstance(favorites_raw, list)
        else set()
    )
    if favorite:
        favorites.add(item_id)
    else:
        favorites.discard(item_id)
    _write_plugin_state(
        enabled_plugins=enabled_plugins,
        scopes=scopes,
        favorites=favorites,
        local_enabled=local_enabled,
    )


def _parse_install_entry(raw: object) -> InstalledPluginEntry | None:
    if not isinstance(raw, dict):
        return None
    scope = raw.get("scope")
    install_path = raw.get("installPath") or raw.get("install_path")
    version = raw.get("version")
    installed_at = raw.get("installedAt") or raw.get("installed_at")
    last_updated = raw.get("lastUpdated") or raw.get("last_updated")
    if (
        scope not in {"user", "project", "local"}
        or not isinstance(install_path, str)
        or not install_path
        or not isinstance(version, str)
        or not version
        or not isinstance(installed_at, str)
        or not isinstance(last_updated, str)
    ):
        return None
    git_sha = raw.get("gitCommitSha") or raw.get("git_commit_sha")
    project_path = raw.get("projectPath") or raw.get("project_path")
    return InstalledPluginEntry(
        scope=cast("InstallScope", scope),
        install_path=install_path,
        version=version,
        installed_at=installed_at,
        last_updated=last_updated,
        git_commit_sha=git_sha if isinstance(git_sha, str) else None,
        project_path=project_path if isinstance(project_path, str) else None,
    )


def load_installed_plugins() -> dict[str, tuple[InstalledPluginEntry, ...]]:
    """Load installed plugin records.

    Returns:
        Map of plugin id to one or more scoped install entries.
    """
    data = _load_json(_installed_plugins_path(), max_version=_INSTALLED_STORAGE_VERSION)
    raw_plugins = data.get("plugins", {})
    if not isinstance(raw_plugins, dict):
        return {}
    result: dict[str, tuple[InstalledPluginEntry, ...]] = {}
    for plugin_id, entries in raw_plugins.items():
        if not isinstance(plugin_id, str) or not isinstance(entries, list):
            continue
        parsed = tuple(
            entry
            for entry in (_parse_install_entry(item) for item in entries)
            if entry is not None
        )
        if parsed:
            result[plugin_id] = parsed
    return result


def _entry_to_json(entry: InstalledPluginEntry) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "scope": entry.scope,
        "installPath": entry.install_path,
        "version": entry.version,
        "installedAt": entry.installed_at,
        "lastUpdated": entry.last_updated,
    }
    if entry.git_commit_sha:
        payload["gitCommitSha"] = entry.git_commit_sha
    if entry.project_path:
        payload["projectPath"] = entry.project_path
    return payload


def _write_installed_plugins(
    plugins: dict[str, tuple[InstalledPluginEntry, ...]],
) -> None:
    _atomic_write_json(
        _installed_plugins_path(),
        {
            "version": _INSTALLED_STORAGE_VERSION,
            "plugins": {
                plugin_id: [_entry_to_json(entry) for entry in entries]
                for plugin_id, entries in sorted(plugins.items())
            },
        },
    )


def add_installed_plugin(
    plugin_id: str,
    *,
    scope: InstallScope,
    install_path: str,
    version: str,
    git_commit_sha: str | None = None,
    project_path: str | None = None,
) -> InstalledPluginEntry:
    """Add or replace a scoped install record for a plugin.

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.
        scope: Install scope.
        install_path: Absolute path to the cached plugin root.
        version: Installed version string.
        git_commit_sha: Optional git SHA for the install.
        project_path: Optional project path for project/local scopes.

    Returns:
        The written install entry.

    Raises:
        ValueError: If a project/local install has no project path.
    """
    now = _utc_now()
    plugins = dict(load_installed_plugins())
    existing = list(plugins.get(plugin_id, ()))
    if scope != "user" and project_path is None:
        msg = f"{scope} installs require a project path"
        raise ValueError(msg)

    def same_target(candidate: InstalledPluginEntry) -> bool:
        if candidate.scope != scope:
            return False
        return scope == "user" or candidate.project_path == project_path

    kept = [entry for entry in existing if not same_target(entry)]
    prior = next((entry for entry in existing if same_target(entry)), None)
    entry = InstalledPluginEntry(
        scope=scope,
        install_path=install_path,
        version=version,
        installed_at=prior.installed_at if prior else now,
        last_updated=now,
        git_commit_sha=git_commit_sha,
        project_path=project_path,
    )
    kept.append(entry)
    plugins[plugin_id] = tuple(kept)
    _write_installed_plugins(plugins)
    return entry


def get_primary_install_entry(
    plugin_id: str, *, project_path: str | None = None
) -> InstalledPluginEntry | None:
    """Return the preferred install entry for a plugin id.

    Prefers `user`, then `project`, then `local`, then first entry.

    Args:
        plugin_id: Plugin id.
        project_path: When provided, only return project/local entries for this path.

    Returns:
        Install entry or `None`.
    """
    entries = load_installed_plugins().get(plugin_id)
    if not entries:
        return None
    for scope in ("user", "project", "local"):
        for entry in entries:
            if entry.scope != scope:
                continue
            if (
                entry.scope != "user"
                and project_path is not None
                and entry.project_path is not None
                and entry.project_path != project_path
            ):
                continue
            return entry
    return None if project_path is not None else entries[0]


def remove_installed_plugin(
    plugin_id: str,
    *,
    scope: InstallScope | None = None,
    project_path: str | None = None,
) -> tuple[InstalledPluginEntry, ...]:
    """Remove install record(s) for a plugin.

    Args:
        plugin_id: Plugin id.
        scope: When set, remove only that scope; otherwise remove all scopes.
        project_path: For project/local scope, remove only this project entry.

    Returns:
        Remaining install entries after removal (empty when fully uninstalled).
    """
    plugins = dict(load_installed_plugins())
    existing = list(plugins.get(plugin_id, ()))
    if not existing:
        return ()
    if scope is None:
        remaining: list[InstalledPluginEntry] = []
    else:
        remaining = [
            entry
            for entry in existing
            if entry.scope != scope
            or (
                scope != "user"
                and project_path is not None
                and entry.project_path != project_path
            )
        ]
    if remaining:
        plugins[plugin_id] = tuple(remaining)
    else:
        plugins.pop(plugin_id, None)
    _write_installed_plugins(plugins)
    return tuple(remaining)


def cache_and_register_plugin(
    plugin_id: str,
    source_dir: Path,
    *,
    scope: InstallScope = "user",
    version: str,
    git_commit_sha: str | None = None,
    project_path: str | None = None,
    force: bool = False,
) -> Path:
    """Copy a plugin into the versioned cache and register the install.

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.
        source_dir: Source plugin root to copy from.
        scope: Install scope.
        version: Version string used for the cache path.
        git_commit_sha: Optional git SHA.
        project_path: Optional project path for project/local scopes.
        force: Whether to replace an existing non-development cache entry.

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
    if cache_path.exists() and version != "dev" and not force:
        try:
            if any(cache_path.iterdir()):
                add_installed_plugin(
                    plugin_id,
                    scope=scope,
                    install_path=str(cache_path),
                    version=version,
                    git_commit_sha=git_commit_sha,
                    project_path=project_path,
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
        scope=scope,
        install_path=str(cache_path.resolve()),
        version=version,
        git_commit_sha=git_commit_sha,
        project_path=project_path,
    )
    return cache_path.resolve()


def uninstall_plugin(
    plugin_id: str,
    *,
    scope: InstallScope | None = None,
    project_path: str | None = None,
) -> None:
    """Disable a plugin, remove install records, and delete orphaned cache dirs.

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.
        scope: Optional scope to uninstall; when omitted, remove all scopes.
        project_path: For project/local scope, remove only this project entry.
    """
    existing = list(load_installed_plugins().get(plugin_id, ()))
    removed_paths = {
        entry.install_path
        for entry in existing
        if scope is None
        or (
            entry.scope == scope
            and (
                scope == "user"
                or project_path is None
                or entry.project_path == project_path
            )
        )
    }
    remaining = remove_installed_plugin(
        plugin_id, scope=scope, project_path=project_path
    )
    remaining_paths = {entry.install_path for entry in remaining}

    data = _load_json(_plugin_state_path())
    enabled_plugins = data.get("enabledPlugins")
    if not isinstance(enabled_plugins, dict):
        enabled_plugins = {}
    scopes = data.get("pluginScopes")
    if not isinstance(scopes, dict):
        scopes = {}
    favorites_raw = data.get("favoritePlugins", [])
    favorites = (
        {item for item in favorites_raw if isinstance(item, str) and item}
        if isinstance(favorites_raw, list)
        else set()
    )
    local_enabled = _load_local_enabled(data)

    removed_entries = [
        entry for entry in existing if entry.install_path in removed_paths
    ]
    for entry in removed_entries:
        if entry.scope == "user":
            enabled_plugins.pop(plugin_id, None)
        elif entry.scope == "project" and entry.project_path:
            project_enabled = _load_project_enabled(entry.project_path)
            project_enabled.pop(plugin_id, None)
            _write_project_enabled(entry.project_path, project_enabled)
        elif entry.scope == "local" and entry.project_path:
            project_key = str(Path(entry.project_path).resolve())
            local_enabled.get(project_key, {}).pop(plugin_id, None)

    if remaining:
        if scope is not None and scopes.get(plugin_id) == scope:
            scopes[plugin_id] = remaining[0].scope
    else:
        enabled_plugins.pop(plugin_id, None)
        if scope is None:
            active_project = project_path or _default_project_path()
            project_enabled = _load_project_enabled(active_project)
            if plugin_id in project_enabled:
                project_enabled.pop(plugin_id)
                _write_project_enabled(active_project, project_enabled)
            project_key = str(Path(active_project).resolve())
            local_enabled.get(project_key, {}).pop(plugin_id, None)
        scopes.pop(plugin_id, None)
        favorites.discard(plugin_id)

    _write_plugin_state(
        enabled_plugins=enabled_plugins,
        scopes=scopes,
        favorites=favorites,
        local_enabled=local_enabled,
    )

    for path_str in removed_paths - remaining_paths:
        path = Path(path_str)
        try:
            resolved = path.resolve()
            cache_root = plugin_install_cache_dir().resolve()
        except OSError:
            logger.warning(
                "Could not validate plugin cache path before deletion: %s", path
            )
            continue
        if not resolved.is_relative_to(cache_root):
            logger.warning("Refusing to delete plugin path outside cache: %s", path)
            continue
        if resolved.is_dir():
            shutil.rmtree(resolved, ignore_errors=True)
