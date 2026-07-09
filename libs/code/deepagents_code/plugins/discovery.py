"""Plugin discovery, install, and enablement helpers."""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import replace
from pathlib import Path

from deepagents_code.plugins.manifest import (
    PluginManifestError,
    build_inventory,
    load_manifest,
)
from deepagents_code.plugins.marketplace import (
    MarketplaceError,
    get_git_commit_sha,
    load_marketplace,
    load_marketplace_location,
    materialize_marketplace_source,
    parse_marketplace_source,
    redact_marketplace_source,
    resolve_plugin_source,
)
from deepagents_code.plugins.models import (
    InstalledPluginEntry,
    InstallScope,
    MarketplacePlugin,
    MarketplaceRecord,
    MarketplaceSource,
    PluginDiscoveryResult,
    PluginErrorCode,
    PluginInstance,
    PluginLoadError,
    PluginMarketplace,
)
from deepagents_code.plugins.store import (
    cache_and_register_plugin,
    get_primary_install_entry,
    load_enabled_plugins,
    load_favorite_plugins,
    load_installed_plugins,
    load_marketplace_records,
    load_plugin_scopes,
    marketplace_cache_dir,
    plugin_data_dir,
    remove_marketplace_record,
    save_marketplace_record,
    set_plugin_enabled,
    uninstall_plugin as uninstall_plugin_record,
)
from deepagents_code.plugins.trust import (
    is_plugin_surface_trusted,
    plugin_surface_fingerprint,
    revoke_plugin_trust,
    trust_plugin_surface,
)

logger = logging.getLogger(__name__)


def _current_project_path() -> str:
    from deepagents_code.project_utils import find_project_root

    cwd = Path.cwd().resolve()
    return str((find_project_root(cwd) or cwd).resolve())


def _project_path_for_scope(scope: InstallScope) -> str | None:
    if scope in {"project", "local"}:
        return _current_project_path()
    return None


def _invalidate_runtime_snapshot() -> None:
    from deepagents_code.plugins.runtime import clear_plugin_snapshot

    clear_plugin_snapshot()


def _install_entry_active(
    entry: InstalledPluginEntry, *, cwd: str | None = None
) -> bool:
    """Return whether a scoped install entry applies to the current project.

    User-scoped installs always apply. Project/local installs apply only when
    their recorded `project_path` matches the current working directory.
    """
    if entry.scope == "user":
        return True
    if entry.project_path is None:
        # Legacy records without a project path stay global to avoid stranding
        # installs created before scope filtering existed.
        return True
    current = cwd if cwd is not None else _current_project_path()
    try:
        return Path(entry.project_path).resolve() == Path(current).resolve()
    except OSError:
        return False


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
            source=redact_marketplace_source(source.value),
            install_location=str(location),
            ref=source.ref,
        )
    )
    return marketplace


def update_marketplace(name: str) -> PluginMarketplace:
    """Refresh a configured marketplace from its recorded source.

    Args:
        name: Marketplace name.

    Returns:
        Refreshed marketplace.

    Raises:
        MarketplaceError: If the marketplace is not configured or cannot load.
    """
    record = load_marketplace_records().get(name)
    if record is None:
        msg = f"Marketplace {name!r} is not configured"
        raise MarketplaceError(msg)
    source = MarketplaceSource(
        source_type=record.source_type,
        value=record.source,
        ref=record.ref,
    )
    marketplace, location = materialize_marketplace_source(source)
    save_marketplace_record(
        MarketplaceRecord(
            name=marketplace.name,
            source_type=source.source_type,
            source=record.source,
            install_location=str(location),
            ref=source.ref,
        )
    )
    return marketplace


def remove_marketplace(name: str) -> bool:
    """Remove a marketplace, its installs, and managed clone.

    Args:
        name: Marketplace name.

    Returns:
        Whether a configured marketplace was removed.
    """
    record = load_marketplace_records().get(name)
    if record is None:
        return False
    plugin_ids = (
        set(load_installed_plugins())
        | set(load_enabled_plugins())
        | set(load_plugin_scopes())
        | load_favorite_plugins()
    )
    for plugin_id in plugin_ids:
        if plugin_id.endswith(f"@{name}"):
            uninstall_plugin(plugin_id)
    removed = remove_marketplace_record(name)
    location = Path(record.install_location)
    try:
        resolved = location.resolve()
        cache_root = marketplace_cache_dir().resolve()
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


def enable_plugin(plugin_id: str) -> None:
    """Enable a plugin id (settings-only; does not materialize cache).

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.
    """
    set_plugin_enabled(plugin_id, True, scope="user")
    _invalidate_runtime_snapshot()


def enable_plugin_with_scope(plugin_id: str, scope: InstallScope) -> None:
    """Enable a plugin id with an install scope (settings-only).

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.
        scope: Install scope: `user`, `project`, or `local`.
    """
    set_plugin_enabled(
        plugin_id,
        True,
        scope=scope,
        project_path=_project_path_for_scope(scope),
    )
    _invalidate_runtime_snapshot()


def disable_plugin(plugin_id: str, *, scope: InstallScope | None = None) -> None:
    """Disable a plugin id (settings-only; keeps install cache).

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.
        scope: Scope to disable. Defaults to the most specific active scope.
    """
    active_scope = scope or load_plugin_scopes(
        project_path=_current_project_path()
    ).get(plugin_id, "user")
    set_plugin_enabled(
        plugin_id,
        False,
        scope=active_scope,
        project_path=_project_path_for_scope(active_scope),
    )
    _invalidate_runtime_snapshot()


def uninstall_plugin(plugin_id: str, *, scope: InstallScope | None = None) -> None:
    """Uninstall a plugin (disable, clear records, delete orphaned cache).

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.
        scope: Optional scope to uninstall; when omitted, remove all scopes.
    """
    uninstall_plugin_record(
        plugin_id,
        scope=scope,
        project_path=_project_path_for_scope(scope) if scope is not None else None,
    )
    if plugin_id not in load_installed_plugins():
        revoke_plugin_trust(plugin_id)
    _invalidate_runtime_snapshot()


def _resolve_marketplace_and_entry(
    plugin_id: str,
) -> tuple[PluginMarketplace, MarketplacePlugin]:
    if "@" not in plugin_id:
        msg = f"Invalid plugin id {plugin_id!r}; expected name@marketplace"
        raise MarketplaceError(msg)
    plugin_name, marketplace_name = plugin_id.rsplit("@", 1)
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


def install_plugin(
    plugin_id: str,
    *,
    scope: InstallScope = "user",
    trust: bool = False,
    force: bool = False,
) -> PluginInstance:
    """Install a marketplace plugin into the versioned cache and enable it.

    Copies the plugin source into `plugins/cache/{marketplace}/{plugin}/{version}/`,
    writes `installed_plugins.json`, and sets `enabledPlugins` (unless the
    manifest declares `defaultEnabled: false`).

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.
        scope: Install scope: `user`, `project`, or `local`.
        trust: Whether to trust this exact executable surface after installation.
        force: Whether to refresh an existing cache entry for the same version.

    Returns:
        Discovered plugin instance loaded from the cache path.

    Raises:
        MarketplaceError: If the marketplace/plugin cannot be resolved, the
            source is unsupported, or the cached plugin fails to load.
        OSError: If the generated manifest cannot be written.
    """
    marketplace, entry = _resolve_marketplace_and_entry(plugin_id)
    source_root = resolve_plugin_source(marketplace, entry)
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

    git_commit_sha = get_git_commit_sha(source_root)
    version = (
        manifest.version if manifest and manifest.version else entry.version
    ) or (git_commit_sha[:12] if git_commit_sha else "dev")
    cache_path = cache_and_register_plugin(
        plugin_id,
        source_root,
        scope=scope,
        version=version,
        git_commit_sha=git_commit_sha,
        project_path=_project_path_for_scope(scope),
        force=force,
    )
    if manifest is None and not entry.strict:
        generated = {"name": entry.name, **entry.manifest_fields}
        generated_path = cache_path / ".claude-plugin" / "plugin.json"
        try:
            generated_path.parent.mkdir(parents=True, exist_ok=True)
            generated_path.write_text(
                json.dumps(generated, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError:
            uninstall_plugin_record(
                plugin_id,
                scope=scope,
                project_path=_project_path_for_scope(scope),
            )
            raise
    elif manifest is not None and not entry.strict and entry.manifest_fields:
        logger.warning(
            "Plugin %s defines both a manifest and strict:false marketplace fields; "
            "using the plugin manifest",
            plugin_id,
        )

    should_enable = True
    if manifest is not None and not manifest.default_enabled:
        should_enable = False
    set_plugin_enabled(
        plugin_id,
        should_enable,
        scope=scope,
        project_path=_project_path_for_scope(scope),
    )

    instance, warnings = _plugin_from_install_path(
        plugin_id=plugin_id,
        root=cache_path,
        marketplace_name=marketplace.name,
        fallback_name=entry.name,
        version=version,
    )
    if instance is None:
        detail = "; ".join(warnings)
        msg = f"Installed {plugin_id} but failed to load from cache: {detail}"
        raise MarketplaceError(msg)
    if trust:
        fingerprint = plugin_surface_fingerprint(
            plugin_id=instance.plugin_id,
            version=instance.version,
            root=instance.root,
            manifest=instance.manifest,
            inventory=instance.inventory,
        )
        trust_plugin_surface(
            instance.plugin_id,
            version=instance.version,
            fingerprint=fingerprint,
        )
        instance = replace(instance, trusted=True)
    _invalidate_runtime_snapshot()
    return instance


def _plugin_from_install_path(
    *,
    plugin_id: str,
    root: Path,
    marketplace_name: str,
    fallback_name: str,
    version: str,
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
    resolved_version = (
        manifest.version if manifest and manifest.version else version
    ) or "dev"
    inventory = build_inventory(root, manifest, tuple(warnings))
    fingerprint = plugin_surface_fingerprint(
        plugin_id=plugin_id,
        version=resolved_version,
        root=root,
        manifest=manifest,
        inventory=inventory,
    )
    trusted = is_plugin_surface_trusted(
        plugin_id,
        version=resolved_version,
        fingerprint=fingerprint,
    )
    instance = PluginInstance(
        plugin_id=plugin_id,
        name=name,
        marketplace=marketplace_name,
        version=resolved_version,
        root=root,
        data_dir=plugin_data_dir(plugin_id),
        manifest=manifest,
        inventory=inventory,
        origin="marketplace",
        in_place=False,
        trusted=trusted,
    )
    return instance, inventory.warnings


def _plugin_from_dev_path(
    root: Path,
) -> tuple[PluginInstance | None, tuple[str, ...]]:
    resolved = root.expanduser().resolve()
    try:
        manifest, _manifest_path, manifest_warnings = load_manifest(
            resolved, fallback_name=resolved.name
        )
    except PluginManifestError as exc:
        return None, (f"Skipping session plugin {resolved}: {exc}",)
    name = manifest.name if manifest and manifest.name else resolved.name
    version = (manifest.version if manifest else None) or "dev"
    inventory = build_inventory(resolved, manifest, manifest_warnings)
    return (
        PluginInstance(
            plugin_id=f"{name}@inline",
            name=name,
            marketplace="inline",
            version=version,
            root=resolved,
            data_dir=plugin_data_dir(f"{name}@inline"),
            manifest=manifest,
            inventory=inventory,
            origin="dev-dir",
            in_place=True,
            trusted=True,
        ),
        inventory.warnings,
    )


def _session_plugin_dirs() -> tuple[Path, ...]:
    from deepagents_code._env_vars import PLUGIN_DIRS

    raw = os.environ.get(PLUGIN_DIRS, "")
    return tuple(
        Path(value).expanduser() for value in raw.split(os.pathsep) if value.strip()
    )


def trust_plugin(plugin_id: str) -> PluginInstance:
    """Trust the currently installed executable surface for a plugin.

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.

    Returns:
        Trusted plugin instance.

    Raises:
        MarketplaceError: If the plugin is invalid, missing, or cannot load.
    """
    if "@" not in plugin_id:
        msg = f"Invalid plugin id {plugin_id!r}; expected name@marketplace"
        raise MarketplaceError(msg)
    plugin_name, marketplace_name = plugin_id.rsplit("@", 1)
    entry = get_primary_install_entry(plugin_id, project_path=_current_project_path())
    if entry is None:
        msg = f"Plugin {plugin_id!r} is not installed for this project"
        raise MarketplaceError(msg)
    instance, warnings = _plugin_from_install_path(
        plugin_id=plugin_id,
        root=Path(entry.install_path),
        marketplace_name=marketplace_name,
        fallback_name=plugin_name,
        version=entry.version,
    )
    if instance is None:
        msg = f"Cannot trust {plugin_id}: {'; '.join(warnings)}"
        raise MarketplaceError(msg)
    fingerprint = plugin_surface_fingerprint(
        plugin_id=instance.plugin_id,
        version=instance.version,
        root=instance.root,
        manifest=instance.manifest,
        inventory=instance.inventory,
    )
    trust_plugin_surface(
        instance.plugin_id,
        version=instance.version,
        fingerprint=fingerprint,
    )
    _invalidate_runtime_snapshot()
    return replace(instance, trusted=True)


def get_plugin_info(plugin_id: str) -> PluginInstance:
    """Load metadata for an installed plugin.

    Args:
        plugin_id: Plugin id in `{name}@{marketplace}` form.

    Returns:
        Installed plugin instance.

    Raises:
        MarketplaceError: If the plugin is invalid, missing, or cannot load.
    """
    if "@" not in plugin_id:
        msg = f"Invalid plugin id {plugin_id!r}; expected name@marketplace"
        raise MarketplaceError(msg)
    plugin_name, marketplace_name = plugin_id.rsplit("@", 1)
    entry = get_primary_install_entry(plugin_id, project_path=_current_project_path())
    if entry is None:
        msg = f"Plugin {plugin_id!r} is not installed for this project"
        raise MarketplaceError(msg)
    instance, warnings = _plugin_from_install_path(
        plugin_id=plugin_id,
        root=Path(entry.install_path),
        marketplace_name=marketplace_name,
        fallback_name=plugin_name,
        version=entry.version,
    )
    if instance is None:
        msg = f"Cannot load {plugin_id}: {'; '.join(warnings)}"
        raise MarketplaceError(msg)
    return instance


def discover_plugins() -> PluginDiscoveryResult:
    """Discover enabled marketplace plugins from their install cache paths.

    Project- and local-scoped installs are only loaded when the current working
    directory matches the install's recorded project path. User-scoped installs
    always load.

    Returns:
        Discovery result. Broken marketplaces/plugins are returned as warnings and
        never abort sibling plugin loading.
    """
    cwd = _current_project_path()
    enabled = load_enabled_plugins(project_path=cwd)
    installed = load_installed_plugins()
    plugins: list[PluginInstance] = []
    warnings: list[str] = []
    errors: list[PluginLoadError] = []

    def record_error(
        code: PluginErrorCode, message: str, *, plugin_id: str | None = None
    ) -> None:
        warnings.append(message)
        errors.append(
            PluginLoadError(
                code=code,
                message=message,
                plugin_id=plugin_id,
            )
        )

    for plugin_id, is_enabled in sorted(enabled.items()):
        if not is_enabled:
            continue
        if "@" not in plugin_id:
            record_error(
                "invalid-plugin-id",
                f"Ignoring invalid plugin id {plugin_id!r}",
                plugin_id=plugin_id,
            )
            continue
        plugin_name, marketplace_name = plugin_id.rsplit("@", 1)
        entry = get_primary_install_entry(plugin_id, project_path=cwd)
        if entry is None:
            record_error(
                "not-installed",
                f"Plugin {plugin_id} is enabled but not installed "
                "(missing installed_plugins.json entry); run install to materialize",
                plugin_id=plugin_id,
            )
            continue
        if not _install_entry_active(entry, cwd=cwd):
            logger.debug(
                "Skipping %s-scoped plugin %s outside its project (%s)",
                entry.scope,
                plugin_id,
                entry.project_path,
            )
            continue
        root = Path(entry.install_path)
        if not root.is_dir():
            record_error(
                "cache-miss",
                f"Plugin {plugin_id} cache miss at {entry.install_path}; "
                "re-run install to refresh",
                plugin_id=plugin_id,
            )
            continue
        plugin, plugin_warnings = _plugin_from_install_path(
            plugin_id=plugin_id,
            root=root,
            marketplace_name=marketplace_name,
            fallback_name=plugin_name,
            version=entry.version,
        )
        warnings.extend(plugin_warnings)
        if plugin is not None:
            plugins.append(plugin)

    for root in _session_plugin_dirs():
        if not root.is_dir():
            record_error(
                "load-failed",
                f"Session plugin directory does not exist: {root}",
            )
            continue
        plugin, plugin_warnings = _plugin_from_dev_path(root)
        warnings.extend(plugin_warnings)
        if plugin is not None:
            plugins.append(plugin)

    # Surface orphaned install records that are not enabled (debug only).
    for plugin_id in installed:
        if plugin_id not in enabled or not enabled.get(plugin_id, False):
            logger.debug("Installed but disabled plugin: %s", plugin_id)

    by_name: dict[str, PluginInstance] = {}
    for plugin in plugins:
        previous = by_name.get(plugin.name)
        if previous is not None:
            record_error(
                "namespace-collision",
                f"Plugin namespace {plugin.name!r} is provided by both "
                f"{previous.plugin_id} and {plugin.plugin_id}; "
                f"using {plugin.plugin_id}",
                plugin_id=plugin.plugin_id,
            )
        by_name[plugin.name] = plugin
    active = tuple(by_name.values())
    active_ids = {plugin.plugin_id for plugin in active}
    active_names = {plugin.name for plugin in active}
    for plugin in active:
        if plugin.manifest is None:
            continue
        for dependency in plugin.manifest.dependencies:
            available = (
                dependency in active_ids
                if "@" in dependency
                else dependency in active_names
            )
            if not available:
                record_error(
                    "dependency-missing",
                    f"{plugin.plugin_id} declares dependency {dependency} "
                    "which is not enabled",
                    plugin_id=plugin.plugin_id,
                )

    return PluginDiscoveryResult(
        plugins=active, warnings=tuple(warnings), errors=tuple(errors)
    )


def list_available_plugins() -> tuple[tuple[str, str, bool], ...]:
    """List plugins from configured marketplaces.

    Returns:
        Tuples of `(plugin_id, description, enabled)`.
    """
    records = load_marketplace_records()
    enabled = load_enabled_plugins(project_path=_current_project_path())
    rows: list[tuple[str, str, bool]] = []
    for name, record in sorted(records.items()):
        try:
            marketplace = load_marketplace_location(Path(record.install_location))
        except MarketplaceError as exc:
            rows.append((f"<marketplace:{name}>", str(exc), False))
            continue
        for plugin in marketplace.plugins:
            plugin_id = f"{plugin.name}@{marketplace.name}"
            rows.append(
                (plugin_id, plugin.description or "", enabled.get(plugin_id, False))
            )
    return tuple(rows)


def list_installed_plugin_ids() -> frozenset[str]:
    """Return plugin ids that have install records.

    Returns:
        Set of installed plugin ids.
    """
    return frozenset(load_installed_plugins())
