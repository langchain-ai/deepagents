"""Process-local snapshots for managed and user TOML configuration."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from deepagents_code.configuration.paths import managed_config_path
from deepagents_code.configuration.providers import TomlFileProvider
from deepagents_code.configuration.resolver import merge_toml_tables
from deepagents_code.configuration.types import (
    ProviderHealth,
    ProviderStatus,
    TomlSnapshot,
)

UNION_PATHS = frozenset(
    {
        ("mcp", "disabled_project_servers"),
        ("mcp", "disabled_servers"),
    }
)
"""Paths whose lists accumulate instead of being replaced.

Deny lists must union across layers: replacing a managed deny list with a
user one would be a fail-open. Every merge path must read this one set, so a
third deny list cannot get union semantics in one place and replace in
another.
"""


def union_paths_under(prefix: tuple[str, ...]) -> frozenset[tuple[str, ...]]:
    """Return `UNION_PATHS` rebased onto a subtree rooted at `prefix`.

    `merge_toml_tables` matches paths relative to wherever it starts, so a
    merge of one option's own subtree never matches an absolute deny-list
    path. Passing `UNION_PATHS` there looks correct and silently does nothing,
    which would replace a nested deny list instead of unioning it.

    Returns:
        The union paths that fall under `prefix`, relative to it.
    """
    depth = len(prefix)
    return frozenset(path[depth:] for path in UNION_PATHS if path[:depth] == prefix)


@dataclass(frozen=True, slots=True)
class ConfigSources:
    """Managed and user TOML snapshots from one resolution generation."""

    managed: TomlSnapshot
    user: TomlSnapshot

    def merged(self) -> tuple[dict[str, Any], dict[str, str]]:
        """Return a deep merge where managed leaves outrank user leaves.

        Lists at `UNION_PATHS` accumulate instead of being replaced. A managed
        scalar replaces a colliding user table whatever its depth, but only
        when the value matches the type the manifest declares; a wrong-typed
        managed value leaves the user value in place.

        Returns:
            Merged table and dotted leaf-to-source mapping.
        """
        return merge_toml_tables(
            self.user.data,
            self.managed.data,
            lower_source="config.toml",
            higher_source="managed config",
            union_paths=UNION_PATHS,
            higher_leaf_is_valid=_is_valid_managed_scalar,
        )


def _is_valid_managed_scalar(path: tuple[str, ...], value: object) -> bool:
    """Return whether a managed scalar has the declared type for its path.

    Unknown and structured paths retain the deep-merger's existing behavior;
    their dedicated readers own validation. Manifest-backed scalar leaves are
    validated before they displace a lower-precedence value.
    """
    from deepagents_code.config_manifest import (
        _INVALID,
        OptionKind,
        _coerce_toml,
        get_config_options,
    )

    option = next(
        (
            candidate
            for candidate in get_config_options()
            if candidate.toml_keys == path
        ),
        None,
    )
    if option is None or option.kind is OptionKind.STRUCTURED:
        return True
    return _coerce_toml(option, value, source="managed config") is not _INVALID


class ManagedConfigError(RuntimeError):
    """Raised when an enforced managed source cannot be read safely."""

    def __init__(self, status: ProviderStatus) -> None:
        """Build a safe startup error from provider health metadata."""
        path = status.path or managed_config_path()
        detail = f": {status.detail}" if status.detail else ""
        super().__init__(
            f"Managed config at {path} is {status.health.value}{detail}. "
            "Ask your administrator to repair or remove the file."
        )
        self.status = status


class _SnapshotState:
    """Mutable process snapshot guarded by `_snapshot_lock`."""

    managed: TomlSnapshot | None = None


_snapshot_lock = threading.RLock()
_snapshot_state = _SnapshotState()


def _load_managed(path: Path | None = None) -> TomlSnapshot:
    """Load the managed provider without applying startup policy.

    Returns:
        Parsed managed snapshot and health.
    """
    resolved = managed_config_path() if path is None else path
    return TomlFileProvider("managed config", resolved).load()


def get_managed_snapshot(
    *, refresh: bool = False, path: Path | None = None
) -> TomlSnapshot:
    """Return the process snapshot, or an isolated snapshot for an explicit path.

    A reload that cannot parse the file never evicts policy that parsed
    cleanly earlier. An unusable snapshot carries `data == {}`, which every
    reader would otherwise treat as "nothing is enforced", so caching it would
    turn one broken write by an administrator into a process-wide fail-open.
    The caller still receives the failed load, so health checks see the error.

    Returns:
        The cached snapshot, or the freshly loaded one when refreshing.
    """
    if path is not None:
        return _load_managed(path)
    with _snapshot_lock:
        cached = _snapshot_state.managed
        if not refresh and cached is not None:
            return cached
        candidate = _load_managed()
        if candidate.status.usable:
            _snapshot_state.managed = candidate
        return candidate


def get_config_sources(
    *,
    user_path: Path | None = None,
    managed_path: Path | None = None,
    refresh_managed: bool = False,
    include_managed: bool = True,
) -> ConfigSources:
    """Load one user snapshot and the current managed snapshot.

    Returns:
        Both snapshots from one resolution generation.
    """
    if user_path is None:
        from deepagents_code.model_config import DEFAULT_CONFIG_PATH

        user_path = DEFAULT_CONFIG_PATH
    user = TomlFileProvider("config.toml", user_path).load()
    if include_managed:
        managed = get_managed_snapshot(
            refresh=refresh_managed,
            path=managed_path,
        )
    else:
        managed = TomlSnapshot(
            {},
            ProviderStatus(
                "managed config",
                managed_path,
                ProviderHealth.MISSING,
            ),
        )
    return ConfigSources(managed=managed, user=user)


def invalidate_config_sources() -> None:
    """Drop the cached managed snapshot.

    Test-only. Production reloads pass `refresh=True` instead, which keeps the
    last snapshot that parsed cleanly if the new one fails; clearing the cache
    first would leave readers with an empty managed table on a failed reload.
    """
    with _snapshot_lock:
        _snapshot_state.managed = None


def require_healthy_managed_config(*, refresh: bool = False) -> None:
    """Fail startup when a present managed policy cannot be parsed or read.

    Raises:
        ManagedConfigError: If managed policy is present but unusable.
    """
    status = get_managed_snapshot(refresh=refresh).status
    if status.health in {ProviderHealth.UNREADABLE, ProviderHealth.CORRUPT}:
        raise ManagedConfigError(status)


def managed_config_status(*, refresh: bool = False) -> ProviderStatus:
    """Return managed provider health for diagnostics and config inspection."""
    return get_managed_snapshot(refresh=refresh).status
