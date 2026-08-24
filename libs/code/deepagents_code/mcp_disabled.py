"""Persistent store of MCP server names the user has disabled.

Disabled servers are skipped at config merge time so their tools never
reach the agent and no connection is attempted. State lives under
`[mcp].disabled_servers` in `~/.deepagents/config.toml`, alongside the
user's other MCP configuration.

The store keys on server *name* alone. Two configs that both declare a
`github` server will both be disabled by a single entry — intentional,
since the agent cannot distinguish overlapping names at runtime anyway
(later configs in the merge order win).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from deepagents_code.configuration.resolver import (
        RankedProviderValue,
        ResolvedValue,
    )
    from deepagents_code.configuration.types import ProviderResult, ProviderStatus

from deepagents_code.model_config import DEFAULT_CONFIG_PATH as _DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)

_SECTION = "mcp"
_KEY = "disabled_servers"
_LEGACY_SECTION = "mcp_disabled"
_LEGACY_KEY = "servers"


class _ManagedDenyListError(Exception):
    """Raised when a managed deny list is present but cannot be read as names.

    `_coerce_entries` reports a wrong-typed value as "key absent", which lets
    the lookup fall through to an empty set. For the *managed* source that is a
    fail-open: an administrator typo turns a deny list into no denials at all.
    """


class _ConfigLoadError(Exception):
    """Raised when the config exists but cannot be parsed or read.

    Distinct from "file does not exist" so callers can refuse to
    overwrite a config they could not parse — otherwise a transient
    read error or a hand-edit typo would silently truncate sibling
    sections (e.g. model profiles) on the next write.
    """


def _load_config(config_path: Path) -> dict[str, Any]:
    """Read the TOML config file.

    Args:
        config_path: Path to the TOML config file.

    Returns:
        Parsed TOML data, or an empty dict if the file does not exist.

    Raises:
        _ConfigLoadError: If the file exists but cannot be read or parsed.
    """
    import tomllib

    if not config_path.exists():
        return {}
    try:
        with config_path.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning(
            "Could not read MCP disabled config at %s: %s",
            config_path,
            exc,
        )
        msg = f"could not load {config_path}: {exc}"
        raise _ConfigLoadError(msg) from exc


def _save_disabled_entry(
    server_name: str, disabled: bool, config_path: Path
) -> str | None:
    """Add or remove one server name through the shared writer.

    The entry set is recomputed from the parse the writer performs inside its
    lock, never from a snapshot read before the lock was taken. Writing a
    pre-lock set would drop a concurrent disable of a *different* server, and
    editing in place keeps a concurrent `[ui]` or `[models]` write intact.

    Returns:
        `None` on success, or the writer's reason for the failure. The reason is
            returned rather than a bare `False` so the caller can show why the
            write failed: `WriteResult.error` carries the path and the errno,
            and "could not write <path>" alone dropped "Permission denied", "No
            space left on device", and the missing-`tomli_w` case.
    """
    from deepagents_code.configuration.writer import update_user_config

    def apply(current: dict[str, Any]) -> bool:
        section = current.get(_SECTION)
        if not isinstance(section, dict):
            section = {}
        before = (section.get(_KEY), _LEGACY_SECTION in current)
        entries = _disabled_entries(current)
        if disabled:
            entries.add(server_name)
        else:
            entries.discard(server_name)
        section[_KEY] = sorted(entries)
        current[_SECTION] = section
        _remove_legacy_disabled_section(current)
        return (section[_KEY], _LEGACY_SECTION in current) != before

    result = update_user_config(apply, config_path=config_path)
    if not result.ok:
        logger.error("Failed to save config to %s: %s", config_path, result.error)
        return result.error or f"could not write {config_path}"
    return None


def _coerce_entries(entries: object) -> set[str] | None:
    """Return valid server names from a TOML value, or `None` when unset."""
    if not isinstance(entries, list):
        return None
    return {name for name in entries if isinstance(name, str) and name}


def _strict_entries(value: object, *, section: str, key: str) -> set[str]:
    """Read one deny-list value, refusing a shape that cannot hold names.

    Mirrors `model_config._toml_str_list`: a bare string is split on commas, so
    the spelling `disabled_servers = "a, b"` yields two names instead of one
    bogus token, and non-string list elements are dropped with a log while the
    valid names survive. Any other type cannot be read as names at all.

    Args:
        value: The raw value read from the deny-list section.
        section: The section name, for log and error context.
        key: The deny-list key name, for log and error context.

    Returns:
        The trimmed, non-empty server names.

    Raises:
        _ManagedDenyListError: If `value` is neither a string nor a list.
    """
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    if not isinstance(value, list):
        msg = (
            f"[{section}].{key} must be an array of server names, "
            f"got {type(value).__name__}"
        )
        raise _ManagedDenyListError(msg)
    names = {item.strip() for item in value if isinstance(item, str) and item.strip()}
    discarded = sum(
        1 for item in value if not isinstance(item, str) or not item.strip()
    )
    if discarded > 0:
        logger.warning(
            "Dropped %d unusable entry/entries from managed [%s].%s",
            discarded,
            section,
            key,
        )
    return names


def _user_entries_result(data: Mapping[str, Any]) -> ProviderResult[list[str]]:
    """Coerce the user tier while preserving its legacy-section fallback.

    A wrong-typed folded value is intentionally `Unset` at that spelling, so
    the legacy list remains eligible. Managed policy has a stricter coercer:
    the same shape is `Invalid` because silently dropping a deny is fail-open.

    Returns:
        A typed provider result for the user file.
    """
    from deepagents_code.configuration.types import Found, Unset

    for section_name, key in ((_SECTION, _KEY), (_LEGACY_SECTION, _LEGACY_KEY)):
        section = data.get(section_name)
        if not isinstance(section, dict):
            continue
        entries = _coerce_entries(section.get(key))
        if entries is not None:
            return Found(sorted(entries))
    return Unset()


def _managed_entries_result(data: Mapping[str, Any]) -> ProviderResult[list[str]]:
    """Coerce the managed tier into names or a fail-closed rejection.

    Returns:
        A typed provider result for managed policy.
    """
    from deepagents_code.configuration.types import Found, Invalid, Unset

    for section_name, key in ((_SECTION, _KEY), (_LEGACY_SECTION, _LEGACY_KEY)):
        section = data.get(section_name)
        if section is None:
            continue
        if not isinstance(section, dict):
            return Invalid(
                f"[{section_name}] must be a table, got {type(section).__name__}"
            )
        if key not in section:
            continue
        try:
            entries = _strict_entries(section[key], section=section_name, key=key)
            return Found(sorted(entries))
        except _ManagedDenyListError as exc:
            return Invalid(str(exc))
    return Unset()


def _ranked_entries(
    data: Mapping[str, Any],
    *,
    rank: int,
    durable: bool,
    status: ProviderStatus,
    managed: bool,
) -> RankedProviderValue[list[str]]:
    """Build one already-coerced deny-list provider tier.

    Returns:
        The rank, status, and typed provider result consumed by the pure resolver.
    """
    from deepagents_code.configuration.resolver import RankedProviderValue
    from deepagents_code.configuration.types import Unset

    if not status.usable:
        result: ProviderResult[list[str]] = Unset()
    elif managed:
        result = _managed_entries_result(data)
    else:
        result = _user_entries_result(data)
    return RankedProviderValue(rank, durable, status, result)


def _resolve_entries(
    providers: tuple[RankedProviderValue[list[str]], ...],
) -> ResolvedValue[list[str]] | None:
    """Resolve deny tiers with the option's manifest-declared union strategy.

    Returns:
        The accumulated names and rank metadata, or `None` when every tier is unset.

    Raises:
        RuntimeError: If the option is missing from the manifest.
    """
    from deepagents_code.config_manifest import get_option
    from deepagents_code.configuration.resolver import resolve_ranked

    option = get_option("mcp.disabled_servers")
    if option is None:
        msg = "mcp.disabled_servers is missing from the config manifest"
        raise RuntimeError(msg)
    return resolve_ranked(providers, strategy=option.merge_strategy.value)


def _raise_for_managed_provider(
    provider: RankedProviderValue[list[str]],
) -> None:
    """Apply the deny-list callsite's fail-closed health policy.

    Raises:
        ManagedConfigError: If the provider is unhealthy or its value is invalid.
    """
    from deepagents_code.configuration.service import ManagedConfigError
    from deepagents_code.configuration.types import (
        Invalid,
        ProviderHealth,
        ProviderStatus,
    )

    if not provider.status.usable:
        raise ManagedConfigError(provider.status)
    if isinstance(provider.result, Invalid):
        raise ManagedConfigError(
            ProviderStatus(
                provider.status.name,
                provider.status.path,
                ProviderHealth.CORRUPT,
                provider.result.reason,
            )
        )


def _disabled_entries(data: dict[str, Any]) -> set[str]:
    """Return disabled names from the current config shape with legacy fallback."""
    section = data.get(_SECTION)
    if isinstance(section, dict):
        entries = _coerce_entries(section.get(_KEY))
        if entries is not None:
            return entries

    legacy_section = data.get(_LEGACY_SECTION)
    if isinstance(legacy_section, dict):
        entries = _coerce_entries(legacy_section.get(_LEGACY_KEY))
        if entries is not None:
            return entries

    return set()


def _managed_disabled_servers() -> set[str]:
    """Return names denied by the read-only managed source.

    Returns:
        Server names the managed source denies.

    Raises:
        ManagedConfigError: If managed policy exists but cannot be parsed, or
            declares a deny list that cannot be read as server names. Both
            cases would otherwise yield an empty set, which is indistinguishable
            from "nothing is denied", so returning it would re-enable every
            administrator-denied server. The caller must fail closed instead.
    """  # noqa: DOC502 - `_raise_for_managed_provider` owns fail-closed policy
    from deepagents_code.configuration.resolver import MANAGED_RANK
    from deepagents_code.configuration.service import get_managed_snapshot

    snapshot = get_managed_snapshot()
    provider = _ranked_entries(
        snapshot.data,
        rank=MANAGED_RANK,
        durable=True,
        status=snapshot.status,
        managed=True,
    )
    _raise_for_managed_provider(provider)
    resolved = _resolve_entries((provider,))
    return set() if resolved is None else set(resolved.value)


def _remove_legacy_disabled_section(data: dict[str, Any]) -> None:
    """Drop the old top-level section after writing the folded config shape."""
    legacy_section = data.get(_LEGACY_SECTION)
    if not isinstance(legacy_section, dict):
        data.pop(_LEGACY_SECTION, None)
        return
    legacy_section.pop(_LEGACY_KEY, None)
    if legacy_section:
        data[_LEGACY_SECTION] = legacy_section
    else:
        data.pop(_LEGACY_SECTION, None)


def get_disabled_servers(*, config_path: Path | None = None) -> set[str]:
    """Return the server names disabled by the user or by managed config.

    Args:
        config_path: Override the default config location; intended for tests.
            Passing a path also excludes managed policy from this read, so
            production callers must pass `None`.

    Returns:
        Union of the user and managed deny sets. Managed denies survive an
        unreadable or corrupt user config, so the result is empty only when
        nothing is denied at either layer.

    Raises:
        ManagedConfigError: If managed policy exists but cannot be parsed.
            Callers must treat this as "deny everything", never as an empty
            deny set.
    """  # noqa: DOC502 - propagates from `_managed_disabled_servers`
    from deepagents_code.configuration.providers import TomlFileProvider
    from deepagents_code.configuration.resolver import MANAGED_RANK, USER_RANK
    from deepagents_code.configuration.service import (
        ConfigSources,
        get_config_sources,
        get_managed_snapshot,
    )
    from deepagents_code.configuration.types import ProviderHealth

    is_default = config_path is None
    path = _DEFAULT_CONFIG_PATH if config_path is None else config_path
    if is_default:
        # `_DEFAULT_CONFIG_PATH` is a long-standing test/embedder seam in this
        # module. Production points it at the same path as `get_config_sources`,
        # while constructing the pair here preserves overrides without turning
        # an explicit user path into an instruction to omit managed policy.
        sources = ConfigSources(
            managed=get_managed_snapshot(),
            user=TomlFileProvider("config.toml", path).load(),
        )
    else:
        sources = get_config_sources(user_path=path)
    managed = _ranked_entries(
        sources.managed.data,
        rank=MANAGED_RANK,
        durable=True,
        status=sources.managed.status,
        managed=True,
    )
    user = _ranked_entries(
        sources.user.data,
        rank=USER_RANK,
        durable=True,
        status=sources.user.status,
        managed=False,
    )
    if is_default:
        _raise_for_managed_provider(managed)
    if sources.user.status.health in {
        ProviderHealth.CORRUPT,
        ProviderHealth.UNREADABLE,
    }:
        logger.warning(
            "Could not read MCP disabled config at %s: %s",
            path,
            sources.user.status.detail or sources.user.status.health.value,
        )
    resolved = _resolve_entries((managed, user))
    return set() if resolved is None else set(resolved.value)


def is_server_disabled(server_name: str, *, config_path: Path | None = None) -> bool:
    """Return `True` when `server_name` is in the disabled set.

    Args:
        server_name: MCP server name from `mcpServers` config.
        config_path: Override the default config location; intended for tests.
            Passing a path also excludes managed policy from this read, so
            production callers must pass `None`.

    Returns:
        `True` when the server is recorded as disabled, and `True` when
        managed policy exists but cannot be parsed — an unreadable deny list
        must not read as permission. `False` when only the user config is
        unreadable, which is the pre-existing behavior.
    """
    from deepagents_code.configuration.service import ManagedConfigError

    try:
        return server_name in get_disabled_servers(config_path=config_path)
    except ManagedConfigError:
        logger.error(  # noqa: TRY400
            "Managed MCP policy is unreadable; treating %r as disabled.",
            server_name,
        )
        return True


def set_server_disabled(
    server_name: str,
    disabled: bool,
    *,
    config_path: Path | None = None,
) -> tuple[bool, str | None]:
    """Add or remove `server_name` from the persistent disabled set.

    Refuses to write when the existing config cannot be parsed so a
    corrupt or permission-denied file is not silently overwritten —
    that would discard sibling sections such as model profiles.

    Args:
        server_name: MCP server name from `mcpServers` config.
        disabled: `True` to disable, `False` to re-enable.
        config_path: Override the default config location; intended for tests.
            Passing a path also excludes managed policy from this read, so
            production callers must pass `None`.

    Returns:
        Tuple of `(ok, detail)`. `detail` is a short user-facing string
        suitable for a toast, and its meaning depends on `ok`: on failure it
        is the error, and on success it is either `None` or a notice that
        managed config keeps the saved preference from taking effect. Check
        `ok` first — a non-`None` `detail` does not by itself mean failure.
    """
    is_default = config_path is None
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH
    try:
        # Parsed only to reject a corrupt file with a specific message before
        # any write is attempted; the writer recomputes the entries itself.
        _load_config(config_path)
    except _ConfigLoadError as exc:
        return False, str(exc)
    from deepagents_code.configuration.service import ManagedConfigError

    try:
        managed_denied = is_default and server_name in _managed_disabled_servers()
    except ManagedConfigError as exc:
        # Re-enabling against policy that cannot be read would be a fail-open,
        # so refuse the write rather than record a preference whose managed
        # shadow is unknown.
        if not disabled:
            return False, str(exc)
        managed_denied = False
    shadowed = managed_denied and not disabled
    shadowed_detail = (
        f"MCP server {server_name!r} remains disabled by managed config."
        if shadowed
        else None
    )
    # The writer recomputes the set under the lock and reports "no change"
    # itself, so there is no pre-lock equality check to make here.
    write_error = _save_disabled_entry(server_name, disabled, config_path)
    if write_error is None:
        return True, shadowed_detail
    return False, write_error
