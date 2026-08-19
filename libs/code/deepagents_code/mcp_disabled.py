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
    from pathlib import Path

from deepagents_code.model_config import DEFAULT_CONFIG_PATH as _DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)

_SECTION = "mcp"
_KEY = "disabled_servers"
_LEGACY_SECTION = "mcp_disabled"
_LEGACY_KEY = "servers"


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


def _save_disabled_entry(server_name: str, disabled: bool, config_path: Path) -> bool:
    """Add or remove one server name through the shared writer.

    The entry set is recomputed from the parse the writer performs inside its
    lock, never from a snapshot read before the lock was taken. Writing a
    pre-lock set would drop a concurrent disable of a *different* server, and
    editing in place keeps a concurrent `[ui]` or `[models]` write intact.

    Returns:
        Whether the transaction succeeded.
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
    return result.ok


def _coerce_entries(entries: object) -> set[str] | None:
    """Return valid server names from a TOML value, or `None` when unset."""
    if not isinstance(entries, list):
        return None
    return {name for name in entries if isinstance(name, str) and name}


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
        ManagedConfigError: If managed policy exists but cannot be parsed. An
            unusable snapshot carries an empty table, which is indistinguishable
            from "nothing is denied", so returning it would re-enable every
            administrator-denied server. The caller must fail closed instead.
    """
    from deepagents_code.configuration.service import (
        ManagedConfigError,
        get_managed_snapshot,
    )

    snapshot = get_managed_snapshot()
    if not snapshot.status.usable:
        raise ManagedConfigError(snapshot.status)
    return _disabled_entries(snapshot.data)


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
    is_default = config_path is None
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH
    # The managed deny set must apply even when the user config is corrupt —
    # otherwise a broken user TOML would silently re-enable admin-denied servers.
    managed = _managed_disabled_servers() if is_default else set()
    try:
        data = _load_config(config_path)
    except _ConfigLoadError:
        return managed
    disabled = _disabled_entries(data)
    disabled.update(managed)
    return disabled


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
    if _save_disabled_entry(server_name, disabled, config_path):
        return True, shadowed_detail
    return False, f"could not write {config_path}"
