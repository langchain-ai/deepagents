"""Persistent store of MCP server names the user has disabled.

Disabled servers are skipped at config merge time so their tools never
reach the agent and no connection is attempted. State lives under
`[mcp_disabled]` in `~/.deepagents/config.toml`, alongside the user's
other configuration sections.

The store keys on server *name* alone. Two configs that both declare a
`github` server will both be disabled by a single entry — intentional,
since the agent cannot distinguish overlapping names at runtime anyway
(later configs in the merge order win).
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from deepagents_code.model_config import DEFAULT_CONFIG_PATH as _DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)

_SECTION = "mcp_disabled"
_KEY = "servers"


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


def _save_config(data: dict[str, Any], config_path: Path) -> bool:
    """Atomic TOML write.

    Returns:
        `True` on success, `False` on I/O failure.
    """
    import tomli_w

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                tomli_w.dump(data, f)
            Path(tmp_path).replace(config_path)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except (OSError, ValueError):
        logger.exception("Failed to save config to %s", config_path)
        return False
    return True


def get_disabled_servers(*, config_path: Path | None = None) -> set[str]:
    """Return the set of server names the user has disabled.

    Args:
        config_path: Override the default config location; intended for tests.

    Returns:
        Set of server names. Empty when nothing is disabled or the config
        cannot be read.
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH
    try:
        data = _load_config(config_path)
    except _ConfigLoadError:
        return set()
    section = data.get(_SECTION)
    if not isinstance(section, dict):
        return set()
    entries = section.get(_KEY)
    if not isinstance(entries, list):
        return set()
    return {name for name in entries if isinstance(name, str) and name}


def is_server_disabled(server_name: str, *, config_path: Path | None = None) -> bool:
    """Return `True` when `server_name` is in the disabled set.

    Args:
        server_name: MCP server name from `mcpServers` config.
        config_path: Override the default config location; intended for tests.

    Returns:
        `True` when the server is recorded as disabled, `False` otherwise
        (including when the config cannot be read).
    """
    return server_name in get_disabled_servers(config_path=config_path)


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

    Returns:
        Tuple of `(ok, error_detail)`. `ok` is `True` on success; on
        failure `error_detail` is a short user-facing string suitable
        for a toast.
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH
    try:
        data = _load_config(config_path)
    except _ConfigLoadError as exc:
        return False, str(exc)
    section = data.get(_SECTION)
    if not isinstance(section, dict):
        section = {}
    entries_raw = section.get(_KEY)
    entries: list[str] = (
        [name for name in entries_raw if isinstance(name, str) and name]
        if isinstance(entries_raw, list)
        else []
    )
    current = set(entries)
    if disabled:
        current.add(server_name)
    else:
        current.discard(server_name)
    if current == set(entries):
        return True, None
    section[_KEY] = sorted(current)
    data[_SECTION] = section
    if _save_config(data, config_path):
        return True, None
    return False, f"could not write {config_path}"
