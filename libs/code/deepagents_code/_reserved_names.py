"""Directory names the app owns under the user profile root.

Agent profiles are siblings of the app's own directories under the profile
root, so an agent named after one of them would resolve onto app state. Both
the picker and the write path need that list.

This module is imported on the CLI startup path (`config.get_agent_dir` and
`main`) and by the agent picker. Keep its module level free of heavy imports.
The constants are read inside the function, so importing this module never
pulls in `deepagents` or LangChain. The function itself must stay light too,
because it runs at startup. See `AGENTS.md` § Startup performance.
"""

from __future__ import annotations

import functools
import sys

__all__ = ["is_reserved_agent_dir_name", "reserved_agent_dir_names"]


@functools.lru_cache(maxsize=1)
def reserved_agent_dir_names() -> frozenset[str]:
    """Return non-agent directory names reserved by the app under the profile root.

    These directories are created by the app for its own use and must never
    appear in the agent picker — even if they contain an `AGENTS.md` file
    (e.g. after `dcode -a plugins` stamps the marker via memory setup):

    - `bin/` holds the managed `rg` binary when the shared installation
      directory is unwritable (`managed_tools.FALLBACK_BIN_DIR`).
    - `plugins/` holds installed plugin state (`_constants`, re-exported by
      `plugins.store`).
    - `conversation_history/` holds offloaded per-thread archives (`offload`).

    `agent/` is deliberately absent: `ProfilePaths.default_skills_dir` names it,
    but nothing reads that field yet, so reserving it would block a legitimate
    agent name for a directory the app does not actually create.

    Each name is derived from the module that owns it. That keeps a single
    source of truth instead of a copy here. `FALLBACK_BIN_DIR` is the bin
    constant that lives under the profile root. The preferred `BIN_DIR` is
    installation-scoped. Deriving from `BIN_DIR` would reserve a profile name
    from an unrelated path.

    `DEFAULT_PLUGIN_DIRNAME` comes from `_constants`, not `plugins.store`.
    Importing `plugins.store` loads the `plugins` package `__init__`, which
    pulls in pydantic through `plugins.models`. This function runs on the
    startup path, so that import would cost every launch about 65 ms.

    The result is cached since the reserved set is constant for the process.

    Returns:
        The reserved directory names.
    """
    from deepagents_code._constants import DEFAULT_PLUGIN_DIRNAME
    from deepagents_code.managed_tools import FALLBACK_BIN_DIR
    from deepagents_code.offload import CONVERSATION_HISTORY_DIRNAME

    return frozenset(
        {
            FALLBACK_BIN_DIR.name,
            DEFAULT_PLUGIN_DIRNAME,
            CONVERSATION_HISTORY_DIRNAME,
        },
    )


def _normalized_for_fs(name: str) -> str:
    """Reduce `name` to the spelling the platform's filesystem compares on.

    Reserved-name guards run against a string the user typed, but the
    filesystem decides which directory that string resolves onto. The default
    macOS and Windows filesystems are case-insensitive, so `Plugins` opens the
    same directory as the reserved `plugins/`; Windows additionally strips
    trailing dots and spaces, so `plugins ` aliases it there too. Comparing
    the raw string on those platforms would let those spellings bypass the
    guard and stamp agent state into an app-owned directory. (`plugins.`
    never reaches this guard: the agent-name character allowlist rejects `.`
    first.)

    The fold applies only where the default filesystem performs it. On Linux
    `Plugins/` is a genuinely different directory from `plugins/`, so folding
    there would reject a harmless name and hide a real agent from the picker.
    Linux can be given a case-insensitive mount, but detecting that is a
    filesystem probe on the startup path and the common case must not reject
    valid names.

    `str.casefold` (rather than `lower`) matches the full Unicode case
    folding the filesystems apply. The trailing dot/space strip applies only
    on Windows, where the filesystem performs it; on POSIX `plugins ` is a
    genuinely different directory from `plugins/`, so folding it there would
    reject a harmless name.

    Returns:
        The normalized name.
    """
    if sys.platform == "darwin" or sys.platform == "win32":
        name = name.casefold()
    if sys.platform == "win32":
        name = name.rstrip(". ")
    return name


@functools.lru_cache(maxsize=1)
def _reserved_names_folded() -> frozenset[str]:
    """Return the reserved names reduced by `_normalized_for_fs`.

    Returns:
        The normalized reserved directory names.
    """
    return frozenset(_normalized_for_fs(name) for name in reserved_agent_dir_names())


def is_reserved_agent_dir_name(name: str) -> bool:
    """Report whether `name` resolves onto an app-owned profile directory.

    Use this instead of `name in reserved_agent_dir_names()` wherever the
    input is a user-supplied agent name: the membership test is exact-string
    and misses case and trailing-dot aliases that the filesystem itself would
    resolve onto the reserved directory.

    Returns:
        `True` when `name` names an app-owned directory.
    """
    return _normalized_for_fs(name) in _reserved_names_folded()
