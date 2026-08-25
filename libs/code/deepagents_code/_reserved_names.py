"""Directory names the app owns under the user profile root.

Agent profiles are siblings of the app's own directories under the profile
root, so an agent named after one of them would resolve onto app state. Both
the picker and the write path need that list.

This module is imported from `config.get_agent_dir`, which is on the CLI
startup path. Keep its module level free of heavy imports: the constants are
read inside the function so importing this module never pulls in `deepagents`
or LangChain. See `AGENTS.md` § Startup performance.
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
    - `plugins/` holds installed plugin state (`plugins.store`).
    - `conversation_history/` holds offloaded per-thread archives (`offload`).

    `agent/` is deliberately absent: `ProfilePaths.default_skills_dir` names it,
    but nothing reads that field yet, so reserving it would block a legitimate
    agent name for a directory the app does not actually create.

    Each name is derived from its owning module so it stays a single source of
    truth rather than being hardcoded here. `FALLBACK_BIN_DIR` is the one that
    lives under the profile root — the preferred `BIN_DIR` is installation-
    scoped, so deriving from it would reserve a profile name based on an
    unrelated path. The result is cached since the reserved set is constant for
    the process.

    Returns:
        The reserved directory names.
    """
    from deepagents_code.managed_tools import FALLBACK_BIN_DIR
    from deepagents_code.offload import CONVERSATION_HISTORY_DIRNAME
    from deepagents_code.plugins.store import DEFAULT_PLUGIN_DIRNAME

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
    the raw string would let those spellings bypass the guard and stamp agent
    state into an app-owned directory. (`plugins.` never reaches this guard:
    the agent-name character allowlist rejects `.` first.)

    `str.casefold` (rather than `lower`) matches the full Unicode case
    folding the filesystems apply. The trailing dot/space strip applies only
    on Windows, where the filesystem performs it; on POSIX `plugins ` is a
    genuinely different directory from `plugins/`, so folding it there would
    reject a harmless name.

    Returns:
        The normalized name.
    """
    normalized = name.casefold()
    if sys.platform == "win32":
        normalized = normalized.rstrip(". ")
    return normalized


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
