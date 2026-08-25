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

__all__ = ["reserved_agent_dir_names"]


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
