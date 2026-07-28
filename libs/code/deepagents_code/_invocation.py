"""Resolution of the command name this process was launched with.

Hints that tell the user how to resume a thread have to echo a command the user
can actually paste back. `dcode` is only one of the names that reach this code:
the package ships both `deepagents-code` and `dcode` console scripts, and
per-project shims (a renamed symlink in `~/.local/bin` pointing at a worktree's
`bin/dcode`) are a common way to run several checkouts side by side. Hardcoding
`dcode` tells those users to run a command that may not exist.

`sys.argv[0]` holds the answer, because the kernel passes the pathname given to
`execve` to the interpreter rather than the symlink target, so a shim invoked as
`abc` reports `abc`. It is not always meaningful, though, so `invoked_name`
falls back to `DEFAULT_INVOKED_NAME` whenever the value is missing or does not
look like a command a user could have typed.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from functools import lru_cache
from pathlib import PurePath

from deepagents_code._env_vars import INVOKED_AS

logger = logging.getLogger(__name__)

DEFAULT_INVOKED_NAME = "dcode"
"""Command name assumed when the launch name cannot be determined."""

_MAX_NAME_LENGTH = 64

_SAFE_NAME_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._+-]*\Z")
"""Plausible console-script names: no separators, spaces, or shell metacharacters.

`sys.argv[0]` and the environment are supplied by whatever started the process,
and the resolved name is rendered into a copy-pasteable command, so the shape is
allowlisted rather than escaped.
"""

_WINDOWS_EXECUTABLE_SUFFIX = ".exe"


def _sanitize(raw: str) -> str | None:
    """Return `raw` as a command name, or `None` when it is not plausible.

    Args:
        raw: A candidate name (an `argv[0]` basename or an env-var value).

    Returns:
        The cleaned command name, or `None` when the value cannot be a console
            script the user typed — empty, absurdly long, a Python source file
            (`python -m deepagents_code` reports `__main__.py`), an interpreter
            name, or anything outside `_SAFE_NAME_RE`.
    """
    name = raw.strip()
    if name.lower().endswith(_WINDOWS_EXECUTABLE_SUFFIX):
        # Windows console scripts are `.exe` wrappers; the user types the stem.
        name = name[: -len(_WINDOWS_EXECUTABLE_SUFFIX)]
    if not name or len(name) > _MAX_NAME_LENGTH:
        return None
    if name.endswith(".py") or name.lower().startswith("python"):
        return None
    if not _SAFE_NAME_RE.match(name):
        return None
    return name


@lru_cache(maxsize=1)
def invoked_name() -> str:
    """Return the command name this process was launched with.

    Cached: `sys.argv[0]` and the launch environment are fixed for the life of
    the process. Tests that vary either must call `invoked_name.cache_clear()`.

    Returns:
        The console-script name the user invoked (for example `dcode`,
            `deepagents-code`, or a shim name), or `DEFAULT_INVOKED_NAME` when it
            cannot be determined.
    """
    override = os.environ.get(INVOKED_AS)
    if override is not None:
        name = _sanitize(override)
        if name is not None:
            return name
        logger.debug("Ignoring implausible %s value", INVOKED_AS)
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0:
        name = _sanitize(PurePath(argv0).name)
        if name is not None:
            return name
    return DEFAULT_INVOKED_NAME
