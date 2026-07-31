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

from deepagents_code._env_vars import DEBUG, INVOKED_AS, is_env_truthy

logger = logging.getLogger(__name__)

DEFAULT_INVOKED_NAME = "dcode"
"""Command name assumed when the launch name cannot be determined."""

STANDARD_INVOKED_NAMES = frozenset({"dcode", "deepagents-code"})
"""Console scripts shipped in `pyproject.toml` (`[project.scripts]`).

Anything else — a per-checkout shim or a user alias — is non-standard and gets a
one-line note in the Debug Console at launch (see `log_nonstandard_invoked_name`).
Duplicated here by hand because this module must stay import-light; reading the
installed entry points at runtime would be slower and can disagree with the shim
the user actually typed. `test_invocation.py` has a drift guard against
`pyproject.toml`.
"""

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


@lru_cache(maxsize=1)
def log_nonstandard_invoked_name() -> None:
    r"""Note a non-standard launch name in the Debug Console, once per process.

    Cached so repeated calls cannot repeat the note; tests that vary the launch
    environment must call `log_nonstandard_invoked_name.cache_clear()`. Shim
    users launch through a name this package does not ship (see
    `STANDARD_INVOKED_NAMES`), and a wrong resume hint is otherwise impossible to
    trace back to how the name was resolved.

    The level is chosen so the note is never user-facing but always reaches the
    in-app Debug Console (`Ctrl+\\`): the in-memory buffer floors the package
    logger at `INFO` and its handler passes `DEBUG` (`_debug_buffer`), so when
    `DEEPAGENTS_CODE_DEBUG` is off a `DEBUG` record would be filtered before the
    buffer saw it — `INFO` still prints nothing to the terminal because the
    buffer is the only handler in the chain. When debug mode is on, `DEBUG` is
    used so the note also lands in the debug log file.
    """
    name = invoked_name()
    if name in STANDARD_INVOKED_NAMES:
        return
    logger.log(
        logging.DEBUG if is_env_truthy(DEBUG) else logging.INFO,
        "Invoked as non-standard command %r; resume hints will use this name",
        name,
    )
