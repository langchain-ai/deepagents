"""Presence lock telling a launch whether other dcode sessions share its install.

A Python migration rebuilds the uv tool environment, which breaks every dcode
process — and every server it spawned — still running from the old one. Startup
auto-update may only migrate when it is the sole user of the installation, and
it cannot tell that from a process listing: shims, renamed symlinks, and the
server bootstrap all run under different names.

Instead, each dcode process of a non-editable install (editable installs never
auto-update) holds a shared `flock` on `SESSION_LOCK_FILE` for its lifetime,
and hands the descriptor to the servers it spawns so they hold
the same lock. A process asking "am I alone?" tries to take the lock
exclusively; that succeeds only when no other process holds it. The kernel
drops a lock when its last descriptor closes, so a crashed session can never
leave a stale claim behind.

POSIX only. On Windows, or whenever the lock cannot be taken, a process is
never reported as alone, so the migration stays deferred to `dcode update`.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

from deepagents_code._paths import PATHS

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

SESSION_LOCK_FILE: Path = PATHS.installation.locks_dir / "sessions.lock"
"""Advisory lock held shared by every running dcode process of one installation.

Lives beside `update.lock`, outside the tool environment, so a migration that
rebuilds the environment keeps the same file. Carries no data.

Installation-scoped only, with no per-profile fallback: a profile-scoped lock
would hide sessions of other profiles, and over-reporting "alone" is the one
failure this lock must not have.
"""

_held_fd: int | None = None
"""Descriptor holding this process's shared lock, or `None` when not held."""


def hold_session_lock() -> None:
    """Register this process as a running session for its lifetime.

    Idempotent and fail-soft: when the lock cannot be taken, the process runs
    unregistered and `is_sole_session` reports `False`. The descriptor is
    non-inheritable, so the re-exec after an update drops it and the new
    process registers itself again.
    """
    global _held_fd  # noqa: PLW0603  # one registration per process
    if _held_fd is not None or sys.platform == "win32":
        return
    import fcntl

    try:
        SESSION_LOCK_FILE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(SESSION_LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        logger.debug("Could not open session lock %s", SESSION_LOCK_FILE, exc_info=True)
        return
    try:
        # Blocks only while another process is inside `is_sole_session`, the
        # one place the lock is ever held exclusively, and only for an instant.
        # Failing fast instead would leave this session invisible to that check.
        fcntl.flock(fd, fcntl.LOCK_SH)
    except OSError:
        logger.debug("Could not take session lock %s", SESSION_LOCK_FILE, exc_info=True)
        os.close(fd)
        return
    _held_fd = fd


def session_lock_fds() -> tuple[int, ...]:
    """Return descriptors a spawned server must inherit to count as this session.

    `flock` locks belong to the open file description, so a child given the
    descriptor holds the same lock: the server keeps the install marked in use
    if it outlives its client, and does not count as a second session.
    """
    return () if _held_fd is None else (_held_fd,)


def is_sole_session() -> bool:
    """Return whether no other dcode process is running from this installation.

    Callers must hold `update_install_lock`. Converting a `flock` is not
    atomic: the shared lock is released before the exclusive one is tried, so
    two processes checking at once could each see the other as absent. The
    install lock admits one checker at a time.

    Returns:
        `True` only when this process holds the session lock and could take
            it exclusively. `False` when another process holds it, or when the
            lock is unavailable and presence cannot be known.
    """
    fd = _held_fd
    if fd is None:
        return False
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sole = False
    else:
        sole = True
    try:
        # Blocking is safe: other processes only ever hold the lock
        # exclusively for the instant of this same check.
        fcntl.flock(fd, fcntl.LOCK_SH)
    except OSError:
        logger.warning(
            "Could not restore the shared session lock %s; other dcode sessions "
            "may not see this one",
            SESSION_LOCK_FILE,
            exc_info=True,
        )
    return sole
