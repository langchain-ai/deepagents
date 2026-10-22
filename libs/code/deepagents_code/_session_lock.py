"""Presence lock telling a launch whether other dcode sessions share its install.

A Python migration rebuilds the uv tool environment, which breaks every dcode
process — and every server it spawned — still running from the old one. Startup
auto-update may only migrate when no other session uses the installation, and
it cannot tell that from a process listing: shims, renamed symlinks, and the
server bootstrap all run under different names.

Instead, each dcode process of a non-editable install (editable installs never
auto-update) holds a shared `flock` on `SESSION_LOCK_FILE` for its lifetime,
and hands the descriptor to the servers it spawns so they hold
the same lock. A process asking "am I alone?" tries to take the lock
exclusively; that succeeds only when no other session's open file holds it.
The kernel drops a lock when its last descriptor closes, so a crashed session
can never leave a stale claim behind.

The answer is a snapshot. It cannot see sessions of versions from before this
lock existed, and nothing stops a session that launches while the migration
runs; both are accepted as rare and recoverable by a relaunch.

POSIX only. On Windows, or whenever the lock cannot be taken, a process is
never reported as alone, so the migration stays deferred to `dcode update`.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

from deepagents_code._paths import PATHS

if sys.platform != "win32":
    import fcntl

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

SESSION_LOCK_FILE: Path = PATHS.installation.locks_dir / "sessions.lock"
"""Advisory lock held shared by every registered dcode process of one installation.

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
    try:
        SESSION_LOCK_FILE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(SESSION_LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        logger.warning(
            "Could not open session lock %s", SESSION_LOCK_FILE, exc_info=True
        )
        return
    try:
        # Blocks only while another process is inside `is_sole_session`, the
        # one place the lock is ever held exclusively, and only for an instant.
        # Failing fast instead would leave this session invisible to that check.
        fcntl.flock(fd, fcntl.LOCK_SH)
    except OSError:
        logger.warning(
            "Could not take session lock %s", SESSION_LOCK_FILE, exc_info=True
        )
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


def session_lock_unavailable() -> bool:
    """Return whether this POSIX process failed to register as a session.

    Lets callers tell "other sessions are running" apart from "presence cannot
    be known", which quitting other sessions would never fix.
    """
    return sys.platform != "win32" and _held_fd is None


def is_sole_session() -> bool:
    """Return whether no other dcode session holds this installation.

    Converting a `flock` is not atomic: the shared lock is released before the
    exclusive one is tried, so two processes checking at once could each see
    the other as absent. An exclusive lock on `sessions.check.lock` beside
    `SESSION_LOCK_FILE` admits one checker at a time; a process that finds it
    taken is not alone.

    Returns:
        `True` only when this process holds the session lock and could take
            it exclusively. `False` when another session holds it, or when the
            lock is unavailable and presence cannot be known.
    """
    if _held_fd is None:
        return False
    check_file = SESSION_LOCK_FILE.with_name("sessions.check.lock")
    try:
        guard = os.open(check_file, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        logger.warning("Could not open %s", check_file, exc_info=True)
        return False
    try:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    else:
        return _probe_exclusive(_held_fd)
    finally:
        os.close(guard)


def _probe_exclusive(fd: int) -> bool:
    """Try to take the session lock exclusively, then hold it shared again.

    Returns:
        Whether the exclusive attempt succeeded and the shared lock is back.
    """
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sole = False
    except OSError:
        logger.warning(
            "Could not check session lock %s", SESSION_LOCK_FILE, exc_info=True
        )
        sole = False
    else:
        sole = True
    # Re-take it on both branches: on Linux a failed upgrade attempt has already
    # released the shared lock.
    return _restore_shared(fd) and sole


def _restore_shared(fd: int) -> bool:
    """Hold the session lock shared again, or unregister on failure.

    Returns:
        Whether this process is still registered.
    """
    global _held_fd  # noqa: PLW0603  # unregister a lock this process lost
    try:
        # Blocking is safe: the check lock is held, so no other process holds
        # this lock exclusively.
        fcntl.flock(fd, fcntl.LOCK_SH)
    except OSError:
        logger.warning(
            "Could not restore the shared session lock %s; other dcode sessions "
            "will not see this one",
            SESSION_LOCK_FILE,
            exc_info=True,
        )
        _held_fd = None
        os.close(fd)
        return False
    return True
