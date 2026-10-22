"""Tests for the session presence lock gating startup Python migrations."""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from deepagents_code import _session_lock
from deepagents_code._session_lock import (
    hold_session_lock,
    is_sole_session,
    session_lock_fds,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the session lock is POSIX-only"
)


@pytest.fixture
def other_session() -> Iterator[int]:
    """Hold the session lock shared through an independent open file.

    `flock` locks belong to the open file description, so a second `os.open`
    in this process contends exactly like another process would.

    Yields:
        The other session's descriptor.
    """
    _session_lock.SESSION_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(_session_lock.SESSION_LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_SH)
    yield fd
    os.close(fd)


def test_unregistered_process_is_never_sole() -> None:
    """Without the lock, presence is unknown, so migration must not run."""
    assert session_lock_fds() == ()
    assert is_sole_session() is False


def test_registered_process_alone_is_sole() -> None:
    """The only registered session may migrate."""
    hold_session_lock()

    assert len(session_lock_fds()) == 1
    assert is_sole_session() is True


def test_hold_is_idempotent() -> None:
    """Registering twice reuses one descriptor rather than leaking another."""
    hold_session_lock()
    fds = session_lock_fds()
    hold_session_lock()

    assert session_lock_fds() == fds
    assert is_sole_session() is True


def test_other_session_blocks_sole(other_session: int) -> None:  # noqa: ARG001
    """A second registered session makes the check report others running."""
    hold_session_lock()

    assert is_sole_session() is False


def test_check_keeps_this_session_registered() -> None:
    """After the check, this process still holds the lock shared."""
    hold_session_lock()
    assert is_sole_session() is True

    probe = os.open(_session_lock.SESSION_LOCK_FILE, os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Shared, not exclusive: other launches can still register.
        fcntl.flock(probe, fcntl.LOCK_SH | fcntl.LOCK_NB)
    finally:
        os.close(probe)


def test_child_given_the_fd_shares_this_session() -> None:
    """A spawned server holds the same lock, not a second one.

    It keeps the install marked in use while it runs, yet does not make its own
    client look like it has company.
    """
    hold_session_lock()
    (fd,) = session_lock_fds()
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE,
        pass_fds=(fd,),
    )
    probe = os.open(_session_lock.SESSION_LOCK_FILE, os.O_RDWR)
    try:
        assert is_sole_session() is True
        # The client exits first; its server still marks the install in use.
        os.close(fd)
        _session_lock._held_fd = None
        with pytest.raises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        assert child.stdin is not None
        child.stdin.close()
        child.wait()
    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(probe)


def test_unwritable_lock_location_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """When the lock file cannot be created, the process is never sole."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    monkeypatch.setattr(_session_lock, "SESSION_LOCK_FILE", blocker / "sessions.lock")

    hold_session_lock()

    assert session_lock_fds() == ()
    assert is_sole_session() is False
