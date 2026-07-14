"""Storage paths for offloaded conversation history."""

from __future__ import annotations

import logging
import os
import stat
import tempfile
from pathlib import Path, PurePath

logger = logging.getLogger(__name__)


def _filesystem_tool_path(path: PurePath) -> str:
    """Represent an absolute host path in the filesystem tool path format.

    Drive-qualified paths are rejected by the SDK's virtual path validation. The
    Windows extended-length form keeps the drive while starting with the `/`
    required by filesystem tools; `pathlib` and Windows APIs still resolve it to
    the same host directory.

    Args:
        path: Absolute host path to represent.

    Returns:
        A forward-slash path accepted by the filesystem tool contract.
    """
    normalized = path.as_posix()
    if path.drive and not path.drive.startswith("\\\\"):
        return f"//?/{normalized}"
    return normalized


_EPHEMERAL_OFFLOAD_STORAGE = False
"""Whether the most recent `_offload_fallback_root` fell back to temp storage."""


def offload_storage_is_ephemeral() -> bool:
    """Return whether offload history is routed to non-persistent storage.

    `True` when the persistent `~/.deepagents` location was unwritable and the
    most recent `_offload_fallback_root` fell back to a temporary directory that
    may not survive a restart. Only meaningful in local mode, where
    `_offload_fallback_root` runs in the same process as the UI; in
    server/sandbox mode persistence is owned by the server backend and this flag
    stays `False` client-side.

    Returns:
        `True` if the local offload root is a temporary, non-persistent
        directory; `False` when it is the persistent per-user location (or was
        never resolved in this process).
    """
    return _EPHEMERAL_OFFLOAD_STORAGE


def _harden_dir(path: Path) -> None:
    """Create `path` if needed and restrict it to the current user.

    Only ever call this on directories owned by this process's storage (a temp
    dir or a dedicated subdirectory), never on the shared `~/.deepagents` config
    root.

    Args:
        path: Directory to create and harden to `0o700`.

    Raises:
        OSError: If the path exists but is not a directory, or the directory
            cannot be created or its mode changed (e.g. a read-only mount).
        PermissionError: If the existing directory is owned by another local user.
    """
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        msg = f"Path is not a directory: {path}"
        raise OSError(msg)
    getuid = getattr(os, "getuid", None)
    if getuid is not None and info.st_uid != getuid():
        msg = f"Directory is owned by another user: {path}"
        raise PermissionError(msg)
    # `mkdir(mode=...)` does not tighten an existing directory. These directories
    # can hold conversation data and offloaded tool results, so they must remain
    # inaccessible to other local accounts regardless of the process umask.
    path.chmod(0o700)


def _probe_writable(path: Path) -> None:
    """Confirm `path` accepts new files (catches read-only mounts).

    Creating the directory is insufficient when it already exists on a read-only
    mount; a temporary file proves writes can succeed.

    Args:
        path: Directory to probe.
    """
    with tempfile.NamedTemporaryFile(dir=path, prefix=".write-test-"):
        pass


def _artifacts_root() -> str:
    """Return a stable, private per-user directory for offloaded artifacts.

    In local mode, large tool results are written here on the real filesystem
    (rather than a hidden virtual backend) so the agent can inspect them with
    `execute` (`jq`, `grep`, `python`) using the exact path the offload message
    hands it. The directory is:

    - stable across process restarts (keyed by user id) so paths embedded in a
      resumed thread's history stay resolvable, and
    - hardened to `0o700` and owned by the current user so other local accounts
      cannot read offloaded results.

    It is used as the `CompositeBackend` `artifacts_root`, which the SDK
    filesystem and summarization middleware turn into the
    `<root>/large_tool_results/` and `<root>/conversation_history/` prefixes.

    If the predictable per-user path is unusable -- e.g. squatted by another
    local user or symlinked (rejected by `_harden_dir`'s ownership / `S_ISDIR`
    guards) -- this falls back to a private unique directory rather than reusing
    a foreign-owned path or failing agent startup. The fallback is not stable
    across restarts, but it never trusts a directory owned by someone else.

    Returns:
        The private, writable directory in the filesystem tool path format.
    """
    getuid = getattr(os, "getuid", None)
    suffix = str(getuid()) if getuid is not None else str(os.getpid())
    temp_root = Path(tempfile.gettempdir())
    root = temp_root / f"dcode-artifacts-{suffix}"
    try:
        _harden_dir(root)
        _probe_writable(root)
    except (OSError, RuntimeError):
        logger.warning(
            "Predictable per-user artifacts directory is unavailable; creating "
            "a private unique directory (paths will not be stable across restarts)",
            exc_info=True,
        )
        unique = Path(
            tempfile.mkdtemp(prefix=f"dcode-artifacts-{suffix}-", dir=temp_root)
        )
        _harden_dir(unique)
        _probe_writable(unique)
        return _filesystem_tool_path(unique)
    return _filesystem_tool_path(root)


def _offload_fallback_root() -> Path:
    """Return a writable base directory for offloaded conversation history.

    Prefers the persistent per-user `~/.deepagents` directory so offloaded
    history survives across sessions and is easy to locate; falls back to a
    private temporary directory when the home directory cannot be resolved or
    written. This is the live root for the local-mode `conversation_history`
    backend in `agent.py`.

    Archives always live in the `conversation_history` subdirectory of the
    returned root. The `0o700` hardening therefore targets that subdirectory,
    never the shared `~/.deepagents` config root -- which also houses
    `config.toml`, `hooks.json`, `.env`, and `.state/`, whose permissions this
    must not disturb. A temporary fallback root is created solely for offload,
    so the whole directory is hardened in that case.

    Note: the `S_ISDIR` check below (which uses `lstat`, deliberately not
    following the link) guards the paths it is applied to -- the
    `conversation_history` subdirectory and, in the fallback case, the temp
    root -- not `~/.deepagents` itself, which is created with a plain `mkdir`.
    So a `conversation_history` (or temp root) that is itself a symlink is
    rejected, whereas a symlinked `~/.deepagents` pointing at a directory the
    current user owns is followed transparently and archives persist normally.
    (A dangling `~/.deepagents` symlink still falls through to temporary
    storage, but via `mkdir` raising, not via this check.)

    Returns:
        A directory whose `conversation_history` subdirectory is private and
        writable.
    """

    def _prepare_user_dir() -> Path:
        base = Path.home() / ".deepagents"
        # Ensure the shared config root exists and is usable, but leave its
        # permissions untouched -- hardening belongs on the archive subdir only.
        base.mkdir(parents=True, exist_ok=True)
        archive_dir = base / "conversation_history"
        _harden_dir(archive_dir)
        _probe_writable(archive_dir)
        return base

    def _prepare_temp_dir(path: Path) -> Path:
        # A temp dir is created solely for offload and is not shared config, so
        # hardening the whole directory (which protects its archive subdir) is
        # both safe and necessary in world-writable temp locations.
        _harden_dir(path)
        _probe_writable(path)
        return path

    global _EPHEMERAL_OFFLOAD_STORAGE  # noqa: PLW0603
    try:
        root = _prepare_user_dir()
    except (RuntimeError, OSError):
        logger.warning(
            "User data directory is not writable; falling back to temporary "
            "offload storage, which may not persist across restarts",
            exc_info=True,
        )
    else:
        _EPHEMERAL_OFFLOAD_STORAGE = False
        return root
    # Only reached on the fallback path: every root produced below is temporary
    # and may not survive a restart.
    _EPHEMERAL_OFFLOAD_STORAGE = True
    getuid = getattr(os, "getuid", None)
    suffix = str(getuid()) if getuid is not None else str(os.getpid())
    temp_root = Path(tempfile.gettempdir())
    path = temp_root / f"deepagents-{suffix}"
    try:
        return _prepare_temp_dir(path)
    except (OSError, RuntimeError):
        logger.warning(
            "Per-user temporary offload directory is unavailable; creating "
            "a private unique directory",
            exc_info=True,
        )
        unique = Path(tempfile.mkdtemp(prefix=f"deepagents-{suffix}-", dir=temp_root))
        return _prepare_temp_dir(unique)
