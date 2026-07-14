"""Storage paths for offloaded conversation history."""

from __future__ import annotations

import logging
import os
import stat
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

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

    def _harden(path: Path) -> None:
        """Create `path` if needed and restrict it to the current user.

        Only ever called on directories owned by offload (a fresh temp dir or
        the dedicated `conversation_history` subdirectory), never on the shared
        `~/.deepagents` config root.

        Raises:
            OSError: If the path exists but is not a directory, or the
                directory cannot be created or its mode changed (e.g. a
                read-only mount).
            PermissionError: If the existing directory is owned by another
                local user.
        """
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode):
            msg = f"Offload path is not a directory: {path}"
            raise OSError(msg)
        getuid = getattr(os, "getuid", None)
        if getuid is not None and info.st_uid != getuid():
            msg = f"Offload directory is owned by another user: {path}"
            raise PermissionError(msg)
        # `mkdir(mode=...)` does not tighten an existing directory. Archives
        # contain conversation data, so the offload directory must remain
        # inaccessible to other local accounts regardless of the process umask.
        path.chmod(0o700)

    def _probe_writable(path: Path) -> None:
        # Creating the directory is insufficient when it already exists on a
        # read-only mount. A temporary file proves archive writes can succeed.
        with tempfile.NamedTemporaryFile(dir=path, prefix=".write-test-"):
            pass

    def _prepare_user_dir() -> Path:
        base = Path.home() / ".deepagents"
        # Ensure the shared config root exists and is usable, but leave its
        # permissions untouched -- hardening belongs on the archive subdir only.
        base.mkdir(parents=True, exist_ok=True)
        archive_dir = base / "conversation_history"
        _harden(archive_dir)
        _probe_writable(archive_dir)
        return base

    def _prepare_temp_dir(path: Path) -> Path:
        # A temp dir is created solely for offload and is not shared config, so
        # hardening the whole directory (which protects its archive subdir) is
        # both safe and necessary in world-writable temp locations.
        _harden(path)
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
