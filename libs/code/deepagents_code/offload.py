"""Storage paths for offloaded conversation history."""

from __future__ import annotations

import logging
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePath

logger = logging.getLogger(__name__)

_FALLBACK_ARTIFACTS_ROOT = "/dcode-artifacts-fallback"


@dataclass(frozen=True)
class _ArtifactsStorage:
    """Agent-visible artifacts root and optional routed large-result directory."""

    root: str
    large_results_dir: Path | None = None


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

_UNIQUE_OFFLOAD_FALLBACK_ROOT: Path | None = None
"""Private random fallback root that cannot be reconstructed on a later call."""


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


def _artifacts_root() -> _ArtifactsStorage:
    """Return storage configuration for offloaded artifacts.

    The normal path is a stable, hardened host directory that filesystem tools
    and shell commands can use directly. If that predictable directory is
    unusable, large results use a private unique directory behind a stable virtual
    root. Keeping the virtual root stable lets conversation archive paths persisted
    in thread state continue matching their dedicated route after a restart.

    Returns:
        The agent-visible artifacts root and an optional directory to which large
        results must be routed.
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
            "Predictable per-user artifacts directory is unavailable; routing "
            "large results from a stable virtual prefix to private temporary storage",
            exc_info=True,
        )
        unique = Path(
            tempfile.mkdtemp(prefix=f"dcode-artifacts-{suffix}-", dir=temp_root)
        )
        _harden_dir(unique)
        _probe_writable(unique)
        return _ArtifactsStorage(
            root=_FALLBACK_ARTIFACTS_ROOT,
            large_results_dir=unique,
        )
    return _ArtifactsStorage(root=_filesystem_tool_path(root))


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

    global _EPHEMERAL_OFFLOAD_STORAGE, _UNIQUE_OFFLOAD_FALLBACK_ROOT  # noqa: PLW0603
    if _UNIQUE_OFFLOAD_FALLBACK_ROOT is not None:
        # Unlike the persistent and predictable temp paths, a directory created
        # by `mkdtemp` cannot be derived again. Keep returning the root already
        # used by the archive backend so cleanup reaches the same files.
        _EPHEMERAL_OFFLOAD_STORAGE = True
        return _UNIQUE_OFFLOAD_FALLBACK_ROOT
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
        _UNIQUE_OFFLOAD_FALLBACK_ROOT = _prepare_temp_dir(unique)
        return _UNIQUE_OFFLOAD_FALLBACK_ROOT


def delete_offloaded_history(thread_id: str) -> bool:
    """Remove a thread's offloaded conversation-history archive.

    Deletes the per-thread markdown file written by the local-mode
    `conversation_history` backend (`{root}/conversation_history/{thread_id}.md`),
    resolving `root` with `_offload_fallback_root` so the persistent
    `~/.deepagents` location and any temporary fallback are both covered.

    Best-effort: filesystem failures are logged and swallowed rather than
    raised, so a failed cleanup never blocks thread deletion. Resolving the
    offload root is not side-effect-free -- it creates (and hardens) the
    `conversation_history` directory and writes a short-lived probe file -- so a
    call for a thread that has no archive still touches the filesystem before
    returning `False`.

    In server/sandbox mode the archive lives on the sandbox backend rather than
    the local `~/.deepagents` directory, so there is no local archive to remove.

    Args:
        thread_id: Thread whose offloaded history should be removed.

    Returns:
        `True` only if an archive file was removed. `False` in every other case:
        an empty or rejected `thread_id`, an unresolvable offload root, a missing
        archive, or an `unlink` failure.
    """
    if not thread_id:
        return False
    try:
        archive_dir = _offload_fallback_root() / "conversation_history"
    except (OSError, RuntimeError):
        logger.warning(
            "Could not resolve offload root to clean history for thread %s",
            thread_id,
            exc_info=True,
        )
        return False
    archive_path = archive_dir / f"{thread_id}.md"
    # Guard against a crafted thread id escaping the archive directory. Thread
    # ids are system-generated UUID7 strings, so a rejection here means either a
    # crafted input or a bug emitting malformed ids -- both worth a trace, and
    # both distinct from the benign "no archive exists" path below.
    if archive_path.parent != archive_dir:
        logger.warning(
            "Refusing to delete offloaded history for suspicious thread id %r",
            thread_id,
        )
        return False
    try:
        archive_path.unlink()
    except FileNotFoundError:
        return False
    except OSError:
        logger.warning(
            "Failed to delete offloaded conversation history for thread %s",
            thread_id,
            exc_info=True,
        )
        return False
    logger.debug("Deleted offloaded conversation history for thread %s", thread_id)
    return True
