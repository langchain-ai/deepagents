"""Shared debug-logging configuration for runtime and file-based tracing.

When the `DEEPAGENTS_CODE_DEBUG` environment variable is set, modules that handle
streaming or remote communication can enable detailed file-based logging. This
helper centralizes the setup so the env-var names, file path, log level, and
format are defined in one place.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import stat
import sys
import weakref
from pathlib import Path

# Windows-only ACL plumbing; see `_apply_windows_owner_only_dacl`. Imported
# under the guard because `_debug` is on the startup path for every command and
# `ctypes` costs a few milliseconds it can never repay on POSIX.
if os.name == "nt":
    import ctypes
    from ctypes import wintypes

from deepagents_code._env_vars import (
    DEBUG,
    DEBUG_DIRECTORY,
    DEBUG_FILE,
    DEFAULT_DEBUG_DIRECTORY,
    LOG_LEVEL,
    is_env_truthy,
)

logger = logging.getLogger(__name__)

_DEBUG_HANDLER_ATTR = "_deepagents_code_debug_handler"
_CONFIGURED_LOGGERS: weakref.WeakSet[logging.Logger] = weakref.WeakSet()
_ACTIVE_THREAD_ID: str | None = None
_SAFE_THREAD_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_MAX_THREAD_FILENAME_LENGTH = 200
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
"""Canonical level-name to `logging` level mapping.

The single source of truth for level names and their numeric values, shared with
the Debug Console's level filter so severity ordering is never re-derived from
hardcoded integers.
"""


def _warn(message: str) -> None:
    """Report a debug-logging failure to stderr and the in-memory buffer.

    stderr covers headless / pre-TUI visibility; the logger also lands the
    record in the always-on buffer behind the Debug Console (installed before
    this module configures anything; see `__init__.py`).
    """
    print(f"Warning: {message}", file=sys.stderr)  # noqa: T201
    logger.warning("%s", message)


def _prepare_debug_file(path: Path) -> None:
    """Create or tighten a debug file before attaching the logging handler.

    On POSIX the file is created or tightened to mode `0o600`. On Windows,
    where `os.open` mode bits and `chmod` do not tighten the DACL, the DACL is
    replaced with one granting read and write access to the current user only.

    `O_NOFOLLOW` refuses a symlink at `path`. The default location is a
    world-writable temp directory, so without it a planted symlink could
    redirect captured MCP server stderr into a file of the attacker's choosing.

    Raises:
        OSError: If the file cannot be created, opened, or tightened. The
            caller must treat this as fatal to file logging.
    """  # noqa: DOC502 - raised by os.open/fchmod, not by an explicit raise
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        if os.name == "nt":
            _set_windows_owner_only_dacl(path)
            return
        fchmod = getattr(os, "fchmod", None)
        if fchmod is None:
            path.chmod(0o600)
        else:
            fchmod(fd, 0o600)
    finally:
        os.close(fd)


def _set_windows_owner_only_dacl(path: Path) -> None:
    """Restrict `path` to the current user on Windows.

    This is a no-op on POSIX, where `_prepare_debug_file` uses mode `0o600`
    instead. The Windows implementation (defined only when `os.name == "nt"`)
    replaces the file's DACL with one granting read and write access to the
    current user and no one else.

    Args:
        path: Debug log file to lock down.

    Raises:
        OSError: If the DACL cannot be built or applied. `_prepare_debug_file`
            propagates it; `configure_debug_logging` catches it and disables
            file logging.
    """  # noqa: DOC502 - raised by the callee, not by an explicit raise
    if os.name != "nt":
        return
    _apply_windows_owner_only_dacl(path)


if os.name == "nt":
    # --- Windows user-only DACL ---------------------------------------------
    # Structures and helpers mirroring the advapi32 API used to build and apply
    # a DACL granting the current user read and write access, and no one else
    # any access. `DELETE` and `WRITE_DAC` are deliberately not granted; the
    # file owner retains them implicitly.

    _SE_FILE_OBJECT = 1
    _DACL_SECURITY_INFORMATION = 0x00000004
    _PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
    _TOKEN_QUERY = 0x0008
    _TOKEN_USER_INFORMATION_CLASS = 1
    _FILE_GENERIC_READ = 0x120089
    _FILE_GENERIC_WRITE = 0x120116
    # `TRUSTEE_FORM` / `TRUSTEE_TYPE` / `ACCESS_MODE` from `accctrl.h`. Named
    # rather than inlined because all three enums start at 0 with unrelated
    # meanings, so a transposed literal still compiles and is rejected only at
    # runtime by `SetEntriesInAclW`.
    _NO_MULTIPLE_TRUSTEE = 0
    _TRUSTEE_IS_SID = 0
    _TRUSTEE_IS_USER = 1
    _SET_ACCESS = 2
    _NO_INHERITANCE = 0

    class _TRUSTEE_W(ctypes.Structure):  # noqa: N801  # mirrors Win32 TRUSTEE_W
        """`TRUSTEE_W` identifying the current-user SID to `SetEntriesInAclW`."""

        _fields_ = [
            ("pMultipleTrustee", ctypes.c_void_p),
            ("MultipleTrusteeOperation", ctypes.c_int),
            ("TrusteeForm", ctypes.c_int),
            ("TrusteeType", ctypes.c_int),
            ("ptstrName", ctypes.c_void_p),
        ]

    class _EXPLICIT_ACCESS_W(ctypes.Structure):  # noqa: N801  # mirrors Win32 type
        """`EXPLICIT_ACCESS_W` describing one access-control entry."""

        _fields_ = [
            ("grfAccessPermissions", wintypes.DWORD),
            ("grfAccessMode", ctypes.c_int),
            ("grfInheritance", wintypes.DWORD),
            ("Trustee", _TRUSTEE_W),
        ]

    def _get_current_user_sid() -> ctypes.c_void_p:
        """Return a pointer to the current user's SID.

        The `TOKEN_USER` buffer the SID points into is attached to the returned
        pointer as `_buffer`, so it stays alive for the DACL construction.
        `ctypes` already retains it through `.contents`; the attribute makes
        that guarantee explicit rather than incidental.

        Returns:
            A pointer to the current user's SID.

        Raises:
            OSError: If the process token or user SID cannot be read. Raised
                via `ctypes.WinError`, which is a factory returning `OSError`.
        """  # noqa: DOC501, DOC502 - `ctypes.WinError` returns an `OSError`
        advapi32 = ctypes.windll.advapi32
        token = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(
            ctypes.windll.kernel32.GetCurrentProcess(),
            _TOKEN_QUERY,
            ctypes.byref(token),
        ):
            raise ctypes.WinError()  # surface the raw OS error
        try:
            needed = wintypes.DWORD(0)
            advapi32.GetTokenInformation(
                token, _TOKEN_USER_INFORMATION_CLASS, None, 0, ctypes.byref(needed)
            )
            if not needed.value:
                raise ctypes.WinError()
            buffer = (ctypes.c_byte * needed.value)()
            if not advapi32.GetTokenInformation(
                token,
                _TOKEN_USER_INFORMATION_CLASS,
                buffer,
                needed,
                ctypes.byref(needed),
            ):
                raise ctypes.WinError()
            # TOKEN_USER begins with a single pointer to the user's SID.
            sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p)).contents
            # Keep the backing buffer alive by attaching it to the pointer object.
            sid._buffer = buffer  # type: ignore[attr-defined]
            return sid
        finally:
            ctypes.windll.kernel32.CloseHandle(token)

    def _apply_windows_owner_only_dacl(path: Path) -> None:
        """Replace `path`'s DACL with a single read/write entry for this user.

        The DACL is marked protected, so entries inherited from the parent
        directory are dropped rather than merged.

        Args:
            path: Debug log file to lock down.

        Raises:
            OSError: If the DACL cannot be built or applied. Raised via
                `ctypes.WinError`, which is a factory returning `OSError`.
        """  # noqa: DOC501, DOC502 - `ctypes.WinError` returns an `OSError`
        advapi32 = ctypes.windll.advapi32
        sid = _get_current_user_sid()

        trustee = _TRUSTEE_W(
            pMultipleTrustee=None,
            MultipleTrusteeOperation=_NO_MULTIPLE_TRUSTEE,
            TrusteeForm=_TRUSTEE_IS_SID,
            TrusteeType=_TRUSTEE_IS_USER,
            ptstrName=ctypes.cast(sid, ctypes.c_void_p).value,
        )
        explicit = _EXPLICIT_ACCESS_W(
            grfAccessPermissions=_FILE_GENERIC_READ | _FILE_GENERIC_WRITE,
            grfAccessMode=_SET_ACCESS,
            grfInheritance=_NO_INHERITANCE,
            Trustee=trustee,
        )
        new_acl = ctypes.c_void_p()
        result = advapi32.SetEntriesInAclW(
            1, ctypes.byref(explicit), None, ctypes.byref(new_acl)
        )
        if result != 0:  # ERROR_SUCCESS
            raise ctypes.WinError(result)
        try:
            apply_result = advapi32.SetNamedSecurityInfoW(
                str(path),
                _SE_FILE_OBJECT,
                _DACL_SECURITY_INFORMATION | _PROTECTED_DACL_SECURITY_INFORMATION,
                None,
                None,
                new_acl,
                None,
            )
            if apply_result != 0:  # ERROR_SUCCESS
                raise ctypes.WinError(apply_result)
        finally:
            ctypes.windll.kernel32.LocalFree(new_acl)


def resolve_log_level(*, debug_enabled: bool | None = None) -> int:
    """Resolve the configured runtime logging level.

    Args:
        debug_enabled: Whether `DEEPAGENTS_CODE_DEBUG` is truthy. When omitted,
            the current environment is checked.

    Returns:
        A standard `logging` level integer. Defaults to `DEBUG` when debug file
        logging is enabled and `INFO` otherwise.
    """
    if debug_enabled is None:
        debug_enabled = is_env_truthy(DEBUG)
    fallback = logging.DEBUG if debug_enabled else logging.INFO
    raw = os.environ.get(LOG_LEVEL)
    if raw is None or not raw.strip():
        return fallback
    level = LOG_LEVELS.get(raw.strip().upper())
    if level is not None:
        return level
    valid = ", ".join(LOG_LEVELS)
    message = f"ignoring invalid {LOG_LEVEL}={raw!r}; expected one of {valid}"
    _warn(message)
    return fallback


def _prepare_debug_directory(path: Path) -> None:
    """Create or tighten the debug directory to owner-only access.

    Raises:
        OSError: If the directory cannot be created, opened, or tightened.
    """
    with contextlib.suppress(FileExistsError):
        path.mkdir(mode=0o700)
    if os.name == "nt":
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            msg = f"debug log directory is not a real directory: {path}"
            raise OSError(msg)
        _set_windows_owner_only_dacl(path)
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        metadata = os.fstat(fd)
        if metadata.st_uid != os.geteuid():
            msg = f"debug log directory is not owned by the current user: {path}"
            raise OSError(msg)
        os.fchmod(fd, 0o700)
    finally:
        os.close(fd)


def _thread_log_name(thread_id: str) -> str:
    """Return a traversal-safe log filename for a thread identifier."""
    if (
        len(thread_id) <= _MAX_THREAD_FILENAME_LENGTH
        and _SAFE_THREAD_ID.fullmatch(thread_id)
        and thread_id not in {".", ".."}
    ):
        return f"{thread_id}.log"
    digest = hashlib.sha256(thread_id.encode()).hexdigest()[:16]
    return f"thread-{digest}.log"


def _remove_debug_handlers(
    target: logging.Logger, *, except_path: Path | None
) -> logging.FileHandler | None:
    """Remove stale tagged handlers.

    Returns:
        The handler for `except_path` that remains attached, or `None`.
    """
    kept: logging.FileHandler | None = None
    for existing in list(target.handlers):
        if not (
            isinstance(existing, logging.FileHandler)
            and getattr(existing, _DEBUG_HANDLER_ATTR, False)
        ):
            continue
        if except_path is not None and Path(existing.baseFilename) == except_path:
            kept = existing
            continue
        target.removeHandler(existing)
        existing.close()
    return kept


def _attach_debug_handler(target: logging.Logger, debug_path: Path, level: int) -> None:
    """Attach one secured debug handler to a configured logger."""
    if kept := _remove_debug_handlers(target, except_path=debug_path):
        kept.setLevel(level)
        return
    try:
        _prepare_debug_file(debug_path)
        handler = logging.FileHandler(str(debug_path), mode="a")
    except OSError as exc:
        _warn(f"could not secure or open debug log file {debug_path}: {exc}")
        return
    setattr(handler, _DEBUG_HANDLER_ATTR, True)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    target.addHandler(handler)


def configure_debug_logging(target: logging.Logger) -> None:
    """Configure runtime logging and register *target* for per-thread files."""
    debug_enabled = is_env_truthy(DEBUG)
    level = resolve_log_level(debug_enabled=debug_enabled)
    target.setLevel(level)
    _CONFIGURED_LOGGERS.add(target)

    if not debug_enabled:
        return
    if _ACTIVE_THREAD_ID is not None:
        bind_debug_logging_to_thread(_ACTIVE_THREAD_ID)


def _debug_directory() -> Path:
    """Return the configured directory, preserving legacy path overrides."""
    if directory := os.environ.get(DEBUG_DIRECTORY):
        return Path(directory)
    if legacy_file := os.environ.get(DEBUG_FILE):
        return Path(legacy_file).parent

    from deepagents_code.config_manifest import load_config_toml

    debug = load_config_toml().get("debug")
    if isinstance(debug, dict):
        if (directory := debug.get("directory")) and isinstance(directory, str):
            return Path(directory)
        if (legacy_file := debug.get("file")) and isinstance(legacy_file, str):
            return Path(legacy_file).parent
    return Path(DEFAULT_DEBUG_DIRECTORY)


def bind_debug_logging_to_thread(thread_id: str) -> None:
    """Route configured debug loggers to the active thread's log file."""
    global _ACTIVE_THREAD_ID  # noqa: PLW0603  # process-wide logging destination
    _ACTIVE_THREAD_ID = thread_id
    if not is_env_truthy(DEBUG):
        return
    directory = _debug_directory()
    try:
        _prepare_debug_directory(directory)
    except OSError as exc:
        for target in list(_CONFIGURED_LOGGERS):
            _remove_debug_handlers(target, except_path=None)
        _warn(f"could not secure debug log directory {directory}: {exc}")
        return
    debug_path = directory / _thread_log_name(thread_id)
    for target in list(_CONFIGURED_LOGGERS):
        _attach_debug_handler(target, debug_path, target.level)


def installed_debug_log_path() -> Path | None:
    """Return the path of the active debug log file, or `None` if not logging.

    Reflects the file handler actually attached by `configure_debug_logging`,
    not the current `DEEPAGENTS_CODE_DEBUG` env value. The two diverge when the
    variable is set after import — e.g. via a project/global `.env` loaded during
    settings bootstrap — in which case the variable reads truthy but no handler
    was installed and no log file exists. Callers that surface "full error in
    <path>" hints must use this rather than the env var to avoid pointing users
    at a file that was never created.
    """
    package_logger = logging.getLogger(__package__ or "deepagents_code")
    for handler in package_logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(
            handler, _DEBUG_HANDLER_ATTR, False
        ):
            return Path(handler.baseFilename)
    return None
