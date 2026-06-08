#!/usr/bin/env python3
"""Self-contained Landlock exec wrapper for `SandboxedShellBackend`.

Run as a standalone subprocess (``python3 _landlock_exec.py ...``). It applies a
Linux Landlock filesystem ruleset to *itself* -- confining reads/writes to the
supplied roots -- and then ``execve``s the agent's command under ``/bin/sh -c``.
Because Landlock restrictions are inherited across ``exec`` and by all child
processes, and can only ever *add* restrictions, every process the shell spawns
stays confined.

Deliberately stdlib-only and designed to be invoked *by path* (not ``-m``) so it
never imports the ``deepagents`` package (which would pull in LangChain and slow
down every shell command). It is launched by
``deepagents.backends.sandboxed_shell.SandboxedShellBackend``.

Threat model: stop the shell from reading or writing files outside the granted
roots (e.g. ``.env`` files, sibling sessions, ``/etc`` secrets). It is a
filesystem boundary only -- it does not restrict network, and on Landlock ABI
< 3 it cannot gate ``truncate()`` on already-accessible files (acceptable:
confinement is about which paths are reachable, not in-place truncation).

Fails closed: if Landlock is unavailable (kernel too old / disabled) or any
mandatory rule cannot be installed, it exits non-zero WITHOUT running the
command, so a misconfiguration can never silently downgrade to an unconfined
shell.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import ctypes.util
import os
import sys
from typing import NoReturn

# x86_64 / arm64 share these syscall numbers for the Landlock syscalls.
_NR_CREATE_RULESET = 444
_NR_ADD_RULE = 445
_NR_RESTRICT_SELF = 446
_PR_SET_NO_NEW_PRIVS = 38
_LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
_LANDLOCK_RULE_PATH_BENEATH = 1

# Landlock ABI versions that unlock additional filesystem access-right bits.
_ABI_REFER = 2  # adds LANDLOCK_ACCESS_FS_REFER (reparent)
_ABI_TRUNCATE = 3  # adds LANDLOCK_ACCESS_FS_TRUNCATE
_ABI_IOCTL_DEV = 5  # adds LANDLOCK_ACCESS_FS_IOCTL_DEV

# Filesystem access-right bits (uapi/linux/landlock.h).
_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
# Read-only roots grant traversal + read + execute (enough to run interpreters
# and shared libraries), but no write/create/remove bits.
_RO_ACCESS = _FS_READ_FILE | _FS_READ_DIR | _FS_EXECUTE
# Device roots (/dev) grant read+write to existing nodes plus dir traversal, so
# shells can redirect to /dev/null, read /dev/urandom, use ttys, etc. -- but not
# execute or create/remove nodes.
_DEV_ACCESS = _FS_READ_FILE | _FS_WRITE_FILE | _FS_READ_DIR

_EXIT_SANDBOX_SETUP_FAILED = 127

_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)


class _RulesetAttr(ctypes.Structure):
    """``struct landlock_ruleset_attr`` -- only ``handled_access_fs`` is passed.

    The size matches the v1 layout, so the call is valid on every ABI that
    supports filesystem rules (v1+) regardless of network-rule support.
    """

    _fields_ = (("handled_access_fs", ctypes.c_uint64),)


class _PathBeneathAttr(ctypes.Structure):
    """``struct landlock_path_beneath_attr`` for ``LANDLOCK_RULE_PATH_BENEATH``."""

    _pack_ = 1
    _fields_ = (
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    )


def _syscall(*args: int) -> int:
    """Invoke ``syscall(2)`` with ``c_long`` arguments and return its result."""
    _libc.syscall.restype = ctypes.c_long
    return _libc.syscall(*[ctypes.c_long(a) for a in args])


def _abi_version() -> int:
    """Return the kernel's Landlock ABI version, or <= 0 if unavailable."""
    return _syscall(_NR_CREATE_RULESET, 0, 0, _LANDLOCK_CREATE_RULESET_VERSION)


def _fs_mask_for_abi(version: int) -> int:
    """Return the full set of filesystem rights the running kernel understands.

    Passing a bit the kernel does not support makes ``create_ruleset`` fail with
    ``EINVAL``, so the handled set must be clamped to the ABI version.
    """
    mask = (1 << 13) - 1  # v1: bits 0..12 (execute/read/write/remove/make-*)
    if version >= _ABI_REFER:
        mask |= 1 << 13  # REFER (reparent)
    if version >= _ABI_TRUNCATE:
        mask |= 1 << 14  # TRUNCATE
    if version >= _ABI_IOCTL_DEV:
        mask |= 1 << 15  # IOCTL_DEV
    return mask


def _add_path_rule(ruleset_fd: int, path: str, access: int) -> None:
    """Grant ``access`` to everything beneath ``path``. Raise ``OSError`` on failure."""
    fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        attr = _PathBeneathAttr(allowed_access=access, parent_fd=fd)
        rc = _syscall(
            _NR_ADD_RULE,
            ruleset_fd,
            _LANDLOCK_RULE_PATH_BENEATH,
            ctypes.addressof(attr),
            0,
        )
        if rc != 0:
            msg = f"landlock_add_rule failed for {path}"
            raise OSError(ctypes.get_errno(), msg)
    finally:
        os.close(fd)


def _die(message: str) -> NoReturn:
    """Write ``message`` to stderr and exit with the sandbox-setup failure code."""
    sys.stderr.write(f"[landlock-exec] {message}\n")
    sys.exit(_EXIT_SANDBOX_SETUP_FAILED)


def _add_optional_roots(ruleset_fd: int, roots: list[str], access: int, label: str) -> None:
    """Grant ``access`` to each existing root; skip absent ones, log other errors.

    Used for read-only and device roots, where a missing path (e.g. ``/lib64``
    on some layouts) should be silently skipped so the allowlist can stay
    generous across images.
    """
    for root in roots:
        try:
            _add_path_rule(ruleset_fd, root, access)
        except FileNotFoundError:
            continue
        except OSError as exc:
            sys.stderr.write(f"[landlock-exec] skipping {label} root {root!r}: {exc}\n")


def _apply_landlock(
    writable_roots: list[str],
    readonly_roots: list[str],
    device_roots: list[str],
) -> None:
    """Create a Landlock ruleset for the given roots and restrict this process."""
    version = _abi_version()
    if version < 1:
        _die(f"Landlock unavailable on this kernel (ABI {version}); refusing to run the command unconfined.")

    fs_mask = _fs_mask_for_abi(version)
    attr = _RulesetAttr(handled_access_fs=fs_mask)
    ruleset_fd = _syscall(_NR_CREATE_RULESET, ctypes.addressof(attr), ctypes.sizeof(attr), 0)
    if ruleset_fd < 0:
        _die(f"landlock_create_ruleset failed (errno={ctypes.get_errno()})")

    # Writable roots must exist -- a missing session dir is a hard error (the
    # workspace should have been prepared first).
    granted_writable = 0
    for root in writable_roots:
        try:
            _add_path_rule(ruleset_fd, root, fs_mask)
            granted_writable += 1
        except OSError as exc:
            _die(f"cannot grant writable root {root!r}: {exc}")
    if granted_writable == 0:
        _die("no writable roots were granted; refusing to run")

    _add_optional_roots(ruleset_fd, readonly_roots, _RO_ACCESS, "ro")
    _add_optional_roots(ruleset_fd, device_roots, _DEV_ACCESS, "dev")

    if _libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        _die(f"prctl(PR_SET_NO_NEW_PRIVS) failed (errno={ctypes.get_errno()})")
    if _syscall(_NR_RESTRICT_SELF, ruleset_fd, 0) != 0:
        _die(f"landlock_restrict_self failed (errno={ctypes.get_errno()})")
    os.close(ruleset_fd)


def main(argv: list[str]) -> int:
    """Parse args, apply the sandbox, then ``execve`` the command under ``/bin/sh``."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rw", action="append", default=[], dest="writable")
    parser.add_argument("--ro", action="append", default=[], dest="readonly")
    parser.add_argument("--dev", action="append", default=[], dest="device")
    parser.add_argument("--cmd-b64", required=True)
    args = parser.parse_args(argv)

    try:
        command = base64.b64decode(args.cmd_b64).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        _die(f"invalid --cmd-b64 payload: {exc}")

    _apply_landlock(args.writable, args.readonly, args.device)

    # Hand off to the shell so pipes/redirects/etc. behave exactly as the
    # unsandboxed backend. execve never returns on success.
    try:
        os.execv("/bin/sh", ["/bin/sh", "-c", command])  # noqa: S606  # confined shell, command from a trusted base64 arg
    except OSError as exc:  # pragma: no cover - only on a broken image
        _die(f"failed to exec /bin/sh: {exc}")
    return _EXIT_SANDBOX_SETUP_FAILED  # unreachable


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
