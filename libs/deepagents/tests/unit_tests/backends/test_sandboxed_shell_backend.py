"""Unit tests for SandboxedShellBackend (Landlock-confined shell).

Skipped automatically on Windows and on kernels without Landlock support
(ABI < 1), so the suite stays green everywhere while still being exercised
wherever Landlock is present.
"""

import ctypes
import ctypes.util
import sys
from pathlib import Path

import pytest

import deepagents.backends.sandboxed_shell as sandboxed_shell_module
from deepagents import backends as backends_pkg
from deepagents.backends.sandboxed_shell import SandboxedShellBackend


def _landlock_abi() -> int:
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        libc.syscall.restype = ctypes.c_long
        return libc.syscall(ctypes.c_long(444), 0, 0, 1)  # create_ruleset(VERSION)
    except Exception:  # noqa: BLE001
        return -1


pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or _landlock_abi() < 1,
    reason="requires Linux with Landlock support",
)

_SYS_ENV = {"PATH": "/usr/bin:/bin"}


@pytest.fixture
def session(tmp_path):
    (tmp_path / "inside.txt").write_text("SESSION DATA")
    return tmp_path


def _backend(session, **kwargs: object) -> SandboxedShellBackend:
    return SandboxedShellBackend(root_dir=str(session), virtual_mode=True, env=_SYS_ENV, **kwargs)


def test_reads_inside_session_allowed(session) -> None:
    r = _backend(session).execute("cat inside.txt")
    assert r.exit_code == 0
    assert "SESSION DATA" in r.output


def test_writes_inside_session_allowed(session) -> None:
    r = _backend(session).execute("echo hi > out.txt && cat out.txt")
    assert r.exit_code == 0
    assert "hi" in r.output
    assert (session / "out.txt").exists()


def test_read_outside_session_denied(session, tmp_path_factory) -> None:
    secret_dir = tmp_path_factory.mktemp("outside")
    secret = secret_dir / "secret.txt"
    secret.write_text("TOPSECRET")
    # extra_writable_roots=[] so /tmp (where pytest tmp lives) is NOT granted.
    r = _backend(session, extra_writable_roots=[]).execute(f"cat {secret}")
    assert r.exit_code != 0
    assert "Permission denied" in r.output
    assert "TOPSECRET" not in r.output


def test_sibling_session_denied(session, tmp_path_factory) -> None:
    """A second session dir must be unreadable from the first."""
    sibling = tmp_path_factory.mktemp("sibling")
    (sibling / "other.txt").write_text("OTHER SESSION")
    r = _backend(session, extra_writable_roots=[]).execute(f"cat {sibling}/other.txt")
    assert r.exit_code != 0
    assert "OTHER SESSION" not in r.output


def test_write_outside_session_denied(session) -> None:
    target = Path("/tmp/landlock_escape_probe")  # noqa: S108  # deliberately probing that writes outside root_dir are denied
    target.unlink(missing_ok=True)
    r = _backend(session, extra_writable_roots=[]).execute(f"echo pwn > {target}")
    assert r.exit_code != 0
    assert not target.exists()


def test_dev_null_and_urandom_work(session) -> None:
    r = _backend(session).execute("echo noise > /dev/null && head -c 4 /dev/urandom | wc -c")
    assert r.exit_code == 0
    assert "4" in r.output


def test_shell_features_and_child_processes(session) -> None:
    """Pipes work and spawned children inherit confinement (still run)."""
    r = _backend(session).execute("echo a b c | tr ' ' '\\n' | wc -l")
    assert r.exit_code == 0
    assert "3" in r.output
    child = _backend(session).execute('python3 -c "print(6*7)"')
    assert child.exit_code == 0
    assert "42" in child.output


def test_extra_writable_root_allows_write(session, tmp_path_factory) -> None:
    scratch = tmp_path_factory.mktemp("scratch")
    r = _backend(session, extra_writable_roots=[str(scratch)]).execute(f"echo ok > {scratch}/w.txt && cat {scratch}/w.txt")
    assert r.exit_code == 0
    assert (scratch / "w.txt").read_text().strip() == "ok"


def test_missing_helper_fails_closed(session, monkeypatch) -> None:
    monkeypatch.setattr(sandboxed_shell_module, "_HELPER_PATH", "/nonexistent/_landlock_exec.py")
    r = _backend(session).execute("echo should-not-run")
    assert r.exit_code == 127
    assert "should-not-run" not in r.output


def test_empty_command_defers_to_parent_contract(session) -> None:
    r = _backend(session).execute("")
    assert r.exit_code == 1
    assert "non-empty string" in r.output


def test_exported_from_backends_package() -> None:
    assert "SandboxedShellBackend" in backends_pkg.__all__
    assert backends_pkg.SandboxedShellBackend is SandboxedShellBackend
