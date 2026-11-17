"""Behaviour of `MXCSandbox` against the real MXC executor on this machine."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

from langchain_mxc import MXCSandbox
from langchain_mxc.sandbox import TIMEOUT_EXIT_CODE

pytestmark = pytest.mark.requires_mxc

IS_WINDOWS = sys.platform == "win32"
HOME = Path.home()


def q(path: os.PathLike[str] | str) -> str:
    """Quote a path for the sandbox shell of this OS."""
    return f'"{path}"' if IS_WINDOWS else shlex.quote(str(path))


def cat(path: os.PathLike[str] | str) -> str:
    return f"type {q(path)}" if IS_WINDOWS else f"cat {q(path)}"


def write(path: os.PathLike[str] | str, text: str) -> str:
    return f"echo {text}> {q(path)}" if IS_WINDOWS else f"echo {shlex.quote(text)} > {q(path)}"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    return root


@pytest.fixture
def sandbox(root: Path, executor: Path) -> MXCSandbox:
    return MXCSandbox(root_dir=root, executor_path=executor)


def test_id_and_probe(sandbox: MXCSandbox) -> None:
    assert sandbox.id.startswith("mxc-")
    probe = sandbox._ensure_probed()
    assert probe.ok, probe.reason
    assert probe.details.get("canary") == "blocked as expected"


def test_success_and_stderr_prefix(sandbox: MXCSandbox) -> None:
    cmd = "echo hello & echo oops 1>&2" if IS_WINDOWS else "echo hello; echo oops >&2"
    resp = sandbox.execute(cmd)
    assert resp.exit_code == 0
    assert "hello" in resp.output
    assert "[stderr] oops" in resp.output
    assert resp.truncated is False


def test_exit_code_passthrough(sandbox: MXCSandbox) -> None:
    resp = sandbox.execute("exit 3" if not IS_WINDOWS else "exit /b 3")
    assert resp.exit_code == 3
    assert resp.output.rstrip().endswith("Exit code: 3")


def test_write_inside_root_succeeds_and_persists(sandbox: MXCSandbox, root: Path) -> None:
    resp = sandbox.execute(write("note.txt", "persisted"))
    assert resp.exit_code == 0, resp.output
    assert (root / "note.txt").read_text().strip() == "persisted"
    # a second command (a fresh sandbox) sees the file
    resp = sandbox.execute(cat("note.txt"))
    assert resp.exit_code == 0
    assert "persisted" in resp.output
    # and the host file tools see it too
    assert sandbox.read("/note.txt").error is None


def test_cwd_is_root(sandbox: MXCSandbox, root: Path) -> None:
    resp = sandbox.execute("cd" if IS_WINDOWS else "pwd")
    assert resp.exit_code == 0
    assert os.path.realpath(root) in resp.output or str(root) in resp.output


def test_write_outside_root_is_blocked(sandbox: MXCSandbox, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "escaped.txt"
    resp = sandbox.execute(write(target, "escaped"))
    assert resp.exit_code != 0, resp.output
    assert not target.exists(), "the sandbox wrote outside root_dir"
    assert "Blocked by sandbox policy" in resp.output
    assert "Ask the user to grant access" in resp.output


def test_write_to_home_is_blocked(sandbox: MXCSandbox) -> None:
    target = HOME / f"langchain-mxc-should-not-exist-{sandbox.id}.txt"
    try:
        resp = sandbox.execute(write(target, "escaped"))
        assert resp.exit_code != 0, resp.output
        assert not target.exists(), "the sandbox wrote into the home directory"
    finally:
        if target.exists():
            target.unlink()


@pytest.mark.skipif(not (HOME / ".ssh").is_dir(), reason="no ~/.ssh on this machine")
def test_reading_ssh_is_blocked(sandbox: MXCSandbox) -> None:
    ssh = HOME / ".ssh"
    names = sorted(p.name for p in ssh.iterdir())
    listing = f"dir /b {q(ssh)}" if IS_WINDOWS else f"ls -la {q(ssh)}"
    resp = sandbox.execute(listing)
    assert resp.exit_code != 0, resp.output
    for name in names:
        assert name not in resp.output, f"{name} leaked from ~/.ssh"
    assert "Blocked by sandbox policy" in resp.output

    for candidate in ("config", "known_hosts", "id_ed25519", "id_rsa", "authorized_keys"):
        if (ssh / candidate).is_file():
            content = (ssh / candidate).read_text(errors="ignore").strip()
            resp = sandbox.execute(cat(ssh / candidate))
            assert resp.exit_code != 0
            if content:
                assert content.splitlines()[0] not in resp.output
            break


def test_tilde_points_at_scratch_not_real_home(sandbox: MXCSandbox) -> None:
    resp = sandbox.execute("echo %USERPROFILE%" if IS_WINDOWS else "echo $HOME")
    assert resp.exit_code == 0
    assert str(sandbox.scratch_dir) in resp.output
    assert str(HOME) + os.sep not in resp.output or str(sandbox.scratch_dir).startswith(str(HOME))


def test_scratch_is_writable(sandbox: MXCSandbox) -> None:
    cmd = "echo x> %TEMP%\\probe.txt && type %TEMP%\\probe.txt" if IS_WINDOWS else 'echo x > "$TMPDIR/probe.txt" && cat "$TMPDIR/probe.txt"'
    resp = sandbox.execute(cmd)
    assert resp.exit_code == 0, resp.output
    assert "x" in resp.output


# Plain HTTP on purpose: the test is about whether the sandbox lets a connection
# out, and HTTPS would also depend on the host's certificate store trusting
# whoever issued example.com's certificate that week.
PY_FETCH = "python -c \"import urllib.request; r = urllib.request.urlopen('http://example.com', timeout=10); print('status', r.status)\""


def test_network_blocked_when_off(sandbox: MXCSandbox) -> None:
    resp = sandbox.execute(PY_FETCH, timeout=30)
    assert resp.exit_code != 0, resp.output
    assert "status 200" not in resp.output
    assert "Blocked by sandbox policy" in resp.output
    assert "network" in resp.output


def test_network_works_when_on(root: Path, executor: Path, host_has_internet: bool) -> None:
    if not host_has_internet:
        pytest.skip("host has no internet access")
    sandbox = MXCSandbox(root_dir=root, executor_path=executor, allow_network=True)
    resp = sandbox.execute(PY_FETCH, timeout=30)
    assert resp.exit_code == 0, resp.output
    assert "status 200" in resp.output
    assert "Blocked by sandbox policy" not in resp.output


def test_timeout_returns_124(sandbox: MXCSandbox) -> None:
    # Python sleeps on every OS. (`ping` cannot be used on Windows: the sandbox blocks loopback, so it exits at once.)
    sleep = 'python -c "import time; time.sleep(30)"'
    cmd = f"echo start & {sleep} & echo end" if IS_WINDOWS else f"echo start; {sleep}; echo end"
    started = time.monotonic()
    resp = sandbox.execute(cmd, timeout=1)
    elapsed = time.monotonic() - started
    assert resp.exit_code == TIMEOUT_EXIT_CODE, resp.output
    assert "timed out after 1 seconds" in resp.output
    assert "end" not in resp.output.replace("timed out", "")
    assert elapsed < 15, f"timeout took {elapsed:.1f}s"


def test_python_available(sandbox: MXCSandbox) -> None:
    resp = sandbox.execute("python --version")
    assert resp.exit_code == 0, resp.output
    assert "Python 3." in resp.output


@pytest.mark.skipif(IS_WINDOWS, reason="on Windows git.exe cannot start while the UI is locked; see the two Windows tests below")
def test_git_available(sandbox: MXCSandbox) -> None:
    resp = sandbox.execute("git --version")
    assert resp.exit_code == 0, resp.output


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows-only behaviour of the UI lock")
def test_windows_git_with_ui_locked_gets_a_clear_hint(sandbox: MXCSandbox) -> None:
    resp = sandbox.execute("git --version")
    assert resp.exit_code != 0
    assert '{"error"' not in resp.output
    assert "Blocked by sandbox policy: window-system access" in resp.output, resp.output


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows-only behaviour of the UI lock")
def test_windows_git_runs_with_allow_ui(root: Path, executor: Path) -> None:
    sandbox = MXCSandbox(root_dir=root, executor_path=executor, allow_ui=True)
    resp = sandbox.execute("git --version")
    assert resp.exit_code == 0, resp.output
    assert "git version" in resp.output


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows-only: cmd's del is denied inside read-write folders")
def test_windows_del_is_denied_with_a_workaround_hint(sandbox: MXCSandbox, root: Path) -> None:
    resp = sandbox.execute("echo a> gone.txt && del gone.txt")
    assert resp.exit_code != 0
    assert "Note: on Windows the sandbox denies `del`" in resp.output, resp.output
    resp = sandbox.execute("python -c \"import os; os.remove('gone.txt')\"")
    assert resp.exit_code == 0, resp.output
    assert not (root / "gone.txt").exists()


def test_output_truncation(root: Path, executor: Path) -> None:
    sandbox = MXCSandbox(root_dir=root, executor_path=executor, max_output_bytes=2000)
    resp = sandbox.execute("python -c \"print('x' * 10000)\"")
    assert resp.exit_code == 0
    assert resp.truncated is True
    assert "Output truncated at 2000 bytes" in resp.output


def test_user_read_only_path_is_readable(tmp_path: Path, root: Path, executor: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "data.txt").write_text("shared data")
    sandbox = MXCSandbox(root_dir=root, executor_path=executor, read_only_paths=[shared])
    resp = sandbox.execute(cat(shared / "data.txt"))
    assert resp.exit_code == 0, resp.output
    assert "shared data" in resp.output
    resp = sandbox.execute(write(shared / "new.txt", "nope"))
    assert resp.exit_code != 0
    assert not (shared / "new.txt").exists()


def test_user_env_is_visible(root: Path, executor: Path) -> None:
    sandbox = MXCSandbox(root_dir=root, executor_path=executor, env={"MXC_DEMO_VAR": "hello-env"})
    resp = sandbox.execute("echo %MXC_DEMO_VAR%" if IS_WINDOWS else "echo $MXC_DEMO_VAR")
    assert "hello-env" in resp.output


def test_aexecute_concurrently(sandbox: MXCSandbox) -> None:
    async def run_all() -> list:
        return await asyncio.gather(*(sandbox.aexecute(f"echo job{i}") for i in range(5)))

    results = asyncio.run(run_all())
    assert [r.exit_code for r in results] == [0] * 5
    for i, r in enumerate(results):
        assert f"job{i}" in r.output


def test_ui_options_are_accepted_by_the_executor(root: Path, executor: Path) -> None:
    """A policy that opens the UI, clipboard and input injection still validates and runs."""
    sandbox = MXCSandbox(root_dir=root, executor_path=executor, allow_ui=True, clipboard="all", allow_input_injection=True)
    assert sandbox.build_request("echo", 1)["ui"] == {"disable": False, "clipboard": "all", "injection": True}
    resp = sandbox.execute("echo ui-ok", timeout=30)
    assert resp.exit_code == 0, resp.output
    assert "ui-ok" in resp.output


@pytest.mark.skipif(sys.platform != "darwin", reason="reads the Seatbelt profile that mxc-exec-mac prints with --debug")
def test_ui_options_change_the_seatbelt_profile(root: Path, executor: Path) -> None:
    """The executor turns the ui block into real profile rules: absent when locked, present when opened."""

    def profile(**kwargs: object) -> str:
        sandbox = MXCSandbox(root_dir=root, executor_path=executor, **kwargs)  # type: ignore[arg-type]
        cfg = root / f"req-{sandbox.id}.json"
        cfg.write_text(json.dumps(sandbox.build_request("true", 10)))
        proc = subprocess.run(  # noqa: S603 - the executor under test, with a config this test wrote
            [str(executor), "--config", str(cfg), "--debug"], capture_output=True, text=True, timeout=60, check=False
        )
        return proc.stdout + proc.stderr

    locked = profile()
    opened = profile(allow_ui=True, clipboard="all", allow_input_injection=True)

    pasteboard = '(allow mach-lookup (global-name "com.apple.pasteboard.1"))'
    assert pasteboard not in locked
    assert "ui enabled" not in locked
    assert pasteboard in opened
    assert "ui enabled" in opened
    assert "clipboard enabled" in opened
