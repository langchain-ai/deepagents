"""Timeout handling: executor-side envelope, host-side kill, and fail-closed probe."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

import langchain_mxc.sandbox as sandbox_module
from langchain_mxc import MXCSandbox
from langchain_mxc._executor import ExecutorResult, ProbeResult, classify_result, run_request
from langchain_mxc.sandbox import TIMEOUT_EXIT_CODE

FAKE = Path(__file__).with_name("fake_executor.py")
FAKE_ARGV = [sys.executable, str(FAKE)]


def request(command: str, timeout_ms: int | None) -> dict:
    process: dict = {"commandLine": command, "cwd": os.getcwd(), "env": ["PATH=/usr/bin"]}
    if timeout_ms:
        process["timeout"] = timeout_ms
    return {"version": "1.0.0", "containment": "process", "process": process, "filesystem": {}, "network": {}, "ui": {}}


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        import subprocess  # noqa: PLC0415

        out = subprocess.run(  # noqa: S603 - fixed argv; the pid is an int
            ["tasklist", "/FI", f"PID eq {pid}"],  # noqa: S607 - system `tasklist`, resolved through PATH
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        # zombie check: a reaped child would raise above; an unreaped one is "alive" to the kernel
        return True


def test_fake_executor_dry_run_passes(tmp_path: Path) -> None:
    res = run_request(FAKE_ARGV, request("echo", 1000), timeout=10, config_dir=tmp_path, extra_args=["--dry-run"])
    assert res.returncode == 0
    assert "validation passed" in res.stdout_text


def test_executor_side_timeout_envelope_is_classified(tmp_path: Path) -> None:
    res = run_request(FAKE_ARGV, request("FAKE_EXEC_TIMEOUT", 300), timeout=5, config_dir=tmp_path)
    assert res.returncode == 255
    assert res.timed_out is False
    outcome = classify_result(res)
    assert outcome.kind == "timeout"
    assert outcome.workload_stdout == "started\n"


def test_host_side_timeout_kills_the_whole_tree(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    started = time.monotonic()
    res = run_request(FAKE_ARGV, request(f"FAKE_HANG={pid_file}", 60_000), timeout=0.5, grace=0.5, config_dir=tmp_path)
    elapsed = time.monotonic() - started

    assert res.timed_out is True
    assert elapsed < 15, f"host-side kill took {elapsed:.1f}s"
    assert res.stdout_text.startswith("hanging")

    deadline = time.monotonic() + 5
    child_pid = int(pid_file.read_text())
    while time.monotonic() < deadline and _alive(child_pid):
        time.sleep(0.1)
    assert not _alive(child_pid), "grandchild survived the host-side kill"


def test_config_file_is_removed_after_run(tmp_path: Path) -> None:
    run_request(FAKE_ARGV, request("echo", 1000), timeout=10, config_dir=tmp_path)
    assert not list(tmp_path.glob("mxc-request-*.json"))


# ---------------------------------------------------------------------------
# MXCSandbox.execute with the executor layer stubbed out
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_executor(tmp_path: Path) -> Path:
    exe = tmp_path / "mxc-exec-dummy"
    exe.write_bytes(b"")
    return exe


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    return root


def test_sandbox_maps_host_timeout_to_124(monkeypatch: pytest.MonkeyPatch, root: Path, dummy_executor: Path) -> None:
    seen: dict = {}

    def fake_probe(*_args, **_kwargs) -> ProbeResult:
        return ProbeResult(ok=True)

    def fake_run(_executor, req, *, timeout, grace, config_dir, extra_args=()) -> ExecutorResult:  # noqa: ARG001
        seen["timeout"] = timeout
        seen["request_timeout_ms"] = req["process"]["timeout"]
        return ExecutorResult(stdout=b"partial\n", stderr=b"", returncode=None, timed_out=True)

    monkeypatch.setattr(sandbox_module, "probe_executor", fake_probe)
    monkeypatch.setattr(sandbox_module, "run_request", fake_run)
    sb = MXCSandbox(root_dir=root, executor_path=dummy_executor, timeout=7)

    resp = sb.execute("sleep 100")
    assert resp.exit_code == TIMEOUT_EXIT_CODE
    assert "timed out after 7 seconds" in resp.output
    assert seen == {"timeout": 7, "request_timeout_ms": 7000}

    resp = sb.execute("sleep 100", timeout=3)
    assert resp.exit_code == TIMEOUT_EXIT_CODE
    assert "timed out after 3 seconds (custom timeout)" in resp.output
    assert seen["request_timeout_ms"] == 3000


def test_sandbox_rejects_non_positive_timeouts(monkeypatch: pytest.MonkeyPatch, root: Path, dummy_executor: Path) -> None:
    with pytest.raises(ValueError, match="timeout must be positive"):
        MXCSandbox(root_dir=root, executor_path=dummy_executor, timeout=0)
    monkeypatch.setattr(sandbox_module, "probe_executor", lambda *a, **k: ProbeResult(ok=True))  # noqa: ARG005
    sb = MXCSandbox(root_dir=root, executor_path=dummy_executor)
    with pytest.raises(ValueError, match="timeout must be positive"):
        sb.execute("echo", timeout=-1)


def test_sandbox_fails_closed_when_probe_fails(monkeypatch: pytest.MonkeyPatch, root: Path, dummy_executor: Path) -> None:
    calls = {"probe": 0, "run": 0}

    def fake_probe(*_args, **_kwargs) -> ProbeResult:
        calls["probe"] += 1
        return ProbeResult(ok=False, reason="MXC rejected the sandbox policy on this host: bubblewrap not installed")

    def fake_run(*_args, **_kwargs) -> ExecutorResult:
        calls["run"] += 1
        return ExecutorResult(stdout=b"ran!\n", stderr=b"", returncode=0)

    monkeypatch.setattr(sandbox_module, "probe_executor", fake_probe)
    monkeypatch.setattr(sandbox_module, "run_request", fake_run)
    sb = MXCSandbox(root_dir=root, executor_path=dummy_executor)

    for _ in range(3):
        resp = sb.execute("echo hi")
        assert resp.exit_code == 1
        assert "not available on this host" in resp.output
        assert "bubblewrap not installed" in resp.output
        assert "never run outside the sandbox" in resp.output
        assert "ran!" not in resp.output

    assert calls == {"probe": 1, "run": 0}, "probe must run once and the command must never run"


def test_probe_runs_once_per_instance(monkeypatch: pytest.MonkeyPatch, root: Path, dummy_executor: Path) -> None:
    calls = {"probe": 0}

    def fake_probe(*_args, **_kwargs) -> ProbeResult:
        calls["probe"] += 1
        return ProbeResult(ok=True)

    monkeypatch.setattr(sandbox_module, "probe_executor", fake_probe)
    monkeypatch.setattr(sandbox_module, "run_request", lambda *a, **k: ExecutorResult(stdout=b"ok\n", stderr=b"", returncode=0))  # noqa: ARG005
    sb = MXCSandbox(root_dir=root, executor_path=dummy_executor)
    for _ in range(4):
        assert sb.execute("echo").output == "ok\n"
    assert calls["probe"] == 1

    other = MXCSandbox(root_dir=root, executor_path=dummy_executor)
    other.execute("echo")
    assert calls["probe"] == 2
    assert other.id != sb.id
    assert sb.id.startswith("mxc-")


def test_empty_command(monkeypatch: pytest.MonkeyPatch, root: Path, dummy_executor: Path) -> None:
    monkeypatch.setattr(sandbox_module, "probe_executor", lambda *a, **k: ProbeResult(ok=True))  # noqa: ARG005
    sb = MXCSandbox(root_dir=root, executor_path=dummy_executor)
    resp = sb.execute("")
    assert resp.exit_code == 1
    assert "non-empty string" in resp.output
