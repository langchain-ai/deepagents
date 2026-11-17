"""Output and denial mapping, driven by output observed from the real MXC executors.

No MXC executor is needed: `MXCSandbox` only requires `executor_path` to be an
existing file at construction time, and the probe runs lazily on `execute`.
Here the recorded stdout/stderr/exit code of each case is fed straight into
the response formatter.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

from langchain_mxc import MXCSandbox
from langchain_mxc._executor import ExecutorResult, classify_result
from langchain_mxc.sandbox import BLOCKED_HINT, TIMEOUT_EXIT_CODE

HOME = str(Path.home())
TIMEOUT_ENVELOPE = json.dumps({"error": {"code": "backend_error", "message": "script timed out after 2000ms"}})
SPAWN_ENVELOPE = json.dumps(
    {"error": {"code": "backend_error", "message": "failed to spawn sandboxed process: No such file or directory (os error 2)"}}
)
REQUEST_ERROR = "Request error\nConfiguration parse error: Invalid one-shot request: Invalid configuration at `bogusField`: unknown field `bogusField`, expected one of `$schema`, `_comment`, `version` at line 48 column 14\n"  # noqa: E501
BACKEND_UNAVAILABLE = "error: the 'bubblewrap' backend is not available on macOS\nScript code length: 49\nWorking directory: /x\nScript timeout: 10000\nContainer name: \n"  # noqa: E501


def result(stdout: str = "", stderr: str = "", returncode: int = 0, *, timed_out: bool = False) -> ExecutorResult:
    return ExecutorResult(stdout=stdout.encode(), stderr=stderr.encode(), returncode=returncode, timed_out=timed_out)


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


@pytest.fixture
def sandbox(root: Path, dummy_executor: Path) -> MXCSandbox:
    return MXCSandbox(root_dir=root, executor_path=dummy_executor)


def fmt(sandbox: MXCSandbox, res: ExecutorResult, *, timeout: float = 120, custom: bool = False):
    return sandbox._format_response(res, effective_timeout=timeout, custom_timeout=custom)


# ---------------------------------------------------------------------------
# classify_result: the three output classes
# ---------------------------------------------------------------------------


def test_classify_workload_success() -> None:
    outcome = classify_result(result("hello\nhello.txt\n", "to stderr\n", 0))
    assert outcome.kind == "workload"
    assert outcome.workload_stdout == "hello\nhello.txt\n"
    assert outcome.workload_stderr == "to stderr\n"


def test_classify_workload_nonzero_is_still_workload() -> None:
    assert classify_result(result("", "/bin/sh: nonexistent_cmd_xyz: command not found\n", 127)).kind == "workload"
    assert classify_result(result("", "", 44)).kind == "workload"


def test_classify_signal_death_without_envelope_is_workload() -> None:
    # `kill -9 $$` gives exit 255 and no envelope: it is the workload's own death.
    assert classify_result(result("pre\n", "", 255)).kind == "workload"


def test_classify_timeout_envelope() -> None:
    outcome = classify_result(result("started\n", TIMEOUT_ENVELOPE + "\n", 255))
    assert outcome.kind == "timeout"
    assert outcome.code == "backend_error"
    assert outcome.workload_stdout == "started\n"
    assert outcome.workload_stderr == ""


def test_classify_envelope_after_workload_stderr() -> None:
    outcome = classify_result(result("started\n", "warning from tool\n" + TIMEOUT_ENVELOPE + "\n", 255))
    assert outcome.kind == "timeout"
    assert outcome.workload_stderr == "warning from tool\n"


def test_classify_spawn_failure_envelope() -> None:
    outcome = classify_result(result("", SPAWN_ENVELOPE + "\n", 255))
    assert outcome.kind == "backend_error"
    assert "failed to spawn" in outcome.message


def test_classify_request_rejected() -> None:
    outcome = classify_result(result("", REQUEST_ERROR, 1))
    assert outcome.kind == "request_rejected"
    assert outcome.message.startswith("Invalid one-shot request")
    assert "bogusField" in outcome.message


def test_classify_backend_unavailable() -> None:
    outcome = classify_result(result("", BACKEND_UNAVAILABLE, 1))
    assert outcome.kind == "backend_unavailable"
    assert outcome.message == "the 'bubblewrap' backend is not available on macOS"


def test_classify_no_config() -> None:
    outcome = classify_result(result("", "Error: No config provided. Use a positional path, --config, or --config-base64\n", 1))
    assert outcome.kind == "backend_error"


def test_classify_host_timeout() -> None:
    assert classify_result(result("partial\n", "", None, timed_out=True)).kind == "host_timeout"


def test_classify_workload_printing_request_error_with_stdout_is_workload() -> None:
    # Only an empty stdout plus exit 1 counts as a rejection.
    assert classify_result(result("some output\n", "Request error\n", 1)).kind == "workload"


# ---------------------------------------------------------------------------
# ExecuteResponse formatting (LocalShellBackend conventions)
# ---------------------------------------------------------------------------


def test_success_combines_streams_with_stderr_prefix(sandbox: MXCSandbox) -> None:
    resp = fmt(sandbox, result("hello\nhello.txt\n", "to stderr\n", 0))
    assert resp.exit_code == 0
    assert resp.truncated is False
    assert resp.output == "hello\nhello.txt\n\n[stderr] to stderr"


def test_nonzero_exit_appends_exit_code(sandbox: MXCSandbox) -> None:
    resp = fmt(sandbox, result("before\n", "bad thing\n", 3))
    assert resp.exit_code == 3
    assert resp.output == "before\n\n[stderr] bad thing\n\nExit code: 3"


def test_no_output(sandbox: MXCSandbox) -> None:
    resp = fmt(sandbox, result("", "", 0))
    assert resp.output == "<no output>"
    assert resp.exit_code == 0


def test_command_not_found_has_no_policy_hint(sandbox: MXCSandbox) -> None:
    resp = fmt(sandbox, result("", "/bin/sh: nonexistent_cmd_xyz: command not found\n", 127))
    assert resp.exit_code == 127
    assert "Blocked by sandbox policy" not in resp.output


def test_truncation(root: Path, dummy_executor: Path) -> None:
    sb = MXCSandbox(root_dir=root, executor_path=dummy_executor, max_output_bytes=1000)
    resp = fmt(sb, result("x" * 5000, "", 0))
    assert resp.truncated is True
    assert resp.output.startswith("x" * 1000)
    assert "Output truncated at 1000 bytes" in resp.output


def test_executor_timeout_maps_to_124(sandbox: MXCSandbox) -> None:
    resp = fmt(sandbox, result("started\n", TIMEOUT_ENVELOPE + "\n", 255), timeout=2, custom=True)
    assert resp.exit_code == TIMEOUT_EXIT_CODE
    assert "started" in resp.output
    assert "timed out after 2 seconds (custom timeout)" in resp.output


def test_default_timeout_message(sandbox: MXCSandbox) -> None:
    resp = fmt(sandbox, result("", TIMEOUT_ENVELOPE + "\n", 255), timeout=120, custom=False)
    assert resp.exit_code == TIMEOUT_EXIT_CODE
    assert resp.output == "Error: Command timed out after 120 seconds. For long-running commands, re-run using the timeout parameter."


def test_host_timeout_maps_to_124(sandbox: MXCSandbox) -> None:
    resp = fmt(sandbox, result("partial\n", "", None, timed_out=True), timeout=5, custom=True)
    assert resp.exit_code == TIMEOUT_EXIT_CODE
    assert resp.output.startswith("partial")


def test_request_rejected_fails_closed_with_hint(sandbox: MXCSandbox) -> None:
    resp = fmt(sandbox, result("", REQUEST_ERROR, 1))
    assert resp.exit_code == 1
    assert "MXC rejected the sandbox policy" in resp.output
    assert "was not run" in resp.output
    assert "Blocked by sandbox policy" in resp.output


def test_backend_unavailable_fails_closed(sandbox: MXCSandbox) -> None:
    resp = fmt(sandbox, result("", BACKEND_UNAVAILABLE, 1))
    assert resp.exit_code == 1
    assert "cannot sandbox commands on this host" in resp.output
    assert "never run outside the sandbox" in resp.output


def test_spawn_failure_is_reported(sandbox: MXCSandbox) -> None:
    resp = fmt(sandbox, result("", SPAWN_ENVELOPE + "\n", 255))
    assert resp.exit_code == 1
    assert "MXC failed while running the command (backend_error)" in resp.output
    assert "failed to spawn" in resp.output


# ---------------------------------------------------------------------------
# Denial hints
# ---------------------------------------------------------------------------


def test_write_outside_root_gets_hint(sandbox: MXCSandbox) -> None:
    target = f"{HOME}/Desktop/mxc_should_fail.txt"
    resp = fmt(sandbox, result("", f"/bin/sh: {target}: Operation not permitted\n", 1))
    assert resp.exit_code == 1
    assert resp.output.rstrip().endswith(BLOCKED_HINT.format(what=target))


def test_read_ssh_gets_hint(sandbox: MXCSandbox) -> None:
    resp = fmt(sandbox, result("", f"ls: {HOME}/.ssh: Operation not permitted\n", 1))
    assert BLOCKED_HINT.format(what=f"{HOME}/.ssh") in resp.output


def test_multiple_denied_paths_are_listed_once(sandbox: MXCSandbox) -> None:
    stderr = f"ls: {HOME}/.ssh: Operation not permitted\ncat: {HOME}/.ssh/config: Operation not permitted\ncat: {HOME}/.ssh/config: Operation not permitted\n"  # noqa: E501
    resp = fmt(sandbox, result("", stderr, 1))
    assert resp.output.count("Blocked by sandbox policy") == 1
    assert f"{HOME}/.ssh, {HOME}/.ssh/config" in resp.output


def test_python_permission_error_with_path(sandbox: MXCSandbox) -> None:
    resp = fmt(sandbox, result("", f"PermissionError: [Errno 1] Operation not permitted: '{HOME}/Desktop/x.txt'\n", 1))
    assert BLOCKED_HINT.format(what=f"{HOME}/Desktop/x.txt") in resp.output


def test_python_permission_error_without_path_is_generic(sandbox: MXCSandbox) -> None:
    resp = fmt(
        sandbox,
        result(
            "", 'Traceback (most recent call last):\n  File "<string>", line 1, in <module>\nPermissionError: [Errno 1] Operation not permitted\n', 1
        ),
    )
    assert "Blocked by sandbox policy: a path outside the allowed folders or the network, which is off." in resp.output


def test_permission_denied_inside_root_is_not_a_policy_hint(sandbox: MXCSandbox, root: Path) -> None:
    # chmod 000 on a file the policy allows: an ordinary error, not a sandbox block.
    resp = fmt(sandbox, result("", f"cat: {root}/secret.txt: Permission denied\n", 1))
    assert "Blocked by sandbox policy" not in resp.output


@pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX paths, which a Windows host does not treat as system folders")
def test_denial_in_allowed_system_path_is_not_a_policy_hint(sandbox: MXCSandbox) -> None:
    resp = fmt(sandbox, result("", "/bin/sh: /usr/bin/x: Operation not permitted\n", 1))
    assert "Blocked by sandbox policy" not in resp.output


def test_network_denied_curl(sandbox: MXCSandbox) -> None:
    resp = fmt(sandbox, result("", "curl: (6) Could not resolve host: example.com\n", 6))
    assert resp.exit_code == 6
    assert BLOCKED_HINT.format(what="network") in resp.output


def test_network_denied_python(sandbox: MXCSandbox) -> None:
    resp = fmt(sandbox, result("ERR URLError <urlopen error [Errno 8] nodename nor servname provided, or not known>\n", "", 0))
    assert BLOCKED_HINT.format(what="network") in resp.output


def test_network_hint_not_added_when_network_allowed(root: Path, dummy_executor: Path) -> None:
    sb = MXCSandbox(root_dir=root, executor_path=dummy_executor, allow_network=True)
    resp = fmt(sb, result("", "curl: (6) Could not resolve host: example.com\n", 6))
    assert "Blocked by sandbox policy" not in resp.output


def test_path_and_network_in_one_hint(sandbox: MXCSandbox) -> None:
    stderr = f"ls: {HOME}/.aws: Operation not permitted\ncurl: (6) Could not resolve host: example.com\n"
    resp = fmt(sandbox, result("", stderr, 1))
    assert BLOCKED_HINT.format(what=f"{HOME}/.aws, network") in resp.output


def test_hint_comes_after_exit_code_line(sandbox: MXCSandbox) -> None:
    resp = fmt(sandbox, result("", f"ls: {HOME}/.ssh: Operation not permitted\n", 1))
    lines = resp.output.splitlines()
    assert lines[-2] == "Exit code: 1"
    assert lines[-1].startswith("Blocked by sandbox policy")


# ---------------------------------------------------------------------------
# Linux (bubblewrap), recorded 2026-11-12 in a Debian container
# ---------------------------------------------------------------------------

BWRAP_START_FAILURE = "bwrap: Can't mount proc on /newroot/proc: Operation not permitted\n"


def test_bwrap_start_failure_is_a_backend_error_not_a_policy_block(sandbox: MXCSandbox) -> None:
    res = result(stderr=BWRAP_START_FAILURE, returncode=1)
    outcome = classify_result(res)
    assert outcome.kind == "backend_error"
    assert outcome.code == "sandbox_start_failed"
    response = sandbox._format_response(res, effective_timeout=10, custom_timeout=False)
    assert response.exit_code == 1
    assert "MXC failed while running the command" in response.output
    assert "Blocked by sandbox policy" not in response.output


@pytest.mark.skipif(sys.platform == "win32", reason="simulates a Linux host with real POSIX paths")
def test_linux_hidden_path_gets_the_blocked_hint(sandbox: MXCSandbox, tmp_path: Path) -> None:
    sandbox._host = dataclasses.replace(sandbox._host, platform="linux")
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("x")

    read = sandbox._format_response(
        result(stderr=f"ls: cannot access '{secret}': No such file or directory\n", returncode=2), effective_timeout=10, custom_timeout=False
    )
    assert read.output.endswith(BLOCKED_HINT.format(what=str(secret)))

    new_file = outside / "new.txt"
    write = sandbox._format_response(
        result(stderr=f"touch: cannot touch '{new_file}': No such file or directory\n", returncode=1), effective_timeout=10, custom_timeout=False
    )
    assert write.output.endswith(BLOCKED_HINT.format(what=str(new_file)))

    redirect = sandbox._format_response(
        result(stderr=f"/bin/sh: 1: cannot create {new_file}: Directory nonexistent\n", returncode=2), effective_timeout=10, custom_timeout=False
    )
    assert redirect.output.endswith(BLOCKED_HINT.format(what=str(new_file)))


def test_linux_genuinely_missing_file_inside_the_project_gets_no_hint(sandbox: MXCSandbox, root: Path) -> None:
    sandbox._host = dataclasses.replace(sandbox._host, platform="linux")
    missing = root / "nope.txt"
    response = sandbox._format_response(
        result(stderr=f"cat: {missing}: No such file or directory\n", returncode=1), effective_timeout=10, custom_timeout=False
    )
    assert "Blocked by sandbox policy" not in response.output


def test_missing_file_wording_is_not_treated_as_a_block_off_linux(sandbox: MXCSandbox, tmp_path: Path) -> None:
    sandbox._host = dataclasses.replace(sandbox._host, platform="darwin")
    response = sandbox._format_response(
        result(stderr=f"cat: {tmp_path}/gone.txt: No such file or directory\n", returncode=1), effective_timeout=10, custom_timeout=False
    )
    assert "Blocked by sandbox policy" not in response.output


WINDOWS_UI_MESSAGE = (
    "The target executable exited with STATUS_DLL_INIT_FAILED (0xC0000142). This often means the sandbox is blocking "
    "Win32k system calls (UI subsystem access), which is often required to initialize."
)


def windows_ui_failure() -> ExecutorResult:
    # Observed on Windows 11 build 28000 with wxc-exec 1.0.1: message and envelope share one line, exit code is the NTSTATUS.
    envelope = json.dumps({"error": {"code": "backend_error", "message": WINDOWS_UI_MESSAGE}})
    return result("", WINDOWS_UI_MESSAGE + envelope + "\r\n", 0xC0000142)


def test_windows_same_line_envelope_is_a_backend_error_with_a_ui_hint(sandbox: MXCSandbox) -> None:
    res = windows_ui_failure()
    outcome = classify_result(res)
    assert outcome.kind == "backend_error"
    assert outcome.code == "backend_error"
    assert outcome.workload_stderr == ""
    resp = fmt(sandbox, res)
    assert resp.exit_code == 1
    assert '{"error"' not in resp.output
    assert resp.output.count("STATUS_DLL_INIT_FAILED") == 1
    assert "Blocked by sandbox policy: window-system access" in resp.output
    assert "allow_ui=False" in resp.output


def test_workload_json_error_on_stderr_is_not_an_executor_failure(sandbox: MXCSandbox) -> None:
    res = result("", '{"error": {"code": "not_found", "message": "no such thing"}}\n', 1)
    assert classify_result(res).kind == "workload"
    resp = fmt(sandbox, res)
    assert resp.exit_code == 1
    assert "MXC failed" not in resp.output


def test_crlf_output_is_normalised(sandbox: MXCSandbox) -> None:
    resp = fmt(sandbox, result("one\r\ntwo\r\n", "", 0))
    assert "\r" not in resp.output
    assert "one\ntwo" in resp.output


def test_windows_crash_with_ui_locked_gets_the_ui_hint(sandbox: MXCSandbox) -> None:
    sandbox._host = dataclasses.replace(sandbox._host, platform="win32")
    crash = fmt(sandbox, result("", "Windows fatal exception: code 0xc06d007e\n", 0xC06D007E))
    assert "Blocked by sandbox policy: window-system access" in crash.output
    dll = fmt(
        sandbox, result("", "ImportError: DLL load failed while importing _ctypes: A dynamic link library (DLL) initialization routine failed.\n", 1)
    )
    assert "Blocked by sandbox policy: window-system access" in dll.output


def test_ordinary_failure_on_windows_gets_no_ui_hint(sandbox: MXCSandbox) -> None:
    sandbox._host = dataclasses.replace(sandbox._host, platform="win32")
    resp = fmt(sandbox, result("", "1 failed, 2 passed\n", 1))
    assert "Blocked by sandbox policy" not in resp.output


def test_windows_del_denied_gets_the_delete_hint(sandbox: MXCSandbox) -> None:
    from langchain_mxc.sandbox import WINDOWS_DELETE_HINT  # noqa: PLC0415

    sandbox._host = dataclasses.replace(sandbox._host, platform="win32")
    for command in ("del notes.txt", "echo a> x.txt && del /q x.txt", 'powershell -Command "Remove-Item x.txt"'):
        resp = sandbox._format_response(result("", "Access is denied.\n", 1), effective_timeout=10, custom_timeout=False, command=command)
        assert resp.output.endswith(WINDOWS_DELETE_HINT), command
        assert "outside the allowed folders" not in resp.output


def test_windows_read_denied_keeps_the_policy_hint(sandbox: MXCSandbox) -> None:
    sandbox._host = dataclasses.replace(sandbox._host, platform="win32")
    resp = sandbox._format_response(
        result("", "Access is denied.\n", 1), effective_timeout=10, custom_timeout=False, command='type "C:\\Users\\me\\secret.txt"'
    )
    assert "Blocked by sandbox policy: a path outside the allowed folders" in resp.output
    assert "Note: on Windows the sandbox denies `del`" not in resp.output


def test_del_hint_is_windows_only(sandbox: MXCSandbox) -> None:
    sandbox._host = dataclasses.replace(sandbox._host, platform="darwin")
    resp = sandbox._format_response(result("", "Access is denied.\n", 1), effective_timeout=10, custom_timeout=False, command="del x")
    assert "Note: on Windows" not in resp.output
