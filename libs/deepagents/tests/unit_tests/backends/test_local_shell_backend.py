"""Unit tests for LocalShellBackend."""

import asyncio
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, suppress
from contextvars import ContextVar
from itertools import chain, repeat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import deepagents.backends.local_shell as local_shell_module
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

_POSIX_SHELL_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="test requires POSIX shell behavior")


@pytest.fixture(autouse=True)
def _reset_background_workers() -> Iterator[None]:
    """Keep the module-global worker set from leaking between tests.

    `_BACKGROUND_WORKERS` is process-wide. A worker left behind by a failing test
    makes a later assertion on the set fail in an unrelated place.
    """
    local_shell_module._BACKGROUND_WORKERS.clear()
    yield
    local_shell_module._BACKGROUND_WORKERS.clear()


def _as_posix() -> AbstractContextManager[bool]:
    """Select the POSIX cleanup branch for the duration of a test.

    Patching the module constant keeps the choice local. Patching `sys.platform`
    instead would change it for every library in the process, which breaks
    anything that resolves platform-specific behavior while the test runs.
    """
    return patch.object(local_shell_module, "_IS_WINDOWS", new=False)


def _as_windows() -> AbstractContextManager[bool]:
    """Select the Windows cleanup branch for the duration of a test."""
    return patch.object(local_shell_module, "_IS_WINDOWS", new=True)


def _heartbeat_command(directory: Path) -> tuple[str, Path, Path]:
    """Build a shell command whose background child updates a heartbeat."""
    heartbeat = directory / "heartbeat"
    pid_file = directory / "child.pid"
    script = (
        f'i=0; while :; do i=$((i + 1)); printf "%s" "$i" > {shlex.quote(str(heartbeat))}; '
        f"sleep 0.02; done & echo $! > {shlex.quote(str(pid_file))}; wait"
    )
    return f"sh -c {shlex.quote(script)}", pid_file, heartbeat


def _wait_for_file(path: Path, timeout: float = 2) -> bool:
    """Wait for a subprocess to create a synchronization file."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.01)
    return path.exists()


def _assert_heartbeat_stopped(heartbeat: Path) -> None:
    """Assert that a descendant is no longer updating its heartbeat."""
    time.sleep(0.05)
    stopped_value = heartbeat.read_text()
    time.sleep(0.1)
    assert heartbeat.read_text() == stopped_value


def _stop_test_descendant(pid_file: Path) -> None:
    """Best-effort cleanup if a descendant-reaping assertion fails."""
    if not pid_file.exists():
        return
    with suppress(ProcessLookupError):
        os.kill(int(pid_file.read_text()), signal.SIGKILL)


def _assert_posix_cleanup(process: MagicMock, killpg: MagicMock) -> None:
    """Assert cleanup killed the POSIX process group and reaped the shell.

    Callers patch `_IS_WINDOWS` to `False`, so this runs on every platform rather
    than testing only whichever branch the host happens to take.
    """
    killpg.assert_called_once_with(1234, signal.SIGKILL)
    process.kill.assert_not_called()
    process.wait.assert_called_once_with(timeout=local_shell_module._PROCESS_REAP_TIMEOUT)


def _execute_controlling_terminal_probe(directory: Path, result_file: Path) -> None:
    """Run the backend probe after proving this process owns `/dev/tty`."""
    descriptor = os.open("/dev/tty", os.O_RDONLY)
    os.close(descriptor)
    result = LocalShellBackend(root_dir=directory).execute(": </dev/tty")
    result_file.write_text(f"{result.exit_code}\n{result.output}", encoding="utf-8")


def _run_controlling_terminal_probe(directory: Path) -> tuple[int, str]:
    """Run the backend inside a child that owns a real controlling terminal."""
    pty = pytest.importorskip("pty")
    result_file = directory / "tty-result"
    child_id, terminal = pty.fork()
    if child_id == 0:  # pragma: no cover - assertions run in the parent process
        try:
            _execute_controlling_terminal_probe(directory, result_file)
        except BaseException as error:  # noqa: BLE001  # Report child setup failures to the parent.
            result_file.write_text(f"harness error: {error}", encoding="utf-8")
            os._exit(1)
        os._exit(0)
    try:
        _, status = os.waitpid(child_id, 0)
    finally:
        os.close(terminal)
    details = result_file.read_text(encoding="utf-8")
    assert os.waitstatus_to_exitcode(status) == 0, details
    exit_code, output = details.split("\n", 1)
    return int(exit_code), output


def test_local_shell_backend_initialization() -> None:
    """Test that LocalShellBackend initializes correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir)

        assert backend.cwd == Path(tmpdir).resolve()
        assert backend.id.startswith("local-")
        assert len(backend.id) == 14  # "local-" + 8 hex chars


def test_local_shell_backend_execute_simple_command() -> None:
    """Test executing a simple shell command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, inherit_env=True)

        result = backend.execute("echo 'Hello World'")

        assert isinstance(result, ExecuteResponse)
        assert result.exit_code == 0
        assert "Hello World" in result.output
        assert result.truncated is False


def test_local_shell_backend_execute_configures_session_for_platform() -> None:
    """Test that only POSIX commands start in a new session."""
    process = MagicMock(returncode=0)
    process.communicate.return_value = ("hello\n", "")
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir)
        with patch.object(local_shell_module, "WindowsProcessReader", return_value=process), patch("subprocess.Popen", return_value=process) as popen:
            backend.execute("echo hello")

    assert popen.call_args.kwargs["start_new_session"] == (sys.platform != "win32")


@_POSIX_SHELL_ONLY
def test_local_shell_backend_execute_process_is_group_leader() -> None:
    """Test the real shell process leads its detached process group."""
    probe = "import os; parent = os.getppid(); print(parent, os.getpgid(parent))"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(probe)}; :"
    with tempfile.TemporaryDirectory() as tmpdir:
        result = LocalShellBackend(root_dir=tmpdir, inherit_env=True).execute(command)

    assert result.exit_code == 0
    process_id, process_group = (int(value) for value in result.output.split())
    assert process_id == process_group
    assert process_group != os.getpgrp()


@_POSIX_SHELL_ONLY
def test_local_shell_backend_timeout_stops_descendant(tmp_path: Path) -> None:
    """Test a real background descendant stops after command timeout."""
    command, pid_file, heartbeat = _heartbeat_command(tmp_path)
    try:
        result = LocalShellBackend(root_dir=tmp_path, inherit_env=True).execute(command, timeout=1)
        assert result.exit_code == 124
        assert _wait_for_file(pid_file)
        assert _wait_for_file(heartbeat)
        _assert_heartbeat_stopped(heartbeat)
    finally:
        _stop_test_descendant(pid_file)


def test_local_shell_backend_interrupt_cleans_up_posix_process_group() -> None:
    """Test an interrupt kills the command's POSIX process group."""
    process = MagicMock(pid=1234)
    process.communicate.side_effect = KeyboardInterrupt
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        _as_posix(),
        patch.object(local_shell_module, "WindowsProcessReader", return_value=process),
        patch("subprocess.Popen", return_value=process),
        patch.object(local_shell_module.os, "killpg", create=True) as killpg,
        pytest.raises(KeyboardInterrupt),
    ):
        LocalShellBackend(root_dir=tmpdir).execute("sleep 10")

    _assert_posix_cleanup(process, killpg)


def test_local_shell_backend_polling_interrupt_kills_process_group() -> None:
    """Test an interrupt in the cancellation-aware polling loop cleans up."""
    process = MagicMock(pid=1234)
    process.communicate.side_effect = KeyboardInterrupt
    with (
        patch.object(local_shell_module, "WindowsProcessReader", return_value=process),
        patch.object(local_shell_module, "_kill_and_reap") as kill_and_reap,
        pytest.raises(KeyboardInterrupt),
    ):
        local_shell_module._communicate(
            process,
            10,
            threading.Event(),
            process_group=1234,
        )

    kill_and_reap.assert_called_once_with(process, 1234)


def test_local_shell_backend_polling_deadline_kills_process_group() -> None:
    """Test the cancellation-aware polling loop enforces its deadline."""
    process = MagicMock(pid=1234)
    with (
        patch.object(local_shell_module.time, "monotonic", side_effect=chain([0], repeat(2))),
        patch.object(local_shell_module, "WindowsProcessReader", return_value=process),
        patch.object(local_shell_module, "_kill_and_reap") as kill_and_reap,
        pytest.raises(subprocess.TimeoutExpired),
    ):
        local_shell_module._communicate(
            process,
            1,
            threading.Event(),
            process_group=1234,
        )

    kill_and_reap.assert_called_once_with(process, 1234)


def test_local_shell_backend_cleanup_without_process_group_kills_process() -> None:
    """Test cleanup falls back to the direct process without a group."""
    process = MagicMock(pid=1234)
    assert local_shell_module._kill_and_reap(process, None) is True
    process.kill.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=local_shell_module._PROCESS_REAP_TIMEOUT)


def test_local_shell_backend_cleanup_accepts_missing_pipes() -> None:
    """Test cleanup accepts processes without captured output pipes."""
    local_shell_module._close_pipe(None, "stdout", 1234)


def test_local_shell_backend_cancelled_background_worker_is_released() -> None:
    """Test a cancelled background worker is dropped without reading its result."""
    worker = MagicMock()
    worker.cancelled.return_value = True
    local_shell_module._BACKGROUND_WORKERS.add(worker)
    local_shell_module._release_background_worker(worker, backend_id="local-test", command="echo hi")
    assert worker not in local_shell_module._BACKGROUND_WORKERS
    worker.result.assert_not_called()


async def test_local_shell_backend_late_cooperative_cancellation_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    worker = asyncio.get_running_loop().create_future()
    worker.set_exception(asyncio.CancelledError())
    local_shell_module._BACKGROUND_WORKERS.add(worker)
    local_shell_module._release_background_worker(worker, backend_id="local-test", command="echo hi")
    assert worker not in local_shell_module._BACKGROUND_WORKERS
    assert not caplog.records


def test_local_shell_backend_failed_background_worker_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Test a late background worker failure is consumed and diagnosed."""
    worker = MagicMock()
    worker.cancelled.return_value = False
    worker.result.side_effect = RuntimeError("backend exploded")
    local_shell_module._BACKGROUND_WORKERS.add(worker)
    with caplog.at_level("WARNING", logger="deepagents.backends.local_shell"):
        local_shell_module._release_background_worker(worker, backend_id="local-test", command="echo hi")

    assert worker not in local_shell_module._BACKGROUND_WORKERS
    assert "failed on backend" in caplog.text
    assert "after its caller was cancelled" in caplog.text
    assert "local-test" in caplog.text
    assert "echo hi" in caplog.text


@_POSIX_SHELL_ONLY
def test_local_shell_backend_cleanup_errors_preserve_interrupt(caplog: pytest.LogCaptureFixture) -> None:
    """Test that cleanup failures cannot replace an active interrupt."""
    process = MagicMock(pid=1234)
    process.communicate.side_effect = KeyboardInterrupt
    process.kill.side_effect = PermissionError
    process.wait.side_effect = OSError
    process.stdout.close.side_effect = OSError
    process.stderr.close.side_effect = OSError
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        _as_posix(),
        patch.object(local_shell_module, "WindowsProcessReader", return_value=process),
        patch("subprocess.Popen", return_value=process),
        patch("os.killpg", side_effect=PermissionError),
        caplog.at_level("WARNING", logger="deepagents.backends.local_shell"),
        pytest.raises(KeyboardInterrupt),
    ):
        LocalShellBackend(root_dir=tmpdir).execute("sleep 10")

    process.kill.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=local_shell_module._PROCESS_REAP_TIMEOUT)
    process.stdout.close.assert_called_once_with()
    process.stderr.close.assert_called_once_with()
    assert "Failed to terminate local shell process group 1234" in caplog.text
    assert "Failed to terminate local shell process 1234" in caplog.text
    assert "Failed to reap local shell process 1234" in caplog.text
    assert "Failed to close stdout for local shell process 1234" in caplog.text
    assert "Failed to close stderr for local shell process 1234" in caplog.text


def test_local_shell_backend_timeout_cleans_up_posix_process_group() -> None:
    """Test a timeout kills the command's POSIX process group."""
    process = MagicMock(pid=1234)
    process.communicate.side_effect = subprocess.TimeoutExpired("sleep 10", 1)
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        _as_posix(),
        patch.object(local_shell_module, "WindowsProcessReader", return_value=process),
        patch("subprocess.Popen", return_value=process),
        patch.object(local_shell_module.os, "killpg", create=True) as killpg,
    ):
        result = LocalShellBackend(root_dir=tmpdir, timeout=1).execute("sleep 10")

    assert result.exit_code == 124
    _assert_posix_cleanup(process, killpg)


def test_local_shell_backend_windows_timeout_kills_direct_process() -> None:
    """Test Windows timeout cleanup terminates and reaps the direct shell."""
    process = MagicMock(pid=1234)
    process.communicate.side_effect = subprocess.TimeoutExpired("sleep 10", 1)
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        _as_windows(),
        patch.object(local_shell_module, "WindowsProcessReader", return_value=process),
        patch("subprocess.Popen", return_value=process) as popen,
        patch.object(local_shell_module.os, "killpg", create=True) as killpg,
    ):
        result = LocalShellBackend(root_dir=tmpdir, timeout=1).execute("sleep 10")

    assert result.exit_code == 124
    assert popen.call_args.kwargs["start_new_session"] is False
    process.kill.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=local_shell_module._PROCESS_REAP_TIMEOUT)
    killpg.assert_not_called()


def test_local_shell_backend_timeout_bounds_process_reaping(caplog: pytest.LogCaptureFixture) -> None:
    """Test that a stuck process cannot extend cleanup indefinitely."""
    process = MagicMock(pid=1234)
    process.communicate.side_effect = subprocess.TimeoutExpired("sleep 10", 1)
    process.wait.side_effect = subprocess.TimeoutExpired("sleep 10", 5)
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch.object(local_shell_module, "WindowsProcessReader", return_value=process),
        patch("subprocess.Popen", return_value=process),
        patch.object(local_shell_module.os, "killpg", create=True),
        caplog.at_level("WARNING", logger="deepagents.backends.local_shell"),
    ):
        result = LocalShellBackend(root_dir=tmpdir, timeout=1).execute("sleep 10")

    assert result.exit_code == 124
    process.wait.assert_called_once_with(timeout=local_shell_module._PROCESS_REAP_TIMEOUT)
    assert "did not exit within 5 seconds after termination" in caplog.text
    # Pipes must close even when the process could not be reaped, or a stuck
    # command leaks two descriptors.
    process.stdout.close.assert_called_once_with()
    process.stderr.close.assert_called_once_with()
    # The caller cannot see the log, so the response has to carry the warning.
    assert "could not be stopped and may still be running" in result.output


@_POSIX_SHELL_ONLY
def test_local_shell_backend_cannot_open_parent_controlling_terminal(tmp_path: Path) -> None:
    """Test a command cannot open the controlling terminal owned by its parent."""
    exit_code, output = _run_controlling_terminal_probe(tmp_path)
    assert exit_code != 0
    assert "/dev/tty" in output


def test_local_shell_backend_execute_with_error() -> None:
    """Test executing a command that fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, inherit_env=True)

        result = backend.execute("cat nonexistent_file.txt")

        assert result.exit_code != 0
        assert "[stderr]" in result.output
        assert "Exit code:" in result.output


@_POSIX_SHELL_ONLY
def test_local_shell_backend_execute_in_working_directory() -> None:
    """Test that commands execute in the specified working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test file
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("test content")

        backend = LocalShellBackend(root_dir=tmpdir, inherit_env=True)

        # Execute command that relies on working directory
        result = backend.execute("cat test.txt")

        assert result.exit_code == 0
        assert "test content" in result.output


def test_local_shell_backend_execute_empty_command() -> None:
    """Test executing an empty command returns an error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir)

        result = backend.execute("")

        assert result.exit_code == 1
        assert "must be a non-empty string" in result.output


@_POSIX_SHELL_ONLY
def test_local_shell_backend_execute_timeout() -> None:
    """Test that long-running commands timeout correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, timeout=1.0, inherit_env=True)

        # Sleep for longer than timeout
        result = backend.execute("sleep 5")

        assert result.exit_code == 124  # Standard timeout exit code
        assert "timed out" in result.output


@_POSIX_SHELL_ONLY
def test_local_shell_backend_execute_output_truncation() -> None:
    """Test that large output gets truncated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, max_output_bytes=100, inherit_env=True)

        # Generate lots of output
        result = backend.execute("seq 1 1000")

        assert result.truncated is True
        assert "Output truncated" in result.output
        assert len(result.output) <= 150  # Some buffer for truncation message


def test_local_shell_backend_filesystem_operations() -> None:
    """Test that filesystem operations work (inherited from FilesystemBackend)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, virtual_mode=True)

        # Write a file
        write_result = backend.write("/test.txt", "Hello\nWorld\n")
        assert write_result.error is None
        assert write_result.path == "/test.txt"

        # Read the file
        content = backend.read("/test.txt")
        assert content.file_data is not None
        assert "Hello" in content.file_data["content"]
        assert "World" in content.file_data["content"]

        # Edit the file
        edit_result = backend.edit("/test.txt", "World", "Universe")
        assert edit_result.error is None
        assert edit_result.occurrences == 1

        # Verify edit
        content = backend.read("/test.txt")
        assert content.file_data is not None
        assert "Universe" in content.file_data["content"]
        assert "World" not in content.file_data["content"]


@_POSIX_SHELL_ONLY
def test_local_shell_backend_integration_shell_and_filesystem() -> None:
    """Test that shell commands and filesystem operations work together."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, virtual_mode=True, inherit_env=True)

        # Create file via filesystem
        backend.write("/script.sh", "#!/bin/bash\necho 'Script output'")

        # Make it executable and run via shell
        backend.execute("chmod +x script.sh")
        result = backend.execute("bash script.sh")

        assert result.exit_code == 0
        assert "Script output" in result.output

        # Create file via shell
        backend.execute("echo 'Shell created' > shell_file.txt")

        # Read via filesystem
        content = backend.read("/shell_file.txt")
        assert content.file_data is not None
        assert "Shell created" in content.file_data["content"]


def test_local_shell_backend_ls_info() -> None:
    """Test listing directory contents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, virtual_mode=True)

        # Create some files
        backend.write("/file1.txt", "content1")
        backend.write("/file2.txt", "content2")

        # List files
        files = backend.ls("/").entries

        assert files is not None
        assert len(files) == 2
        paths = [f["path"] for f in files]
        assert "/file1.txt" in paths
        assert "/file2.txt" in paths


def test_local_shell_backend_grep() -> None:
    """Test grep functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, virtual_mode=True)

        # Create files with searchable content
        backend.write("/file1.txt", "TODO: implement this")
        backend.write("/file2.txt", "DONE: completed")

        # Search for TODO
        matches = backend.grep("TODO").matches

        assert matches is not None
        assert len(matches) == 1
        assert matches[0]["text"] == "TODO: implement this"


def test_local_shell_backend_glob() -> None:
    """Test glob functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, virtual_mode=True)

        # Create files with different extensions
        backend.write("/file1.txt", "content")
        backend.write("/file2.py", "content")
        backend.write("/file3.txt", "content")

        # Find all .txt files
        txt_files = backend.glob("*.txt").matches

        assert txt_files is not None
        assert len(txt_files) == 2
        paths = [f["path"] for f in txt_files]
        assert "/file1.txt" in paths
        assert "/file3.txt" in paths
        assert "/file2.py" not in paths


def test_local_shell_backend_virtual_mode_restrictions() -> None:
    """Test that virtual_mode restricts filesystem paths but not shell commands."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, virtual_mode=True)

        # Filesystem operations should be restricted
        with pytest.raises(ValueError, match="Path traversal not allowed"):
            backend.read("/../etc/passwd")

        # But shell commands are NOT restricted (by design)
        result = backend.execute("cat /etc/passwd")
        # Command will succeed or fail based on permissions, but won't be blocked
        assert isinstance(result, ExecuteResponse)


@_POSIX_SHELL_ONLY
def test_local_shell_backend_environment_variables() -> None:
    """Test that custom environment variables are passed to commands."""
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_env = {"CUSTOM_VAR": "custom_value", "PATH": "/usr/bin:/bin"}
        backend = LocalShellBackend(root_dir=tmpdir, env=custom_env)

        result = backend.execute("sh -c 'echo $CUSTOM_VAR'")

        assert result.exit_code == 0
        assert "custom_value" in result.output


@_POSIX_SHELL_ONLY
def test_local_shell_backend_inherit_env() -> None:
    """Test that inherit_env=True inherits parent environment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, inherit_env=True)

        # PATH should be available from parent environment
        result = backend.execute("echo $PATH")

        assert result.exit_code == 0
        assert len(result.output.strip()) > 0  # PATH should not be empty


@_POSIX_SHELL_ONLY
def test_local_shell_backend_empty_env_by_default() -> None:
    """Test that environment is empty by default (secure default)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir)

        # Without inherit_env, PATH should not be available
        result = backend.execute("sh -c 'echo PATH is: $PATH'")

        assert result.exit_code == 0
        # PATH should be empty (the string "PATH is: " with no value after)
        assert "PATH is:" in result.output


def test_local_shell_backend_stderr_formatting() -> None:
    """Test that stderr is properly prefixed with [stderr]."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, inherit_env=True)

        # Command that outputs to stderr
        result = backend.execute("echo 'error message' >&2")

        assert result.exit_code == 0
        assert "[stderr]" in result.output
        assert "error message" in result.output


async def test_local_shell_backend_async_execute() -> None:
    """Test async execute method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, inherit_env=True)

        result = await backend.aexecute("echo 'async test'")

        assert isinstance(result, ExecuteResponse)
        assert result.exit_code == 0
        assert "async test" in result.output


async def test_local_shell_backend_async_execute_honors_execute_override() -> None:
    """Test async execution preserves subclass command restrictions."""
    calls: list[tuple[str, int | None]] = []

    class RestrictedLocalShellBackend(LocalShellBackend):
        def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
            calls.append((command, timeout))
            msg = f"Command is not allowed: {command}"
            raise PermissionError(msg)

    with tempfile.TemporaryDirectory() as tmpdir, patch("subprocess.Popen") as popen:
        backend = RestrictedLocalShellBackend(root_dir=tmpdir)
        with pytest.raises(PermissionError, match="Command is not allowed"):
            await backend.aexecute("blocked", timeout=5)

    assert calls == [("blocked", 5)]
    popen.assert_not_called()


async def test_local_shell_backend_async_preserves_context_and_legacy_override() -> None:
    context: ContextVar[str] = ContextVar("request", default="missing")

    class LegacyBackend(LocalShellBackend):
        def execute(self, command: str) -> ExecuteResponse:
            value = context.get()
            context.set("worker")
            return ExecuteResponse(output=value, exit_code=0, truncated=False)

    token = context.set("caller")
    try:
        response = await LegacyBackend().aexecute("ignored", timeout=5)
        assert response.output == "caller"
        assert context.get() == "caller"
    finally:
        context.reset(token)


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_local_shell_backend_async_override_interrupt_is_catchable(exception_type: type[BaseException]) -> None:
    class InterruptingBackend(LocalShellBackend):
        def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
            msg = "override interrupted"
            raise exception_type(msg)

    async def run() -> None:
        with pytest.raises(exception_type, match="override interrupted"):
            await InterruptingBackend().aexecute("ignored")

    try:
        asyncio.run(run())
    except exception_type:
        pytest.fail("The override interruption escaped the caller's exception handler")


async def test_local_shell_backend_async_cancellation_cleans_up_platform_process_scope() -> None:
    """Test async cancellation cleans up the platform's supported process scope."""
    communication_started = threading.Event()
    process = MagicMock(pid=1234)

    def block_communication(*, timeout: float) -> tuple[str, str]:
        communication_started.set()
        timeout_error = subprocess.TimeoutExpired(process.args, timeout)
        raise timeout_error

    process.communicate.side_effect = block_communication
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch.object(local_shell_module, "WindowsProcessReader", return_value=process),
        patch("subprocess.Popen", return_value=process),
        patch.object(local_shell_module.os, "killpg", create=True) as killpg,
    ):
        task = asyncio.create_task(LocalShellBackend(root_dir=tmpdir).aexecute("sleep 10"))
        assert await asyncio.to_thread(communication_started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    _assert_posix_cleanup(process, killpg)


@_POSIX_SHELL_ONLY
async def test_local_shell_backend_async_cancellation_stops_descendant(tmp_path: Path) -> None:
    """Test cancelling a real command stops its background descendant."""
    command, pid_file, heartbeat = _heartbeat_command(tmp_path)
    task = asyncio.create_task(LocalShellBackend(root_dir=tmp_path, inherit_env=True).aexecute(command))
    try:
        assert await asyncio.to_thread(_wait_for_file, pid_file)
        assert await asyncio.to_thread(_wait_for_file, heartbeat)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.to_thread(_assert_heartbeat_stopped, heartbeat)
    finally:
        if not task.done():
            task.cancel()
        _stop_test_descendant(pid_file)


@_POSIX_SHELL_ONLY
def test_local_shell_backend_cancellation_after_output_stops_descendant(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat"
    command = (
        f'i=0; while :; do i=$((i + 1)); printf "%s" "$i" > {shlex.quote(str(heartbeat))}; '
        "sleep 0.02; done >/dev/null 2>&1 & "
        f"while ! test -s {shlex.quote(str(heartbeat))}; do sleep 0.01; done"
    )
    cancellation_event = threading.Event()
    process = subprocess.Popen(  # noqa: S602  # Fixed shell probe with a quoted pytest temporary path.
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    )
    communicate = process.communicate

    def cancel_before_return(*, timeout: float) -> tuple[str, str]:
        output = communicate(timeout=timeout)
        cancellation_event.set()
        return output

    try:
        with patch.object(process, "communicate", side_effect=cancel_before_return), pytest.raises(local_shell_module._CommandCancelled):
            local_shell_module._communicate(process, 5, cancellation_event, process_group=process.pid)
        assert process.returncode == 0
        assert process.stdout is not None and process.stdout.closed
        assert process.stderr is not None and process.stderr.closed
        _assert_heartbeat_stopped(heartbeat)
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=5)


def test_local_shell_backend_async_start_race_skips_execution() -> None:
    """Test a worker observing cancellation before start skips execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir)
        cancellation_event = threading.Event()
        execution_started = threading.Event()
        cancellation_event.set()
        with patch.object(backend, "execute") as execute, pytest.raises(asyncio.CancelledError):
            backend._execute_in_thread("echo skipped", None, cancellation_event, execution_started)

    assert execution_started.is_set()
    execute.assert_not_called()


async def test_local_shell_backend_async_cancellation_preserves_cancelled_error(caplog: pytest.LogCaptureFixture) -> None:
    """Test a worker failure cannot replace async cancellation, but is still reported.

    Cancellation wins the race, so the caller sees `CancelledError`. The real
    failure must still reach the log, or a broken command is indistinguishable
    from an ordinary cancellation.
    """
    execution_started = threading.Event()
    release_execution = threading.Event()

    class FailingLocalShellBackend(LocalShellBackend):
        def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
            execution_started.set()
            release_execution.wait()
            msg = "backend exploded"
            raise RuntimeError(msg)

    with (
        tempfile.TemporaryDirectory() as tmpdir,
        caplog.at_level("WARNING", logger="deepagents.backends.local_shell"),
    ):
        task = asyncio.create_task(FailingLocalShellBackend(root_dir=tmpdir).aexecute("explode"))
        assert await asyncio.to_thread(execution_started.wait, 1)
        task.cancel()
        release_execution.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert task.cancelled()
    assert "failed on backend" in caplog.text
    assert "explode" in caplog.text
    assert "backend exploded" in caplog.text


async def test_local_shell_backend_async_cancellation_bypasses_execute_wrappers() -> None:
    """Test that cancellation is not exposed to wrappers as command output."""
    communication_started = threading.Event()
    observed_results: list[ExecuteResponse] = []
    process = MagicMock(pid=1234)

    def block_communication(*, timeout: float) -> tuple[str, str]:
        communication_started.set()
        raise subprocess.TimeoutExpired(process.args, timeout)

    class ObservingLocalShellBackend(LocalShellBackend):
        def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
            result = super().execute(command, timeout=timeout)
            observed_results.append(result)
            return result

    process.communicate.side_effect = block_communication
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch.object(local_shell_module, "WindowsProcessReader", return_value=process),
        patch("subprocess.Popen", return_value=process),
        patch.object(local_shell_module, "_kill_and_reap"),
    ):
        task = asyncio.create_task(ObservingLocalShellBackend(root_dir=tmpdir).aexecute("sleep 10"))
        assert await asyncio.to_thread(communication_started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert observed_results == []


async def test_local_shell_backend_async_cancellation_bounds_override_wait(caplog: pytest.LogCaptureFixture) -> None:
    """Test that an uncooperative override cannot block cancellation."""
    execution_started = threading.Event()
    release_execution = threading.Event()
    execution_finished = threading.Event()

    class SlowLocalShellBackend(LocalShellBackend):
        def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
            execution_started.set()
            release_execution.wait()
            execution_finished.set()
            return ExecuteResponse(output="done", exit_code=0, truncated=False)

    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch.object(local_shell_module, "_ASYNC_CANCELLATION_GRACE_PERIOD", 0.01),
        caplog.at_level("WARNING", logger="deepagents.backends.local_shell"),
    ):
        task = asyncio.create_task(SlowLocalShellBackend(root_dir=tmpdir).aexecute("slow override"))
        assert await asyncio.to_thread(execution_started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.5)

        release_execution.set()
        assert await asyncio.to_thread(execution_finished.wait, 1)
        for _ in range(100):
            if not local_shell_module._BACKGROUND_WORKERS:
                break
            await asyncio.sleep(0)

    assert task.cancelled()
    assert not local_shell_module._BACKGROUND_WORKERS
    assert "overridden execute method may still be running" in caplog.text


def test_local_shell_backend_async_cancellation_skips_queued_command() -> None:
    """Test that cancellation does not wait for or run queued executor work."""

    async def run_scenario() -> None:
        loop = asyncio.get_running_loop()
        executor_started = asyncio.Event()
        release_executor = threading.Event()

        def occupy_executor() -> None:
            loop.call_soon_threadsafe(executor_started.set)
            release_executor.wait()

        # Occupying the default executor keeps the command queued until cancellation.
        # The submit-count assertion ensures the command reached that queue.
        executor = ThreadPoolExecutor(max_workers=1)
        with patch.object(executor, "submit", wraps=executor.submit) as submit:
            loop.set_default_executor(executor)
            blocker = loop.run_in_executor(None, occupy_executor)
            await executor_started.wait()

            with tempfile.TemporaryDirectory() as tmpdir, patch("subprocess.Popen") as popen:
                task = asyncio.create_task(LocalShellBackend(root_dir=tmpdir).aexecute("echo queued"))
                for _ in range(100):
                    if submit.call_count > 1:
                        break
                    await asyncio.sleep(0)
                assert submit.call_count > 1, "command did not reach the executor queue"
                task.cancel()
                try:
                    for _ in range(100):
                        if task.done():
                            break
                        await asyncio.sleep(0)
                    assert task.done(), "cancellation waited for queued executor work"
                finally:
                    release_executor.set()
                    with suppress(asyncio.CancelledError):
                        await task
                    await blocker
                    await loop.run_in_executor(None, lambda: None)

            popen.assert_not_called()

    asyncio.run(run_scenario())


async def test_local_shell_backend_async_filesystem_operations() -> None:
    """Test async filesystem operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, virtual_mode=True)

        # Async write
        write_result = await backend.awrite("/async_test.txt", "async content")
        assert write_result.error is None

        # Async read
        content = await backend.aread("/async_test.txt")
        assert content.file_data is not None
        assert "async content" in content.file_data["content"]

        # Async edit
        edit_result = await backend.aedit("/async_test.txt", "async", "modified")
        assert edit_result.error is None

        # Verify
        content = await backend.aread("/async_test.txt")
        assert content.file_data is not None
        assert "modified content" in content.file_data["content"]


class TestLocalShellVirtualModeDefault:
    """`virtual_mode` defaults to `True` and never emits a deprecation."""

    def test_omitted_virtual_mode_defaults_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            be = LocalShellBackend(root_dir=tmpdir)

        deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning) and "virtual_mode" in str(w.message)]
        assert deprecations == []
        assert be.virtual_mode is True

    def test_explicit_virtual_mode_does_not_warn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            LocalShellBackend(root_dir=tmpdir, virtual_mode=False)
            LocalShellBackend(root_dir=tmpdir, virtual_mode=True)

        deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning) and "virtual_mode" in str(w.message)]
        assert deprecations == []


@_POSIX_SHELL_ONLY
def test_local_shell_backend_polling_loop_keeps_output_from_every_attempt() -> None:
    """Test the cancellation-aware loop keeps output read by earlier attempts.

    The loop retries `communicate` with a short timeout and depends on each retry
    keeping the bytes already read. A command that prints across several poll
    intervals shows this: if a retry reset the buffers, only the last chunk would
    survive and no other test would notice.
    """
    chunks = 8
    script = f'for i in $(seq 1 {chunks}); do printf "out$i\\n"; printf "err$i\\n" >&2; sleep 0.05; done'
    process = subprocess.Popen(  # noqa: S602  # Fixed shell probe with no external input.
        script, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    )
    try:
        stdout, stderr = local_shell_module._communicate(process, 30, threading.Event(), process_group=process.pid)
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)

    assert stdout == "".join(f"out{index}\n" for index in range(1, chunks + 1))
    assert stderr == "".join(f"err{index}\n" for index in range(1, chunks + 1))


@_POSIX_SHELL_ONLY
def test_local_shell_backend_polling_loop_drains_more_than_a_pipe_buffer() -> None:
    """Test output larger than the pipe buffer cannot deadlock the poll loop.

    A pipe holds roughly 64 KB. If the loop stopped draining one stream, the
    command would block on its write and the command would time out instead of
    returning.
    """
    script = "import sys; sys.stdout.write('o' * 500_000); sys.stderr.write('e' * 200_000)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    process = subprocess.Popen(  # noqa: S602  # Fixed probe that runs the test interpreter.
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    )
    try:
        stdout, stderr = local_shell_module._communicate(process, 30, threading.Event(), process_group=process.pid)
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)

    assert len(stdout) == 500_000
    assert len(stderr) == 200_000


@_POSIX_SHELL_ONLY
def test_local_shell_backend_timeout_reports_output_printed_before_it() -> None:
    """Test a timed-out command still reports what it printed first.

    What a command printed before it wedged is usually the only clue about where
    it stopped, so the response must carry it.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        result = LocalShellBackend(root_dir=tmpdir, inherit_env=True).execute('printf "progress line\n"; sleep 30', timeout=1)

    assert result.exit_code == 124
    assert "timed out" in result.output
    assert "progress line" in result.output


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
@pytest.mark.parametrize("data", [b"progress \xe2", b"progress \xff"])
def test_local_shell_backend_timeout_replaces_undecodable_output(tmp_path: Path, stream: str, data: bytes) -> None:
    """Incomplete or invalid UTF-8 must preserve the timeout and partial output."""
    process = MagicMock(pid=1234)
    pipe = getattr(process, stream)
    pipe.encoding = "utf-8"
    pipe.errors = "strict"
    process.communicate.side_effect = subprocess.TimeoutExpired(
        "command", 1, output=data if stream == "stdout" else None, stderr=data if stream == "stderr" else None
    )
    with (
        _as_posix(),
        patch("subprocess.Popen", return_value=process),
        patch.object(local_shell_module.os, "killpg", create=True),
    ):
        result = LocalShellBackend(root_dir=tmp_path).execute("command", timeout=1)

    assert result.exit_code == 124
    assert "timed out after 1 seconds (custom timeout)" in result.output
    assert "The command may be stuck or require more time." in result.output
    assert "progress \ufffd" in result.output


def test_local_shell_backend_already_exited_group_is_not_a_cleanup_failure(caplog: pytest.LogCaptureFixture) -> None:
    """Test an empty process group counts as success, not as a failed kill.

    Cancellation arriving just after a command finished finds nothing left to
    kill. Reporting that as a failure would train readers to ignore this logger.
    """
    process = MagicMock(pid=1234)
    with (
        _as_posix(),
        patch.object(local_shell_module.os, "killpg", create=True, side_effect=ProcessLookupError),
        caplog.at_level("WARNING", logger="deepagents.backends.local_shell"),
    ):
        assert local_shell_module._kill_and_reap(process, 1234) is True

    process.kill.assert_not_called()
    assert "Failed to terminate" not in caplog.text


def test_local_shell_backend_failed_termination_is_reported_to_its_caller() -> None:
    """Test cleanup reports failure so the caller can warn about an orphan."""
    process = MagicMock(pid=1234)
    process.kill.side_effect = PermissionError
    with (
        _as_posix(),
        patch.object(local_shell_module.os, "killpg", create=True, side_effect=PermissionError),
    ):
        assert local_shell_module._kill_and_reap(process, 1234) is False


def test_local_shell_backend_unexpected_failure_is_reported_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Test an unexpected error becomes an error response and leaves a traceback.

    Exit code 1 is indistinguishable from the command itself failing, so the
    traceback has to reach the log or the failure is undiagnosable.
    """
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("subprocess.Popen", side_effect=OSError("boom")),
        caplog.at_level("ERROR", logger="deepagents.backends.local_shell"),
    ):
        result = LocalShellBackend(root_dir=tmpdir).execute("echo hi")

    assert result.exit_code == 1
    assert "OSError" in result.output
    assert "boom" in result.output
    assert "Local shell command failed" in caplog.text
    assert "echo hi" in caplog.text


async def test_local_shell_backend_cancellation_does_not_stop_another_backend() -> None:
    """Test one backend's cancellation cannot stop a command run by another.

    An override runs inside the cancelling backend's copied context, so a second
    backend called from that override sees the same context. `execute` must
    ignore a cancellation event that belongs to a different backend.
    """
    observed: list[threading.Event | None] = []
    execution_started = threading.Event()
    release_execution = threading.Event()
    original_execute = LocalShellBackend._execute

    with tempfile.TemporaryDirectory() as tmpdir:
        inner = LocalShellBackend(root_dir=tmpdir, inherit_env=True)

        def spy(
            self: LocalShellBackend,
            command: str,
            *,
            timeout: int | None,
            cancellation_event: threading.Event | None = None,
        ) -> ExecuteResponse:
            if self is inner:
                observed.append(cancellation_event)
            return original_execute(self, command, timeout=timeout, cancellation_event=cancellation_event)

        class NestingLocalShellBackend(LocalShellBackend):
            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                execution_started.set()
                release_execution.wait(5)
                return inner.execute("echo nested")

        with patch.object(LocalShellBackend, "_execute", spy):
            task = asyncio.create_task(NestingLocalShellBackend(root_dir=tmpdir, inherit_env=True).aexecute("outer"))
            assert await asyncio.to_thread(execution_started.wait, 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release_execution.set()
            for _ in range(200):
                if observed:
                    break
                await asyncio.sleep(0.01)

    assert observed == [None], "the inner backend inherited another backend's cancellation event"


@_POSIX_SHELL_ONLY
async def test_local_shell_backend_cancelling_one_command_leaves_a_sibling_running() -> None:
    """Test cancelling one async command does not disturb a concurrent one.

    Each worker gets its own cancellation event through `copy_context`. Holding
    the event on the backend instead would make sibling commands kill each other.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, inherit_env=True)
        survivor = asyncio.create_task(backend.aexecute("sleep 0.5; echo survivor"))
        doomed = asyncio.create_task(backend.aexecute("sleep 30"))
        await asyncio.sleep(0.1)
        doomed.cancel()
        with pytest.raises(asyncio.CancelledError):
            await doomed
        result = await survivor

    assert result.exit_code == 0
    assert "survivor" in result.output


async def test_local_shell_backend_repeated_cancellation_still_tracks_the_worker(caplog: pytest.LogCaptureFixture) -> None:
    """Test a second cancellation during the grace period does not skip cleanup.

    If the repeat `CancelledError` escaped, the worker would never be added to
    `_BACKGROUND_WORKERS` and asyncio would later report its result as never
    retrieved at an unrelated point.
    """
    execution_started = threading.Event()
    release_execution = threading.Event()

    class SlowLocalShellBackend(LocalShellBackend):
        def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
            execution_started.set()
            release_execution.wait(5)
            return ExecuteResponse(output="done", exit_code=0, truncated=False)

    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch.object(local_shell_module, "_ASYNC_CANCELLATION_GRACE_PERIOD", 0.3),
        caplog.at_level("WARNING", logger="deepagents.backends.local_shell"),
    ):
        task = asyncio.create_task(SlowLocalShellBackend(root_dir=tmpdir).aexecute("slow override"))
        assert await asyncio.to_thread(execution_started.wait, 2)
        task.cancel()
        # Let the coroutine reach the grace-period wait, then cancel again.
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        tracked = len(local_shell_module._BACKGROUND_WORKERS)

        release_execution.set()
        for _ in range(200):
            if not local_shell_module._BACKGROUND_WORKERS:
                break
            await asyncio.sleep(0.01)

    assert task.cancelled()
    assert tracked == 1, "the repeated cancellation skipped the background-worker bookkeeping"
    assert "overridden execute method may still be running" in caplog.text
    assert not local_shell_module._BACKGROUND_WORKERS
