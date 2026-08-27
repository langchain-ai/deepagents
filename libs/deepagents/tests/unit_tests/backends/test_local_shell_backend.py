"""Unit tests for LocalShellBackend."""

import asyncio
import signal
import subprocess
import sys
import tempfile
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="LocalShellBackend requires sh, not available on Windows")


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


def test_local_shell_backend_execute_starts_new_session() -> None:
    """Test that commands cannot access the parent's controlling terminal."""
    process = MagicMock(returncode=0)
    process.communicate.return_value = ("hello\n", "")
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir)
        with patch("subprocess.Popen", return_value=process) as popen:
            backend.execute("echo hello")

    assert popen.call_args.kwargs["start_new_session"] is True


def test_local_shell_backend_interrupt_kills_process_group() -> None:
    """Test that an interrupt cannot leave detached descendants running."""
    process = MagicMock(pid=1234)
    process.communicate.side_effect = KeyboardInterrupt
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("subprocess.Popen", return_value=process),
        patch("os.killpg") as killpg,
        pytest.raises(KeyboardInterrupt),
    ):
        LocalShellBackend(root_dir=tmpdir).execute("sleep 10")

    killpg.assert_called_once_with(1234, signal.SIGKILL)
    process.wait.assert_called_once_with()


def test_local_shell_backend_timeout_kills_process_group() -> None:
    """Test that a timeout cannot leave detached descendants running."""
    process = MagicMock(pid=1234)
    process.communicate.side_effect = subprocess.TimeoutExpired("sleep 10", 1)
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("subprocess.Popen", return_value=process),
        patch("os.killpg") as killpg,
    ):
        result = LocalShellBackend(root_dir=tmpdir, timeout=1).execute("sleep 10")

    assert result.exit_code == 124
    killpg.assert_called_once_with(1234, signal.SIGKILL)
    process.wait.assert_called_once_with()


def test_local_shell_backend_execute_with_error() -> None:
    """Test executing a command that fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, inherit_env=True)

        result = backend.execute("cat nonexistent_file.txt")

        assert result.exit_code != 0
        assert "[stderr]" in result.output
        assert "Exit code:" in result.output


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


def test_local_shell_backend_execute_timeout() -> None:
    """Test that long-running commands timeout correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, timeout=1.0, inherit_env=True)

        # Sleep for longer than timeout
        result = backend.execute("sleep 5")

        assert result.exit_code == 124  # Standard timeout exit code
        assert "timed out" in result.output


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


def test_local_shell_backend_environment_variables() -> None:
    """Test that custom environment variables are passed to commands."""
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_env = {"CUSTOM_VAR": "custom_value", "PATH": "/usr/bin:/bin"}
        backend = LocalShellBackend(root_dir=tmpdir, env=custom_env)

        result = backend.execute("sh -c 'echo $CUSTOM_VAR'")

        assert result.exit_code == 0
        assert "custom_value" in result.output


def test_local_shell_backend_inherit_env() -> None:
    """Test that inherit_env=True inherits parent environment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, inherit_env=True)

        # PATH should be available from parent environment
        result = backend.execute("echo $PATH")

        assert result.exit_code == 0
        assert len(result.output.strip()) > 0  # PATH should not be empty


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


async def test_local_shell_backend_async_cancellation_kills_process_group() -> None:
    """Test that async cancellation reaps the detached command."""
    communication_started = threading.Event()
    process = MagicMock(pid=1234)

    def block_communication(*, timeout: float) -> tuple[str, str]:
        communication_started.set()
        timeout_error = subprocess.TimeoutExpired(process.args, timeout)
        raise timeout_error

    process.communicate.side_effect = block_communication
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("subprocess.Popen", return_value=process),
        patch("os.killpg") as killpg,
    ):
        task = asyncio.create_task(LocalShellBackend(root_dir=tmpdir).aexecute("sleep 10"))
        assert await asyncio.to_thread(communication_started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    killpg.assert_called_once_with(1234, signal.SIGKILL)
    process.wait.assert_called_once_with()


def test_local_shell_backend_async_cancellation_skips_queued_command() -> None:
    """Test that cancellation does not wait for or run queued executor work."""

    async def run_scenario() -> None:
        loop = asyncio.get_running_loop()
        executor_started = asyncio.Event()
        release_executor = threading.Event()

        def occupy_executor() -> None:
            loop.call_soon_threadsafe(executor_started.set)
            release_executor.wait()

        executor = ThreadPoolExecutor(max_workers=1)
        with patch.object(executor, "submit", wraps=executor.submit) as submit:
            loop.set_default_executor(executor)
            blocker = loop.run_in_executor(None, occupy_executor)
            await executor_started.wait()

            with tempfile.TemporaryDirectory() as tmpdir, patch("subprocess.Popen") as popen, patch("os.killpg"):
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
