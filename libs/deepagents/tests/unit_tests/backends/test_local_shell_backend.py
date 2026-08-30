"""Unit tests for LocalShellBackend."""

import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from deepagents.backends.local_shell import LocalShellBackend, _resolve_pipeline_status
from deepagents.backends.protocol import ExecuteResponse

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="LocalShellBackend requires sh, not available on Windows")


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


def test_local_shell_backend_execute_starts_new_session() -> None:
    """Test that commands cannot access the parent's controlling terminal."""
    completed = subprocess.CompletedProcess(args="echo hello", returncode=0, stdout="hello\n", stderr="")
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir)
        with patch("subprocess.run", return_value=completed) as run:
            backend.execute("echo hello")

    assert run.call_args.kwargs["start_new_session"] is True


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


@pytest.mark.skipif(shutil.which("bash") is None, reason="pipeline status recovery requires bash")
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        pytest.param("sh -c 'echo out; exit 1' | tail -1", 1, id="failure-piped-into-filter"),
        pytest.param("sh -c 'exit 3' | sh -c 'exit 0' | tail -1", 3, id="first-of-three-stages-fails"),
        pytest.param("echo hi | cat", 0, id="clean-pipeline"),
    ],
)
def test_local_shell_backend_reports_failure_masked_by_a_pipeline(command: str, expected: int) -> None:
    """A shell reports only a pipeline's last stage, hiding `pytest ... | tail`-style failures."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = LocalShellBackend(root_dir=temp_dir).execute(command)
        assert result.exit_code == expected


@pytest.mark.skipif(shutil.which("bash") is None, reason="pipeline status recovery requires bash")
@pytest.mark.parametrize(
    "command",
    [
        pytest.param("seq 1 100000 | head -3", id="head-closes-the-pipe"),
        pytest.param("yes | head -1", id="infinite-writer-closed-early"),
        pytest.param("seq 1 100000 | grep -q 5", id="grep-q-exits-on-first-match"),
    ],
)
def test_local_shell_backend_ignores_sigpipe_from_a_reader_exiting_early(command: str) -> None:
    """`head` and `grep -q` close the pipe on purpose; the writer's SIGPIPE is not a failure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = LocalShellBackend(root_dir=temp_dir).execute(command)
        assert result.exit_code == 0


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        pytest.param("sh -c 'exit 1'; echo done", 0, id="semicolon-chain-reports-last-command"),
        pytest.param("sh -c 'exit 1' || true", 0, id="explicit-suppression-is-honoured"),
        pytest.param("echo a; exit 5", 5, id="command-exiting-directly"),
        pytest.param("sh -c 'exit 7'", 7, id="plain-failure"),
        pytest.param("echo ok", 0, id="plain-success"),
    ],
)
def test_local_shell_backend_leaves_non_pipeline_status_alone(command: str, expected: int) -> None:
    """`a; b` and `a || true` mean what they say -- recovering a status there would break them."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = LocalShellBackend(root_dir=temp_dir).execute(command)
        assert result.exit_code == expected


def test_local_shell_backend_pipeline_recovery_preserves_output() -> None:
    """The status side-channel must not leak into the output the agent reads."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = LocalShellBackend(root_dir=temp_dir).execute("echo HELLO | cat")
        assert result.output.strip() == "HELLO"


@pytest.mark.parametrize(
    ("statuses", "returncode", "expected"),
    [
        pytest.param([1, 0], 0, 1, id="earlier-stage-failed"),
        pytest.param([141, 0], 0, 0, id="sigpipe-before-last-stage-is-benign"),
        pytest.param([0, 141], 141, 141, id="sigpipe-on-last-stage-is-real"),
        pytest.param([0], 5, 5, id="single-stage-defers-to-returncode"),
        pytest.param([], 5, 5, id="unavailable-defers-to-returncode"),
        pytest.param([0, 0], 0, 0, id="all-clean"),
    ],
)
def test_resolve_pipeline_status(statuses: list[int], returncode: int, expected: int) -> None:
    assert _resolve_pipeline_status(statuses, returncode) == expected
