"""Unit tests for LocalShellBackend."""

import errno
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepagents.backends.local_shell import LocalShellBackend, _describe_pipeline_stages, _prepare_pipeline_status_capture
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
    process = MagicMock()
    process.communicate.return_value = ("hello\n", "")
    process.returncode = 0
    process.__enter__.return_value = process
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir)
        with patch("subprocess.Popen", return_value=process) as popen:
            backend.execute("echo hello")

    assert popen.call_args.kwargs["start_new_session"] is True


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


@pytest.mark.skipif(shutil.which("bash") is None, reason="pipeline status detail requires bash")
@pytest.mark.parametrize(
    ("command", "expected_stages"),
    [
        pytest.param("sh -c 'echo out; exit 1' | tail -1", "1, 0", id="failure-hidden-by-a-filter"),
        pytest.param("sh -c 'exit 3' | sh -c 'exit 0' | tail -1", "3, 0, 0", id="first-of-three-stages"),
        pytest.param("grep zzz /etc/hosts | wc -l", "1, 0", id="non-zero-as-information-is-still-shown"),
    ],
)
def test_local_shell_backend_reports_pipeline_stage_statuses(command: str, expected_stages: str) -> None:
    """The stages a shell hides are surfaced for the agent to interpret."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = LocalShellBackend(root_dir=temp_dir).execute(command)
        assert f"Pipeline stages exited: {expected_stages}" in result.output


@pytest.mark.parametrize(
    "command",
    [
        pytest.param("grep zzz /etc/hosts | wc -l", id="grep-no-match"),
        pytest.param("diff /etc/hosts /etc/passwd | head -2", id="diff-differs"),
        pytest.param("sh -c 'echo out; exit 1' | tail -1", id="failure-hidden-by-a-filter"),
        pytest.param("seq 1 100000 | head -3", id="reader-exits-early"),
        pytest.param("false | cat; exit 9", id="command-exiting-after-a-pipeline"),
        pytest.param("{ sh -c 'exit 3' | cat; } || exit 0", id="explicit-suppression"),
        pytest.param("echo a; exit 5", id="command-exiting-directly"),
        pytest.param("sh -c 'exit 7'", id="plain-failure"),
        pytest.param("echo ok", id="plain-success"),
    ],
)
def test_local_shell_backend_never_changes_the_exit_code(command: str) -> None:
    """Reporting stage detail must not rewrite the status.

    Nothing at this layer can tell `pytest` exiting 1 from `grep` exiting 1, so it does not
    try. Every command must return exactly what a plain shell would.
    """
    expected = subprocess.run(command, shell=True, capture_output=True, check=False).returncode  # noqa: S602
    with tempfile.TemporaryDirectory() as temp_dir:
        result = LocalShellBackend(root_dir=temp_dir).execute(command)
        assert result.exit_code == expected


@pytest.mark.skipif(shutil.which("bash") is None, reason="pipeline status detail requires bash")
def test_local_shell_backend_omits_stale_pipeline_detail() -> None:
    """`exit` terminates before PIPESTATUS is reset, so the trap can see an older pipeline."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = LocalShellBackend(root_dir=temp_dir).execute("false | cat; exit 9")
        assert "Pipeline stages exited" not in result.output
        assert result.exit_code == 9


@pytest.mark.skipif(shutil.which("bash") is None, reason="pipeline status detail requires bash")
def test_local_shell_backend_wrapper_preserves_an_unterminated_heredoc() -> None:
    """The wrapper must not turn a command a plain shell accepts into a syntax error."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = LocalShellBackend(root_dir=temp_dir).execute("cat <<EOF\nhello")
        assert result.exit_code == 0
        assert "hello" in result.output


def test_local_shell_backend_pipeline_detail_preserves_output() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        result = LocalShellBackend(root_dir=temp_dir).execute("echo HELLO | cat")
        assert result.output.splitlines()[0] == "HELLO"


@pytest.mark.parametrize(
    ("statuses", "returncode", "expected"),
    [
        pytest.param([1, 0], 0, "1, 0", id="earlier-stage-failed"),
        pytest.param([3, 0, 0], 0, "3, 0, 0", id="first-of-three-stages-failed"),
        pytest.param([2, 141, 0], 0, "2, 141 (SIGPIPE), 0", id="sigpipe-spelled-out-alongside-a-real-failure"),
        pytest.param([0, 1], 1, "", id="only-the-last-stage-failed-so-the-exit-code-says-it"),
        pytest.param([0, 0, 5], 5, "", id="only-the-last-of-three-failed"),
        pytest.param([141, 0], 0, "", id="sigpipe-alone-is-how-a-reader-stops-a-producer"),
        pytest.param([141, 141, 0], 0, "", id="sigpipe-throughout-adds-nothing"),
        pytest.param([0, 0], 0, "", id="all-clean-adds-nothing"),
        pytest.param([0], 5, "", id="single-stage-adds-nothing"),
        pytest.param([], 5, "", id="unavailable"),
        pytest.param([1, 0], 9, "", id="stale-array-suppressed"),
    ],
)
def test_describe_pipeline_stages(statuses: list[int], returncode: int, expected: str) -> None:
    assert _describe_pipeline_stages(statuses, returncode) == expected


@pytest.mark.skipif(shutil.which("bash") is None, reason="pipeline status detail requires bash")
@pytest.mark.parametrize(
    "command",
    [
        pytest.param("cat /etc/hosts | grep -c zzz", id="only-the-last-stage-failed"),
        pytest.param("seq 1 100000 | head -3", id="reader-stops-the-producer"),
        pytest.param("yes | head -2", id="endless-producer-stopped-by-its-reader"),
        pytest.param("echo hi | cat", id="nothing-went-wrong"),
    ],
)
def test_local_shell_backend_omits_detail_a_reader_cannot_use(command: str) -> None:
    """Detail that only repeats the exit code, or that reports a pipeline working, is noise.

    `grep -c` exiting 1 as the last stage is already in `Exit code: 1`, and the 141 that
    `head` provokes in `seq` is how the command is supposed to end. A bare number next to
    either reads as a second, unexplained failure.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        result = LocalShellBackend(root_dir=temp_dir).execute(command)
        assert "Pipeline stages exited" not in result.output


@pytest.mark.skipif(shutil.which("bash") is None, reason="pipeline status detail requires bash")
def test_local_shell_backend_names_sigpipe_when_it_reports_a_hidden_failure() -> None:
    """When there is a real failure to report, any `SIGPIPE` beside it is named, not numbered."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = LocalShellBackend(root_dir=temp_dir).execute("sh -c 'exit 2' | seq 1 100000 | head -3")
        assert "Pipeline stages exited: 2, 141 (SIGPIPE), 0" in result.output


def test_local_shell_backend_runs_when_the_status_file_cannot_be_created() -> None:
    """A temp directory that is read-only, full, or out of descriptors must not fail the command."""
    unwritable = OSError(errno.EROFS, "Read-only file system")
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = LocalShellBackend(root_dir=temp_dir)
        with patch("tempfile.mkstemp", side_effect=unwritable):
            result = backend.execute("echo still here | cat")
    assert result.exit_code == 0
    assert "still here" in result.output


def test_prepare_pipeline_status_capture_degrades_to_the_plain_command() -> None:
    """Without a status file the command must run exactly as it did before this diagnostic."""
    with patch("tempfile.mkstemp", side_effect=OSError(errno.ENOSPC, "No space left on device")):
        assert _prepare_pipeline_status_capture("echo hi | cat") == ("echo hi | cat", None, None)


TIMEOUT_SECONDS = 2
"""Short enough to keep the timeout tests quick, long enough not to race a loaded machine."""

RETURN_DEADLINE_SECONDS = 15
"""`execute` must be back long before its descendant finishes, and with room over the timeout.

The markers below double as the descendants' lifetimes, so each is at least 27 seconds -- far
enough past this deadline that a run hitting it has genuinely waited for the descendant.
"""


def _reap_survivors(pattern: str) -> int:
    """Count -- and then kill -- any process whose command line still matches `pattern`."""
    # S603/S607: fixed argv, no shell; the only interpolation is a constant marker from a test.
    found = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, check=False)  # noqa: S603, S607
    survivors = found.stdout.split()
    if survivors:
        subprocess.run(["pkill", "-f", pattern], check=False)  # noqa: S603, S607
    return len(survivors)


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX-only")
def test_local_shell_backend_timeout_kills_a_background_descendant_and_returns() -> None:
    """A timed-out command must die at the timeout, and `execute` must come back then too.

    `echo hi; sleep N &` is the awkward shape: the shell returns immediately, so by the time
    the timeout fires it is an exited-but-unreaped zombie whose pgid can no longer be looked
    up, while the descendant it left behind is still holding the pipes `execute` reads.
    `THREAT_MODEL.md` lists this timeout as the mitigation at trust boundary TB5, so neither
    the kill nor the return may depend on the shell still being alive.
    """
    marker = "31.41592"
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = LocalShellBackend(root_dir=temp_dir)
        start = time.monotonic()
        result = backend.execute(f"echo hi; sleep {marker} &", timeout=TIMEOUT_SECONDS)
        elapsed = time.monotonic() - start
    survivors = _reap_survivors(f"^sleep {marker}$")
    assert result.exit_code == 124
    assert elapsed < RETURN_DEADLINE_SECONDS, f"execute blocked for {elapsed:.1f}s on a {TIMEOUT_SECONDS}s timeout"
    assert survivors == 0, f"timeout left {survivors} process(es) running"


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX-only")
def test_local_shell_backend_timeout_returns_even_when_a_descendant_escapes_the_kill() -> None:
    """A descendant the kill cannot reach must not be able to hold `execute` open either.

    Nothing can signal a process that has put itself in another session, so this one outlives
    the timeout by design -- and it still holds the inherited pipes. Draining those pipes
    after the kill would wait for it. Waiting for the shell alone does not.
    """
    marker = "27.182818"
    escapee = f"import os, time; os.setsid(); time.sleep({marker})"
    command = f"echo hi; {shlex.quote(sys.executable)} -c {shlex.quote(escapee)} &"
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = LocalShellBackend(root_dir=temp_dir)
        start = time.monotonic()
        result = backend.execute(command, timeout=TIMEOUT_SECONDS)
        elapsed = time.monotonic() - start
    _reap_survivors(f"time.sleep\\({marker}\\)")
    assert result.exit_code == 124
    assert elapsed < RETURN_DEADLINE_SECONDS, f"execute blocked for {elapsed:.1f}s on a {TIMEOUT_SECONDS}s timeout"
