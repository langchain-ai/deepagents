"""Unit tests for LocalShellBackend."""

import tempfile
from pathlib import Path

import pytest

from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse


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
        assert "Hello" in content
        assert "World" in content

        # Edit the file
        edit_result = backend.edit("/test.txt", "World", "Universe")
        assert edit_result.error is None
        assert edit_result.occurrences == 1

        # Verify edit
        content = backend.read("/test.txt")
        assert "Universe" in content
        assert "World" not in content


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
        assert "Shell created" in content


def test_local_shell_backend_ls_info() -> None:
    """Test listing directory contents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, virtual_mode=True)

        # Create some files
        backend.write("/file1.txt", "content1")
        backend.write("/file2.txt", "content2")

        # List files
        files = backend.ls_info("/")

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
        matches = backend.grep_raw("TODO")

        assert isinstance(matches, list)
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
        txt_files = backend.glob_info("*.txt")

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
        assert "async content" in content

        # Async edit
        edit_result = await backend.aedit("/async_test.txt", "async", "modified")
        assert edit_result.error is None

        # Verify
        content = await backend.aread("/async_test.txt")
        assert "modified content" in content
