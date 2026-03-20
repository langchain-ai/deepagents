"""Tests for BaseSandbox backend operations.

Verifies that write/edit use upload_files/download_files (bypassing ARG_MAX),
and that remaining command templates format correctly.

Related issues:
- https://github.com/langchain-ai/deepagents/pull/872
- ARG_MAX failures when writing large payloads via docker exec
"""

import base64
import json

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import (
    _GLOB_COMMAND_TEMPLATE,
    _READ_COMMAND_TEMPLATE,
    _WRITE_CHECK_TEMPLATE,
    BaseSandbox,
)


class MockSandbox(BaseSandbox):
    """Minimal concrete implementation of BaseSandbox for testing."""

    def __init__(self) -> None:
        self.last_command: str | None = None
        self._next_output: str = "1"
        self._uploaded: list[tuple[str, bytes]] = []
        self._file_store: dict[str, bytes] = {}

    @property
    def id(self) -> str:
        return "mock-sandbox"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        self.last_command = command
        output = self._next_output
        self._next_output = "1"
        return ExecuteResponse(output=output, exit_code=0, truncated=False)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        self._uploaded.extend(files)
        for path, content in files:
            self._file_store[path] = content
        return [FileUploadResponse(path=f[0], error=None) for f in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        results = []
        for p in paths:
            if p in self._file_store:
                results.append(FileDownloadResponse(path=p, content=self._file_store[p], error=None))
            else:
                results.append(FileDownloadResponse(path=p, content=None, error="file_not_found"))
        return results


# -- template formatting tests -----------------------------------------------


def test_write_check_template_format() -> None:
    """Test that _WRITE_CHECK_TEMPLATE can be formatted without KeyError."""
    path_b64 = base64.b64encode(b"/test/file.txt").decode("ascii")
    cmd = _WRITE_CHECK_TEMPLATE.format(path_b64=path_b64)

    assert "python3 -c" in cmd
    assert path_b64 in cmd


def test_glob_command_template_format() -> None:
    """Test that _GLOB_COMMAND_TEMPLATE can be formatted without KeyError."""
    path_b64 = base64.b64encode(b"/test").decode("ascii")
    pattern_b64 = base64.b64encode(b"*.py").decode("ascii")

    cmd = _GLOB_COMMAND_TEMPLATE.format(path_b64=path_b64, pattern_b64=pattern_b64)

    assert "python3 -c" in cmd
    assert path_b64 in cmd
    assert pattern_b64 in cmd


def test_read_command_template_format() -> None:
    """Test that _READ_COMMAND_TEMPLATE can be formatted without KeyError."""
    payload = json.dumps({"path": "/test/file.txt", "offset": 0, "limit": 100})
    payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    cmd = _READ_COMMAND_TEMPLATE.format(payload_b64=payload_b64)

    assert "python3 -c" in cmd
    assert payload_b64 in cmd
    assert "__DEEPAGENTS_EOF__" in cmd


def test_read_command_template_ends_with_newline() -> None:
    """Test that heredoc-based read template terminates with a trailing newline."""
    payload = json.dumps({"path": "/test/file.txt", "offset": 0, "limit": 100})
    payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    read_cmd = _READ_COMMAND_TEMPLATE.format(payload_b64=payload_b64)

    assert read_cmd.endswith("\n")


# -- write tests --------------------------------------------------------------


def test_sandbox_write_uses_upload_files() -> None:
    """Test that write() delegates data transfer to upload_files()."""
    sandbox = MockSandbox()

    sandbox.write("/test/file.txt", "test content")

    assert len(sandbox._uploaded) == 1
    path, data = sandbox._uploaded[0]
    assert path == "/test/file.txt"
    assert data == b"test content"


def test_sandbox_write_check_command_is_small() -> None:
    """Test that write() only sends a small check command to execute(), not the content."""
    sandbox = MockSandbox()
    large_content = "x" * 500_000  # 500KB — would blow ARG_MAX if embedded

    sandbox.write("/test/big.txt", large_content)

    # The command sent to execute() should be the small check, not the content
    assert sandbox.last_command is not None
    assert len(sandbox.last_command) < 1000
    assert large_content not in sandbox.last_command


def test_sandbox_write_with_special_content() -> None:
    """Test write with content containing curly braces and special characters."""
    sandbox = MockSandbox()
    content = "def foo(): return {key: value for key, value in items.items()}"

    sandbox.write("/test/code.py", content)

    assert sandbox._uploaded[0][1] == content.encode("utf-8")


def test_sandbox_write_returns_error_on_existing_file() -> None:
    """Test that write() returns an error when the check command fails."""
    sandbox = MockSandbox()
    sandbox._next_output = "Error: File already exists"

    def fail_execute(command: str, *, timeout: int | None = None) -> ExecuteResponse:  # noqa: ARG001
        sandbox.last_command = command
        return ExecuteResponse(output="Error: File already exists", exit_code=1)

    sandbox.execute = fail_execute

    result = sandbox.write("/test/existing.txt", "content")
    assert result.error is not None
    assert "Error:" in result.error
    assert len(sandbox._uploaded) == 0  # upload should not have been called


# -- edit tests ----------------------------------------------------------------


def test_sandbox_edit_uses_download_upload() -> None:
    """Test that edit() uses download_files + upload_files, not execute()."""
    sandbox = MockSandbox()
    sandbox._file_store["/test/file.txt"] = b"hello old world"

    result = sandbox.edit("/test/file.txt", "old", "new")

    assert result.error is None
    assert result.occurrences == 1
    assert sandbox._file_store["/test/file.txt"] == b"hello new world"


def test_sandbox_edit_file_not_found() -> None:
    """Test that edit() returns error when file doesn't exist."""
    sandbox = MockSandbox()

    result = sandbox.edit("/test/missing.txt", "old", "new")

    assert result.error is not None
    assert "not found" in result.error


def test_sandbox_edit_string_not_found() -> None:
    """Test that edit() returns error when old_string is not in the file."""
    sandbox = MockSandbox()
    sandbox._file_store["/test/file.txt"] = b"hello world"

    result = sandbox.edit("/test/file.txt", "missing", "new")

    assert result.error is not None
    assert "not found" in result.error


def test_sandbox_edit_multiple_occurrences_without_replace_all() -> None:
    """Test that edit() errors on multiple occurrences without replace_all."""
    sandbox = MockSandbox()
    sandbox._file_store["/test/file.txt"] = b"foo bar foo"

    result = sandbox.edit("/test/file.txt", "foo", "baz")

    assert result.error is not None
    assert "multiple times" in result.error.lower()


def test_sandbox_edit_replace_all() -> None:
    """Test that edit() with replace_all replaces all occurrences."""
    sandbox = MockSandbox()
    sandbox._file_store["/test/file.txt"] = b"foo bar foo"

    result = sandbox.edit("/test/file.txt", "foo", "baz", replace_all=True)

    assert result.error is None
    assert result.occurrences == 2
    assert sandbox._file_store["/test/file.txt"] == b"baz bar baz"


def test_sandbox_edit_with_special_strings() -> None:
    """Test edit with strings containing curly braces."""
    sandbox = MockSandbox()
    sandbox._file_store["/test/file.txt"] = b"value = {old_key}"

    result = sandbox.edit("/test/file.txt", "{old_key}", "{new_key}")

    assert result.error is None
    assert sandbox._file_store["/test/file.txt"] == b"value = {new_key}"


# -- remaining template tests --------------------------------------------------


def test_sandbox_read_uses_payload() -> None:
    """Test that read() bundles all params into a single base64 payload."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"content": "mock content", "encoding": "utf-8"})

    sandbox.read("/test/file.txt", offset=5, limit=50)

    assert sandbox.last_command is not None
    assert "__DEEPAGENTS_EOF__" in sandbox.last_command
    assert "/test/file.txt" not in sandbox.last_command


def test_sandbox_grep_literal_search() -> None:
    """Test that grep performs literal search using grep -F flag."""
    sandbox = MockSandbox()

    # Override execute to return mock grep results
    def mock_execute(command: str, *, timeout: int | None = None) -> ExecuteResponse:  # noqa: ARG001
        sandbox.last_command = command
        # Return mock grep output for literal search tests
        if "grep" in command:
            # Check that -F flag (fixed-strings/literal) is present in the flags
            # -F can appear as standalone "-F" or combined like "-rHnF"
            assert "-F" in command or "F" in command.split("grep", 1)[1].split(maxsplit=1)[0], "grep should use -F flag for literal search"
            return ExecuteResponse(
                output="/test/code.py:1:def __init__(self):\n/test/types.py:1:str | int",
                exit_code=0,
                truncated=False,
            )
        return ExecuteResponse(output="", exit_code=0, truncated=False)

    sandbox.execute = mock_execute

    # Test with parentheses (should be literal, not regex grouping)
    matches = sandbox.grep("def __init__(", path="/test").matches
    assert matches is not None
    assert len(matches) == 2

    # Test with pipe character (should be literal, not regex OR)
    matches = sandbox.grep("str | int", path="/test").matches
    assert matches is not None

    # Verify the command uses grep -rHnF for literal search (combined flags)
    assert sandbox.last_command is not None
    assert "grep -rHnF" in sandbox.last_command


# -- upload/download failure tests --------------------------------------------


def test_sandbox_write_returns_error_on_upload_failure() -> None:
    """Test that write() surfaces upload_files errors."""
    sandbox = MockSandbox()

    def failing_upload(
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=files[0][0], error="permission_denied")]

    sandbox.upload_files = failing_upload  # type: ignore[assignment]

    result = sandbox.write("/test/file.txt", "content")

    assert result.error is not None
    assert "Failed to write" in result.error
    assert result.path is None


def test_sandbox_edit_returns_error_on_upload_failure() -> None:
    """Test that edit() surfaces upload_files errors during re-upload."""
    sandbox = MockSandbox()
    sandbox._file_store["/test/file.txt"] = b"hello old world"

    original_upload = sandbox.upload_files

    def failing_upload(
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=files[0][0], error="permission_denied")]

    sandbox.upload_files = failing_upload  # type: ignore[assignment]

    result = sandbox.edit("/test/file.txt", "old", "new")

    assert result.error is not None
    assert "Error editing file" in result.error
    sandbox.upload_files = original_upload  # restore


def test_sandbox_edit_binary_file_returns_error() -> None:
    """Test that edit() returns a clear error for non-UTF-8 files."""
    sandbox = MockSandbox()
    sandbox._file_store["/test/binary.bin"] = b"\x80\x81\x82\xff"

    result = sandbox.edit("/test/binary.bin", "old", "new")

    assert result.error is not None
    assert "not a text file" in result.error


def test_sandbox_edit_does_not_embed_content_in_command() -> None:
    """Test that edit() uses download/upload, not execute(), for data transfer."""
    sandbox = MockSandbox()
    large = "x" * 500_000
    sandbox._file_store["/test/big.txt"] = large.encode("utf-8")
    sandbox.last_command = None  # reset

    sandbox.edit("/test/big.txt", "x", "y", replace_all=True)

    # edit() should never call execute() — only download_files + upload_files
    assert sandbox.last_command is None
