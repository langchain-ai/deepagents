"""Tests for BaseSandbox backend operations.

Verifies that read and edit (small payload) use execute() for server-side
operations, write uses upload_files, edit (large payload) uploads old/new as
temp files with a server-side replace script, and command templates format
correctly.
"""

import base64
import json
import re

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import (
    _EDIT_COMMAND_TEMPLATE,
    _EDIT_INLINE_MAX_BYTES,
    _EDIT_TMPFILE_TEMPLATE,
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
        # Simulate the _EDIT_TMPFILE_TEMPLATE server-side script
        if "old_path = base64.b64decode(" in command and ".deepagents_edit_" in str(self._file_store):
            return self._simulate_edit_tmpfile(command)
        output = self._next_output
        self._next_output = "1"
        return ExecuteResponse(output=output, exit_code=0, truncated=False)

    def _simulate_edit_tmpfile(self, command: str) -> ExecuteResponse:
        """Simulate the _EDIT_TMPFILE_TEMPLATE server-side script."""
        old_m = re.search(r"old_path = base64\.b64decode\('([^']+)'\)", command)
        new_m = re.search(r"new_path = base64\.b64decode\('([^']+)'\)", command)
        tgt_m = re.search(r"target = base64\.b64decode\('([^']+)'\)", command)
        ra_m = re.search(r"replace_all = (True|False)", command)

        old_path = base64.b64decode(old_m.group(1)).decode("utf-8")
        new_path = base64.b64decode(new_m.group(1)).decode("utf-8")
        target = base64.b64decode(tgt_m.group(1)).decode("utf-8")
        replace_all = ra_m.group(1) == "True"

        old_data = self._file_store.pop(old_path, None)
        new_data = self._file_store.pop(new_path, None)
        if old_data is None or new_data is None:
            return ExecuteResponse(output=json.dumps({"error": "temp_file_not_found"}), exit_code=0)

        old_str = old_data.decode("utf-8")
        new_str = new_data.decode("utf-8")

        if target not in self._file_store:
            return ExecuteResponse(output=json.dumps({"error": "file_not_found"}), exit_code=0)

        raw = self._file_store[target]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return ExecuteResponse(output=json.dumps({"error": "not_a_text_file"}), exit_code=0)

        count = text.count(old_str)
        if count == 0:
            return ExecuteResponse(output=json.dumps({"error": "string_not_found"}), exit_code=0)
        if count > 1 and not replace_all:
            return ExecuteResponse(output=json.dumps({"error": "multiple_occurrences", "count": count}), exit_code=0)

        result = text.replace(old_str, new_str) if replace_all else text.replace(old_str, new_str, 1)
        self._file_store[target] = result.encode("utf-8")
        return ExecuteResponse(output=json.dumps({"count": count}), exit_code=0)

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


def test_read_uses_execute() -> None:
    """Test that read() delegates to execute() for server-side pagination."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"encoding": "utf-8", "content": "line one\nline two\nline three"})

    result = sandbox.read("/test/file.txt")

    assert result.error is None
    assert result.file_data is not None
    assert "line one" in result.file_data["content"]
    # read() should call execute()
    assert sandbox.last_command is not None


def test_read_parses_offset_and_limit() -> None:
    """Test that read() parses paginated content from the server-side script."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"encoding": "utf-8", "content": "b\nc"})

    result = sandbox.read("/test/file.txt", offset=1, limit=2)

    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["content"] == "b\nc"


def test_read_offset_exceeds_length() -> None:
    """Test that read() returns error when offset exceeds file length."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"error": "Line offset 5 exceeds file length (1 lines)"})

    result = sandbox.read("/test/file.txt", offset=5)

    assert result.error is not None
    assert "exceeds file length" in result.error


def test_read_empty_file() -> None:
    """Test that read() returns a sentinel message for empty files."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps(
        {
            "encoding": "utf-8",
            "content": "System reminder: File exists but has empty contents",
        }
    )

    result = sandbox.read("/test/empty.txt")

    assert result.error is None
    assert result.file_data is not None
    assert "empty contents" in result.file_data["content"]


def test_read_binary_file() -> None:
    """Test that read() returns base64-encoded content for non-UTF-8 files."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"encoding": "base64", "content": "gIGC/w=="})

    result = sandbox.read("/test/binary.bin")

    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["encoding"] == "base64"


def test_read_file_not_found() -> None:
    """Test that read() returns error for missing files."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"error": "file_not_found"})

    result = sandbox.read("/test/missing.txt")

    assert result.error is not None
    assert "file_not_found" in result.error


def test_read_handles_malformed_output() -> None:
    """Test that read() gracefully handles non-JSON output from execute()."""
    sandbox = MockSandbox()
    sandbox._next_output = "not json at all"

    result = sandbox.read("/test/file.txt")

    assert result.error is not None
    assert "not found" in result.error


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

    def fail_execute(command: str, *, timeout: int | None = None) -> ExecuteResponse:  # noqa: ARG001
        sandbox.last_command = command
        return ExecuteResponse(output="Error: File already exists", exit_code=1)

    sandbox.execute = fail_execute

    result = sandbox.write("/test/existing.txt", "content")
    assert result.error is not None
    assert "Error:" in result.error
    assert len(sandbox._uploaded) == 0  # upload should not have been called


# -- edit tests (inline path: small strings → execute()) ----------------------


def test_sandbox_edit_inline_basic() -> None:
    """Test that edit() with small strings uses execute() for server-side replace."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"count": 1})

    result = sandbox.edit("/test/file.txt", "old", "new")

    assert result.error is None
    assert result.occurrences == 1
    # Should have called execute(), not download_files
    assert sandbox.last_command is not None
    assert len(sandbox._uploaded) == 0


def test_sandbox_edit_inline_file_not_found() -> None:
    """Test that inline edit returns error when file doesn't exist."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"error": "file_not_found"})

    result = sandbox.edit("/test/missing.txt", "old", "new")

    assert result.error is not None
    assert "Failed to read" in result.error


def test_sandbox_edit_inline_string_not_found() -> None:
    """Test that inline edit returns error when old_string is not in the file."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"error": "string_not_found"})

    result = sandbox.edit("/test/file.txt", "missing", "new")

    assert result.error is not None
    assert "not found" in result.error


def test_sandbox_edit_inline_multiple_occurrences() -> None:
    """Test that inline edit errors on multiple occurrences without replace_all."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"error": "multiple_occurrences", "count": 2})

    result = sandbox.edit("/test/file.txt", "foo", "baz")

    assert result.error is not None
    assert "multiple times" in result.error.lower()


def test_sandbox_edit_inline_replace_all() -> None:
    """Test that inline edit with replace_all returns correct count."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"count": 2})

    result = sandbox.edit("/test/file.txt", "foo", "baz", replace_all=True)

    assert result.error is None
    assert result.occurrences == 2


def test_sandbox_edit_inline_special_strings() -> None:
    """Test inline edit with strings containing curly braces."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"count": 1})

    result = sandbox.edit("/test/file.txt", "{old_key}", "{new_key}")

    assert result.error is None
    assert result.occurrences == 1


def test_sandbox_edit_inline_binary_file() -> None:
    """Test that inline edit returns error for non-UTF-8 files."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"error": "not_a_text_file"})

    result = sandbox.edit("/test/binary.bin", "old", "new")

    assert result.error is not None
    assert "not a text file" in result.error


def test_sandbox_edit_inline_malformed_output() -> None:
    """Test that inline edit handles non-JSON output from execute()."""
    sandbox = MockSandbox()
    sandbox._next_output = "not json at all"

    result = sandbox.edit("/test/file.txt", "old", "new")

    assert result.error is not None
    assert "not found" in result.error


def test_sandbox_edit_inline_does_not_download() -> None:
    """Test that inline edit never calls download_files or upload_files."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"count": 1})
    download_called = False
    original_download = sandbox.download_files

    def tracking_download(paths: list[str]) -> list[FileDownloadResponse]:
        nonlocal download_called
        download_called = True
        return original_download(paths)

    sandbox.download_files = tracking_download  # type: ignore[assignment]

    sandbox.edit("/test/file.txt", "old", "new")

    assert not download_called
    assert len(sandbox._uploaded) == 0


# -- edit tests (upload path: large strings → temp file upload + server-side replace)


def test_sandbox_edit_transfer_basic() -> None:
    """Test that edit() with large strings uses temp-file upload + server-side replace."""
    sandbox = MockSandbox()
    large_old = "x" * (_EDIT_INLINE_MAX_BYTES + 1)
    sandbox._file_store["/test/file.txt"] = f"prefix {large_old} suffix".encode()

    result = sandbox.edit("/test/file.txt", large_old, "new")

    assert result.error is None
    assert result.occurrences == 1
    assert sandbox._file_store["/test/file.txt"] == b"prefix new suffix"


def test_sandbox_edit_transfer_file_not_found() -> None:
    """Test that upload-path edit returns error when file doesn't exist."""
    sandbox = MockSandbox()
    large_old = "x" * (_EDIT_INLINE_MAX_BYTES + 1)

    result = sandbox.edit("/test/missing.txt", large_old, "new")

    assert result.error is not None
    assert "Failed to read" in result.error


def test_sandbox_edit_transfer_replace_all() -> None:
    """Test that upload-path edit with replace_all replaces all occurrences."""
    sandbox = MockSandbox()
    large_old = "x" * (_EDIT_INLINE_MAX_BYTES + 1)
    sandbox._file_store["/test/file.txt"] = f"a{large_old}b{large_old}c".encode()

    result = sandbox.edit("/test/file.txt", large_old, "y", replace_all=True)

    assert result.error is None
    assert result.occurrences == 2
    assert sandbox._file_store["/test/file.txt"] == b"aybyc"


def test_sandbox_edit_upload_does_not_embed_content_in_command() -> None:
    """Test that upload-path edit does not embed large strings in the command."""
    sandbox = MockSandbox()
    large_old = "x" * (_EDIT_INLINE_MAX_BYTES + 1)
    sandbox._file_store["/test/file.txt"] = f"prefix {large_old} suffix".encode()

    sandbox.edit("/test/file.txt", large_old, "new")

    # The execute command should NOT contain the large old/new strings —
    # only base64-encoded temp file paths (small, fixed-size).
    assert sandbox.last_command is not None
    assert large_old not in sandbox.last_command


# -- remaining template tests --------------------------------------------------


def test_read_command_template_format() -> None:
    """Test that _READ_COMMAND_TEMPLATE can be formatted without KeyError."""
    path_b64 = base64.b64encode(b"/test/file.txt").decode("ascii")
    cmd = _READ_COMMAND_TEMPLATE.format(
        path_b64=path_b64,
        file_type="text",
        offset=0,
        limit=2000,
    )

    assert "python3 -c" in cmd
    assert path_b64 in cmd


def test_edit_command_template_format() -> None:
    """Test that _EDIT_COMMAND_TEMPLATE can be formatted without KeyError."""
    payload_b64 = base64.b64encode(b'{"path":"/f","old":"a","new":"b"}').decode("ascii")
    cmd = _EDIT_COMMAND_TEMPLATE.format(payload_b64=payload_b64)

    assert "python3 -c" in cmd
    assert payload_b64 in cmd
    assert "__DEEPAGENTS_EDIT_EOF__" in cmd


def test_edit_tmpfile_template_format() -> None:
    """Test that _EDIT_TMPFILE_TEMPLATE can be formatted without KeyError."""
    old_b64 = base64.b64encode(b"/tmp/old").decode("ascii")
    new_b64 = base64.b64encode(b"/tmp/new").decode("ascii")
    tgt_b64 = base64.b64encode(b"/test/file.txt").decode("ascii")

    cmd = _EDIT_TMPFILE_TEMPLATE.format(
        old_path_b64=old_b64,
        new_path_b64=new_b64,
        target_b64=tgt_b64,
        replace_all=False,
    )

    assert "python3 -c" in cmd
    assert old_b64 in cmd
    assert new_b64 in cmd
    assert tgt_b64 in cmd


def test_sandbox_read_embeds_b64_path_not_raw() -> None:
    """Test that read() uses base64-encoded path, not raw path in execute()."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"encoding": "utf-8", "content": "content"})

    sandbox.read("/test/file.txt", offset=0, limit=50)

    # read() should call execute() with base64-encoded path
    assert sandbox.last_command is not None
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


def test_sandbox_edit_transfer_returns_error_on_upload_failure() -> None:
    """Test that upload-path edit surfaces upload_files errors."""
    sandbox = MockSandbox()
    large_old = "x" * (_EDIT_INLINE_MAX_BYTES + 1)
    sandbox._file_store["/test/file.txt"] = f"prefix {large_old} suffix".encode()

    def failing_upload(
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=f[0], error="permission_denied") for f in files]

    sandbox.upload_files = failing_upload  # type: ignore[assignment]

    result = sandbox.edit("/test/file.txt", large_old, "new")

    assert result.error is not None
    assert "Error editing file" in result.error


def test_sandbox_edit_transfer_binary_file_returns_error() -> None:
    """Test that upload-path edit returns a clear error for non-UTF-8 files."""
    sandbox = MockSandbox()
    large_old = "x" * (_EDIT_INLINE_MAX_BYTES + 1)
    sandbox._file_store["/test/binary.bin"] = b"\x80\x81\x82\xff"

    result = sandbox.edit("/test/binary.bin", large_old, "new")

    assert result.error is not None
    assert "not a text file" in result.error


def test_sandbox_write_returns_correct_result_on_success() -> None:
    """Test that write() returns a well-formed WriteResult on success."""
    sandbox = MockSandbox()

    result = sandbox.write("/test/file.txt", "content")

    assert result.error is None
    assert result.path == "/test/file.txt"
    assert result.files_update is None


def test_sandbox_write_returns_error_on_empty_upload_response() -> None:
    """Test that write() handles upload_files returning an empty list."""
    sandbox = MockSandbox()
    sandbox.upload_files = lambda _files: []  # type: ignore[assignment]

    result = sandbox.write("/test/file.txt", "content")

    assert result.error is not None
    assert "no response" in result.error


def test_sandbox_edit_transfer_returns_error_on_empty_upload_response() -> None:
    """Test that upload-path edit handles upload_files returning empty list."""
    sandbox = MockSandbox()
    large_old = "x" * (_EDIT_INLINE_MAX_BYTES + 1)
    sandbox._file_store["/test/file.txt"] = f"prefix {large_old} suffix".encode()
    sandbox.upload_files = lambda _files: []  # type: ignore[assignment]

    result = sandbox.edit("/test/file.txt", large_old, "new")

    assert result.error is not None
    assert "no response" in result.error


def test_sandbox_edit_upload_surfaces_upload_error_code() -> None:
    """Test that upload-path edit includes error code from upload_files."""
    sandbox = MockSandbox()
    large_old = "x" * (_EDIT_INLINE_MAX_BYTES + 1)

    def upload_with_error(
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        return [
            FileUploadResponse(path=files[0][0], error="permission_denied"),
            FileUploadResponse(path=files[1][0], error="permission_denied"),
        ]

    sandbox.upload_files = upload_with_error  # type: ignore[assignment]

    result = sandbox.edit("/test/file.txt", large_old, "new")

    assert result.error is not None
    assert "permission_denied" in result.error
