"""Tests for BaseSandbox backend operations.

Verifies that read and edit (small payload) use execute() for server-side
operations, write uses upload_files, edit (large payload) uploads old/new as
temp files with a server-side replace script, and command templates format
correctly.
"""

import base64
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

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
        # Detect temp-file upload path: upload_files() stores .deepagents_edit_*
        # keys in _file_store before execute() is called.
        has_tmp = any(".deepagents_edit_" in k for k in self._file_store)
        if "old_path = base64.b64decode(" in command and has_tmp:
            return self._simulate_edit_tmpfile(command)
        output = self._next_output
        self._next_output = "1"
        return ExecuteResponse(output=output, exit_code=0, truncated=False)

    def _simulate_edit_tmpfile(self, command: str) -> ExecuteResponse:
        """Simulate the server-side temp-file edit script.

        Reads temp file entries placed in `_file_store` by `upload_files()`,
        performs the replacement on the target file, and removes the temp
        entries.
        """
        old_m = re.search(r"old_path = base64\.b64decode\('([^']+)'\)", command)
        new_m = re.search(r"new_path = base64\.b64decode\('([^']+)'\)", command)
        tgt_m = re.search(r"target = base64\.b64decode\('([^']+)'\)", command)
        ra_m = re.search(r"replace_all = (True|False)", command)
        assert old_m is not None, "Could not parse old_path from command"
        assert new_m is not None, "Could not parse new_path from command"
        assert tgt_m is not None, "Could not parse target from command"
        assert ra_m is not None, "Could not parse replace_all from command"

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
    assert "unexpected server response" in result.error
    assert "not json at all" in result.error


def test_read_handles_non_dict_json_output() -> None:
    """Test that read() returns error when execute() returns valid JSON that is not a dict."""
    sandbox = MockSandbox()
    sandbox._next_output = "[1, 2, 3]"

    result = sandbox.read("/test/file.txt")

    assert result.error is not None
    assert "unexpected server response" in result.error


def test_read_allows_truncated_paginated_output() -> None:
    """Test that read() accepts truncated paginated content returned by the server."""
    sandbox = MockSandbox()
    truncated_content = (
        "line one\n\n"
        "[Output was truncated due to size limits. Continue reading with a larger "
        "offset or smaller limit to inspect the rest of the file.]"
    )
    sandbox._next_output = json.dumps(
        {
            "encoding": "utf-8",
            "content": truncated_content,
        }
    )

    result = sandbox.read("/test/file.txt")

    assert result.error is None
    assert result.file_data == {
        "encoding": "utf-8",
        "content": truncated_content,
    }


# -- ls tests -----------------------------------------------------------------


def test_ls_returns_entries_for_directory_with_files() -> None:
    """ls() parses one JSON object per line into FileInfo entries."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"path": "/test/a.txt", "is_dir": False}) + "\n" + json.dumps({"path": "/test/sub", "is_dir": True})

    result = sandbox.ls("/test")

    assert result.error is None
    assert result.entries == [
        {"path": "/test/a.txt", "is_dir": False},
        {"path": "/test/sub", "is_dir": True},
    ]


def test_ls_empty_directory_returns_empty_entries_no_error() -> None:
    """A genuinely empty directory yields entries=[] with error=None."""
    sandbox = MockSandbox()
    sandbox._next_output = ""

    result = sandbox.ls("/test/empty")

    assert result.error is None
    assert result.entries == []


def test_ls_nonexistent_path_sets_error() -> None:
    """When the inline script reports a missing path, ls() surfaces it on .error.

    Mirrors read()/edit(): the sandbox emits a short snake_case code; the host
    wraps it with the path prefix.
    """
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"error": "path_not_found"})

    result = sandbox.ls("/test/does_not_exist")

    assert result.entries is None
    assert result.error == "Path '/test/does_not_exist': path_not_found"


def test_ls_permission_denied_sets_error() -> None:
    """When the inline script reports permission denied, ls() surfaces it on .error."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"error": "permission_denied"})

    result = sandbox.ls("/test/locked")

    assert result.entries is None
    assert result.error == "Path '/test/locked': permission_denied"


def test_ls_not_a_directory_sets_error() -> None:
    """When the inline script reports the path is a file, ls() surfaces it on .error."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"error": "not_a_directory"})

    result = sandbox.ls("/test/file.txt")

    assert result.entries is None
    assert result.error == "Path '/test/file.txt': not_a_directory"


def test_ls_error_line_amongst_entries_takes_precedence() -> None:
    """If any line is an error record, entries=None and error is reported.

    A top-level scandir failure raises before any entries are emitted, but
    `entry.is_dir()` can raise per-entry after some entries have already been
    printed (e.g., a child becomes unreadable mid-scan). The parser is
    defensive: any error line wins over partial entries, preserving the
    documented contract (entries=None on failure).
    """
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"path": "/test/a.txt", "is_dir": False}) + "\n" + json.dumps({"error": "permission_denied"})

    result = sandbox.ls("/test")

    assert result.entries is None
    assert result.error == "Path '/test': permission_denied"


def test_ls_command_base64_encodes_path() -> None:
    """The path is base64-encoded into the inline script to prevent injection."""
    sandbox = MockSandbox()
    sandbox._next_output = ""

    sandbox.ls("/test/dir")

    assert sandbox.last_command is not None
    expected_b64 = base64.b64encode(b"/test/dir").decode("ascii")
    assert expected_b64 in sandbox.last_command
    assert "python3 -c" in sandbox.last_command


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
    assert "not found" in result.error.lower()


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
    assert "unexpected server response" in result.error
    assert "not json at all" in result.error


def test_sandbox_edit_inline_non_dict_json_output() -> None:
    """Test that inline edit returns error when execute() returns non-dict JSON."""
    sandbox = MockSandbox()
    sandbox._next_output = "[1, 2, 3]"

    result = sandbox.edit("/test/file.txt", "old", "new")

    assert result.error is not None
    assert "unexpected server response" in result.error


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


def test_sandbox_edit_upload_basic() -> None:
    """Test that edit() with large strings uses temp-file upload + server-side replace."""
    sandbox = MockSandbox()
    large_old = "x" * (_EDIT_INLINE_MAX_BYTES + 1)
    sandbox._file_store["/test/file.txt"] = f"prefix {large_old} suffix".encode()

    result = sandbox.edit("/test/file.txt", large_old, "new")

    assert result.error is None
    assert result.occurrences == 1
    assert sandbox._file_store["/test/file.txt"] == b"prefix new suffix"


def test_sandbox_edit_upload_file_not_found() -> None:
    """Test that upload-path edit returns error when file doesn't exist."""
    sandbox = MockSandbox()
    large_old = "x" * (_EDIT_INLINE_MAX_BYTES + 1)

    result = sandbox.edit("/test/missing.txt", large_old, "new")

    assert result.error is not None
    assert "not found" in result.error.lower()


def test_sandbox_edit_upload_replace_all() -> None:
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


def test_sandbox_edit_upload_cleans_up_temp_files() -> None:
    """Test that temp files are removed from the sandbox after a successful edit."""
    sandbox = MockSandbox()
    large_old = "x" * (_EDIT_INLINE_MAX_BYTES + 1)
    sandbox._file_store["/test/file.txt"] = f"prefix {large_old} suffix".encode()

    result = sandbox.edit("/test/file.txt", large_old, "new")

    assert result.error is None
    assert not any(k.startswith("/tmp/.deepagents_edit_") for k in sandbox._file_store)  # noqa: S108


def test_sandbox_edit_upload_string_not_found() -> None:
    """Test that upload-path edit returns error when old_string is absent."""
    sandbox = MockSandbox()
    large_old = "x" * (_EDIT_INLINE_MAX_BYTES + 1)
    sandbox._file_store["/test/file.txt"] = b"completely different content"

    result = sandbox.edit("/test/file.txt", large_old, "new")

    assert result.error is not None
    assert "not found" in result.error.lower()


def test_sandbox_edit_upload_multiple_occurrences_without_replace_all() -> None:
    """Test that upload-path edit errors on multiple matches without replace_all."""
    sandbox = MockSandbox()
    large_old = "x" * (_EDIT_INLINE_MAX_BYTES + 1)
    sandbox._file_store["/test/file.txt"] = f"a{large_old}b{large_old}c".encode()

    result = sandbox.edit("/test/file.txt", large_old, "y")

    assert result.error is not None
    assert "multiple times" in result.error.lower()


def test_sandbox_edit_upload_partial_upload_failure() -> None:
    """Test that upload-path edit surfaces error when one of two uploads fails."""
    sandbox = MockSandbox()
    large_old = "x" * (_EDIT_INLINE_MAX_BYTES + 1)
    sandbox._file_store["/test/file.txt"] = f"prefix {large_old} suffix".encode()

    def partial_failure(
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        return [
            FileUploadResponse(path=files[0][0], error=None),
            FileUploadResponse(path=files[1][0], error="disk_full"),
        ]

    sandbox.upload_files = partial_failure  # type: ignore[assignment]

    result = sandbox.edit("/test/file.txt", large_old, "new")

    assert result.error is not None
    assert "disk_full" in result.error


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


def test_edit_command_template_ends_with_newline() -> None:
    """Test that _EDIT_COMMAND_TEMPLATE preserves the trailing newline after EOF."""
    assert _EDIT_COMMAND_TEMPLATE.endswith("\n")


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


def test_sandbox_grep_quotes_include_glob() -> None:
    """Test that grep shell-quotes the include glob pattern."""
    sandbox = MockSandbox()

    def mock_execute(command: str, *, timeout: int | None = None) -> ExecuteResponse:  # noqa: ARG001
        sandbox.last_command = command
        return ExecuteResponse(output="", exit_code=0, truncated=False)

    sandbox.execute = mock_execute

    sandbox.grep("needle", path="/test", glob="x' ; echo injected ; #")

    assert sandbox.last_command is not None
    assert "--include='x'\"'\"' ; echo injected ; #'" in sandbox.last_command
    assert "--include='x'\"'\"' ; echo injected ; #' -e needle /test" in sandbox.last_command


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


def test_sandbox_edit_upload_returns_error_on_upload_failure() -> None:
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


def test_sandbox_edit_upload_binary_file_returns_error() -> None:
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


def test_sandbox_edit_upload_returns_error_on_empty_upload_response() -> None:
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


# -- boundary + catch-all tests ------------------------------------------------


def test_sandbox_edit_at_exact_threshold_uses_inline() -> None:
    """Test that payload of exactly _EDIT_INLINE_MAX_BYTES uses the inline path."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"count": 1})
    # Create old+new whose combined UTF-8 byte length == threshold
    old = "x" * _EDIT_INLINE_MAX_BYTES
    new = ""

    result = sandbox.edit("/test/file.txt", old, new)

    assert result.error is None
    assert len(sandbox._uploaded) == 0  # inline path — no uploads


def test_sandbox_edit_one_over_threshold_uses_upload() -> None:
    """Test that payload of _EDIT_INLINE_MAX_BYTES + 1 uses the upload path."""
    sandbox = MockSandbox()
    old = "x" * (_EDIT_INLINE_MAX_BYTES + 1)
    sandbox._file_store["/test/file.txt"] = f"prefix {old} suffix".encode()

    result = sandbox.edit("/test/file.txt", old, "new")

    assert result.error is None
    assert len(sandbox._uploaded) > 0  # upload path — temp files uploaded


def test_map_edit_error_unknown_code_falls_through() -> None:
    """Test that _map_edit_error returns a generic error for unrecognized codes."""
    result = BaseSandbox._map_edit_error("temp_read_failed", "/test/file.txt", "old")

    assert result.error is not None
    assert "temp_read_failed" in result.error
    assert "/test/file.txt" in result.error


def test_sandbox_edit_upload_malformed_output_cleans_up() -> None:
    """Test that upload-path edit cleans up temp files on malformed output."""
    sandbox = MockSandbox()
    large_old = "x" * (_EDIT_INLINE_MAX_BYTES + 1)
    sandbox._file_store["/test/file.txt"] = f"prefix {large_old} suffix".encode()
    cleanup_commands: list[str] = []

    original_execute = sandbox.execute

    def tracking_execute(command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if command.startswith("rm -f"):
            cleanup_commands.append(command)
            return ExecuteResponse(output="", exit_code=0)
        # For the edit command, return malformed output
        resp = original_execute(command, timeout=timeout)
        if "old_path = base64.b64decode(" in command:
            return ExecuteResponse(output="crash traceback", exit_code=1)
        return resp

    sandbox.execute = tracking_execute  # type: ignore[assignment]

    result = sandbox.edit("/test/file.txt", large_old, "new")

    assert result.error is not None
    assert "unexpected server response" in result.error
    assert len(cleanup_commands) == 1
    assert ".deepagents_edit_" in cleanup_commands[0]


# -- read script binary-detection behavior -----------------------------------
# Direct execution of the formatted _READ_COMMAND_TEMPLATE script via
# subprocess. Exercises the binary-vs-text classification logic that
# _FakeSandbox-style tests cannot reach because they stub execute() output.


def _run_read_script(target: Path, *, file_type: str = "text", offset: int = 0, limit: int = 2000) -> dict:
    cmd = _READ_COMMAND_TEMPLATE.format(
        path_b64=base64.b64encode(str(target).encode("utf-8")).decode("ascii"),
        file_type=file_type,
        offset=offset,
        limit=limit,
    )
    _, _, tail = cmd.partition('python3 -c "')
    script, _, _ = tail.rpartition('" 2>&1')
    proc = subprocess.run(  # noqa: S603  # script is the project's own _READ_COMMAND_TEMPLATE, not user input
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip())


def test_read_script_cjk_at_prefix_boundary(tmp_path: Path) -> None:
    """3-byte CJK char straddling byte 8192 must classify as text, not binary."""
    target = tmp_path / "cjk.md"
    target.write_bytes((b"a" * 8190) + "가나다".encode())

    result = _run_read_script(target)

    assert result["encoding"] == "utf-8"
    assert result["content"].endswith("가나다")


@pytest.mark.parametrize("pad", [8189, 8190, 8191])
def test_read_script_emoji_at_prefix_boundary(tmp_path: Path, pad: int) -> None:
    """4-byte emoji at any sub-boundary offset must classify as text."""
    target = tmp_path / f"emoji_{pad}.md"
    target.write_bytes((b"a" * pad) + "😀tail".encode())

    result = _run_read_script(target)

    assert result["encoding"] == "utf-8"
    assert "😀tail" in result["content"]


def test_read_script_genuine_binary_returns_base64(tmp_path: Path) -> None:
    target = tmp_path / "bin.dat"
    target.write_bytes(b"\x00\x01\x02\xff\xfe" * 2000)

    result = _run_read_script(target)

    assert result["encoding"] == "base64"


def test_read_script_mid_buffer_invalid_utf8_returns_base64(tmp_path: Path) -> None:
    """Corruption inside the prefix must still route to base64 (not swallowed)."""
    target = tmp_path / "midbad.dat"
    target.write_bytes(b"a" * 100 + b"\xff\xff" + b"a" * 9000)

    result = _run_read_script(target)

    assert result["encoding"] == "base64"


def test_read_script_ascii_larger_than_prefix(tmp_path: Path) -> None:
    """Pure-ASCII control: file >8192 bytes must classify as text."""
    target = tmp_path / "ascii.txt"
    target.write_bytes(b"hello\n" * 2000)

    result = _run_read_script(target)

    assert result["encoding"] == "utf-8"


# -- script-level permission/error tests --------------------------------------
# Direct subprocess runs of read/edit/glob inline scripts on the local FS,
# exercising the OSError-handling branches that mock-bypassed tests cannot
# reach.


_PERMISSION_DENIED_SKIP = pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="chmod 000 does not deny access on Windows or as root",
)


def _run_edit_script(
    path: Path,
    old: str,
    new: str,
    replace_all: bool = False,  # noqa: FBT001, FBT002
) -> dict:
    payload = json.dumps({"path": str(path), "old": old, "new": new, "replace_all": replace_all})
    payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    cmd = _EDIT_COMMAND_TEMPLATE.format(payload_b64=payload_b64)
    _, _, tail = cmd.partition('python3 -c "')
    script, _, _ = tail.partition('" 2>&1')
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        input=payload_b64,
        capture_output=True,
        text=True,
        check=False,
    )
    return json.loads(proc.stdout.strip())


def _run_glob_script(path: Path, pattern: str) -> str:
    cmd = _GLOB_COMMAND_TEMPLATE.format(
        path_b64=base64.b64encode(str(path).encode("utf-8")).decode("ascii"),
        pattern_b64=base64.b64encode(pattern.encode("utf-8")).decode("ascii"),
    )
    _, _, tail = cmd.partition('python3 -c "')
    script, _, _ = tail.partition('" 2>&1')
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout


@_PERMISSION_DENIED_SKIP
def test_read_script_permission_denied(tmp_path: Path) -> None:
    """Read script must surface permission_denied, not crash with a traceback."""
    target = tmp_path / "locked.txt"
    target.write_text("secret")
    target.chmod(0o000)
    try:
        result = _run_read_script(target)
        assert result == {"error": "permission_denied"}
    finally:
        target.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_read_script_is_a_directory(tmp_path: Path) -> None:
    """Read script must surface a structured error when path is a directory."""
    result = _run_read_script(tmp_path)
    assert result.get("error") == "not_a_file"


@_PERMISSION_DENIED_SKIP
def test_edit_script_permission_denied(tmp_path: Path) -> None:
    """Edit script must surface permission_denied, not crash with a traceback."""
    target = tmp_path / "locked.txt"
    target.write_text("old content")
    target.chmod(0o000)
    try:
        result = _run_edit_script(target, "old", "new")
        assert result == {"error": "permission_denied"}
    finally:
        target.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_glob_script_path_not_found(tmp_path: Path) -> None:
    """Glob script must surface path_not_found instead of crashing on chdir."""
    missing = tmp_path / "does_not_exist"
    output = _run_glob_script(missing, "*.py")
    data = json.loads(output.strip().split("\n")[0])
    assert data == {"error": "path_not_found"}


@_PERMISSION_DENIED_SKIP
def test_glob_script_permission_denied(tmp_path: Path) -> None:
    """Glob script must surface permission_denied instead of crashing on chdir."""
    locked = tmp_path / "locked_dir"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        output = _run_glob_script(locked, "*.py")
        data = json.loads(output.strip().split("\n")[0])
        assert data == {"error": "permission_denied"}
    finally:
        locked.chmod(stat.S_IRWXU)


# -- glob host-side error surfacing -------------------------------------------


def test_glob_surfaces_error_from_script() -> None:
    """When the inline script emits an error JSON line, GlobResult.error is set.

    Mirrors read()/ls() convention: sandbox emits a short code; host wraps
    with the path prefix and reports entries=None.
    """
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"error": "permission_denied"})

    result = sandbox.glob("*.py", path="/locked")

    assert result.matches is None
    assert result.error == "Path '/locked': permission_denied"


def test_glob_path_not_found_sets_error() -> None:
    """Glob path_not_found code is wrapped with the search path."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"error": "path_not_found"})

    result = sandbox.glob("*.py", path="/missing")

    assert result.matches is None
    assert result.error == "Path '/missing': path_not_found"


def test_glob_empty_returns_empty_matches() -> None:
    """Empty stdout still means a successful, empty search."""
    sandbox = MockSandbox()
    sandbox._next_output = ""

    result = sandbox.glob("*.py", path="/some/dir")

    assert result.error is None
    assert result.matches == []


# -- _map_edit_error coverage for new codes -----------------------------------


def test_map_edit_error_permission_denied() -> None:
    """_map_edit_error returns a readable message for permission_denied."""
    result = BaseSandbox._map_edit_error("permission_denied", "/test/file.txt", "old")
    assert result.error is not None
    assert "permission" in result.error.lower()
    assert "/test/file.txt" in result.error


# -- read host-side error wrapping for new codes -----------------------------


def test_read_permission_denied_surfaces_error() -> None:
    """read() wraps permission_denied from the inline script onto ReadResult.error."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"error": "permission_denied"})

    result = sandbox.read("/test/locked.txt")

    assert result.file_data is None
    assert result.error is not None
    assert "permission_denied" in result.error
    assert "/test/locked.txt" in result.error


def test_sandbox_edit_inline_permission_denied() -> None:
    """edit() (inline) wraps permission_denied from the inline script."""
    sandbox = MockSandbox()
    sandbox._next_output = json.dumps({"error": "permission_denied"})

    result = sandbox.edit("/test/locked.txt", "old", "new")

    assert result.error is not None
    assert "permission" in result.error.lower()
    assert "/test/locked.txt" in result.error
