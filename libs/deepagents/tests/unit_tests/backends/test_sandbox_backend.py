"""Tests for BaseSandbox backend template formatting.

These tests verify that the command templates in BaseSandbox can be properly
formatted without raising KeyError due to unescaped curly braces.

Related issue: https://github.com/langchain-ai/deepagents/pull/872
The heredoc templates introduced in PR #872 contain {e} in exception handlers
that need to be escaped as {{e}} for Python's .format() method.
"""

import base64
import json

from deepagents.backends.protocol import ExecuteResponse
from deepagents.backends.sandbox import (
    _EDIT_COMMAND_TEMPLATE,
    _GLOB_COMMAND_TEMPLATE,
    _READ_COMMAND_TEMPLATE,
    _WRITE_COMMAND_TEMPLATE,
    BaseSandbox,
)


class MockSandbox(BaseSandbox):
    """Minimal concrete implementation of BaseSandbox for testing."""

    def __init__(self) -> None:
        self.last_command = None

    @property
    def id(self) -> str:
        return "mock-sandbox"

    def execute(self, command: str) -> ExecuteResponse:
        self.last_command = command
        # Return "1" for edit commands (simulates 1 occurrence replaced)
        return ExecuteResponse(output="1", exit_code=0, truncated=False)


def test_write_command_template_format() -> None:
    """Test that _WRITE_COMMAND_TEMPLATE can be formatted without KeyError."""
    content = "test content with special chars: {curly} and 'quotes'"
    content_b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    payload = json.dumps({"path": "/test/file.txt", "content": content_b64})
    payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")

    # This should not raise KeyError
    cmd = _WRITE_COMMAND_TEMPLATE.format(payload_b64=payload_b64)

    assert "python3 -c" in cmd
    assert payload_b64 in cmd


def test_edit_command_template_format() -> None:
    """Test that _EDIT_COMMAND_TEMPLATE can be formatted without KeyError."""
    payload = json.dumps({"path": "/test/file.txt", "old": "foo", "new": "bar"})
    payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")

    # This should not raise KeyError
    cmd = _EDIT_COMMAND_TEMPLATE.format(payload_b64=payload_b64, replace_all=False)

    assert "python3 -c" in cmd
    assert payload_b64 in cmd


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
    cmd = _READ_COMMAND_TEMPLATE.format(file_path="/test/file.txt", offset=0, limit=100)

    assert "python3 -c" in cmd
    assert "/test/file.txt" in cmd


def test_sandbox_write_method() -> None:
    """Test that BaseSandbox.write() successfully formats the command."""
    sandbox = MockSandbox()

    # This should not raise KeyError
    sandbox.write("/test/file.txt", "test content")

    # The command should have been formatted and passed to execute()
    assert sandbox.last_command is not None
    assert "python3 -c" in sandbox.last_command


def test_sandbox_edit_method() -> None:
    """Test that BaseSandbox.edit() successfully formats the command."""
    sandbox = MockSandbox()

    # This should not raise KeyError
    sandbox.edit("/test/file.txt", "old", "new", replace_all=False)

    # The command should have been formatted and passed to execute()
    assert sandbox.last_command is not None
    assert "python3 -c" in sandbox.last_command


def test_sandbox_write_with_special_content() -> None:
    """Test write with content containing curly braces and special characters."""
    sandbox = MockSandbox()

    # Content with curly braces that could confuse format()
    content = "def foo(): return {key: value for key, value in items.items()}"

    sandbox.write("/test/code.py", content)

    assert sandbox.last_command is not None


def test_sandbox_edit_with_special_strings() -> None:
    """Test edit with strings containing curly braces."""
    sandbox = MockSandbox()

    old_string = "{old_key}"
    new_string = "{new_key}"

    sandbox.edit("/test/file.txt", old_string, new_string, replace_all=True)

    assert sandbox.last_command is not None


def test_sandbox_grep_literal_search() -> None:
    """Test that grep performs literal search using grep -F flag."""
    sandbox = MockSandbox()

    # Override execute to return mock grep results
    def mock_execute(command: str) -> ExecuteResponse:
        sandbox.last_command = command
        # Return mock grep output for literal search tests
        if "grep" in command:
            # Check that -F flag (fixed-strings/literal) is present in the flags
            # -F can appear as standalone "-F" or combined like "-rHnF"
            assert "-F" in command or "F" in command.split("grep")[1].split()[0], "grep should use -F flag for literal search"
            return ExecuteResponse(
                output="/test/code.py:1:def __init__(self):\n/test/types.py:1:str | int",
                exit_code=0,
                truncated=False,
            )
        return ExecuteResponse(output="", exit_code=0, truncated=False)

    sandbox.execute = mock_execute

    # Test with parentheses (should be literal, not regex grouping)
    matches = sandbox.grep_raw("def __init__(", path="/test")
    assert isinstance(matches, list)
    assert len(matches) == 2

    # Test with pipe character (should be literal, not regex OR)
    matches = sandbox.grep_raw("str | int", path="/test")
    assert isinstance(matches, list)

    # Verify the command uses grep -rHnF for literal search (combined flags)
    assert sandbox.last_command is not None
    assert "grep -rHnF" in sandbox.last_command


# -- Upload/download tests ------------------------------------------------


def test_upload_small_file_uses_single_command() -> None:
    """Test that small files are uploaded in a single execute() call."""
    commands: list[str] = []

    class TrackingSandbox(MockSandbox):
        def execute(self, command: str) -> ExecuteResponse:
            commands.append(command)
            return ExecuteResponse(output="", exit_code=0, truncated=False)

    sandbox = TrackingSandbox()
    small_content = b"hello world"
    sandbox.upload_files([("/test.txt", small_content)])

    # Should be exactly one command (no chunking).
    assert len(commands) == 1
    assert "__DEEPAGENTS_EOF__" in commands[0]


def test_upload_large_file_uses_chunked_commands() -> None:
    """Test that large files are split into multiple execute() calls."""
    commands: list[str] = []

    class TrackingSandbox(MockSandbox):
        def execute(self, command: str) -> ExecuteResponse:
            commands.append(command)
            return ExecuteResponse(output="", exit_code=0, truncated=False)

    sandbox = TrackingSandbox()
    # Create content larger than _UPLOAD_CHUNK_BYTES (64KB).
    large_content = b"x" * 100_000
    sandbox.upload_files([("/large.bin", large_content)])

    # Should have: 1 mkdir + N chunk appends + 1 decode = more than 1 command.
    assert len(commands) > 1
    # Last command should be the decode step.
    assert "base64.b64decode" in commands[-1]
    assert "unlink" in commands[-1]


def test_upload_returns_error_on_failure() -> None:
    """Test that upload returns error when execute() fails."""

    class FailingSandbox(MockSandbox):
        def execute(self, command: str) -> ExecuteResponse:
            return ExecuteResponse(output="some error", exit_code=1, truncated=False)

    sandbox = FailingSandbox()
    responses = sandbox.upload_files([("/fail.txt", b"content")])

    assert len(responses) == 1
    assert responses[0].error == "permission_denied"


def test_download_small_file() -> None:
    """Test downloading a small file in a single command."""
    content = b"hello world"
    b64_content = base64.b64encode(content).decode("ascii")
    call_count = 0

    class DownloadSandbox(MockSandbox):
        def execute(self, command: str) -> ExecuteResponse:
            nonlocal call_count
            call_count += 1
            if "getsize" in command:
                return ExecuteResponse(output=str(len(content)), exit_code=0, truncated=False)
            return ExecuteResponse(output=b64_content, exit_code=0, truncated=False)

    sandbox = DownloadSandbox()
    responses = sandbox.download_files(["/test.txt"])

    assert len(responses) == 1
    assert responses[0].content == content
    assert responses[0].error is None
    # 1 size check + 1 download = 2 calls.
    assert call_count == 2


def test_download_large_file_uses_chunks() -> None:
    """Test that large files are downloaded in multiple chunks."""
    # 100KB file — larger than _DOWNLOAD_CHUNK_BYTES (64KB).
    full_content = b"A" * 100_000
    chunk_size = BaseSandbox._DOWNLOAD_CHUNK_BYTES
    call_count = 0

    class ChunkedDownloadSandbox(MockSandbox):
        def execute(self, command: str) -> ExecuteResponse:
            nonlocal call_count
            call_count += 1
            if "getsize" in command:
                return ExecuteResponse(
                    output=str(len(full_content)),
                    exit_code=0,
                    truncated=False,
                )
            if "f.seek" in command:
                # Parse offset from the command.
                offset_str = command.split("f.seek(")[1].split(")")[0]
                offset = int(offset_str)
                chunk = full_content[offset : offset + chunk_size]
                b64_chunk = base64.b64encode(chunk).decode("ascii")
                return ExecuteResponse(output=b64_chunk, exit_code=0, truncated=False)
            return ExecuteResponse(output="", exit_code=0, truncated=False)

    sandbox = ChunkedDownloadSandbox()
    responses = sandbox.download_files(["/large.bin"])

    assert len(responses) == 1
    assert responses[0].content == full_content
    assert responses[0].error is None
    # 1 size check + 2 chunk downloads = 3 calls.
    assert call_count == 3


def test_download_returns_error_for_missing_file() -> None:
    """Test that download returns error when file does not exist."""

    class MissingSandbox(MockSandbox):
        def execute(self, command: str) -> ExecuteResponse:
            return ExecuteResponse(output="FileNotFoundError", exit_code=1, truncated=False)

    sandbox = MissingSandbox()
    responses = sandbox.download_files(["/missing.txt"])

    assert len(responses) == 1
    assert responses[0].error == "file_not_found"
    assert responses[0].content is None
