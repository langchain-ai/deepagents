"""Tests for BaseSandbox backend template formatting.

These tests verify that the command templates in BaseSandbox can be properly
formatted without raising KeyError due to unescaped curly braces.

Related issue: https://github.com/langchain-ai/deepagents/pull/872
The heredoc templates introduced in PR #872 contain {e} in exception handlers
that need to be escaped as {{e}} for Python's .format() method.
"""

import base64
import json

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
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

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        self.last_command = command
        # Return "1" for edit commands (simulates 1 occurrence replaced)
        return ExecuteResponse(output="1", exit_code=0, truncated=False)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=f[0], error=None) for f in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return [FileDownloadResponse(path=p, content=None, error="not_implemented") for p in paths]


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
