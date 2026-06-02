"""Tests for LangSmithSandbox backend."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

from langsmith.sandbox import ResourceNotFoundError, SandboxClientError

from deepagents.backends import sandbox as base_sandbox
from deepagents.backends.langsmith import LangSmithSandbox
from deepagents.backends.sandbox import MAX_BINARY_BYTES, MAX_OUTPUT_BYTES, TRUNCATION_MSG


def _make_sandbox() -> tuple[LangSmithSandbox, MagicMock]:
    mock_sdk = MagicMock()
    mock_sdk.name = "test-sandbox"
    sb = LangSmithSandbox(sandbox=mock_sdk)
    return sb, mock_sdk


def test_id_returns_sandbox_name() -> None:
    sb, _ = _make_sandbox()
    assert sb.id == "test-sandbox"


def test_execute_returns_stdout() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run.return_value = SimpleNamespace(stdout="hello world", stderr="", exit_code=0)

    result = sb.execute("echo hello world")

    assert result.output == "hello world"
    assert result.exit_code == 0
    assert result.truncated is False
    mock_sdk.run.assert_called_once_with("echo hello world", timeout=30 * 60)


def test_execute_combines_stdout_and_stderr() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run.return_value = SimpleNamespace(stdout="out", stderr="err", exit_code=1)

    result = sb.execute("failing-cmd")

    assert result.output == "out\nerr"
    assert result.exit_code == 1


def test_execute_stderr_only() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run.return_value = SimpleNamespace(stdout="", stderr="error msg", exit_code=1)

    result = sb.execute("bad-cmd")

    assert result.output == "error msg"


def test_execute_with_explicit_timeout() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run.return_value = SimpleNamespace(stdout="ok", stderr="", exit_code=0)

    sb.execute("cmd", timeout=60)

    mock_sdk.run.assert_called_once_with("cmd", timeout=60)


def test_execute_with_zero_timeout() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run.return_value = SimpleNamespace(stdout="ok", stderr="", exit_code=0)

    sb.execute("cmd", timeout=0)

    mock_sdk.run.assert_called_once_with("cmd", timeout=0)


def test_write_success() -> None:
    sb, mock_sdk = _make_sandbox()
    # Preflight: file does not exist
    mock_sdk.run.return_value = SimpleNamespace(stdout="", stderr="", exit_code=0)

    result = sb.write("/app/test.txt", "hello world")

    assert result.path == "/app/test.txt"
    assert result.error is None
    mock_sdk.write.assert_called_once_with("/app/test.txt", b"hello world")


def test_write_existing_file_returns_error() -> None:
    sb, mock_sdk = _make_sandbox()
    # Preflight: file already exists
    mock_sdk.run.return_value = SimpleNamespace(stdout="Error: File already exists: '/app/test.txt'", stderr="", exit_code=1)

    result = sb.write("/app/test.txt", "content")

    assert result.error is not None
    assert "already exists" in result.error.lower()
    mock_sdk.write.assert_not_called()


def test_write_error() -> None:
    sb, mock_sdk = _make_sandbox()
    # Preflight succeeds, but SDK write fails
    mock_sdk.run.return_value = SimpleNamespace(stdout="", stderr="", exit_code=0)
    mock_sdk.write.side_effect = SandboxClientError("permission denied")

    result = sb.write("/readonly/test.txt", "content")

    assert result.error is not None
    assert "Failed to write file" in result.error
    assert "/readonly/test.txt" in result.error


def test_download_files_success() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.read.return_value = b"file content"

    responses = sb.download_files(["/app/test.txt"])

    assert len(responses) == 1
    assert responses[0].path == "/app/test.txt"
    assert responses[0].content == b"file content"
    assert responses[0].error is None


def test_download_files_not_found() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.read.side_effect = ResourceNotFoundError("file not found", resource_type="file")

    responses = sb.download_files(["/missing.txt"])

    assert len(responses) == 1
    assert responses[0].path == "/missing.txt"
    assert responses[0].content is None
    assert responses[0].error == "file_not_found"


def test_download_files_partial_success() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.read.side_effect = [
        b"content1",
        ResourceNotFoundError("file not found", resource_type="file"),
        b"content3",
    ]

    responses = sb.download_files(["/a.txt", "/b.txt", "/c.txt"])

    assert len(responses) == 3
    assert responses[0].content == b"content1"
    assert responses[0].error is None
    assert responses[1].content is None
    assert responses[1].error == "file_not_found"
    assert responses[2].content == b"content3"
    assert responses[2].error is None


def test_upload_files_success() -> None:
    sb, mock_sdk = _make_sandbox()

    responses = sb.upload_files([("/app/test.txt", b"content")])

    assert len(responses) == 1
    assert responses[0].path == "/app/test.txt"
    assert responses[0].error is None
    mock_sdk.write.assert_called_once_with("/app/test.txt", b"content")


def test_upload_files_error() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.write.side_effect = SandboxClientError("permission denied")

    responses = sb.upload_files([("/readonly/test.txt", b"content")])

    assert len(responses) == 1
    assert responses[0].path == "/readonly/test.txt"
    assert responses[0].error == "permission_denied"


def test_upload_files_partial_success() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.write.side_effect = [None, SandboxClientError("fail"), None]

    responses = sb.upload_files(
        [
            ("/a.txt", b"a"),
            ("/b.txt", b"b"),
            ("/c.txt", b"c"),
        ]
    )

    assert len(responses) == 3
    assert responses[0].error is None
    assert responses[1].error == "permission_denied"
    assert responses[2].error is None


# -- read() tests -----------------------------------------------------------


def test_write_preflight_empty_output_nonzero_exit() -> None:
    sb, mock_sdk = _make_sandbox()
    # Preflight fails with empty output (e.g., sandbox crash)
    mock_sdk.run.return_value = SimpleNamespace(stdout="", stderr="", exit_code=1)

    result = sb.write("/app/test.txt", "content")

    assert result.error == "Failed to write file '/app/test.txt'"
    mock_sdk.write.assert_not_called()


def test_read_text_file() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.read.return_value = b"line1\nline2\nline3"

    result = sb.read("/app/test.txt")

    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["content"] == "line1\nline2\nline3"
    assert result.file_data["encoding"] == "utf-8"


def test_read_with_pagination() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.read.return_value = b"line0\nline1\nline2\nline3\nline4"

    result = sb.read("/app/test.txt", offset=1, limit=2)

    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["content"] == "line1\nline2"


def test_read_trailing_newline_does_not_add_extra_line() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.read.return_value = b"line0\nline1\nline2\n"

    result = sb.read("/app/test.txt", offset=2, limit=10)

    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["content"] == "line2"


def test_read_sandbox_client_error() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.read.side_effect = SandboxClientError("connection timeout")

    result = sb.read("/app/test.txt")

    assert result.error is not None
    assert "connection timeout" in result.error


def test_read_file_not_found() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.read.side_effect = ResourceNotFoundError("not found", resource_type="file")

    result = sb.read("/missing.txt")

    assert result.error is not None
    assert "file_not_found" in result.error


def test_read_empty_file() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.read.return_value = b""

    result = sb.read("/app/empty.txt")

    assert result.error is None
    assert result.file_data is not None
    assert "empty contents" in result.file_data["content"]


def test_read_binary_file() -> None:
    sb, mock_sdk = _make_sandbox()
    raw = b"\x89PNG\r\n\x1a\n"
    mock_sdk.read.return_value = raw

    result = sb.read("/app/image.png")

    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["encoding"] == "base64"
    assert result.file_data["content"] == base64.b64encode(raw).decode("ascii")


def test_read_large_binary_returns_error() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.read.return_value = b"\x89PNG" + b"\x00" * (500 * 1024)

    result = sb.read("/app/large.png")

    assert result.error is not None
    assert "maximum preview size" in result.error


def test_read_text_extension_with_invalid_utf8_falls_back_to_binary() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.read.return_value = b"\xff\xfe invalid utf8 \x80\x81"

    result = sb.read("/app/corrupted.txt")

    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["encoding"] == "base64"


def test_read_offset_exceeds_length() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.read.return_value = b"only one line"

    result = sb.read("/app/test.txt", offset=5)

    assert result.error is not None
    assert "offset" in result.error.lower()


def test_read_normalizes_crlf_to_lf() -> None:
    r"""CRLF and bare-CR line endings collapse to LF, matching `BaseSandbox.read()`.

    Without this, files written on Windows round-trip with stray `\r`
    characters that break `edit()` (issue #2880).
    """
    sb, mock_sdk = _make_sandbox()
    mock_sdk.read.return_value = b"line1\r\nline2\r\nline3"

    result = sb.read("/app/test.txt")

    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["content"] == "line1\nline2\nline3"


def test_read_normalizes_bare_cr_to_lf() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.read.return_value = b"line1\rline2\rline3"

    result = sb.read("/app/test.txt")

    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["content"] == "line1\nline2\nline3"


def test_read_text_with_only_newline_returns_empty_content() -> None:
    r"""A file containing only `\n` paginates as a single empty line."""
    sb, mock_sdk = _make_sandbox()
    mock_sdk.read.return_value = b"\n"

    result = sb.read("/app/test.txt")

    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["content"] == ""


def test_read_truncates_at_max_output_bytes() -> None:
    """Pages exceeding `MAX_OUTPUT_BYTES` are truncated with `TRUNCATION_MSG`."""
    sb, mock_sdk = _make_sandbox()
    # Single-line payload comfortably above the cap. Using ASCII so byte length
    # equals string length for the assertion below.
    payload = b"x" * (MAX_OUTPUT_BYTES + 10_000)
    mock_sdk.read.return_value = payload

    result = sb.read("/app/big.txt")

    assert result.error is None
    assert result.file_data is not None
    content = result.file_data["content"]
    assert content.endswith(TRUNCATION_MSG)
    assert len(content.encode("utf-8")) <= MAX_OUTPUT_BYTES


def test_read_binary_at_exact_max_size_succeeds() -> None:
    sb, mock_sdk = _make_sandbox()
    raw = b"\x00" * MAX_BINARY_BYTES
    mock_sdk.read.return_value = raw

    result = sb.read("/app/exact.png")

    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["encoding"] == "base64"


def test_read_binary_one_byte_over_max_returns_error() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.read.return_value = b"\x00" * (MAX_BINARY_BYTES + 1)

    result = sb.read("/app/over.png")

    assert result.error is not None
    assert "maximum preview size" in result.error
    # Error message includes the path prefix to match `BaseSandbox.read()` shape.
    assert "/app/over.png" in result.error


def test_read_error_messages_include_file_path() -> None:
    """All `read()` error paths prefix with `File '<path>': ` for parity with base."""
    sb, mock_sdk = _make_sandbox()

    mock_sdk.read.side_effect = ResourceNotFoundError("not found", resource_type="file")
    not_found = sb.read("/missing.txt")
    assert not_found.error is not None
    assert not_found.error.startswith("File '/missing.txt':")

    mock_sdk.read.side_effect = SandboxClientError("boom")
    sdk_err = sb.read("/app/test.txt")
    assert sdk_err.error is not None
    assert sdk_err.error.startswith("File '/app/test.txt':")
    assert "SandboxClientError" in sdk_err.error
    assert "boom" in sdk_err.error


def test_write_preflight_runs_existence_check() -> None:
    """`write()` invokes `_write_preflight` before delegating to the SDK."""
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run.return_value = SimpleNamespace(stdout="", stderr="", exit_code=0)

    sb.write("/app/test.txt", "hello")

    # The preflight executes the base-64-encoded `_WRITE_CHECK_TEMPLATE` script.
    assert mock_sdk.run.call_count == 1
    cmd_arg = mock_sdk.run.call_args.args[0]
    assert "python3 -c" in cmd_arg
    assert base64.b64encode(b"/app/test.txt").decode("ascii") in cmd_arg


def test_max_binary_bytes_constant_matches_template() -> None:
    """Python `MAX_BINARY_BYTES` constant stays in lockstep with the heredoc literal.

    Drift here would silently desync `LangSmithSandbox.read()` from
    `BaseSandbox.read()` because the template does not import the constant.
    """
    assert "MAX_BINARY_BYTES = 500 * 1024" in base_sandbox._READ_COMMAND_TEMPLATE
    assert MAX_BINARY_BYTES == 500 * 1024


def test_max_output_bytes_constant_matches_template() -> None:
    assert "MAX_OUTPUT_BYTES = 500 * 1024" in base_sandbox._READ_COMMAND_TEMPLATE
    assert MAX_OUTPUT_BYTES == 500 * 1024
