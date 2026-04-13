"""Tests for LangSmithSandbox backend."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from langsmith.sandbox import ResourceNotFoundError, SandboxClientError

from deepagents.backends.langsmith import LangSmithSandbox


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
    import base64  # noqa: PLC0415 — local to avoid top-level import for one test

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
