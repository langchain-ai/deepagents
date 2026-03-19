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
