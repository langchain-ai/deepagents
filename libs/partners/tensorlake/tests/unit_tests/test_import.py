from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import langchain_tensorlake
from langchain_tensorlake.sandbox import TensorlakeSandbox


def _make_sandbox() -> tuple[TensorlakeSandbox, MagicMock]:
    mock_sdk = MagicMock()
    mock_sdk.sandbox_id = "sb-123"
    sb = TensorlakeSandbox(sandbox=mock_sdk)
    return sb, mock_sdk


def test_import_tensorlake() -> None:
    assert langchain_tensorlake is not None


def test_execute_returns_stdout() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run.return_value = SimpleNamespace(stdout="hello world", stderr="", exit_code=0)

    result = sb.execute("echo hello world")

    assert result.output == "hello world"
    assert result.exit_code == 0
    assert result.truncated is False


def test_execute_combines_stdout_and_stderr() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run.return_value = SimpleNamespace(stdout="out", stderr="err", exit_code=1)

    result = sb.execute("bad-cmd")

    assert result.output == "out\nerr"
    assert result.exit_code == 1


def test_download_files_success() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.read_file.return_value = b"content"

    responses = sb.download_files(["/app/file.txt"])

    assert len(responses) == 1
    assert responses[0].content == b"content"
    assert responses[0].error is None


def test_download_files_not_found() -> None:
    from tensorlake.sandbox.exceptions import SandboxError

    sb, mock_sdk = _make_sandbox()
    mock_sdk.read_file.side_effect = SandboxError("file not found")

    responses = sb.download_files(["/missing.txt"])

    assert responses[0].error == "file_not_found"


def test_upload_files_success() -> None:
    sb, mock_sdk = _make_sandbox()

    responses = sb.upload_files([("/app/file.txt", b"content")])

    assert len(responses) == 1
    assert responses[0].error is None


def test_upload_files_permission_denied() -> None:
    from tensorlake.sandbox.exceptions import SandboxError

    sb, mock_sdk = _make_sandbox()
    mock_sdk.write_file.side_effect = SandboxError("permission denied")

    responses = sb.upload_files([("/readonly/file.txt", b"content")])

    assert responses[0].error == "permission_denied"
