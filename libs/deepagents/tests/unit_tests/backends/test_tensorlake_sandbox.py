"""Tests for TensorlakeSandbox backend."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from tensorlake.sandbox.exceptions import SandboxError

from deepagents.backends.tensorlake import TensorlakeSandbox


def _make_sandbox() -> tuple[TensorlakeSandbox, MagicMock]:
    mock_sdk = MagicMock()
    mock_sdk.sandbox_id = "sb-123"
    sb = TensorlakeSandbox(sandbox=mock_sdk)
    return sb, mock_sdk


def test_id_returns_sandbox_id() -> None:
    sb, _ = _make_sandbox()
    assert sb.id == "sb-123"


def test_execute_returns_stdout() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run.return_value = SimpleNamespace(stdout="hello", stderr="", exit_code=0)

    result = sb.execute("echo hello")

    assert result.output == "hello"
    assert result.exit_code == 0


def test_execute_combines_stdout_and_stderr() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run.return_value = SimpleNamespace(stdout="out", stderr="err", exit_code=1)

    result = sb.execute("command")

    assert result.output == "out\nerr"


def test_download_files_not_found() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.read_file.side_effect = SandboxError("file not found")

    response = sb.download_files(["/missing.txt"])[0]

    assert response.error == "file_not_found"


def test_upload_files_permission_denied() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.write_file.side_effect = SandboxError("permission denied")

    response = sb.upload_files([("/readonly.txt", b"x")])[0]

    assert response.error == "permission_denied"
