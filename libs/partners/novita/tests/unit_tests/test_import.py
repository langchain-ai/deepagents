from __future__ import annotations

from unittest.mock import MagicMock

import langchain_novita
from langchain_novita.sandbox import NovitaSandbox


def _make_sandbox() -> tuple[NovitaSandbox, MagicMock]:
    mock_sdk = MagicMock()
    mock_sdk.sandbox_id = "novita-sb-123"
    sb = NovitaSandbox(sandbox=mock_sdk)
    return sb, mock_sdk


def test_import_novita() -> None:
    assert langchain_novita is not None


def test_sandbox_id() -> None:
    sb, mock_sdk = _make_sandbox()
    assert sb.id == "novita-sb-123"


def test_execute_returns_stdout() -> None:
    from deepagents.backends.protocol import ExecuteResponse

    sb, mock_sdk = _make_sandbox()
    mock_result = MagicMock()
    mock_result.stdout = "hello world"
    mock_result.stderr = ""
    mock_result.exit_code = 0
    mock_sdk.commands.run.return_value = mock_result

    result = sb.execute("echo hello world")

    assert isinstance(result, ExecuteResponse)
    assert result.output == "hello world"
    assert result.exit_code == 0
    assert result.truncated is False


def test_execute_nonzero_exit_code() -> None:
    from novita_sandbox.code_interpreter import CommandExitException

    sb, mock_sdk = _make_sandbox()
    exc = CommandExitException(
        stdout="",
        stderr="command not found",
        exit_code=127,
        error=None,
    )
    mock_sdk.commands.run.side_effect = exc

    result = sb.execute("bad_command")

    assert result.exit_code == 127
    assert "command not found" in result.output


def test_execute_uses_default_timeout() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_result = MagicMock()
    mock_result.stdout = "ok"
    mock_result.stderr = ""
    mock_result.exit_code = 0
    mock_sdk.commands.run.return_value = mock_result

    sb.execute("echo ok")

    call_kwargs = mock_sdk.commands.run.call_args
    assert call_kwargs.kwargs.get("timeout") == int(30 * 60)


def test_execute_timeout_zero_passes_none() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_result = MagicMock()
    mock_result.stdout = "ok"
    mock_result.stderr = ""
    mock_result.exit_code = 0
    mock_sdk.commands.run.return_value = mock_result

    sb.execute("echo ok", timeout=0)

    call_kwargs = mock_sdk.commands.run.call_args
    assert call_kwargs.kwargs.get("timeout") is None


def test_download_files_invalid_path() -> None:
    sb, _ = _make_sandbox()
    results = sb.download_files(["relative/path"])
    assert results[0].error == "invalid_path"
    assert results[0].content is None


def test_download_files_success() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.files.read.return_value = bytearray(b"file content")

    results = sb.download_files(["/home/user/file.txt"])

    assert results[0].content == b"file content"
    assert results[0].error is None


def test_download_files_is_directory() -> None:
    from novita_sandbox.code_interpreter import InvalidArgumentException

    sb, mock_sdk = _make_sandbox()
    mock_sdk.files.read.side_effect = InvalidArgumentException(
        "path '/tmp/somedir' is a directory"
    )

    results = sb.download_files(["/tmp/somedir"])
    assert results[0].error == "is_directory"
    assert results[0].content is None


def test_upload_files_invalid_path() -> None:
    sb, _ = _make_sandbox()
    results = sb.upload_files([("relative/path", b"data")])
    assert results[0].error == "invalid_path"


def test_upload_files_success() -> None:
    sb, mock_sdk = _make_sandbox()
    results = sb.upload_files([("/home/user/file.txt", b"data")])
    mock_sdk.files.write.assert_called_once_with("/home/user/file.txt", b"data")
    assert results[0].error is None
