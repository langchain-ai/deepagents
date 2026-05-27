"""Unit tests for `CubeSandbox` covering execute/upload/download paths."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_cubesandbox.sandbox import (
    _DOWNLOAD_FILE_NOT_FOUND,
    _DOWNLOAD_IS_DIRECTORY,
    _DOWNLOAD_PERMISSION_DENIED,
    _EXEC_RC_PREFIX,
    _EXEC_RC_SUFFIX,
    CubeSandbox,
)

COMMAND_TIMEOUT_EXIT_CODE = 124


def _make_sandbox() -> tuple[CubeSandbox, MagicMock]:
    mock_sdk = MagicMock()
    mock_sdk.sandbox_id = "sb-cube-123"
    backend = CubeSandbox(sandbox=mock_sdk)
    return backend, mock_sdk


def _make_execution(
    *,
    stdout: str = "",
    stderr: list[str] | None = None,
    error: Any | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        logs=SimpleNamespace(stdout=[stdout] if stdout else [], stderr=stderr or []),
        error=error,
    )


def _exec_stdout(captured: str, rc: int) -> str:
    """Build the stdout payload our in-sandbox wrapper would emit."""
    return f"{captured}{_EXEC_RC_PREFIX}{rc}{_EXEC_RC_SUFFIX}"


# --- id / construction -----------------------------------------------------


def test_id_returns_underlying_sandbox_id() -> None:
    sb, _ = _make_sandbox()
    assert sb.id == "sb-cube-123"


# --- execute ---------------------------------------------------------------


def test_execute_returns_stdout_and_exit_code() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run_code.return_value = _make_execution(
        stdout=_exec_stdout("hello world\n", 0),
    )

    result = sb.execute("echo hello world")

    assert result.exit_code == 0
    assert result.output == "hello world\n"
    assert result.truncated is False
    mock_sdk.run_code.assert_called_once()
    args, kwargs = mock_sdk.run_code.call_args
    # Wrapper code must contain the original command (as a Python literal).
    assert "'echo hello world'" in args[0]
    assert kwargs.get("timeout") == 30 * 60
    # `commands.run` must NOT be used — it has the trailing-newline corruption bug.
    mock_sdk.commands.run.assert_not_called()


def test_execute_preserves_stdout_without_trailing_newline() -> None:
    """Regression: `cat` on a no-trailing-newline file used to leak `0` from
    the SDK's `commands.run()` wrapper. Our `execute()` must keep stdout
    byte-for-byte regardless of trailing newline.
    """
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run_code.return_value = _make_execution(
        stdout=_exec_stdout("First content", 0),
    )

    result = sb.execute("cat /tmp/no-newline")

    assert result.exit_code == 0
    assert result.output == "First content"


def test_execute_appends_stderr_in_envelope() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run_code.return_value = _make_execution(
        stdout=_exec_stdout("line1\n", 2),
        stderr=["oops"],
    )

    result = sb.execute("bad-cmd")

    assert result.exit_code == 2
    assert result.output == "line1\n\n<stderr>oops</stderr>"


def test_execute_handles_negative_exit_code() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run_code.return_value = _make_execution(
        stdout=_exec_stdout("", -1),
    )

    result = sb.execute("kill-by-signal")

    assert result.exit_code == -1
    assert result.output == ""


def test_execute_uses_custom_timeout() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run_code.return_value = _make_execution(stdout=_exec_stdout("", 0))

    sb.execute("ls", timeout=5)

    _, kwargs = mock_sdk.run_code.call_args
    assert kwargs.get("timeout") == 5


def test_execute_handles_timeout_error() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run_code.side_effect = TimeoutError("boom")

    result = sb.execute("sleep 999", timeout=10)

    assert result.exit_code == COMMAND_TIMEOUT_EXIT_CODE
    assert "timed out" in result.output


def test_execute_handles_httpx_read_timeout() -> None:
    sb, mock_sdk = _make_sandbox()

    class ReadTimeout(Exception):
        """Stand-in for `httpx.ReadTimeout` to avoid pulling httpx into tests."""

    mock_sdk.run_code.side_effect = ReadTimeout("read timed out")

    result = sb.execute("sleep 999", timeout=10)

    assert result.exit_code == COMMAND_TIMEOUT_EXIT_CODE
    assert "timed out" in result.output


def test_execute_falls_back_when_sentinel_missing() -> None:
    """If the wrapper dies before emitting the sentinel, we must not crash.

    Exit code falls back to `0` when no `Execution.error` is reported,
    `1` otherwise. The raw stdout is surfaced as-is.
    """
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run_code.return_value = _make_execution(
        stdout="partial output",
        error=SimpleNamespace(name="MemoryError"),
    )

    result = sb.execute("dump-the-universe")

    assert result.exit_code == 1
    assert result.output == "partial output"


def test_execute_reraises_unknown_exception() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run_code.side_effect = RuntimeError("unexpected")

    with pytest.raises(RuntimeError, match="unexpected"):
        sb.execute("ls")


# --- upload_files ----------------------------------------------------------


def test_upload_files_rejects_relative_path() -> None:
    sb, mock_sdk = _make_sandbox()

    responses = sb.upload_files([("relative.txt", b"data")])

    assert responses == [
        type(responses[0])(path="relative.txt", error="invalid_path"),
    ]
    mock_sdk.run_code.assert_not_called()


def test_upload_files_calls_run_code_with_base64_payload() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run_code.return_value = _make_execution()

    payload = b"\x00\x01binary\xff"
    responses = sb.upload_files([("/tmp/x.bin", payload)])

    assert responses[0].path == "/tmp/x.bin"
    assert responses[0].error is None
    code = mock_sdk.run_code.call_args.args[0]
    expected_b64 = base64.b64encode(payload).decode("ascii")
    assert expected_b64 in code
    assert "/tmp/x.bin" in code
    assert "open(_path, 'wb')" in code


def test_upload_files_maps_python_error_to_file_operation_error() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run_code.return_value = _make_execution(
        error=SimpleNamespace(
            name="PermissionError",
            value="[Errno 13] Permission denied: '/root/x'",
            traceback=[],
        ),
    )

    responses = sb.upload_files([("/root/x", b"x")])

    assert responses[0].error == "permission_denied"


def test_upload_files_preserves_order_with_mixed_results() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run_code.return_value = _make_execution()

    responses = sb.upload_files(
        [
            ("bad", b"x"),
            ("/ok/a.txt", b"a"),
            ("also-bad", b"y"),
            ("/ok/b.txt", b"b"),
        ]
    )

    assert [r.path for r in responses] == [
        "bad",
        "/ok/a.txt",
        "also-bad",
        "/ok/b.txt",
    ]
    assert [r.error for r in responses] == [
        "invalid_path",
        None,
        "invalid_path",
        None,
    ]
    assert mock_sdk.run_code.call_count == 2


def test_upload_files_surfaces_sdk_exception_per_file() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run_code.side_effect = RuntimeError("network unreachable")

    responses = sb.upload_files([("/tmp/x", b"data")])

    assert responses[0].path == "/tmp/x"
    assert responses[0].error is not None
    assert "RuntimeError" in responses[0].error
    assert "network unreachable" in responses[0].error


# --- download_files --------------------------------------------------------


def test_download_files_rejects_relative_path() -> None:
    sb, mock_sdk = _make_sandbox()

    responses = sb.download_files(["relative.txt"])

    assert responses[0].path == "relative.txt"
    assert responses[0].error == "invalid_path"
    assert responses[0].content is None
    mock_sdk.run_code.assert_not_called()


def test_download_files_decodes_base64_payload() -> None:
    sb, mock_sdk = _make_sandbox()
    payload = b"\x00binary\xffdata"
    encoded = base64.b64encode(payload).decode("ascii")
    mock_sdk.run_code.return_value = _make_execution(stdout=encoded)

    responses = sb.download_files(["/tmp/x.bin"])

    assert responses[0].path == "/tmp/x.bin"
    assert responses[0].error is None
    assert responses[0].content == payload


def test_download_files_maps_sentinels_to_error_codes() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run_code.side_effect = [
        _make_execution(stdout=_DOWNLOAD_FILE_NOT_FOUND),
        _make_execution(stdout=_DOWNLOAD_IS_DIRECTORY),
        _make_execution(stdout=_DOWNLOAD_PERMISSION_DENIED),
    ]

    responses = sb.download_files(["/a", "/b", "/c"])

    assert [r.error for r in responses] == [
        "file_not_found",
        "is_directory",
        "permission_denied",
    ]
    assert all(r.content is None for r in responses)


def test_download_files_handles_invalid_base64_payload() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.run_code.return_value = _make_execution(stdout="not-base64!!!")

    responses = sb.download_files(["/tmp/x"])

    assert responses[0].error == "file_not_found"
    assert responses[0].content is None


def test_download_files_preserves_order_with_mixed_results() -> None:
    sb, mock_sdk = _make_sandbox()
    payload_a = b"alpha"
    encoded_a = base64.b64encode(payload_a).decode("ascii")
    mock_sdk.run_code.side_effect = [
        _make_execution(stdout=encoded_a),
        _make_execution(stdout=_DOWNLOAD_FILE_NOT_FOUND),
    ]

    responses = sb.download_files(["bad", "/ok/a", "/missing"])

    assert [r.path for r in responses] == ["bad", "/ok/a", "/missing"]
    assert responses[0].error == "invalid_path"
    assert responses[1].content == payload_a
    assert responses[1].error is None
    assert responses[2].error == "file_not_found"
