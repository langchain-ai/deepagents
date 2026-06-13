from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest

from langchain_vercel_sandbox import VercelSandbox
from langchain_vercel_sandbox.sandbox import MAX_OUTPUT_BYTES

if TYPE_CHECKING:
    from vercel.sandbox import Sandbox

NON_ZERO_EXIT_CODE = 7
TIMEOUT_EXIT_CODE = 124
DEFAULT_TIMEOUT = 42
EXPLICIT_TIMEOUT = 7


class _Command:
    def __init__(
        self,
        *,
        exit_code: int | None = 0,
        stdout: str = "",
        stderr: str = "",
        wait_exc: Exception | None = None,
        stdout_exc: Exception | None = None,
    ) -> None:
        self.exit_code = exit_code
        self._stdout = stdout
        self._stderr = stderr
        self._wait_exc = wait_exc
        self._stdout_exc = stdout_exc
        self.wait_event: threading.Event | None = None
        self.kill = MagicMock()

    def wait(self) -> _Command:
        if self.wait_event is not None:
            self.wait_event.wait()
        if self._wait_exc is not None:
            raise self._wait_exc
        return self

    def stdout(self) -> str:
        if self._stdout_exc is not None:
            raise self._stdout_exc
        return self._stdout

    def stderr(self) -> str:
        return self._stderr


class _Sandbox:
    def __init__(self) -> None:
        self.sandbox_id = "sb_123"
        self.detached_command = _Command(stdout="hello\n")
        self.writes: list[list[dict[str, object]]] = []
        self.files: dict[str, bytes | Exception | None] = {}
        self.write_error: Exception | None = None

    def run_command_detached(self, cmd: str, args: list[str]) -> _Command:
        self.detached_args = (cmd, args)
        return self.detached_command

    def read_file(self, path: str) -> bytes | None:
        value = self.files[path]
        if isinstance(value, Exception):
            raise value
        return value

    def write_files(self, files: list[dict[str, object]]) -> None:
        self.writes.append(files)
        if self.write_error is not None:
            raise self.write_error

    def as_backend(self) -> VercelSandbox:
        return VercelSandbox(sandbox=cast("Sandbox", self))


def _wait_immediately(cmd: _Command, _timeout: int, _started_at: float) -> _Command:
    return cmd.wait()


def test_id_returns_sandbox_id() -> None:
    sandbox = _Sandbox()

    assert sandbox.as_backend().id == "sb_123"


def test_execute_returns_stdout() -> None:
    sandbox = _Sandbox()
    sandbox.detached_command = _Command(exit_code=0, stdout="hello\n")

    result = sandbox.as_backend().execute("echo hello")

    assert result.output == "hello\n"
    assert result.exit_code == 0
    assert sandbox.detached_args == ("bash", ["-lc", "echo hello"])


def test_execute_appends_stderr() -> None:
    sandbox = _Sandbox()
    sandbox.detached_command = _Command(exit_code=0, stdout="out", stderr="err\n")

    result = sandbox.as_backend().execute("echo hello")

    assert result.output == "out\n<stderr>err</stderr>"
    assert result.exit_code == 0


def test_execute_preserves_non_zero_exit_code() -> None:
    sandbox = _Sandbox()
    sandbox.detached_command = _Command(exit_code=NON_ZERO_EXIT_CODE, stdout="failed")

    result = sandbox.as_backend().execute("exit 7")

    assert result.output == "failed"
    assert result.exit_code == NON_ZERO_EXIT_CODE


def test_execute_truncates_large_stdout() -> None:
    sandbox = _Sandbox()
    sandbox.detached_command = _Command(stdout="x" * (MAX_OUTPUT_BYTES + 1))

    result = sandbox.as_backend().execute("yes | head -c 100001")

    assert result.output == (
        "x" * MAX_OUTPUT_BYTES
        + f"\n\n... Output truncated at {MAX_OUTPUT_BYTES} bytes."
    )
    assert result.truncated is True


def test_execute_truncates_combined_stdout_and_stderr() -> None:
    sandbox = _Sandbox()
    sandbox.detached_command = _Command(
        stdout="x" * (MAX_OUTPUT_BYTES - 1),
        stderr="err",
    )

    result = sandbox.as_backend().execute("python noisy.py")

    assert result.output == (
        "x" * (MAX_OUTPUT_BYTES - 1)
        + "\n"
        + f"\n\n... Output truncated at {MAX_OUTPUT_BYTES} bytes."
    )
    assert "<stderr>" not in result.output
    assert result.truncated is True


def test_execute_enforces_timeout_and_kills_command() -> None:
    sandbox = _Sandbox()
    pending = _Command(exit_code=None)
    pending.wait_event = threading.Event()
    sandbox.detached_command = pending
    backend = VercelSandbox(sandbox=cast("Sandbox", sandbox))

    result = backend.execute("sleep 10", timeout=-1)

    assert result.output == "Command timed out after -1 seconds"
    assert result.exit_code == TIMEOUT_EXIT_CODE
    pending.kill.assert_called_once_with()


def test_execute_waits_until_complete() -> None:
    sandbox = _Sandbox()
    sandbox.detached_command = _Command(exit_code=0, stdout="done")

    with patch.object(
        sandbox.detached_command,
        "wait",
        wraps=sandbox.detached_command.wait,
    ) as wait:
        result = sandbox.as_backend().execute("echo done")

    assert result.output == "done"
    assert result.exit_code == 0
    wait.assert_called_once_with()


def test_execute_uses_default_timeout() -> None:
    sandbox = _Sandbox()
    backend = VercelSandbox(
        sandbox=cast("Sandbox", sandbox),
        timeout=DEFAULT_TIMEOUT,
    )

    with patch(
        "langchain_vercel_sandbox.sandbox._wait_for_command",
        wraps=_wait_immediately,
    ) as wait:
        backend.execute("echo done")

    assert wait.call_args.args[1] == DEFAULT_TIMEOUT


def test_execute_uses_explicit_timeout() -> None:
    sandbox = _Sandbox()
    backend = VercelSandbox(
        sandbox=cast("Sandbox", sandbox),
        timeout=DEFAULT_TIMEOUT,
    )

    with patch(
        "langchain_vercel_sandbox.sandbox._wait_for_command",
        wraps=_wait_immediately,
    ) as wait:
        backend.execute("echo done", timeout=EXPLICIT_TIMEOUT)

    assert wait.call_args.args[1] == EXPLICIT_TIMEOUT


def test_upload_files_rejects_relative_paths_and_preserves_order() -> None:
    sandbox = _Sandbox()

    responses = sandbox.as_backend().upload_files(
        [
            ("relative.txt", b"bad"),
            ("/vercel/sandbox/ok.txt", b"ok"),
            ("other.txt", b"bad"),
        ]
    )

    assert [response.path for response in responses] == [
        "relative.txt",
        "/vercel/sandbox/ok.txt",
        "other.txt",
    ]
    assert [response.error for response in responses] == [
        "invalid_path",
        None,
        "invalid_path",
    ]
    assert sandbox.writes == [[{"path": "/vercel/sandbox/ok.txt", "content": b"ok"}]]


def test_upload_files_maps_provider_errors_to_valid_paths() -> None:
    sandbox = _Sandbox()
    sandbox.write_error = PermissionError("permission denied")

    responses = sandbox.as_backend().upload_files(
        [("relative.txt", b"bad"), ("/vercel/sandbox/ok.txt", b"ok")]
    )

    assert [response.error for response in responses] == [
        "invalid_path",
        "permission_denied",
    ]


def test_download_files_rejects_relative_paths_and_preserves_order() -> None:
    sandbox = _Sandbox()
    sandbox.files["/vercel/sandbox/ok.txt"] = b"ok"
    sandbox.files["/vercel/sandbox/missing.txt"] = FileNotFoundError("missing")

    responses = sandbox.as_backend().download_files(
        ["relative.txt", "/vercel/sandbox/ok.txt", "/vercel/sandbox/missing.txt"]
    )

    assert [response.path for response in responses] == [
        "relative.txt",
        "/vercel/sandbox/ok.txt",
        "/vercel/sandbox/missing.txt",
    ]
    assert responses[0].error == "invalid_path"
    assert responses[1].content == b"ok"
    assert responses[1].error is None
    assert responses[2].content is None
    assert responses[2].error == "file_not_found"


def test_download_files_surfaces_missing_sandbox() -> None:
    sandbox = _Sandbox()
    # The SDK returns None only when the sandbox itself is gone, not for a
    # missing file; it must not be collapsed to file_not_found.
    sandbox.files["/vercel/sandbox/file.txt"] = None

    response = sandbox.as_backend().download_files(["/vercel/sandbox/file.txt"])[0]

    assert response.content is None
    assert response.error == "sandbox not found"


def test_download_files_maps_missing_file_errors() -> None:
    sandbox = _Sandbox()
    sandbox.files["/vercel/sandbox/missing.txt"] = FileNotFoundError("missing")

    response = sandbox.as_backend().download_files(["/vercel/sandbox/missing.txt"])[0]

    assert response.content is None
    assert response.error == "file_not_found"


def test_download_files_maps_directory_errors() -> None:
    sandbox = _Sandbox()
    sandbox.files["/vercel/sandbox/dir"] = IsADirectoryError("is a directory")

    response = sandbox.as_backend().download_files(["/vercel/sandbox/dir"])[0]

    assert response.content is None
    assert response.error == "is_directory"


def test_download_files_maps_generic_directory_errors() -> None:
    sandbox = _Sandbox()
    sandbox.files["/vercel/sandbox/dir"] = RuntimeError("Is a directory")

    response = sandbox.as_backend().download_files(["/vercel/sandbox/dir"])[0]

    assert response.content is None
    assert response.error == "is_directory"


def test_download_files_does_not_treat_missing_path_as_directory() -> None:
    sandbox = _Sandbox()
    paths = [
        "/vercel/sandbox/no-such-file.txt",
        "/vercel/sandbox/not-a-directory/file.txt",
    ]
    sandbox.files[paths[0]] = RuntimeError("No such file or directory")
    sandbox.files[paths[1]] = RuntimeError("not a directory")

    responses = sandbox.as_backend().download_files(paths)

    # "No such file or directory" is the POSIX ENOENT phrasing and maps to
    # file_not_found; "not a directory" (ENOTDIR) is neither a missing file nor
    # a directory, so its raw message is surfaced rather than mislabeled.
    assert [response.content for response in responses] == [None, None]
    assert [response.error for response in responses] == [
        "file_not_found",
        "not a directory",
    ]


def test_download_files_maps_permission_substring() -> None:
    sandbox = _Sandbox()
    sandbox.files["/vercel/sandbox/secret"] = RuntimeError("Access denied")

    response = sandbox.as_backend().download_files(["/vercel/sandbox/secret"])[0]

    assert response.content is None
    assert response.error == "permission_denied"


def test_download_files_surfaces_unrecognized_error() -> None:
    sandbox = _Sandbox()
    sandbox.files["/vercel/sandbox/file.txt"] = RuntimeError("connection reset by peer")

    response = sandbox.as_backend().download_files(["/vercel/sandbox/file.txt"])[0]

    assert response.content is None
    # Unrecognized failures (network/auth/transient) are surfaced verbatim
    # instead of being masked as file_not_found.
    assert response.error == "connection reset by peer"


def test_upload_files_surfaces_unrecognized_error() -> None:
    sandbox = _Sandbox()
    sandbox.write_error = RuntimeError("network unreachable")

    responses = sandbox.as_backend().upload_files(
        [("relative.txt", b"bad"), ("/vercel/sandbox/ok.txt", b"ok")]
    )

    assert [response.error for response in responses] == [
        "invalid_path",
        "network unreachable",
    ]


def test_execute_timeout_zero_waits_indefinitely() -> None:
    sandbox = _Sandbox()
    sandbox.detached_command = _Command(exit_code=0, stdout="done")

    result = sandbox.as_backend().execute("echo done", timeout=0)

    assert result.output == "done"
    assert result.exit_code == 0


def test_execute_reraises_command_wait_error() -> None:
    sandbox = _Sandbox()
    sandbox.detached_command = _Command(wait_exc=RuntimeError("wait blew up"))

    with pytest.raises(RuntimeError, match="wait blew up"):
        sandbox.as_backend().execute("echo done", timeout=EXPLICIT_TIMEOUT)


def test_execute_timeout_logs_kill_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sandbox = _Sandbox()
    pending = _Command(exit_code=None)
    pending.wait_event = threading.Event()
    pending.kill.side_effect = RuntimeError("kill failed")
    sandbox.detached_command = pending

    with caplog.at_level(logging.WARNING):
        result = VercelSandbox(sandbox=cast("Sandbox", sandbox)).execute(
            "sleep 10", timeout=-1
        )

    assert result.exit_code == TIMEOUT_EXIT_CODE
    pending.kill.assert_called_once_with()
    assert "Failed to kill timed-out command" in caplog.text


def test_execute_returns_response_when_log_fetch_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sandbox = _Sandbox()
    sandbox.detached_command = _Command(
        exit_code=NON_ZERO_EXIT_CODE,
        stdout_exc=RuntimeError("network down"),
    )

    with caplog.at_level(logging.WARNING):
        result = sandbox.as_backend().execute("echo hi")

    # A failed (network) log fetch must not escape execute(): the command
    # already ran, so the exit code is preserved and output is reported missing.
    assert result.exit_code == NON_ZERO_EXIT_CODE
    assert result.output == "<output unavailable: failed to fetch command logs>"
    assert result.truncated is False
    assert "Failed to fetch output" in caplog.text


def test_execute_completes_through_threaded_queue() -> None:
    sandbox = _Sandbox()
    sandbox.detached_command = _Command(exit_code=0, stdout="done")

    # A positive timeout exercises the real threaded wait + queue drain rather
    # than the timeout=0 direct-wait shortcut.
    result = sandbox.as_backend().execute("echo done", timeout=EXPLICIT_TIMEOUT)

    assert result.output == "done"
    assert result.exit_code == 0


def test_init_rejects_negative_timeout() -> None:
    with pytest.raises(ValueError, match="timeout must be non-negative"):
        VercelSandbox(sandbox=cast("Sandbox", _Sandbox()), timeout=-1)
