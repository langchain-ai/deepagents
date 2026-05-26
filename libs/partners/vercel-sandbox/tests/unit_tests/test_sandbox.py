from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

from langchain_vercel_sandbox import VercelSandbox

if TYPE_CHECKING:
    from vercel.sandbox import Sandbox

NON_ZERO_EXIT_CODE = 7
TIMEOUT_EXIT_CODE = 124


@dataclass
class _CommandModel:
    exit_code: int | None


class _Command:
    def __init__(
        self,
        *,
        cmd_id: str = "cmd_123",
        exit_code: int | None = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.cmd_id = cmd_id
        self.cmd = _CommandModel(exit_code=exit_code)
        self._stdout = stdout
        self._stderr = stderr
        self.wait_event: threading.Event | None = None
        self.kill = MagicMock()

    def wait(self) -> _Command:
        if self.wait_event is not None:
            self.wait_event.wait()
        return self

    def stdout(self) -> str:
        return self._stdout

    def stderr(self) -> str:
        return self._stderr


class _Sandbox:
    def __init__(self) -> None:
        self.sandbox_id = "sb_123"
        self.detached_command = _Command(stdout="hello\n")
        self.commands: list[_Command] = []
        self.writes: list[list[dict[str, object]]] = []
        self.files: dict[str, bytes | Exception | None] = {}
        self.write_error: Exception | None = None

    def run_command_detached(self, cmd: str, args: list[str]) -> _Command:
        self.detached_args = (cmd, args)
        return self.detached_command

    def get_command(self, cmd_id: str) -> _Command:
        assert cmd_id == self.detached_command.cmd_id
        return self.commands.pop(0)

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
    sandbox.files["/vercel/sandbox/missing.txt"] = None

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
