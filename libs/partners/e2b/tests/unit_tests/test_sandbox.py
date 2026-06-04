from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import e2b
import pytest

from langchain_e2b import E2BSandbox

if TYPE_CHECKING:
    from e2b import Sandbox

TEST_DIR_PATH = "/home/user/data"
TEST_FILE_PATH = "/home/user/file.txt"
TEST_TIMEOUT = 7
TIMEOUT_EXIT_CODE = 124


@dataclass
class FakeEntryInfo:
    type: e2b.FileType


@dataclass
class FakeCommandResult:
    stdout: str
    stderr: str
    exit_code: int


@dataclass
class FakeFiles:
    entries: dict[str, tuple[e2b.FileType, bytes]] = field(default_factory=dict)
    get_info_exc: BaseException | None = None
    write_exc: BaseException | None = None

    def get_info(self, path: str) -> FakeEntryInfo:
        if self.get_info_exc is not None:
            raise self.get_info_exc
        if path not in self.entries:
            raise e2b.FileNotFoundException(path)

        file_type, _ = self.entries[path]
        return FakeEntryInfo(type=file_type)

    def read(
        self,
        path: str,
        *,
        format: Literal["bytes"] = "bytes",  # noqa: A002
    ) -> bytearray:
        assert format == "bytes"
        return bytearray(self.entries[path][1])

    def write(self, path: str, data: bytes) -> None:
        if self.write_exc is not None:
            raise self.write_exc
        self.entries[path] = (e2b.FileType.FILE, data)


@dataclass
class FakeCommands:
    result: FakeCommandResult | None = None
    exc: BaseException | None = None
    cwd: str | None = None
    timeout: int | None = None

    def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> FakeCommandResult:
        self.cwd = cwd
        self.timeout = timeout
        if self.exc is not None:
            raise self.exc
        assert command
        assert self.result is not None
        return self.result


@dataclass
class FakeSandbox:
    commands: FakeCommands
    files: FakeFiles
    sandbox_id: str = "sbx_test"


def _backend(
    *,
    commands: FakeCommands | None = None,
    files: FakeFiles | None = None,
    workdir: str = "/home/user",
    timeout: int = 30 * 60,
) -> E2BSandbox:
    fake = FakeSandbox(
        commands=commands
        or FakeCommands(result=FakeCommandResult(stdout="", stderr="", exit_code=0)),
        files=files or FakeFiles(),
    )
    return E2BSandbox(
        sandbox=cast("Sandbox", fake),
        workdir=workdir,
        timeout=timeout,
    )


def test_id_returns_e2b_sandbox_id() -> None:
    backend = _backend()

    assert backend.id == "sbx_test"


def test_execute_success_uses_workdir_and_timeout() -> None:
    commands = FakeCommands(
        result=FakeCommandResult(stdout="hello\n", stderr="", exit_code=0)
    )
    backend = _backend(commands=commands, workdir="/workspace", timeout=TEST_TIMEOUT)

    result = backend.execute("echo hello")

    assert result.output == "hello\n"
    assert result.exit_code == 0
    assert result.truncated is False
    assert commands.cwd == "/workspace"
    assert commands.timeout == TEST_TIMEOUT


def test_execute_combines_stdout_and_stderr() -> None:
    backend = _backend(
        commands=FakeCommands(
            result=FakeCommandResult(stdout="out", stderr="err", exit_code=0)
        )
    )

    result = backend.execute("echo hello")

    assert result.output == "out\nerr"
    assert result.exit_code == 0


def test_execute_nonzero_exit_returns_response() -> None:
    backend = _backend(
        commands=FakeCommands(
            exc=e2b.CommandExitException(
                stdout="",
                stderr="boom",
                exit_code=1,
                error="boom",
            )
        )
    )

    result = backend.execute("false")

    assert result.output == "boom"
    assert result.exit_code == 1


def test_execute_timeout_returns_timeout_response() -> None:
    backend = _backend(commands=FakeCommands(exc=e2b.TimeoutException("timed out")))

    result = backend.execute("sleep 10", timeout=5)

    assert result.output == "Command timed out after 5 seconds"
    assert result.exit_code == TIMEOUT_EXIT_CODE


def test_execute_rejects_negative_timeout() -> None:
    backend = _backend()

    with pytest.raises(ValueError, match="timeout must be non-negative"):
        backend.execute("echo hello", timeout=-1)


def test_download_rejects_relative_path() -> None:
    response = _backend().download_files(["relative.txt"])[0]

    assert response.error == "invalid_path"
    assert response.content is None


def test_download_missing_file_maps_to_file_not_found() -> None:
    response = _backend().download_files([TEST_FILE_PATH])[0]

    assert response.error == "file_not_found"
    assert response.content is None


def test_download_directory_maps_to_is_directory() -> None:
    files = FakeFiles(entries={TEST_DIR_PATH: (e2b.FileType.DIR, b"")})
    response = _backend(files=files).download_files([TEST_DIR_PATH])[0]

    assert response.error == "is_directory"
    assert response.content is None


def test_download_invalid_argument_maps_to_invalid_path() -> None:
    files = FakeFiles(get_info_exc=e2b.InvalidArgumentException("invalid path"))
    response = _backend(files=files).download_files([TEST_FILE_PATH])[0]

    assert response.error == "invalid_path"
    assert response.content is None


def test_upload_rejects_relative_path() -> None:
    response = _backend().upload_files([("relative.txt", b"hello")])[0]

    assert response.error == "invalid_path"


def test_upload_existing_directory_maps_to_is_directory() -> None:
    files = FakeFiles(entries={TEST_DIR_PATH: (e2b.FileType.DIR, b"")})
    response = _backend(files=files).upload_files([(TEST_DIR_PATH, b"hello")])[0]

    assert response.error == "is_directory"


def test_upload_invalid_argument_maps_to_invalid_path() -> None:
    files = FakeFiles(write_exc=e2b.InvalidArgumentException("invalid path"))
    response = _backend(files=files).upload_files([(TEST_FILE_PATH, b"hello")])[0]

    assert response.error == "invalid_path"


def test_upload_and_download_round_trip() -> None:
    files = FakeFiles()
    backend = _backend(files=files)

    upload = backend.upload_files([(TEST_FILE_PATH, b"hello")])[0]
    download = backend.download_files([TEST_FILE_PATH])[0]

    assert upload.error is None
    assert download.error is None
    assert download.content == b"hello"
