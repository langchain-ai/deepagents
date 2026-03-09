from __future__ import annotations

from types import SimpleNamespace

from e2b.exceptions import NotFoundException, TimeoutException
from e2b.sandbox.commands.command_handle import CommandExitException
from e2b.sandbox.filesystem.filesystem import FileType

from langchain_e2b import E2BSandbox

TIMEOUT_EXIT_CODE = 124
TEST_DIR_PATH = "/home/user/data"
TEST_FILE_PATH = "/home/user/file.txt"


class FakeFiles:
    def __init__(self) -> None:
        self.entries: dict[str, tuple[FileType, bytes]] = {}

    def get_info(self, path: str) -> SimpleNamespace:
        if path not in self.entries:
            raise NotFoundException(path)

        file_type, _ = self.entries[path]
        return SimpleNamespace(type=file_type)

    def read(self, path: str, format: str = "bytes") -> bytes:  # noqa: A002
        _ = format
        return self.entries[path][1]

    def write(self, path: str, content: bytes) -> None:
        self.entries[path] = (FileType.FILE, content)


class FakeCommands:
    def __init__(
        self, *, result: SimpleNamespace | None = None, exc: Exception | None = None
    ) -> None:
        self.result = result
        self.exc = exc

    def run(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
        if self.exc is not None:
            raise self.exc
        assert self.result is not None
        return self.result


class FakeSandbox:
    def __init__(self, *, commands: FakeCommands, files: FakeFiles) -> None:
        self.sandbox_id = "sbx_test"
        self.commands = commands
        self.files = files


def test_execute_success() -> None:
    backend = E2BSandbox(
        sandbox=FakeSandbox(
            commands=FakeCommands(
                result=SimpleNamespace(stdout="hello\n", stderr="", exit_code=0)
            ),
            files=FakeFiles(),
        )
    )

    result = backend.execute("echo hello")

    assert result.output == "hello\n"
    assert result.exit_code == 0


def test_execute_nonzero_exit_returns_response() -> None:
    backend = E2BSandbox(
        sandbox=FakeSandbox(
            commands=FakeCommands(
                exc=CommandExitException(
                    stdout="",
                    stderr="boom",
                    exit_code=1,
                    error="boom",
                )
            ),
            files=FakeFiles(),
        )
    )

    result = backend.execute("false")

    assert result.output == "boom"
    assert result.exit_code == 1


def test_execute_timeout_returns_timeout_response() -> None:
    backend = E2BSandbox(
        sandbox=FakeSandbox(
            commands=FakeCommands(exc=TimeoutException("timed out")),
            files=FakeFiles(),
        )
    )

    result = backend.execute("sleep 10", timeout=5)

    assert "timed out after 5 seconds" in result.output
    assert result.exit_code == TIMEOUT_EXIT_CODE


def test_download_directory_maps_to_is_directory() -> None:
    files = FakeFiles()
    files.entries[TEST_DIR_PATH] = (FileType.DIR, b"")
    backend = E2BSandbox(
        sandbox=FakeSandbox(
            commands=FakeCommands(
                result=SimpleNamespace(stdout="", stderr="", exit_code=0)
            ),
            files=files,
        )
    )

    response = backend.download_files([TEST_DIR_PATH])[0]

    assert response.error == "is_directory"
    assert response.content is None


def test_upload_and_download_round_trip() -> None:
    files = FakeFiles()
    backend = E2BSandbox(
        sandbox=FakeSandbox(
            commands=FakeCommands(
                result=SimpleNamespace(stdout="", stderr="", exit_code=0)
            ),
            files=files,
        )
    )

    upload = backend.upload_files([(TEST_FILE_PATH, b"hello")])[0]
    download = backend.download_files([TEST_FILE_PATH])[0]

    assert upload.error is None
    assert download.error is None
    assert download.content == b"hello"
