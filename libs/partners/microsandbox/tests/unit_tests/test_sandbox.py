from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest
from deepagents.backends.protocol import ExecuteResponse
from microsandbox import ExecTimeoutError, FilesystemError

from langchain_microsandbox import MicrosandboxSandbox

if TYPE_CHECKING:
    from microsandbox import Sandbox

DEFAULT_TIMEOUT = 30 * 60
EXPLICIT_TIMEOUT = 17
TIMEOUT_EXIT_CODE = 124
NON_ZERO_EXIT_CODE = 7


@dataclass
class _Output:
    stdout_text: str = ""
    stderr_text: str = ""
    exit_code: int = 0


class _Filesystem:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.read_errors: dict[str, Exception] = {}
        self.write_errors: dict[str, Exception] = {}
        self.mkdir_errors: dict[str, Exception] = {}
        self.mkdir_calls: list[str] = []
        self.write_calls: list[tuple[str, bytes]] = []
        self.operation_loops: list[asyncio.AbstractEventLoop] = []

    async def mkdir(self, path: str) -> None:
        self.operation_loops.append(asyncio.get_running_loop())
        self.mkdir_calls.append(path)
        if error := self.mkdir_errors.get(path):
            raise error

    async def write(self, path: str, content: bytes) -> None:
        self.operation_loops.append(asyncio.get_running_loop())
        self.write_calls.append((path, content))
        if error := self.write_errors.get(path):
            raise error
        self.files[path] = content

    async def read(self, path: str) -> bytes:
        self.operation_loops.append(asyncio.get_running_loop())
        if error := self.read_errors.get(path):
            raise error
        return self.files[path]


class _Sandbox:
    def __init__(
        self,
        *,
        name: str = "deepagents-session",
        output: _Output | None = None,
    ) -> None:
        self._name = name
        self.output = output or _Output(stdout_text="ok")
        self.name_calls = 0
        self.shell_calls: list[tuple[str, float]] = []
        self.shell_error: Exception | None = None
        self.shell_loop: asyncio.AbstractEventLoop | None = None
        self.shell_thread: int | None = None
        self.fs = _Filesystem()

    async def name(self) -> str:
        self.name_calls += 1
        return self._name

    async def shell(
        self,
        command: str,
        *,
        timeout: float,  # noqa: ASYNC109  # mirrors provider signature
    ) -> _Output:
        self.shell_loop = asyncio.get_running_loop()
        self.shell_thread = threading.get_ident()
        self.shell_calls.append((command, timeout))
        if self.shell_error is not None:
            raise self.shell_error
        return self.output


class _AwaitableNameSandbox(_Sandbox):
    @property
    def name(self) -> asyncio.Future[str]:
        self.name_calls += 1
        future = asyncio.get_running_loop().create_future()
        future.set_result(self._name)
        return future


async def _backend(
    sandbox: _Sandbox,
    *,
    timeout: int = DEFAULT_TIMEOUT,  # noqa: ASYNC109  # forwarded default
) -> MicrosandboxSandbox:
    return await MicrosandboxSandbox.create(
        cast("Sandbox", sandbox),
        timeout=timeout,
    )


async def test_create_caches_sandbox_name() -> None:
    sandbox = _Sandbox(name="cached-name")

    backend = await _backend(sandbox)

    assert backend.id == "cached-name"
    assert backend.id == "cached-name"
    assert sandbox.name_calls == 1


async def test_create_supports_awaitable_name_property() -> None:
    sandbox = _AwaitableNameSandbox(name="runtime-property-name")

    backend = await _backend(sandbox)

    assert backend.id == "runtime-property-name"
    assert sandbox.name_calls == 1


async def test_create_rejects_negative_default_timeout() -> None:
    sandbox = _Sandbox()

    with pytest.raises(ValueError, match="timeout must be non-negative"):
        await _backend(sandbox, timeout=-1)


async def test_aexecute_maps_output_and_uses_current_loop() -> None:
    sandbox = _Sandbox(
        output=_Output(
            stdout_text="out",
            stderr_text="err\n",
            exit_code=NON_ZERO_EXIT_CODE,
        )
    )
    backend = await _backend(sandbox)
    current_loop = asyncio.get_running_loop()

    result = await backend.aexecute("python fail.py", timeout=EXPLICIT_TIMEOUT)

    assert result == ExecuteResponse(
        output="out\n<stderr>err</stderr>",
        exit_code=NON_ZERO_EXIT_CODE,
        truncated=False,
    )
    assert sandbox.shell_calls == [("python fail.py", float(EXPLICIT_TIMEOUT))]
    assert sandbox.shell_loop is current_loop


async def test_aexecute_uses_default_timeout() -> None:
    sandbox = _Sandbox()
    backend = await _backend(sandbox)

    await backend.aexecute("echo ok")

    assert sandbox.shell_calls == [("echo ok", float(DEFAULT_TIMEOUT))]


async def test_aexecute_maps_timeout() -> None:
    sandbox = _Sandbox()
    sandbox.shell_error = ExecTimeoutError("execution timed out")
    backend = await _backend(sandbox)

    result = await backend.aexecute("sleep 100", timeout=EXPLICIT_TIMEOUT)

    assert result == ExecuteResponse(
        output=f"Command timed out after {EXPLICIT_TIMEOUT} seconds",
        exit_code=TIMEOUT_EXIT_CODE,
        truncated=False,
    )


async def test_execute_works_inside_running_event_loop() -> None:
    sandbox = _Sandbox(output=_Output(stdout_text="sync"))
    backend = await _backend(sandbox)
    caller_thread = threading.get_ident()

    result = backend.execute("echo sync")

    assert result.output == "sync"
    assert sandbox.shell_thread != caller_thread


async def test_execute_propagates_provider_error_inside_running_event_loop() -> None:
    sandbox = _Sandbox()
    sandbox.shell_error = RuntimeError("provider unavailable")
    backend = await _backend(sandbox)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        backend.execute("echo fail")


def test_sync_operations_work_without_running_event_loop() -> None:
    sandbox = _Sandbox(output=_Output(stdout_text="sync"))
    sandbox.fs.files["/download.txt"] = b"download"
    backend = asyncio.run(_backend(sandbox))

    execution = backend.execute("echo sync")
    upload = backend.upload_files([("/upload.txt", b"upload")])
    download = backend.download_files(["/download.txt"])

    assert execution.output == "sync"
    assert upload[0].error is None
    assert download[0].content == b"download"


async def test_aupload_files_preserves_order_and_partial_success() -> None:
    sandbox = _Sandbox()
    sandbox.fs.write_errors["/denied.txt"] = FilesystemError("permission denied")
    backend = await _backend(sandbox)
    current_loop = asyncio.get_running_loop()

    responses = await backend.aupload_files(
        [
            ("relative.txt", b"bad"),
            ("/workspace/ok.txt", b"ok"),
            ("/denied.txt", b"no"),
        ]
    )

    assert [response.path for response in responses] == [
        "relative.txt",
        "/workspace/ok.txt",
        "/denied.txt",
    ]
    assert [response.error for response in responses] == [
        "invalid_path",
        None,
        "permission_denied",
    ]
    assert sandbox.fs.mkdir_calls == ["/workspace"]
    assert sandbox.fs.write_calls == [
        ("/workspace/ok.txt", b"ok"),
        ("/denied.txt", b"no"),
    ]
    assert all(loop is current_loop for loop in sandbox.fs.operation_loops)


async def test_aupload_files_maps_parent_creation_error_per_file() -> None:
    sandbox = _Sandbox()
    sandbox.fs.mkdir_errors["/readonly"] = FilesystemError("access denied")
    backend = await _backend(sandbox)

    responses = await backend.aupload_files(
        [("/readonly/a.txt", b"a"), ("/ok.txt", b"ok")]
    )

    assert [response.error for response in responses] == ["permission_denied", None]
    assert sandbox.fs.write_calls == [("/ok.txt", b"ok")]


async def test_adownload_files_maps_errors_and_preserves_order() -> None:
    sandbox = _Sandbox()
    sandbox.fs.files["/ok.txt"] = b"ok"
    sandbox.fs.read_errors["/missing.txt"] = FilesystemError("no such file")
    sandbox.fs.read_errors["/folder"] = FilesystemError("is a directory")
    backend = await _backend(sandbox)

    responses = await backend.adownload_files(
        ["relative.txt", "/ok.txt", "/missing.txt", "/folder"]
    )

    assert [response.path for response in responses] == [
        "relative.txt",
        "/ok.txt",
        "/missing.txt",
        "/folder",
    ]
    assert [response.content for response in responses] == [None, b"ok", None, None]
    assert [response.error for response in responses] == [
        "invalid_path",
        None,
        "file_not_found",
        "is_directory",
    ]


async def test_file_operations_surface_unknown_provider_errors() -> None:
    sandbox = _Sandbox()
    sandbox.fs.write_errors["/write.txt"] = FilesystemError("transport closed")
    sandbox.fs.read_errors["/read.txt"] = FilesystemError("transport closed")
    backend = await _backend(sandbox)

    upload = await backend.aupload_files([("/write.txt", b"data")])
    download = await backend.adownload_files(["/read.txt"])

    assert upload[0].error == "transport closed"
    assert download[0].error == "transport closed"


async def test_sync_file_operations_work_inside_running_event_loop() -> None:
    sandbox = _Sandbox()
    sandbox.fs.files["/download.txt"] = b"download"
    backend = await _backend(sandbox)

    upload = backend.upload_files([("/upload.txt", b"upload")])
    download = backend.download_files(["/download.txt"])

    assert upload[0].error is None
    assert download[0].content == b"download"
