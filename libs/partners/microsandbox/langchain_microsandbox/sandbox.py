"""Microsandbox backend implementation."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, TypeVar, cast

from deepagents.backends.protocol import (
    FILE_NOT_FOUND,
    INVALID_PATH,
    IS_DIRECTORY,
    PERMISSION_DENIED,
    ExecuteResponse,
    FileDownloadResponse,
    FileOperationError,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from microsandbox import ExecTimeoutError, PathNotFoundError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine
    from concurrent.futures import Future

    from microsandbox import ExecOutput, Sandbox

_ResultT = TypeVar("_ResultT")
_TIMEOUT_EXIT_CODE = 124


class MicrosandboxSandbox(BaseSandbox):
    """Microsandbox implementation conforming to `SandboxBackendProtocol`.

    The adapter wraps an existing sandbox. The calling application remains
    responsible for stopping it and choosing its persistence or removal policy.
    """

    def __init__(
        self,
        *,
        sandbox: Sandbox,
        sandbox_id: str,
        timeout: int,
    ) -> None:
        """Initialize an adapter after asynchronous construction.

        Use [`create`][langchain_microsandbox.MicrosandboxSandbox.create] so the
        asynchronous Microsandbox name is cached for the synchronous `id`
        property.

        Args:
            sandbox: Existing Microsandbox instance to wrap.
            sandbox_id: Name resolved from the Microsandbox instance.
            timeout: Default command timeout in seconds.
        """
        self._sandbox = sandbox
        self._id = sandbox_id
        self._default_timeout = timeout

    @classmethod
    async def create(
        cls,
        sandbox: Sandbox,
        *,
        timeout: int = 30 * 60,  # noqa: ASYNC109  # provider execution default
    ) -> MicrosandboxSandbox:
        """Create an adapter around an existing Microsandbox instance.

        Args:
            sandbox: Existing Microsandbox instance to wrap.
            timeout: Default command timeout in seconds used when execution is
                called without an explicit `timeout`.

        Returns:
            An initialized adapter with the sandbox name cached as its `id`.

        Raises:
            ValueError: If `timeout` is negative.
        """
        if timeout < 0:
            msg = f"timeout must be non-negative, got {timeout}"
            raise ValueError(msg)
        # v0.6.12 exposes an awaitable property at runtime even though its stub
        # declares an async method; accept both forms across the supported range.
        name: object = sandbox.name
        if callable(name):
            name_factory = cast("Callable[[], Awaitable[str]]", name)
            sandbox_id = await name_factory()
        else:
            sandbox_id = await cast("Awaitable[str]", name)
        return cls(
            sandbox=sandbox,
            sandbox_id=sandbox_id,
            timeout=timeout,
        )

    @property
    def id(self) -> str:
        """Return the cached Microsandbox name."""
        return self._id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command through the synchronous backend interface.

        Args:
            command: Shell command string to execute.
            timeout: Maximum execution time in seconds. If `None`, use the
                adapter's default timeout.

        Returns:
            The command output and exit status.
        """
        return _run_sync(lambda: self.aexecute(command, timeout=timeout))

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,  # noqa: ASYNC109  # forwarded to Microsandbox
    ) -> ExecuteResponse:
        """Execute a shell command through Microsandbox's native async API.

        Args:
            command: Shell command string to execute.
            timeout: Maximum execution time in seconds. If `None`, use the
                adapter's default timeout.

        Returns:
            The command output and exit status. Provider timeouts use shell exit
            code 124, matching the other Deep Agents sandbox integrations.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        try:
            output = await self._sandbox.shell(
                command,
                timeout=float(effective_timeout),
            )
        except ExecTimeoutError:
            msg = f"Command timed out after {effective_timeout} seconds"
            return ExecuteResponse(
                output=msg,
                exit_code=_TIMEOUT_EXIT_CODE,
                truncated=False,
            )
        return _map_execute_output(output)

    def upload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        """Upload files through the synchronous backend interface.

        Args:
            files: Absolute sandbox paths paired with file contents.

        Returns:
            One response per input file, in input order.
        """
        return _run_sync(lambda: self.aupload_files(files))

    async def aupload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        """Upload files through Microsandbox's native async filesystem API.

        Args:
            files: Absolute sandbox paths paired with file contents.

        Returns:
            One response per input file, in input order.
        """
        return [await self._aupload_file(path, content) for path, content in files]

    async def _aupload_file(self, path: str, content: bytes) -> FileUploadResponse:
        if not path.startswith("/"):
            return FileUploadResponse(path=path, error=INVALID_PATH)

        try:
            parent = str(PurePosixPath(path).parent)
            if parent != "/":
                await self._sandbox.fs.mkdir(parent)
            await self._sandbox.fs.write(path, content)
        except Exception as exc:  # noqa: BLE001  # Provider errors vary by operation
            return FileUploadResponse(path=path, error=_map_file_error(exc))
        return FileUploadResponse(path=path, error=None)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files through the synchronous backend interface.

        Args:
            paths: Absolute sandbox paths to download.

        Returns:
            One response per input path, in input order.
        """
        return _run_sync(lambda: self.adownload_files(paths))

    async def adownload_files(
        self,
        paths: list[str],
    ) -> list[FileDownloadResponse]:
        """Download files through Microsandbox's native async filesystem API.

        Args:
            paths: Absolute sandbox paths to download.

        Returns:
            One response per input path, in input order.
        """
        return [await self._adownload_file(path) for path in paths]

    async def _adownload_file(self, path: str) -> FileDownloadResponse:
        if not path.startswith("/"):
            return FileDownloadResponse(path=path, content=None, error=INVALID_PATH)

        try:
            content = await self._sandbox.fs.read(path)
        except Exception as exc:  # noqa: BLE001  # Provider errors vary by operation
            return FileDownloadResponse(
                path=path,
                content=None,
                error=_map_file_error(exc),
            )
        return FileDownloadResponse(path=path, content=content, error=None)


def _map_execute_output(output: ExecOutput) -> ExecuteResponse:
    """Convert Microsandbox's command result into the Deep Agents response."""
    combined = output.stdout_text
    stderr = output.stderr_text.strip()
    if stderr:
        combined += f"\n<stderr>{stderr}</stderr>"
    return ExecuteResponse(
        output=combined,
        exit_code=output.exit_code,
        truncated=False,
    )


def _map_file_error(exc: Exception) -> FileOperationError | str:
    """Normalize known filesystem failures and preserve unknown provider errors."""
    if isinstance(exc, PermissionError):
        return PERMISSION_DENIED
    if isinstance(exc, IsADirectoryError):
        return IS_DIRECTORY
    if isinstance(exc, (FileNotFoundError, PathNotFoundError)):
        return FILE_NOT_FOUND

    message = str(exc)
    normalized = message.lower()
    substring_errors: tuple[tuple[tuple[str, ...], FileOperationError], ...] = (
        (("permission denied", "access denied", "forbidden"), PERMISSION_DENIED),
        (("is a directory",), IS_DIRECTORY),
        (("invalid path",), INVALID_PATH),
        (("no such file", "not found"), FILE_NOT_FOUND),
    )
    for needles, error in substring_errors:
        if any(needle in normalized for needle in needles):
            return error
    return message or type(exc).__name__


def _run_sync(factory: Callable[[], Coroutine[object, object, _ResultT]]) -> _ResultT:
    """Run an async provider call from synchronous code.

    When the current thread already owns a running event loop, both coroutine
    creation and execution happen in a worker thread. This avoids nesting
    `asyncio.run()` in the caller's loop while keeping the bridge stateless.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    def run() -> _ResultT:
        return asyncio.run(factory())

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="microsandbox-sync",
    ) as executor:
        future: Future[_ResultT] = executor.submit(run)
        return future.result()
