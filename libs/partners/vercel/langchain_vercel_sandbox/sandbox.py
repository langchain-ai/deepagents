"""Vercel Sandbox backend implementation."""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from vercel.sandbox import Command, CommandFinished, Sandbox, WriteFile


class VercelSandbox(BaseSandbox):
    """Vercel Sandbox implementation conforming to SandboxBackendProtocol."""

    def __init__(
        self,
        *,
        sandbox: Sandbox,
        timeout: int = 30 * 60,
    ) -> None:
        """Create a backend wrapping an existing Vercel sandbox.

        Args:
            sandbox: Existing Vercel sandbox instance to wrap.
            timeout: Default command timeout in seconds used when `execute()` is
                called without an explicit `timeout`.
        """
        self._sandbox = sandbox
        self._default_timeout = timeout

    @property
    def id(self) -> str:
        """Return the Vercel sandbox id."""
        return self._sandbox.sandbox_id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If None, uses the backend's default timeout.

                A timeout of 0 waits indefinitely.

        Returns:
            ExecuteResponse containing output, exit code, and truncation flag.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        started_at = time.monotonic()
        cmd = self._sandbox.run_command_detached("bash", ["-lc", command])
        current = _wait_for_command(cmd, effective_timeout, started_at)
        if current is None:
            with contextlib.suppress(Exception):
                cmd.kill()
            msg = f"Command timed out after {effective_timeout} seconds"
            return ExecuteResponse(output=msg, exit_code=124, truncated=False)

        output = current.stdout() or ""
        stderr = current.stderr() or ""
        if stderr.strip():
            output += f"\n<stderr>{stderr.strip()}</stderr>"

        return ExecuteResponse(
            output=output,
            exit_code=current.exit_code,
            truncated=False,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox."""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(
                    FileDownloadResponse(path=path, content=None, error=INVALID_PATH)
                )
                continue
            try:
                content = self._sandbox.read_file(path)
            except Exception as exc:  # noqa: BLE001  # Provider exceptions vary by SDK version
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error=_map_file_error(exc),
                    )
                )
                continue
            if content is None:
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error=FILE_NOT_FOUND,
                    )
                )
            else:
                responses.append(
                    FileDownloadResponse(path=path, content=content, error=None)
                )
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the sandbox."""
        write_files: list[WriteFile] = []
        responses: list[FileUploadResponse] = []

        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error=INVALID_PATH))
                continue
            write_files.append({"path": path, "content": content})
            responses.append(FileUploadResponse(path=path, error=None))

        if not write_files:
            return responses

        try:
            self._sandbox.write_files(write_files)
        except Exception as exc:  # noqa: BLE001  # Provider exceptions vary by SDK version
            error = _map_file_error(exc)
            for i, (path, _content) in enumerate(files):
                if path.startswith("/"):
                    responses[i] = FileUploadResponse(path=path, error=error)

        return responses


def _map_file_error(exc: Exception) -> FileOperationError:
    """Map provider filesystem failures to Deep Agents file error literals."""
    error: FileOperationError = FILE_NOT_FOUND
    if isinstance(exc, PermissionError):
        error = PERMISSION_DENIED
    elif isinstance(exc, IsADirectoryError):
        error = IS_DIRECTORY
    elif isinstance(exc, FileNotFoundError):
        error = FILE_NOT_FOUND
    else:
        message = str(exc).lower()
        if (
            "permission" in message
            or "forbidden" in message
            or "access denied" in message
        ):
            error = PERMISSION_DENIED
        elif "is a directory" in message:
            error = IS_DIRECTORY
        elif "invalid path" in message:
            error = INVALID_PATH
    return error


def _wait_for_command(
    cmd: Command,
    effective_timeout: int,
    started_at: float,
) -> CommandFinished | None:
    """Wait for a Vercel command while preserving local timeout semantics."""
    if effective_timeout == 0:
        return cmd.wait()

    remaining = max(0.0, effective_timeout - (time.monotonic() - started_at))
    result_queue: queue.Queue[tuple[CommandFinished | None, Exception | None]] = (
        queue.Queue(maxsize=1)
    )

    def wait() -> None:
        try:
            result_queue.put((cmd.wait(), None))
        except Exception as exc:  # noqa: BLE001  # re-raise provider wait errors on caller thread
            result_queue.put((None, exc))

    thread = threading.Thread(target=wait, daemon=True)
    thread.start()
    thread.join(remaining)

    if thread.is_alive():
        return None

    current, exc = result_queue.get_nowait()
    if exc is not None:
        raise exc
    return current
