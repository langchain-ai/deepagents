"""Vercel Sandbox backend implementation."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


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
            try:
                cmd.kill()
            except Exception:  # noqa: BLE001  # best-effort cleanup; surface to logs
                logger.warning(
                    "Failed to kill timed-out command in Vercel sandbox %s; the "
                    "command may still be running and incurring cost.",
                    self._sandbox.sandbox_id,
                    exc_info=True,
                )
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
            # `write_files` is a single batched call, so a failure is
            # batch-level: every valid path receives the same error. For
            # unrecognized failures this is the real provider message (see
            # `_map_file_error`) rather than a fabricated per-file code.
            error = _map_file_error(exc)
            for i, (path, _content) in enumerate(files):
                if path.startswith("/"):
                    responses[i] = FileUploadResponse(path=path, error=error)

        return responses


def _map_file_error(exc: Exception) -> FileOperationError | str:
    """Map a provider filesystem failure to a Deep Agents file error.

    Recognized failures map to a `FileOperationError` literal. Unrecognized
    exceptions return their string representation rather than defaulting to
    `FILE_NOT_FOUND`, so that auth, network, or transient SDK failures are
    surfaced to the agent instead of masquerading as a missing file.
    """
    if isinstance(exc, PermissionError):
        return PERMISSION_DENIED
    if isinstance(exc, IsADirectoryError):
        return IS_DIRECTORY
    if isinstance(exc, FileNotFoundError):
        return FILE_NOT_FOUND

    # Ordered substring heuristics for SDKs that raise plain exceptions. Order
    # matters: more specific phrases are matched before broader ones.
    message = str(exc).lower()
    substring_errors: tuple[tuple[tuple[str, ...], FileOperationError], ...] = (
        (("permission", "forbidden", "access denied"), PERMISSION_DENIED),
        (("is a directory",), IS_DIRECTORY),
        (("invalid path",), INVALID_PATH),
        (("no such file",), FILE_NOT_FOUND),
    )
    for needles, error in substring_errors:
        if any(needle in message for needle in needles):
            return error
    # Unrecognized: surface the backend's own message instead of guessing.
    return str(exc) or FILE_NOT_FOUND


def _wait_for_command(
    cmd: Command,
    effective_timeout: int,
    started_at: float,
) -> CommandFinished | None:
    """Wait for a Vercel command while preserving local timeout semantics.

    The Vercel SDK's `wait()` has no native timeout or cancellation, so on
    timeout this returns None and intentionally leaves the daemon wait thread
    running until the underlying command finishes. The caller is expected to
    `kill()` the command to unblock it.
    """
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
