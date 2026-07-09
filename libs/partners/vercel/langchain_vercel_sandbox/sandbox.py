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
from deepagents.backends.sandbox import MAX_BINARY_BYTES, BaseSandbox

if TYPE_CHECKING:
    from vercel.sandbox import Command, CommandFinished, Sandbox, WriteFile

logger = logging.getLogger(__name__)

MAX_OUTPUT_BYTES = 100_000
_DOWNLOAD_CHUNK_BYTES = 64 * 1024


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
                called without an explicit `timeout`. A timeout of 0 waits
                indefinitely; negative values are rejected.

        Raises:
            ValueError: If `timeout` is negative.
        """
        if timeout < 0:
            msg = f"timeout must be non-negative, got {timeout}"
            raise ValueError(msg)
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

        # `stdout()`/`stderr()` re-fetch logs over the network (the Vercel SDK
        # streams them via `get_logs`), so a transient failure here must not
        # escape `execute()` and discard the result of a command that already
        # ran. Preserve the exit code and report that output was unavailable.
        try:
            output = current.stdout() or ""
            stderr = current.stderr() or ""
        except Exception:  # noqa: BLE001  # log fetch is a network call; keep the exit code
            logger.warning(
                "Failed to fetch output for completed command in Vercel sandbox "
                "%s; returning the exit code without output.",
                self._sandbox.sandbox_id,
                exc_info=True,
            )
            return ExecuteResponse(
                output="<output unavailable: failed to fetch command logs>",
                exit_code=current.exit_code,
                truncated=False,
            )

        if stderr.strip():
            output += f"\n<stderr>{stderr.strip()}</stderr>"

        truncated = False
        if len(output) > MAX_OUTPUT_BYTES:
            output = output[:MAX_OUTPUT_BYTES]
            output += f"\n\n... Output truncated at {MAX_OUTPUT_BYTES} bytes."
            truncated = True

        return ExecuteResponse(
            output=output,
            exit_code=current.exit_code,
            truncated=truncated,
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
                content, too_large = _read_file_with_limit(self._sandbox, path)
            except Exception as exc:  # noqa: BLE001  # Provider exceptions vary by SDK version
                if _is_sandbox_not_found_error(exc):
                    content = None
                    too_large = False
                else:
                    responses.append(
                        FileDownloadResponse(
                            path=path,
                            content=None,
                            error=_map_file_error(exc),
                        )
                    )
                    continue
            if too_large:
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error=f"file_too_large:max_size_bytes={MAX_BINARY_BYTES}",
                    )
                )
            elif content is None:
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error="sandbox not found",
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


def _read_file_with_limit(sandbox: Sandbox, path: str) -> tuple[bytes | None, bool]:
    chunks_iter = sandbox.iter_file(path, chunk_size=_DOWNLOAD_CHUNK_BYTES)
    if chunks_iter is None:
        return None, False

    chunks: list[bytes] = []
    total = 0
    for chunk in chunks_iter:
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_BINARY_BYTES:
            return None, True
        chunks.append(chunk)
    return b"".join(chunks), False


def _is_sandbox_not_found_error(exc: Exception) -> bool:
    return exc.__class__.__name__ == "SandboxNotFoundError"


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

    # Substring heuristics for SDKs that raise plain exceptions. The groups are
    # disjoint today; ordered defensively so a more specific phrase still wins
    # first should future entries ever overlap.
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
