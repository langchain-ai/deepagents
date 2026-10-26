"""Daytona sandbox backend implementation."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import daytona
from daytona import FileDownloadRequest, FileUpload, SessionExecuteRequest
from daytona.common.errors import DaytonaError, DaytonaTimeoutError
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    from daytona.common.process import SessionCommandLogsResponse

logger = logging.getLogger(__name__)

SyncPollingInterval = float | Callable[[float], float]
PollingStrategy = Callable[[float], float]

_MIN_REQUEST_TIMEOUT = 1.0
"""Smallest safe bounded HTTP request timeout in seconds.

Daytona 0.225.0 rounds positive request timeouts below one second up to one
second and treats zero as no timeout. Do not start a bounded request when less
than one second remains in its budget.
"""

_EXECUTE_REQUEST_TIMEOUT_BUFFER = 5
"""Extra seconds Daytona 0.225.0 adds to `execute_session_command(timeout=...)`.

The integration subtracts this buffer before calling the SDK so the effective
HTTP timeout does not exceed the command's remaining deadline.
"""

_CLEANUP_GRACE_SECONDS = 2.0
"""Total best-effort cleanup budget after an execution timeout, in seconds."""

_CLEANUP_REQUEST_TIMEOUT = 1.0
"""Maximum timeout per cleanup request, in seconds."""


def _remaining(deadline: float | None) -> float | None:
    """Return non-negative seconds left before `deadline`, or `None` if unbounded."""
    if deadline is None:
        return None
    return max(deadline - time.monotonic(), 0.0)


def _has_request_budget(deadline: float | None) -> bool:
    """Whether there is enough time to make a bounded request safely."""
    remaining = _remaining(deadline)
    return remaining is None or remaining >= _MIN_REQUEST_TIMEOUT


def _sleep_until_deadline(deadline: float | None) -> None:
    """Wait only for the remainder of a finite command deadline."""
    remaining = _remaining(deadline)
    if remaining is not None and remaining > 0:
        time.sleep(remaining)


def _cleanup_request_timeout(cleanup_deadline: float) -> float | None:
    """Return a cleanup request timeout within the shared grace budget."""
    remaining = _remaining(cleanup_deadline)
    if remaining is None or remaining < _MIN_REQUEST_TIMEOUT:
        return None
    return min(_CLEANUP_REQUEST_TIMEOUT, remaining)


def _execute_request_timeout(remaining: float | None) -> int:
    """Convert a remaining deadline into Daytona 0.225.0's execute timeout.

    That SDK adds five seconds to the supplied value when it builds the HTTP
    request timeout. Flooring keeps the effective HTTP timeout at or below the
    available time. A resulting zero is avoided because the SDK interprets zero
    as no timeout; -1 produces a four-second HTTP budget instead.
    """
    if remaining is None:
        return 0
    whole_seconds = math.floor(remaining)
    timeout = whole_seconds - _EXECUTE_REQUEST_TIMEOUT_BUFFER
    return -1 if timeout == 0 else timeout


def _format_logs(logs: SessionCommandLogsResponse) -> str:
    """Combine a command's stdout and stderr into a single output string."""
    output = logs.stdout or ""
    if logs.stderr is not None and logs.stderr.strip():
        output += f"\n<stderr>{logs.stderr.strip()}</stderr>"
    return output


class DaytonaSandbox(BaseSandbox):
    """Daytona sandbox implementation conforming to SandboxBackendProtocol.

    This implementation inherits all file operation methods from BaseSandbox
    and only implements the execute() method using Daytona's API.
    """

    def __init__(
        self,
        *,
        sandbox: daytona.Sandbox,
        timeout: int = 30 * 60,
        sync_polling_interval: SyncPollingInterval = 0.1,
    ) -> None:
        """Create a backend wrapping an existing Daytona sandbox.

        Args:
            sandbox: Existing Daytona sandbox instance to wrap.
            timeout: Default command timeout in seconds used when `execute()` is
                called without an explicit `timeout`.
            sync_polling_interval: Delay in seconds between polling Daytona for
                command completion on the sync execution path, or a callable
                that receives elapsed execution time in seconds and returns the
                next polling delay. This will eventually only appear on the
                sync path once an optimized async implementation is available.
        """
        self._sandbox = sandbox
        self._default_timeout = timeout
        polling_strategy: PollingStrategy
        if callable(sync_polling_interval):
            polling_strategy = cast("PollingStrategy", sync_polling_interval)
        else:

            def polling_strategy(_elapsed: float) -> float:
                return sync_polling_interval

        self._sync_polling_interval = polling_strategy

    @property
    def id(self) -> str:
        """Return the Daytona sandbox id."""
        return self._sandbox.id

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

                Note that in Daytona's implementation, a timeout of 0 means
                "wait indefinitely".
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        return self._execute_via_session_logs(command, timeout=effective_timeout)

    def _execute_via_session_logs(
        self,
        command: str,
        *,
        timeout: int,
    ) -> ExecuteResponse:
        """Execute a command with one deadline for all command-phase requests.

        Timed-out commands get one shared, two-second best-effort cleanup budget
        for retrieving partial logs and deleting the session.
        """
        started_at = time.monotonic()
        deadline = None if timeout == 0 else started_at + timeout
        if not _has_request_budget(deadline):
            _sleep_until_deadline(deadline)
            return self._timeout_result(timeout)

        session_id = str(uuid4())
        cleanup_deadline: float | None = None
        session_creation_attempted = False
        try:
            # A transport timeout does not prove that the remote session wasn't
            # created, so the finally block still makes a bounded delete attempt.
            session_creation_attempted = True
            try:
                self._sandbox.process.create_session(
                    session_id, request_timeout=_remaining(deadline)
                )
            except DaytonaTimeoutError:
                if deadline is None:
                    raise
                response, cleanup_deadline = self._timeout_response_with_grace(
                    session_id, None, timeout=timeout
                )
                return response

            result, cleanup_deadline = self._run_in_session(
                session_id,
                command,
                timeout=timeout,
                started_at=started_at,
                deadline=deadline,
            )
            return result
        finally:
            if session_creation_attempted:
                if cleanup_deadline is None:
                    cleanup_deadline = time.monotonic() + _CLEANUP_GRACE_SECONDS
                self._delete_session(session_id, cleanup_deadline)

    def _run_in_session(
        self,
        session_id: str,
        command: str,
        *,
        timeout: int,
        started_at: float,
        deadline: float | None,
    ) -> tuple[ExecuteResponse, float | None]:
        """Start a command, then delegate waiting and final-log collection."""
        process = self._sandbox.process
        remaining = _remaining(deadline)
        if remaining is not None and remaining < _MIN_REQUEST_TIMEOUT:
            _sleep_until_deadline(deadline)
            return self._timeout_response_with_grace(session_id, None, timeout=timeout)

        try:
            result = process.execute_session_command(
                session_id,
                SessionExecuteRequest(command=command, run_async=True),
                timeout=_execute_request_timeout(remaining),
            )
        except DaytonaTimeoutError:
            if deadline is None:
                raise
            return self._timeout_response_with_grace(session_id, None, timeout=timeout)

        return self._finish_started_command(
            session_id,
            result.cmd_id,
            timeout=timeout,
            started_at=started_at,
            deadline=deadline,
        )

    def _finish_started_command(
        self,
        session_id: str,
        cmd_id: str,
        *,
        timeout: int,
        started_at: float,
        deadline: float | None,
    ) -> tuple[ExecuteResponse, float | None]:
        """Wait for a started command and collect logs within the deadline."""
        if deadline is not None and time.monotonic() >= deadline:
            return self._timeout_response_with_grace(
                session_id, cmd_id, timeout=timeout
            )

        exit_code = self._wait_for_exit(
            session_id, cmd_id, started_at=started_at, deadline=deadline
        )
        if exit_code is None:
            return self._timeout_response_with_grace(
                session_id, cmd_id, timeout=timeout
            )

        if not _has_request_budget(deadline):
            _sleep_until_deadline(deadline)
            return self._timeout_response_with_grace(
                session_id, cmd_id, timeout=timeout
            )

        try:
            logs = self._sandbox.process.get_session_command_logs(
                session_id, cmd_id, request_timeout=_remaining(deadline)
            )
        except DaytonaTimeoutError:
            if deadline is None:
                raise
            return self._timeout_response_with_grace(
                session_id, cmd_id, timeout=timeout
            )

        if deadline is not None and time.monotonic() >= deadline:
            return self._timeout_response_with_grace(
                session_id, cmd_id, timeout=timeout, partial_logs=logs
            )

        return (
            ExecuteResponse(
                output=_format_logs(logs),
                exit_code=exit_code,
                truncated=False,
            ),
            None,
        )

    def _wait_for_exit(
        self,
        session_id: str,
        cmd_id: str,
        *,
        started_at: float,
        deadline: float | None,
    ) -> int | None:
        """Poll until exit, or return ``None`` when the command deadline expires."""
        while True:
            if not _has_request_budget(deadline):
                _sleep_until_deadline(deadline)
                return None

            try:
                status = self._sandbox.process.get_session_command(
                    session_id, cmd_id, request_timeout=_remaining(deadline)
                )
            except DaytonaTimeoutError:
                if deadline is None:
                    raise
                # A timeout from the SDK's status request is not itself proof
                # the whole deadline elapsed; wait out the unused portion.
                _sleep_until_deadline(deadline)
                return None

            if deadline is not None and time.monotonic() >= deadline:
                return None
            if status.exit_code is not None:
                return status.exit_code

            delay = self._sync_polling_interval(time.monotonic() - started_at)
            remaining = _remaining(deadline)
            if remaining is not None:
                if remaining < _MIN_REQUEST_TIMEOUT:
                    _sleep_until_deadline(deadline)
                    return None
                delay = min(delay, remaining)
            time.sleep(delay)

    def _timeout_response_with_grace(
        self,
        session_id: str,
        cmd_id: str | None,
        *,
        timeout: int,
        partial_logs: SessionCommandLogsResponse | None = None,
    ) -> tuple[ExecuteResponse, float]:
        """Create a timeout response and its one shared cleanup deadline."""
        cleanup_deadline = time.monotonic() + _CLEANUP_GRACE_SECONDS
        response = self._timeout_response(
            session_id,
            cmd_id,
            timeout=timeout,
            cleanup_deadline=cleanup_deadline,
            partial_logs=partial_logs,
        )
        return response, cleanup_deadline

    def _timeout_result(self, timeout: int) -> ExecuteResponse:
        """Return a timeout result when no command id is available for log recovery."""
        return ExecuteResponse(
            output=f"Command timed out after {timeout} seconds",
            exit_code=124,
            truncated=False,
        )

    def _timeout_response(
        self,
        session_id: str,
        cmd_id: str | None,
        *,
        timeout: int,
        cleanup_deadline: float,
        partial_logs: SessionCommandLogsResponse | None = None,
    ) -> ExecuteResponse:
        """Build a timeout result and best-effort recover logs within shared grace."""
        output = f"Command timed out after {timeout} seconds"
        logs = partial_logs
        request_timeout = _cleanup_request_timeout(cleanup_deadline)
        if logs is None and cmd_id is not None and request_timeout is not None:
            try:
                logs = self._sandbox.process.get_session_command_logs(
                    session_id, cmd_id, request_timeout=request_timeout
                )
            except DaytonaError:
                logger.warning(
                    "Could not read partial output after timeout", exc_info=True
                )

        if logs is not None:
            partial_output = _format_logs(logs)
            if partial_output:
                output = f"{output}\n{partial_output}"
        return ExecuteResponse(output=output, exit_code=124, truncated=False)

    def _delete_session(self, session_id: str, cleanup_deadline: float) -> None:
        """Delete the session without exceeding the caller's shared cleanup grace."""
        request_timeout = _cleanup_request_timeout(cleanup_deadline)
        if request_timeout is None:
            logger.warning(
                "Skipping session deletion for %s: cleanup grace exhausted", session_id
            )
            return
        try:
            self._sandbox.process.delete_session(
                session_id, request_timeout=request_timeout
            )
        except DaytonaError:
            logger.warning("Could not delete session %s", session_id, exc_info=True)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox."""
        download_requests: list[FileDownloadRequest] = []
        responses: list[FileDownloadResponse] = []

        for path in paths:
            if not path.startswith("/"):
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="invalid_path")
                )
                continue
            download_requests.append(FileDownloadRequest(source=path))
            responses.append(FileDownloadResponse(path=path, content=None, error=None))

        if not download_requests:
            return responses

        daytona_responses = self._sandbox.fs.download_files(download_requests)

        mapped_responses: list[FileDownloadResponse] = []
        for resp in daytona_responses:
            content = resp.result
            if content is None:
                mapped_responses.append(
                    FileDownloadResponse(
                        path=resp.source,
                        content=None,
                        error="file_not_found",
                    )
                )
            else:
                mapped_responses.append(
                    FileDownloadResponse(
                        path=resp.source,
                        content=content,  # ty: ignore[invalid-argument-type]  # Daytona SDK returns bytes for file content
                        error=None,
                    )
                )

        mapped_iter = iter(mapped_responses)
        for i, path in enumerate(paths):
            if not path.startswith("/"):
                continue
            responses[i] = next(
                mapped_iter,
                FileDownloadResponse(path=path, content=None, error="file_not_found"),
            )

        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the sandbox."""
        upload_requests: list[FileUpload] = []
        responses: list[FileUploadResponse] = []

        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            upload_requests.append(FileUpload(source=content, destination=path))
            responses.append(FileUploadResponse(path=path, error=None))

        if upload_requests:
            self._sandbox.fs.upload_files(upload_requests)

        return responses
