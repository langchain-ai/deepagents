"""Daytona sandbox backend implementation."""

from __future__ import annotations

import time
import uuid

import daytona
from daytona import FileDownloadRequest, FileUpload, SessionExecuteRequest
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

_POLL_INTERVAL = 0.5
_MAX_POLL_INTERVAL = 5.0


class DaytonaSandbox(BaseSandbox):
    """Daytona sandbox implementation conforming to SandboxBackendProtocol.

    This implementation inherits all file operation methods from BaseSandbox
    and only implements the execute() method using Daytona's API.

    Execution uses Daytona's session-based API (`execute_session_command` with
    `run_async=True`) instead of `process.exec` to avoid the 5-minute HTTP
    connection timeout in the Python SDK.  The command is dispatched
    asynchronously and then polled for completion, so arbitrarily long-running
    commands are supported without holding open a single HTTP connection.
    """

    def __init__(self, *, sandbox: daytona.Sandbox) -> None:
        """Create a backend wrapping an existing Daytona sandbox."""
        self._sandbox = sandbox
        self._default_timeout: int = 30 * 60
        self._session_id: str | None = None

    @property
    def id(self) -> str:
        """Return the Daytona sandbox id."""
        return self._sandbox.id

    def _ensure_session(self) -> str:
        """Return an existing session id, creating one if needed."""
        if self._session_id is None:
            sid = f"da-exec-{uuid.uuid4().hex[:12]}"
            self._sandbox.process.create_session(sid)
            self._session_id = sid
        return self._session_id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox.

        Uses Daytona's session API with async dispatch and polling to avoid
        the 5-minute HTTP connection timeout in the Python SDK.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If None, uses the backend's default timeout (30 minutes).
                A timeout of 0 means "wait indefinitely".
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        session_id = self._ensure_session()

        resp = self._sandbox.process.execute_session_command(
            session_id,
            SessionExecuteRequest(command=command, run_async=True),
        )
        cmd_id = resp.cmd_id

        deadline = time.monotonic() + effective_timeout if effective_timeout else None
        interval = _POLL_INTERVAL

        while True:
            cmd = self._sandbox.process.get_session_command(session_id, cmd_id)
            if cmd.exit_code is not None:
                break
            if deadline is not None and time.monotonic() >= deadline:
                return ExecuteResponse(
                    output=f"Command timed out after {effective_timeout}s",
                    exit_code=124,
                    truncated=False,
                )
            time.sleep(interval)
            interval = min(interval * 1.5, _MAX_POLL_INTERVAL)

        logs = self._sandbox.process.get_session_command_logs(session_id, cmd_id)
        output_parts: list[str] = []
        if logs.stdout:
            output_parts.append(logs.stdout)
        if logs.stderr:
            output_parts.append(logs.stderr)
        output = "\n".join(output_parts) if output_parts else ""

        return ExecuteResponse(
            output=output,
            exit_code=cmd.exit_code,
            truncated=False,
        )

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
