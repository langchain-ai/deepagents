"""Daytona sandbox backend implementation."""

from __future__ import annotations

import daytona
from daytona import FileDownloadRequest, FileUpload
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox


class DaytonaSandbox(BaseSandbox):
    """Daytona sandbox implementation conforming to SandboxBackendProtocol.

    This implementation inherits all file operation methods from BaseSandbox
    and only implements the execute() method using Daytona's API.
    """

    def __init__(self, *, sandbox: daytona.Sandbox) -> None:
        """Create a backend wrapping an existing Daytona sandbox."""
        self._sandbox = sandbox
        self._default_timeout: int = 30 * 60

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
        result = self._sandbox.process.exec(command, timeout=effective_timeout)

        return ExecuteResponse(
            output=result.result,
            exit_code=result.exit_code,
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
