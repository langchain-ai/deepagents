"""Daytona sandbox backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.backends.protocol import ExecuteResponse
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    from daytona import Sandbox, FileDownloadRequest


class DaytonaBackend(BaseSandbox):
    """Daytona backend implementation conforming to SandboxBackendProtocol.

    This implementation inherits all file operation methods from BaseSandbox
    and only implements the execute() method using Daytona's API.
    """

    def __init__(self, sandbox: Sandbox) -> None:
        """Initialize the DaytonaBackend with a Daytona sandbox client.

        Args:
            sandbox: Daytona sandbox instance
        """
        self._sandbox = sandbox
        self._timeout: int = 30 * 60  # 30 mins

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""
        return self._sandbox.id

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """Execute a command in the sandbox and return ExecuteResponse.

        Args:
            command: Full shell command string to execute.

        Returns:
            ExecuteResponse with combined output, exit code, optional signal, and truncation flag.
        """
        result = self._sandbox.process.exec(command, timeout=self._timeout)

        return ExecuteResponse(
            output=result.result,  # Daytona combines stdout/stderr
            exit_code=result.exit_code,
            truncated=False,
        )

    def download_file(self, path: str) -> bytes:
        """Download a file from the Modal sandbox.

        Args:
            path: Full path of the file to download.

        Returns:
            File contents as bytes.
        """
        files_download_requests = [
            FileDownloadRequest(
                source=path,
            )
        ]
        results = self._sandbox.fs.download_files(files_download_requests)
        if len(results) > 1:
            raise ValueError("Expected a single file download result.")
        if len(results) == 0:
            raise ValueError("No file download results returned.")
        result = results[0]
        if result.error is not None:




    def upload_file(self, path: str, content: bytes) -> None:
        """Upload a file to the Modal sandbox.

        Args:
            path: Full path where the file should be uploaded.
            content: File contents as bytes.
        """
        # This implementation relies on the Modal sandbox file API.
        # https://modal.com/doc/guide/sandbox-files
        # The API is currently in alpha and is not recommended for production use.
        # We're OK using it here as it's targeting the CLI application.
        with self._sandbox.open(path, "wb") as f:
            f.write(content)
