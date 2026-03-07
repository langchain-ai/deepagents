"""Daytona sandbox backend implementation."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    SandboxBackendProtocol,
)
from deepagents.backends.sandbox import BaseSandbox

from deepagents_cli.integrations.sandbox_provider import (
    SandboxProvider,
)

if TYPE_CHECKING:
    from daytona import Sandbox


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
        self._default_timeout: int = 30 * 60  # 30 mins

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""
        return self._sandbox.id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a command in the sandbox and return ExecuteResponse.

        Args:
            command: Full shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If None, uses the backend's default timeout.

                Note that in Daytona's implementation, a timeout of 0 means
                "wait indefinitely".

        Returns:
            ExecuteResponse with combined output, exit code, and truncation flag.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        result = self._sandbox.process.exec(command, timeout=effective_timeout)

        return ExecuteResponse(
            output=result.result,  # Daytona combines stdout/stderr
            exit_code=result.exit_code,
            truncated=False,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the Daytona sandbox.

        Leverages Daytona's native batch download API for efficiency.
        Supports partial success - individual downloads may fail without
        affecting others.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.
            Response order matches input order.

        TODO: Map Daytona API error strings to standardized FileOperationError codes.
        Currently only implements happy path.
        """
        from daytona import FileDownloadRequest

        # Create batch download request using Daytona's native batch API
        download_requests = [FileDownloadRequest(source=path) for path in paths]
        daytona_responses = self._sandbox.fs.download_files(download_requests)

        # Convert Daytona results to our response format
        # TODO: Map resp.error to standardized error codes when available
        return [
            FileDownloadResponse(
                path=resp.source,
                content=resp.result.encode()
                if isinstance(resp.result, str)
                else resp.result,
                error=None,  # TODO: map resp.error to FileOperationError
            )
            for resp in daytona_responses
        ]

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the Daytona sandbox.

        Leverages Daytona's native batch upload API for efficiency.
        Supports partial success - individual uploads may fail without
        affecting others.

        Args:
            files: List of (path, content) tuples to upload.

        Returns:
            List of FileUploadResponse objects, one per input file.
            Response order matches input order.

        TODO: Map Daytona API error strings to standardized FileOperationError codes.
        Currently only implements happy path.
        """
        from daytona import FileUpload

        # Create batch upload request using Daytona's native batch API
        upload_requests = [
            FileUpload(source=content, destination=path) for path, content in files
        ]
        self._sandbox.fs.upload_files(upload_requests)

        # TODO: Check if Daytona returns error info and map to FileOperationError codes
        return [FileUploadResponse(path=path, error=None) for path, _ in files]


class DaytonaProvider(SandboxProvider):
    """Daytona sandbox provider implementation.

    Manages Daytona sandbox lifecycle using the Daytona SDK.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize Daytona provider.

        Args:
            api_key: Daytona API key (defaults to DAYTONA_API_KEY env var)

        Raises:
            ValueError: If DAYTONA_API_KEY environment variable not set
        """
        from daytona import Daytona, DaytonaConfig

        self._api_key = api_key or os.environ.get("DAYTONA_API_KEY")
        if not self._api_key:
            msg = "DAYTONA_API_KEY environment variable not set"
            raise ValueError(msg)
        self._client = Daytona(DaytonaConfig(api_key=self._api_key))

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        **kwargs: Any,  # noqa: ARG002  # Required by SandboxFactory interface
    ) -> SandboxBackendProtocol:
        """Get existing or create new Daytona sandbox.

        Args:
            sandbox_id: Not supported yet - must be None
            timeout: Timeout in seconds for sandbox startup (default: 180)
            **kwargs: Additional Daytona-specific parameters

        Returns:
            DaytonaBackend instance

        Raises:
            NotImplementedError: Connecting to existing sandbox not supported
            RuntimeError: Sandbox startup failed
        """
        if sandbox_id:
            msg = (
                "Connecting to existing Daytona sandbox by ID not yet supported. "
                "Create a new sandbox by omitting sandbox_id parameter."
            )
            raise NotImplementedError(msg)

        sandbox = self._client.create()

        # Poll until running
        for _ in range(timeout // 2):
            try:
                result = sandbox.process.exec("echo ready", timeout=5)
                if result.exit_code == 0:
                    break
            except Exception:  # noqa: S110, BLE001  # Sandbox not ready yet, continue polling
                pass
            time.sleep(2)
        else:
            try:
                sandbox.delete()
            finally:
                msg = f"Daytona sandbox failed to start within {timeout} seconds"
                raise RuntimeError(msg)

        return DaytonaBackend(sandbox)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002  # Required by SandboxFactory interface
        """Delete a Daytona sandbox.

        Args:
            sandbox_id: Sandbox ID to delete
            **kwargs: Additional parameters
        """
        sandbox = self._client.get(sandbox_id)
        self._client.delete(sandbox)
