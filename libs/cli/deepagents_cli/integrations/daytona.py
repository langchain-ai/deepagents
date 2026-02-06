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
from deepagents.backends.sandbox import (
    BaseSandbox,
    SandboxClient,
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
            ExecuteResponse with combined output, exit code, optional signal, and
                truncation flag.
        """
        result = self._sandbox.process.exec(command, timeout=self._timeout)

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
                content=resp.result,
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


class DaytonaSandboxClient(SandboxClient):
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

    def get(
        self,
        *,
        sandbox_id: str,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"DaytonaSandboxClient.get() got unsupported kwargs: {keys}"
            raise ValueError(msg)
        sandbox = self._client.get(sandbox_id)
        return DaytonaBackend(sandbox)

    def create(
        self,
        *,
        timeout: int = 180,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"DaytonaSandboxClient.create() got unsupported kwargs: {keys}"
            raise ValueError(msg)

        sandbox = self._client.create()

        for _ in range(timeout // 2):
            try:
                result = sandbox.process.exec("echo ready", timeout=5)
                if result.exit_code == 0:
                    break
            except Exception:  # noqa: BLE001
                time.sleep(2)
                continue
            time.sleep(2)
        else:
            try:
                self._client.delete(sandbox)
            finally:
                msg = f"Daytona sandbox failed to start within {timeout} seconds"
                raise RuntimeError(msg)

        return DaytonaBackend(sandbox)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        if sandbox_id is None:
            return self.create(timeout=timeout, **kwargs)
        return self.get(sandbox_id=sandbox_id, **kwargs)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"DaytonaSandboxClient.delete() got unsupported kwargs: {keys}"
            raise ValueError(msg)
        try:
            sandbox = self._client.get(sandbox_id)
        except Exception:
            return
        try:
            self._client.delete(sandbox)
        except Exception:
            return
n
