"""Daytona sandbox backend implementation."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from daytona import Daytona, DaytonaConfig, FileDownloadRequest, FileUpload
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    SandboxBackendProtocol,
)
from deepagents.backends.sandbox import (
    BaseSandbox,
    SandboxClient,
    SandboxNotFoundError,
)

if TYPE_CHECKING:
    from daytona import Sandbox


class DaytonaSandbox(BaseSandbox):
    """Daytona sandbox implementation conforming to SandboxBackendProtocol.

    This implementation inherits all file operation methods from BaseSandbox
    and only implements the execute() method using Daytona's API.
    """

    def __init__(self, sandbox: Sandbox) -> None:
        """Create a backend wrapping an existing Daytona sandbox."""
        self._sandbox = sandbox
        self._timeout: int = 30 * 60

    @property
    def id(self) -> str:
        """Return the Daytona sandbox id."""
        return self._sandbox.id

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox."""
        result = self._sandbox.process.exec(command, timeout=self._timeout)

        return ExecuteResponse(
            output=result.result,
            exit_code=result.exit_code,
            truncated=False,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox."""
        download_requests = [FileDownloadRequest(source=path) for path in paths]
        daytona_responses = self._sandbox.fs.download_files(download_requests)

        return [
            FileDownloadResponse(
                path=resp.source,
                content=resp.result,
                error=None,
            )
            for resp in daytona_responses
        ]

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the sandbox."""
        upload_requests = [
            FileUpload(source=content, destination=path) for path, content in files
        ]
        self._sandbox.fs.upload_files(upload_requests)

        return [FileUploadResponse(path=path, error=None) for path, _ in files]


DaytonaBackend = DaytonaSandbox


class DaytonaSandboxClient(SandboxClient):
    """Daytona sandbox provider implementation."""

    def __init__(
        self, *, api_key: str | None = None, client: Daytona | None = None
    ) -> None:
        """Create a provider backed by the Daytona SDK."""
        if api_key and client:
            msg = "Provide either api_key or client, not both."
            raise ValueError(msg)

        if client is not None:
            self._client = client
            return

        api_key = api_key or os.environ.get("DAYTONA_API_KEY")
        if not api_key:
            msg = "Provide either client or api_key (or set DAYTONA_API_KEY)."
            raise ValueError(msg)

        self._client = Daytona(DaytonaConfig(api_key=api_key))

    @property
    def client(self) -> Daytona:
        """Expose the underlying Daytona client instance."""
        return self._client

    def get(
        self,
        *,
        sandbox_id: str,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Get an existing Daytona sandbox."""
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"DaytonaProvider.get() got unsupported kwargs: {keys}"
            raise ValueError(msg)
        try:
            sandbox = self._client.get(sandbox_id)
        except Exception as e:
            raise SandboxNotFoundError(sandbox_id) from e
        return DaytonaSandbox(sandbox)

    def create(
        self,
        *,
        timeout: int = 180,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Create a new Daytona sandbox."""
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"DaytonaProvider.create() got unsupported kwargs: {keys}"
            raise ValueError(msg)

        sandbox = self._client.create()

        for _ in range(timeout // 2):
            try:
                result = sandbox.process.exec("echo ready", timeout=5)
                if result.exit_code == 0:
                    break
            except Exception:  # noqa: BLE001
                # Ok: startup errors vary; we retry then timeout.
                time.sleep(2)
                continue
            time.sleep(2)
        else:
            try:
                sandbox.delete()
            finally:
                msg = f"Daytona sandbox failed to start within {timeout} seconds"
                raise RuntimeError(msg)

        return DaytonaSandbox(sandbox)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Deprecated: use get() or create()."""
        if sandbox_id is None:
            return self.create(timeout=timeout, **kwargs)
        return self.get(sandbox_id=sandbox_id, **kwargs)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:
        """Delete a sandbox by id."""
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"DaytonaProvider.delete() got unsupported kwargs: {keys}"
            raise ValueError(msg)
        try:
            sandbox = self._client.get(sandbox_id)
        except Exception:  # noqa: BLE001
            return
        try:
            self._client.delete(sandbox)
        except Exception:  # noqa: BLE001
            return
