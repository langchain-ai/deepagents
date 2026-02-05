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
    SandboxListResponse,
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
        self._sandbox = sandbox
        self._timeout: int = 30 * 60

    @property
    def id(self) -> str:
        return self._sandbox.id

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        result = self._sandbox.process.exec(command, timeout=self._timeout)

        return ExecuteResponse(
            output=result.result,
            exit_code=result.exit_code,
            truncated=False,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        from daytona import FileDownloadRequest

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
        from daytona import FileUpload

        upload_requests = [
            FileUpload(source=content, destination=path) for path, content in files
        ]
        self._sandbox.fs.upload_files(upload_requests)

        return [FileUploadResponse(path=path, error=None) for path, _ in files]


class DaytonaProvider(SandboxProvider[dict[str, Any]]):
    """Daytona sandbox provider implementation."""

    def __init__(self, api_key: str | None = None) -> None:
        from daytona import Daytona, DaytonaConfig

        self._api_key = api_key or os.environ.get("DAYTONA_API_KEY")
        if not self._api_key:
            msg = "DAYTONA_API_KEY environment variable not set"
            raise ValueError(msg)
        self._client = Daytona(DaytonaConfig(api_key=self._api_key))

    def list(
        self,
        *,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> SandboxListResponse[dict[str, Any]]:
        msg = "Listing with Daytona SDK not yet implemented"
        raise NotImplementedError(msg)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        if sandbox_id:
            msg = (
                "Connecting to existing Daytona sandbox by ID not yet supported. "
                "Create a new sandbox by omitting sandbox_id parameter."
            )
            raise NotImplementedError(msg)

        sandbox = self._client.create()

        for _ in range(timeout // 2):
            try:
                result = sandbox.process.exec("echo ready", timeout=5)
                if result.exit_code == 0:
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
        else:
            try:
                sandbox.delete()
            finally:
                msg = f"Daytona sandbox failed to start within {timeout} seconds"
                raise RuntimeError(msg)

        return DaytonaBackend(sandbox)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:
        sandbox = self._client.get(sandbox_id)
        self._client.delete(sandbox)
