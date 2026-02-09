"""Runloop sandbox implementation."""

from __future__ import annotations

import importlib.util
import time
from typing import Any

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    SandboxBackendProtocol,
)
from deepagents.backends.sandbox import BaseSandbox


class RunloopSandbox(BaseSandbox):
    """Sandbox backend that operates on a Runloop devbox."""

    def __init__(
        self,
        *,
        devbox: Any,
    ) -> None:
        """Create a sandbox backend connected to an existing Runloop devbox."""
        self._devbox = devbox
        self._devbox_id = devbox.id
        self._timeout = 30 * 60

    @property
    def id(self) -> str:
        """Return the devbox id."""
        return self._devbox_id

    def execute(self, command: str) -> ExecuteResponse:
        """Execute a shell command inside the devbox."""
        result = self._devbox.execute_and_await_completion(
            command=command,
            timeout=self._timeout,
        )

        output = result.stdout or ""
        if result.stderr:
            output += "\n" + result.stderr if output else result.stderr

        return ExecuteResponse(
            output=output, exit_code=result.exit_status, truncated=False
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the devbox."""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            resp = self._devbox.download_file(path=path)
            content = resp.read()
            responses.append(
                FileDownloadResponse(path=path, content=content, error=None)
            )
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the devbox."""
        responses: list[FileUploadResponse] = []
        for path, content in files:
            self._devbox.upload_file(path=path, file=content)
            responses.append(FileUploadResponse(path=path, error=None))
        return responses

