"""Runloop sandbox implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runloop_api_client.sdk import Devbox

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox


class RunloopSandbox(BaseSandbox):
    """Sandbox backend that operates on a Runloop devbox."""

    def __init__(
        self,
        *,
        devbox: Devbox,
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
        result = self._devbox.cmd.exec(command)

        output = result.stdout() if result.stdout() is not None else ""
        stderr = result.stderr() if result.stderr() is not None else ""
        if stderr:
            output += "\n" + stderr if output else stderr

        return ExecuteResponse(
            output=output,
            exit_code=result.exit_code,
            truncated=False,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the devbox."""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            content = self._devbox.file.download(path=path)
            responses.append(
                FileDownloadResponse(path=path, content=content, error=None)
            )
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the devbox."""
        responses: list[FileUploadResponse] = []
        for path, content in files:
            self._devbox.file.upload(path=path, file=content)
            responses.append(FileUploadResponse(path=path, error=None))
        return responses
