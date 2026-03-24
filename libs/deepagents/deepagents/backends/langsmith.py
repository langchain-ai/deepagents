"""LangSmith sandbox backend implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    WriteResult,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    from langsmith.sandbox import Sandbox

logger = logging.getLogger(__name__)


class LangSmithSandbox(BaseSandbox):
    """LangSmith sandbox implementation conforming to `SandboxBackendProtocol`.

    This implementation inherits all file operation methods from `BaseSandbox`
    and only implements the execute() method using LangSmith's API.
    """

    def __init__(self, sandbox: Sandbox) -> None:
        """Create a backend wrapping an existing LangSmith sandbox.

        Args:
            sandbox: LangSmith Sandbox instance to wrap.
        """
        self._sandbox = sandbox
        self._default_timeout: int = 30 * 60

    @property
    def id(self) -> str:
        """Return the LangSmith sandbox name."""
        return self._sandbox.name

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Execute a shell command inside the sandbox.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If None, uses the backend's default timeout.
                A value of 0 disables the command timeout when the
                `langsmith[sandbox]` extra is installed.

        Returns:
            `ExecuteResponse` containing output, exit code, and truncation flag.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        result = self._sandbox.run(command, timeout=effective_timeout)

        output = result.stdout or ""
        if result.stderr:
            output += "\n" + result.stderr if output else result.stderr

        return ExecuteResponse(
            output=output,
            exit_code=result.exit_code,
            truncated=False,
        )

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write content using the LangSmith SDK to avoid ARG_MAX.

        `BaseSandbox.write()` sends the full content in a shell command, which
        can exceed ARG_MAX for large content. This override uses the SDK's
        native `write()`, which sends content in the HTTP body.

        Args:
            file_path: Destination path inside the sandbox.
            content: Text content to write.

        Returns:
            `WriteResult` with the written path on success, or an error message.
        """
        from langsmith.sandbox import SandboxClientError  # noqa: PLC0415

        try:
            self._sandbox.write(file_path, content.encode("utf-8"))
            return WriteResult(path=file_path, files_update=None)
        except SandboxClientError as e:
            return WriteResult(error=f"Failed to write file '{file_path}': {e}")

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the LangSmith sandbox.

        Supports partial success -- individual downloads may fail without
        affecting others.

        Args:
            paths: List of file paths to download.

        Returns:
            List of `FileDownloadResponse` objects, one per input path.

                Response order matches input order.
        """
        from langsmith.sandbox import ResourceNotFoundError, SandboxClientError  # noqa: PLC0415

        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(FileDownloadResponse(path=path, content=None, error="invalid_path"))
                continue
            try:
                content = self._sandbox.read(path)
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
            except ResourceNotFoundError:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
            except SandboxClientError as e:
                msg = str(e).lower()
                error = "is_directory" if "is a directory" in msg else "file_not_found"
                responses.append(FileDownloadResponse(path=path, content=None, error=error))
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the LangSmith sandbox.

        Supports partial success -- individual uploads may fail without
        affecting others.

        Args:
            files: List of `(path, content)` tuples to upload.

        Returns:
            List of `FileUploadResponse` objects, one per input file.

                Response order matches input order.
        """
        from langsmith.sandbox import SandboxClientError  # noqa: PLC0415

        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            try:
                self._sandbox.write(path, content)
                responses.append(FileUploadResponse(path=path, error=None))
            except SandboxClientError as e:
                logger.debug("Failed to upload %s: %s", path, e)
                responses.append(FileUploadResponse(path=path, error="permission_denied"))
        return responses
