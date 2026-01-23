"""Koyeb sandbox backend implementation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    from koyeb import AsyncSandbox


class KoyebBackend(BaseSandbox):
    """Koyeb backend implementation conforming to SandboxBackendProtocol.

    This implementation uses Koyeb's async API natively and provides both sync
    and async methods. The async methods use Koyeb's native async API, while
    sync methods wrap the async calls with asyncio.run().
    """

    def __init__(self, sandbox: AsyncSandbox) -> None:
        """Initialize the KoyebBackend with a Koyeb AsyncSandbox instance.

        Args:
            sandbox: Koyeb AsyncSandbox instance
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

        Sync wrapper that calls the async implementation.

        Args:
            command: Full shell command string to execute.

        Returns:
            ExecuteResponse with combined output, exit code, and truncation flag.
        """
        return asyncio.run(self.aexecute(command))

    async def aexecute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """Execute a command in the sandbox and return ExecuteResponse (async).

        Native async implementation using Koyeb's async API.

        Args:
            command: Full shell command string to execute.

        Returns:
            ExecuteResponse with combined output, exit code, and truncation flag.
        """
        # Execute command using Koyeb's async exec API
        result = await self._sandbox.exec(command, timeout=self._timeout)

        # Koyeb's CommandResult combines stdout and stderr via the output property
        return ExecuteResponse(
            output=result.output,
            exit_code=result.exit_code,
            truncated=False,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the Koyeb sandbox.

        Sync wrapper that calls the async implementation.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.
            Response order matches input order.
        """
        return asyncio.run(self.adownload_files(paths))

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the Koyeb sandbox (async).

        Native async implementation using Koyeb's async API.
        Supports partial success - individual downloads may fail without
        affecting others.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.
            Response order matches input order. If a download fails, the
            response will contain an error with a standardized error code.

        Error codes:
            - file_not_found: File does not exist
            - permission_denied: Insufficient permissions (not currently raised by Koyeb)
            - is_directory: Path is a directory, not a file (not currently raised by Koyeb)
            - invalid_path: Invalid path format (not currently raised by Koyeb)
        """
        from koyeb.sandbox.filesystem import (
            SandboxFileNotFoundError,
            SandboxFilesystemError,
        )

        responses = []
        for path in paths:
            try:
                # Try base64 first (handles binary files stored with base64)
                # If that fails, fall back to UTF-8 for text files
                file_info = None
                try:
                    file_info = await self._sandbox.filesystem.read_file(path, encoding="base64")
                except SandboxFilesystemError as e:
                    error_msg = str(e).lower()
                    if "base64" in error_msg or "invalid" in error_msg:
                        file_info = await self._sandbox.filesystem.read_file(path, encoding="utf-8")
                    else:
                        raise

                # Process the file content
                if file_info:
                    if isinstance(file_info.content, (bytes, bytearray, memoryview)):
                        content = bytes(file_info.content)
                    elif isinstance(file_info.content, str):
                        content = file_info.content.encode("utf-8")
                    else:
                        content = bytes(file_info.content)
                    responses.append(FileDownloadResponse(path=path, content=content, error=None))
            except SandboxFileNotFoundError:
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error="file_not_found",
                    )
                )
            except SandboxFilesystemError:
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error="invalid_path",
                    )
                )
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the Koyeb sandbox.

        Sync wrapper that calls the async implementation.

        Args:
            files: List of (path, content) tuples to upload.

        Returns:
            List of FileUploadResponse objects, one per input file.
            Response order matches input order.
        """
        return asyncio.run(self.aupload_files(files))

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the Koyeb sandbox (async).

        Native async implementation using Koyeb's async API.
        Supports partial success - individual uploads may fail without
        affecting others.

        Args:
            files: List of (path, content) tuples to upload.

        Returns:
            List of FileUploadResponse objects, one per input file.
            Response order matches input order. If an upload fails, the
            response will contain an error with a standardized error code.

        Error codes:
            - file_not_found: Parent directory does not exist
              (not currently raised by Koyeb - auto-creates parent dirs)
            - permission_denied: Insufficient permissions
              (not currently raised by Koyeb)
            - is_directory: Path is a directory, not a file
              (not currently raised by Koyeb)
            - invalid_path: Invalid path format (catch-all for errors)
        """
        from koyeb.sandbox.filesystem import (
            SandboxFileExistsError,
            SandboxFileNotFoundError,
            SandboxFilesystemError,
        )

        responses = []
        for path, content in files:
            try:
                # Try to decode as UTF-8 text; if successful, write as text
                # Otherwise, use base64 encoding for binary content
                try:
                    text_content = content.decode("utf-8")
                    await self._sandbox.filesystem.write_file(path, text_content, encoding="utf-8")
                except UnicodeDecodeError:
                    await self._sandbox.filesystem.write_file(path, content, encoding="base64")
                responses.append(FileUploadResponse(path=path, error=None))
            except SandboxFileNotFoundError:
                responses.append(
                    FileUploadResponse(
                        path=path,
                        error="file_not_found",
                    )
                )
            except SandboxFileExistsError:
                responses.append(
                    FileUploadResponse(
                        path=path,
                        error="invalid_path",
                    )
                )
            except SandboxFilesystemError:
                responses.append(
                    FileUploadResponse(
                        path=path,
                        error="invalid_path",
                    )
                )
        return responses
