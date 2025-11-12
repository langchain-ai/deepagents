"""Modal sandbox backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    import modal


class ModalBackend(BaseSandbox):
    """Modal backend implementation conforming to SandboxBackendProtocol.

    This implementation inherits all file operation methods from BaseSandbox
    and only implements the execute() method using Modal's API.
    """

    def __init__(self, sandbox: modal.Sandbox) -> None:
        """Initialize the ModalBackend with a Modal sandbox instance.

        Args:
            sandbox: Active Modal Sandbox instance
        """
        self._sandbox = sandbox
        self._timeout = 30 * 60

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""
        return self._sandbox.object_id

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """Execute a command in the sandbox and return ExecuteResponse.

        Args:
            command: Full shell command string to execute.

        Returns:
            ExecuteResponse with combined output, exit code, and truncation flag.
        """
        # Execute command using Modal's exec API
        process = self._sandbox.exec("bash", "-c", command, timeout=self._timeout)

        # Wait for process to complete
        process.wait()

        # Read stdout and stderr
        stdout = process.stdout.read()
        stderr = process.stderr.read()

        # Combine stdout and stderr (matching Runloop's approach)
        output = stdout or ""
        if stderr:
            output += "\n" + stderr if output else stderr

        return ExecuteResponse(
            output=output,
            exit_code=process.returncode,
            truncated=False,  # Modal doesn't provide truncation info
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the Modal sandbox.

        Supports partial success - individual downloads may fail without
        affecting others.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.
            Response order matches input order.
        """
        # This implementation relies on the Modal sandbox file API.
        # https://modal.com/doc/guide/sandbox-files
        # The API is currently in alpha and is not recommended for production use.
        # We're OK using it here as it's targeting the CLI application.
        responses = []
        for path in paths:
            try:
                with self._sandbox.open(path, "rb") as f:
                    content = f.read()
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
            except FileNotFoundError:
                responses.append(
                    FileDownloadResponse(path=path, content=None, error=f"File not found: {path}")
                )
            except PermissionError:
                responses.append(
                    FileDownloadResponse(
                        path=path, content=None, error=f"Permission denied: {path}"
                    )
                )
            except Exception as e:
                responses.append(
                    FileDownloadResponse(
                        path=path, content=None, error=f"Error downloading {path}: {e}"
                    )
                )
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the Modal sandbox.

        Supports partial success - individual uploads may fail without
        affecting others.

        Args:
            files: List of (path, content) tuples to upload.

        Returns:
            List of FileUploadResponse objects, one per input file.
            Response order matches input order.
        """
        # This implementation relies on the Modal sandbox file API.
        # https://modal.com/doc/guide/sandbox-files
        # The API is currently in alpha and is not recommended for production use.
        # We're OK using it here as it's targeting the CLI application.
        responses = []
        for path, content in files:
            try:
                with self._sandbox.open(path, "wb") as f:
                    f.write(content)
                responses.append(FileUploadResponse(path=path, error=None))
            except FileNotFoundError:
                responses.append(
                    FileUploadResponse(
                        path=path,
                        error=f"Directory not found for path: {path}",
                    )
                )
            except PermissionError:
                responses.append(FileUploadResponse(path=path, error=f"Permission denied: {path}"))
            except Exception as e:
                responses.append(
                    FileUploadResponse(path=path, error=f"Error uploading {path}: {e}")
                )
        return responses
