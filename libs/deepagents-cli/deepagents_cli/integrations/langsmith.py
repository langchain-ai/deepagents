"""LangSmith sandbox backend implementation."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    from langsmith.sandbox import Sandbox, SandboxClient

from deepagents_cli.config import console


class LangSmithBackend(BaseSandbox):
    """LangSmith backend implementation conforming to SandboxBackendProtocol.

    This implementation inherits all file operation methods from BaseSandbox
    and only implements the execute() method using LangSmith's API.
    """

    def __init__(self, sandbox: Sandbox) -> None:
        """Initialize the LangSmithBackend with a sandbox instance.

        Args:
            sandbox: LangSmith Sandbox instance
        """
        self._sandbox = sandbox
        self._timeout: int = 30 * 60  # 30 mins default

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""
        return self._sandbox.name

    def execute(self, command: str) -> ExecuteResponse:
        """Execute a command in the sandbox and return ExecuteResponse.

        Args:
            command: Full shell command string to execute.

        Returns:
            ExecuteResponse with combined output, exit code, and truncation flag.
        """
        result = self._sandbox.run(command, timeout=self._timeout)

        # Combine stdout and stderr (matching other backends' approach)
        output = result.stdout or ""
        if result.stderr:
            output += "\n" + result.stderr if output else result.stderr

        return ExecuteResponse(
            output=output,
            exit_code=result.exit_code,
            truncated=False,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the LangSmith sandbox.

        Supports partial success - individual downloads may fail without
        affecting others.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.

        Note:
            Error handling with standardized FileOperationError codes is not yet
            implemented in existing backends. Currently only the happy path is
            implemented. See existing backends for reference.
        """
        responses: list[FileDownloadResponse] = []

        for path in paths:
            # Use LangSmith's native file read API (returns bytes)
            content = self._sandbox.read(path)
            responses.append(FileDownloadResponse(path=path, content=content, error=None))

        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the LangSmith sandbox.

        Supports partial success - individual uploads may fail without
        affecting others.

        Args:
            files: List of (path, content) tuples to upload.

        Returns:
            List of FileUploadResponse objects, one per input file.

        Note:
            Error handling with standardized FileOperationError codes is not yet
            implemented in existing backends. Currently only the happy path is
            implemented. See existing backends for reference.
        """
        responses: list[FileUploadResponse] = []

        for path, content in files:
            # Use LangSmith's native file write API
            self._sandbox.write(path, content)
            responses.append(FileUploadResponse(path=path, error=None))

        return responses


# Default template configuration
DEFAULT_TEMPLATE_NAME = "deepagents-cli"
DEFAULT_TEMPLATE_IMAGE = "ubuntu:24.04"


def ensure_template(client: SandboxClient, template_name: str = DEFAULT_TEMPLATE_NAME) -> None:
    """Ensure template exists, creating it if needed.

    Args:
        client: LangSmith SandboxClient instance
        template_name: Name of the template to ensure exists

    Raises:
        RuntimeError: If template check or creation fails
    """
    from langsmith.sandbox import ResourceNotFoundError

    try:
        client.get_template(template_name)
    except ResourceNotFoundError as e:
        if e.resource_type != "template":
            raise
        console.print(f"[dim]Creating template '{template_name}'...[/dim]")
        try:
            client.create_template(name=template_name, image=DEFAULT_TEMPLATE_IMAGE)
        except Exception as e:
            msg = f"Failed to create template '{template_name}': {e}"
            raise RuntimeError(msg) from e
    except Exception as e:
        msg = f"Failed to check template '{template_name}': {e}"
        raise RuntimeError(msg) from e


def create_sandbox_instance(
    client: SandboxClient, template_name: str = DEFAULT_TEMPLATE_NAME
) -> Sandbox:
    """Create a new sandbox and verify it's ready.

    Args:
        client: LangSmith SandboxClient instance
        template_name: Name of the template to use

    Returns:
        Ready Sandbox instance

    Raises:
        RuntimeError: If sandbox creation or readiness check fails
    """
    console.print("[dim]Creating sandbox...[/dim]")
    try:
        sb = client.create_sandbox(template_name=template_name, timeout=180)
    except Exception as e:
        msg = f"Failed to create sandbox from template '{template_name}': {e}"
        raise RuntimeError(msg) from e

    # Verify sandbox is ready
    readiness_error: Exception | None = None
    try:
        result = sb.run("echo ready", timeout=5)
        if result.exit_code != 0:
            readiness_error = RuntimeError("Sandbox readiness check failed")
    except Exception as e:
        readiness_error = e

    if readiness_error:
        with contextlib.suppress(Exception):
            client.delete_sandbox(sb.name)
        msg = f"LangSmith sandbox failed readiness check: {readiness_error}"
        raise RuntimeError(msg) from readiness_error

    console.print(f"[green]✓ LangSmith sandbox ready: {sb.name}[/green]")
    return sb
