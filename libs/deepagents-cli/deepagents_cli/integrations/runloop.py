"""BackendProtocol implementation for Runloop."""

try:
    import runloop_api_client
except ImportError:
    msg = (
        "runloop_api_client package is required for RunloopBackend. "
        "Install with `pip install runloop_api_client`."
    )
    raise ImportError(msg)

import os
import time
from typing import Any

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    SandboxBackendProtocol,
)
from deepagents.backends.sandbox import BaseSandbox, SandboxListResponse, SandboxProvider
from runloop_api_client import Runloop


class RunloopBackend(BaseSandbox):
    """Backend that operates on files in a Runloop devbox.

    This implementation uses the Runloop API client to execute commands
    and manipulate files within a remote devbox environment.
    """

    def __init__(
        self,
        devbox_id: str,
        client: Runloop | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize Runloop protocol.

        Args:
            devbox_id: ID of the Runloop devbox to operate on.
            client: Optional existing Runloop client instance
            api_key: Optional API key for creating a new client
                         (defaults to RUNLOOP_API_KEY environment variable)
        """
        if client and api_key:
            msg = "Provide either client or bearer_token, not both."
            raise ValueError(msg)

        if client is None:
            api_key = api_key or os.environ.get("RUNLOOP_API_KEY", None)
            if api_key is None:
                msg = "Either client or bearer_token must be provided."
                raise ValueError(msg)
            client = Runloop(bearer_token=api_key)

        self._client = client
        self._devbox_id = devbox_id
        self._timeout = 30 * 60

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""
        return self._devbox_id

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """Execute a command in the devbox and return ExecuteResponse.

        Args:
            command: Full shell command string to execute.
            timeout: Maximum execution time in seconds (default: 30 minutes).

        Returns:
            ExecuteResponse with combined output, exit code, optional signal, and truncation flag.
        """
        result = self._client.devboxes.execute_and_await_completion(
            devbox_id=self._devbox_id,
            command=command,
            timeout=self._timeout,
        )
        # Combine stdout and stderr
        output = result.stdout or ""
        if result.stderr:
            output += "\n" + result.stderr if output else result.stderr

        return ExecuteResponse(
            output=output,
            exit_code=result.exit_status,
            truncated=False,  # Runloop doesn't provide truncation info
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the Runloop devbox.

        Downloads files individually using the Runloop API. Returns a list of
        FileDownloadResponse objects preserving order and reporting per-file
        errors rather than raising exceptions.

        TODO: Implement proper error handling with standardized FileOperationError codes.
        Currently only implements happy path.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            # devboxes.download_file returns a BinaryAPIResponse which exposes .read()
            resp = self._client.devboxes.download_file(self._devbox_id, path=path)
            content = resp.read()
            responses.append(FileDownloadResponse(path=path, content=content, error=None))

        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the Runloop devbox.

        Uploads files individually using the Runloop API. Returns a list of
        FileUploadResponse objects preserving order and reporting per-file
        errors rather than raising exceptions.

        TODO: Implement proper error handling with standardized FileOperationError codes.
        Currently only implements happy path.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            # The Runloop client expects 'file' as bytes or a file-like object
            self._client.devboxes.upload_file(self._devbox_id, path=path, file=content)
            responses.append(FileUploadResponse(path=path, error=None))

        return responses


class RunloopProvider(SandboxProvider[dict[str, Any]]):
    """Runloop sandbox provider implementation.

    Manages Runloop devbox lifecycle using the Runloop SDK.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize Runloop provider.

        Args:
            api_key: Runloop API key (defaults to RUNLOOP_API_KEY env var)
        """
        self._api_key = api_key or os.environ.get("RUNLOOP_API_KEY")
        if not self._api_key:
            msg = "RUNLOOP_API_KEY environment variable not set"
            raise ValueError(msg)
        self._client = Runloop(bearer_token=self._api_key)

    def list(
        self,
        *,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> SandboxListResponse[dict[str, Any]]:
        """List available Runloop devboxes.

        Raises:
            NotImplementedError: Runloop SDK doesn't expose a list API yet.
        """
        msg = "Listing with Runloop SDK not yet implemented"
        raise NotImplementedError(msg)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Get existing or create new Runloop devbox.

        Args:
            sandbox_id: Existing devbox ID to connect to (if None, creates new)
            timeout: Timeout in seconds for devbox startup (default: 180)
            **kwargs: Additional Runloop-specific parameters

        Returns:
            RunloopBackend instance

        Raises:
            ImportError: Runloop SDK not installed
            RuntimeError: Devbox startup failed
        """
        # Import console here to avoid circular import
        from deepagents_cli.config import console

        console.print("[yellow]Starting Runloop devbox...[/yellow]")

        if sandbox_id:
            devbox = self._client.devboxes.retrieve(id=sandbox_id)
        else:
            devbox = self._client.devboxes.create()
            sandbox_id = devbox.id

            # Poll until running
            for _ in range(timeout // 2):
                status = self._client.devboxes.retrieve(id=devbox.id)
                if status.status == "running":
                    break
                time.sleep(2)
            else:
                self._client.devboxes.shutdown(id=devbox.id)
                msg = f"Devbox failed to start within {timeout} seconds"
                raise RuntimeError(msg)

        console.print(f"[green]✓ Runloop devbox ready: {sandbox_id}[/green]")
        return RunloopBackend(devbox_id=devbox.id, client=self._client)

    def delete(self, sandbox_id: str, **kwargs: Any) -> None:
        """Delete a Runloop devbox.

        Args:
            sandbox_id: Devbox ID to delete
            **kwargs: Additional parameters
        """
        # Import console here to avoid circular import
        from deepagents_cli.config import console

        console.print(f"[dim]Shutting down Runloop devbox {sandbox_id}...[/dim]")
        self._client.devboxes.shutdown(id=sandbox_id)
        console.print(f"[dim]✓ Runloop devbox {sandbox_id} terminated[/dim]")

        return responses
