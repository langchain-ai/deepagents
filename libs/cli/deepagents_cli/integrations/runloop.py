"""BackendProtocol implementation for Runloop."""

import importlib.util

if importlib.util.find_spec("runloop_api_client") is None:
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
from deepagents.backends.sandbox import BaseSandbox
from runloop_api_client import Runloop

from deepagents_cli.integrations.sandbox_provider import (
    SandboxNotFoundError,
    SandboxProvider,
)


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

        Raises:
            ValueError: If both client and api_key are provided, or if neither
                is available.
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
        self._default_timeout = 30 * 60

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""
        return self._devbox_id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a command in the devbox and return ExecuteResponse.

        Args:
            command: Full shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If None, uses the backend's default timeout.

        Returns:
            ExecuteResponse with combined output, exit code, and truncation flag.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        result = self._client.devboxes.execute_and_await_completion(
            devbox_id=self._devbox_id,
            command=command,
            timeout=effective_timeout,
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

        TODO: Implement proper error handling with standardized
        FileOperationError codes. Currently only implements happy path.

        Returns:
            List of FileDownloadResponse objects preserving input order.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            # devboxes.download_file returns a BinaryAPIResponse which exposes .read()
            resp = self._client.devboxes.download_file(self._devbox_id, path=path)
            content = resp.read()
            responses.append(
                FileDownloadResponse(path=path, content=content, error=None)
            )

        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the Runloop devbox.

        Uploads files individually using the Runloop API. Returns a list of
        FileUploadResponse objects preserving order and reporting per-file
        errors rather than raising exceptions.

        TODO: Implement proper error handling with standardized
            FileOperationError codes. Currently only implements happy path.

        Returns:
            List of FileUploadResponse objects preserving input order.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            # The Runloop client expects 'file' as bytes or a file-like object
            self._client.devboxes.upload_file(self._devbox_id, path=path, file=content)
            responses.append(FileUploadResponse(path=path, error=None))

        return responses


class RunloopProvider(SandboxProvider):
    """Runloop sandbox provider implementation.

    Manages Runloop devbox lifecycle using the Runloop SDK.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize Runloop provider.

        Args:
            api_key: Runloop API key (defaults to RUNLOOP_API_KEY env var)

        Raises:
            ValueError: If RUNLOOP_API_KEY environment variable not set
        """
        self._api_key = api_key or os.environ.get("RUNLOOP_API_KEY")
        if not self._api_key:
            msg = "RUNLOOP_API_KEY environment variable not set"
            raise ValueError(msg)
        self._client = Runloop(bearer_token=self._api_key)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        **kwargs: Any,  # noqa: ARG002  # Required by SandboxFactory interface
    ) -> SandboxBackendProtocol:
        """Get existing or create new Runloop devbox.

        Args:
            sandbox_id: Existing devbox ID to connect to (if None, creates new)
            timeout: Timeout in seconds for devbox startup (default: 180)
            **kwargs: Additional Runloop-specific parameters

        Returns:
            RunloopBackend instance

        Raises:
            RuntimeError: Devbox startup failed
            SandboxNotFoundError: If sandbox_id is provided but does not exist
        """
        if sandbox_id:
            try:
                devbox = self._client.devboxes.retrieve(id=sandbox_id)
            except KeyError as e:
                raise SandboxNotFoundError(sandbox_id) from e
        else:
            devbox = self._client.devboxes.create()

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

        return RunloopBackend(devbox_id=devbox.id, client=self._client)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002  # Required by SandboxFactory interface
        """Delete a Runloop devbox.

        Args:
            sandbox_id: Devbox ID to delete
            **kwargs: Additional parameters
        """
        self._client.devboxes.shutdown(id=sandbox_id)
