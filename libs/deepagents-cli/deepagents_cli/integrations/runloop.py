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

from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox
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
