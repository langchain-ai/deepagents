"""BackendProtocol implementation for Runloop."""

try:
    import runloop_api_client
except ImportError:
    raise ImportError(
        "runloop_api_client package is required for RunloopBackend. "
        "Install with `pip install runloop_api_client`."
    )

import os

from runloop_api_client import Runloop

from deepagents.backends.protocol import ExecuteResponse
from deepagents.backends.sandbox import BaseSandbox


class RunloopBackend(BaseSandbox):
    """Backend that operates on files in a Runloop devbox.

    This implementation uses the Runloop API client to execute commands
    and manipulate files within a remote devbox environment.
    """

    def __init__(
        self,
        devbox_id: str,
        client: Runloop | None = None,
        bearer_token: str | None = None,
    ) -> None:
        """Initialize Runloop protocol.

        Args:
            devbox_id: ID of the Runloop devbox to operate on.
            client: Optional existing Runloop client instance
            bearer_token: Optional API key for creating a new client
                         (defaults to RUNLOOP_API_KEY environment variable)
        """
        if client and bearer_token:
            raise ValueError("Provide either client or bearer_token, not both.")

        if client is None:
            bearer_token = bearer_token or os.environ.get("RUNLOOP_API_KEY", None)
            if bearer_token is None:
                raise ValueError("Either client or bearer_token must be provided.")
            client = Runloop(bearer_token=bearer_token)

        self._client = client
        self._devbox_id = devbox_id

    def execute(
        self,
        command: str,
        *,
        timeout: int = 30 * 60,
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
            timeout=timeout,
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
