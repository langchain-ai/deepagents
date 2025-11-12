"""Modal sandbox backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.backends.protocol import ExecuteResponse
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
