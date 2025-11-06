"""Daytona sandbox backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileInfo,
)

from deepagents_cli.integrations.base_sandbox import BaseSandbox

if TYPE_CHECKING:
    from daytona import Sandbox


class DaytonaBackend(BaseSandbox):
    """Daytona backend implementation conforming to SandboxBackendProtocol.

    This implementation inherits all file operation methods from BaseSandbox
    and only implements the execute() method using Daytona's API.
    """

    def __init__(self, sandbox: Sandbox) -> None:
        """Initialize the DaytonaBackend with a Daytona sandbox client."""
        self._sandbox = sandbox

    def execute(
        self,
        command: str,
        *,
        timeout: int = 30 * 60,
    ) -> ExecuteResponse:
        """Execute a command in the sandbox and return ExecuteResponse.

        Args:
            command: Full shell command string to execute.
            timeout: Maximum execution time in seconds (default: 30 minutes).

        Returns:
            ExecuteResponse with combined output, exit code, optional signal, and truncation flag.
        """
        result = self._sandbox.process.exec(command, timeout=timeout)

        return ExecuteResponse(
            output=result.result,  # Daytona combines stdout/stderr
            exit_code=result.exit_code,
            truncated=False,
        )
