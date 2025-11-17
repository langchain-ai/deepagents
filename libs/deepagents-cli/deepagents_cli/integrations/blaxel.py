"""Blaxel sandbox backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.backends.protocol import ExecuteResponse
from deepagents.backends.sandbox import BaseSandbox
from blaxel.core.sandbox.client.models import ProcessRequest

if TYPE_CHECKING:
    from blaxel.core import SyncSandboxInstance


class BlaxelBackend(BaseSandbox):
    """Blaxel backend implementation conforming to SandboxBackendProtocol.

    This implementation inherits all file operation methods from BaseSandbox
    and only implements the execute() method using Blaxel's API.
    """

    def __init__(self, sandbox: SyncSandboxInstance) -> None:
        """Initialize the BlaxelBackend with a Blaxel sandbox client.

        Args:
            sandbox: Blaxel sandbox instance
        """
        self._sandbox = sandbox
        self._timeout: int = 30 * 60  # 30 mins

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""
        return self._sandbox.metadata.name or ""

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """Execute a command in the sandbox and return ExecuteResponse.

        Args:
            command: Full shell command string to execute.

        Returns:
            ExecuteResponse with combined output, exit code, optional signal, and truncation flag.
        """
        exec_request = ProcessRequest(
            command=command,
            wait_for_completion=True,
            timeout=self._timeout,
        )
        result = self._sandbox.process.exec(exec_request)

        return ExecuteResponse(
            output=result.logs,  # Blaxel combines stdout/stderr
            exit_code=result.exit_code,
            truncated=False,
        )
