"""Harbor-specific tools for DeepAgents to interact with the environment."""

import asyncio

from harbor.environments.base import BaseEnvironment

from deepagents.backends.protocol import ExecuteResponse
from deepagents.backends.sandbox import BaseSandbox


class HarborSandbox(BaseSandbox):
    def __init__(self, environment: BaseEnvironment) -> None:
        """Initialize HarborSandbox with the given environment."""
        self.environment = environment

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """Execute a bash command in the task environment."""
        # TODO: This is a temporary hack to run async code from a sync function.
        # We need to add async support in deepagents.
        result = asyncio.run(self.environment.exec(command))
        output = (result.stdout or "") + "\n stderr: " + (result.stderr or "")
        return ExecuteResponse(
            output=output,
            exit_code=result.return_code,
        )

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""
        return self.environment.session_id
