"""Harbor-specific tools for DeepAgents to interact with the environment."""

import asyncio
import threading

from harbor.environments.base import BaseEnvironment

from deepagents.backends.protocol import ExecuteResponse
from deepagents.backends.sandbox import BaseSandbox

_loop = asyncio.new_event_loop()


def run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


_thread = threading.Thread(
    target=run_loop, args=(_loop,), daemon=True, name="HarborSandboxLoop"
)
_thread.start()


class HarborSandbox(BaseSandbox):
    def __init__(self, environment: BaseEnvironment) -> None:
        """Initialize HarborSandbox with the given environment."""
        self.environment = environment

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """Execute a bash command in the task environment."""
        coro = self.environment.exec(command)

        # Submit the async task to the background loop and wait for the result
        future = asyncio.run_coroutine_threadsafe(coro, _loop)
        result = future.result()
        output = (result.stdout or "") + "\n stderr: " + (result.stderr or "")
        return ExecuteResponse(
            output=output,
            exit_code=result.return_code,
        )

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""
        return self.environment.session_id

