"""Example shell middleware that has ZERO additional abstractions."""

import logging
import os
from typing import Any, NotRequired

from daytona import CreateSandboxFromSnapshotParams, Daytona, DaytonaConfig
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

logger = logging.getLogger(__name__)


class SandboxState(AgentState):
    """Keep track of the Daytona sandbox ID."""

    sandbox_id: NotRequired[str]


class DaytonaSandboxMiddleware(AgentMiddleware[SandboxState]):
    """Middleware that manages a Daytona sandbox for code execution."""

    state_schema = SandboxState

    def __init__(
        self,
        *,
        api_key: str | None = None,
        env: dict[str, str] | None = None,
        auto_stop_minutes: int | None = None,
        auto_delete_minutes: int | None = None,
        terminate_on_complete: bool = True,
    ):
        """Initialize the sandbox middleware.

        Args:
            api_key: Daytona API key. If not provided, uses DAYTONA_API_KEY env var.
            env: Dictionary of environment variables to set in the sandbox.
            auto_stop_minutes: Minutes of inactivity before sandbox auto-stops. Defaults to 15.
            auto_delete_minutes: Minutes after stopping before sandbox is deleted. Defaults to 0
                                (delete immediately on stop).
        """
        super().__init__()

        # Initialize Daytona client
        api_key = api_key or os.environ.get("DAYTONA_API_KEY")
        if api_key is None:
            raise ValueError("Either api_key parameter or DAYTONA_API_KEY environment variable must be provided.")

        config = DaytonaConfig(api_key=api_key)
        self.daytona_client = Daytona(config)
        self.auto_stop_interval = auto_stop_minutes
        self.auto_delete_interval = auto_delete_minutes
        self.terminate_on_complete = terminate_on_complete
        self.env = env or {}

        @tool
        def bash(command: str, runtime: ToolRuntime) -> Command:
            """Execute a bash command in the isolated sandbox environment.

            Use this tool to run shell commands, execute scripts, install dependencies,
            or perform any other bash operations needed to verify the documentation.

            Args:
                command: The bash command to execute (e.g., "python script.py", "pip install requests")

            Returns:
                The output from the command execution, including stdout and stderr.
            """
            sandbox_id = runtime.state.get("sandbox_id")
            created = False

            # Get or create sandbox
            if sandbox_id:
                sandbox = self.daytona_client.get(sandbox_id)
            else:
                # Create new sandbox with auto-stop and auto-delete configured
                sandbox = self.daytona_client.create(
                    params=CreateSandboxFromSnapshotParams(auto_stop_interval=self.auto_stop_interval, auto_delete_interval=self.auto_delete_interval)
                )
                created = True

            # Execute command directly on sandbox
            execute_response = sandbox.process.exec(command, env=self.env)
            content = f"Output:\n{execute_response.result}\nExit Code: {execute_response.exit_code}"

            update = {
                "messages": [
                    ToolMessage(
                        content=content,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }

            if created:
                update["sandbox_id"] = sandbox.id

            return Command(update=update)

        self.tools = [bash]

    def after_agent(self, state: SandboxState, runtime: Runtime) -> dict[str, Any] | None:
        """Clean up the sandbox after the agent completes."""
        sandbox_id = state.get("sandbox_id")

        if sandbox_id and self.terminate_on_complete:
            try:
                sandbox = self.daytona_client.get(sandbox_id)
                self.daytona_client.delete(sandbox)
            except Exception as e:
                logger.info("Error deleting sandbox %s: %s", sandbox_id, str(e))
        return None
