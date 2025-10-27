"""Generalized factory for creating file system or shell tools based on configuration.

This module is meant to support interacting with ephemeral containers/sandboxes.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.typing import ContextT

from deepagents.backends.sandbox import SandboxProvider


class SandboxState(AgentState):
    """State schema for Daytona sandbox middleware."""

    sandbox_id: NotRequired[str]


def _get_tools(sandbox_provider: SandboxProvider) -> list[BaseTool]:
    """Generate tools for interacting with the sandbox.

    Args:
        sandbox_provider: The sandbox provider to use.

    Returns:
        List of tools for sandbox interaction.
    """
    capabilities = sandbox_provider.get_capabilities()
    tools = []

    fs_capabilities = capabilities["fs"]

    if fs_capabilities["can_read"]:

        @tool
        def read(
            file_path: str,
            runtime: ToolRuntime,
            offset: int = 0,
            limit: int = 2000,
        ) -> Command:
            """Read a file from the sandbox filesystem."""
            sandbox_id = runtime.state.get("sandbox_id")
            # If None then, the sandbox will be created now
            created = sandbox_id is None
            sandbox = sandbox_provider.get_or_create(sandbox_id)
            result = sandbox.fs.read(file_path, offset, limit)
            update = {
                "messages": [
                    ToolMessage(
                        content=result,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }

            if created:
                update["sandbox_id"] = sandbox.id
            return Command(update=update)

        tools.append(read)

    if fs_capabilities["can_edit"]:

        @tool
        def edit(
            file_path: str,
            old_string: str,
            new_string: str,
            runtime: ToolRuntime,
            replace_all: bool = False,
        ) -> Command:
            """Edit a file in the sandbox filesystem."""
            sandbox_id = runtime.state.get("sandbox_id")
            # If None then, the sandbox will be created now
            created = sandbox_id is None
            sandbox = sandbox_provider.get_or_create(sandbox_id)
            result = sandbox.fs.edit(file_path, old_string, new_string, replace_all)
            update = {
                "messages": [
                    ToolMessage(
                        content=result,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }

            if created:
                update["sandbox_id"] = sandbox.id
            return Command(update=update)

        tools.append(edit)

    if fs_capabilities["can_list_files"]:

        @tool
        def ls(
            runtime: ToolRuntime,
            prefix: str | None = None,
        ) -> Command:
            """List files in the sandbox filesystem."""
            sandbox_id = runtime.state.get("sandbox_id")
            # If None then, the sandbox will be created now
            created = sandbox_id is None
            sandbox = sandbox_provider.get_or_create(sandbox_id)
            result = sandbox.fs.ls(prefix)
            update = {
                "messages": [
                    ToolMessage(
                        content=str(result),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }

            if created:
                update["sandbox_id"] = sandbox.id
            return Command(update=update)

        tools.append(ls)

    if fs_capabilities["can_delete"]:

        @tool
        def delete(
            file_path: str,
            runtime: ToolRuntime,
        ) -> Command:
            """Delete a file from the sandbox filesystem."""
            sandbox_id = runtime.state.get("sandbox_id")
            # If None then, the sandbox will be created now
            created = sandbox_id is None
            sandbox = sandbox_provider.get_or_create(sandbox_id)
            sandbox.fs.delete(file_path)
            result = f"Successfully deleted {file_path}"
            update = {
                "messages": [
                    ToolMessage(
                        content=result,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }

            if created:
                update["sandbox_id"] = sandbox.id
            return Command(update=update)

        tools.append(delete)

    if fs_capabilities["can_grep"]:

        @tool
        def grep(
            pattern: str,
            runtime: ToolRuntime,
            path: str = "/",
            include: str | None = None,
            output_mode: str = "files_with_matches",
        ) -> Command:
            """Search for a pattern in files in the sandbox filesystem."""
            sandbox_id = runtime.state.get("sandbox_id")
            # If None then, the sandbox will be created now
            created = sandbox_id is None
            sandbox = sandbox_provider.get_or_create(sandbox_id)
            result = sandbox.fs.grep(pattern, path, include, output_mode)
            update = {
                "messages": [
                    ToolMessage(
                        content=result,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }

            if created:
                update["sandbox_id"] = sandbox.id
            return Command(update=update)

        tools.append(grep)

    if fs_capabilities["can_glob"]:

        @tool
        def glob(
            pattern: str,
            runtime: ToolRuntime,
            path: str = "/",
        ) -> Command:
            """Find files matching a glob pattern in the sandbox filesystem."""
            sandbox_id = runtime.state.get("sandbox_id")
            # If None then, the sandbox will be created now
            created = sandbox_id is None
            sandbox = sandbox_provider.get_or_create(sandbox_id)
            result = sandbox.fs.glob(pattern, path)
            update = {
                "messages": [
                    ToolMessage(
                        content=str(result),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }

            if created:
                update["sandbox_id"] = sandbox.id
            return Command(update=update)

        tools.append(glob)

    return tools


class GeneralizedFilesystemMiddleware(AgentMiddleware):
    state_schema = SandboxState

    def __init__(
        self, sandbox_provider: SandboxProvider, *, terminate_on_complete: bool = True, create_on: Literal["start", "usage"] = "start"
    ) -> None:
        """Initialize the Daytona sandbox middleware."""
        self.sandbox_provider = sandbox_provider
        self.terminate_on_complete = terminate_on_complete
        self.create_on = create_on

        self.tools = _get_tools(sandbox_provider)

    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        if "sandbox_id" not in state and self.create_on == "start":
            sandbox = self.sandbox_provider.get_or_create()
            return {"sandbox_id": sandbox.id}
        return {}

    def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Terminate the sandbox after agent completion if configured to do so."""
        if sandbox_id := state.get("sandbox_id") and self.terminate_on_complete:
            self.sandbox_provider.delete(sandbox_id)
        return None


class GeneralizedShellMiddleware(AgentMiddleware):
    state_schema = SandboxState

    def __init__(
        self,
        sandbox_provider: SandboxProvider,
        *,
        env: dict[str, str] | None = None,
        terminate_on_complete: bool = True,
    ) -> None:
        """Initialize the Daytona shell middleware."""
        self.sandbox_provider = sandbox_provider
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
            # If sandbox_id was not found, create a new sandbox
            created = sandbox_id is None
            sandbox = self.sandbox_provider.get_or_create(sandbox_id)
            # Execute command directly on sandbox
            # TODO(Eugene): Add env support
            execute_response = sandbox.process.execute(command)
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

    def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Clean up the sandbox after the agent completes."""
        sandbox_id = state.get("sandbox_id")

        if sandbox_id and self.terminate_on_complete:
            self.sandbox_provider.delete(sandbox_id)

        return None
