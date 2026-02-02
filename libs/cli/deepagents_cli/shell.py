"""Simplified middleware that exposes a basic shell tool to agents."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import ToolException

_DEFAULT_SHELL_TIMEOUT = 120


class ShellMiddleware(AgentMiddleware[AgentState, Any]):
    """Give basic shell access to agents via the shell.

    This shell will execute on the local machine and has NO safeguards except
    for the human in the loop safeguard provided by the CLI itself.
    """

    def __init__(
        self,
        *,
        workspace_root: str,
        timeout: int = _DEFAULT_SHELL_TIMEOUT,
        max_output_bytes: int = 100_000,
        env: dict[str, str] | None = None,
    ) -> None:
        """Initialize an instance of `ShellMiddleware`.

        Args:
            workspace_root: Working directory for shell commands.
            timeout: Default maximum time in seconds to wait for command completion.

                Defaults to 120 seconds.
            max_output_bytes: Maximum number of bytes to capture from command output.
                Defaults to 100,000 bytes.
            env: Environment variables to pass to the subprocess. If None,
                uses the current process's environment. Defaults to None.
        """
        super().__init__()
        if timeout <= 0:
            msg = f"timeout must be positive, got {timeout}"
            raise ValueError(msg)
        self._default_timeout = timeout
        self._max_output_bytes = max_output_bytes
        self._tool_name = "shell"
        self._env = env if env is not None else os.environ.copy()
        self._workspace_root = workspace_root

        # Build description with working directory information
        description = (
            f"Execute a shell command directly on the host. Commands will run in "
            f"the working directory: {workspace_root}. Each command runs in a fresh shell "
            f"environment with the current process's environment variables. Commands may "
            f"be truncated if they exceed the configured timeout ({self._default_timeout}s) "
            f"or output limits. Use the optional timeout parameter for long-running commands."
        )

        @tool(self._tool_name, description=description)
        def shell_tool(
            command: str,
            runtime: ToolRuntime[None, AgentState],
            timeout: int | None = None,
        ) -> ToolMessage | str:
            """Execute a shell command.

            Args:
                command: The shell command to execute.
                runtime: The tool runtime context.
                timeout: Optional timeout in seconds for this command. Use for
                    long-running commands that may exceed the default timeout.
            """
            return self._run_shell_command(
                command, tool_call_id=runtime.tool_call_id, timeout=timeout
            )

        self._shell_tool = shell_tool
        self.tools = [self._shell_tool]

    def _run_shell_command(
        self,
        command: str,
        *,
        tool_call_id: str | None,
        timeout: int | None = None,
    ) -> ToolMessage:
        """Execute a shell command and return the result.

        Args:
            command: The shell command to execute.
            tool_call_id: The tool call ID for creating a ToolMessage.
            timeout: Optional per-command timeout override.

        Returns:
            A ToolMessage with the command output or an error message.
        """
        if not command or not isinstance(command, str):
            msg = "Shell tool expects a non-empty command string."
            raise ToolException(msg)

        # Determine effective timeout: per-command > default
        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ToolException(msg)

        try:
            result = subprocess.run(
                command,
                check=False,
                shell=True,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                env=self._env,
                cwd=self._workspace_root,
            )

            # Combine stdout and stderr
            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                stderr_lines = result.stderr.strip().split("\n")
                for line in stderr_lines:
                    output_parts.append(f"[stderr] {line}")

            output = "\n".join(output_parts) if output_parts else "<no output>"

            # Truncate output if needed
            if len(output) > self._max_output_bytes:
                output = output[: self._max_output_bytes]
                output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."

            # Add exit code info if non-zero
            if result.returncode != 0:
                output = f"{output.rstrip()}\n\nExit code: {result.returncode}"
                status = "error"
            else:
                status = "success"

        except subprocess.TimeoutExpired:
            output = (
                f"Error: Command timed out after {effective_timeout} seconds. "
                f"For long-running commands, re-run use the timeout parameter."
            )
            status = "error"

        return ToolMessage(
            content=output,
            tool_call_id=tool_call_id,
            name=self._tool_name,
            status=status,
        )


__all__ = ["ShellMiddleware"]
