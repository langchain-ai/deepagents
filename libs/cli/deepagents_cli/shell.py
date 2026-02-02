"""Simplified middleware that exposes a basic shell tool to agents."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import ToolException


class ShellMiddleware(AgentMiddleware[AgentState, Any]):
    """Give basic shell access to agents via the shell.

    This shell will execute on the local machine and has NO safeguards except
    for the human in the loop safeguard provided by the CLI itself.
    """

    def __init__(
        self,
        *,
        workspace_root: str,
        timeout: float = 120.0,
        max_output_bytes: int = 100_000,
        env: dict[str, str] | None = None,
    ) -> None:
        """Initialize an instance of `ShellMiddleware`.

        Args:
            workspace_root: Working directory for shell commands.
            timeout: Maximum time in seconds to wait for command completion.
                Defaults to 120 seconds.
            max_output_bytes: Maximum number of bytes to capture from command output.
                Defaults to 100,000 bytes.
            env: Environment variables to pass to the subprocess. If None,
                uses the current process's environment. Defaults to None.
        """
        super().__init__()
        self._timeout = timeout
        self._max_output_bytes = max_output_bytes
        self._tool_name = "shell"
        self._env = env if env is not None else os.environ.copy()
        self._workspace_root = workspace_root

        # Build description with working directory information
        description = f"""Execute shell commands directly on the host machine.

Working Directory: {workspace_root}

**When to Use:**
- Git operations (git status, git diff, git commit, git push)
- Build commands (make, npm build, cargo build, go build)
- Package management (pip install, npm install, cargo add)
- Running tests (pytest, npm test, go test)
- Process management (docker, systemctl)
- Running scripts (python script.py, node script.js)

**When NOT to Use:**
- Reading files - use `read_file` instead
- Editing files - use `edit_file` instead
- Writing files - use `write_file` instead
- Searching file contents - use `grep` tool instead
- Finding files by pattern - use `glob` tool instead
- Communicating with user - output text directly, never use echo

**Usage Guidelines:**

1. **Path Quoting**: ALWAYS quote paths containing spaces with double quotes
   - Good: `cd "/Users/name/My Documents"`
   - Bad: `cd /Users/name/My Documents`

2. **Working Directory**: Use absolute paths, avoid `cd` commands
   - Good: `pytest /foo/bar/tests`
   - Bad: `cd /foo/bar && pytest tests`

3. **Chaining Commands**:
   - Use `&&` when commands depend on each other
   - Use `;` when you don't care if earlier commands fail
   - DO NOT use newlines to separate commands

4. **Output Limits**: Output truncated at {self._max_output_bytes} bytes, timeout after {self._timeout}s

**Important:**
- Each command runs in a fresh shell (environment variables don't persist)
- Shell state resets between calls
- For long output, consider redirecting to temp files and reading with `read_file`"""

        @tool(self._tool_name, description=description)
        def shell_tool(
            command: str,
            runtime: ToolRuntime[None, AgentState],
        ) -> ToolMessage | str:
            """Execute a shell command.

            Args:
                command: The shell command to execute.
                runtime: The tool runtime context.
            """
            return self._run_shell_command(command, tool_call_id=runtime.tool_call_id)

        self._shell_tool = shell_tool
        self.tools = [self._shell_tool]

    def _run_shell_command(
        self,
        command: str,
        *,
        tool_call_id: str | None,
    ) -> ToolMessage | str:
        """Execute a shell command and return the result.

        Args:
            command: The shell command to execute.
            tool_call_id: The tool call ID for creating a ToolMessage.

        Returns:
            A ToolMessage with the command output or an error message.
        """
        if not command or not isinstance(command, str):
            msg = "Shell tool expects a non-empty command string."
            raise ToolException(msg)

        try:
            result = subprocess.run(
                command,
                check=False,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self._timeout,
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
            output = f"Error: Command timed out after {self._timeout:.1f} seconds."
            status = "error"

        return ToolMessage(
            content=output,
            tool_call_id=tool_call_id,
            name=self._tool_name,
            status=status,
        )


__all__ = ["ShellMiddleware"]
