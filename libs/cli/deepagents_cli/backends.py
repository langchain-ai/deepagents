"""CLI-specific backend and middleware overrides for per-command timeout support.

Subclasses `LocalShellBackend` and `FilesystemMiddleware` to add a per-command
`timeout` keyword argument to `execute()`, so the LLM can override the default
timeout for long-running commands. Uses monkey-patching via
`patch_filesystem_middleware()` to replace the SDK's `FilesystemMiddleware`
class reference with `CLIFilesystemMiddleware`, so the SDK constructs the CLI
subclass transparently.

When the SDK adds this natively, these subclasses can be removed.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess  # noqa: S404
from typing import Annotated, cast

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import (
    BackendProtocol,
    ExecuteResponse,
    SandboxBackendProtocol,
)
from deepagents.middleware.filesystem import (
    EXECUTE_TOOL_DESCRIPTION,
    FilesystemMiddleware,
    FilesystemState,
)
from langchain.tools import ToolRuntime  # noqa: TC002
from langchain_core.tools import BaseTool, StructuredTool

logger = logging.getLogger(__name__)

_TIMEOUT_DESC = (
    "Optional timeout in seconds. Overrides the default. Use for long-running commands."
)

# Must match LocalShellBackend's default timeout parameter.
DEFAULT_EXECUTE_TIMEOUT = 120
"""Default timeout in seconds for shell command execution."""


class CLIShellBackend(LocalShellBackend):
    """Local shell backend with per-command timeout override support.

    Extends `LocalShellBackend` to accept an optional `timeout` keyword on
    each `execute()` call.
    """

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        r"""Execute a shell command with optional per-command timeout.

        Overrides `LocalShellBackend.execute()` to accept a per-command
        `timeout` that overrides the default set at init time.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for this command.

                Overrides the default timeout set at init.

                If `None`, falls back to the instance-level timeout
                configured at init time (defaults to 120s if not overridden).

        Returns:
            ExecuteResponse containing output, exit code, and truncation flag.

        Raises:
            ValueError: If the effective timeout (per-command or instance
                default) is not positive.
        """
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        effective_timeout = timeout if timeout is not None else self._timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ValueError(msg)

        try:
            result = subprocess.run(  # noqa: S602
                command,
                check=False,
                shell=True,  # Intentional: designed for LLM-controlled shell execution
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                env=self._env,
                cwd=str(self.cwd),
            )

            # Combine stdout and stderr
            # Prefix each stderr line with [stderr] for clear attribution.
            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                stderr_lines = result.stderr.strip().split("\n")
                output_parts.extend(f"[stderr] {line}" for line in stderr_lines)

            output = "\n".join(output_parts) if output_parts else "<no output>"

            truncated = False
            if len(output) > self._max_output_bytes:
                output = output[: self._max_output_bytes]
                output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
                truncated = True

            return ExecuteResponse(
                output=output,
                exit_code=result.returncode,
                truncated=truncated,
            )

        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output=(
                    f"Error: Command timed out after {effective_timeout} seconds."
                    " For long-running commands, re-run using the timeout parameter."
                ),
                exit_code=124,  # Standard timeout exit code
                truncated=False,
            )
        except Exception as e:  # noqa: BLE001
            # Broad exception catch is intentional: we want to catch all
            # execution errors and return a consistent ExecuteResponse rather
            # than propagating exceptions that would crash the agent loop.
            return ExecuteResponse(
                output=f"Error executing command ({type(e).__name__}): {e}",
                exit_code=1,
                truncated=False,
            )


def _format_execute_result(result: ExecuteResponse) -> str:
    """Format an `ExecuteResponse` for LLM consumption.

    Args:
        result: The execution response to format.

    Returns:
        Formatted string with output, status, and truncation info.
    """
    parts = [result.output]
    if result.exit_code is not None:
        status = "succeeded" if result.exit_code == 0 else "failed"
        parts.append(f"\n[Command {status} with exit code {result.exit_code}]")
    if result.truncated:
        parts.append("\n[Output was truncated due to size limits]")
    return "".join(parts)


def _get_sandbox_backend(
    backend: BackendProtocol,
) -> SandboxBackendProtocol | None:
    """Unwrap a `CompositeBackend` to find the sandbox backend.

    `CompositeBackend.execute()` delegates to `self.default.execute()` but
    doesn't forward extra kwargs like `timeout`. This helper reaches through
    to the actual sandbox backend so we can call `execute(timeout=...)` on it.

    The returned backend must support a `timeout` kwarg on `execute()`
    (e.g., `CLIShellBackend`). The base `SandboxBackendProtocol` does not
    define this parameter -- the `timeout` kwarg is specific to the CLI
    subclass.

    Args:
        backend: The resolved backend, possibly a `CompositeBackend`.

    Returns:
        The sandbox backend if found, `None` otherwise.
    """
    if isinstance(backend, CompositeBackend):
        default = backend.default
        if isinstance(default, SandboxBackendProtocol):
            return default
        logger.warning(
            "CompositeBackend.default is %s, not a SandboxBackendProtocol. "
            "The execute tool timeout parameter will be unavailable.",
            type(default).__name__,
        )
        return None
    if isinstance(backend, SandboxBackendProtocol):
        return backend
    return None


class CLIFilesystemMiddleware(FilesystemMiddleware):
    """Filesystem middleware with per-command timeout on the execute tool.

    Overrides `_create_execute_tool` to expose a `timeout` parameter in the
    tool schema so the LLM can pass it for long-running commands.
    """

    def _create_execute_tool(self) -> BaseTool:
        """Create the execute tool with an additional `timeout` parameter.

        Returns:
            A `StructuredTool` for shell command execution with timeout support.
        """
        tool_description = (
            self._custom_tool_descriptions.get("execute") or EXECUTE_TOOL_DESCRIPTION
        )

        def sync_execute(
            command: Annotated[
                str, "Shell command to execute in the sandbox environment."
            ],
            runtime: ToolRuntime[None, FilesystemState],
            timeout: Annotated[int | None, _TIMEOUT_DESC] = None,
        ) -> str:
            """Synchronous wrapper for execute tool.

            Args:
                command: Shell command to execute.
                runtime: Tool runtime context for backend resolution.
                timeout: Optional per-command timeout in seconds.

            Returns:
                Formatted command output for LLM consumption.
            """
            if timeout is not None and timeout <= 0:
                return f"Error: timeout must be a positive integer, got {timeout}."

            resolved_backend = self._get_backend(runtime)  # type: ignore[arg-type]
            proto = _get_sandbox_backend(resolved_backend)

            if proto is None:
                return (
                    "Error: Execution not available. This agent's backend "
                    "does not support command execution."
                )

            sandbox = cast("CLIShellBackend", proto)
            try:
                result = sandbox.execute(command, timeout=timeout)
            except NotImplementedError as e:
                return f"Error: Execution not available. {e}"
            except ValueError as e:
                return f"Error: Invalid parameter. {e}"

            return _format_execute_result(result)

        async def async_execute(
            command: Annotated[
                str, "Shell command to execute in the sandbox environment."
            ],
            runtime: ToolRuntime[None, FilesystemState],
            timeout: Annotated[  # noqa: ASYNC109
                int | None, _TIMEOUT_DESC
            ] = None,
        ) -> str:
            """Asynchronous wrapper for execute tool.

            Args:
                command: Shell command to execute.
                runtime: Tool runtime context for backend resolution.
                timeout: Optional per-command timeout in seconds.

            Returns:
                Formatted command output for LLM consumption.
            """
            if timeout is not None and timeout <= 0:
                return f"Error: timeout must be a positive integer, got {timeout}."

            resolved_backend = self._get_backend(runtime)  # type: ignore[arg-type]
            proto = _get_sandbox_backend(resolved_backend)

            if proto is None:
                return (
                    "Error: Execution not available. This agent's backend "
                    "does not support command execution."
                )

            sandbox = cast("CLIShellBackend", proto)
            try:
                result = await asyncio.to_thread(
                    lambda: sandbox.execute(command, timeout=timeout),
                )
            except NotImplementedError as e:
                return f"Error: Execution not available. {e}"
            except ValueError as e:
                return f"Error: Invalid parameter. {e}"

            return _format_execute_result(result)

        return StructuredTool.from_function(
            name="execute",
            description=tool_description,
            func=sync_execute,
            coroutine=async_execute,
        )


def patch_filesystem_middleware() -> None:
    """Monkey-patch the SDK to use `CLIFilesystemMiddleware`.

    Must be called before `create_deep_agent` is invoked so the SDK's internal
    `FilesystemMiddleware(backend=...)` calls construct our subclass instead.

    Patches two module-level references:

    - `deepagents.middleware.filesystem.FilesystemMiddleware`
    - `deepagents.graph.FilesystemMiddleware`

    Both must be patched because `graph.py` imports the class at the top level
    and uses it directly when constructing middleware stacks. If the SDK adds
    additional import sites, this patch must be updated accordingly. Validate
    when upgrading the SDK version.
    """
    import deepagents.graph as graph_module
    import deepagents.middleware.filesystem as fs_module

    fs_module.FilesystemMiddleware = CLIFilesystemMiddleware  # type: ignore[misc]
    graph_module.FilesystemMiddleware = CLIFilesystemMiddleware  # type: ignore[misc]
