"""`LocalShellBackend`: Filesystem backend with unrestricted local shell execution.

This backend extends `FilesystemBackend` to add shell command execution on
the local host system. It provides NO sandboxing or isolation - all operations
run directly on the host machine with full system access.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import suppress
from contextvars import ContextVar
from typing import TYPE_CHECKING

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol, execute_accepts_timeout

if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)

DEFAULT_EXECUTE_TIMEOUT = 120
"""Default timeout in seconds for shell command execution."""

_CANCELLATION_POLL_INTERVAL = 0.1
_ASYNC_CANCELLATION_GRACE_PERIOD = 1
"""Maximum seconds to wait for cooperative cleanup after async cancellation."""

_PROCESS_REAP_TIMEOUT = 5
"""Maximum seconds to wait for a killed shell process to exit."""

_ASYNC_EXECUTION_CONTEXT: ContextVar[tuple[object, threading.Event] | None] = ContextVar(
    "_ASYNC_EXECUTION_CONTEXT",
    default=None,
)
_BACKGROUND_WORKERS: set[asyncio.Task[ExecuteResponse]] = set()
"""Workers retained until an uncooperative `execute` override finishes."""


class _CommandCancelled(BaseException):
    """Signal async cancellation through synchronous execution wrappers."""


def _release_background_worker(worker: asyncio.Task[ExecuteResponse]) -> None:
    """Consume the result of an execution worker retained after cancellation."""
    _BACKGROUND_WORKERS.discard(worker)
    if worker.cancelled():
        return
    try:
        worker.result()
    except BaseException:  # noqa: BLE001  # Done callbacks cannot propagate worker control-flow exceptions.
        logger.warning("Local shell execution failed after its caller was cancelled", exc_info=True)


async def _wait_for_worker_shutdown(worker: asyncio.Task[ExecuteResponse]) -> bool:
    """Wait through repeated cancellation up to the cleanup grace period."""
    deadline = asyncio.get_running_loop().time() + _ASYNC_CANCELLATION_GRACE_PERIOD
    while not worker.done():
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return False
        with suppress(asyncio.CancelledError):
            await asyncio.wait({worker}, timeout=remaining)
    return True


def _kill_and_reap(process: subprocess.Popen[str], process_group: int | None) -> None:
    """Kill a command's process group and reap its shell."""
    kill_succeeded = False
    with suppress(BaseException):
        if process_group is None:
            process.kill()
        else:
            os.killpg(process_group, signal.SIGKILL)
        kill_succeeded = True
    if not kill_succeeded:
        with suppress(BaseException):
            process.kill()

    with suppress(BaseException):
        process.wait(timeout=_PROCESS_REAP_TIMEOUT)
    with suppress(BaseException):
        if process.stdout is not None:
            process.stdout.close()
    with suppress(BaseException):
        if process.stderr is not None:
            process.stderr.close()


def _communicate(
    process: subprocess.Popen[str],
    timeout: int,
    cancellation_event: threading.Event | None = None,
    *,
    process_group: int | None = None,
) -> tuple[str, str]:
    """Collect output while ensuring interrupted commands cannot outlive us."""
    if cancellation_event is None:
        try:
            return process.communicate(timeout=timeout)
        except BaseException:
            _kill_and_reap(process, process_group)
            raise

    deadline = time.monotonic() + timeout
    while not cancellation_event.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_and_reap(process, process_group)
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            return process.communicate(timeout=min(remaining, _CANCELLATION_POLL_INTERVAL))
        except subprocess.TimeoutExpired:
            continue
        except BaseException:
            _kill_and_reap(process, process_group)
            raise
    _kill_and_reap(process, process_group)
    raise _CommandCancelled


class LocalShellBackend(FilesystemBackend, SandboxBackendProtocol):
    """Filesystem backend with unrestricted local shell command execution.

    This backend extends `FilesystemBackend` to add shell command execution
    capabilities. Commands are executed directly on the host system without any
    sandboxing, process isolation, or security restrictions.

    !!! warning "Security Warning"

        This backend grants agents BOTH direct filesystem access AND
        unrestricted shell execution on your local machine. Use with extreme
        caution and only in appropriate environments.

        **Appropriate use cases:**

        - Local development CLIs (coding assistants, development tools)
        - Personal development environments where you trust the agent's code
        - CI/CD pipelines with proper secret management (see
            security considerations)

        **Inappropriate use cases:**

        - Production environments (e.g., web servers, APIs, multi-tenant systems)
        - Processing untrusted user input or executing untrusted code

        Use `StateBackend`, `StoreBackend`, or extend `BaseSandbox` for production.

        **Security risks:**

        - Agents can execute **arbitrary shell commands** with your
            user's permissions
        - Agents can read **any accessible file**, including secrets (API keys,
            credentials, `.env` files, SSH keys, etc.)
        - Combined with network tools, secrets may be exfiltrated via SSRF attacks
        - File modifications and command execution are **permanent and irreversible**
        - Agents can install packages, modify system files, spawn processes, etc.
        - **No process isolation** - commands run directly on your host system
        - **No resource limits** - commands can consume unlimited CPU, memory, disk

        **Recommended safeguards:**

        Since shell access is unrestricted and can bypass
        filesystem restrictions:

        1. **Enable Human-in-the-Loop (HITL) middleware** to review and
            approve ALL operations before execution. This is
            STRONGLY RECOMMENDED as your primary safeguard when using this backend.
        2. Run in dedicated development environments only - never on shared or
            production systems
        3. Never expose to untrusted users or allow execution of untrusted code
        4. For production environments requiring code execution, extend `BaseSandbox`
            to create a properly isolated backend (Docker containers, VMs, or
            other sandboxed execution environments)

        !!! note

            `virtual_mode=True` and path-based restrictions provide NO security
            with shell access enabled, since commands can access any path on
            the system

    Examples:
        ```python
        from deepagents.backends import LocalShellBackend

        # Create backend with explicit environment
        backend = LocalShellBackend(root_dir="/home/user/project", env={"PATH": "/usr/bin:/bin"})

        # Execute shell commands (runs directly on host)
        result = backend.execute("ls -la")
        print(result.output)
        print(result.exit_code)

        # Use filesystem operations (inherited from FilesystemBackend)
        content = backend.read("/README.md")
        backend.write("/output.txt", "Hello world")

        # Inherit all environment variables
        backend = LocalShellBackend(root_dir="/home/user/project", inherit_env=True)
        ```
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        *,
        virtual_mode: bool = True,
        timeout: int = DEFAULT_EXECUTE_TIMEOUT,
        max_output_bytes: int = 100_000,
        env: dict[str, str] | None = None,
        inherit_env: bool = False,
    ) -> None:
        """Initialize local shell backend with filesystem access.

        Args:
            root_dir: Working directory for both filesystem operations and shell commands.

                - If not provided, defaults to the current working directory.
                - Shell commands execute with this as their working directory.
                - When `virtual_mode=False`: Paths are used as-is.

                    Agents can access any file using absolute paths or `..` sequences.
                - When `virtual_mode=True` (default): Acts as a virtual root for filesystem operations.

                    Useful with `CompositeBackend` to support routing file
                    operations across different backend implementations.

                    **Note:** This does NOT restrict shell commands.

            virtual_mode: Enable virtual path mode for filesystem operations.

                When `True` (default), treats `root_dir` as a virtual root filesystem.
                All paths are interpreted relative to `root_dir`
                (e.g., `/file.txt` maps to `{root_dir}/file.txt`).
                Path traversal (`..`, `~`) is blocked.

                **Primary use case:** Working with `CompositeBackend`, which
                routes different path prefixes to different backends. Virtual
                mode allows the `CompositeBackend` to strip route prefixes and
                pass normalized paths to each backend, enabling file operations
                to work correctly across multiple backend implementations.

                **Important:** This only affects filesystem operations.
                Shell commands executed via `execute()` are NOT restricted
                and can access any path.

            timeout: Default maximum time in seconds to wait for shell command execution.

                Defaults to 120 seconds (2 minutes).

                Commands exceeding this timeout will be terminated.

                Can be overridden per-command via the `timeout` parameter
                on `execute()`.

            max_output_bytes: Maximum number of bytes to capture from command output.
                Output exceeding this limit will be truncated.

                Defaults to 100,000 bytes.

            env: Environment variables for shell commands.

                If `None`, starts with an empty environment
                (unless `inherit_env=True`).

            inherit_env: Whether to inherit the parent process's environment variables.

                When `False` (default), only variables in `env` dict are available.

                When `True`, inherits all `os.environ` variables
                and applies `env` overrides.

        Raises:
            ValueError: If timeout is not positive.
        """
        if timeout <= 0:
            msg = f"timeout must be positive, got {timeout}"
            raise ValueError(msg)

        # Initialize parent FilesystemBackend
        super().__init__(
            root_dir=root_dir,
            virtual_mode=virtual_mode,
            max_file_size_mb=10,
        )

        # Store execution parameters
        self._default_timeout = timeout
        self._max_output_bytes = max_output_bytes

        # Build environment based on inherit_env setting
        if inherit_env:
            self._env = os.environ.copy()
            if env is not None:
                self._env.update(env)
        else:
            self._env = env if env is not None else {}

        # Generate unique sandbox ID
        self._sandbox_id = f"local-{uuid.uuid4().hex[:8]}"

    @property
    def id(self) -> str:
        """Unique identifier for this backend instance.

        Returns:
            String identifier in format "local-{random_hex}".
        """
        return self._sandbox_id

    def _execute_in_thread(
        self,
        command: str,
        timeout: int | None,
        cancellation_event: threading.Event,
        execution_started: threading.Event,
    ) -> ExecuteResponse:
        """Run a command only if cancellation did not win the start race."""
        # Publishing first makes it safe for `aexecute` to stop waiting when
        # this event is unset: a worker that starts later will see cancellation.
        execution_started.set()
        if cancellation_event.is_set():
            raise asyncio.CancelledError
        token = _ASYNC_EXECUTION_CONTEXT.set((self, cancellation_event))
        try:
            if timeout is not None and execute_accepts_timeout(type(self)):
                return self.execute(command, timeout=timeout)
            return self.execute(command)
        except _CommandCancelled:
            raise asyncio.CancelledError from None
        finally:
            _ASYNC_EXECUTION_CONTEXT.reset(token)

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,  # noqa: ASYNC109  # Command timeout, not coroutine timeout.
    ) -> ExecuteResponse:
        """Execute a shell command asynchronously.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for the command.

        Returns:
            The command output, exit code, and truncation status.

        Raises:
            asyncio.CancelledError: If the caller cancels command execution.

        Note:
            Cancellation allows synchronous execution one second for cleanup.
            An overridden `execute` method that does not cooperate may continue
            running in a background thread after cancellation is raised.
        """
        cancellation_event = threading.Event()
        execution_started = threading.Event()
        worker = asyncio.create_task(
            asyncio.to_thread(
                self._execute_in_thread,
                command,
                timeout,
                cancellation_event,
                execution_started,
            )
        )
        try:
            await asyncio.wait({worker})
            return worker.result()
        except asyncio.CancelledError:
            cancellation_event.set()
            if not execution_started.is_set():
                worker.cancel()
            if not await _wait_for_worker_shutdown(worker):
                logger.warning(
                    "Cancellation of local shell backend %s (%s) exceeded %s second; its overridden execute method may still be running",
                    self.id,
                    type(self).__name__,
                    _ASYNC_CANCELLATION_GRACE_PERIOD,
                )
                _BACKGROUND_WORKERS.add(worker)
                worker.add_done_callback(_release_background_worker)
                raise
            if not worker.cancelled():
                worker.exception()
            raise

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        r"""Execute a shell command directly on the host system.

        !!! danger "Unrestricted Execution"

            Commands are executed directly on your host system
            using `subprocess.Popen()` with `shell=True`. There is **no sandboxing,
            isolation, or security restrictions**. The command runs with
            your user's full permissions and can:

            - Access any file on the filesystem (regardless of `virtual_mode`)
            - Execute any program or script
            - Make network connections
            - Modify system configuration
            - Spawn additional processes
            - Install packages or modify dependencies

            **Always use Human-in-the-Loop (HITL) middleware when using this method.**

        The command is executed using the system shell (`/bin/sh` or equivalent)
        with the working directory set to the backend's `root_dir`.
        Stdout and stderr are combined into a single output stream.

        On POSIX systems, each command starts in a new session without sharing
        the parent's controlling terminal. Timeout, interruption, and async
        cancellation terminate the command's entire process group with
        `SIGKILL`, then wait briefly for the shell process to exit.

        On Windows, commands do not start in a new process group. Cleanup
        terminates only the direct shell process, so descendants may continue
        running after timeout, interruption, or async cancellation.

        Args:
            command: Shell command string to execute.

                Examples: `"python script.py"`, `"ls -la"`, `"grep pattern file.txt"`

                **Security:** This string is passed directly to the shell.
                Agents can execute arbitrary commands including pipes,
                redirects, command substitution, etc.
            timeout: Maximum time in seconds to wait for this command.

                Overrides the default timeout set at init.

                If `None`, uses the default.

        Returns:
            `ExecuteResponse` containing:
                - `output`: Combined stdout and stderr (stderr lines prefixed with `[stderr]`)
                - `exit_code`: Process exit code (0 for success, non-zero for failure)
                - `truncated`: `True` if output was truncated due to size limits

        Raises:
            ValueError: If per-command timeout is not positive.

        Examples:
            ```python
            # Run a simple command
            result = backend.execute("echo hello")
            assert result.output == "hello\\n"
            assert result.exit_code == 0

            # Handle errors
            result = backend.execute("cat nonexistent.txt")
            assert result.exit_code != 0
            assert "[stderr]" in result.output

            # Check for truncation
            result = backend.execute("cat huge_file.txt")
            if result.truncated:
                print("Output was truncated")

            # Override timeout for long-running commands
            result = backend.execute("make build", timeout=300)

            # Commands run in root_dir, but can access any path
            result = backend.execute("cat /etc/passwd")  # Can read system files!
            ```
        """
        execution_context = _ASYNC_EXECUTION_CONTEXT.get()
        cancellation_event = execution_context[1] if execution_context is not None and execution_context[0] is self else None
        return self._execute(command, timeout=timeout, cancellation_event=cancellation_event)

    def _execute(
        self,
        command: str,
        *,
        timeout: int | None,
        cancellation_event: threading.Event | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command, optionally observing async cancellation."""
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ValueError(msg)

        try:
            start_new_session = sys.platform != "win32"
            process = subprocess.Popen(  # noqa: S602
                command,
                shell=True,  # Intentional: designed for LLM-controlled shell execution
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,  # Prevent hanging on commands that read stdin (e.g. python, cat)
                text=True,
                env=self._env,
                cwd=str(self.cwd),  # Use the root_dir from FilesystemBackend
                start_new_session=start_new_session,
            )
            process_group = process.pid if start_new_session else None
            stdout, stderr = _communicate(
                process,
                effective_timeout,
                cancellation_event,
                process_group=process_group,
            )

            # Combine stdout and stderr
            # Prefix each stderr line with [stderr] for clear attribution.
            # Example: "hello\n[stderr] error: file not found"  # noqa: ERA001
            output_parts = []
            if stdout:
                output_parts.append(stdout)
            if stderr:
                stderr_lines = stderr.strip().split("\n")
                output_parts.extend(f"[stderr] {line}" for line in stderr_lines)

            output = "\n".join(output_parts) if output_parts else "<no output>"

            # Check for truncation
            truncated = False
            if len(output) > self._max_output_bytes:
                output = output[: self._max_output_bytes]
                output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
                truncated = True

            # Add exit code info if non-zero
            if process.returncode != 0:
                output = f"{output.rstrip()}\n\nExit code: {process.returncode}"

            return ExecuteResponse(
                output=output,
                exit_code=process.returncode,
                truncated=truncated,
            )

        except subprocess.TimeoutExpired:
            if timeout is not None:
                msg = f"Error: Command timed out after {effective_timeout} seconds (custom timeout). The command may be stuck or require more time."
            else:
                msg = f"Error: Command timed out after {effective_timeout} seconds. For long-running commands, re-run using the timeout parameter."
            return ExecuteResponse(
                output=msg,
                exit_code=124,  # Standard timeout exit code
                truncated=False,
            )
        except Exception as e:  # noqa: BLE001
            # Broad exception catch is intentional: we want to catch all execution errors
            # and return a consistent ExecuteResponse rather than propagating exceptions
            return ExecuteResponse(
                output=f"Error executing command ({type(e).__name__}): {e}",
                exit_code=1,
                truncated=False,
            )


__all__ = ["DEFAULT_EXECUTE_TIMEOUT", "LocalShellBackend"]
