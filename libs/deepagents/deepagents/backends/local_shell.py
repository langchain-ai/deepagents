"""`LocalShellBackend`: Filesystem backend with unrestricted local shell execution.

This backend extends `FilesystemBackend` to add shell command execution on
the local host system. It provides NO sandboxing or isolation - all operations
run directly on the host machine with full system access.
"""

from __future__ import annotations

import codecs
import locale
import os
import subprocess
import sys
import time
import uuid
from typing import TYPE_CHECKING

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

if TYPE_CHECKING:
    from pathlib import Path


DEFAULT_EXECUTE_TIMEOUT = 120
"""Default timeout in seconds for shell command execution."""

_OUTPUT_POLL_INTERVAL = 0.05
_OUTPUT_DRAIN_TIMEOUT = 0.5


def _decode_partial_output(output: bytes | str | None, encoding: str) -> str:
    """Apply text-mode decoding and newline normalization to partial pipe output."""
    text = codecs.getincrementaldecoder(encoding)().decode(output) if isinstance(output, bytes) else output or ""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _collect_output(process: subprocess.Popen[str], timeout: int, encoding: str) -> tuple[str, str, bool]:
    """Drain both pipes while monitoring shell exit, then bound the EOF wait."""
    deadline = time.monotonic() + timeout
    while process.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            return (*process.communicate(timeout=min(remaining, _OUTPUT_POLL_INTERVAL)), False)
        except subprocess.TimeoutExpired:
            pass
    try:
        return (*process.communicate(timeout=_OUTPUT_DRAIN_TIMEOUT), False)
    except subprocess.TimeoutExpired as error:
        return (
            _decode_partial_output(error.output, encoding),
            _decode_partial_output(error.stderr, encoding),
            True,
        )


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

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute an unrestricted shell command with bounded post-exit output collection."""
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
            result, background_running = self._run_command(command, effective_timeout)

            # Combine stdout and stderr
            # Prefix each stderr line with [stderr] for clear attribution.
            # Example: "hello\n[stderr] error: file not found"  # noqa: ERA001
            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                stderr_lines = result.stderr.strip().split("\n")
                output_parts.extend(f"[stderr] {line}" for line in stderr_lines)

            output = "\n".join(output_parts) if output_parts else "<no output>"

            # Check for truncation
            truncated = False
            if len(output) > self._max_output_bytes:
                output = output[: self._max_output_bytes]
                output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
                truncated = True

            output = f"{output.rstrip()}\n\nNote: background processes are still running." if background_running else output

            # Add exit code info if non-zero
            if result.returncode != 0:
                output = f"{output.rstrip()}\n\nExit code: {result.returncode}"

            return ExecuteResponse(
                output=output,
                exit_code=result.returncode,
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

    def _run_command(self, command: str, timeout: int) -> tuple[subprocess.CompletedProcess[str], bool]:
        """Collect command output without waiting for background descendants to exit."""
        encoding = locale.getpreferredencoding(do_setlocale=False)
        with subprocess.Popen(  # noqa: S602  # Intentional unrestricted shell execution.
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding=encoding,
            env=self._env,
            cwd=str(self.cwd),
            start_new_session=(sys.platform != "win32"),
        ) as process:
            try:
                stdout, stderr, background_running = _collect_output(process, timeout, encoding)
            except BaseException:
                process.kill()
                process.wait()
                raise
            return subprocess.CompletedProcess(command, process.wait(), stdout, stderr), background_running


__all__ = ["DEFAULT_EXECUTE_TIMEOUT", "LocalShellBackend"]
