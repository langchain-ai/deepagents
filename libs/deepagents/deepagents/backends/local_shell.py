"""`LocalShellBackend`: Filesystem backend with unrestricted local shell execution.

This backend extends `FilesystemBackend` to add shell command execution on
the local host system. It provides NO sandboxing or isolation - all operations
run directly on the host machine with full system access.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from typing import TYPE_CHECKING

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

if TYPE_CHECKING:
    from pathlib import Path


DEFAULT_EXECUTE_TIMEOUT = 120
"""Default timeout in seconds for shell command execution."""

_SIGPIPE_STATUS = 128 + 13
"""Exit status of a process killed by `SIGPIPE`, i.e. one whose reader closed the pipe."""

_MIN_PIPELINE_STAGES = 2
"""Below this, the recorded statuses say nothing `returncode` does not already say."""

_PIPESTATUS_WRAPPER = """( trap 'printf "%s " "${{PIPESTATUS[@]}}" > {statusfile}' EXIT
{command}
)"""
"""Run the command under bash and side-channel the per-stage statuses of its last pipeline.

The subshell is what makes this safe: a command containing `exit` terminates the subshell
rather than the wrapper, and the `EXIT` trap still fires, so the statuses are recorded either
way. Writing them to a file rather than a stream keeps the command's own stdout and stderr
untouched.
"""


def _prepare_pipeline_status_capture(command: str) -> tuple[str, str | None, str | None]:
    """Wrap `command` so its pipeline statuses are recoverable, when the shell allows it.

    Returns the command to run, the shell to run it with, and the status file to read
    afterwards. Where bash is unavailable -- POSIX `sh` has no `PIPESTATUS`, and Windows has
    no `sh` at all -- returns the command untouched and `None` for the rest, leaving the
    existing behaviour exactly as it was.
    """
    bash = shutil.which("bash") if sys.platform != "win32" else None
    if bash is None:
        return command, None, None
    descriptor, statusfile = tempfile.mkstemp(prefix="deepagents-pipestatus-")
    os.close(descriptor)
    wrapped = _PIPESTATUS_WRAPPER.format(statusfile=shlex.quote(statusfile), command=command)
    return wrapped, bash, statusfile


def _read_pipeline_statuses(statusfile: str | None) -> list[int]:
    """Read the statuses the wrapper recorded, tolerating anything unexpected.

    An unreadable or malformed file must never fail the command the agent ran, so this
    degrades to an empty list, which `_resolve_pipeline_status` treats as "use returncode".
    """
    if statusfile is None:
        return []
    try:
        with open(statusfile) as handle:  # noqa: PTH123
            return [int(token) for token in handle.read().split()]
    except (OSError, ValueError):
        return []


def _discard_status_file(statusfile: str | None) -> None:
    """Remove the status side-channel, tolerating it never having been created."""
    if statusfile is None:
        return
    with contextlib.suppress(OSError):
        os.unlink(statusfile)  # noqa: PTH108


def _resolve_pipeline_status(statuses: list[int], returncode: int) -> int:
    """Reduce a pipeline's per-stage statuses to the one worth reporting.

    A shell reports only the last stage of a pipeline, so `pytest ... | tail` looks
    successful whenever `tail` succeeds. Walking the stages instead surfaces the failure
    the agent actually needs to see.

    Two rules matter:

    - A stage killed by `SIGPIPE` *before the last one* is not a failure. It means a later
      stage closed the pipe early, which is exactly what `head` and `grep -q` do on
      purpose. Treating it as failure would break far more commands than it fixes.
    - Fewer than two stages carries no more information than `returncode` and can carry
      less: for `echo hi; exit 5` the last pipeline succeeded, so the statuses read `[0]`
      while the command as a whole exited 5. Defer to `returncode` there.
    """
    if len(statuses) < _MIN_PIPELINE_STAGES:
        return returncode
    last = len(statuses) - 1
    for index, status in enumerate(statuses):
        if status == _SIGPIPE_STATUS and index < last:
            continue
        if status != 0:
            return status
    return 0


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

    def _combine_output(self, stdout: str, stderr: str) -> tuple[str, bool]:
        r"""Merge stdout and stderr into one stream, truncating past the size limit.

        Each stderr line is prefixed with `[stderr]` for clear attribution, e.g.
        `hello\n[stderr] error: file not found`.
        """
        output_parts = []
        if stdout:
            output_parts.append(stdout)
        if stderr:
            output_parts.extend(f"[stderr] {line}" for line in stderr.strip().split("\n"))

        output = "\n".join(output_parts) if output_parts else "<no output>"

        if len(output) > self._max_output_bytes:
            truncated_output = output[: self._max_output_bytes]
            return (
                f"{truncated_output}\n\n... Output truncated at {self._max_output_bytes} bytes.",
                True,
            )
        return output, False

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        r"""Execute a shell command directly on the host system.

        !!! danger "Unrestricted Execution"

            Commands are executed directly on your host system
            using `subprocess.run()` with `shell=True`. There is **no sandboxing,
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

        run_command, bash, statusfile = _prepare_pipeline_status_capture(command)

        try:
            result = subprocess.run(  # noqa: S602
                run_command,
                check=False,
                shell=True,  # Intentional: designed for LLM-controlled shell execution
                executable=bash,
                capture_output=True,
                stdin=subprocess.DEVNULL,  # Prevent hanging on commands that read stdin (e.g. python, cat)
                text=True,
                timeout=effective_timeout,
                env=self._env,
                cwd=str(self.cwd),  # Use the root_dir from FilesystemBackend
                start_new_session=(sys.platform != "win32"),
            )
            returncode = result.returncode
            if statusfile is not None:
                returncode = _resolve_pipeline_status(_read_pipeline_statuses(statusfile), result.returncode)

            output, truncated = self._combine_output(result.stdout, result.stderr)
            if returncode != 0:
                output = f"{output.rstrip()}\n\nExit code: {returncode}"

            return ExecuteResponse(
                output=output,
                exit_code=returncode,
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
        finally:
            _discard_status_file(statusfile)


__all__ = ["DEFAULT_EXECUTE_TIMEOUT", "LocalShellBackend"]
