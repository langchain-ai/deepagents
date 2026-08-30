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
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

DEFAULT_EXECUTE_TIMEOUT = 120
"""Default timeout in seconds for shell command execution."""

_POSIX = sys.platform != "win32"
"""Whether this host has POSIX process groups and a POSIX shell. Windows has neither."""

_MIN_PIPELINE_STAGES = 2
"""Below this, the recorded statuses say nothing `returncode` does not already say."""

_SIGPIPE_STATUS = 141
"""128 + `SIGPIPE`: how a shell reports a stage stopped because a later stage stopped reading.

`seq 1 100000 | head -3` and `yes | head` end this way by construction, so this status on a
non-final stage is the pipeline working rather than a fault. It is excluded from the check
for a hidden failure, and spelled out wherever it is shown, because a bare `141` reads as an
error to a model.
"""

_PIPESTATUS_WRAPPER = """trap 'printf "%s " "${{PIPESTATUS[@]}}" > {statusfile}' EXIT
{command}"""
"""Run the command under bash and side-channel the per-stage statuses of its last pipeline.

The `EXIT` trap fires however the command ends, including on `exit`, so the statuses are
always recorded. `exit` is a builtin that returns before `PIPESTATUS` is reset, though, so
what the trap records then belongs to an earlier pipeline; `_describe_pipeline_stages` detects
that and reports nothing. Writing to a file rather than a stream keeps the command's own
stdout and stderr untouched.
"""


def _run_shell(command: str, *, executable: str | None, timeout: int, env: dict[str, str], cwd: str) -> subprocess.CompletedProcess[str]:
    """Run `command` under the shell, killing the whole process group on timeout.

    `subprocess.run` kills only the process it started. That was enough when bash could
    `exec` the command in place, but wrapping it to capture `PIPESTATUS` means the direct
    child is the shell and the real work is a grandchild, which survives. The timeout is
    listed in `THREAT_MODEL.md` as the mitigation at trust boundary TB5, so it has to reach
    everything the command started.
    """
    with subprocess.Popen(  # noqa: S602
        command,
        shell=True,  # Intentional: designed for LLM-controlled shell execution
        executable=executable,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,  # Prevent hanging on commands that read stdin
        text=True,
        env=env,
        cwd=cwd,
        start_new_session=_POSIX,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            _collect_killed_process(process)
            raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Kill the timed-out command and anything it spawned.

    `start_new_session=True` makes the shell the leader of a new process group, so its pgid
    is its pid and `os.killpg` reaches every descendant. Looking the pgid up with
    `os.getpgid` instead fails in exactly the case that matters: a shell that has already
    returned while leaving a background descendant behind is an unreaped zombie, and on macOS
    `os.getpgid` raises `ProcessLookupError` for it, so nothing is signalled at all.
    """
    if not _POSIX:
        process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        process.kill()


def _collect_killed_process(process: subprocess.Popen[str]) -> None:
    """Collect the killed shell, doing what `subprocess.run` does on each platform.

    Windows accumulates the output on reader threads that only `communicate` joins. On POSIX
    `communicate` would read the pipes to EOF with no timeout, so a descendant that inherited
    them could hold `execute` open for its whole lifetime, long past the timeout. `wait`
    waits only for the shell, which has just been killed.
    """
    if _POSIX:
        process.wait()
    else:
        process.communicate()


def _prepare_pipeline_status_capture(command: str) -> tuple[str, str | None, str | None]:
    """Wrap `command` so its pipeline statuses are recoverable, when the shell allows it.

    Returns the command to run, the shell to run it with, and the status file to read
    afterwards. Where the wrapper cannot be set up -- POSIX `sh` has no `PIPESTATUS`, Windows
    has no `sh` at all, and a read-only or exhausted temp directory has nowhere to record the
    statuses -- returns the command untouched and `None` for the rest, leaving the existing
    behaviour exactly as it was. The detail is a diagnostic, so failing to arrange it must
    never stop the agent's command from running.
    """
    bash = shutil.which("bash") if _POSIX else None
    if bash is None:
        return command, None, None
    try:
        descriptor, statusfile = tempfile.mkstemp(prefix="deepagents-pipestatus-")
    except OSError:
        return command, None, None
    os.close(descriptor)
    wrapped = _PIPESTATUS_WRAPPER.format(statusfile=shlex.quote(statusfile), command=command)
    return wrapped, bash, statusfile


def _read_pipeline_statuses(statusfile: str | None) -> list[int]:
    """Read the statuses the wrapper recorded, tolerating anything unexpected.

    An unreadable or malformed file must never fail the command the agent ran, so this
    degrades to an empty list, which `_describe_pipeline_stages` reports nothing for.
    """
    if statusfile is None:
        return []
    try:
        return [int(token) for token in Path(statusfile).read_text().split()]
    except (OSError, ValueError):
        return []


def _discard_status_file(statusfile: str | None) -> None:
    """Remove the status side-channel, tolerating it never having been created."""
    if statusfile is None:
        return
    with contextlib.suppress(OSError):
        Path(statusfile).unlink()


def _describe_pipeline_stages(statuses: list[int], returncode: int) -> str:
    """Describe a pipeline's per-stage statuses, or "" when there is nothing to add.

    Deliberately does not decide whether the command failed. Nothing at this layer can:
    `pytest` exiting 1 means the tests failed, `grep` exiting 1 means it matched nothing,
    and the shell cannot tell them apart. The agent can, because it chose the command, so
    the stage statuses are reported and the judgement is left to the reader.

    Returns "" when the statuses add nothing or cannot be trusted:

    - Fewer than two stages says no more than `returncode` already does.
    - A last stage disagreeing with `returncode` means the array is stale. `exit` is a
      builtin that terminates before `PIPESTATUS` is reset, so for `false | cat; exit 9`
      the trap still sees the earlier pipeline's `[1, 0]` while the command exited 9.
    - No non-final stage failing. `returncode` is the last stage's status, so for
      `cat hosts | grep -c zzz` the detail would only repeat it, and a pipeline whose
      earlier stages were all fine has nothing hidden in it. `SIGPIPE` does not count as a
      failure here, since it is how `seq 1 100000 | head -3` is meant to end.
    """
    if len(statuses) < _MIN_PIPELINE_STAGES:
        return ""
    if statuses[-1] != returncode:
        return ""
    if all(status in (0, _SIGPIPE_STATUS) for status in statuses[:-1]):
        return ""
    return ", ".join(f"{status} (SIGPIPE)" if status == _SIGPIPE_STATUS else str(status) for status in statuses)


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
            using `subprocess.Popen` with `shell=True`. There is **no sandboxing,
            isolation, or security restrictions**. The command runs with
            your user's full permissions and can:

            - Access any file on the filesystem (regardless of `virtual_mode`)
            - Execute any program or script
            - Make network connections
            - Modify system configuration
            - Spawn additional processes
            - Install packages or modify dependencies

            **Always use Human-in-the-Loop (HITL) middleware when using this method.**

        The command is executed using bash where it is available, falling back to the
        system shell (`/bin/sh` or equivalent) otherwise, with the working directory set to
        the backend's `root_dir`. Bash is preferred because it exposes `PIPESTATUS`, which
        is what lets a failure hidden inside a pipeline be reported rather than dropped.
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
            result = _run_shell(
                run_command,
                executable=bash,
                timeout=effective_timeout,
                env=self._env,
                cwd=str(self.cwd),  # Use the root_dir from FilesystemBackend
            )
            stages = _describe_pipeline_stages(_read_pipeline_statuses(statusfile), result.returncode)

            output, truncated = self._combine_output(result.stdout, result.stderr)
            if stages:
                output = f"{output.rstrip()}\n\nPipeline stages exited: {stages}"
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
        finally:
            _discard_status_file(statusfile)


__all__ = ["DEFAULT_EXECUTE_TIMEOUT", "LocalShellBackend"]
