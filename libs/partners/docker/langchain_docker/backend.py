"""Docker sandbox backend."""

from __future__ import annotations

import atexit
import contextlib
import json
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Self

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

DEFAULT_EXECUTE_TIMEOUT = 120
DEFAULT_IMAGE = "python:3.12-bookworm"
CONTAINER_WORKDIR = "/shared"


class DockerError(RuntimeError):
    """Raised when a Docker CLI invocation fails."""


class DockerSandbox(FilesystemBackend, SandboxBackendProtocol):
    """Docker-backed sandbox backend for DeepAgents."""

    def __init__(  # noqa: PLR0913  # sandbox constructor exposes docker run options
        self,
        *,
        image: str = DEFAULT_IMAGE,
        allow_outbound_traffic: bool = True,
        shared_dir: str | Path | None = None,
        timeout: int = DEFAULT_EXECUTE_TIMEOUT,
        max_output_bytes: int = 100_000,
        memory: str = "256m",
        cpus: float = 0.5,
        pids_limit: int = 128,
        auto_remove: bool = True,
        extra_run_args: list[str] | None = None,
    ) -> None:
        """Create a sandbox container and shared directory.

        Args:
            image: Docker image for command execution (default: official
                `python:3.12-bookworm`).
            allow_outbound_traffic: Allow/deny outbound network traffic
                (default: allow).
            shared_dir: Host directory shared with the container. A temporary
                directory is created when omitted.
            timeout: Default command timeout in seconds.
            max_output_bytes: Maximum combined stdout/stderr captured per command.
            memory: Docker memory limit (for example ``"256m"``).
            cpus: Docker CPU limit.
            pids_limit: Maximum number of PIDs inside the container.
            auto_remove: Remove the container on ``close()``.
            extra_run_args: Additional ``docker run`` flags appended before the image.
        """
        if timeout <= 0:
            msg = f"timeout must be positive, got {timeout}"
            raise ValueError(msg)
        if cpus <= 0:
            msg = f"cpus must be positive, got {cpus}"
            raise ValueError(msg)
        if pids_limit <= 0:
            msg = f"pids_limit must be positive, got {pids_limit}"
            raise ValueError(msg)

        self._owns_shared_dir = shared_dir is None
        self._shared_dir = Path(
            tempfile.mkdtemp(prefix="deepagents-docker-shared-")
            if shared_dir is None
            else shared_dir,
        ).resolve()
        self._shared_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(root_dir=self._shared_dir, virtual_mode=True)

        self._image = image
        self._default_timeout = timeout
        self._max_output_bytes = max_output_bytes
        self._memory = memory
        self._cpus = cpus
        self._pids_limit = pids_limit
        self._network_mode = "bridge" if allow_outbound_traffic else "none"
        self._auto_remove = auto_remove
        self._extra_run_args = list(extra_run_args or [])

        self._container_id: str = f"{uuid.uuid4().hex[:12]}"
        self._container_name = f"deepagents-docker-{self._container_id}"
        self._closed = False

        self._start_container()
        atexit.register(self.close)

    @property
    def shared_dir(self) -> Path:
        """Host path of the folder shared with the container."""
        return self._shared_dir

    @property
    def id(self) -> str:
        """Unique identifier for this sandbox instance."""
        return self._container_id

    def _start_container(self) -> None:
        if not self._docker_available():
            msg = (
                "Docker is not available. Install Docker, ensure the daemon "
                "is running, and pull the default image with "
                f"`docker pull {DEFAULT_IMAGE}`"
            )
            raise DockerError(msg)

        run_args = [
            "run",
            "-d",
            "--name",
            self._container_name,
            "--network",
            self._network_mode,
            "--cpus",
            str(self._cpus),
            "--memory",
            self._memory,
            "--pids-limit",
            str(self._pids_limit),
            "-v",
            f"{self._shared_dir}:{CONTAINER_WORKDIR}:rw",
            "-w",
            CONTAINER_WORKDIR,
            *self._extra_run_args,
            self._image,
            "sleep",
            "infinity",
        ]
        result = self._run_docker(run_args)
        if result.returncode != 0:
            detail = self._format_docker_error(result)
            msg = f"failed to start sandbox container: {detail}"
            raise DockerError(msg)

    def execute(  # noqa: C901  # timeout and docker error paths stay in one method
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox container."""
        if self._closed:
            return ExecuteResponse(
                output="Error: Sandbox has been closed.",
                exit_code=1,
                truncated=False,
            )

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

        docker_args = [
            "exec",
            self._container_name,
            "sh",
            "-c",
            command,
        ]

        try:
            completed = self._run_docker(docker_args, timeout=effective_timeout)
        except DockerError as exc:
            detail = str(exc)
            if "timed out" in detail:
                if timeout is not None:
                    msg = (
                        f"Error: Command timed out after {effective_timeout} "
                        "seconds (custom timeout). The command may be stuck or "
                        "require more time."
                    )
                else:
                    msg = (
                        f"Error: Command timed out after {effective_timeout} seconds. "
                        "For long-running commands, re-run using the timeout parameter."
                    )
                return ExecuteResponse(output=msg, exit_code=124, truncated=False)
            if "not found on PATH" in detail:
                return ExecuteResponse(
                    output=(
                        "Error executing command (FileNotFoundError): "
                        "docker executable not found on PATH"
                    ),
                    exit_code=1,
                    truncated=False,
                )
            return ExecuteResponse(
                output=f"Error executing command (DockerError): {exc}",
                exit_code=1,
                truncated=False,
            )

        output_parts: list[str] = []
        if completed.stdout:
            output_parts.append(completed.stdout)
        if completed.stderr:
            stderr_lines = completed.stderr.strip().split("\n")
            output_parts.extend(f"[stderr] {line}" for line in stderr_lines)

        output = "\n".join(output_parts) if output_parts else "<no output>"
        truncated = False
        if len(output) > self._max_output_bytes:
            output = output[: self._max_output_bytes]
            output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
            truncated = True

        if completed.returncode != 0:
            output = f"{output.rstrip()}\n\nExit code: {completed.returncode}"

        return ExecuteResponse(
            output=output,
            exit_code=completed.returncode,
            truncated=truncated,
        )

    def close(self) -> None:
        """Stop and remove the sandbox container."""
        if self._closed:
            return
        self._closed = True

        stop = self._run_docker(["stop", "-t", "2", self._container_name], timeout=30)
        if stop.returncode != 0 and "No such container" not in stop.stderr:
            # Best-effort shutdown; container may already be gone.
            pass

        if self._auto_remove:
            self._run_docker(["rm", "-f", self._container_name], timeout=30)

        if self._owns_shared_dir:
            shutil.rmtree(self._shared_dir, ignore_errors=True)

    def __enter__(self) -> Self:
        """Return this sandbox for use as a context manager."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the sandbox when leaving a context manager."""
        self.close()

    def __del__(self) -> None:
        """Close the sandbox during garbage collection."""
        with contextlib.suppress(Exception):
            self.close()

    def _docker_available(self) -> bool:
        result = self._run_docker(["info", "--format", "{{.ServerVersion}}"])
        return result.returncode == 0

    def _format_docker_error(self, result: subprocess.CompletedProcess[str]) -> str:
        detail = (result.stderr or result.stdout).strip()
        if not detail:
            return f"exit code {result.returncode}"
        try:
            payload = json.loads(detail)
        except json.JSONDecodeError:
            return detail
        if isinstance(payload, dict) and "message" in payload:
            return str(payload["message"])
        return detail

    def _run_docker(
        self,
        args: list[str],
        *,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(  # noqa: S603
                ["docker", *args],  # noqa: S607  # resolve `docker` from PATH
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            msg = f"docker command timed out after {timeout} seconds"
            raise DockerError(msg) from exc
        except FileNotFoundError as exc:
            msg = "docker executable not found on PATH"
            raise DockerError(msg) from exc
