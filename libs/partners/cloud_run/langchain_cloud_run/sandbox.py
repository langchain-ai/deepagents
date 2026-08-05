"""Cloud Run guest sandbox implementation for Deep Agents."""

import logging
import subprocess
from pathlib import Path

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

logger = logging.getLogger("langchain_cloud_run." + __name__)


class CloudRunSandbox(BaseSandbox):
    """Cloud Run guest sandbox backend for Deep Agents.

    Executes commands in an isolated guest sandbox environment inside Cloud Run
    via `/usr/local/gcp/bin/sandbox do`.

    Args:
        allow_egress: Whether to allow network access within the sandbox.
            Defaults to `False` for maximum network security and metadata protection.
        sandbox_bin: Absolute path to the Cloud Run sandbox binary.
            Defaults to `"/usr/local/gcp/bin/sandbox"`.
        default_timeout: Default timeout in seconds for command execution.
            Defaults to `1800` (30 minutes).
        env: Default environment variables to set in the sandbox (`-e KEY=VAL`).
        workdir: Default working directory to execute commands in (`--workdir DIR`).
        extra_sandbox_args: Pass-through list of additional flags to pass
            to `sandbox do`.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        allow_egress: bool = False,
        sandbox_bin: str = "/usr/local/gcp/bin/sandbox",
        default_timeout: int = 1800,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        extra_sandbox_args: list[str] | None = None,
    ) -> None:
        """Initialize the Cloud Run sandbox backend."""
        self.allow_egress = allow_egress
        self.sandbox_bin = sandbox_bin
        self.default_timeout = default_timeout
        self.env = env or {}
        self.workdir = workdir
        self.extra_sandbox_args = extra_sandbox_args or []

    @property
    def id(self) -> str:
        """Return unique string identifier for this sandbox instance."""
        return f"cloud-run-sandbox-{id(self)}"

    def execute(  # noqa: C901, PLR0912
        self,
        command: str | list[str],
        *,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        extra_sandbox_args: list[str] | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command or direct binary inside the Cloud Run sandbox.

        Args:
            command: Shell command string or list of command arguments to execute.
            timeout: Timeout in seconds for execution. Overrides default_timeout.
            env: Environment variables to set for this execution.
                Merged with default env.
            workdir: Working directory for this execution. Overrides default workdir.
            extra_sandbox_args: Additional flags to pass to `sandbox do`.
                Combined with default extra_sandbox_args.

        Returns:
            `ExecuteResponse` containing exit code, stdout, and formatted stderr.
        """
        cmd: list[str] = [self.sandbox_bin, "do"]

        # Control network access inside guest sandbox container
        if self.allow_egress:
            cmd.append("--allow-egress")

        # Set working directory if specified
        effective_workdir = workdir if workdir is not None else self.workdir
        if effective_workdir:
            cmd.extend(["--workdir", effective_workdir])

        # Append pass-through extra arguments (e.g., --mount, --write, --sync-tar)
        effective_extra = list(self.extra_sandbox_args)
        if extra_sandbox_args:
            effective_extra.extend(extra_sandbox_args)

        if effective_extra:
            cmd.extend(effective_extra)

        # Merge instance default environment variables with per-call overrides
        effective_env = dict(self.env)
        if env:
            effective_env.update(env)

        # Cloud Run's `sandbox do` constructs the guest container environment
        # array using ONLY passed `-e` flags. If `PATH` is not explicitly provided,
        # the guest container starts with an empty `PATH: []`, causing `sh` and
        # runtime binaries to fail with "no such file or directory". Fallback
        # to standard Linux PATH.
        if "PATH" not in effective_env:
            effective_env["PATH"] = (
                "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            )

        for k, v in effective_env.items():
            cmd.extend(["-e", f"{k}={v}"])

        cmd.append("--")

        # Support direct binary invocation (list of args) or shell string execution
        if isinstance(command, list):
            cmd.extend(command)
        else:
            cmd.extend(["sh", "-c", command])

        effective_timeout = timeout if timeout is not None else self.default_timeout

        logger.debug("Executing command in Cloud Run Sandbox: %s", command)

        try:
            res = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                check=False,
            )
            stdout = res.stdout or ""
            stderr = res.stderr or ""
            exit_code = res.returncode

            output = stdout
            if stderr:
                if output and not output.endswith("\n"):
                    output += "\n"
                output += f"<stderr>\n{stderr}\n</stderr>"

            return ExecuteResponse(
                output=output,
                exit_code=exit_code,
                truncated=False,
            )
        except FileNotFoundError:
            # Handle scenario where sandbox binary is missing (outside Cloud Run)
            err_msg = (
                f"Cloud Run sandbox binary '{self.sandbox_bin}' not found. "
                "Ensure your application is deployed inside Cloud Run with "
                "Sandbox enabled."
            )
            logger.error(err_msg)  # noqa: TRY400
            return ExecuteResponse(
                output=f"<stderr>\n{err_msg}\n</stderr>",
                exit_code=127,
                truncated=False,
            )
        except subprocess.TimeoutExpired:
            # Handle process execution timeout
            err_msg = f"Command execution timed out after {effective_timeout} seconds."
            logger.warning(err_msg)
            return ExecuteResponse(
                output=f"<stderr>\n{err_msg}\n</stderr>",
                exit_code=124,
                truncated=False,
            )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload binary/text files directly to container disk.

        The container storage is shared between parent container and guest sandbox.

        Args:
            files: List of `(path, contents)` tuples.

        Returns:
            List of `FileUploadResponse` objects indicating success or failure.
        """
        results: list[FileUploadResponse] = []
        for raw_path, contents in files:
            file_path = Path(raw_path)
            try:
                if file_path.parent:
                    file_path.parent.mkdir(parents=True, exist_ok=True)

                file_path.write_bytes(contents)
                results.append(FileUploadResponse(path=raw_path, error=None))
            except Exception as e:  # noqa: BLE001
                results.append(FileUploadResponse(path=raw_path, error=str(e)))

        return results

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download binary/text files directly from container disk.

        The container storage is shared between parent container and guest sandbox.

        Args:
            paths: List of absolute file paths to download.

        Returns:
            List of `FileDownloadResponse` objects containing contents or error details.
        """
        results: list[FileDownloadResponse] = []
        for raw_path in paths:
            file_path = Path(raw_path)
            try:
                content = file_path.read_bytes()
                results.append(
                    FileDownloadResponse(path=raw_path, content=content, error=None)
                )
            except Exception as e:  # noqa: BLE001
                results.append(
                    FileDownloadResponse(path=raw_path, content=None, error=str(e))
                )

        return results
