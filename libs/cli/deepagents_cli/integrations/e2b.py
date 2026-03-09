"""E2B sandbox backend implementation."""

from __future__ import annotations

import importlib.util
import os
import time
from typing import TYPE_CHECKING, Any

if importlib.util.find_spec("e2b") is None:
    msg = "e2b package is required for E2BProvider. Install with `pip install e2b`."
    raise ImportError(msg)

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileOperationError,
    FileUploadResponse,
    SandboxBackendProtocol,
)
from deepagents.backends.sandbox import BaseSandbox

from deepagents_cli.integrations.sandbox_provider import (
    SandboxNotFoundError,
    SandboxProvider,
)

if TYPE_CHECKING:
    from e2b import Sandbox

DEFAULT_SANDBOX_LIFETIME = 3600
DEFAULT_STARTUP_TIMEOUT = 180
DEFAULT_COMMAND_TIMEOUT = 30 * 60
DEFAULT_WORKDIR = "/home/user"
READINESS_POLL_INTERVAL = 2


def _combine_output(stdout: str | None, stderr: str | None) -> str:
    """Combine stdout and stderr into the protocol output shape.

    Args:
        stdout: Command standard output.
        stderr: Command standard error.

    Returns:
        Combined command output.
    """
    output = stdout or ""
    if stderr:
        output += "\n" + stderr if output else stderr
    return output or "<no output>"


def _is_invalid_path(path: str) -> bool:
    """Return whether a path is malformed for sandbox file operations."""
    return not path.startswith("/") or "\x00" in path


def _map_error_message(
    message: str, *, default: FileOperationError
) -> FileOperationError:
    """Map provider error messages to standardized file operation errors.

    Args:
        message: Provider error message.
        default: Fallback error code.

    Returns:
        Normalized error code.
    """
    lowered = message.lower()
    if "permission" in lowered:
        return "permission_denied"
    if "directory" in lowered:
        return "is_directory"
    if "not found" in lowered or "no such file" in lowered:
        return "file_not_found"
    return default


class E2BBackend(BaseSandbox):
    """E2B backend implementation conforming to `SandboxBackendProtocol`."""

    def __init__(self, sandbox: Sandbox, *, workdir: str = DEFAULT_WORKDIR) -> None:
        """Initialize the E2B backend with a connected sandbox.

        Args:
            sandbox: Active E2B sandbox instance.
            workdir: Working directory used for command execution.
        """
        self._sandbox = sandbox
        self._default_timeout = DEFAULT_COMMAND_TIMEOUT
        self._workdir = workdir

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""
        return self._sandbox.sandbox_id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a command in the sandbox and return `ExecuteResponse`.

        Args:
            command: Full shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If None, uses the backend's default timeout.

                Note that in E2B's implementation, a timeout of 0 means
                "wait indefinitely".

        Returns:
            `ExecuteResponse` with combined output, exit code, and truncation flag.

        Raises:
            ValueError: If `timeout` is negative.
        """
        if not command:
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout < 0:
            msg = f"timeout must be non-negative, got {effective_timeout}"
            raise ValueError(msg)

        from e2b.exceptions import TimeoutException
        from e2b.sandbox.commands.command_handle import CommandExitException

        try:
            result = self._sandbox.commands.run(
                command,
                cwd=self._workdir,
                timeout=effective_timeout,
            )
        except CommandExitException as exc:
            return ExecuteResponse(
                output=_combine_output(exc.stdout, exc.stderr),
                exit_code=exc.exit_code,
                truncated=False,
            )
        except TimeoutException:
            if timeout is not None:
                msg = (
                    "Error: Command timed out after "
                    f"{effective_timeout} seconds (custom timeout). "
                    "The command may be stuck or require more time."
                )
            else:
                msg = (
                    "Error: Command timed out after "
                    f"{effective_timeout} seconds. For long-running commands, "
                    "re-run using the timeout parameter."
                )
            return ExecuteResponse(output=msg, exit_code=124, truncated=False)
        except Exception as exc:  # noqa: BLE001
            return ExecuteResponse(
                output=f"Error executing command ({type(exc).__name__}): {exc}",
                exit_code=1,
                truncated=False,
            )

        return ExecuteResponse(
            output=_combine_output(result.stdout, result.stderr),
            exit_code=result.exit_code,
            truncated=False,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the E2B sandbox.

        Args:
            paths: Absolute sandbox file paths to fetch.

        Returns:
            Download responses in the same order as `paths`.
        """
        from e2b.exceptions import InvalidArgumentException, NotFoundException
        from e2b.sandbox.filesystem.filesystem import FileType

        responses: list[FileDownloadResponse] = []
        for path in paths:
            if _is_invalid_path(path):
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="invalid_path")
                )
                continue

            try:
                info = self._sandbox.files.get_info(path)
                if info.type == FileType.DIR:
                    responses.append(
                        FileDownloadResponse(
                            path=path,
                            content=None,
                            error="is_directory",
                        )
                    )
                    continue
                content = bytes(self._sandbox.files.read(path, format="bytes"))
            except NotFoundException:
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error="file_not_found",
                    )
                )
                continue
            except InvalidArgumentException:
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="invalid_path")
                )
                continue
            except PermissionError:
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error="permission_denied",
                    )
                )
                continue
            except Exception as exc:  # noqa: BLE001
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error=_map_error_message(str(exc), default="invalid_path"),
                    )
                )
                continue

            responses.append(
                FileDownloadResponse(path=path, content=content, error=None)
            )

        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the E2B sandbox.

        Args:
            files: `(path, content)` pairs to write.

        Returns:
            Upload responses in the same order as `files`.
        """
        from e2b.exceptions import InvalidArgumentException, NotFoundException
        from e2b.sandbox.filesystem.filesystem import FileType

        responses: list[FileUploadResponse] = []
        for path, content in files:
            if _is_invalid_path(path):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue

            try:
                info = self._sandbox.files.get_info(path)
            except NotFoundException:
                info = None
            except InvalidArgumentException:
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            except PermissionError:
                responses.append(
                    FileUploadResponse(path=path, error="permission_denied")
                )
                continue
            except Exception as exc:  # noqa: BLE001
                responses.append(
                    FileUploadResponse(
                        path=path,
                        error=_map_error_message(str(exc), default="invalid_path"),
                    )
                )
                continue

            if info is not None and info.type == FileType.DIR:
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue

            try:
                self._sandbox.files.write(path, content)
            except InvalidArgumentException:
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            except PermissionError:
                responses.append(
                    FileUploadResponse(path=path, error="permission_denied")
                )
                continue
            except Exception as exc:  # noqa: BLE001
                responses.append(
                    FileUploadResponse(
                        path=path,
                        error=_map_error_message(str(exc), default="invalid_path"),
                    )
                )
                continue

            responses.append(FileUploadResponse(path=path, error=None))

        return responses


class E2BProvider(SandboxProvider):
    """E2B sandbox provider implementation."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the E2B provider.

        Args:
            api_key: E2B API key (defaults to `E2B_API_KEY`).

        Raises:
            ValueError: If `E2B_API_KEY` is not configured.
        """
        self._api_key = api_key or os.environ.get("E2B_API_KEY")
        if not self._api_key:
            msg = "E2B_API_KEY environment variable not set"
            raise ValueError(msg)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = DEFAULT_STARTUP_TIMEOUT,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Get an existing sandbox or create a new one.

        Args:
            sandbox_id: Existing sandbox ID to connect to.
            timeout: Startup/request timeout in seconds.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Connected E2B backend.

        Raises:
            RuntimeError: If the sandbox cannot be started or validated.
            SandboxNotFoundError: If an existing sandbox ID cannot be resolved.
            TypeError: If unsupported keyword arguments are provided.
        """
        from e2b import Sandbox
        from e2b.exceptions import NotFoundException

        if kwargs:
            msg = f"Received unsupported arguments: {list(kwargs.keys())}"
            raise TypeError(msg)

        if sandbox_id:
            try:
                sandbox = Sandbox.connect(
                    sandbox_id,
                    timeout=DEFAULT_SANDBOX_LIFETIME,
                    request_timeout=timeout,
                    api_key=self._api_key,
                )
            except NotFoundException as exc:
                raise SandboxNotFoundError(sandbox_id) from exc
            except Exception as exc:
                msg = f"Failed to connect to E2B sandbox '{sandbox_id}': {exc}"
                raise RuntimeError(msg) from exc
        else:
            try:
                sandbox = Sandbox.create(
                    timeout=DEFAULT_SANDBOX_LIFETIME,
                    request_timeout=timeout,
                    api_key=self._api_key,
                )
            except Exception as exc:
                msg = f"Failed to create E2B sandbox: {exc}"
                raise RuntimeError(msg) from exc

        self._wait_until_ready(sandbox=sandbox, timeout=timeout)
        backend = E2BBackend(sandbox)
        self._validate_runtime_tools(backend)
        return backend

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002
        """Delete an E2B sandbox.

        Args:
            sandbox_id: E2B sandbox ID.
            **kwargs: Additional provider-specific parameters.

        Raises:
            SandboxNotFoundError: If the sandbox no longer exists.
        """
        from e2b import Sandbox
        from e2b.exceptions import NotFoundException

        try:
            Sandbox.kill(sandbox_id, api_key=self._api_key)
        except NotFoundException as exc:
            raise SandboxNotFoundError(sandbox_id) from exc

    @staticmethod
    def _wait_until_ready(*, sandbox: Sandbox, timeout: int) -> None:
        """Poll the sandbox until it accepts commands.

        Args:
            sandbox: Sandbox being initialized.
            timeout: Startup budget in seconds.

        Raises:
            RuntimeError: If the sandbox never becomes command-ready.
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                result = sandbox.commands.run(
                    "echo ready",
                    cwd=DEFAULT_WORKDIR,
                    timeout=5,
                    request_timeout=5,
                )
                if result.exit_code == 0 and result.stdout.strip() == "ready":
                    return
            except Exception:  # noqa: S110, BLE001  # Sandbox may still be starting up
                pass

            if time.monotonic() >= deadline:
                sandbox.kill()
                msg = f"E2B sandbox failed to start within {timeout} seconds"
                raise RuntimeError(msg)

            time.sleep(READINESS_POLL_INTERVAL)

    @staticmethod
    def _validate_runtime_tools(backend: E2BBackend) -> None:
        """Fail early if the selected sandbox image is not BaseSandbox compatible.

        Args:
            backend: Connected E2B backend.

        Raises:
            RuntimeError: If `bash` or `python3` is missing.
        """
        result = backend.execute(
            "command -v bash >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1",
            timeout=5,
        )
        if result.exit_code == 0:
            return

        msg = (
            "E2B sandbox is missing required tools (`bash` and `python3`). "
            "Deep Agents' BaseSandbox helpers rely on both. "
            "Use an E2B template that includes them."
        )
        raise RuntimeError(msg)
