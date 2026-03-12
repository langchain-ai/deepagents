"""E2B sandbox backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileOperationError,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from e2b.exceptions import InvalidArgumentException, NotFoundException, TimeoutException
from e2b.sandbox.commands.command_handle import CommandExitException
from e2b.sandbox.filesystem.filesystem import FileType

if TYPE_CHECKING:
    from e2b import Sandbox

DEFAULT_COMMAND_TIMEOUT = 30 * 60
DEFAULT_WORKDIR = "/home/user"


def _combine_output(stdout: str | None, stderr: str | None) -> str:
    """Combine stdout and stderr into a single output string.

    Args:
        stdout: Command standard output.
        stderr: Command standard error.

    Returns:
        Combined output string.
    """
    output = stdout or ""
    if stderr:
        output += "\n" + stderr if output else stderr
    return output or "<no output>"


def _is_invalid_path(path: str) -> bool:
    """Return whether a path is malformed for sandbox file operations.

    Args:
        path: Candidate sandbox path.

    Returns:
        `True` when the path is malformed for file operations.
    """
    return not path.startswith("/") or "\x00" in path


def _map_error_message(
    message: str, *, default: FileOperationError
) -> FileOperationError:
    """Map provider error messages to standardized file operation errors.

    Args:
        message: Provider error message.
        default: Fallback error code.

    Returns:
        Normalized file operation error code.
    """
    lowered = message.lower()
    if "permission" in lowered:
        return "permission_denied"
    if "directory" in lowered:
        return "is_directory"
    if "not found" in lowered or "no such file" in lowered:
        return "file_not_found"
    return default


def _download_error(path: str, error: FileOperationError) -> FileDownloadResponse:
    """Create a failed download response.

    Args:
        path: Requested file path.
        error: Standardized error code.

    Returns:
        Failed `FileDownloadResponse`.
    """
    return FileDownloadResponse(path=path, content=None, error=error)


def _upload_error(path: str, error: FileOperationError) -> FileUploadResponse:
    """Create a failed upload response.

    Args:
        path: Target file path.
        error: Standardized error code.

    Returns:
        Failed `FileUploadResponse`.
    """
    return FileUploadResponse(path=path, error=error)


def _get_existing_file_type(
    sandbox: Sandbox, path: str
) -> FileType | None | FileUploadResponse:
    """Get the current file type for an upload target.

    Args:
        sandbox: Connected E2B sandbox.
        path: Candidate upload path.

    Returns:
        Existing `FileType`, `None` if the path does not exist, or a failed
        `FileUploadResponse` when the path cannot be inspected.
    """
    try:
        info = sandbox.files.get_info(path)
    except NotFoundException:
        return None
    except InvalidArgumentException:
        return _upload_error(path, "invalid_path")
    except PermissionError:
        return _upload_error(path, "permission_denied")
    except Exception as exc:  # noqa: BLE001
        return _upload_error(
            path,
            _map_error_message(str(exc), default="invalid_path"),
        )
    return info.type


class E2BSandbox(BaseSandbox):
    """E2B sandbox implementation conforming to SandboxBackendProtocol."""

    def __init__(self, *, sandbox: Sandbox, workdir: str = DEFAULT_WORKDIR) -> None:
        """Create a backend wrapping an existing E2B sandbox.

        Args:
            sandbox: Connected E2B sandbox instance.
            workdir: Working directory used for shell command execution.
        """
        self._sandbox = sandbox
        self._default_timeout = DEFAULT_COMMAND_TIMEOUT
        self._workdir = workdir

    @property
    def id(self) -> str:
        """Return the E2B sandbox id."""
        return self._sandbox.sandbox_id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for this command.

                If None, uses the backend's default timeout.

                Note that in E2B's implementation, a timeout of 0 means
                "wait indefinitely".

        Returns:
            ExecuteResponse containing output, exit code, and truncation flag.

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
            return ExecuteResponse(
                output=f"Error: Command timed out after {effective_timeout} seconds.",
                exit_code=124,
                truncated=False,
            )
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
        """Download files from the sandbox.

        Args:
            paths: Absolute sandbox file paths.

        Returns:
            Download responses in the same order as `paths`.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if _is_invalid_path(path):
                responses.append(_download_error(path, "invalid_path"))
                continue

            try:
                info = self._sandbox.files.get_info(path)
                if info.type == FileType.DIR:
                    responses.append(_download_error(path, "is_directory"))
                    continue
                content = bytes(self._sandbox.files.read(path, format="bytes"))
            except NotFoundException:
                responses.append(_download_error(path, "file_not_found"))
                continue
            except InvalidArgumentException:
                responses.append(_download_error(path, "invalid_path"))
                continue
            except PermissionError:
                responses.append(_download_error(path, "permission_denied"))
                continue
            except Exception as exc:  # noqa: BLE001
                responses.append(
                    _download_error(
                        path,
                        _map_error_message(str(exc), default="invalid_path"),
                    )
                )
                continue

            responses.append(
                FileDownloadResponse(path=path, content=content, error=None)
            )

        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the sandbox.

        Args:
            files: `(path, content)` pairs to write.

        Returns:
            Upload responses in the same order as `files`.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            if _is_invalid_path(path):
                responses.append(_upload_error(path, "invalid_path"))
                continue

            existing_type = _get_existing_file_type(self._sandbox, path)
            if isinstance(existing_type, FileUploadResponse):
                responses.append(existing_type)
                continue

            if existing_type == FileType.DIR:
                responses.append(_upload_error(path, "invalid_path"))
                continue

            try:
                self._sandbox.files.write(path, content)
            except InvalidArgumentException:
                responses.append(_upload_error(path, "invalid_path"))
                continue
            except PermissionError:
                responses.append(_upload_error(path, "permission_denied"))
                continue
            except Exception as exc:  # noqa: BLE001
                responses.append(
                    _upload_error(
                        path=path,
                        error=_map_error_message(str(exc), default="invalid_path"),
                    )
                )
                continue

            responses.append(FileUploadResponse(path=path, error=None))

        return responses
