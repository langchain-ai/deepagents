"""Runloop sandbox implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runloop_api_client.sdk import Devbox

from deepagents.backends.protocol import (
    FILE_NOT_FOUND,
    INVALID_PATH,
    IS_DIRECTORY,
    PERMISSION_DENIED,
    ExecuteResponse,
    FileDownloadResponse,
    FileOperationError,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox


class RunloopSandbox(BaseSandbox):
    """Sandbox backend that operates on a Runloop devbox."""

    def __init__(
        self,
        *,
        devbox: Devbox,
    ) -> None:
        """Create a sandbox backend connected to an existing Runloop devbox."""
        self._devbox = devbox
        self._devbox_id = devbox.id
        self._default_timeout = 30 * 60

    @property
    def id(self) -> str:
        """Return the devbox id."""
        return self._devbox_id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Execute a shell command inside the devbox.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for this command.

                If None, uses the backend's default timeout.

        Returns:
            ExecuteResponse containing output, exit code, and truncation flag.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        result = self._devbox.cmd.exec(command, timeout=effective_timeout)

        output = result.stdout() if result.stdout() is not None else ""
        stderr = result.stderr() if result.stderr() is not None else ""
        if stderr:
            output += "\n" + stderr if output else stderr

        return ExecuteResponse(
            output=output,
            exit_code=result.exit_code,
            truncated=False,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the devbox."""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(
                    FileDownloadResponse(path=path, content=None, error=INVALID_PATH)
                )
                continue
            try:
                content = self._devbox.file.download(path=path)
            except Exception as exc:  # noqa: BLE001  # Provider exceptions vary by SDK version
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error=_map_file_error(exc),
                    )
                )
                continue
            responses.append(
                FileDownloadResponse(path=path, content=content, error=None)
            )
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the devbox."""
        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error=INVALID_PATH))
                continue
            try:
                self._devbox.file.upload(path=path, file=content)
            except Exception as exc:  # noqa: BLE001  # Provider exceptions vary by SDK version
                responses.append(
                    FileUploadResponse(path=path, error=_map_file_error(exc))
                )
                continue
            responses.append(FileUploadResponse(path=path, error=None))
        return responses


def _map_file_error(exc: Exception) -> FileOperationError | str:
    """Map a provider filesystem failure to a Deep Agents file error.

    Recognized failures map to a ``FileOperationError`` literal. Unrecognized
    exceptions return their string representation rather than defaulting to
    ``FILE_NOT_FOUND``, so that auth, network, or transient SDK failures are
    surfaced to the agent instead of masquerading as a missing file.
    """
    if isinstance(exc, PermissionError):
        return PERMISSION_DENIED
    if isinstance(exc, IsADirectoryError):
        return IS_DIRECTORY
    if isinstance(exc, FileNotFoundError):
        return FILE_NOT_FOUND

    message = str(exc).lower()
    substring_errors: tuple[tuple[tuple[str, ...], FileOperationError], ...] = (
        (("permission", "forbidden", "access denied"), PERMISSION_DENIED),
        (("is a directory",), IS_DIRECTORY),
        (("invalid path",), INVALID_PATH),
        (("no such file",), FILE_NOT_FOUND),
    )
    for needles, error in substring_errors:
        if any(needle in message for needle in needles):
            return error
    return str(exc) or FILE_NOT_FOUND
