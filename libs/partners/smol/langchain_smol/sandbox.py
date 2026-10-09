"""Smol Machines sandbox backend for Deep Agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

import smol
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    from smol import Machine

_DEFAULT_TIMEOUT = 30 * 60


def _file_error(error: smol.SmolError) -> str:
    """Normalize Cloud HTTP and local guest file errors for Deep Agents."""
    message = str(error).lower()
    if error.code == "NOT_FOUND" or "no such file or directory (os error 2)" in message:
        return "file_not_found"
    if "permission denied (os error 13)" in message:
        return "permission_denied"
    if "is a directory (os error 21)" in message:
        return "is_directory"
    return error.code.lower()


class SmolSandbox(BaseSandbox):
    """Run Deep Agents tools inside an existing local or Cloud microVM.

    The caller owns the VM's lifecycle; call `machine.delete()` when finished.
    The image must contain `sh` and `python3` for Deep Agents file tools.
    """

    def __init__(self, *, machine: Machine, timeout: int = _DEFAULT_TIMEOUT) -> None:
        """Wrap a running machine.

        Args:
            machine: A ready local or Cloud Smol Machines VM.
            timeout: Default execution timeout in seconds.
        """
        if timeout <= 0:
            msg = "timeout must be greater than zero"
            raise ValueError(msg)
        self._machine = machine
        self._default_timeout = timeout

    @property
    def id(self) -> str:
        """Return the Smol Machines VM ID."""
        return self._machine.id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run a shell command in the VM with a bounded execution time."""
        effective_timeout = self._default_timeout if timeout is None else timeout
        if effective_timeout <= 0:
            msg = "timeout must be greater than zero"
            raise ValueError(msg)
        result = self._machine.exec(
            ["sh", "-lc", command], smol.ExecOptions(timeout=effective_timeout)
        )
        output = result.stdout
        if result.stderr:
            output += (
                f"\n<stderr>{result.stderr}</stderr>"
                if output
                else (f"<stderr>{result.stderr}</stderr>")
            )
        return ExecuteResponse(
            output=output,
            exit_code=result.exit_code,
            truncated=result.stdout_truncated or result.stderr_truncated,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Read byte-exact file contents from the VM."""
        return [self._download_file(path) for path in paths]

    def _download_file(self, path: str) -> FileDownloadResponse:
        if not path.startswith("/"):
            return FileDownloadResponse(path=path, content=None, error="invalid_path")
        try:
            return FileDownloadResponse(
                path=path, content=self._machine.read_file(path), error=None
            )
        except smol.SmolError as exc:
            return FileDownloadResponse(path=path, content=None, error=_file_error(exc))

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Write byte-exact file contents into the VM."""
        return [self._upload_file(path, data) for path, data in files]

    def _upload_file(self, path: str, data: bytes) -> FileUploadResponse:
        if not path.startswith("/"):
            return FileUploadResponse(path=path, error="invalid_path")
        try:
            self._machine.write_file(path, data)
        except smol.SmolError as exc:
            return FileUploadResponse(path=path, error=_file_error(exc))
        return FileUploadResponse(path=path, error=None)
