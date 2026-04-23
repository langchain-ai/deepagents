"""Novita sandbox backend implementation."""

from __future__ import annotations

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from novita_sandbox.code_interpreter import (
    CommandExitException,
    InvalidArgumentException,
    NotFoundException,
    Sandbox,
)


class NovitaSandbox(BaseSandbox):
    """Novita sandbox implementation conforming to SandboxBackendProtocol."""

    def __init__(self, *, sandbox: Sandbox) -> None:
        """Create a backend wrapping an existing Novita sandbox."""
        self._sandbox = sandbox
        self._default_timeout = 30 * 60

    @property
    def id(self) -> str:
        """Return the sandbox id."""
        return self._sandbox.sandbox_id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Execute a shell command inside the sandbox.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If None, uses the backend's default timeout.

                A timeout of 0 means "wait indefinitely".
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        sdk_timeout = None if effective_timeout == 0 else int(effective_timeout)

        try:
            result = self._sandbox.commands.run(command, timeout=sdk_timeout)
            output = result.stdout or ""
            if result.stderr and result.stderr.strip():
                output += f"\n<stderr>{result.stderr.strip()}</stderr>"
            return ExecuteResponse(
                output=output, exit_code=result.exit_code, truncated=False
            )
        except CommandExitException as e:
            output = e.stdout or ""
            if e.stderr and e.stderr.strip():
                output += f"\n<stderr>{e.stderr.strip()}</stderr>"
            return ExecuteResponse(
                output=output, exit_code=e.exit_code, truncated=False
            )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox."""
        results: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                results.append(
                    FileDownloadResponse(path=path, content=None, error="invalid_path")
                )
                continue
            try:
                raw = self._sandbox.files.read(path, format="bytes")
                results.append(
                    FileDownloadResponse(path=path, content=bytes(raw), error=None)
                )
            except NotFoundException:
                results.append(
                    FileDownloadResponse(
                        path=path, content=None, error="file_not_found"
                    )
                )
            except InvalidArgumentException as e:
                msg = str(e).lower()
                error = "is_directory" if "is a directory" in msg else "file_not_found"
                results.append(
                    FileDownloadResponse(path=path, content=None, error=error)
                )
        return results

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the sandbox."""
        results: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                results.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            self._sandbox.files.write(path, content)
            results.append(FileUploadResponse(path=path, error=None))
        return results
