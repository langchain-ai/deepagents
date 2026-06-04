"""E2B sandbox backend implementation."""

from __future__ import annotations

import e2b
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

DEFAULT_WORKDIR = "/home/user"
TIMEOUT_EXIT_CODE = 124


def _combine_output(stdout: str | None, stderr: str | None) -> str:
    output = stdout or ""
    if stderr:
        output += "\n" + stderr if output else stderr
    return output


class E2BSandbox(BaseSandbox):
    """Sandbox backend that operates on an existing E2B sandbox."""

    def __init__(
        self,
        *,
        sandbox: e2b.Sandbox,
        workdir: str = DEFAULT_WORKDIR,
        timeout: int = 30 * 60,
    ) -> None:
        """Create a backend wrapping an existing E2B sandbox.

        Args:
            sandbox: Existing E2B sandbox instance to wrap.
            workdir: Working directory for command execution.
            timeout: Default command timeout in seconds when `execute()` is
                called without an explicit `timeout`.
        """
        self._sandbox = sandbox
        self._workdir = workdir
        self._default_timeout = timeout

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

        Returns:
            ExecuteResponse containing output, exit code, and truncation flag.

        Raises:
            ValueError: If `timeout` is negative.
        """
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
        except e2b.CommandExitException as exc:
            return ExecuteResponse(
                output=_combine_output(exc.stdout, exc.stderr),
                exit_code=exc.exit_code,
                truncated=False,
            )
        except e2b.TimeoutException:
            return ExecuteResponse(
                output=f"Command timed out after {effective_timeout} seconds",
                exit_code=TIMEOUT_EXIT_CODE,
                truncated=False,
            )
        except e2b.SandboxException as exc:
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

    def _read_file(self, path: str) -> FileDownloadResponse:
        if not path.startswith("/"):
            return FileDownloadResponse(path=path, content=None, error="invalid_path")

        try:
            info = self._sandbox.files.get_info(path)
            if info.type == e2b.FileType.DIR:
                return FileDownloadResponse(
                    path=path,
                    content=None,
                    error="is_directory",
                )
            content = bytes(self._sandbox.files.read(path, format="bytes"))
            return FileDownloadResponse(path=path, content=content, error=None)
        except e2b.FileNotFoundException:
            return FileDownloadResponse(path=path, content=None, error="file_not_found")
        except e2b.InvalidArgumentException:
            return FileDownloadResponse(path=path, content=None, error="invalid_path")
        except PermissionError:
            return FileDownloadResponse(
                path=path,
                content=None,
                error="permission_denied",
            )

    def _write_file(self, path: str, content: bytes) -> FileUploadResponse:
        if not path.startswith("/"):
            return FileUploadResponse(path=path, error="invalid_path")

        error: str | None = None
        try:
            info = self._sandbox.files.get_info(path)
            if info.type == e2b.FileType.DIR:
                error = "is_directory"
        except e2b.FileNotFoundException:
            pass
        except e2b.InvalidArgumentException:
            error = "invalid_path"
        except PermissionError:
            error = "permission_denied"

        if error is None:
            try:
                self._sandbox.files.write(path, content)
            except e2b.FileNotFoundException:
                error = "file_not_found"
            except e2b.InvalidArgumentException:
                error = "invalid_path"
            except PermissionError:
                error = "permission_denied"

        return FileUploadResponse(path=path, error=error)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox.

        Args:
            paths: Absolute sandbox file paths to download.

        Returns:
            Download responses in the same order as `paths`.
        """
        return [self._read_file(path) for path in paths]

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the sandbox.

        Args:
            files: `(path, content)` pairs to write.

        Returns:
            Upload responses in the same order as `files`.
        """
        return [self._write_file(path, content) for path, content in files]
