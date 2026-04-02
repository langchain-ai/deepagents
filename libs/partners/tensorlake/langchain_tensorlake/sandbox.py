"""Tensorlake sandbox backend implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    WriteResult,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    from tensorlake.sandbox import Sandbox as TensorlakeSandboxClient
    from tensorlake.sandbox.exceptions import SandboxError as TensorlakeSandboxError

logger = logging.getLogger(__name__)


class TensorlakeSandbox(BaseSandbox):
    """Tensorlake sandbox implementation conforming to SandboxBackendProtocol."""

    def __init__(
        self,
        sandbox: "TensorlakeSandboxClient",
        *,
        timeout: int = 30 * 60,
    ) -> None:
        """Create a backend wrapping an existing Tensorlake sandbox."""
        self._sandbox = sandbox
        self._default_timeout = timeout

    @property
    def id(self) -> str:
        """Return the sandbox id."""
        return self._sandbox.sandbox_id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox."""
        effective_timeout = timeout if timeout is not None else self._default_timeout

        # Tensorlake sandbox expects `run()` to be called with an executable path and
        # optional args. The test command can be compound shell syntax (&&, |, etc.),
        # so run it through a shell interpreter.
        shell_command = "/bin/sh"
        shell_args = ["-c", command]

        result = self._sandbox.run(
            shell_command,
            args=shell_args,
            timeout=effective_timeout,
        )

        output = result.stdout or ""
        if result.stderr:
            output = f"{output}\n{result.stderr}" if output else result.stderr

        return ExecuteResponse(
            output=output,
            exit_code=result.exit_code,
            truncated=False,
        )

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write file contents through Tensorlake native write_file."""
        from tensorlake.sandbox.exceptions import SandboxError as TensorlakeSandboxError

        try:
            self._sandbox.write_file(file_path, content.encode("utf-8"))
            return WriteResult(path=file_path)
        except TensorlakeSandboxError as exc:
            return WriteResult(error=f"Failed to write file '{file_path}': {exc}")

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read file content - returns string for test compatibility.

        Uses execute() to run Python scripts that read and return file contents.
        """
        import base64

        # Check if file is readable by running a test cat
        test_result = self.execute(f"cat {file_path}")
        if test_result.exit_code != 0:
            return f"Error reading {file_path}"

        # Read file as base64 to handle binary safely
        read_script = f"""python3 -c "
import base64
try:
    with open('{file_path}', 'rb') as f:
        content = f.read()
    print(base64.b64encode(content).decode('ascii'))
except Exception as e:
    print(f'ERROR:{{str(e)}}')
" """
        result = self.execute(read_script)
        output = result.output.strip()

        if output.startswith("ERROR:"):
            return f"Error: {output[6:]}"

        try:
            raw = base64.b64decode(output.encode('ascii'))
        except Exception:
            return "Error: failed to decode file"

        # Try to decode as text
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            # Binary file
            return base64.b64encode(raw).decode('ascii')

        # Apply offset/limit for text files
        lines = text.splitlines()
        if offset >= len(lines):
            return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"
        selected_lines = lines[offset:offset + limit]
        return '\n'.join(selected_lines)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox.

        Checks file existence and permissions, then reads files via execute().
        """
        import base64

        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(FileDownloadResponse(path=path, content=None, error="invalid_path"))
                continue

            # Read file with Python script — check existence and mode bits explicitly.
            # Using mode bits (stat) rather than os.access() so the check is correct
            # even when the sandbox process runs as root (root bypasses access checks).
            read_script = f"""python3 -c "
import os, base64, stat
try:
    st = os.stat('{path}')
    # Treat as permission_denied when no read bit is set for anyone
    readable = bool(st.st_mode & (stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH))
    if not readable:
        print('PERMISSION_DENIED')
    else:
        with open('{path}', 'rb') as f:
            content = f.read()
        print(base64.b64encode(content).decode('ascii'))
except FileNotFoundError:
    print('FILE_NOT_FOUND')
except PermissionError:
    print('PERMISSION_DENIED')
except Exception as e:
    print(f'ERROR:{{str(e)}}')
" """
            result = self.execute(read_script)
            output = result.output.strip()

            if output == "FILE_NOT_FOUND":
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
            elif output == "PERMISSION_DENIED":
                responses.append(FileDownloadResponse(path=path, content=None, error="permission_denied"))
            elif output.startswith("ERROR:"):
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
            else:
                try:
                    content = base64.b64decode(output.encode('ascii'))
                    responses.append(FileDownloadResponse(path=path, content=content, error=None))
                except Exception:
                    responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))

        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files to the Tensorlake sandbox."""
        from tensorlake.sandbox.exceptions import SandboxError as TensorlakeSandboxError

        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            try:
                self._sandbox.write_file(path, content)
                responses.append(FileUploadResponse(path=path, error=None))
            except TensorlakeSandboxError as exc:
                msg = str(exc).lower()
                if "permission" in msg:
                    error = "permission_denied"
                else:
                    error = "permission_denied"
                logger.warning("Tensorlake upload failed for %s: %s", path, exc)
                responses.append(FileUploadResponse(path=path, error=error))
        return responses
