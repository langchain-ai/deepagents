"""Tensorlake sandbox backend implementation."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

from tensorlake.sandbox.exceptions import SandboxError as TensorlakeSandboxError

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileData,
    FileDownloadResponse,
    FileUploadResponse,
    ReadResult,
    WriteResult,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    from tensorlake.sandbox import Sandbox as TensorlakeSandboxClient

logger = logging.getLogger(__name__)


class TensorlakeSandbox(BaseSandbox):
    """Tensorlake sandbox implementation conforming to SandboxBackendProtocol."""

    def __init__(
        self,
        sandbox: TensorlakeSandboxClient,
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

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read file content using the native read_file API."""
        try:
            result = self._sandbox.read_file(file_path)
            raw: bytes = result.value if hasattr(result, "value") else result

            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                return ReadResult(
                    file_data=FileData(
                        content=base64.b64encode(raw).decode("ascii"),
                        encoding="base64",
                    )
                )

            lines = text.splitlines()
            if offset >= len(lines):
                return ReadResult(error=f"Line offset {offset} exceeds file length ({len(lines)} lines)")
            selected_lines = lines[offset : offset + limit]
            return ReadResult(
                file_data=FileData(
                    content="\n".join(selected_lines),
                    encoding="utf-8",
                )
            )
        except TensorlakeSandboxError as exc:
            return ReadResult(error=f"File '{file_path}': {exc}")

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox."""
        effective_timeout = timeout if timeout is not None else self._default_timeout

        # Tensorlake requires an executable path + args; map shell-like
        # command strings to /bin/sh -c for compatibility with test harness.
        result = self._sandbox.run(
            "/bin/sh",
            args=["-c", command],
            timeout=effective_timeout,
        )

        output = result.stdout or ""
        if result.stderr:
            output = f"{output}\n{result.stderr}" if output else result.stderr

        return ExecuteResponse(output=output, exit_code=result.exit_code, truncated=False)

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write file contents through Tensorlake native write_file."""
        try:
            self._sandbox.write_file(file_path, content.encode("utf-8"))
            return WriteResult(path=file_path)
        except TensorlakeSandboxError as exc:
            return WriteResult(error=f"Failed to write file '{file_path}': {exc}")

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox using the native read_file API."""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(FileDownloadResponse(path=path, content=None, error="invalid_path"))
                continue
            try:
                raw = self._sandbox.read_file(path)
                content: bytes = raw.value if hasattr(raw, "value") else raw
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
            except TensorlakeSandboxError as exc:
                error_msg = str(exc).lower()
                error = "permission_denied" if "permission" in error_msg else "file_not_found"
                responses.append(FileDownloadResponse(path=path, content=None, error=error))

        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files to the Tensorlake sandbox."""
        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            try:
                self._sandbox.write_file(path, content)
                responses.append(FileUploadResponse(path=path, error=None))
            except TensorlakeSandboxError as exc:
                logger.warning("Tensorlake upload failed for %s: %s", path, exc)
                responses.append(FileUploadResponse(path=path, error="permission_denied"))

        return responses
