"""LangSmith sandbox backend implementation."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileData,
    FileDownloadResponse,
    FileUploadResponse,
    ReadResult,
    WriteResult,
)
from deepagents.backends.sandbox import _WRITE_CHECK_TEMPLATE, BaseSandbox
from deepagents.backends.utils import _get_file_type

if TYPE_CHECKING:
    from langsmith.sandbox import Sandbox

logger = logging.getLogger(__name__)


class LangSmithSandbox(BaseSandbox):
    """LangSmith sandbox implementation conforming to `SandboxBackendProtocol`."""

    def __init__(self, sandbox: Sandbox) -> None:
        """Create a backend wrapping an existing LangSmith sandbox.

        Args:
            sandbox: LangSmith Sandbox instance to wrap.
        """
        self._sandbox = sandbox
        self._default_timeout: int = 30 * 60

    @property
    def id(self) -> str:
        """Return the LangSmith sandbox name."""
        return self._sandbox.name

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Execute a shell command inside the sandbox.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If None, uses the backend's default timeout.
                A value of 0 disables the command timeout when the
                `langsmith[sandbox]` extra is installed.

        Returns:
            `ExecuteResponse` containing output, exit code, and truncation flag.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        result = self._sandbox.run(command, timeout=effective_timeout)

        output = result.stdout or ""
        if result.stderr:
            output += "\n" + result.stderr if output else result.stderr

        return ExecuteResponse(
            output=output,
            exit_code=result.exit_code,
            truncated=False,
        )

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write content using the LangSmith SDK to avoid ARG_MAX.

        `BaseSandbox.write()` sends the full content in a shell command, which
        can exceed ARG_MAX for large content. This override uses the SDK's
        native `write()`, which sends content in the HTTP body, but preserves
        the same existence-check preflight as `BaseSandbox.write()`.

        Args:
            file_path: Destination path inside the sandbox.
            content: Text content to write.

        Returns:
            `WriteResult` with the written path on success, or an error message.
        """
        from langsmith.sandbox import SandboxClientError  # noqa: PLC0415

        # Existence check + mkdir (same preflight as BaseSandbox.write)
        path_b64 = base64.b64encode(file_path.encode("utf-8")).decode("ascii")
        check_cmd = _WRITE_CHECK_TEMPLATE.format(path_b64=path_b64)
        check = self.execute(check_cmd)
        if check.exit_code != 0 or "Error:" in check.output:
            error_msg = check.output.strip() or f"Failed to write file '{file_path}'"
            return WriteResult(error=error_msg)

        try:
            self._sandbox.write(file_path, content.encode("utf-8"))
            return WriteResult(path=file_path)
        except SandboxClientError as e:
            return WriteResult(error=f"Failed to write file '{file_path}': {e}")

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read file content using the LangSmith SDK.

        `BaseSandbox.read()` pipes file content through `execute()`, which
        can hang or exceed transport limits for large files. This override
        uses the SDK's native `read()` to fetch bytes directly and applies
        line-based pagination locally.

        Args:
            file_path: Absolute path to the file to read.
            offset: Starting line number (0-indexed).
            limit: Maximum number of lines to return.

        Returns:
            `ReadResult` with `file_data` on success or `error` on failure.
        """
        from langsmith.sandbox import ResourceNotFoundError, SandboxClientError  # noqa: PLC0415

        try:
            raw = self._sandbox.read(file_path)
        except (ResourceNotFoundError, SandboxClientError) as e:
            error = "file_not_found" if isinstance(e, ResourceNotFoundError) else str(e)
            return ReadResult(error=f"File '{file_path}': {error}")

        if not raw:
            return ReadResult(
                file_data=FileData(
                    content="System reminder: File exists but has empty contents",
                    encoding="utf-8",
                )
            )

        # Determine whether content is decodable text
        is_text = _get_file_type(file_path) == "text"
        text: str | None = None
        if is_text:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                is_text = False

        if not is_text:
            max_binary = 500 * 1024  # must match MAX_BINARY_BYTES in _READ_COMMAND_TEMPLATE
            if len(raw) > max_binary:
                return ReadResult(error=f"Binary file exceeds maximum preview size of {max_binary} bytes")
            return ReadResult(file_data=FileData(content=base64.b64encode(raw).decode("ascii"), encoding="base64"))

        # Line-based pagination matching _READ_COMMAND_TEMPLATE behavior:
        # split on \n, trailing \n does not produce an extra line.
        assert text is not None  # noqa: S101 — narrowing: is_text=True and decode succeeded
        lines = text.split("\n")
        if lines and lines[-1] == "" and text.endswith("\n"):
            lines.pop()

        offset = int(offset)
        limit = int(limit)

        if offset >= len(lines) and lines:
            return ReadResult(error=f"File '{file_path}': Line offset {offset} exceeds file length ({len(lines)} lines)")

        page = lines[offset : offset + limit]
        return ReadResult(file_data=FileData(content="\n".join(page), encoding="utf-8"))

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the LangSmith sandbox.

        Supports partial success -- individual downloads may fail without
        affecting others.

        Args:
            paths: List of file paths to download.

        Returns:
            List of `FileDownloadResponse` objects, one per input path.

                Response order matches input order.
        """
        from langsmith.sandbox import ResourceNotFoundError, SandboxClientError  # noqa: PLC0415

        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(FileDownloadResponse(path=path, content=None, error="invalid_path"))
                continue
            try:
                content = self._sandbox.read(path)
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
            except ResourceNotFoundError:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
            except SandboxClientError as e:
                msg = str(e).lower()
                error = "is_directory" if "is a directory" in msg else "file_not_found"
                responses.append(FileDownloadResponse(path=path, content=None, error=error))
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the LangSmith sandbox.

        Supports partial success -- individual uploads may fail without
        affecting others.

        Args:
            files: List of `(path, content)` tuples to upload.

        Returns:
            List of `FileUploadResponse` objects, one per input file.

                Response order matches input order.
        """
        from langsmith.sandbox import SandboxClientError  # noqa: PLC0415

        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            try:
                self._sandbox.write(path, content)
                responses.append(FileUploadResponse(path=path, error=None))
            except SandboxClientError as e:
                logger.debug("Failed to upload %s: %s", path, e)
                responses.append(FileUploadResponse(path=path, error="permission_denied"))
        return responses
