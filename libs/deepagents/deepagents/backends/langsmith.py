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
from deepagents.backends.sandbox import (
    MAX_BINARY_BYTES,
    MAX_OUTPUT_BYTES,
    TRUNCATION_MSG,
    BaseSandbox,
)
from deepagents.backends.utils import _get_file_type

if TYPE_CHECKING:
    from langsmith.sandbox import Sandbox

logger = logging.getLogger(__name__)


def _binary_read_result(file_path: str, raw: bytes) -> ReadResult:
    """Build the binary `ReadResult` shape used by `LangSmithSandbox.read()`.

    Mirrors the `error` / `encoding=base64` outputs produced server-side by
    `_READ_COMMAND_TEMPLATE` in `sandbox.py`, including the `File '<path>': `
    prefix that `BaseSandbox.read()` adds when it wraps script errors.
    """
    if len(raw) > MAX_BINARY_BYTES:
        return ReadResult(error=(f"File '{file_path}': Binary file exceeds maximum preview size of {MAX_BINARY_BYTES} bytes"))
    return ReadResult(
        file_data=FileData(
            content=base64.b64encode(raw).decode("ascii"),
            encoding="base64",
        )
    )


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
        the same existence check and parent-directory creation as
        `BaseSandbox.write()`.

        Args:
            file_path: Destination path inside the sandbox.
            content: Text content to write.

        Returns:
            `WriteResult` with the written path on success, or an error message.
        """
        from langsmith.sandbox import SandboxClientError  # noqa: PLC0415

        preflight_error = self._write_preflight(file_path)
        if preflight_error is not None:
            return preflight_error

        try:
            self._sandbox.write(file_path, content.encode("utf-8"))
            return WriteResult(path=file_path)
        except SandboxClientError as e:
            return WriteResult(error=f"Failed to write file '{file_path}': {e}")

    def read(  # noqa: PLR0911 - early returns for distinct error conditions
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        r"""Read file content using the LangSmith SDK.

        `BaseSandbox.read()` pipes file content through `execute()`, which
        can hang or exceed transport limits for large files. This override
        fetches bytes directly via the SDK and reproduces the base-class
        pagination semantics locally:

        - Empty files surface the "empty contents" reminder.
        - Files routed as binary by extension (or that fail UTF-8 decode) are
            returned base64-encoded, capped at `MAX_BINARY_BYTES`.
        - Text content is normalized for universal newlines (`\r\n` and bare
            `\r` collapse to `\n`), split on `\n`, paginated by `offset` /
            `limit`, joined back with `\n`, and capped at `MAX_OUTPUT_BYTES`
            with `TRUNCATION_MSG` appended on overflow.

        Args:
            file_path: Absolute path to the file to read.
            offset: Number of leading text lines to skip.
            limit: Maximum number of text lines to return.

        Returns:
            `ReadResult` with `file_data` on success or `error` on failure.
        """
        from langsmith.sandbox import ResourceNotFoundError, SandboxClientError  # noqa: PLC0415

        try:
            raw = self._sandbox.read(file_path)
        except ResourceNotFoundError:
            return ReadResult(error=f"File '{file_path}': file_not_found")
        except SandboxClientError as e:
            logger.warning("LangSmith read failed for %s: %s", file_path, e)
            return ReadResult(error=f"File '{file_path}': {type(e).__name__}: {e}")

        if not raw:
            return ReadResult(
                file_data=FileData(
                    content="System reminder: File exists but has empty contents",
                    encoding="utf-8",
                )
            )

        # Route by extension first, mirroring _READ_COMMAND_TEMPLATE: anything
        # not classified as text goes straight to base64 without a decode
        # attempt.
        if _get_file_type(file_path) != "text":
            return _binary_read_result(file_path, raw)

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            # Text-by-extension file with non-UTF-8 bytes: fall back to base64
            # rather than guessing an encoding. Log so a corrupted file or
            # mis-named extension is observable rather than silently reshaped.
            logger.info(
                "Text-extension file %s contained invalid UTF-8; returning as base64",
                file_path,
            )
            return _binary_read_result(file_path, raw)

        # Universal-newline normalization to match `open(..., newline=None)` +
        # `rstrip('\n').rstrip('\r')` in _READ_COMMAND_TEMPLATE: \r\n and bare
        # \r both collapse to \n. Without this, CRLF files round-trip with
        # stray \r in returned content, which then breaks `edit()` (issue
        # #2880).
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized.split("\n")
        if lines and lines[-1] == "":
            lines.pop()

        offset = int(offset)
        limit = int(limit)

        if not lines or offset >= len(lines):
            return ReadResult(error=f"File '{file_path}': Line offset {offset} exceeds file length ({len(lines)} lines)")

        page = lines[offset : offset + limit]
        content = "\n".join(page)

        # Cap rendered text at MAX_OUTPUT_BYTES and append TRUNCATION_MSG, so
        # large pages don't reintroduce the transport-size symptom this
        # override fixes.
        encoded = content.encode("utf-8")
        msg_bytes = TRUNCATION_MSG.encode("utf-8")
        effective_limit = MAX_OUTPUT_BYTES - len(msg_bytes)
        if len(encoded) > effective_limit:
            content = encoded[:effective_limit].decode("utf-8", errors="ignore") + TRUNCATION_MSG

        return ReadResult(file_data=FileData(content=content, encoding="utf-8"))

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
