"""LangSmith sandbox backend implementation."""

from __future__ import annotations

import base64
import logging
import zlib
from typing import TYPE_CHECKING, Final

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
from deepagents.backends.utils import (
    MAX_CURSOR_ROWS,
    _chunked_rows,
    _get_backend_read_file_type,
    make_read_cursor,
    normalize_read_bounds,
    parse_read_cursor,
)

_UTF8_2_BYTE: Final = 0xE0
_UTF8_3_BYTE: Final = 0xF0
_ASCII_MAX: Final = 0x80


def _utf8_char_len(data: bytes) -> int:
    """Count characters in a UTF-8-safe byte prefix without decoding tail garbage."""
    count = 0
    index = 0
    while index < len(data):
        byte = data[index]
        width = 1 if byte < _ASCII_MAX else 2 if byte < _UTF8_2_BYTE else 3 if byte < _UTF8_3_BYTE else 4
        if width > 1 and index + width > len(data):
            break
        index += width
        count += 1
    return count


if TYPE_CHECKING:
    from langsmith.sandbox import AsyncSandbox, AsyncSandboxClient, ExecutionResult, Sandbox

logger = logging.getLogger(__name__)


def _execute_response(result: ExecutionResult) -> ExecuteResponse:
    """Build an `ExecuteResponse` from a LangSmith SDK execution result."""
    output = result.stdout or ""
    if result.stderr:
        output += "\n" + result.stderr if output else result.stderr
    return ExecuteResponse(output=output, exit_code=result.exit_code, truncated=False)


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
    """LangSmith sandbox implementation conforming to [`SandboxBackendProtocol`][deepagents.backends.protocol.SandboxBackendProtocol]."""

    # LangSmith sandbox images ship a POSIX shell + coreutils compatible with the
    # capture wrapper, so opt in to capture-at-source offload for `execute`.
    enable_capture_offload = True

    def __init__(self, sandbox: Sandbox) -> None:
        """Create a backend wrapping an existing LangSmith sandbox.

        Args:
            sandbox: LangSmith Sandbox instance to wrap.
        """
        self._sandbox = sandbox
        self._default_timeout: int = 30 * 60
        self._async_sandbox: AsyncSandbox | None = None
        self._async_client: AsyncSandboxClient | None = None

    @property
    def id(self) -> str:
        """Return the LangSmith sandbox name."""
        return self._sandbox.name

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Execute a shell command inside the sandbox.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If `None`, uses the backend's default timeout.

                A value of 0 disables the command timeout when the
                `langsmith[sandbox]` extra is installed.

        Returns:
            `ExecuteResponse` containing output, exit code, and truncation flag.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        return _execute_response(self._sandbox.run(command, timeout=effective_timeout))

    def _aget_sandbox(self) -> AsyncSandbox:
        """Return the cached `AsyncSandbox`, creating it on first async use.

        `Sandbox.to_async()` builds a fresh client with its own connection pool
        on every call, so it is cached: rebuilding per command would add a TCP
        and TLS handshake to each one. The client belongs to the event loop that
        created it, as does this backend — reusing one instance across loops is
        not supported.
        """
        if self._async_sandbox is None:
            # The SDK exposes no public accessor for a sandbox's client, and
            # the async client is held here so `aclose()` can reach its pool.
            self._async_client = self._sandbox._client.to_async()
            self._async_sandbox = self._sandbox.to_async(client=self._async_client)
        return self._async_sandbox

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:  # noqa: ASYNC109
        """Execute a shell command inside the sandbox.

        Overrides the protocol default, which offloads the blocking `execute()`
        to a worker thread. `BaseSandbox` routes every async filesystem
        operation through `aexecute`, so using the SDK's async client here keeps
        all of them off the sync transport.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If `None`, uses the backend's default timeout.

        Returns:
            `ExecuteResponse` containing output, exit code, and truncation flag.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        sandbox = self._aget_sandbox()
        return _execute_response(await sandbox.run(command, timeout=effective_timeout))

    async def aclose(self) -> None:
        """Close the cached async client's connection pool, if one was created."""
        client = self._async_client
        self._async_sandbox = self._async_client = None
        if client is not None:
            await client.aclose()

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

    def read(  # noqa: PLR0911, PLR0912, PLR0915, C901 - early returns for distinct error conditions; the read pipeline's branches stay inline to mirror _READ_COMMAND_TEMPLATE
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        *,
        cursor: str | None = None,
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
            `limit` with `limit` bounding displayed rows of
            `MAX_LINE_LENGTH` characters, and capped at `MAX_OUTPUT_BYTES`
            with `TRUNCATION_MSG` appended on overflow.
        - A page that stops mid-source-line carries a `continuation_cursor`;
            passing it back resumes at the exact character and takes
            precedence over `offset`/`limit`. A negative `offset` is clamped
            to the start of the file, and a non-positive `limit` returns
            empty content with no pagination metadata.

        Args:
            file_path: Absolute path to the file to read.
            offset: Number of leading text lines to skip.
            limit: Maximum number of displayed rows to return.
            cursor: Opaque continuation cursor from a previous read.

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
        if _get_backend_read_file_type(file_path) != "text":
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

        offset, limit = normalize_read_bounds(offset, limit)

        # Nothing was requested: no line range to describe, and nothing for the
        # byte cap below to shorten. Guarded at `<= 0` so this holds even if the
        # clamp above is ever bypassed or removed. `no_lines_requested` flags
        # the window as never inspected so the middleware can tell it apart
        # from a genuinely empty file.
        if limit <= 0 and cursor is None:
            return ReadResult(file_data=FileData(content="", encoding="utf-8"), no_lines_requested=True)

        total_lines = len(lines)
        identity = (len(raw), zlib.crc32(raw) & 0xFFFFFFFF)
        if cursor is not None:
            parsed = parse_read_cursor(cursor, content=normalized, size=identity[0], mtime_ns=identity[1])
            if isinstance(parsed, str):
                return ReadResult(error=f"File '{file_path}': {parsed}")
            start_index, first_char_offset = parsed
            page_lines = lines[start_index : start_index + MAX_CURSOR_ROWS]
            rows, next_cursor = _chunked_rows(page_lines, identity, MAX_CURSOR_ROWS, first_char_offset=first_char_offset)
        else:
            if not lines or offset >= total_lines:
                return ReadResult(error=f"File '{file_path}': Line offset {offset} exceeds file length ({total_lines} lines)")
            start_index = offset
            first_char_offset = 0
            rows, next_cursor = _chunked_rows(lines[offset:], identity, limit)

        # Cap rendered text at MAX_OUTPUT_BYTES and append TRUNCATION_MSG, so
        # large pages don't reintroduce the transport-size symptom this
        # override fixes.
        encoded_rows = [chunk.encode("utf-8") for _, chunk, _ in rows]
        effective_limit = MAX_OUTPUT_BYTES - len(TRUNCATION_MSG.encode("utf-8"))
        current_bytes = 0
        cut_row: int | None = None
        for row_pos, piece in enumerate(encoded_rows):
            separator = 1 if row_pos else 0
            if current_bytes + separator + len(piece) > effective_limit:
                cut_row = row_pos
                break
            current_bytes += separator + len(piece)
        if cut_row is not None:
            remaining = effective_limit - current_bytes - (1 if cut_row else 0)
            line_index, _chunk_text, chunk_offset = rows[cut_row]
            partial_text = encoded_rows[cut_row][:remaining].decode("utf-8", errors="ignore") if remaining > 0 else ""
            shown = _utf8_char_len(partial_text.encode("utf-8"))
            rows = rows[:cut_row]
            if shown:
                rows = [*rows, (line_index, partial_text, chunk_offset)]
            source_line = lines[start_index + line_index]
            resume_char = chunk_offset + shown
            if resume_char < len(source_line):
                next_cursor = make_read_cursor(identity[0], identity[1], start_index + line_index, resume_char, source_line)
            elif start_index + line_index + 1 < total_lines:
                next_cursor = make_read_cursor(identity[0], identity[1], start_index + line_index + 1, 0, lines[start_index + line_index + 1])
            else:
                next_cursor = None
            content = "\n".join(chunk_text for _, chunk_text, _ in rows) + TRUNCATION_MSG
        else:
            content = "\n".join(chunk for _, chunk, _ in rows)
        row_chunk_offsets = [(line_index, chunk_offset) for line_index, _, chunk_offset in rows]

        end_line = start_index + rows[-1][0] + 1 if rows else start_index + 1
        next_offset = end_line if next_cursor is None and end_line < total_lines else None

        return ReadResult(
            file_data=FileData(content=content, encoding="utf-8"),
            total_lines=total_lines,
            start_line=start_index + 1,
            end_line=end_line,
            next_offset=next_offset,
            continuation_cursor=next_cursor,
            first_row_char_offset=rows[0][2] if rows else first_char_offset,
            row_chunk_offsets=row_chunk_offsets,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the LangSmith sandbox.

        Supports partial success. Individual downloads may fail without
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
