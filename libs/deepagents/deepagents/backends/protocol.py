"""Protocol definition for pluggable memory backends.

This module defines the BackendProtocol that all backend implementations
must follow. Backends can store files in different locations (state, filesystem,
database, etc.) and provide a uniform interface for file operations.
"""

import abc
import asyncio
import inspect
import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, NotRequired, TypeAlias

from langchain.tools import ToolRuntime
from typing_extensions import TypedDict

FileFormat = Literal["v1", "v2"]
r"""File storage format version.

- `'v1'`: Legacy format — `content` stored as `list[str]` (lines split
    on `\\n`), no `encoding` field.
- `'v2'`: Current format — `content` stored as a plain `str` (UTF-8 text
    or base64-encoded binary), with an `encoding` field (`"utf-8"` or
    `"base64"`).
"""

logger = logging.getLogger(__name__)

FileOperationError = Literal[
    "file_not_found",
    "permission_denied",
    "is_directory",
    "invalid_path",
]
"""Standardized error codes for file upload/download operations.

These represent common, recoverable errors that an LLM can understand and
potentially fix:

- file_not_found: The requested file doesn't exist (download)
- permission_denied: Access denied for the operation
- is_directory: Attempted to download a directory as a file
- invalid_path: Path syntax is malformed or contains invalid characters
"""


@dataclass
class FileDownloadResponse:
    """Result of a single file download operation.

    The response is designed to allow partial success in batch operations.

    The errors are standardized using `FileOperationError` literals for certain
    recoverable conditions for use cases that involve LLMs performing
    file operations.

    Examples:
        >>> # Success
        >>> FileDownloadResponse(path="/app/config.json", content=b"{...}", error=None)
        >>> # Failure
        >>> FileDownloadResponse(path="/wrong/path.txt", content=None, error="file_not_found")
    """

    path: str
    """The file path that was requested. Included for easy correlation when
    processing batch results, especially useful for error messages."""

    content: bytes | None = None
    """File contents as bytes on success, `None` on failure."""

    error: FileOperationError | None = None
    """A `FileOperationError` literal for known conditions, or a
    backend-specific error string when the failure cannot be normalized.

    `None` on success.
    """


@dataclass
class FileUploadResponse:
    """Result of a single file upload operation.

    The response is designed to allow partial success in batch operations.

    The errors are standardized using `FileOperationError` literals for certain
    recoverable conditions for use cases that involve LLMs performing
    file operations.

    Examples:
        >>> # Success
        >>> FileUploadResponse(path="/app/data.txt", error=None)
        >>> # Failure
        >>> FileUploadResponse(path="/readonly/file.txt", error="permission_denied")
    """

    path: str
    """The file path that was requested.

    Included for easy correlation when processing batch results and for clear
    error messages.
    """

    error: FileOperationError | None = None
    """error: A `FileOperationError` literal for known conditions, or a
    backend-specific error string when the failure cannot be normalized.

    `None` on success.
    """


class FileInfo(TypedDict):
    """Structured file listing info.

    Minimal contract used across backends. Only `path` is required.
    Other fields are best-effort and may be absent depending on backend.
    """

    path: str
    """Absolute or relative file path."""

    is_dir: NotRequired[bool]
    """Whether the entry is a directory."""

    size: NotRequired[int]
    """File size in bytes (approximate)."""

    modified_at: NotRequired[str]
    """ISO 8601 timestamp of last modification, if known."""


class GrepMatch(TypedDict):
    """A single match from a grep search."""

    path: str
    """Path to the file containing the match."""

    line: int
    """1-indexed line number of the match."""

    text: str
    """Content of the matching line."""


class FileData(TypedDict):
    """Data structure for storing file contents with metadata."""

    content: str
    """File content as a plain string (utf-8 text or base64-encoded binary)."""

    encoding: str
    """Content encoding: `"utf-8"` for text, `"base64"` for binary."""

    created_at: NotRequired[str]
    """ISO 8601 timestamp of file creation."""

    modified_at: NotRequired[str]
    """ISO 8601 timestamp of last modification."""


@dataclass
class ReadResult:
    """Result from backend read operations.

    Attributes:
        error: Error message on failure, None on success.
        file_data: FileData dict on success, None on failure.
    """

    error: str | None = None
    file_data: FileData | None = None


@dataclass
class WriteResult:
    """Result from backend write operations.

    Attributes:
        error: Error message on failure, None on success.
        path: Absolute path of written file, None on failure.

    Examples:
        >>> WriteResult(path="/f.txt")
        >>> WriteResult(error="File exists")
    """

    error: str | None = None
    path: str | None = None


@dataclass
class EditResult:
    """Result from backend edit operations.

    Attributes:
        error: Error message on failure, None on success.
        path: Absolute path of edited file, None on failure.
        occurrences: Number of replacements made, None on failure.

    Examples:
        >>> EditResult(path="/f.txt", occurrences=1)
        >>> EditResult(error="File not found")
    """

    error: str | None = None
    path: str | None = None
    occurrences: int | None = None


@dataclass
class LsResult:
    """Result from backend ls operations.

    Attributes:
        error: Error message on failure, None on success.
        entries: List of file info dicts on success, None on failure.
    """

    error: str | None = None
    entries: list["FileInfo"] | None = None


@dataclass
class GrepResult:
    """Result from backend grep operations.

    Attributes:
        error: Error message on failure, None on success.
        matches: List of grep match dicts on success, None on failure.
    """

    error: str | None = None
    matches: list["GrepMatch"] | None = None


@dataclass
class GlobResult:
    """Result from backend glob operations.

    Attributes:
        error: Error message on failure, None on success.
        matches: List of matching file info dicts on success, None on failure.
    """

    error: str | None = None
    matches: list["FileInfo"] | None = None


# @abstractmethod to avoid breaking subclasses that only implement a subset
class BackendProtocol(abc.ABC):  # noqa: B024
    r"""Protocol for pluggable memory backends (single, unified).

    Backends can store files in different locations (state, filesystem, database, etc.)
    and provide a uniform interface for file operations.

    All file data is represented as dicts with the following structure::

        {
            "content": str,  # Text content (utf-8) or base64-encoded binary
            "encoding": str,  # "utf-8" for text, "base64" for binary data
            "created_at": str,  # ISO format timestamp
            "modified_at": str,  # ISO format timestamp
        }

    Note:
        Legacy data may still contain `"content": list[str]` (lines split on
        `\\n`).  Backends accept this for backwards compatibility and emit a
        `DeprecationWarning`.
    """

    def ls(self, path: str) -> "LsResult":
        """List all files in a directory with metadata.

        Args:
            path: Absolute path to the directory to list. Must start with '/'.

        Returns:
            LsResult with directory entries or error.
        """
        if type(self).ls_info is not BackendProtocol.ls_info:
            warnings.warn(
                "`ls_info` is deprecated and will be removed in v0.7; rename to `ls` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.ls_info(path)

        raise NotImplementedError

    async def als(self, path: str) -> "LsResult":
        """Async version of `ls`."""
        return await asyncio.to_thread(self.ls, path)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read file content with line numbers.

        Args:
            file_path: Absolute path to the file to read. Must start with '/'.
            offset: Line number to start reading from (0-indexed). Default: 0.
            limit: Maximum number of lines to read. Default: 2000.

        Returns:
            String containing file content formatted with line numbers (cat -n format),
            starting at line 1. Lines longer than 2000 characters are truncated.

            Returns an error string if the file doesn't exist or can't be read.
        """
        raise NotImplementedError

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Async version of read."""
        return await asyncio.to_thread(self.read, file_path, offset, limit)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> "GrepResult":
        """Search for a literal text pattern in files.

        Args:
            pattern: Literal string to search for (NOT regex).

                Performs exact substring matching within file content.

                Example: "TODO" matches any line containing "TODO"

            path: Optional directory path to search in.

                If None, searches in current working directory.

                Example: `'/workspace/src'`

            glob: Optional glob pattern to filter which FILES to search.

                Filters by filename/path, not content.

                Supports standard glob wildcards:

                - `*` matches any characters in filename
                - `**` matches any directories recursively
                - `?` matches single character
                - `[abc]` matches one character from set

        Examples:
            - `'*.py'` - only search Python files
            - `'**/*.txt'` - search all `.txt` files recursively
            - `'src/**/*.js'` - search JS files under src/
            - `'test[0-9].txt'` - search `test0.txt`, `test1.txt`, etc.

        Returns:
            `GrepResult` with matches or error.
        """
        if type(self).grep_raw is not BackendProtocol.grep_raw:
            warnings.warn(
                "`grep_raw` is deprecated and will be removed in v0.7; rename to `grep` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.grep_raw(pattern, path, glob)

        raise NotImplementedError

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> "GrepResult":
        """Async version of `grep`."""
        return await asyncio.to_thread(self.grep, pattern, path, glob)

    def glob(self, pattern: str, path: str = "/") -> "GlobResult":
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern with wildcards to match file paths.

                Supports standard glob syntax:

                - `*` matches any characters within a filename/directory
                - `**` matches any directories recursively
                - `?` matches a single character
                - `[abc]` matches one character from set

            path: Base directory to search from.

                Default: `'/'` (root).

                The pattern is applied relative to this path.

        Returns:
            GlobResult with matching files or error.
        """
        if type(self).glob_info is not BackendProtocol.glob_info:
            warnings.warn(
                "`glob_info` is deprecated and will be removed in v0.7; rename to `glob` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.glob_info(pattern, path)

        raise NotImplementedError

    async def aglob(self, pattern: str, path: str = "/") -> "GlobResult":
        """Async version of `glob`."""
        return await asyncio.to_thread(self.glob, pattern, path)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Write content to a new file in the filesystem, error if file exists.

        Args:
            file_path: Absolute path where the file should be created.

                Must start with '/'.
            content: String content to write to the file.

        Returns:
            WriteResult
        """
        raise NotImplementedError

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Async version of write."""
        return await asyncio.to_thread(self.write, file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Perform exact string replacements in an existing file.

        Args:
            file_path: Absolute path to the file to edit. Must start with `'/'`.
            old_string: Exact string to search for and replace.

                Must match exactly including whitespace and indentation.
            new_string: String to replace old_string with.

                Must be different from old_string.
            replace_all: If True, replace all occurrences.

                If False (default), `old_string` must be unique in the file or
                the edit fails.

        Returns:
            EditResult
        """
        raise NotImplementedError

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Async version of edit."""
        return await asyncio.to_thread(self.edit, file_path, old_string, new_string, replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the sandbox.

        This API is designed to allow developers to use it either directly or by
        exposing it to LLMs via custom tools.

        Args:
            files: List of (path, content) tuples to upload.

        Returns:
            List of FileUploadResponse objects, one per input file.

                Response order matches input order (response[i] for files[i]).

                Check the error field to determine success/failure per file.

        Examples:
            ```python
            responses = sandbox.upload_files(
                [
                    ("/app/config.json", b"{...}"),
                    ("/app/data.txt", b"content"),
                ]
            )
            ```
        """
        raise NotImplementedError

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Async version of upload_files."""
        return await asyncio.to_thread(self.upload_files, files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the sandbox.

        This API is designed to allow developers to use it either directly or
        by exposing it to LLMs via custom tools.

        Args:
            paths: List of file paths to download.

        Returns:
            List of `FileDownloadResponse` objects, one per input path.

                Response order matches input order (response[i] for paths[i]).

                Check the error field to determine success/failure per file.
        """
        raise NotImplementedError

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Async version of download_files."""
        return await asyncio.to_thread(self.download_files, paths)

    # -- deprecated methods --------------------------------------------------

    def ls_info(self, path: str) -> "LsResult":
        """List all files in a directory with metadata.

        !!! warning "Deprecated"

            Use `ls` instead.
        """
        warnings.warn(
            "`ls_info` is deprecated and will be removed in v0.7; use `ls` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.ls(path)

    async def als_info(self, path: str) -> "LsResult":
        """Async version of `ls_info`.

        !!! warning "Deprecated"

            Use `als` instead.
        """
        warnings.warn(
            "`als_info` is deprecated and will be removed in v0.7; use `als` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.als(path)

    def glob_info(self, pattern: str, path: str = "/") -> "GlobResult":
        """Find files matching a glob pattern.

        !!! warning "Deprecated"

            Use `glob` instead.
        """
        warnings.warn(
            "`glob_info` is deprecated and will be removed in v0.7; use `glob` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.glob(pattern, path)

    async def aglob_info(self, pattern: str, path: str = "/") -> "GlobResult":
        """Async version of `glob_info`.

        !!! warning "Deprecated"

            Use `aglob` instead.
        """
        warnings.warn(
            "`aglob_info` is deprecated and will be removed in v0.7; use `aglob` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.aglob(pattern, path)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> "GrepResult":
        """Search for a literal text pattern in files.

        !!! warning "Deprecated"

            Use `grep` instead.
        """
        warnings.warn(
            "`grep_raw` is deprecated and will be removed in v0.7; use `grep` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.grep(pattern, path, glob)

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> "GrepResult":
        """Async version of `grep_raw`.

        !!! warning "Deprecated"

            Use `agrep` instead.
        """
        warnings.warn(
            "`agrep_raw` is deprecated and will be removed in v0.7; use `agrep` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.agrep(pattern, path, glob)


@dataclass
class ExecuteResponse:
    """Result of code execution.

    Simplified schema optimized for LLM consumption.
    """

    output: str
    """Combined stdout and stderr output of the executed command."""

    exit_code: int | None = None
    """The process exit code.

    0 indicates success, non-zero indicates failure.
    """

    truncated: bool = False
    """Whether the output was truncated due to backend limitations."""


class SandboxBackendProtocol(BackendProtocol):
    """Extension of `BackendProtocol` that adds shell command execution.

    Designed for backends running in isolated environments (containers, VMs,
    remote hosts).

    Adds `execute()`/`aexecute()` for shell commands and an `id` property.

    See `BaseSandbox` for a base class that implements all inherited file
    operations by delegating to `execute()`.
    """

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend instance."""
        raise NotImplementedError

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command in the sandbox environment.

        Simplified interface optimized for LLM consumption.

        Args:
            command: Full shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If None, uses the backend's default timeout.

                Callers should provide non-negative integer values for portable
                behavior across backends. A value of 0 may disable timeouts on
                backends that support no-timeout execution.

        Returns:
            `ExecuteResponse` with combined output, exit code, and truncation flag.
        """
        raise NotImplementedError

    async def aexecute(
        self,
        command: str,
        *,
        # ASYNC109 - timeout is a semantic parameter forwarded to the sync
        # implementation, not an asyncio.timeout() contract.
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> ExecuteResponse:
        """Async version of execute."""
        # The middleware layer validates timeout support before calling, so
        # this guard only protects direct callers bypassing the middleware.
        if timeout is not None and execute_accepts_timeout(type(self)):
            return await asyncio.to_thread(self.execute, command, timeout=timeout)
        return await asyncio.to_thread(self.execute, command)


@lru_cache(maxsize=128)
def execute_accepts_timeout(cls: type[SandboxBackendProtocol]) -> bool:
    """Check whether a backend class's `execute` accepts a `timeout` kwarg.

    Older backend packages didn't lower-bound their SDK dependency, so they
    may not accept the `timeout` keyword added to `SandboxBackendProtocol`.

    Results are cached per class to avoid repeated introspection overhead.
    """
    try:
        sig = inspect.signature(cls.execute)
    except (ValueError, TypeError):
        logger.warning(
            "Could not inspect signature of %s.execute; assuming timeout is not supported. This may indicate a backend packaging issue.",
            cls.__qualname__,
            exc_info=True,
        )
        return False
    else:
        return "timeout" in sig.parameters


BackendFactory: TypeAlias = Callable[[ToolRuntime], BackendProtocol]
BACKEND_TYPES = BackendProtocol | BackendFactory
