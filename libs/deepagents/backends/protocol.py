"""Protocol definition for pluggable memory backends.

This module defines the BackendProtocol that all backend implementations
must follow. Backends can store files in different locations (state, filesystem,
database, etc.) and provide a uniform interface for file operations.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, TypedDict, runtime_checkable

from langchain.tools import ToolRuntime


class FileInfo(TypedDict, total=False):
    """Structured file listing info.

    Minimal contract used across backends. Only "path" is required.
    Other fields are best-effort and may be absent depending on backend.
    """

    path: str
    is_dir: bool
    size: int  # bytes (approx)
    modified_at: str  # ISO timestamp if known


class GrepMatch(TypedDict):
    """Structured grep match entry."""

    path: str
    line: int
    text: str


@dataclass
class WriteResult:
    """Result from backend write operations.

    Attributes:
        error: Error message on failure, None on success.
        path: Absolute path of written file, None on failure.
        files_update: State update dict for checkpoint backends, None for external storage.
            Checkpoint backends populate this with {file_path: file_data} for LangGraph state.
            External backends set None (already persisted to disk/S3/database/etc).

    Examples:
        >>> # Checkpoint storage
        >>> WriteResult(path="/f.txt", files_update={"/f.txt": {...}})
        >>> # External storage
        >>> WriteResult(path="/f.txt", files_update=None)
        >>> # Error
        >>> WriteResult(error="File exists")
    """

    error: str | None = None
    path: str | None = None
    files_update: dict[str, Any] | None = None


@dataclass
class EditResult:
    """Result from backend edit operations.

    Attributes:
        error: Error message on failure, None on success.
        path: Absolute path of edited file, None on failure.
        files_update: State update dict for checkpoint backends, None for external storage.
            Checkpoint backends populate this with {file_path: file_data} for LangGraph state.
            External backends set None (already persisted to disk/S3/database/etc).
        occurrences: Number of replacements made, None on failure.

    Examples:
        >>> # Checkpoint storage
        >>> EditResult(path="/f.txt", files_update={"/f.txt": {...}}, occurrences=1)
        >>> # External storage
        >>> EditResult(path="/f.txt", files_update=None, occurrences=2)
        >>> # Error
        >>> EditResult(error="File not found")
    """

    error: str | None = None
    path: str | None = None
    files_update: dict[str, Any] | None = None
    occurrences: int | None = None


@runtime_checkable
class BackendProtocol(Protocol):
    """Protocol for pluggable memory backends (single, unified).

    Backends can store files in different locations (state, filesystem, database, etc.)
    and provide a uniform interface for file operations.

    All file data is represented as dicts with the following structure:
    {
        "content": list[str],      # Lines of text content
        "created_at": str,         # ISO format timestamp
        "modified_at": str,        # ISO format timestamp
    }
    """

    def ls_info(self, path: str) -> list["FileInfo"]:
        """Structured listing with file metadata."""
        ...

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers or an error string."""
        ...

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list["GrepMatch"] | str:
        """Structured search results or error string for invalid input."""
        ...

    def glob_info(self, pattern: str, path: str = "/") -> list["FileInfo"]:
        """Structured glob matching returning FileInfo dicts."""
        ...

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file. Returns WriteResult; error populated on failure."""
        ...

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences. Returns EditResult."""
        ...


@dataclass
class ExecuteResponse:
    """Result of code execution.

    Simplified schema optimized for LLM consumption.
    """

    output: str
    """Combined stdout and stderr output of the executed command."""

    exit_code: int | None = None
    """The process exit code. 0 indicates success, non-zero indicates failure."""

    truncated: bool = False
    """Whether the output was truncated due to backend limitations."""


@dataclass
class FileDownloadResponse:
    """Result of a single file download operation.

    Supports partial success in batch operations - if an LLM requests multiple
    files and some paths are incorrect, the valid files can still be downloaded
    while errors are reported for invalid paths.

    Attributes:
        path: The file path that was requested. Included for easy correlation
            when processing batch results, especially useful for error messages.
        content: File contents as bytes on success, None on failure.
        error: Human-readable error message on failure, None on success.
            Stored as string rather than exception for LLM consumption - allows
            the error to be serialized in tool responses so LLMs can read the
            error and correct mistakes (e.g., fix an incorrect file path).

    Examples:
        >>> # Success
        >>> FileDownloadResponse(path="/app/config.json", content=b"{...}", error=None)
        >>> # Failure
        >>> FileDownloadResponse(path="/wrong/path.txt", content=None, error="No such file or directory: /wrong/path.txt")
    """

    path: str
    content: bytes | None = None
    error: str | None = None


@dataclass
class FileUploadResponse:
    """Result of a single file upload operation.

    Supports partial success in batch operations - if an LLM attempts to upload
    multiple files and some operations fail (e.g., permission denied), successful
    uploads still complete while errors are reported for failures.

    Attributes:
        path: The file path that was requested. Included for easy correlation
            when processing batch results and for clear error messages.
        error: Human-readable error message on failure, None on success.
            Stored as string rather than exception for LLM consumption - allows
            the error to be serialized in tool responses so LLMs can read the
            error and correct mistakes (e.g., fix permission issues or paths).

    Examples:
        >>> # Success
        >>> FileUploadResponse(path="/app/data.txt", error=None)
        >>> # Failure
        >>> FileUploadResponse(path="/readonly/file.txt", error="Permission denied: /readonly/file.txt")
    """

    path: str
    error: str | None = None


@runtime_checkable
class SandboxBackendProtocol(BackendProtocol, Protocol):
    """Protocol for sandboxed backends with isolated runtime.

    Sandboxed backends run in isolated environments (e.g., separate processes,
    containers) and communicate via defined interfaces.
    """

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """Execute a command in the process.

        Simplified interface optimized for LLM consumption.

        Args:
            command: Full shell command string to execute.

        Returns:
            ExecuteResponse with combined output, exit code, optional signal, and truncation flag.
        """
        ...

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend instance."""
        ...

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the sandbox.

        This is NOT meant to be used by LLMs directly. LLMs should use write/edit
        methods for file content manipulation. This method is provided to allow
        users to define custom tools that use upload functionality as needed.

        Supports partial success - individual file uploads may fail without
        affecting others. Implementations must catch exceptions per-file and
        return error messages in FileUploadResponse objects rather than raising.

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
        ...

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the sandbox.

        This is NOT meant to be used by LLMs directly. LLMs should use read/grep
        methods for file content retrieval. This method is provided to allow
        users to define custom tools that use download functionality as needed.

        Supports partial success - individual file downloads may fail without
        affecting others. Implementations must catch exceptions per-file and
        return error messages in FileDownloadResponse objects rather than raising.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.
            Response order matches input order (response[i] for paths[i]).
            Check the error field to determine success/failure per file.
        """
        ...


BackendFactory: TypeAlias = Callable[[ToolRuntime], BackendProtocol]
BACKEND_TYPES = BackendProtocol | BackendFactory
