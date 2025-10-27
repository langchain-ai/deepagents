"""Type definitions for backend interface.

This module defines the unified Backend interface that all backend implementations
must follow. Backends can store files in different locations (state, filesystem,
database, etc.) and provide a uniform interface for file operations.
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain.tools import ToolRuntime


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


class Backend(ABC):
    """Abstract interface for pluggable file storage backends.

    Backends store files in different locations (LangGraph state, filesystem, database, etc.)
    and provide a uniform interface for file operations.

    Storage Models:
        Checkpoint Storage (StateBackend): Files stored in LangGraph state, persisted via
            checkpointing. Returns WriteResult/EditResult with files_update populated for
            tool layer to convert to Command objects.

        External Storage (FilesystemBackend, StoreBackend, etc.): Files persisted directly
            to external systems (disk, S3, BaseStore, database). Returns WriteResult/EditResult
            with files_update=None.

    File Data Format:
        Checkpoint backends use dicts: {"content": list[str], "created_at": str, "modified_at": str}.
        External backends manage their own format internally.
    """

    @abstractmethod
    def ls(self, path: str) -> list[str]:
        """List all file paths in a directory.

        Args:
            path: Absolute directory path (e.g., "/", "/subdir/", "/memories/").

        Returns:
            List of absolute file paths. Empty list if directory doesn't exist or is empty.
        """
        ...

    @abstractmethod
    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.

        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md").
            offset: Line offset to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            Formatted content with line numbers (cat -n style), or error message if file not found.
        """
        ...

    @abstractmethod
    def write(self, file_path: str, content: str) -> WriteResult:
        """Create a new file with content.

        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md").
            content: File content as a string.

        Returns:
            WriteResult with error=None on success (files_update populated for checkpoint
            backends, None for external). Returns error message if file exists, permission
            denied, path traversal attempt, or I/O error.
        """
        ...

    @abstractmethod
    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences.

        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md").
            old_string: String to find and replace.
            new_string: Replacement string.
            replace_all: If True, replace all occurrences; if False, require unique match.

        Returns:
            EditResult with error=None on success (files_update populated for checkpoint
            backends, None for external). Returns error if file not found, string not found,
            or multiple matches without replace_all=True.
        """
        ...

    @abstractmethod
    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        output_mode: str = "files_with_matches",
    ) -> str:
        """Search for a pattern in files.

        Args:
            pattern: Search pattern (implementation-specific: literal or regex).
            path: Directory to search in (default "/").
            glob: Optional glob pattern to filter files (e.g., "*.py").
            output_mode: Output format - "files_with_matches" (file paths), "content"
                (matching lines with context), or "count" (match counts per file).

        Returns:
            Formatted search results based on output_mode, or message if no matches found.

        Note:
            This is a basic implementation. Missing features: context lines (-A/-B/-C),
            line numbers (-n), case sensitivity (-i), output limiting, file type filters,
            multiline support.
        """
        ...

    @abstractmethod
    def glob(self, pattern: str, path: str = "/") -> list[str]:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "**/*.py", "*.txt", "/subdir/**/*.md").
            path: Base directory to search from (default "/").

        Returns:
            List of absolute file paths matching pattern. Empty list if no matches.
        """
        ...


# Backend factory function types
BackendProvider = Callable[[ToolRuntime], Backend]
AsyncBackendProvider = Callable[[ToolRuntime], Awaitable[Backend]]
