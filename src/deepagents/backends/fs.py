from typing import Literal, NotRequired, Protocol, TypedDict


class FileSystemCapabilities(TypedDict):
    """Filesystem capabilities of the sandbox."""

    can_upload: bool
    """Whether the sandbox can upload files."""
    can_download: bool
    """Whether the sandbox can download files."""
    can_list_files: bool
    """Whether the sandbox can list files."""
    can_read: bool
    """Whether the sandbox can read file contents."""
    can_edit: bool
    """Whether the sandbox can edit files."""
    can_delete: bool
    """Whether the sandbox can delete files."""
    can_grep: bool
    """Whether the sandbox can search file contents."""
    can_glob: bool
    """Whether the sandbox can find files by pattern."""


class FileInfo(TypedDict):
    """Information about the file."""

    path: str
    """Fully qualified file name (absolute path)."""
    kind: NotRequired[Literal["file", "dir"]]
    """Whether the file is a directory."""
    size: NotRequired[int]
    """Size of the file in bytes."""
    created_at: NotRequired[str]
    """Modification time in ISO format."""
    modified_at: NotRequired[str]
    """Creation time in ISO format."""


class FileSystem(Protocol):
    """Protocol for pluggable memory backends.

    Backends can store files in different locations (state, filesystem, database, etc.)
    and provide a uniform interface for file operations.
    """

    def ls(self, prefix: str | None = None) -> list[FileInfo]:
        """List all file paths, optionally filtered by prefix.

        Args:
            prefix: Optional path prefix to filter results (e.g., "/subdir/", "/memories/")
                   If None, returns all files.

        Returns:
            List of absolute file paths matching the prefix.
        """
        ...

    def upload_file(self, file: bytes, path: str, *, timeout: int = 30 * 60) -> None:
        """Upload a file to the sandbox."""
        raise NotImplementedError

    def download_file(self, path: str, *, timeout: int = 30 * 60) -> bytes:
        """Download a file from the sandbox."""
        raise NotImplementedError

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.

        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md")
            offset: Line offset to start reading from (0-indexed)
            limit: Maximum number of lines to read

        Returns:
            Formatted file content with line numbers (cat -n style), or error message.
            Returns "Error: File '{file_path}' not found" if file doesn't exist.
            Returns "System reminder: File exists but has empty contents" for empty files.
        """
        ...

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Edit a file by replacing string occurrences.

        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md")
            old_string: String to find and replace
            new_string: Replacement string
            replace_all: If True, replace all occurrences; if False, require unique match

        Returns:
            - Command object for StateBackend (uses_state=True) to update LangGraph state
            - Success message string for other backends, or error message on failure

        Error cases:
            - "Error: File '{file_path}' not found" if file doesn't exist
            - "Error: String not found in file: '{old_string}'" if string not found
            - "Error: String '{old_string}' appears {n} times. Use replace_all=True..."
              if multiple matches found and replace_all=False
        """
        ...

    def delete(self, file_path: str) -> None:
        """Delete a file by path.

        Args:
            file_path: Absolute file path to delete
            runtime: Optional ToolRuntime to access state (required for StateBackend).

        Returns:
            - None for backends that modify storage directly (uses_state=False)
            - Command object for StateBackend (uses_state=True) to update LangGraph state
        """
        ...

    def grep(
        self,
        pattern: str,
        path: str = "/",
        include: str | None = None,
        output_mode: str = "files_with_matches",
    ) -> str:
        """Search for a pattern in files.

        Args:
            pattern: String pattern to search for
            path: Path to search in (default "/")
            include: Optional glob pattern to filter files (e.g., "*.py")
            output_mode: Output format - "files_with_matches", "content", or "count"
                - files_with_matches: List file paths that contain matches
                - content: Show matching lines with file paths and line numbers
                - count: Show count of matches per file

        Returns:
            Formatted search results based on output_mode, or message if no matches found.
        """
        ...

    def glob(self, pattern: str, path: str = "/") -> list[str]:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "**/*.py", "*.txt", "/subdir/**/*.md")
            path: Base path to search from (default "/")

        Returns:
            List of absolute file paths matching the pattern.
        """
        ...

    @property
    def get_capabilities(self) -> FileSystemCapabilities:
        """Get the filesystem capabilities."""
        ...
