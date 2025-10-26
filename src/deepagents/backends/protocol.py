"""Protocol definition for pluggable memory backends.

This module defines the BackendProtocol that all backend implementations
must follow. Backends can store files in different locations (state, filesystem,
database, etc.) and provide a uniform interface for file operations.
"""

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable
from langgraph.types import Command
from langchain.tools import ToolRuntime


@runtime_checkable
class _BackendProtocol(Protocol):
    """Protocol for pluggable memory backends.

    Backends can store files in different locations (state, filesystem, database, etc.)
    and provide a uniform interface for file operations.

    All file data is represented as dicts with the following structure:
    {
        "content": list[str],      # Lines of text content
        "created_at": str,         # ISO format timestamp
        "modified_at": str,        # ISO format timestamp
    }
    """
    
    def ls(self, path: str) -> list[str]:
        """List all file paths in a directory.
        
        Args:
            path: Absolute path to directory (e.g., "/", "/subdir/", "/memories/")
        
        Returns:
            List of absolute file paths in the specified directory.
        """
        ...
    
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

    
    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
        output_mode: str = "files_with_matches",
    ) -> str:
        """Search for a pattern in files.
        
        TODO: This implementation is significantly less capable than Claude Code's Grep tool.
        Missing features to add in the future:
        - Context lines: -A (after), -B (before), -C (context) parameters
        - Line numbers: -n parameter to show line numbers in output
        - Case sensitivity: -i parameter for case-insensitive search
        - Output limiting: head_limit parameter for large result sets
        - File type filter: type parameter (e.g., "py", "js")
        - Multiline support: multiline parameter for cross-line pattern matching
        - Pattern semantics: Clarify if pattern is regex or literal string
        See /memories/memory_backend_vs_claude_code_comparison.md for full details.
        
        Args:
            pattern: String pattern to search for (currently literal string)
            path: Path to search in (default "/")
            glob: Optional glob pattern to filter files (e.g., "*.py")
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
            path: Base path to search from (default: "/")

        Returns:
            List of absolute file paths matching the pattern.
        """
        ...


class BackendProtocol(_BackendProtocol):
    def write(
            self,
            file_path: str,
            content: str,
    ) -> str:
        """Create a new file with content.

        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md")
            content: File content as a string

        Returns:
            - Command object for StateBackend (uses_state=True) to update LangGraph state
            - Success message string for other backends, or error if file already exists

        Error cases:
            - Returns error message if file already exists (should use edit instead)
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


@runtime_checkable
class BackendProvider(Protocol):

    def get_backend(self, runtime: ToolRuntime) -> BackendProtocol:
        """Get the backend."""
        ...


class StateBackendProtocol(_BackendProtocol):

    def write(
            self,
            file_path: str,
            content: str,
    ) -> Command | str:
        """Create a new file with content.

        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md")
            content: File content as a string

        Returns:
            - Command object for StateBackend (uses_state=True) to update LangGraph state
            - Success message string for other backends, or error if file already exists

        Error cases:
            - Returns error message if file already exists (should use edit instead)
        """
        ...

    def edit(
            self,
            file_path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
    ) -> Command | str:
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

@runtime_checkable
class StateBackendProvider(Protocol):

    def get_backend(self, runtime: ToolRuntime) -> StateBackendProtocol:
        """Get the backend."""
        ...