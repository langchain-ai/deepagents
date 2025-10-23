"""Protocol definition for pluggable memory backends.

This module defines the MemoryBackend protocol that all backend implementations
must follow. Backends can store files in different locations (state, filesystem,
database, etc.) and provide a uniform interface for file operations.
"""

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable
from langgraph.types import Command

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime


@runtime_checkable
class MemoryBackend(Protocol):
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
    
    def ls(self, prefix: Optional[str] = None, runtime: Optional["ToolRuntime"] = None) -> list[str]:
        """List all file paths, optionally filtered by prefix.
        
        Args:
            prefix: Optional path prefix to filter results (e.g., "/subdir/", "/memories/")
                   If None, returns all files.
            runtime: Optional ToolRuntime to access state (required for StateBackend).
        
        Returns:
            List of absolute file paths matching the prefix.
        """
        ...
    
    def read(
        self, 
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        runtime: Optional["ToolRuntime"] = None,
    ) -> str:
        """Read file content with line numbers.
        
        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md")
            offset: Line offset to start reading from (0-indexed)
            limit: Maximum number of lines to read
            runtime: Optional ToolRuntime to access state (required for StateBackend).
        
        Returns:
            Formatted file content with line numbers (cat -n style), or error message.
            Returns "Error: File '{file_path}' not found" if file doesn't exist.
            Returns "System reminder: File exists but has empty contents" for empty files.
        """
        ...
    
    def write(
        self, 
        file_path: str,
        content: str,
        runtime: Optional["ToolRuntime"] = None,
    ) -> Command | str:
        """Create a new file with content.
        
        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md")
            content: File content as a string
            runtime: Optional ToolRuntime to access state (required for StateBackend).
        
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
        runtime: Optional["ToolRuntime"] = None,
    ) -> Command | str:
        """Edit a file by replacing string occurrences.
        
        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md")
            old_string: String to find and replace
            new_string: Replacement string
            replace_all: If True, replace all occurrences; if False, require unique match
            runtime: Optional ToolRuntime to access state (required for StateBackend).
        
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
    
    def delete(self, file_path: str, runtime: Optional["ToolRuntime"] = None) -> Command | None:
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
        include: Optional[str] = None,
        output_mode: str = "files_with_matches",
        runtime: Optional["ToolRuntime"] = None,
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
            runtime: Optional ToolRuntime to access state (required for StateBackend).
        
        Returns:
            Formatted search results based on output_mode, or message if no matches found.
        """
        ...
    
    def glob(self, pattern: str, runtime: Optional["ToolRuntime"] = None) -> list[str]:
        """Find files matching a glob pattern.
        
        Args:
            pattern: Glob pattern (e.g., "**/*.py", "*.txt", "/subdir/**/*.md")
            runtime: Optional ToolRuntime to access state (required for StateBackend).
        
        Returns:
            List of absolute file paths matching the pattern.
        """
        ...