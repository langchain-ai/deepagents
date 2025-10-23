"""Protocol definition for pluggable memory backends.

This module defines the MemoryBackend protocol that all backend implementations
must follow. Backends can store files in different locations (state, filesystem,
database, etc.) and provide a uniform interface for file operations.
"""

from typing import Any, Optional, Protocol, Union, runtime_checkable


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
    
    @property
    def uses_state(self) -> bool:
        """Flag indicating if this backend uses agent state.
        
        When True, put() operations should return Command objects instead of None.
        This is required for backends that store data in LangGraph state, which
        must be updated via Command objects for proper checkpointing.
        
        Default: False (most backends modify external storage directly)
        """
        ...
    
    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Retrieve file data by key.
        
        Args:
            key: File path (e.g., "/notes.txt")
        
        Returns:
            FileData dict with keys: 'content' (list[str]), 'created_at' (str), 
            'modified_at' (str), or None if not found.
        """
        ...
    
    def put(self, key: str, value: dict[str, Any]) -> Any:
        """Store or update file data.
        
        Args:
            key: File path (e.g., "/notes.txt")
            value: FileData dict with keys: 'content' (list[str]), 'created_at' (str),
                   'modified_at' (str)
        
        Returns:
            - None for most backends (direct storage)
            - Command object for StateBackend (uses_state=True)
            - String message for backends that want custom confirmation
        """
        ...
    
    def ls(self, prefix: Optional[str] = None) -> list[str]:
        """List all file keys, optionally filtered by prefix.
        
        Args:
            prefix: Optional path prefix to filter results (e.g., "/subdir/")
        
        Returns:
            List of file paths matching the prefix.
        """
        ...
    
    def delete(self, key: str) -> Any:
        """Delete a file by key.
        
        Args:
            key: File path to delete
        
        Returns:
            - None for most backends
            - Command object for StateBackend (uses_state=True)
        """
        ...
