"""Protocol definition for pluggable memory backends."""

from typing import Any, Protocol


class MemoryBackend(Protocol):
    """Protocol for pluggable long-term memory backends.
    
    Keys are file paths like "/agent.md", "/notes.txt", etc.
    Values are dictionaries with 'content' (list[str]), 'created_at' (str),
    and 'modified_at' (str) fields.
    
    The backend is responsible for organizing/isolating data internally
    (e.g., by agent_id if needed).
    
    Note: No delete operation is needed - agents don't have a tool to delete
    files from long-term memory. Cleanup is done externally if needed.
    """
    
    def get(self, key: str) -> dict[str, Any] | None:
        """Retrieve a single item by key.
        
        Args:
            key: File path like "/agent.md"
        
        Returns:
            Dictionary with 'content' (list[str]), 'created_at' (str), 
            'modified_at' (str) fields, or None if not found.
        """
        ...
    
    def put(self, key: str, value: dict[str, Any]) -> None:
        """Store or update an item.
        
        Args:
            key: File path like "/agent.md"
            value: Dict with 'content' (list[str]), 'created_at' (str), 
                   'modified_at' (str) fields
        """
        ...
    
    def ls(self) -> list[str]:
        """List all keys in this backend.
        
        Used by the agent's ls tool to show available files in long-term memory.
        
        Returns:
            List of all keys (file paths)
        """
        ...
