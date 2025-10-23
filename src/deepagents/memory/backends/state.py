"""StateBackend: Store files in LangGraph agent state (ephemeral)."""

from typing import Any, Optional

from langchain.tools import get_runtime
from langchain_core.messages import ToolMessage
from langgraph.types import Command


class StateBackend:
    """Backend that stores files in agent state (ephemeral).
    
    Uses LangGraph's state management and checkpointing. Files persist within
    a conversation thread but not across threads. State is automatically
    checkpointed after each agent step.
    
    Special handling: Since LangGraph state must be updated via Command objects
    (not direct mutation), put() operations return Command objects instead of None.
    This is indicated by the uses_state=True flag.
    """
    
    @property
    def uses_state(self) -> bool:
        """Always True for StateBackend - must return Commands for writes."""
        return True
    
    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Get file from state.
        
        Args:
            key: File path (e.g., "/notes.txt")
        
        Returns:
            FileData dict or None if not found.
        """
        runtime = get_runtime()
        files = runtime.state.get("files", {})
        return files.get(key)
    
    def put(self, key: str, value: dict[str, Any]) -> Command:
        """Store file in state via Command.
        
        Returns Command to update state, not None like other backends.
        This is required for LangGraph's state management.
        
        Args:
            key: File path (e.g., "/notes.txt")
            value: FileData dict
        
        Returns:
            Command object to update state.
        """
        runtime = get_runtime()
        tool_call_id = runtime.tool_call_id
        return Command(
            update={
                "files": {key: value},
                "messages": [
                    ToolMessage(
                        content=f"Updated file {key}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
    
    def ls(self, prefix: Optional[str] = None) -> list[str]:
        """List files from state.
        
        Args:
            prefix: Optional path prefix to filter results.
        
        Returns:
            List of file paths.
        """
        runtime = get_runtime()
        files = runtime.state.get("files", {})
        keys = list(files.keys())
        
        if prefix is not None:
            keys = [k for k in keys if k.startswith(prefix)]
        
        return keys
    
    def delete(self, key: str) -> Command:
        """Delete file from state via Command.
        
        Args:
            key: File path to delete
        
        Returns:
            Command object to update state (sets file to None for deletion).
        """
        runtime = get_runtime()
        tool_call_id = runtime.tool_call_id
        return Command(
            update={
                "files": {key: None},
                "messages": [
                    ToolMessage(
                        content=f"Deleted file {key}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
