"""StateBackend: Store files in LangGraph agent state (ephemeral)."""

import re
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from .utils import (
    create_file_data,
    update_file_data,
    file_data_to_string,
    format_read_response,
    perform_string_replacement,
    _glob_search_files,
    _grep_search_files,
)


class StateBackend:
    """Backend that stores files in agent state (ephemeral).
    
    Uses LangGraph's state management and checkpointing. Files persist within
    a conversation thread but not across threads. State is automatically
    checkpointed after each agent step.
    
    Special handling: Since LangGraph state must be updated via Command objects
    (not direct mutation), put() operations return Command objects instead of None.
    This is indicated by the uses_state=True flag.
    
    Note: All methods require a runtime parameter to access state.
    """
    
    def ls(self, prefix: Optional[str] = None, runtime: Optional["ToolRuntime"] = None) -> list[str]:
        """List files from state.
        
        Args:
            prefix: Optional path prefix to filter results.
            runtime: ToolRuntime to access state.
        
        Returns:
            List of file paths.
        """
        if runtime is None:
            raise ValueError("StateBackend requires runtime parameter")
        files = runtime.state.get("files", {})
        keys = list(files.keys())
        
        if prefix is not None:
            keys = [k for k in keys if k.startswith(prefix)]
        
        return keys
    
    def read(
        self, 
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        runtime: Optional["ToolRuntime"] = None,
    ) -> str:
        """Read file content with line numbers.
        
        Args:
            file_path: Absolute file path
            offset: Line offset to start reading from (0-indexed)
            limit: Maximum number of lines to read
            runtime: ToolRuntime to access state.
        
        Returns:
            Formatted file content with line numbers, or error message.
        """
        if runtime is None:
            raise ValueError("StateBackend requires runtime parameter")
        files = runtime.state.get("files", {})
        file_data = files.get(file_path)
        
        if file_data is None:
            return f"Error: File '{file_path}' not found"
        
        return format_read_response(file_data, offset, limit)
    
    def write(
        self, 
        file_path: str,
        content: str,
        runtime: Optional["ToolRuntime"] = None,
    ) -> Command | str:
        """Create a new file with content.
        
        Args:
            file_path: Absolute file path
            content: File content as a string
            runtime: ToolRuntime to access state.
        
        Returns:
            Command object to update state, or error message if file exists.
        """
        if runtime is None:
            raise ValueError("StateBackend requires runtime parameter")
        files = runtime.state.get("files", {})
        
        if file_path in files:
            return f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path."
        
        new_file_data = create_file_data(content)
        tool_call_id = runtime.tool_call_id
        
        return Command(
            update={
                "files": {file_path: new_file_data},
                "messages": [
                    ToolMessage(
                        content=f"Updated file {file_path}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
    
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
            file_path: Absolute file path
            old_string: String to find and replace
            new_string: Replacement string
            replace_all: If True, replace all occurrences
            runtime: ToolRuntime to access state.
        
        Returns:
            Command object to update state, or error message on failure.
        """
        if runtime is None:
            raise ValueError("StateBackend requires runtime parameter")
        files = runtime.state.get("files", {})
        file_data = files.get(file_path)
        
        if file_data is None:
            return f"Error: File '{file_path}' not found"
        
        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)
        
        if isinstance(result, str):
            return result
        
        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)
        tool_call_id = runtime.tool_call_id
        
        return Command(
            update={
                "files": {file_path: new_file_data},
                "messages": [
                    ToolMessage(
                        content=f"Successfully replaced {occurrences} instance(s) of the string in '{file_path}'",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
    
    def delete(self, file_path: str, runtime: Optional["ToolRuntime"] = None) -> Command | None:
        """Delete file from state via Command.
        
        Args:
            file_path: File path to delete
            runtime: ToolRuntime to access state.
        
        Returns:
            Command object to update state (sets file to None for deletion).
        """
        if runtime is None:
            raise ValueError("StateBackend requires runtime parameter")
        tool_call_id = runtime.tool_call_id
        return Command(
            update={
                "files": {file_path: None},
                "messages": [
                    ToolMessage(
                        content=f"Deleted file {file_path}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
    
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
            runtime: ToolRuntime to access state.
        
        Returns:
            Formatted search results based on output_mode.
        """
        if runtime is None:
            raise ValueError("StateBackend requires runtime parameter")
        files = runtime.state.get("files", {})
        
        return _grep_search_files(files, pattern, path, include, output_mode)
    
    def glob(self, pattern: str, path: str = "/", runtime: Optional["ToolRuntime"] = None) -> list[str]:
        """Find files matching a glob pattern.
        
        Args:
            pattern: Glob pattern (e.g., "**/*.py", "*.txt", "/subdir/**/*.md")
            path: Base path to search from (default "/")
            runtime: ToolRuntime to access state.
        
        Returns:
            List of absolute file paths matching the pattern.
        """
        if runtime is None:
            raise ValueError("StateBackend requires runtime parameter")
        files = runtime.state.get("files", {})
        
        result = _glob_search_files(files, pattern, path)
        if result == "No files found":
            return []
        return result.split("\n")