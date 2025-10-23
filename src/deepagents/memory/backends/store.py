"""StoreBackend: Adapter for LangGraph's BaseStore (persistent, cross-thread)."""

import re
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

from langgraph.config import get_config
from langgraph.store.base import BaseStore, Item
from langgraph.types import Command

from deepagents.memory.backends.utils import (
    create_file_data,
    update_file_data,
    file_data_to_string,
    format_read_response,
    perform_string_replacement,
    _glob_search_files,
    _grep_search_files,
)


class StoreBackend:
    """Backend that stores files in LangGraph's BaseStore (persistent).
    
    Uses LangGraph's Store for persistent, cross-conversation storage.
    Files are organized via namespaces and persist across all threads.
    
    The namespace can include an optional assistant_id for multi-agent isolation.
    """

    def _get_store(self, runtime: Optional["ToolRuntime"] = None) -> BaseStore:
        """Get the store instance.
        
        Args:
            runtime: ToolRuntime to access store.
        
        Returns:
            BaseStore instance
        
        Raises:
            ValueError: If no store is available or runtime not provided
        """
        if runtime is None:
            msg = "StoreBackend requires runtime parameter"
            raise ValueError(msg)
        store = runtime.store
        if store is None:
            msg = "Store is required but not available in runtime"
            raise ValueError(msg)
        return store
    
    def _get_namespace(self) -> tuple[str, ...]:
        """Get the namespace for store operations.
        
        Returns a tuple for organizing files in the store. If an assistant_id is
        available in the config metadata, returns (assistant_id, "filesystem") to
        provide per-assistant isolation. Otherwise, returns ("filesystem",).
        
        Returns:
            Namespace tuple for store operations.
        """
        namespace = "filesystem"
        config = get_config()
        if config is None:
            return (namespace,)
        assistant_id = config.get("metadata", {}).get("assistant_id")
        if assistant_id is None:
            return (namespace,)
        return (assistant_id, namespace)
    
    def _convert_store_item_to_file_data(self, store_item: Item) -> dict[str, Any]:
        """Convert a store Item to FileData format.
        
        Args:
            store_item: The store Item containing file data.
        
        Returns:
            FileData dict with content, created_at, and modified_at fields.
        
        Raises:
            ValueError: If required fields are missing or have incorrect types.
        """
        if "content" not in store_item.value or not isinstance(store_item.value["content"], list):
            msg = f"Store item does not contain valid content field. Got: {store_item.value.keys()}"
            raise ValueError(msg)
        if "created_at" not in store_item.value or not isinstance(store_item.value["created_at"], str):
            msg = f"Store item does not contain valid created_at field. Got: {store_item.value.keys()}"
            raise ValueError(msg)
        if "modified_at" not in store_item.value or not isinstance(store_item.value["modified_at"], str):
            msg = f"Store item does not contain valid modified_at field. Got: {store_item.value.keys()}"
            raise ValueError(msg)
        return {
            "content": store_item.value["content"],
            "created_at": store_item.value["created_at"],
            "modified_at": store_item.value["modified_at"],
        }
    
    def _convert_file_data_to_store_value(self, file_data: dict[str, Any]) -> dict[str, Any]:
        """Convert FileData to a dict suitable for store.put().
        
        Args:
            file_data: The FileData to convert.
        
        Returns:
            Dictionary with content, created_at, and modified_at fields.
        """
        return {
            "content": file_data["content"],
            "created_at": file_data["created_at"],
            "modified_at": file_data["modified_at"],
        }
    
    def ls(self, prefix: Optional[str] = None, runtime: Optional["ToolRuntime"] = None) -> list[str]:
        """List files from store.
        
        Args:
            prefix: Optional path prefix to filter results.
            runtime: ToolRuntime to access store.
        
        Returns:
            List of file paths.
        """
        store = self._get_store(runtime)
        namespace = self._get_namespace()
        
        # Search store with optional prefix filter
        items = store.search(namespace, filter={"prefix": prefix} if prefix else None)
        
        return [item.key for item in items]
    
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
            runtime: ToolRuntime to access store.
            limit: Maximum number of lines to read
        
        Returns:
            Formatted file content with line numbers, or error message.
        """
        store = self._get_store(runtime)
        namespace = self._get_namespace()
        item: Optional[Item] = store.get(namespace, file_path)
        
        if item is None:
            return f"Error: File '{file_path}' not found"
        
        try:
            file_data = self._convert_store_item_to_file_data(item)
        except ValueError as e:
            return f"Error: {e}"
        
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
            runtime: ToolRuntime to access store.
        
        Returns:
            Success message or error if file already exists.
        """
        store = self._get_store(runtime)
        namespace = self._get_namespace()
        
        # Check if file exists
        existing = store.get(namespace, file_path)
        if existing is not None:
            return f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path."
        
        # Create new file
        file_data = create_file_data(content)
        store_value = self._convert_file_data_to_store_value(file_data)
        store.put(namespace, file_path, store_value)
        
        return f"Updated file {file_path}"
    
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
            runtime: ToolRuntime to access store.
        
        Returns:
            Success message or error message on failure.
        """
        store = self._get_store(runtime)
        namespace = self._get_namespace()
        
        # Get existing file
        item = store.get(namespace, file_path)
        if item is None:
            return f"Error: File '{file_path}' not found"
        
        try:
            file_data = self._convert_store_item_to_file_data(item)
        except ValueError as e:
            return f"Error: {e}"
        
        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)
        
        if isinstance(result, str):
            return result
        
        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)
        
        # Update file in store
        store_value = self._convert_file_data_to_store_value(new_file_data)
        store.put(namespace, file_path, store_value)
        
        return f"Successfully replaced {occurrences} instance(s) of the string in '{file_path}'"
    
    def delete(self, file_path: str, runtime: Optional["ToolRuntime"] = None) -> Command | None:
        """Delete file from store.
        
        Args:
            file_path: File path to delete
            runtime: ToolRuntime to access store.
        
        Returns:
            None (direct store modification)
        """
        store = self._get_store(runtime)
        namespace = self._get_namespace()
        store.delete(namespace, file_path)
        
        return None
    
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
            runtime: ToolRuntime to access store.
        
        Returns:
            Formatted search results based on output_mode.
        """
        store = self._get_store(runtime)
        namespace = self._get_namespace()
        
        items = store.search(namespace)
        
        files = {}
        for item in items:
            if item is None:
                continue
            try:
                file_data = self._convert_store_item_to_file_data(item)
                files[item.key] = file_data
            except ValueError:
                continue
        
        return _grep_search_files(files, pattern, path, include, output_mode)
    
    def glob(self, pattern: str, path: str = "/", runtime: Optional["ToolRuntime"] = None) -> list[str]:
        """Find files matching a glob pattern.
        
        Args:
            pattern: Glob pattern (e.g., "**/*.py", "*.txt", "/subdir/**/*.md")
            path: Base path to search from (default "/")
            runtime: ToolRuntime to access store.
        
        Returns:
            List of absolute file paths matching the pattern.
        """
        store = self._get_store(runtime)
        namespace = self._get_namespace()
        
        items = store.search(namespace)
        
        files = {}
        for item in items:
            if item is None:
                continue
            try:
                file_data = self._convert_store_item_to_file_data(item)
                files[item.key] = file_data
            except ValueError:
                continue
        
        result = _glob_search_files(files, pattern, path)
        if result == "No files found":
            return []
        return result.split("\n")