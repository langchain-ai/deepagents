"""StoreBackend: Adapter for LangGraph's BaseStore (persistent, cross-thread)."""

from typing import Any, Optional

from langchain.tools import ToolRuntime
from langgraph.config import get_config
from langgraph.store.base import BaseStore, Item


class StoreBackend:
    """Backend that stores files in LangGraph's BaseStore (persistent).
    
    Uses LangGraph's Store for persistent, cross-conversation storage.
    Files are organized via namespaces and persist across all threads.
    
    The namespace can include an optional assistant_id for multi-agent isolation.
    """
    
    def __init__(self, runtime: ToolRuntime, store: Optional[BaseStore] = None) -> None:
        """Initialize with runtime and optional store.
        
        Args:
            runtime: ToolRuntime providing access to store via runtime.store.
            store: Optional explicit store instance. If None, uses runtime.store.
        """
        self.runtime = runtime
        self._store = store
    
    @property
    def uses_state(self) -> bool:
        """False for StoreBackend - modifies external storage directly."""
        return False
    
    def _get_store(self) -> BaseStore:
        """Get the store instance.
        
        Returns:
            BaseStore instance
        
        Raises:
            ValueError: If no store is available
        """
        store = self._store or self.runtime.store
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
    
    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Get file from store.
        
        Args:
            key: File path (e.g., "/notes.txt")
        
        Returns:
            FileData dict or None if not found.
        """
        store = self._get_store()
        namespace = self._get_namespace()
        item: Optional[Item] = store.get(namespace, key)
        
        if item is None:
            return None
        
        return self._convert_store_item_to_file_data(item)
    
    def put(self, key: str, value: dict[str, Any]) -> None:
        """Store file in store.
        
        Args:
            key: File path (e.g., "/notes.txt")
            value: FileData dict
        """
        store = self._get_store()
        namespace = self._get_namespace()
        store_value = self._convert_file_data_to_store_value(value)
        store.put(namespace, key, store_value)
    
    def ls(self, prefix: Optional[str] = None) -> list[str]:
        """List files from store.
        
        Args:
            prefix: Optional path prefix to filter results.
        
        Returns:
            List of file paths.
        """
        store = self._get_store()
        namespace = self._get_namespace()
        
        # Search store with optional prefix filter
        items = store.search(namespace, filter={"prefix": prefix} if prefix else None)
        
        return [item.key for item in items]
    
    def delete(self, key: str) -> None:
        """Delete file from store.
        
        Args:
            key: File path to delete
        """
        store = self._get_store()
        namespace = self._get_namespace()
        store.delete(namespace, key)
