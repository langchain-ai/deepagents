"""StoreBackend: Adapter for LangGraph's BaseStore (persistent, cross-thread)."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

from langgraph.config import get_config
from langgraph.store.base import BaseStore, Item

from deepagents.backends.protocol import Backend, EditResult, WriteResult
from deepagents.backends.utils import (
    _glob_search_files,
    _grep_search_files,
    create_file_data,
    file_data_to_string,
    format_read_response,
    perform_string_replacement,
    truncate_if_too_long,
    update_file_data,
)


class StoreBackend(Backend):
    """Backend that stores files in LangGraph's BaseStore (persistent).

    Uses LangGraph's Store for persistent, cross-conversation storage.
    Files are organized via namespaces and persist across all threads.

    The namespace can include an optional assistant_id for multi-agent isolation.

    Storage Model: External Storage
    --------------------------------
    This backend uses external storage (LangGraph BaseStore). Write and edit
    operations persist directly to the store and return WriteResult/EditResult
    with files_update=None.
    """

    def __init__(self, runtime: "ToolRuntime"):
        """Initialize StoreBackend with runtime.

        Args:
            runtime: Tool runtime with access to LangGraph BaseStore
        """
        self.runtime = runtime

    def _get_store(self) -> BaseStore:
        """Get the store instance.

        Args:Returns:
            BaseStore instance

        Raises:
            ValueError: If no store is available or runtime not provided
        """
        store = self.runtime.store
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

    def _search_store_paginated(
        self,
        store: BaseStore,
        namespace: tuple[str, ...],
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        page_size: int = 100,
    ) -> list[Item]:
        """Search store with automatic pagination to retrieve all results.

        Args:
            store: The store to search.
            namespace: Hierarchical path prefix to search within.
            query: Optional query for natural language search.
            filter: Key-value pairs to filter results.
            page_size: Number of items to fetch per page (default: 100).

        Returns:
            List of all items matching the search criteria.

        Example:
            ```python
            store = _get_store(runtime)
            namespace = _get_namespace()
            all_items = _search_store_paginated(store, namespace)
            ```
        """
        all_items: list[Item] = []
        offset = 0
        while True:
            page_items = store.search(
                namespace,
                query=query,
                filter=filter,
                limit=page_size,
                offset=offset,
            )
            if not page_items:
                break
            all_items.extend(page_items)
            if len(page_items) < page_size:
                break
            offset += page_size

        return all_items

    def ls(self, path: str) -> list[str]:
        """List files from store.

        Args:
            path: Absolute path to directory.

        Returns:
            List of file paths.
        """
        store = self._get_store()
        namespace = self._get_namespace()

        # Search store with path filter
        items = self._search_store_paginated(store, namespace, filter={"prefix": path})

        return truncate_if_too_long([item.key for item in items])

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.

        Args:
            file_path: Absolute file path
            offset: Line offset to start reading from (0-indexed)limit: Maximum number of lines to read

        Returns:
            Formatted file content with line numbers, or error message.
        """
        store = self._get_store()
        namespace = self._get_namespace()
        item: Item | None = store.get(namespace, file_path)

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
    ) -> WriteResult:
        """Create a new file with content.

        Args:
            file_path: Absolute file path
            content: File content as a string

        Returns:
            WriteResult with files_update=None (external storage).
        """
        store = self._get_store()
        namespace = self._get_namespace()

        # Check if file exists
        existing = store.get(namespace, file_path)
        if existing is not None:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

        # Create new file and persist to store
        file_data = create_file_data(content)
        store_value = self._convert_file_data_to_store_value(file_data)
        store.put(namespace, file_path, store_value)

        return WriteResult(
            path=file_path,
            files_update=None,  # External storage: already persisted to store
        )

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences.

        Args:
            file_path: Absolute file path
            old_string: String to find and replace
            new_string: Replacement string
            replace_all: If True, replace all occurrences

        Returns:
            EditResult with files_update=None (external storage).
        """
        store = self._get_store()
        namespace = self._get_namespace()

        # Get existing file
        item = store.get(namespace, file_path)
        if item is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        try:
            file_data = self._convert_store_item_to_file_data(item)
        except ValueError as e:
            return EditResult(error=f"Error: {e}")

        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)

        if isinstance(result, str):
            # Error message from perform_string_replacement
            return EditResult(error=result)

        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)

        # Update file in store
        store_value = self._convert_file_data_to_store_value(new_file_data)
        store.put(namespace, file_path, store_value)

        return EditResult(
            path=file_path,
            files_update=None,  # External storage: already persisted to store
            occurrences=occurrences,
        )

    def grep(
        self,
        pattern: str,
        path: str = "/",
        glob: str | None = None,
        output_mode: str = "files_with_matches",
    ) -> str:
        """Search for a pattern in files.

        Args:
            pattern: String pattern to search for
            path: Path to search in (default "/")
            glob: Optional glob pattern to filter files (e.g., "*.py")
            output_mode: Output format - "files_with_matches", "content", or "count"Returns:
            Formatted search results based on output_mode.
        """
        store = self._get_store()
        namespace = self._get_namespace()

        items = self._search_store_paginated(store, namespace)

        files = {}
        for item in items:
            if item is None:
                continue
            try:
                file_data = self._convert_store_item_to_file_data(item)
                files[item.key] = file_data
            except ValueError:
                continue

        return truncate_if_too_long(_grep_search_files(files, pattern, path, glob, output_mode))

    def glob(self, pattern: str, path: str = "/") -> list[str]:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "**/*.py", "*.txt", "/subdir/**/*.md")
            path: Base path to search from (default "/")Returns:
            List of absolute file paths matching the pattern.
        """
        store = self._get_store()
        namespace = self._get_namespace()

        items = self._search_store_paginated(store, namespace)

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
        return truncate_if_too_long(result.split("\n"))
