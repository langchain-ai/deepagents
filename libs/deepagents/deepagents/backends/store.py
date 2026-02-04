"""StoreBackend: Adapter for LangGraph's BaseStore (persistent, cross-thread)."""

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langgraph.config import get_config
from langgraph.store.base import BaseStore, Item

from deepagents.backends.protocol import (
    BackendContext,
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)
from deepagents.backends.utils import (
    _glob_search_files,
    create_file_data,
    file_data_to_string,
    format_read_response,
    grep_matches_from_files,
    perform_string_replacement,
    update_file_data,
)

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

# Type alias for namespace factory functions
NamespaceFactory = Callable[[BackendContext], tuple[str, ...]]


class StoreBackend(BackendProtocol):
    """Backend that stores files in LangGraph's BaseStore (persistent).

    Uses LangGraph's Store for persistent, cross-conversation storage.
    Files are organized via namespaces and persist across all threads.

    The namespace can include an optional assistant_id for multi-agent isolation.

    .. versionchanged:: 0.4
        The `runtime` parameter is now optional and deprecated.
        Pass `ctx` to individual methods instead.
    """

    def __init__(
        self,
        runtime: "ToolRuntime | None" = None,
        *,
        namespace: NamespaceFactory | None = None,
    ):
        """Initialize StoreBackend.

        Args:
            runtime: Optional ToolRuntime for backwards compatibility.
                Deprecated: pass `ctx` to individual methods instead.
            namespace: Optional callable that takes a BackendContext and returns
                a namespace tuple. This provides full flexibility for namespace resolution.

        Example:
                    namespace=lambda ctx: ("filesystem", ctx.runtime.context.user_id)
                If None, uses legacy assistant_id detection from metadata (deprecated).

        .. deprecated:: 0.4
            The `runtime` parameter is deprecated and will be removed in 0.5.
            Pass `ctx` to individual methods instead.
        """
        if runtime is not None:
            warnings.warn(
                "Passing `runtime` to StoreBackend is deprecated and will be removed in 0.5. Pass `ctx` to individual methods instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._runtime = runtime
        self._namespace = namespace

    def _get_store(self, ctx: "BackendContext | None" = None) -> BaseStore:
        """Get the store instance.

        Args:
            ctx: Optional backend context.

        Returns:
            BaseStore instance from context or runtime.

        Raises:
            ValueError: If no store is available.
        """
        if ctx is not None and ctx.runtime is not None:
            store = ctx.runtime.store
            if store is not None:
                return store
        if self._runtime is not None:
            store = self._runtime.store
            if store is not None:
                return store
        msg = "Store is required but not available. Pass ctx with runtime.store set, or set runtime in __init__."
        raise ValueError(msg)

    def _get_namespace(self, ctx: "BackendContext | None" = None) -> tuple[str, ...]:
        """Get the namespace for store operations.

        Args:
            ctx: Optional backend context.

        Returns:
            Namespace tuple for store operations.
        """
        if self._namespace is not None:
            if ctx is not None:
                return self._namespace(ctx)
            # Fall back to constructing ctx from runtime (deprecated path, removed in 0.5)
            if self._runtime is not None:
                state = getattr(self._runtime, "state", {})
                fallback_ctx = BackendContext(state=state, runtime=self._runtime)  # type: ignore[arg-type]
                return self._namespace(fallback_ctx)
            msg = "namespace factory requires ctx or runtime"
            raise ValueError(msg)

        return self._get_namespace_legacy(ctx)

    def _get_namespace_legacy(self, ctx: "BackendContext | None" = None) -> tuple[str, ...]:
        """Legacy namespace resolution: check metadata for assistant_id.

        Args:
            ctx: Optional backend context.

        Preference order:
        1) Use config from ctx.runtime if present.
        2) Use `self._runtime.config` if present.
        3) Fallback to `langgraph.config.get_config()` if available.
        4) Default to ("filesystem",).

        If an assistant_id is available in the config metadata, return
        (assistant_id, "filesystem") to provide per-assistant isolation.

        .. deprecated::
            Pass `namespace` to StoreBackend instead of relying on legacy detection.
        """
        warnings.warn(
            "StoreBackend without explicit `namespace` is deprecated. Pass `namespace=lambda ctx: (...)` to StoreBackend.",
            DeprecationWarning,
            stacklevel=3,
        )
        namespace = "filesystem"

        # Try to get config from ctx or runtime
        runtime_cfg = None
        if ctx is not None and ctx.runtime is not None:
            runtime_cfg = getattr(ctx.runtime, "config", None)
        if runtime_cfg is None and self._runtime is not None:
            runtime_cfg = getattr(self._runtime, "config", None)

        if isinstance(runtime_cfg, dict):
            assistant_id = runtime_cfg.get("metadata", {}).get("assistant_id")
            if assistant_id:
                return (assistant_id, namespace)
            return (namespace,)

        # Fallback to langgraph's context, but guard against errors when
        # called outside of a runnable context
        try:
            cfg = get_config()
        except Exception:
            return (namespace,)

        try:
            assistant_id = cfg.get("metadata", {}).get("assistant_id")  # type: ignore[assignment]
        except Exception:
            assistant_id = None

        if assistant_id:
            return (assistant_id, namespace)
        return (namespace,)

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

    def ls(
        self,
        path: str,
        *,
        ctx: "BackendContext | None" = None,
    ) -> list[FileInfo]:
        """List files and directories in the specified directory (non-recursive).

        Args:
            path: Absolute path to directory.
            ctx: Optional backend context for accessing store.

        Returns:
            List of FileInfo-like dicts for files and directories directly in the directory.
            Directories have a trailing / in their path and is_dir=True.
        """
        store = self._get_store(ctx)
        namespace = self._get_namespace(ctx)

        # Retrieve all items and filter by path prefix locally to avoid
        # coupling to store-specific filter semantics
        items = self._search_store_paginated(store, namespace)
        infos: list[FileInfo] = []
        subdirs: set[str] = set()

        # Normalize path to have trailing slash for proper prefix matching
        normalized_path = path if path.endswith("/") else path + "/"

        for item in items:
            # Check if file is in the specified directory or a subdirectory
            if not str(item.key).startswith(normalized_path):
                continue

            # Get the relative path after the directory
            relative = str(item.key)[len(normalized_path) :]

            # If relative path contains '/', it's in a subdirectory
            if "/" in relative:
                # Extract the immediate subdirectory name
                subdir_name = relative.split("/")[0]
                subdirs.add(normalized_path + subdir_name + "/")
                continue

            # This is a file directly in the current directory
            try:
                fd = self._convert_store_item_to_file_data(item)
            except ValueError:
                continue
            size = len("\n".join(fd.get("content", [])))
            infos.append(
                {
                    "path": item.key,
                    "is_dir": False,
                    "size": int(size),
                    "modified_at": fd.get("modified_at", ""),
                }
            )

        # Add directories to the results
        for subdir in sorted(subdirs):
            infos.append(
                {
                    "path": subdir,
                    "is_dir": True,
                    "size": 0,
                    "modified_at": "",
                }
            )

        infos.sort(key=lambda x: x.get("path", ""))
        return infos

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        *,
        ctx: "BackendContext | None" = None,
    ) -> str:
        """Read file content with line numbers.

        Args:
            file_path: Absolute file path.
            offset: Line offset to start reading from (0-indexed).
            limit: Maximum number of lines to read.
            ctx: Optional backend context for accessing store.

        Returns:
            Formatted file content with line numbers, or error message.
        """
        store = self._get_store(ctx)
        namespace = self._get_namespace(ctx)
        item: Item | None = store.get(namespace, file_path)

        if item is None:
            return f"Error: File '{file_path}' not found"

        try:
            file_data = self._convert_store_item_to_file_data(item)
        except ValueError as e:
            return f"Error: {e}"

        return format_read_response(file_data, offset, limit)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        *,
        ctx: "BackendContext | None" = None,
    ) -> str:
        """Async version of read using native store async methods.

        Args:
            file_path: Absolute file path.
            offset: Line offset to start reading from (0-indexed).
            limit: Maximum number of lines to read.
            ctx: Optional backend context for accessing store.

        Returns:
            Formatted file content with line numbers, or error message.
        """
        store = self._get_store(ctx)
        namespace = self._get_namespace(ctx)
        item: Item | None = await store.aget(namespace, file_path)

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
        *,
        ctx: "BackendContext | None" = None,
    ) -> WriteResult:
        """Create a new file with content.

        Args:
            file_path: Absolute path where the file should be created.
            content: String content to write to the file.
            ctx: Optional backend context for accessing store.

        Returns:
            WriteResult. External storage sets files_update=None.
        """
        store = self._get_store(ctx)
        namespace = self._get_namespace(ctx)

        # Check if file exists
        existing = store.get(namespace, file_path)
        if existing is not None:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

        # Create new file
        file_data = create_file_data(content)
        store_value = self._convert_file_data_to_store_value(file_data)
        store.put(namespace, file_path, store_value)
        return WriteResult(path=file_path, files_update=None)

    async def awrite(
        self,
        file_path: str,
        content: str,
        *,
        ctx: "BackendContext | None" = None,
    ) -> WriteResult:
        """Async version of write using native store async methods.

        Args:
            file_path: Absolute path where the file should be created.
            content: String content to write to the file.
            ctx: Optional backend context for accessing store.

        Returns:
            WriteResult. External storage sets files_update=None.
        """
        store = self._get_store(ctx)
        namespace = self._get_namespace(ctx)

        # Check if file exists using async method
        existing = await store.aget(namespace, file_path)
        if existing is not None:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

        # Create new file using async method
        file_data = create_file_data(content)
        store_value = self._convert_file_data_to_store_value(file_data)
        await store.aput(namespace, file_path, store_value)
        return WriteResult(path=file_path, files_update=None)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        *,
        ctx: "BackendContext | None" = None,
    ) -> EditResult:
        """Edit a file by replacing string occurrences.

        Args:
            file_path: Absolute path to the file to edit.
            old_string: Exact string to search for and replace.
            new_string: String to replace old_string with.
            replace_all: If True, replace all occurrences.
            ctx: Optional backend context for accessing store.

        Returns:
            EditResult. External storage sets files_update=None.
        """
        store = self._get_store(ctx)
        namespace = self._get_namespace(ctx)

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
            return EditResult(error=result)

        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)

        # Update file in store
        store_value = self._convert_file_data_to_store_value(new_file_data)
        store.put(namespace, file_path, store_value)
        return EditResult(path=file_path, files_update=None, occurrences=int(occurrences))

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        *,
        ctx: "BackendContext | None" = None,
    ) -> EditResult:
        """Async version of edit using native store async methods.

        Args:
            file_path: Absolute path to the file to edit.
            old_string: Exact string to search for and replace.
            new_string: String to replace old_string with.
            replace_all: If True, replace all occurrences.
            ctx: Optional backend context for accessing store.

        Returns:
            EditResult. External storage sets files_update=None.
        """
        store = self._get_store(ctx)
        namespace = self._get_namespace(ctx)

        # Get existing file using async method
        item = await store.aget(namespace, file_path)
        if item is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        try:
            file_data = self._convert_store_item_to_file_data(item)
        except ValueError as e:
            return EditResult(error=f"Error: {e}")

        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)

        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)

        # Update file in store using async method
        store_value = self._convert_file_data_to_store_value(new_file_data)
        await store.aput(namespace, file_path, store_value)
        return EditResult(path=file_path, files_update=None, occurrences=int(occurrences))

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        ctx: "BackendContext | None" = None,
    ) -> list[GrepMatch] | str:
        """Search for a literal text pattern in files.

        Args:
            pattern: Literal string to search for.
            path: Optional directory path to search in.
            glob: Optional glob pattern to filter files.
            ctx: Optional backend context for accessing store.

        Returns:
            List of GrepMatch on success, error string on failure.
        """
        store = self._get_store(ctx)
        namespace = self._get_namespace(ctx)
        items = self._search_store_paginated(store, namespace)
        files: dict[str, Any] = {}
        for item in items:
            try:
                files[item.key] = self._convert_store_item_to_file_data(item)
            except ValueError:
                continue
        return grep_matches_from_files(files, pattern, path or "/", glob)

    def glob(
        self,
        pattern: str,
        path: str = "/",
        *,
        ctx: "BackendContext | None" = None,
    ) -> list[FileInfo]:
        """Get FileInfo for files matching glob pattern.

        Args:
            pattern: Glob pattern to match files.
            path: Base directory to search from.
            ctx: Optional backend context for accessing store.

        Returns:
            List of FileInfo for matching files.
        """
        store = self._get_store(ctx)
        namespace = self._get_namespace(ctx)
        items = self._search_store_paginated(store, namespace)
        files: dict[str, Any] = {}
        for item in items:
            try:
                files[item.key] = self._convert_store_item_to_file_data(item)
            except ValueError:
                continue
        result = _glob_search_files(files, pattern, path)
        if result == "No files found":
            return []
        paths = result.split("\n")
        infos: list[FileInfo] = []
        for p in paths:
            fd = files.get(p)
            size = len("\n".join(fd.get("content", []))) if fd else 0
            infos.append(
                {
                    "path": p,
                    "is_dir": False,
                    "size": int(size),
                    "modified_at": fd.get("modified_at", "") if fd else "",
                }
            )
        return infos

    def upload_files(
        self,
        files: list[tuple[str, bytes]],
        *,
        ctx: "BackendContext | None" = None,
    ) -> list[FileUploadResponse]:
        """Upload multiple files to the store.

        Args:
            files: List of (path, content) tuples where content is bytes.
            ctx: Optional backend context for accessing store.

        Returns:
            List of FileUploadResponse objects, one per input file.
            Response order matches input order.
        """
        store = self._get_store(ctx)
        namespace = self._get_namespace(ctx)
        responses: list[FileUploadResponse] = []

        for path, content in files:
            content_str = content.decode("utf-8")
            # Create file data
            file_data = create_file_data(content_str)
            store_value = self._convert_file_data_to_store_value(file_data)

            # Store the file
            store.put(namespace, path, store_value)
            responses.append(FileUploadResponse(path=path, error=None))

        return responses

    def download_files(
        self,
        paths: list[str],
        *,
        ctx: "BackendContext | None" = None,
    ) -> list[FileDownloadResponse]:
        """Download multiple files from the store.

        Args:
            paths: List of file paths to download.
            ctx: Optional backend context for accessing store.

        Returns:
            List of FileDownloadResponse objects, one per input path.
            Response order matches input order.
        """
        store = self._get_store(ctx)
        namespace = self._get_namespace(ctx)
        responses: list[FileDownloadResponse] = []

        for path in paths:
            item = store.get(namespace, path)

            if item is None:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
                continue

            file_data = self._convert_store_item_to_file_data(item)
            # Convert file data to bytes
            content_str = file_data_to_string(file_data)
            content_bytes = content_str.encode("utf-8")

            responses.append(FileDownloadResponse(path=path, content=content_bytes, error=None))

        return responses
