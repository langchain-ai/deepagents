"""`StoreBackend`: Adapter for LangGraph's BaseStore (persistent, cross-thread)."""

import base64
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from langgraph.config import get_store
from langgraph.runtime import get_runtime
from langgraph.store.base import BaseStore, Item, PutOp

from deepagents.backends.protocol import (
    BackendProtocol,
    DeleteResult,
    EditResult,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.utils import (
    _get_backend_read_file_type,
    _glob_search_files,
    create_file_data,
    file_data_to_string,
    grep_matches_from_files,
    perform_string_replacement,
    slice_read_response,
    update_file_data,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


NamespaceFactory = Callable[["Runtime[Any]"], tuple[str, ...]]

# Allowed characters in namespace components: alphanumeric, plus characters
# common in user IDs (hyphen, underscore, dot, @, +, colon, tilde).
_NAMESPACE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9\-_.@+:~]+$")


def _validate_namespace(namespace: tuple[str, ...]) -> tuple[str, ...]:
    """Validate a namespace tuple returned by a NamespaceFactory.

    Each component must be a non-empty string containing only safe characters:
    alphanumeric (a-z, A-Z, 0-9), hyphen (-), underscore (_), dot (.),
    at sign (@), plus (+), colon (:), and tilde (~).

    Characters like `*`, `?`, `[`, `]`, `{`, `}`, etc. are
    rejected to prevent wildcard or glob injection in store lookups.

    Args:
        namespace: The namespace tuple to validate.

    Returns:
        The validated namespace tuple (unchanged).

    Raises:
        ValueError: If the namespace is empty, contains non-string elements,
            empty strings, or strings with disallowed characters.
    """
    if not namespace:
        msg = "Namespace tuple must not be empty."
        raise ValueError(msg)

    for i, component in enumerate(namespace):
        if not isinstance(component, str):
            msg = f"Namespace component at index {i} must be a string, got {type(component).__name__}."
            raise TypeError(msg)
        if not component:
            msg = f"Namespace component at index {i} must not be empty."
            raise ValueError(msg)
        if not _NAMESPACE_COMPONENT_RE.match(component):
            msg = (
                f"Namespace component at index {i} contains disallowed characters: {component!r}. "
                f"Only alphanumeric characters, hyphens, underscores, dots, @, +, colons, and tildes are allowed."
            )
            raise ValueError(msg)

    return namespace


class StoreBackend(BackendProtocol):
    """Backend that stores files in LangGraph's BaseStore (persistent).

    Uses LangGraph's Store for persistent, cross-conversation storage.
    Files are organized via namespaces and persist across all threads.

    Files are scoped by the caller-supplied `namespace` factory (e.g. per-user
    or per-assistant isolation).
    """

    def __init__(
        self,
        *,
        namespace: NamespaceFactory,
        store: BaseStore | None = None,
    ) -> None:
        r"""Initialize `StoreBackend`.

        Args:
            namespace: Callable that receives a `Runtime` and returns a namespace
                tuple for scoping store operations. Wildcards (`*`) are forbidden.
            store: Optional `BaseStore` instance. When provided, this store
                is used directly. When `None`, the store is obtained at call time
                via `get_store()`, which requires a LangGraph graph execution
                context.

        Example:
            `namespace=lambda rt: (rt.server_info.user.identity, "filesystem")`
        """
        self._store = store
        self._namespace = namespace

    def _get_store(self) -> BaseStore:
        """Return the store instance.

        Uses the store passed at init if available, otherwise falls back to
        `get_store()` which reads from the LangGraph execution context.
        """
        if self._store is not None:
            return self._store
        try:
            return get_store()
        except (RuntimeError, KeyError):
            msg = (
                "StoreBackend must be used inside a LangGraph graph execution "
                "(e.g. via create_deep_agent), or initialized with an explicit "
                "store and namespace: StoreBackend(store=my_store, "
                "namespace=lambda _rt: ('filesystem',))"
            )
            raise RuntimeError(msg) from None

    def _get_namespace(self) -> tuple[str, ...]:
        """Get the namespace for store operations.

        Resolves the `Runtime` from the graph execution context and passes it to
        the namespace factory. When called outside a graph (e.g. direct backend
        use), the runtime is unavailable and `None` is passed instead, so
        factories that ignore their argument still work. A factory that reads
        the runtime (e.g. `lambda rt: (rt.server_info.user.identity, ...)`)
        raises a clear `RuntimeError` in that case rather than an opaque
        `AttributeError` on `None`.
        """
        try:
            runtime: Runtime[Any] | None = get_runtime()
        except (RuntimeError, KeyError):
            runtime = None
        try:
            namespace = self._namespace(cast("Runtime[Any]", runtime))
        except AttributeError as exc:
            if runtime is None:
                msg = (
                    "The namespace factory tried to read the Runtime, but it is "
                    "unavailable (running outside a LangGraph graph execution). "
                    "Use StoreBackend inside a graph (e.g. via create_deep_agent), "
                    "or pass a namespace factory that does not read the Runtime."
                )
                raise RuntimeError(msg) from exc
            raise
        return _validate_namespace(namespace)

    def _convert_store_item_to_file_data(self, store_item: Item) -> FileData:
        """Convert a store `Item` to `FileData` format.

        Args:
            store_item: The store `Item` containing file data.

        Returns:
            `FileData` dict with content and encoding.

                Includes `created_at` and `modified_at` when present in the store item.
        """
        raw_content = store_item.value.get("content")
        if raw_content is None:
            msg = f"Store item does not contain valid content field. Got: {store_item.value.keys()}"
            raise ValueError(msg)

        if not isinstance(raw_content, str):
            msg = f"Store item `content` must be a `str`, got {type(raw_content).__name__}. Legacy `list[str]` content is no longer supported."
            raise TypeError(msg)
        content = raw_content

        result = FileData(
            content=content,
            encoding=store_item.value.get("encoding", "utf-8"),
        )
        if "created_at" in store_item.value and isinstance(store_item.value["created_at"], str):
            result["created_at"] = store_item.value["created_at"]
        if "modified_at" in store_item.value and isinstance(store_item.value["modified_at"], str):
            result["modified_at"] = store_item.value["modified_at"]
        return result

    def _convert_file_data_to_store_value(self, file_data: FileData) -> dict[str, Any]:
        """Convert `FileData` to a dict suitable for `store.put()`.

        Args:
            file_data: The `FileData` to convert.

        Returns:
            Dictionary with content and encoding.

                Includes `created_at` and `modified_at` when present in
                the `FileData`.
        """
        result: dict[str, Any] = {
            "content": file_data["content"],
            "encoding": file_data["encoding"],
        }
        if "created_at" in file_data:
            result["created_at"] = file_data["created_at"]
        if "modified_at" in file_data:
            result["modified_at"] = file_data["modified_at"]
        return result

    def _search_store_paginated(
        self,
        store: BaseStore,
        namespace: tuple[str, ...],
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002  # Matches LangGraph BaseStore.search() API
        page_size: int = 100,
    ) -> list[Item]:
        """Search store with automatic pagination to retrieve all results.

        Args:
            store: The store to search.
            namespace: Hierarchical path prefix to search within.
            query: Optional query for natural language search.
            filter: Key-value pairs to filter results.
            page_size: Number of items to fetch per page.

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

    async def _asearch_store_paginated(
        self,
        store: BaseStore,
        namespace: tuple[str, ...],
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002  # Matches LangGraph BaseStore.asearch() API
        page_size: int = 100,
    ) -> list[Item]:
        """Async version of `_search_store_paginated`."""
        all_items: list[Item] = []
        offset = 0
        while True:
            page_items = await store.asearch(
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

    def ls(self, path: str) -> LsResult:
        """List files and directories in the specified directory (non-recursive).

        Args:
            path: Absolute path to directory.

        Returns:
            List of `FileInfo`-like dicts for files and directories directly
                in the directory.

                Directories have a trailing `/` in their path and `is_dir=True`.
        """
        store = self._get_store()
        namespace = self._get_namespace()

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
            size = len(fd.get("content", ""))
            infos.append(
                {
                    "path": item.key,
                    "is_dir": False,
                    "size": int(size),
                    "modified_at": fd.get("modified_at", ""),
                }
            )

        # Add directories to the results
        infos.extend(FileInfo(path=subdir, is_dir=True, size=0, modified_at="") for subdir in sorted(subdirs))

        infos.sort(key=lambda x: x.get("path", ""))
        return LsResult(entries=infos)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read file content for the requested line range.

        Args:
            file_path: Absolute file path.
            offset: Line offset to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            `ReadResult` with raw (unformatted) content for the requested window.

                Line-number formatting is applied by the middleware.
        """
        store = self._get_store()
        namespace = self._get_namespace()
        item: Item | None = store.get(namespace, file_path)

        if item is None:
            return ReadResult(error=f"File '{file_path}' not found")

        try:
            file_data = self._convert_store_item_to_file_data(item)
        except ValueError as e:
            return ReadResult(error=str(e))

        if _get_backend_read_file_type(file_path) != "text":
            return ReadResult(file_data=file_data)

        return slice_read_response(file_data, offset, limit)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Async version of read using native store async methods.

        This avoids sync calls in async context by using `store.aget` directly.
        """
        store = self._get_store()
        namespace = self._get_namespace()
        item: Item | None = await store.aget(namespace, file_path)

        if item is None:
            return ReadResult(error=f"File '{file_path}' not found")

        try:
            file_data = self._convert_store_item_to_file_data(item)
        except ValueError as e:
            return ReadResult(error=str(e))

        if _get_backend_read_file_type(file_path) != "text":
            return ReadResult(file_data=file_data)

        return slice_read_response(file_data, offset, limit)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Write content to a file, creating it or overwriting it if it already exists.

        Returns `WriteResult` on success or error.
        """
        store = self._get_store()
        namespace = self._get_namespace()

        existing = store.get(namespace, file_path)
        if existing is not None:
            existing_file_data = self._convert_store_item_to_file_data(existing)
            file_data = update_file_data(existing_file_data, content)
        else:
            file_data = create_file_data(content)
        store_value = self._convert_file_data_to_store_value(file_data)
        store.put(namespace, file_path, store_value)
        return WriteResult(path=file_path)

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Async version of write using native store async methods.

        This avoids sync calls in async context by using `store.aget`/`aput` directly.
        """
        store = self._get_store()
        namespace = self._get_namespace()

        existing = await store.aget(namespace, file_path)
        if existing is not None:
            existing_file_data = self._convert_store_item_to_file_data(existing)
            file_data = update_file_data(existing_file_data, content)
        else:
            file_data = create_file_data(content)
        store_value = self._convert_file_data_to_store_value(file_data)
        await store.aput(namespace, file_path, store_value)
        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Edit a file by replacing string occurrences.

        Returns `EditResult` on success or error.
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
            return EditResult(error=result)

        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)

        # Update file in store
        store_value = self._convert_file_data_to_store_value(new_file_data)
        store.put(namespace, file_path, store_value)
        return EditResult(path=file_path, occurrences=int(occurrences))

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Async version of edit using native store async methods.

        This avoids sync calls in async context by using `store.aget`/`aput` directly.
        """
        store = self._get_store()
        namespace = self._get_namespace()

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
        return EditResult(path=file_path, occurrences=int(occurrences))

    def delete(self, file_path: str) -> DeleteResult:
        """Delete a file or directory from the store.

        Deleting a path removes the exact key `file_path` plus every key nested
        under it (the prefix `file_path` + "/"), so a directory is removed
        recursively. Wildcards (e.g. `*`) in `file_path` are treated literally.

        Args:
            file_path: Path of the file or directory to delete.

        Returns:
            `DeleteResult` with `file_path` on success, or an error if no key is
                stored at or under it.
        """
        store = self._get_store()
        namespace = self._get_namespace()

        items = self._search_store_paginated(store, namespace)
        # A recursive delete removes the exact key plus everything nested under it.
        base = file_path.rstrip("/")
        prefix = base + "/"
        to_delete = [key for item in items if (key := str(item.key)) == base or key.startswith(prefix)]
        if not to_delete:
            return DeleteResult(error=f"Error: File '{file_path}' not found")

        store.batch([PutOp(namespace, key, None) for key in to_delete])
        return DeleteResult(path=file_path)

    async def adelete(self, file_path: str) -> DeleteResult:
        """Async version of `delete` using native store async methods."""
        store = self._get_store()
        namespace = self._get_namespace()

        items = await self._asearch_store_paginated(store, namespace)
        base = file_path.rstrip("/")
        prefix = base + "/"
        to_delete = [key for item in items if (key := str(item.key)) == base or key.startswith(prefix)]
        if not to_delete:
            return DeleteResult(error=f"Error: File '{file_path}' not found")

        await store.abatch([PutOp(namespace, key, None) for key in to_delete])
        return DeleteResult(path=file_path)

    # Removed legacy grep() convenience to keep lean surface

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Search store files for a literal text pattern."""
        store = self._get_store()
        namespace = self._get_namespace()
        items = self._search_store_paginated(store, namespace)
        files: dict[str, Any] = {}
        for item in items:
            try:
                files[item.key] = self._convert_store_item_to_file_data(item)
            except ValueError:
                continue
        return grep_matches_from_files(files, pattern, path, glob, max_count=max_count)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find files matching a glob pattern in the store."""
        store = self._get_store()
        namespace = self._get_namespace()
        items = self._search_store_paginated(store, namespace)
        files: dict[str, Any] = {}
        for item in items:
            try:
                files[item.key] = self._convert_store_item_to_file_data(item)
            except ValueError:
                continue
        result = _glob_search_files(files, pattern, path)
        if result == "No files found":
            return GlobResult(matches=[])
        paths = result.split("\n")
        infos: list[FileInfo] = []
        for p in paths:
            fd = files.get(p)
            size = len(fd.get("content", "")) if fd else 0
            infos.append(
                {
                    "path": p,
                    "is_dir": False,
                    "size": int(size),
                    "modified_at": fd.get("modified_at", "") if fd else "",
                }
            )
        return GlobResult(matches=infos)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the store.

        Binary files (images, PDFs, etc.) are stored as base64-encoded strings.
        Text files are stored as utf-8 strings.

        Args:
            files: List of `(path, content)` tuples where content is bytes.

        Returns:
            List of `FileUploadResponse` objects, one per input file.

                Response order matches input order.
        """
        store = self._get_store()
        namespace = self._get_namespace()
        responses: list[FileUploadResponse] = []

        for path, content in files:
            try:
                content_str = content.decode("utf-8")
                encoding = "utf-8"
            except UnicodeDecodeError:
                content_str = base64.standard_b64encode(content).decode("ascii")
                encoding = "base64"

            file_data = create_file_data(content_str, encoding=encoding)
            store_value = self._convert_file_data_to_store_value(file_data)

            store.put(namespace, path, store_value)
            responses.append(FileUploadResponse(path=path, error=None))

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the store.

        Args:
            paths: List of file paths to download.

        Returns:
            List of `FileDownloadResponse` objects, one per input path.

                Response order matches input order.
        """
        store = self._get_store()
        namespace = self._get_namespace()
        responses: list[FileDownloadResponse] = []

        for path in paths:
            item = store.get(namespace, path)

            if item is None:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
                continue

            file_data = self._convert_store_item_to_file_data(item)
            content_str = file_data_to_string(file_data)

            encoding = file_data["encoding"]
            content_bytes = base64.standard_b64decode(content_str) if encoding == "base64" else content_str.encode("utf-8")

            responses.append(FileDownloadResponse(path=path, content=content_bytes, error=None))

        return responses
