"""StoreBackend: Adapter for LangGraph's BaseStore (persistent, cross-thread)."""

import base64
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from langgraph.config import get_config, get_store
from langgraph.runtime import get_runtime
from langgraph.store.base import BaseStore, Item
from langgraph.typing import ContextT, StateT

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileData,
    FileDownloadResponse,
    FileFormat,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.utils import (
    _get_file_type,
    _glob_search_files,
    _to_legacy_file_data,
    create_file_data,
    file_data_to_string,
    grep_matches_from_files,
    perform_string_replacement,
    slice_read_response,
    update_file_data,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


@dataclass
class BackendContext(Generic[StateT, ContextT]):
    """Context passed to namespace factory functions.

    Deprecated: `BackendContext` will be removed in version 0.7.0.
    Namespace factories now receive a `Runtime` instance directly.
    Migrate from `lambda ctx: (ctx.runtime.context.user_id, "fs")`
    to `lambda rt: (rt.server_info.user.identity, "fs")`.
    """

    state: StateT
    runtime: "Runtime[ContextT]"


class _NamespaceRuntimeCompat:
    """Wrapper that duck-types as both Runtime and BackendContext.

    Allows old-style namespace factories (accessing ``.runtime`` / ``.state``)
    to work alongside new-style factories (accessing ``Runtime`` attrs directly).

    Will be removed in v0.7.
    """

    def __init__(self, runtime: "Runtime[Any] | None", state: object = None) -> None:
        self._runtime = runtime
        self._state = state

    @property
    def runtime(self) -> "Runtime[Any] | None":
        warnings.warn(
            "Accessing .runtime on the namespace factory argument is deprecated. "
            "The argument is now a Runtime instance — use its attributes directly "
            "(e.g. `rt.context` instead of `ctx.runtime.context`). "
            "This will be removed in v0.7.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._runtime

    @property
    def state(self) -> object:
        warnings.warn(
            "Accessing .state on the namespace factory argument is deprecated. "
            "Namespace resolution should not depend on state. "
            "This will be removed in v0.7.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._state

    def __getattr__(self, name: str) -> object:
        if self._runtime is None:
            msg = f"Runtime is not available (running outside graph execution), cannot access '.{name}'"
            raise AttributeError(msg)
        return getattr(self._runtime, name)


# Type alias for namespace factory functions.
# Accepts Runtime directly. Old-style BackendContext callables still work
# via _NamespaceRuntimeCompat but are deprecated (removed in v0.7).
NamespaceFactory = Callable[["Runtime[Any]"], tuple[str, ...]]

# Allowed characters in namespace components: alphanumeric, plus characters
# common in user IDs (hyphen, underscore, dot, @, +, colon, tilde).
_NAMESPACE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9\-_.@+:~]+$")


def _validate_namespace(namespace: tuple[str, ...]) -> tuple[str, ...]:
    """Validate a namespace tuple returned by a NamespaceFactory.

    Each component must be a non-empty string containing only safe characters:
    alphanumeric (a-z, A-Z, 0-9), hyphen (-), underscore (_), dot (.),
    at sign (@), plus (+), colon (:), and tilde (~).

    Characters like ``*``, ``?``, ``[``, ``]``, ``{``, ``}``, etc. are
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

    The namespace can include an optional assistant_id for multi-agent isolation.
    """

    def __init__(
        self,
        runtime: object = None,
        *,
        store: BaseStore | None = None,
        namespace: NamespaceFactory | None = None,
        file_format: FileFormat = "v2",
    ) -> None:
        r"""Initialize StoreBackend.

        Args:
            runtime: Deprecated - accepted for backward compatibility but
                ignored.  Store and context are now obtained via
                ``get_store()`` / ``get_runtime()``.
            store: Optional ``BaseStore`` instance.  When provided, this store
                is used directly.  When ``None`` (the default), the store is
                obtained at call time via ``get_store()``, which requires
                a LangGraph graph execution context.
            namespace: Optional callable that receives a ``Runtime`` and returns
                a namespace tuple for scoping store operations.
                Wildcards (``*``) are forbidden.
                If ``None``, uses legacy assistant_id detection from metadata (deprecated).

                Old-style callables that accept ``BackendContext`` still work
                but are deprecated and will be removed in v0.7.

            file_format: Storage format version. `"v1"` (default) stores
                content as `list[str]` (lines split on `\\n`) without an
                `encoding` field.  `"v2"` stores content as a plain `str`
                with an `encoding` field.

        Example:
                    namespace=lambda rt: (rt.server_info.user.identity, "filesystem")
        """
        if runtime is not None:
            warnings.warn(
                "Passing `runtime` to StoreBackend is deprecated and will be "
                "removed in v0.7. StoreBackend now obtains store "
                "and context via `get_store()` / `get_runtime()`. Simply use "
                "`StoreBackend()` or `StoreBackend(store=my_store)` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._store = store
        self._namespace = namespace
        self._file_format = file_format

    def _get_store(self) -> BaseStore:
        """Return the store instance.

        Uses the store passed at init if available, otherwise falls back to
        ``get_store()`` which reads from the LangGraph execution context.
        """
        if self._store is not None:
            return self._store
        try:
            return get_store()
        except (RuntimeError, KeyError):
            msg = (
                "StoreBackend must be used inside a LangGraph graph execution "
                "(e.g. via create_deep_agent), or initialized with an explicit "
                "store: StoreBackend(store=my_store)"
            )
            raise RuntimeError(msg) from None

    def _get_namespace(self) -> tuple[str, ...]:
        """Get the namespace for store operations.

        If namespace was provided at init, calls it with a ``_NamespaceRuntimeCompat``
        wrapper that duck-types as both ``Runtime`` (new) and ``BackendContext`` (legacy).
        Otherwise, uses legacy assistant_id detection from metadata (deprecated).
        """
        if self._namespace is not None:
            try:
                runtime = get_runtime()
            except (RuntimeError, KeyError):
                runtime = None
            compat = _NamespaceRuntimeCompat(runtime)
            return _validate_namespace(self._namespace(compat))  # type: ignore[arg-type]

        return self._get_namespace_legacy()

    def _get_namespace_legacy(self) -> tuple[str, ...]:
        """Legacy namespace resolution: check metadata for assistant_id.

        Uses ``get_config()`` to find assistant_id in metadata.
        Defaults to ``("filesystem",)``.

        .. deprecated::
            Pass `namespace` to StoreBackend instead of relying on legacy detection.
        """
        warnings.warn(
            "StoreBackend without explicit `namespace` is deprecated and will be removed in v0.7. "
            "Pass `namespace=lambda ctx: (...)` to StoreBackend.",
            DeprecationWarning,
            stacklevel=3,
        )
        namespace = "filesystem"

        try:
            cfg = get_config()
        except Exception:  # noqa: BLE001  # Intentional for resilient config fallback
            return (namespace,)

        try:
            assistant_id = cfg.get("metadata", {}).get("assistant_id")
        except Exception:  # noqa: BLE001  # Intentional for resilient config fallback
            assistant_id = None

        if assistant_id:
            return _validate_namespace((assistant_id, namespace))
        return (namespace,)

    def _convert_store_item_to_file_data(self, store_item: Item) -> FileData:
        """Convert a store Item to FileData format.

        Args:
            store_item: The store Item containing file data.

        Returns:
            FileData dict with content and encoding. Includes created_at and
            modified_at when present in the store item.
        """
        raw_content = store_item.value.get("content")
        if raw_content is None:
            msg = f"Store item does not contain valid content field. Got: {store_item.value.keys()}"
            raise ValueError(msg)

        # BACKWARDS COMPAT: legacy list[str] format
        if isinstance(raw_content, list):
            warnings.warn(
                "Store item with list[str] content is deprecated and will be removed in v0.7. Content should be stored as a plain str.",
                DeprecationWarning,
                stacklevel=2,
            )
            content = "\n".join(raw_content)
        elif isinstance(raw_content, str):
            content = raw_content
        else:
            msg = f"Store item does not contain valid content field. Got: {store_item.value.keys()}"
            raise TypeError(msg)

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
        """Convert FileData to a dict suitable for store.put().

        When `file_format="v1"`, returns the legacy format with `content`
        as `list[str]` and no `encoding` key.

        Args:
            file_data: The FileData to convert.

        Returns:
            Dictionary with content and encoding. Includes created_at and
            modified_at when present in the FileData.
        """
        if self._file_format == "v1":
            return _to_legacy_file_data(file_data)
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

    def ls(self, path: str) -> LsResult:
        """List files and directories in the specified directory (non-recursive).

        Args:
            path: Absolute path to directory.

        Returns:
            List of FileInfo-like dicts for files and directories directly in the directory.
            Directories have a trailing / in their path and is_dir=True.
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
            # BACKWARDS COMPAT: handle legacy list[str] content for size computation
            raw = fd.get("content", "")
            size = len("\n".join(raw)) if isinstance(raw, list) else len(raw)
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
            ReadResult with raw (unformatted) content for the requested
            window. Line-number formatting is applied by the middleware.
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

        if _get_file_type(file_path) != "text":
            return ReadResult(file_data=file_data)

        sliced = slice_read_response(file_data, offset, limit)
        if isinstance(sliced, ReadResult):
            return sliced
        sliced_fd = FileData(
            content=sliced,
            encoding=file_data.get("encoding", "utf-8"),
        )
        if "created_at" in file_data:
            sliced_fd["created_at"] = file_data["created_at"]
        if "modified_at" in file_data:
            sliced_fd["modified_at"] = file_data["modified_at"]
        return ReadResult(file_data=sliced_fd)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Async version of read using native store async methods.

        This avoids sync calls in async context by using store.aget directly.
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

        if _get_file_type(file_path) != "text":
            return ReadResult(file_data=file_data)

        sliced = slice_read_response(file_data, offset, limit)
        if isinstance(sliced, ReadResult):
            return sliced
        sliced_fd = FileData(
            content=sliced,
            encoding=file_data.get("encoding", "utf-8"),
        )
        if "created_at" in file_data:
            sliced_fd["created_at"] = file_data["created_at"]
        if "modified_at" in file_data:
            sliced_fd["modified_at"] = file_data["modified_at"]
        return ReadResult(file_data=sliced_fd)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file with content.

        Returns WriteResult on success or error.
        """
        store = self._get_store()
        namespace = self._get_namespace()

        # Check if file exists
        existing = store.get(namespace, file_path)
        if existing is not None:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

        # Create new file
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

        This avoids sync calls in async context by using store.aget/aput directly.
        """
        store = self._get_store()
        namespace = self._get_namespace()

        # Check if file exists using async method
        existing = await store.aget(namespace, file_path)
        if existing is not None:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

        # Create new file using async method
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

        Returns EditResult on success or error.
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

        This avoids sync calls in async context by using store.aget/aput directly.
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

    # Removed legacy grep() convenience to keep lean surface

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
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
        return grep_matches_from_files(files, pattern, path, glob)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
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
            if fd:
                # BACKWARDS COMPAT: handle legacy list[str] content for size computation
                raw = fd.get("content", "")
                size = len("\n".join(raw)) if isinstance(raw, list) else len(raw)
            else:
                size = 0
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
            files: List of (path, content) tuples where content is bytes.

        Returns:
            List of FileUploadResponse objects, one per input file.
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
            List of FileDownloadResponse objects, one per input path.
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
