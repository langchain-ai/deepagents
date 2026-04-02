"""Helpers for lazily migrating `StateBackend` files into `StoreBackend`."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ContextT, ResponseT
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from langgraph.store.base import Item

from deepagents.backends.protocol import FileData
from deepagents.backends.store import StoreBackend
from deepagents.middleware.filesystem import FilesystemState

PathTransform = Callable[[str], str]


@dataclass
class StateToStoreMigrationResult:
    """Summary of a state-to-store migration attempt."""

    migrated_paths: dict[str, str] = field(default_factory=dict)
    """State paths copied into the store, mapped to their target store paths."""

    already_migrated_paths: dict[str, str] = field(default_factory=dict)
    """State paths whose target store entries already matched the state content."""

    conflicted_paths: dict[str, str] = field(default_factory=dict)
    """State paths whose target store entries already existed with different content."""

    failed_paths: dict[str, str] = field(default_factory=dict)
    """State paths that could not be migrated, mapped to an error message."""

    @property
    def clearable_paths(self) -> list[str]:
        """Return source paths that are safe to delete from state."""
        return [*self.migrated_paths, *self.already_migrated_paths]


def _normalize_state_file_data(raw_file_data: Mapping[str, Any]) -> FileData:
    """Normalize state-backed file data into modern `FileData` format."""
    raw_content = raw_file_data.get("content")
    if isinstance(raw_content, list):
        if not all(isinstance(line, str) for line in raw_content):
            msg = "Legacy state file content must be a list of strings."
            raise TypeError(msg)
        content = "\n".join(raw_content)
        encoding = "utf-8"
    elif isinstance(raw_content, str):
        content = raw_content
        raw_encoding = raw_file_data.get("encoding", "utf-8")
        if not isinstance(raw_encoding, str):
            msg = "State file encoding must be a string when present."
            raise TypeError(msg)
        encoding = raw_encoding
    else:
        msg = "State file content must be a string or legacy list[str]."
        raise TypeError(msg)

    result = FileData(content=content, encoding=encoding)
    created_at = raw_file_data.get("created_at")
    modified_at = raw_file_data.get("modified_at")
    if isinstance(created_at, str):
        result["created_at"] = created_at
    if isinstance(modified_at, str):
        result["modified_at"] = modified_at
    return result


def _resolve_target_path(source_path: str, path_transform: PathTransform | None) -> str:
    """Resolve the destination store path for a state-backed file."""
    target_path = path_transform(source_path) if path_transform is not None else source_path
    if not target_path:
        msg = "Path transform must return a non-empty path."
        raise ValueError(msg)
    if not target_path.startswith("/"):
        msg = "Path transform must return an absolute path."
        raise ValueError(msg)
    return target_path


def _file_data_matches(left: FileData, right: FileData) -> bool:
    """Compare file payloads, ignoring timestamps."""
    return left["content"] == right["content"] and left.get("encoding", "utf-8") == right.get("encoding", "utf-8")


def _record_failure(result: StateToStoreMigrationResult, source_path: str, error: Exception) -> None:
    """Record a per-path migration failure."""
    msg = str(error) or error.__class__.__name__
    result.failed_paths[source_path] = msg


def _existing_item_matches(item: Item, store_backend: StoreBackend, file_data: FileData) -> bool:
    """Check whether an existing store item already matches the state payload."""
    existing_data = store_backend._convert_store_item_to_file_data(item)
    return _file_data_matches(file_data, existing_data)


def migrate_state_files_to_store(
    state_files: Mapping[str, Mapping[str, Any]],
    store_backend: StoreBackend,
    *,
    overwrite: bool = False,
    path_transform: PathTransform | None = None,
) -> StateToStoreMigrationResult:
    """Copy state-backed files into a `StoreBackend`.

    This is intended for lazy migrations when an application switches from
    `StateBackend` to `StoreBackend` but existing threads still carry `files`
    state from earlier runs.

    Args:
        state_files: Mapping of source paths to raw state-backed file payloads.
        store_backend: Target `StoreBackend` instance.
        overwrite: Whether to overwrite existing store entries when content differs.
        path_transform: Optional function to remap a state path to a target store path.

    Returns:
        A `StateToStoreMigrationResult` describing which paths were copied,
        already present, conflicted, or failed.
    """
    result = StateToStoreMigrationResult()
    if not state_files:
        return result

    store = store_backend._get_store()
    namespace = store_backend._get_namespace()

    for source_path, raw_file_data in state_files.items():
        try:
            target_path = _resolve_target_path(source_path, path_transform)
            normalized_file_data = _normalize_state_file_data(raw_file_data)
            existing_item = store.get(namespace, target_path)
            if existing_item is not None and not overwrite:
                if _existing_item_matches(existing_item, store_backend, normalized_file_data):
                    result.already_migrated_paths[source_path] = target_path
                else:
                    result.conflicted_paths[source_path] = target_path
                continue

            store_value = store_backend._convert_file_data_to_store_value(normalized_file_data)
            store.put(namespace, target_path, store_value)
            result.migrated_paths[source_path] = target_path
        except (TypeError, ValueError) as e:
            _record_failure(result, source_path, e)

    return result


async def amigrate_state_files_to_store(
    state_files: Mapping[str, Mapping[str, Any]],
    store_backend: StoreBackend,
    *,
    overwrite: bool = False,
    path_transform: PathTransform | None = None,
) -> StateToStoreMigrationResult:
    """Async version of `migrate_state_files_to_store`."""
    result = StateToStoreMigrationResult()
    if not state_files:
        return result

    store = store_backend._get_store()
    namespace = store_backend._get_namespace()

    for source_path, raw_file_data in state_files.items():
        try:
            target_path = _resolve_target_path(source_path, path_transform)
            normalized_file_data = _normalize_state_file_data(raw_file_data)
            existing_item = await store.aget(namespace, target_path)
            if existing_item is not None and not overwrite:
                if _existing_item_matches(existing_item, store_backend, normalized_file_data):
                    result.already_migrated_paths[source_path] = target_path
                else:
                    result.conflicted_paths[source_path] = target_path
                continue

            store_value = store_backend._convert_file_data_to_store_value(normalized_file_data)
            await store.aput(namespace, target_path, store_value)
            result.migrated_paths[source_path] = target_path
        except (TypeError, ValueError) as e:
            _record_failure(result, source_path, e)

    return result


def _build_state_clear_update(paths: list[str]) -> dict[str, dict[str, None]] | None:
    """Build a `files` state update that deletes migrated paths."""
    if not paths:
        return None
    return {"files": dict.fromkeys(paths, None)}


class StateToStoreMigrationMiddleware(AgentMiddleware[FilesystemState, ContextT, ResponseT]):
    """Lazily migrate `files` state into a `StoreBackend` before each turn.

    When migration succeeds for a path, the middleware can delete that path
    from state so future turns read directly from the store.
    """

    state_schema = FilesystemState

    def __init__(
        self,
        *,
        store_backend: StoreBackend,
        overwrite: bool = False,
        clear_state: bool = True,
        path_transform: PathTransform | None = None,
    ) -> None:
        """Initialize the migration middleware.

        Args:
            store_backend: Target `StoreBackend` used for migrated files.
            overwrite: Whether to overwrite existing store entries when content differs.
            clear_state: Whether to remove successfully migrated paths from `files` state.
            path_transform: Optional function to remap source paths into store paths.
        """
        self._store_backend = store_backend
        self._overwrite = overwrite
        self._clear_state = clear_state
        self._path_transform = path_transform

    def before_agent(
        self,
        state: FilesystemState,
        _runtime: Runtime[ContextT],
        _config: RunnableConfig,
    ) -> dict[str, dict[str, None]] | None:  # ty: ignore[invalid-method-override]
        """Migrate any state-backed files before the agent runs."""
        raw_state_files = state.get("files")
        if not raw_state_files:
            return None

        result = migrate_state_files_to_store(
            raw_state_files,
            self._store_backend,
            overwrite=self._overwrite,
            path_transform=self._path_transform,
        )
        if not self._clear_state:
            return None
        return _build_state_clear_update(result.clearable_paths)

    async def abefore_agent(
        self,
        state: FilesystemState,
        _runtime: Runtime[ContextT],
        _config: RunnableConfig,
    ) -> dict[str, dict[str, None]] | None:  # ty: ignore[invalid-method-override]
        """Async version of `before_agent`."""
        raw_state_files = state.get("files")
        if not raw_state_files:
            return None

        result = await amigrate_state_files_to_store(
            raw_state_files,
            self._store_backend,
            overwrite=self._overwrite,
            path_transform=self._path_transform,
        )
        if not self._clear_state:
            return None
        return _build_state_clear_update(result.clearable_paths)


__all__ = [
    "StateToStoreMigrationMiddleware",
    "StateToStoreMigrationResult",
    "amigrate_state_files_to_store",
    "migrate_state_files_to_store",
]
