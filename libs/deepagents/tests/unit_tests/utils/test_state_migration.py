"""Unit tests for lazy StateBackend -> StoreBackend migration helpers."""

from typing import Any

from langgraph.store.memory import InMemoryStore

from deepagents.backends.store import StoreBackend
from deepagents.utils import (
    StateToStoreMigrationMiddleware,
    amigrate_state_files_to_store,
    migrate_state_files_to_store,
)


def _make_store_backend(*, file_format: str = "v2") -> tuple[StoreBackend, InMemoryStore]:
    store = InMemoryStore()
    backend = StoreBackend(
        store=store,
        namespace=lambda _ctx: ("filesystem",),
        file_format=file_format,  # type: ignore[arg-type]
    )
    return backend, store


class TestMigrateStateFilesToStore:
    def test_migrates_v2_state_files_and_preserves_metadata(self) -> None:
        backend, store = _make_store_backend()
        state_files = {
            "/notes/todo.md": {
                "content": "ship it",
                "encoding": "utf-8",
                "created_at": "2026-01-01T00:00:00+00:00",
                "modified_at": "2026-01-02T00:00:00+00:00",
            }
        }

        result = migrate_state_files_to_store(state_files, backend)

        assert result.migrated_paths == {"/notes/todo.md": "/notes/todo.md"}
        assert result.clearable_paths == ["/notes/todo.md"]
        item = store.get(("filesystem",), "/notes/todo.md")
        assert item is not None
        assert item.value == {
            "content": "ship it",
            "encoding": "utf-8",
            "created_at": "2026-01-01T00:00:00+00:00",
            "modified_at": "2026-01-02T00:00:00+00:00",
        }

    def test_migrates_legacy_state_files(self) -> None:
        backend, store = _make_store_backend()
        state_files = {
            "/legacy.txt": {
                "content": ["hello", "world"],
                "created_at": "2026-01-01T00:00:00+00:00",
                "modified_at": "2026-01-02T00:00:00+00:00",
            }
        }

        result = migrate_state_files_to_store(state_files, backend)

        assert result.migrated_paths == {"/legacy.txt": "/legacy.txt"}
        item = store.get(("filesystem",), "/legacy.txt")
        assert item is not None
        assert item.value["content"] == "hello\nworld"
        assert item.value["encoding"] == "utf-8"

    def test_respects_store_backend_file_format(self) -> None:
        backend, store = _make_store_backend(file_format="v1")
        state_files = {
            "/legacy-target.txt": {
                "content": "alpha\nbeta",
                "encoding": "utf-8",
            }
        }

        migrate_state_files_to_store(state_files, backend)

        item = store.get(("filesystem",), "/legacy-target.txt")
        assert item is not None
        assert item.value["content"] == ["alpha", "beta"]
        assert "encoding" not in item.value

    def test_leaves_conflicts_in_state_by_default(self) -> None:
        backend, store = _make_store_backend()
        store.put(("filesystem",), "/notes/todo.md", {"content": "newer", "encoding": "utf-8"})
        state_files = {
            "/notes/todo.md": {
                "content": "older",
                "encoding": "utf-8",
            }
        }

        result = migrate_state_files_to_store(state_files, backend)

        assert result.conflicted_paths == {"/notes/todo.md": "/notes/todo.md"}
        assert result.clearable_paths == []
        item = store.get(("filesystem",), "/notes/todo.md")
        assert item is not None
        assert item.value["content"] == "newer"

    def test_overwrites_conflicts_when_requested(self) -> None:
        backend, store = _make_store_backend()
        store.put(("filesystem",), "/notes/todo.md", {"content": "newer", "encoding": "utf-8"})
        state_files = {
            "/notes/todo.md": {
                "content": "older",
                "encoding": "utf-8",
            }
        }

        result = migrate_state_files_to_store(state_files, backend, overwrite=True)

        assert result.migrated_paths == {"/notes/todo.md": "/notes/todo.md"}
        item = store.get(("filesystem",), "/notes/todo.md")
        assert item is not None
        assert item.value["content"] == "older"

    async def test_async_migration_supports_path_remapping(self) -> None:
        backend, store = _make_store_backend()
        state_files = {
            "/memories/profile.md": {
                "content": "Name: Ada",
                "encoding": "utf-8",
            }
        }

        result = await amigrate_state_files_to_store(
            state_files,
            backend,
            path_transform=lambda path: path.removeprefix("/memories"),
        )

        assert result.migrated_paths == {"/memories/profile.md": "/profile.md"}
        assert store.get(("filesystem",), "/profile.md") is not None


class TestStateToStoreMigrationMiddleware:
    def test_before_agent_clears_only_paths_that_are_safe_to_remove(self) -> None:
        backend, store = _make_store_backend()
        store.put(("filesystem",), "/existing.txt", {"content": "store version", "encoding": "utf-8"})
        middleware = StateToStoreMigrationMiddleware(store_backend=backend)
        state: dict[str, Any] = {
            "files": {
                "/migrate.txt": {"content": "move me", "encoding": "utf-8"},
                "/existing.txt": {"content": "state version", "encoding": "utf-8"},
            }
        }

        update = middleware.before_agent(state, None, {})  # type: ignore[arg-type]

        assert update == {"files": {"/migrate.txt": None}}
        migrated = store.get(("filesystem",), "/migrate.txt")
        existing = store.get(("filesystem",), "/existing.txt")
        assert migrated is not None
        assert migrated.value["content"] == "move me"
        assert existing is not None
        assert existing.value["content"] == "store version"

    async def test_abefore_agent_supports_async_migration(self) -> None:
        backend, store = _make_store_backend()
        middleware = StateToStoreMigrationMiddleware(store_backend=backend)
        state: dict[str, Any] = {
            "files": {
                "/async.txt": {"content": "async move", "encoding": "utf-8"},
            }
        }

        update = await middleware.abefore_agent(state, None, {})  # type: ignore[arg-type]

        assert update == {"files": {"/async.txt": None}}
        item = store.get(("filesystem",), "/async.txt")
        assert item is not None
        assert item.value["content"] == "async move"
