"""Tests for the TaskMiddleware and related functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from deepagents_cli.tasks import (
    Task,
    TaskList,
    TaskMiddleware,
    TaskStorage,
    _compute_blocked_status,
    _validate_dependencies,
)


class TestTaskStorage:
    """Tests for TaskStorage class."""

    def test_create_task_list(self, tmp_path: Path) -> None:
        """Test creating a new task list."""
        storage = TaskStorage(tmp_path)
        task_list = storage.create("test-list")

        assert task_list["id"] == "test-list"
        assert task_list["tasks"] == []
        assert task_list["version"] == 1  # Incremented on save
        assert "created_at" in task_list
        assert "updated_at" in task_list

        # Verify file exists
        file_path = tmp_path / "test-list.json"
        assert file_path.exists()

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        """Test loading a non-existent task list returns None."""
        storage = TaskStorage(tmp_path)
        result = storage.load("nonexistent")
        assert result is None

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading a task list."""
        storage = TaskStorage(tmp_path)

        task_list: TaskList = {
            "id": "test-list",
            "tasks": [
                {
                    "id": "task-1",
                    "content": "Test task",
                    "status": "pending",
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-01T00:00:00Z",
                }
            ],
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "version": 0,
        }

        storage.save(task_list)
        loaded = storage.load("test-list")

        assert loaded is not None
        assert loaded["id"] == "test-list"
        assert len(loaded["tasks"]) == 1
        assert loaded["tasks"][0]["content"] == "Test task"
        assert loaded["version"] == 1  # Incremented on save

    def test_version_increment(self, tmp_path: Path) -> None:
        """Test that version increments on each save."""
        storage = TaskStorage(tmp_path)
        task_list = storage.create("test-list")

        initial_version = task_list["version"]

        # Save again
        storage.save(task_list)
        loaded = storage.load("test-list")

        assert loaded is not None
        assert loaded["version"] == initial_version + 1


class TestDependencyValidation:
    """Tests for dependency validation functions."""

    def test_valid_dependencies(self) -> None:
        """Test validation passes for valid dependencies."""
        tasks: list[Task] = [
            {"id": "task-1", "content": "First", "status": "completed", "created_at": "", "updated_at": ""},
            {"id": "task-2", "content": "Second", "status": "pending", "blocked_by": ["task-1"], "created_at": "", "updated_at": ""},
        ]
        result = _validate_dependencies(tasks)
        assert result is None

    def test_invalid_reference(self) -> None:
        """Test validation fails for invalid dependency reference."""
        tasks: list[Task] = [
            {"id": "task-1", "content": "First", "status": "pending", "blocked_by": ["nonexistent"], "created_at": "", "updated_at": ""},
        ]
        result = _validate_dependencies(tasks)
        assert result is not None
        assert "non-existent dependency" in result

    def test_self_reference(self) -> None:
        """Test validation fails for self-reference."""
        tasks: list[Task] = [
            {"id": "task-1", "content": "First", "status": "pending", "blocked_by": ["task-1"], "created_at": "", "updated_at": ""},
        ]
        result = _validate_dependencies(tasks)
        assert result is not None
        assert "cannot depend on itself" in result

    def test_circular_dependency(self) -> None:
        """Test validation fails for circular dependencies."""
        tasks: list[Task] = [
            {"id": "task-1", "content": "First", "status": "pending", "blocked_by": ["task-2"], "created_at": "", "updated_at": ""},
            {"id": "task-2", "content": "Second", "status": "pending", "blocked_by": ["task-1"], "created_at": "", "updated_at": ""},
        ]
        result = _validate_dependencies(tasks)
        assert result is not None
        assert "Circular dependency" in result

    def test_transitive_circular_dependency(self) -> None:
        """Test validation fails for transitive circular dependencies."""
        tasks: list[Task] = [
            {"id": "task-1", "content": "First", "status": "pending", "blocked_by": ["task-3"], "created_at": "", "updated_at": ""},
            {"id": "task-2", "content": "Second", "status": "pending", "blocked_by": ["task-1"], "created_at": "", "updated_at": ""},
            {"id": "task-3", "content": "Third", "status": "pending", "blocked_by": ["task-2"], "created_at": "", "updated_at": ""},
        ]
        result = _validate_dependencies(tasks)
        assert result is not None
        assert "Circular dependency" in result


class TestBlockedStatusComputation:
    """Tests for blocked status computation."""

    def test_no_dependencies(self) -> None:
        """Test status is unchanged when no dependencies."""
        task: Task = {"id": "task-1", "content": "Test", "status": "pending", "created_at": "", "updated_at": ""}
        all_tasks = [task]
        result = _compute_blocked_status(task, all_tasks)
        assert result == "pending"

    def test_completed_dependencies(self) -> None:
        """Test status is unchanged when all dependencies completed."""
        tasks: list[Task] = [
            {"id": "task-1", "content": "First", "status": "completed", "created_at": "", "updated_at": ""},
            {"id": "task-2", "content": "Second", "status": "pending", "blocked_by": ["task-1"], "created_at": "", "updated_at": ""},
        ]
        result = _compute_blocked_status(tasks[1], tasks)
        assert result == "pending"

    def test_incomplete_dependencies(self) -> None:
        """Test status becomes blocked when dependencies incomplete."""
        tasks: list[Task] = [
            {"id": "task-1", "content": "First", "status": "in_progress", "created_at": "", "updated_at": ""},
            {"id": "task-2", "content": "Second", "status": "pending", "blocked_by": ["task-1"], "created_at": "", "updated_at": ""},
        ]
        result = _compute_blocked_status(tasks[1], tasks)
        assert result == "blocked"

    def test_partial_dependencies_completed(self) -> None:
        """Test status becomes blocked when some dependencies incomplete."""
        tasks: list[Task] = [
            {"id": "task-1", "content": "First", "status": "completed", "created_at": "", "updated_at": ""},
            {"id": "task-2", "content": "Second", "status": "pending", "created_at": "", "updated_at": ""},
            {"id": "task-3", "content": "Third", "status": "pending", "blocked_by": ["task-1", "task-2"], "created_at": "", "updated_at": ""},
        ]
        result = _compute_blocked_status(tasks[2], tasks)
        assert result == "blocked"


class TestTaskMiddleware:
    """Tests for TaskMiddleware class."""

    def test_init_auto_generates_id(self, tmp_path: Path) -> None:
        """Test middleware auto-generates task list ID."""
        middleware = TaskMiddleware(tasks_dir=tmp_path)
        assert middleware.task_list_id is not None
        assert len(middleware.task_list_id) == 8  # Short UUID

    def test_init_uses_provided_id(self, tmp_path: Path) -> None:
        """Test middleware uses provided task list ID."""
        middleware = TaskMiddleware(task_list_id="my-tasks", tasks_dir=tmp_path)
        assert middleware.task_list_id == "my-tasks"

    def test_init_uses_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test middleware uses environment variable."""
        monkeypatch.setenv("DEEPAGENTS_TASK_LIST_ID", "env-tasks")
        middleware = TaskMiddleware(tasks_dir=tmp_path)
        assert middleware.task_list_id == "env-tasks"

    def test_tools_available(self, tmp_path: Path) -> None:
        """Test middleware provides write_tasks and get_tasks tools."""
        middleware = TaskMiddleware(tasks_dir=tmp_path)
        tool_names = [t.name for t in middleware.tools]
        assert "write_tasks" in tool_names
        assert "get_tasks" in tool_names

    def test_has_state_schema(self, tmp_path: Path) -> None:
        """Test middleware has proper state schema."""
        middleware = TaskMiddleware(tasks_dir=tmp_path)
        assert middleware.state_schema is not None
        assert "tasks" in middleware.state_schema.__annotations__
        assert "task_list_id" in middleware.state_schema.__annotations__


class TestTaskMiddlewareIntegration:
    """Integration tests for TaskMiddleware with file system."""

    def test_write_and_get_tasks_roundtrip(self, tmp_path: Path) -> None:
        """Test writing and reading tasks through storage."""
        storage = TaskStorage(tmp_path)

        # Create initial task list
        task_list = storage.create("integration-test")

        # Add tasks
        task_list["tasks"] = [
            {
                "id": "t1",
                "content": "First task",
                "status": "completed",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z",
            },
            {
                "id": "t2",
                "content": "Second task",
                "status": "pending",
                "blocked_by": ["t1"],
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z",
            },
        ]
        storage.save(task_list)

        # Reload and verify
        loaded = storage.load("integration-test")
        assert loaded is not None
        assert len(loaded["tasks"]) == 2
        assert loaded["tasks"][0]["status"] == "completed"
        assert loaded["tasks"][1]["blocked_by"] == ["t1"]

    def test_concurrent_access_with_locking(self, tmp_path: Path) -> None:
        """Test that file locking prevents corruption."""
        storage = TaskStorage(tmp_path)
        task_list = storage.create("concurrent-test")

        # Simulate concurrent updates by saving multiple times
        for i in range(10):
            current = storage.load("concurrent-test")
            assert current is not None
            current["tasks"].append({
                "id": f"task-{i}",
                "content": f"Task {i}",
                "status": "pending",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z",
            })
            storage.save(current)

        # Verify all tasks are present
        final = storage.load("concurrent-test")
        assert final is not None
        assert len(final["tasks"]) == 10
        assert final["version"] == 11  # 1 from create + 10 saves
