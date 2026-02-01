"""Unit tests for TaskBoardMiddleware."""

import pytest

from deepagents_cli.swarm.middleware import TaskBoardMiddleware
from deepagents_cli.swarm.task_store import TaskStore


@pytest.fixture
def task_store(tmp_path):
    """TaskStore with temp database."""
    return TaskStore(db_path=tmp_path / "test.db")


@pytest.fixture
def middleware(task_store):
    """TaskBoardMiddleware instance."""
    return TaskBoardMiddleware(task_store=task_store)


class TestTaskBoardMiddleware:
    def test_has_four_tools(self, middleware):
        """Middleware should provide 4 task board tools."""
        assert len(middleware.tools) == 4
        tool_names = {t.name for t in middleware.tools}
        assert tool_names == {"TaskCreate", "TaskGet", "TaskUpdate", "TaskList"}

    def test_tools_are_structured_tools(self, middleware):
        """All tools should be StructuredTool instances."""
        from langchain_core.tools import StructuredTool

        for tool in middleware.tools:
            assert isinstance(tool, StructuredTool)

    def test_tool_descriptions(self, middleware):
        """Tools should have meaningful descriptions."""
        tools_dict = {t.name: t for t in middleware.tools}

        assert "task" in tools_dict["TaskCreate"].description.lower()
        assert "details" in tools_dict["TaskGet"].description.lower()
        assert "update" in tools_dict["TaskUpdate"].description.lower()
        assert "list" in tools_dict["TaskList"].description.lower()

    def test_default_task_store(self):
        """Middleware should create default TaskStore if none provided."""
        middleware = TaskBoardMiddleware()
        assert middleware.task_store is not None
        assert isinstance(middleware.task_store, TaskStore)

    def test_custom_task_store(self, task_store):
        """Middleware should use provided TaskStore."""
        middleware = TaskBoardMiddleware(task_store=task_store)
        assert middleware.task_store is task_store
