"""Unit tests for task board tools."""

import pytest

from deepagents_cli.swarm.task_board import create_task_board_tools
from deepagents_cli.swarm.task_store import TaskStore
from deepagents_cli.swarm.types import TaskStatus


@pytest.fixture
def task_store(tmp_path):
    """TaskStore with temp database."""
    return TaskStore(db_path=tmp_path / "test.db")


@pytest.fixture
def tools(task_store):
    """Task board tools for test session."""
    tool_list = create_task_board_tools(task_store, "test-session")
    return {t.name: t for t in tool_list}


class TestTaskCreateTool:
    @pytest.mark.asyncio
    async def test_create_returns_confirmation(self, tools):
        result = await tools["TaskCreate"].ainvoke(
            {
                "subject": "Build feature",
                "description": "Build the feature with tests",
            }
        )
        assert "Task #1 created" in result

    @pytest.mark.asyncio
    async def test_create_sequential(self, tools):
        r1 = await tools["TaskCreate"].ainvoke(
            {"subject": "Task 1", "description": "D1"}
        )
        r2 = await tools["TaskCreate"].ainvoke(
            {"subject": "Task 2", "description": "D2"}
        )
        assert "Task #1" in r1
        assert "Task #2" in r2

    @pytest.mark.asyncio
    async def test_create_with_optional_params(self, tools):
        result = await tools["TaskCreate"].ainvoke(
            {
                "subject": "Task",
                "description": "Desc",
                "active_form": "Working on task",
                "metadata": {"priority": "high"},
            }
        )
        assert "Task #1 created" in result


class TestTaskGetTool:
    @pytest.mark.asyncio
    async def test_get_existing(self, tools, task_store):
        await task_store.create_task("test-session", "My Task", "Task description")
        result = await tools["TaskGet"].ainvoke({"task_id": "1"})

        assert "Task #1: My Task" in result
        assert "Status: pending" in result
        assert "Task description" in result

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, tools):
        result = await tools["TaskGet"].ainvoke({"task_id": "999"})
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_get_shows_blocked_by(self, tools, task_store):
        await task_store.create_task("test-session", "Task 1", "D1")
        await task_store.create_task("test-session", "Task 2", "D2")
        await task_store.update_task("test-session", "2", add_blocked_by=["1"])

        result = await tools["TaskGet"].ainvoke({"task_id": "2"})
        assert "Blocked by:" in result
        assert "#1" in result


class TestTaskUpdateTool:
    @pytest.mark.asyncio
    async def test_update_status(self, tools, task_store):
        await task_store.create_task("test-session", "Task", "Desc")

        result = await tools["TaskUpdate"].ainvoke(
            {"task_id": "1", "status": "in_progress"}
        )
        assert "updated" in result

        task = await task_store.get_task("test-session", "1")
        assert task["status"] == TaskStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, tools):
        result = await tools["TaskUpdate"].ainvoke(
            {"task_id": "999", "status": "completed"}
        )
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_add_blocked_by(self, tools, task_store):
        await task_store.create_task("test-session", "Task 1", "D1")
        await task_store.create_task("test-session", "Task 2", "D2")
        await task_store.create_task("test-session", "Task 3", "D3")

        await tools["TaskUpdate"].ainvoke(
            {"task_id": "3", "add_blocked_by": ["1", "2"]}
        )

        task = await task_store.get_task("test-session", "3")
        assert "1" in task["blocked_by"]
        assert "2" in task["blocked_by"]


class TestTaskListTool:
    @pytest.mark.asyncio
    async def test_list_empty(self, tools):
        result = await tools["TaskList"].ainvoke({})
        assert result == "No tasks"

    @pytest.mark.asyncio
    async def test_list_multiple(self, tools, task_store):
        await task_store.create_task("test-session", "Task A", "Da")
        await task_store.create_task("test-session", "Task B", "Db")
        await task_store.create_task("test-session", "Task C", "Dc")

        result = await tools["TaskList"].ainvoke({})

        assert "#1 [pending] Task A" in result
        assert "#2 [pending] Task B" in result
        assert "#3 [pending] Task C" in result

    @pytest.mark.asyncio
    async def test_list_shows_blockers(self, tools, task_store):
        await task_store.create_task("test-session", "Task A", "Da")
        await task_store.create_task("test-session", "Task B", "Db")
        await task_store.create_task("test-session", "Task C", "Dc")
        await task_store.update_task("test-session", "3", add_blocked_by=["1", "2"])

        result = await tools["TaskList"].ainvoke({})

        assert "#3" in result
        assert "blocked by" in result

    @pytest.mark.asyncio
    async def test_list_hides_completed_blockers(self, tools, task_store):
        await task_store.create_task("test-session", "Task A", "Da")
        await task_store.create_task("test-session", "Task B", "Db")
        await task_store.update_task("test-session", "2", add_blocked_by=["1"])

        # Complete the blocker
        await task_store.update_task("test-session", "1", status=TaskStatus.COMPLETED)

        result = await tools["TaskList"].ainvoke({})

        # Task 2 should not show as blocked since task 1 is completed
        assert "#2 [pending] Task B" in result
        assert "blocked by" not in result.split("\n")[1]  # Check the Task B line

    @pytest.mark.asyncio
    async def test_list_shows_status(self, tools, task_store):
        await task_store.create_task("test-session", "Pending Task", "D")
        await task_store.create_task("test-session", "In Progress Task", "D")
        await task_store.create_task("test-session", "Completed Task", "D")

        await task_store.update_task("test-session", "2", status=TaskStatus.IN_PROGRESS)
        await task_store.update_task("test-session", "3", status=TaskStatus.COMPLETED)

        result = await tools["TaskList"].ainvoke({})

        assert "[pending]" in result
        assert "[in_progress]" in result
        assert "[completed]" in result
