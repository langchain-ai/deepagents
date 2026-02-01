"""Unit tests for TaskStore."""

import pytest

from deepagents_cli.swarm.task_store import TaskStore
from deepagents_cli.swarm.types import TaskStatus


@pytest.fixture
def task_store(tmp_path):
    """TaskStore with temp database."""
    return TaskStore(db_path=tmp_path / "test.db")


class TestTaskCreate:
    @pytest.mark.asyncio
    async def test_create_basic(self, task_store):
        task = await task_store.create_task("sess", "Do X", "Details")
        assert task["id"] == "1"
        assert task["subject"] == "Do X"
        assert task["status"] == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_sequential_ids(self, task_store):
        t1 = await task_store.create_task("sess", "Task 1", "D1")
        t2 = await task_store.create_task("sess", "Task 2", "D2")
        t3 = await task_store.create_task("sess", "Task 3", "D3")
        assert (t1["id"], t2["id"], t3["id"]) == ("1", "2", "3")

    @pytest.mark.asyncio
    async def test_session_isolation(self, task_store):
        await task_store.create_task("sess-a", "Task A", "Da")
        await task_store.create_task("sess-b", "Task B", "Db")

        tasks_a = await task_store.list_tasks("sess-a")
        tasks_b = await task_store.list_tasks("sess-b")

        assert len(tasks_a) == 1
        assert len(tasks_b) == 1
        assert tasks_a[0]["subject"] == "Task A"

    @pytest.mark.asyncio
    async def test_create_with_metadata(self, task_store):
        task = await task_store.create_task(
            "sess",
            "Task",
            "Desc",
            metadata={"priority": "high", "tags": ["urgent"]},
        )
        assert task["metadata"]["priority"] == "high"
        assert task["metadata"]["tags"] == ["urgent"]

    @pytest.mark.asyncio
    async def test_create_with_active_form(self, task_store):
        task = await task_store.create_task(
            "sess",
            "Fix bug",
            "Fix the auth bug",
            active_form="Fixing bug",
        )
        assert task["active_form"] == "Fixing bug"


class TestTaskGet:
    @pytest.mark.asyncio
    async def test_get_existing(self, task_store):
        await task_store.create_task("sess", "Task", "Desc")
        task = await task_store.get_task("sess", "1")
        assert task is not None
        assert task["subject"] == "Task"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, task_store):
        task = await task_store.get_task("sess", "999")
        assert task is None

    @pytest.mark.asyncio
    async def test_get_wrong_session(self, task_store):
        await task_store.create_task("sess-a", "Task A", "Da")
        task = await task_store.get_task("sess-b", "1")
        assert task is None


class TestTaskUpdate:
    @pytest.mark.asyncio
    async def test_update_status(self, task_store):
        task = await task_store.create_task("sess", "Task", "Desc")

        updated = await task_store.update_task(
            "sess", "1", status=TaskStatus.IN_PROGRESS
        )
        assert updated["status"] == TaskStatus.IN_PROGRESS

        updated = await task_store.update_task(
            "sess", "1", status=TaskStatus.COMPLETED
        )
        assert updated["status"] == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_add_blocked_by(self, task_store):
        await task_store.create_task("sess", "Task 1", "D1")
        await task_store.create_task("sess", "Task 2", "D2")
        await task_store.create_task("sess", "Task 3", "D3")

        await task_store.update_task("sess", "3", add_blocked_by=["1", "2"])

        task3 = await task_store.get_task("sess", "3")
        assert "1" in task3["blocked_by"]
        assert "2" in task3["blocked_by"]

    @pytest.mark.asyncio
    async def test_add_blocks(self, task_store):
        await task_store.create_task("sess", "Task 1", "D1")
        await task_store.create_task("sess", "Task 2", "D2")

        await task_store.update_task("sess", "1", add_blocks=["2"])

        task1 = await task_store.get_task("sess", "1")
        assert "2" in task1["blocks"]

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, task_store):
        result = await task_store.update_task(
            "sess", "999", status=TaskStatus.COMPLETED
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_update_subject(self, task_store):
        await task_store.create_task("sess", "Old title", "Desc")
        updated = await task_store.update_task("sess", "1", subject="New title")
        assert updated["subject"] == "New title"

    @pytest.mark.asyncio
    async def test_update_metadata_merge(self, task_store):
        await task_store.create_task(
            "sess", "Task", "Desc", metadata={"key1": "val1"}
        )
        updated = await task_store.update_task(
            "sess", "1", metadata={"key2": "val2"}
        )
        assert updated["metadata"]["key1"] == "val1"
        assert updated["metadata"]["key2"] == "val2"

    @pytest.mark.asyncio
    async def test_update_owner(self, task_store):
        await task_store.create_task("sess", "Task", "Desc")
        updated = await task_store.update_task("sess", "1", owner="agent-1")
        assert updated["owner"] == "agent-1"


class TestTaskList:
    @pytest.mark.asyncio
    async def test_list_empty(self, task_store):
        tasks = await task_store.list_tasks("sess")
        assert tasks == []

    @pytest.mark.asyncio
    async def test_list_ordered(self, task_store):
        await task_store.create_task("sess", "First", "D")
        await task_store.create_task("sess", "Second", "D")
        await task_store.create_task("sess", "Third", "D")

        tasks = await task_store.list_tasks("sess")
        assert len(tasks) == 3
        assert tasks[0]["id"] == "1"
        assert tasks[1]["id"] == "2"
        assert tasks[2]["id"] == "3"

    @pytest.mark.asyncio
    async def test_list_session_isolation(self, task_store):
        await task_store.create_task("sess-a", "Task A1", "D")
        await task_store.create_task("sess-a", "Task A2", "D")
        await task_store.create_task("sess-b", "Task B1", "D")

        tasks_a = await task_store.list_tasks("sess-a")
        tasks_b = await task_store.list_tasks("sess-b")

        assert len(tasks_a) == 2
        assert len(tasks_b) == 1
