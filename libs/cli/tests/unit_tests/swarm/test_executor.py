"""Unit tests for swarm executor."""

import asyncio
import json
import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_cli.swarm.executor import SwarmExecutor, get_default_output_dir
from deepagents_cli.swarm.graph import CycleError
from deepagents_cli.swarm.types import SwarmResultStatus, SwarmTask


class MockSubagent:
    """Mock subagent for testing."""

    def __init__(self, response: str = "Task completed", delay: float = 0.0, error: Exception | None = None):
        self.response = response
        self.delay = delay
        self.error = error
        self.invocations: list[dict] = []

    async def ainvoke(self, state: dict) -> dict:
        self.invocations.append(state)

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.error:
            raise self.error

        return {"messages": [AIMessage(content=self.response)]}


def make_task(id: str, blocked_by: list[str] | None = None, type: str | None = None) -> SwarmTask:
    """Helper to create SwarmTask for testing."""
    task: SwarmTask = {"id": id, "description": f"Task {id}"}
    if blocked_by:
        task["blocked_by"] = blocked_by
    if type:
        task["type"] = type
    return task


@pytest.fixture
def mock_subagent():
    return MockSubagent()


@pytest.fixture
def subagent_graphs(mock_subagent):
    return {"general-purpose": mock_subagent}


@pytest.fixture
def executor(subagent_graphs):
    return SwarmExecutor(subagent_graphs, timeout_seconds=5.0)


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "results"


class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_executes_single_task(self, executor, output_dir):
        tasks = [make_task("1")]

        summary = await executor.execute(tasks, concurrency=1, output_dir=output_dir)

        assert summary["run_id"]
        assert summary["started_at"]
        assert summary["total"] == 1
        assert summary["succeeded"] == 1
        assert summary["failed"] == 0

    @pytest.mark.asyncio
    async def test_executes_multiple_independent_tasks(self, executor, output_dir):
        tasks = [make_task("1"), make_task("2"), make_task("3")]

        summary = await executor.execute(tasks, concurrency=3, output_dir=output_dir)

        assert summary["total"] == 3
        assert summary["succeeded"] == 3

    @pytest.mark.asyncio
    async def test_writes_results_file(self, executor, output_dir):
        tasks = [make_task("1"), make_task("2")]

        await executor.execute(tasks, concurrency=2, output_dir=output_dir)

        results_path = output_dir / "results.jsonl"
        assert results_path.exists()

        results = [json.loads(line) for line in results_path.read_text().strip().split("\n")]
        assert len(results) == 2
        task_ids = {r["task_id"] for r in results}
        assert task_ids == {"1", "2"}

    @pytest.mark.asyncio
    async def test_writes_summary_file(self, executor, output_dir):
        tasks = [make_task("1")]

        await executor.execute(tasks, concurrency=1, output_dir=output_dir)

        summary_path = output_dir / "summary.json"
        assert summary_path.exists()

        summary = json.loads(summary_path.read_text())
        assert summary["run_id"]
        assert summary["started_at"]
        assert summary["total"] == 1
        assert summary["succeeded"] == 1


class TestDependencyHandling:
    @pytest.mark.asyncio
    async def test_respects_dependencies(self, output_dir):
        """Tasks with dependencies should wait for their blockers."""
        execution_order = []

        class OrderTrackingSubagent:
            async def ainvoke(self, state: dict):
                desc = state["messages"][0].content
                task_id = desc.split()[-1]  # "Task X" -> "X"
                execution_order.append(task_id)
                return {"messages": [AIMessage(content=f"Completed {task_id}")]}

        executor = SwarmExecutor(
            {"general-purpose": OrderTrackingSubagent()},
            timeout_seconds=5.0,
        )

        tasks = [
            make_task("1"),
            make_task("2"),
            make_task("3", blocked_by=["1", "2"]),
        ]

        await executor.execute(tasks, concurrency=2, output_dir=output_dir)

        # Task 3 must come after 1 and 2
        assert execution_order.index("3") > execution_order.index("1")
        assert execution_order.index("3") > execution_order.index("2")

    @pytest.mark.asyncio
    async def test_skips_dependents_on_failure(self, output_dir):
        """When a task fails, its dependents should be skipped."""
        failing_subagent = MockSubagent(error=ValueError("Task failed"))

        executor = SwarmExecutor(
            {"general-purpose": failing_subagent},
            timeout_seconds=5.0,
        )

        tasks = [
            make_task("1"),  # Will fail
            make_task("2", blocked_by=["1"]),  # Should be skipped
            make_task("3", blocked_by=["2"]),  # Should also be skipped (transitive)
        ]

        summary = await executor.execute(tasks, concurrency=1, output_dir=output_dir)

        assert summary["failed"] == 1
        assert summary["skipped"] == 2

        # Check failures file
        failures_path = output_dir / "failures.jsonl"
        failures = [json.loads(line) for line in failures_path.read_text().strip().split("\n")]

        statuses = {f["task_id"]: f["status"] for f in failures}
        assert statuses["1"] == "failed"
        assert statuses["2"] == "skipped"
        assert statuses["3"] == "skipped"


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_handles_subagent_error(self, output_dir):
        failing_subagent = MockSubagent(error=ValueError("Something went wrong"))

        executor = SwarmExecutor(
            {"general-purpose": failing_subagent},
            timeout_seconds=5.0,
        )

        tasks = [make_task("1")]

        summary = await executor.execute(tasks, concurrency=1, output_dir=output_dir)

        assert summary["failed"] == 1
        assert summary["succeeded"] == 0

        # Check error is recorded
        results = json.loads((output_dir / "results.jsonl").read_text().strip())
        assert results["status"] == "failed"
        assert results["error"] == "ValueError"

    @pytest.mark.asyncio
    async def test_handles_timeout(self, output_dir):
        slow_subagent = MockSubagent(delay=10.0)  # Will timeout

        executor = SwarmExecutor(
            {"general-purpose": slow_subagent},
            timeout_seconds=0.1,  # Very short timeout
        )

        tasks = [make_task("1")]

        summary = await executor.execute(tasks, concurrency=1, output_dir=output_dir)

        assert summary["failed"] == 1

        results = json.loads((output_dir / "results.jsonl").read_text().strip())
        assert results["status"] == "failed"
        assert results["error"] == "TimeoutError"

    @pytest.mark.asyncio
    async def test_handles_unknown_subagent_type(self, executor, output_dir):
        tasks = [make_task("1", type="nonexistent-type")]

        summary = await executor.execute(tasks, concurrency=1, output_dir=output_dir)

        assert summary["failed"] == 1

        results = json.loads((output_dir / "results.jsonl").read_text().strip())
        assert "Unknown subagent type" in results["message"]

    @pytest.mark.asyncio
    async def test_rejects_cyclic_dependencies(self, executor, output_dir):
        tasks = [
            make_task("1", blocked_by=["2"]),
            make_task("2", blocked_by=["1"]),
        ]

        with pytest.raises(CycleError):
            await executor.execute(tasks, concurrency=1, output_dir=output_dir)


class TestProgressReporting:
    @pytest.mark.asyncio
    async def test_calls_progress_callback(self, executor, output_dir):
        progress_updates = []

        def track_progress(progress):
            progress_updates.append({
                "completed": progress.completed,
                "running": len(progress.running),
            })

        tasks = [make_task("1"), make_task("2")]

        await executor.execute(
            tasks,
            concurrency=2,
            output_dir=output_dir,
            progress_callback=track_progress,
        )

        # Should have received multiple progress updates
        assert len(progress_updates) > 0
        # Final update should show all completed
        assert progress_updates[-1]["completed"] == 2


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_respects_concurrency_limit(self, output_dir):
        max_concurrent = 0
        current_concurrent = 0

        class ConcurrencyTracker:
            async def ainvoke(self, state: dict):
                nonlocal max_concurrent, current_concurrent
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
                await asyncio.sleep(0.1)
                current_concurrent -= 1
                return {"messages": [AIMessage(content="Done")]}

        executor = SwarmExecutor(
            {"general-purpose": ConcurrencyTracker()},
            timeout_seconds=5.0,
        )

        tasks = [make_task(str(i)) for i in range(10)]

        await executor.execute(tasks, concurrency=3, output_dir=output_dir)

        assert max_concurrent <= 3


class TestOutputDirectory:
    def test_default_output_dir_is_timestamped_with_run_id(self):
        output_dir = get_default_output_dir()

        assert "batch_results" in str(output_dir)
        # Name is expected to be: YYYY-MM-DD_HHMMSS_<8hex_run_id>
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{6}_[0-9a-f]{8}", output_dir.name)
