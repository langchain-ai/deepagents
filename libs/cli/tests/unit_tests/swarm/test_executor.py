"""Unit tests for swarm executor."""

import asyncio
import json
import re
from typing import TYPE_CHECKING, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_cli.swarm.executor import SwarmExecutor, get_default_output_dir
from deepagents_cli.swarm.types import SwarmProgress, SwarmTask

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable


class MockSubagent:
    """Mock subagent for testing."""

    def __init__(
        self,
        response: str = "Task completed",
        delay: float = 0.0,
        error: Exception | None = None,
    ) -> None:
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


def make_task(task_id: str, subagent_type: str | None = None) -> SwarmTask:
    """Helper to create SwarmTask for testing."""
    task: SwarmTask = {"id": task_id, "description": f"Task {task_id}"}
    if subagent_type:
        task["type"] = subagent_type
    return task


@pytest.fixture
def mock_subagent():
    return MockSubagent()


@pytest.fixture
def subagent_graphs(mock_subagent):
    return {"general-purpose": cast("Runnable", mock_subagent)}


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
    async def test_executes_multiple_tasks(self, executor, output_dir):
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

        results = [
            json.loads(line) for line in results_path.read_text().strip().split("\n")
        ]
        assert len(results) == 2
        assert {result["task_id"] for result in results} == {"1", "2"}

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


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_handles_subagent_error(self, output_dir):
        failing_subagent = MockSubagent(error=ValueError("Something went wrong"))

        executor = SwarmExecutor(
            {"general-purpose": cast("Runnable", failing_subagent)},
            timeout_seconds=5.0,
        )

        summary = await executor.execute(
            [make_task("1")],
            concurrency=1,
            output_dir=output_dir,
        )

        assert summary["failed"] == 1
        assert summary["succeeded"] == 0

        results = json.loads((output_dir / "results.jsonl").read_text().strip())
        assert results["status"] == "failed"
        assert results["error"] == "ValueError"
        assert "traceback" in results
        assert "ValueError: Something went wrong" in results["traceback"]

    @pytest.mark.asyncio
    async def test_handles_timeout(self, output_dir):
        slow_subagent = MockSubagent(delay=10.0)

        executor = SwarmExecutor(
            {"general-purpose": cast("Runnable", slow_subagent)},
            timeout_seconds=0.1,
        )

        summary = await executor.execute(
            [make_task("1")],
            concurrency=1,
            output_dir=output_dir,
        )

        assert summary["failed"] == 1
        result = json.loads((output_dir / "results.jsonl").read_text().strip())
        assert result["status"] == "failed"
        assert result["error"] == "TimeoutError"

    @pytest.mark.asyncio
    async def test_captures_traceback_for_recursion_error(self, output_dir):
        recursive_subagent = MockSubagent(
            error=RecursionError("maximum recursion depth exceeded")
        )
        executor = SwarmExecutor(
            {"general-purpose": cast("Runnable", recursive_subagent)},
            timeout_seconds=5.0,
        )

        summary = await executor.execute(
            [make_task("1")],
            concurrency=1,
            output_dir=output_dir,
        )

        assert summary["failed"] == 1
        result = json.loads((output_dir / "results.jsonl").read_text().strip())
        assert result["status"] == "failed"
        assert result["error"] == "RecursionError"
        assert "traceback" in result
        assert "RecursionError: maximum recursion depth exceeded" in result["traceback"]

    @pytest.mark.asyncio
    async def test_handles_unknown_subagent_type(self, executor, output_dir):
        summary = await executor.execute(
            [make_task("1", subagent_type="nonexistent-type")],
            concurrency=1,
            output_dir=output_dir,
        )

        assert summary["failed"] == 1
        result = json.loads((output_dir / "results.jsonl").read_text().strip())
        assert "Unknown subagent type" in result["message"]


class TestProgressReporting:
    @pytest.mark.asyncio
    async def test_calls_progress_callback(self, executor, output_dir):
        progress_updates = []

        def track_progress(progress: SwarmProgress) -> None:
            progress_updates.append(
                {
                    "completed": progress.completed,
                    "running": len(progress.running),
                }
            )

        tasks = [make_task("1"), make_task("2")]

        await executor.execute(
            tasks,
            concurrency=2,
            output_dir=output_dir,
            progress_callback=track_progress,
        )

        assert progress_updates
        assert progress_updates[-1]["completed"] == 2


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_respects_concurrency_limit(self, output_dir):
        max_concurrent = 0
        current_concurrent = 0

        class ConcurrencyTracker:
            async def ainvoke(self, _state: dict) -> dict:
                nonlocal max_concurrent, current_concurrent
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
                await asyncio.sleep(0.1)
                current_concurrent -= 1
                return {"messages": [AIMessage(content="Done")]}

        executor = SwarmExecutor(
            {"general-purpose": cast("Runnable", ConcurrencyTracker())},
            timeout_seconds=5.0,
        )

        await executor.execute(
            [make_task(str(index)) for index in range(10)],
            concurrency=3,
            output_dir=output_dir,
        )

        assert max_concurrent <= 3


class TestSubagentInput:
    @pytest.mark.asyncio
    async def test_passes_description_as_human_message(self, output_dir):
        subagent = MockSubagent()
        executor = SwarmExecutor(
            {"general-purpose": cast("Runnable", subagent)},
            timeout_seconds=5.0,
        )

        await executor.execute(
            [{"id": "1", "description": "Research ACME"}],
            concurrency=1,
            output_dir=output_dir,
        )

        invocation = subagent.invocations[0]
        message = invocation["messages"][0]
        assert isinstance(message, HumanMessage)
        assert message.content == "Research ACME"


class TestOutputDirectory:
    def test_default_output_dir_is_timestamped_with_run_id(self):
        output_dir = get_default_output_dir()

        assert "batch_results" in str(output_dir)
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{6}_[0-9a-f]{8}", output_dir.name)
