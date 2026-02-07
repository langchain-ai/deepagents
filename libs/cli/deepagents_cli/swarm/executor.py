"""Swarm executor with async worker pool for parallel task execution."""

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from deepagents_cli.swarm.graph import CycleError, DependencyGraph
from deepagents_cli.swarm.types import (
    SwarmProgress,
    SwarmResult,
    SwarmResultStatus,
    SwarmSummary,
    SwarmTask,
)


class SwarmExecutionError(Exception):
    """Error during swarm execution."""

    pass


# Sentinel value for signaling workers to stop
_DONE_SENTINEL = object()


class SwarmExecutor:
    """Executes a batch of tasks in parallel using subagents.

    Manages a pool of async workers that execute tasks respecting dependencies.
    Tasks without dependencies start immediately; tasks with dependencies wait
    until all dependencies complete successfully.
    """

    def __init__(
        self,
        subagent_graphs: dict[str, Runnable],
        *,
        default_subagent_type: str = "general-purpose",
        timeout_seconds: float = 300.0,
    ) -> None:
        """Initialize the swarm executor.

        Args:
            subagent_graphs: Map of subagent type names to runnable instances.
            default_subagent_type: Default subagent type when not specified in task.
            timeout_seconds: Timeout for individual task execution.
        """
        self.subagent_graphs = subagent_graphs
        self.default_subagent_type = default_subagent_type
        self.timeout_seconds = timeout_seconds

    async def execute(
        self,
        tasks: list[SwarmTask],
        *,
        concurrency: int = 10,
        output_dir: Path,
        run_id: str | None = None,
        started_at: str | None = None,
        progress_callback: Callable[[SwarmProgress], None] | None = None,
    ) -> SwarmSummary:
        """Execute a batch of tasks in parallel.

        Args:
            tasks: List of tasks to execute.
            concurrency: Maximum number of parallel workers.
            output_dir: Directory to write results.
            run_id: Optional identifier for this swarm run.
            started_at: Optional ISO timestamp for when the run was started.
            progress_callback: Optional callback for progress updates.

        Returns:
            SwarmSummary with execution statistics.

        Raises:
            SwarmExecutionError: If there's an error during execution.
            CycleError: If tasks contain circular dependencies.
        """
        start_time = time.time()

        # Build dependency graph (validates no cycles)
        try:
            graph = DependencyGraph.from_tasks(tasks)
        except CycleError:
            raise

        # Create task lookup by ID
        task_by_id: dict[str, SwarmTask] = {t["id"]: t for t in tasks}

        # Track state
        results: dict[str, SwarmResult] = {}
        completed: set[str] = set()  # Successfully completed tasks
        failed: set[str] = set()  # Failed tasks
        skipped: set[str] = set()  # Skipped due to dependency failure
        running: set[str] = set()  # Currently running tasks

        # Ready queue for tasks that can be executed
        ready_queue: asyncio.Queue[SwarmTask | object] = asyncio.Queue()

        # Results file handle (streaming writes)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "results.jsonl"
        failures_path = output_dir / "failures.jsonl"

        # Initialize output files
        results_path.write_text("")
        failures_path.write_text("")

        # Seed with independent tasks
        for task_id in graph.get_independent_tasks():
            await ready_queue.put(task_by_id[task_id])

        def _report_progress() -> None:
            """Report current progress if callback provided."""
            if progress_callback:
                progress = SwarmProgress(
                    total=len(tasks),
                    completed=len(completed) + len(failed),
                    succeeded=len(completed),
                    failed=len(failed),
                    skipped=len(skipped),
                    running=list(running),
                    blocked=len(tasks) - len(completed) - len(failed) - len(skipped) - len(running),
                )
                progress_callback(progress)

        def _write_result(result: SwarmResult) -> None:
            """Write a result to the appropriate output file."""
            with results_path.open("a") as f:
                f.write(json.dumps(_result_to_dict(result)) + "\n")

            if result["status"] != SwarmResultStatus.SUCCESS:
                with failures_path.open("a") as f:
                    f.write(json.dumps(_result_to_dict(result)) + "\n")

        async def _execute_single_task(task: SwarmTask) -> SwarmResult:
            """Execute a single task using the appropriate subagent."""
            task_start = time.time()

            subagent_type = task.get("type", self.default_subagent_type)

            if subagent_type not in self.subagent_graphs:
                available = list(self.subagent_graphs.keys())
                return SwarmResult(
                    task_id=task["id"],
                    status=SwarmResultStatus.FAILED,
                    error="ValueError",
                    message=f"Unknown subagent type: {subagent_type}. Available: {available}",
                    duration_ms=int((time.time() - task_start) * 1000),
                    metadata=task.get("metadata"),
                )

            subagent = self.subagent_graphs[subagent_type]

            try:
                # Create isolated state with task description
                subagent_state = {"messages": [HumanMessage(content=task["description"])]}

                # Execute with timeout
                result = await asyncio.wait_for(
                    subagent.ainvoke(subagent_state),
                    timeout=self.timeout_seconds,
                )

                # Extract output from final message
                output = ""
                if "messages" in result and result["messages"]:
                    final_msg = result["messages"][-1]
                    if hasattr(final_msg, "content"):
                        content = final_msg.content
                        # Handle multimodal content (list of content blocks)
                        if isinstance(content, list):
                            # Extract text from content blocks
                            text_parts = []
                            for block in content:
                                if isinstance(block, str):
                                    text_parts.append(block)
                                elif isinstance(block, dict) and block.get("type") == "text":
                                    text_parts.append(block.get("text", ""))
                            output = "\n".join(text_parts)
                        else:
                            output = content
                    else:
                        output = str(final_msg)

                return SwarmResult(
                    task_id=task["id"],
                    status=SwarmResultStatus.SUCCESS,
                    output=output,
                    duration_ms=int((time.time() - task_start) * 1000),
                    metadata=task.get("metadata"),
                )

            except asyncio.TimeoutError:
                return SwarmResult(
                    task_id=task["id"],
                    status=SwarmResultStatus.FAILED,
                    error="TimeoutError",
                    message=f"Task timed out after {self.timeout_seconds}s",
                    duration_ms=int((time.time() - task_start) * 1000),
                    metadata=task.get("metadata"),
                )

            except Exception as e:
                return SwarmResult(
                    task_id=task["id"],
                    status=SwarmResultStatus.FAILED,
                    error=type(e).__name__,
                    message=str(e),
                    duration_ms=int((time.time() - task_start) * 1000),
                    metadata=task.get("metadata"),
                )

        async def worker() -> None:
            """Worker coroutine that processes tasks from the ready queue."""
            while True:
                item = await ready_queue.get()

                if item is _DONE_SENTINEL:
                    ready_queue.task_done()
                    break

                task = item
                task_id = task["id"]
                running.add(task_id)
                _report_progress()

                try:
                    result = await _execute_single_task(task)
                    results[task_id] = result
                    _write_result(result)

                    if result["status"] == SwarmResultStatus.SUCCESS:
                        completed.add(task_id)

                        # Check if any dependents are now unblocked
                        for dependent_id in graph.get_dependents(task_id):
                            deps = graph.get_dependencies(dependent_id)
                            # All dependencies must be in completed (not just finished)
                            if deps.issubset(completed) and dependent_id not in skipped:
                                await ready_queue.put(task_by_id[dependent_id])
                    else:
                        failed.add(task_id)

                        # Mark all downstream tasks as skipped
                        for downstream_id in graph.get_all_downstream(task_id):
                            if downstream_id not in results:
                                skip_result = SwarmResult(
                                    task_id=downstream_id,
                                    status=SwarmResultStatus.SKIPPED,
                                    message=f"Skipped: dependency task '{task_id}' failed",
                                    duration_ms=0,
                                    metadata=task_by_id[downstream_id].get("metadata"),
                                )
                                results[downstream_id] = skip_result
                                skipped.add(downstream_id)
                                _write_result(skip_result)

                finally:
                    running.discard(task_id)
                    ready_queue.task_done()
                    _report_progress()

        # Start workers
        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]

        # Wait for all tasks to complete
        # We need to check periodically if we're done since skipped tasks don't go through the queue
        while len(results) < len(tasks):
            # Wait a bit for queue to drain
            await asyncio.sleep(0.1)

            # Check if we're stuck (no running tasks and nothing in queue but not all done)
            if not running and ready_queue.empty() and len(results) < len(tasks):
                # All remaining tasks must be blocked by failed tasks
                for task in tasks:
                    if task["id"] not in results:
                        skip_result = SwarmResult(
                            task_id=task["id"],
                            status=SwarmResultStatus.SKIPPED,
                            message="Skipped: blocked by failed dependency",
                            duration_ms=0,
                            metadata=task.get("metadata"),
                        )
                        results[task["id"]] = skip_result
                        skipped.add(task["id"])
                        _write_result(skip_result)
                break

        # Signal workers to stop
        for _ in workers:
            await ready_queue.put(_DONE_SENTINEL)
        await asyncio.gather(*workers)

        # Final progress report
        _report_progress()

        # Calculate duration
        duration_seconds = time.time() - start_time

        # Build summary
        summary = SwarmSummary(
            run_id=run_id or generate_swarm_run_id(),
            started_at=started_at or datetime.now().isoformat(timespec="seconds"),
            total=len(tasks),
            succeeded=len(completed),
            failed=len(failed),
            skipped=len(skipped),
            duration_seconds=round(duration_seconds, 2),
            concurrency=concurrency,
            results_path=str(results_path),
            failures_path=str(failures_path),
        )

        # Write summary
        summary_path = output_dir / "summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

        return summary


def _result_to_dict(result: SwarmResult) -> dict[str, Any]:
    """Convert SwarmResult to a JSON-serializable dict."""
    d: dict[str, Any] = {
        "task_id": result["task_id"],
        "status": result["status"].value,
        "duration_ms": result["duration_ms"],
    }

    if "output" in result:
        d["output"] = result["output"]
    if "error" in result:
        d["error"] = result["error"]
    if "message" in result:
        d["message"] = result["message"]
    if "metadata" in result and result["metadata"]:
        d["metadata"] = result["metadata"]

    return d


def get_default_output_dir(run_id: str | None = None) -> Path:
    """Get the default output directory for batch results.

    Returns a timestamped directory under ./batch_results/.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    resolved_run_id = run_id or generate_swarm_run_id()
    return Path.cwd() / "batch_results" / f"{timestamp}_{resolved_run_id}"


def generate_swarm_run_id() -> str:
    """Generate a compact run ID for swarm executions."""
    return uuid.uuid4().hex[:8]
