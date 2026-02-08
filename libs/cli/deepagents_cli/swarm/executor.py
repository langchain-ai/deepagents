"""Swarm executor with bounded parallelism for independent task execution."""

import asyncio
import json
import time
import traceback
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from deepagents_cli.swarm.types import (
    SwarmProgress,
    SwarmResult,
    SwarmResultStatus,
    SwarmSummary,
    SwarmTask,
)


class SwarmExecutionError(Exception):
    """Error during swarm execution."""


class SwarmExecutor:
    """Executes independent tasks in parallel using subagents."""

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
        """Execute independent tasks in parallel.

        Args:
            tasks: List of tasks to execute.
            concurrency: Maximum number of parallel workers.
            output_dir: Directory to write results.
            run_id: Optional identifier for this swarm run.
            started_at: Optional ISO timestamp for when the run started.
            progress_callback: Optional callback for progress updates.

        Returns:
            SwarmSummary with execution statistics.
        """
        start_time = time.time()
        run_concurrency = max(1, concurrency)

        await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
        results_path = output_dir / "results.jsonl"
        failures_path = output_dir / "failures.jsonl"
        await asyncio.to_thread(results_path.write_text, "")
        await asyncio.to_thread(failures_path.write_text, "")

        completed: set[str] = set()
        failed: set[str] = set()
        running: set[str] = set()

        def _report_progress() -> None:
            if progress_callback is None:
                return
            progress_callback(
                SwarmProgress(
                    total=len(tasks),
                    completed=len(completed) + len(failed),
                    succeeded=len(completed),
                    failed=len(failed),
                    running=sorted(running),
                )
            )

        def _write_result(result: SwarmResult) -> None:
            with results_path.open("a", encoding="utf-8") as results_file:
                results_file.write(json.dumps(_result_to_dict(result)) + "\n")

            if result["status"] == SwarmResultStatus.FAILED:
                with failures_path.open("a", encoding="utf-8") as failures_file:
                    failures_file.write(json.dumps(_result_to_dict(result)) + "\n")

        async def _execute_single_task(task: SwarmTask) -> SwarmResult:
            task_start = time.time()
            task_metadata = task.get("metadata")
            subagent_type = task.get("type", self.default_subagent_type)

            if subagent_type not in self.subagent_graphs:
                available = sorted(self.subagent_graphs.keys())
                msg = (
                    f"Unknown subagent type: {subagent_type}. Available: {available}. "
                    "Use one of the available types or omit 'type' to use the default."
                )
                result: SwarmResult = {
                    "task_id": task["id"],
                    "status": SwarmResultStatus.FAILED,
                    "error": "ValueError",
                    "message": msg,
                    "duration_ms": int((time.time() - task_start) * 1000),
                }
                if task_metadata is not None:
                    result["metadata"] = task_metadata
                return result

            subagent = self.subagent_graphs[subagent_type]
            try:
                subagent_state = {
                    "messages": [HumanMessage(content=task["description"])]
                }
                invoke_result = await asyncio.wait_for(
                    subagent.ainvoke(subagent_state),
                    timeout=self.timeout_seconds,
                )
            except TimeoutError:
                timeout_result: SwarmResult = {
                    "task_id": task["id"],
                    "status": SwarmResultStatus.FAILED,
                    "error": "TimeoutError",
                    "message": f"Task timed out after {self.timeout_seconds}s",
                    "duration_ms": int((time.time() - task_start) * 1000),
                }
                if task_metadata is not None:
                    timeout_result["metadata"] = task_metadata
                return timeout_result
            except Exception as exc:  # noqa: BLE001
                traceback_text = traceback.format_exc()
                error_result: SwarmResult = {
                    "task_id": task["id"],
                    "status": SwarmResultStatus.FAILED,
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback_text,
                    "duration_ms": int((time.time() - task_start) * 1000),
                }
                if task_metadata is not None:
                    error_result["metadata"] = task_metadata
                return error_result

            output = ""
            if invoke_result.get("messages"):
                final_msg = invoke_result["messages"][-1]
                content = getattr(final_msg, "content", final_msg)
                if isinstance(content, list):
                    text_parts: list[str] = []
                    for block in content:
                        if isinstance(block, str):
                            text_parts.append(block)
                        elif isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(str(block.get("text", "")))
                    output = "\n".join(text_parts)
                else:
                    output = str(content)

            success_result: SwarmResult = {
                "task_id": task["id"],
                "status": SwarmResultStatus.SUCCESS,
                "output": output,
                "duration_ms": int((time.time() - task_start) * 1000),
            }
            if task_metadata is not None:
                success_result["metadata"] = task_metadata
            return success_result

        semaphore = asyncio.Semaphore(run_concurrency)

        async def _run_with_limit(task: SwarmTask) -> SwarmResult:
            async with semaphore:
                task_id = task["id"]
                running.add(task_id)
                _report_progress()
                try:
                    return await _execute_single_task(task)
                finally:
                    running.discard(task_id)
                    _report_progress()

        _report_progress()
        task_futures = [asyncio.create_task(_run_with_limit(task)) for task in tasks]

        for task_future in asyncio.as_completed(task_futures):
            result = await task_future
            _write_result(result)
            if result["status"] == SwarmResultStatus.SUCCESS:
                completed.add(result["task_id"])
            else:
                failed.add(result["task_id"])
            _report_progress()

        duration_seconds = time.time() - start_time

        summary: SwarmSummary = {
            "run_id": run_id or generate_swarm_run_id(),
            "started_at": started_at or datetime.now(UTC).isoformat(timespec="seconds"),
            "total": len(tasks),
            "succeeded": len(completed),
            "failed": len(failed),
            "duration_seconds": round(duration_seconds, 2),
            "concurrency": run_concurrency,
            "results_path": str(results_path),
            "failures_path": str(failures_path),
        }

        summary_path = output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)

        return summary


def _result_to_dict(result: SwarmResult) -> dict[str, Any]:
    """Convert SwarmResult to a JSON-serializable dictionary.

    Returns:
        Dictionary representation suitable for JSONL output.
    """
    data: dict[str, Any] = {
        "task_id": result["task_id"],
        "status": result["status"].value,
        "duration_ms": result["duration_ms"],
    }
    if "output" in result:
        data["output"] = result["output"]
    if "error" in result:
        data["error"] = result["error"]
    if "message" in result:
        data["message"] = result["message"]
    if "traceback" in result:
        data["traceback"] = result["traceback"]
    if "metadata" in result:
        data["metadata"] = result["metadata"]
    return data


def get_default_output_dir(run_id: str | None = None) -> Path:
    """Get the default output directory for swarm results.

    Returns:
        Output directory path with UTC timestamp and run ID suffix.
    """
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
    resolved_run_id = run_id or generate_swarm_run_id()
    return Path.cwd() / "batch_results" / f"{timestamp}_{resolved_run_id}"


def generate_swarm_run_id() -> str:
    """Generate a compact run ID for swarm executions.

    Returns:
        Eight-character lowercase hexadecimal identifier.
    """
    return uuid.uuid4().hex[:8]
