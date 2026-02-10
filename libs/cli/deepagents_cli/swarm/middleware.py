"""Minimal JSONL swarm execution middleware and core implementation."""

import asyncio
import json
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool

SWARM_SYSTEM_PROMPT = """## Swarm Tools (parallel JSONL execution)

You have access to `swarm_execute` for running many independent tasks in parallel.

Use it when the user provides a JSONL task file like:
```
{"id": "1", "description": "Analyze company A"}
{"id": "2", "description": "Analyze company B"}
```

Each task runs independently in its own subagent execution.
Use `num_parallel` to control worker count.

Task type support:
- If `type` is provided, it must match a configured subagent type.
- If unsure, omit `type` and use the default.
"""


class SwarmTask(TypedDict):
    """Task definition for batch swarm execution."""

    id: str
    description: str
    type: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]


class SwarmResultStatus(StrEnum):
    """Status values for swarm task results."""

    SUCCESS = "success"
    FAILED = "failed"


class SwarmResult(TypedDict):
    """Result of a single swarm task execution."""

    task_id: str
    status: SwarmResultStatus
    output: NotRequired[str]
    error: NotRequired[str]
    message: NotRequired[str]
    traceback: NotRequired[str]
    duration_ms: int
    metadata: NotRequired[dict[str, Any]]


@dataclass
class SwarmProgress:
    """Progress information for swarm execution."""

    total: int
    completed: int = 0
    succeeded: int = 0
    failed: int = 0
    running: list[str] = field(default_factory=list)

    @property
    def pending(self) -> int:
        """Number of tasks not yet started (excluding currently running)."""
        return self.total - self.completed - len(self.running)


class SwarmSummary(TypedDict):
    """Summary of a completed swarm batch execution."""

    run_id: str
    started_at: str
    total: int
    succeeded: int
    failed: int
    duration_seconds: float
    concurrency: int
    results_path: str
    failures_path: str


class TaskFileError(Exception):
    """Error parsing a task file."""


class SwarmExecutionError(Exception):
    """Error during swarm execution."""


def parse_task_file(path: str | Path) -> list[SwarmTask]:
    """Parse a JSONL task file into SwarmTask objects.

    Args:
        path: Path to a JSONL task file.

    Returns:
        List of parsed task definitions.

    Raises:
        FileNotFoundError: If the file does not exist.
        TaskFileError: If the file has invalid format or content.
    """
    task_path = Path(path)
    if not task_path.exists():
        msg = f"Task file not found: {task_path}"
        raise FileNotFoundError(msg)

    if task_path.suffix.lower() != ".jsonl":
        msg = f"Task file must be JSONL (.jsonl). Got: {task_path.name}"
        raise TaskFileError(msg)

    tasks: list[SwarmTask] = []

    with task_path.open("r", encoding="utf-8") as file_handle:
        for line_num, raw_line in enumerate(file_handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                raw_data = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"Invalid JSON on line {line_num}: {exc}"
                raise TaskFileError(msg) from exc

            data = _normalize_task_payload(raw_data, line_num)
            task = _validate_and_convert_task(
                data,
                line_num,
                default_id=f"auto-{line_num}",
            )
            tasks.append(task)

    if not tasks:
        msg = "Task file is empty"
        raise TaskFileError(msg)

    _validate_task_ids(tasks)
    return tasks


def _normalize_task_payload(
    raw_data: dict[str, Any] | str,
    line_num: int,
) -> dict[str, Any]:
    """Normalize a task payload into canonical dict form.

    Returns:
        Normalized task dictionary.

    Raises:
        TaskFileError: If the payload is not a JSON object or string.
    """
    if isinstance(raw_data, str):
        return {"description": raw_data}
    if not isinstance(raw_data, dict):
        msg = f"Line {line_num}: task entry must be a JSON object or string"
        raise TaskFileError(msg)

    data = dict(raw_data)
    if "description" not in data:
        for alias in ("task", "prompt"):
            if alias in data:
                data["description"] = data[alias]
                break
    return data


def _validate_and_convert_task(
    data: dict[str, Any], line_num: int, *, default_id: str
) -> SwarmTask:
    """Validate task data and convert to SwarmTask.

    Returns:
        Validated task dictionary.

    Raises:
        TaskFileError: If required fields or optional structures are invalid.
    """
    if "description" not in data or not str(data["description"]).strip():
        msg = f"Line {line_num}: missing required field 'description'"
        raise TaskFileError(msg)

    if "blocked_by" in data:
        msg = (
            "Field 'blocked_by' is not supported in simplified swarm mode. "
            "All tasks run independently in parallel."
        )
        raise TaskFileError(msg)

    task: SwarmTask = {
        "id": str(data.get("id", "")).strip() or default_id,
        "description": str(data["description"]).strip(),
    }

    if "type" in data:
        task["type"] = str(data["type"]).strip()

    if "metadata" in data:
        if not isinstance(data["metadata"], dict):
            msg = f"Line {line_num}: metadata must be a dict"
            raise TaskFileError(msg)
        task["metadata"] = data["metadata"]

    return task


def _validate_task_ids(tasks: list[SwarmTask]) -> None:
    """Validate that all task IDs are unique.

    Raises:
        TaskFileError: If duplicate IDs are found.
    """
    task_ids: set[str] = set()
    for task in tasks:
        task_id = task["id"]
        if task_id in task_ids:
            msg = f"Duplicate task ID: {task_id}"
            raise TaskFileError(msg)
        task_ids.add(task_id)


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


def generate_swarm_run_id() -> str:
    """Generate a compact run ID for swarm executions.

    Returns:
        Eight-character lowercase hexadecimal identifier.
    """
    return uuid.uuid4().hex[:8]


def get_default_output_dir(run_id: str | None = None) -> Path:
    """Get the default output directory for swarm results.

    Returns:
        Output directory path with UTC timestamp and run ID suffix.
    """
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
    resolved_run_id = run_id or generate_swarm_run_id()
    return Path.cwd() / "batch_results" / f"{timestamp}_{resolved_run_id}"


def _append_to_system_message(
    system_message: SystemMessage | None,
    text: str,
) -> SystemMessage:
    """Append text to a system message.

    Returns:
        Updated system message containing the additional text block.
    """
    if system_message is None:
        content: list[str | dict[str, object]] = []
    elif isinstance(system_message.content, str):
        content = [system_message.content]
    else:
        content = list(system_message.content)

    if content:
        text = f"\n\n{text}"
    content.append({"type": "text", "text": text})
    return SystemMessage(content=content)


class SwarmMiddleware(AgentMiddleware):
    """Middleware that provides the `swarm_execute` tool."""

    def __init__(
        self,
        subagent_graphs: dict[str, Runnable] | None = None,
        *,
        subagent_factory: Callable[[], dict[str, Runnable]] | None = None,
        default_concurrency: int = 10,
        max_concurrency: int = 50,
        timeout_seconds: float = 300.0,
        progress_callback: Callable[[SwarmProgress], None] | None = None,
    ) -> None:
        """Initialize middleware.

        Args:
            subagent_graphs: Pre-built map of subagent type names to runnable instances.
            subagent_factory: Lazy factory when subagent_graphs are not pre-built.
            default_concurrency: Default number of parallel workers.
            max_concurrency: Maximum allowed parallel workers.
            timeout_seconds: Timeout for individual task execution.
            progress_callback: Optional callback for progress updates.

        Raises:
            ValueError: If neither subagent_graphs nor subagent_factory is provided.
        """
        super().__init__()
        if subagent_graphs is None and subagent_factory is None:
            msg = "Either subagent_graphs or subagent_factory must be provided"
            raise ValueError(msg)

        self._subagent_graphs = subagent_graphs
        self._subagent_factory = subagent_factory
        self.default_concurrency = default_concurrency
        self.max_concurrency = max_concurrency
        self.timeout_seconds = timeout_seconds
        self.progress_callback = progress_callback

    @property
    def subagent_graphs(self) -> dict[str, Runnable]:
        """Get subagent graphs, initializing lazily if needed.

        Returns:
            Mapping of subagent type to runnable graph.

        Raises:
            RuntimeError: If no prebuilt graphs or factory are configured.
        """
        if self._subagent_graphs is None:
            if self._subagent_factory is None:
                msg = "No subagent_graphs or subagent_factory configured"
                raise RuntimeError(msg)
            self._subagent_graphs = self._subagent_factory()
        return self._subagent_graphs

    @property
    def tools(self) -> list[StructuredTool]:
        """Return the swarm execution tool."""
        return _create_swarm_tools(
            subagent_graphs_getter=lambda: self.subagent_graphs,
            default_concurrency=self.default_concurrency,
            max_concurrency=self.max_concurrency,
            timeout_seconds=self.timeout_seconds,
            progress_callback=self.progress_callback,
        )

    def wrap_model_call(  # noqa: PLR6301
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Append swarm guidance to the system prompt.

        Returns:
            Response from the downstream model handler.
        """
        system_message = _append_to_system_message(
            request.system_message,
            SWARM_SYSTEM_PROMPT,
        )
        return handler(request.override(system_message=system_message))

    async def awrap_model_call(  # noqa: PLR6301
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Append swarm guidance to the system prompt (async).

        Returns:
            Response from the downstream async model handler.
        """
        system_message = _append_to_system_message(
            request.system_message,
            SWARM_SYSTEM_PROMPT,
        )
        return await handler(request.override(system_message=system_message))


def _create_swarm_tools(
    subagent_graphs_getter: Callable[[], dict[str, Runnable]],
    default_concurrency: int,
    max_concurrency: int,
    timeout_seconds: float,
    progress_callback: Callable[[SwarmProgress], None] | None,
) -> list[StructuredTool]:
    """Create swarm tools.

    Returns:
        List containing the `swarm_execute` tool.
    """

    def _resolve_parallelism(*, num_parallel: int) -> int:
        return min(max(1, num_parallel), max_concurrency)

    async def swarm_execute(
        source: Annotated[
            str,
            "Path to a JSONL task file. Required field: description. "
            "Optional fields: id, type, metadata.",
        ],
        num_parallel: Annotated[
            int,
            "Maximum number of parallel subagent executions.",
        ] = default_concurrency,
        output_dir: Annotated[
            str | None,
            "Directory to write results. Defaults to "
            "./batch_results/<timestamp>_<run_id>/",
        ] = None,
    ) -> str:
        """Execute independent JSONL tasks in parallel.

        Returns:
            Human-readable summary of the swarm run and output paths.
        """
        parallelism = _resolve_parallelism(
            num_parallel=num_parallel,
        )

        try:
            tasks = parse_task_file(source)
        except (FileNotFoundError, TaskFileError) as exc:
            return f"Error: {exc}"

        run_id = generate_swarm_run_id()
        started_at = datetime.now(UTC).isoformat(timespec="seconds")
        resolved_output_dir = (
            Path(output_dir) if output_dir else get_default_output_dir(run_id=run_id)
        )

        executor = SwarmExecutor(
            subagent_graphs=subagent_graphs_getter(),
            timeout_seconds=timeout_seconds,
        )

        summary = await executor.execute(
            tasks=tasks,
            concurrency=parallelism,
            output_dir=resolved_output_dir,
            run_id=run_id,
            started_at=started_at,
            progress_callback=progress_callback,
        )

        lines = [
            (
                f"Batch execution complete: run {summary['run_id']} "
                f"({summary['started_at']}) - {summary['total']} tasks in "
                f"{summary['duration_seconds']}s"
            ),
            f"  Parallel workers: {summary['concurrency']}",
            f"  Succeeded: {summary['succeeded']}",
            f"  Failed: {summary['failed']}",
            "",
            "Results written to:",
            f"  {resolved_output_dir}/",
            "    summary.json",
            "    results.jsonl",
            "    failures.jsonl",
        ]
        return "\n".join(lines)

    return [
        StructuredTool.from_function(
            name="swarm_execute",
            coroutine=swarm_execute,
            description=(
                "Execute tasks from a JSONL file in parallel using subagents. "
                "Tasks run independently in parallel."
            ),
        )
    ]
