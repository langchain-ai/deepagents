"""Middleware for parallel subagent execution via a ``swarm`` tool."""

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, NotRequired, TypedDict

from langchain.agents.middleware.types import AgentMiddleware, ContextT, ModelRequest, ModelResponse, ResponseT
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphInterrupt
from pydantic import TypeAdapter, ValidationError

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.subagents import _SubagentSpec

# Keys excluded from parent state when building subagent state.
# Imported inline to avoid circular imports; mirrors the constant in subagents.py.
_EXCLUDED_STATE_KEYS = {"messages", "todos", "structured_response", "skills_metadata", "memory_contents"}

DEFAULT_SWARM_CONCURRENCY = 10
"""Default concurrency limit for swarm execution."""

MAX_SWARM_CONCURRENCY = 50
"""Maximum allowed concurrency."""

SWARM_TASK_TIMEOUT_SECONDS = 300
"""Per-task timeout in seconds."""

SWARM_SYSTEM_PROMPT = """
## `swarm` (parallel subagent execution)

Use `swarm` to fan out many independent tasks across multiple subagents and aggregate their results.

### When to use swarm

**Trigger condition**: Use swarm when the input contains too much data to process in a single pass. Indicators: the file or dataset contains hundreds of items that each need individual analysis, or exceeds ~500 lines. When in doubt, check the size and prefer swarm over attempting to process a large input inline.

Also use `swarm` when:
- A task requires applying intelligence to each item in a large collection
- Work can be decomposed into many independent, parallel subtasks

Use `task` instead when:
- You have a small number of independent subtasks
- Each subtask depends on the result of a previous one
- The work is exploratory or adaptive

### How to use swarm

Before calling swarm, understand what you're working with. Explore the data to learn its structure, format, and content using whatever tools are available. The goal is to write task descriptions detailed enough that each subagent can execute without needing to figure anything out on its own.

Once you understand the data:

1. **Generate tasks.** Write a generation script via `execute` that produces a `tasks.jsonl` file — one JSON object per line, each with `id`, `description`, and optional `subagent_type`. Each task should be a self-contained unit of work. **Prefer many small tasks over few large ones** — all tasks run in parallel, so 50 small tasks finish in roughly the same wall-clock time as 5 large ones. When splitting a file, aim for **30-60 lines** per chunk.
2. **Call swarm.** Pass the path to your `tasks.jsonl` file.
3. **Aggregate results.** Write an aggregation script via `execute` that reads `<results_dir>/results.jsonl` and combines the subagent outputs into a final answer.

### Task description quality

Each subagent receives **only its task description** — no other context. The quality of your descriptions determines the quality of swarm results. Invest time upfront to get them right.

Good task descriptions are **prescriptive**: they tell the subagent the data format, the processing logic, the exact range of data to work on, and the expected output format. The subagent should not need to explore or interpret — just execute.

When subagent results need to be aggregated (counting, classification, extraction), instruct each subagent to respond with **structured JSON only** — no explanations, no tables, just the JSON object. Include the exact output schema in the task description.

### Error handling

Each task runs exactly once — there are no automatic retries. If some tasks fail, the swarm summary includes a `failed_tasks` array with each failed task's ID and error message. Use this to decide:
- **Retry via swarm**: generate a new tasks.jsonl targeting just the failures (with modifications) and call swarm again.
- **Retry individually**: use `task` for a small number of failures.
- **Proceed with partial results**: aggregate what completed and skip the rest.

### Important: one swarm call per question

**Never re-run swarm to verify or cross-check results.** Swarm is expensive — treat the first run's per-task outputs as authoritative. If you need to validate, do it in the aggregation script (e.g., check that each chunk returned the expected number of items). Do not generate a second tasks.jsonl or call swarm again for the same question.

### Decomposition patterns

**Flat fan-out**: Split a dataset into equal chunks. All tasks are identical in structure.
Good for: large files, classification, extraction.

**One-per-item**: One task per discrete unit (file, document, URL).
Good for: summarizing collections, processing independent documents.

**Dimensional**: Multiple tasks examine the same input from different angles.
Good for: code review, multi-criteria evaluation."""  # noqa: E501

SWARM_TOOL_DESCRIPTION = """Execute a batch of independent tasks in parallel across multiple subagents.

## Workflow

1. Write a generation script via `execute` that produces a tasks.jsonl file with one JSON object per line:
   ```json
   {{"id": "chunk_0", "description": "Read lines 1-100 of data.txt. Process each item. Return JSON results.", "subagent_type": "general-purpose"}}
   {{"id": "chunk_1", "description": "Read lines 101-200 of data.txt. Process each item. Return JSON results.", "subagent_type": "general-purpose"}}
   ```
2. Call `swarm` with the path to the tasks.jsonl file.
3. The tool returns a JSON summary with `total`, `completed`, `failed`, and `results_dir`.
   Results are written to `<results_dir>/results.jsonl` — each line is the original task enriched with `status`, `result`, and/or `error` fields.
4. Write an aggregation script via `execute` that reads `<results_dir>/results.jsonl` and combines the outputs.

## tasks.jsonl fields

- "id" (string, required): unique task identifier
- "description" (string, required): complete, self-contained prompt — the subagent receives NOTHING else
- "subagent_type" (string, optional): which subagent to use (default: "general-purpose")

## After execution

The tool returns:
```json
{{"total": 20, "completed": 19, "failed": 1,
  "results_dir": "swarm_runs/<uuid>",
  "failed_tasks": [{{"id": "chunk_5", "error": "timed out"}}]}}
```

Each task runs exactly once — there are no automatic retries. Use the `failed_tasks` array to decide how to handle failures.

Available subagent types: {available_agents}
"""


class SwarmTaskSpec(TypedDict):
    """A single task line in a ``tasks.jsonl`` file.

    Fields:
        id: Unique task identifier (must be unique within the task list).
        description: Complete, self-contained prompt for the subagent.
        subagent_type: Which subagent to dispatch to. Defaults to
            ``"general-purpose"`` when omitted.
    """

    id: str
    description: str
    subagent_type: NotRequired[str]


class ParsedSwarmConfig(TypedDict):
    """Result of parsing a tasks JSONL file."""

    tasks: list[SwarmTaskSpec] | None
    error: str | None


def _parse_tasks_jsonl(content: str) -> ParsedSwarmConfig:
    """Parse and validate a ``tasks.jsonl`` string into task specs.

    Validates that each line is valid JSON with required fields, that all
    task IDs are unique, and that at least one task is present.

    Args:
        content: Raw JSONL string (one JSON object per line).

    Returns:
        Parsed config with task list or error message.
    """
    lines = [line for line in content.split("\n") if line.strip()]
    if not lines:
        return {"tasks": None, "error": "tasks.jsonl is empty. The generation script must write at least one task."}

    tasks: list[SwarmTaskSpec] = []
    seen_ids: set[str] = set()
    errors: list[str] = []

    adapter = TypeAdapter(SwarmTaskSpec)
    for idx, line in enumerate(lines):
        line_number = idx + 1
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"Line {line_number}: invalid JSON")
            continue

        try:
            task = adapter.validate_python(parsed)
        except ValidationError as e:
            messages = [issue["msg"] for issue in e.errors()]
            errors.append(f"Line {line_number}: {', '.join(messages)}")
            continue

        if task["id"] in seen_ids:
            errors.append(f'Line {line_number}: duplicate task id "{task["id"]}"')
            continue

        seen_ids.add(task["id"])
        tasks.append(task)

    if errors:
        return {"tasks": None, "error": "tasks.jsonl validation failed:\n" + "\n".join(errors)}

    return {"tasks": tasks, "error": None}


def _build_swarm_tool(  # noqa: C901, PLR0915
    subagent_specs: list[_SubagentSpec],
    backend: BackendProtocol | BackendFactory,
) -> StructuredTool:
    """Create the ``swarm`` tool.

    The ``swarm`` tool reads a JSON config file containing a list of task specs
    and runs those tasks in parallel across subagents.

    Args:
        subagent_specs: Compiled subagent specs with name, description, and runnable.
        backend: Backend for file I/O.

    Returns:
        A structured tool named ``swarm``.
    """
    subagent_graphs: dict[str, Runnable] = {spec["name"]: spec["runnable"] for spec in subagent_specs}
    subagent_description_str = "\n".join(f"- {s['name']}: {s['description']}" for s in subagent_specs)
    description = SWARM_TOOL_DESCRIPTION.format(available_agents=subagent_description_str)

    async def _run_single_task(
        task: SwarmTaskSpec,
        subagent: Runnable,
        parent_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a single task with a timeout. Returns a lean result dict."""
        subagent_state = dict(parent_state)
        subagent_state["messages"] = [HumanMessage(content=task["description"])]
        subagent_type = task.get("subagent_type", "general-purpose")

        try:
            result = await asyncio.wait_for(
                subagent.ainvoke(subagent_state),
                timeout=SWARM_TASK_TIMEOUT_SECONDS,
            )
            messages = result.get("messages", [])
            text = messages[-1].text.rstrip() if messages and messages[-1].text else ""
            return {"id": task["id"], "subagent_type": subagent_type, "status": "completed", "result": text}
        except Exception as exc:  # noqa: BLE001
            return {"id": task["id"], "subagent_type": subagent_type, "status": "failed", "error": str(exc)}

    async def _execute_swarm(
        tasks: list[SwarmTaskSpec],
        parent_state: dict[str, Any],
        effective_concurrency: int,
    ) -> list[dict[str, Any]]:
        """Dispatch all tasks in parallel under a concurrency semaphore.

        Each task runs exactly once — there are no retries. The
        orchestrator owns error recovery.
        """
        semaphore = asyncio.Semaphore(effective_concurrency)

        for task in tasks:
            subagent_type = task.get("subagent_type", "general-purpose")
            if subagent_type not in subagent_graphs:
                allowed = ", ".join(f'"{k}"' for k in subagent_graphs)
                msg = f'Task "{task["id"]}" references unknown subagent_type "{subagent_type}". Available: {allowed}'
                raise ValueError(msg)

        async def run_with_semaphore(task: SwarmTaskSpec) -> dict[str, Any]:
            subagent = subagent_graphs[task.get("subagent_type", "general-purpose")]
            async with semaphore:
                return await _run_single_task(task, subagent, parent_state)

        results = await asyncio.gather(
            *[run_with_semaphore(t) for t in tasks],
            return_exceptions=True,
        )

        final: list[dict[str, Any]] = []
        for idx, raw_result in enumerate(results):
            if isinstance(raw_result, GraphInterrupt):
                raise raw_result
            if isinstance(raw_result, BaseException):
                subagent_type = tasks[idx].get("subagent_type", "general-purpose")
                final.append({"id": tasks[idx]["id"], "subagent_type": subagent_type, "status": "failed", "error": str(raw_result)})
            else:
                final.append(raw_result)
        return final

    async def aswarm(
        tasks_path: Annotated[str, "Path to the tasks.jsonl file produced by the generation script."],
        runtime: ToolRuntime,
        concurrency: Annotated[
            int | None,
            f"Maximum number of subagents running simultaneously. Default: {DEFAULT_SWARM_CONCURRENCY}, max: {MAX_SWARM_CONCURRENCY}.",
        ] = None,
    ) -> ToolMessage:
        """Run swarm tasks."""
        resolved_backend = backend(runtime) if callable(backend) else backend  # ty: ignore[call-top-callable]

        responses = await resolved_backend.adownload_files([tasks_path])
        response = responses[0]
        if response.error:
            return ToolMessage(
                content=f'Failed to read tasks file at "{tasks_path}". '
                f"Ensure the generation script writes the file to this exact path and try again.",
                status="error",
                tool_call_id=runtime.tool_call_id,
            )
        file_content = response.content
        if not isinstance(file_content, bytes):
            return ToolMessage(
                content=f"Content was expected to be bytes. Got {type(file_content)}.",
                status="error",
                tool_call_id=runtime.tool_call_id,
            )
        parsed = _parse_tasks_jsonl(file_content.decode("utf-8"))
        if parsed["error"] is not None:
            return ToolMessage(content=parsed["error"], status="error", tool_call_id=runtime.tool_call_id)
        tasks = parsed["tasks"]
        if tasks is None:
            msg = "parsed swarm tasks unexpectedly missing"
            raise AssertionError(msg)

        parent_state = {k: v for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}

        effective_concurrency = max(1, min(concurrency or DEFAULT_SWARM_CONCURRENCY, MAX_SWARM_CONCURRENCY))

        task_results = await _execute_swarm(tasks, parent_state, effective_concurrency)

        artifacts_root = resolved_backend.artifacts_root if isinstance(resolved_backend, CompositeBackend) else "/"
        _root = artifacts_root.rstrip("/")
        results_dir = f"{_root}/swarm_runs/{uuid.uuid4()}"
        results_path = f"{results_dir}/results.jsonl"
        results_content = "\n".join(json.dumps(r) for r in task_results) + "\n"
        await resolved_backend.aupload_files([(results_path, results_content.encode("utf-8"))])

        completed_count = sum(1 for r in task_results if r["status"] == "completed")
        failed_results = [r for r in task_results if r["status"] == "failed"]
        summary: dict[str, Any] = {
            "total": len(task_results),
            "completed": completed_count,
            "failed": len(failed_results),
            "results_dir": results_dir,
            "failed_tasks": [{"id": r["id"], "error": r.get("error", "unknown")} for r in failed_results],
        }

        return ToolMessage(
            content=json.dumps(summary),
            status="success",
            tool_call_id=runtime.tool_call_id,
        )

    return StructuredTool.from_function(
        name="swarm",
        coroutine=aswarm,
        description=description,
    )


class SwarmMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware for parallel subagent execution via a ``swarm`` tool.

    Registers the ``swarm`` tool and injects the system prompt that teaches
    the orchestrator the three-step workflow (generate → swarm → aggregate).

    Args:
        backend: Backend for file operations and execution.
        subagent_specs: Compiled subagent specs (name, description, runnable).
            These are the same subagent graphs used by the task tool.

    Example:
        ```python
        from deepagents.middleware.swarm import SwarmMiddleware

        SwarmMiddleware(
            backend=my_backend,
            subagent_specs=subagent_middleware.subagent_specs,
        )
        ```
    """

    def __init__(
        self,
        *,
        backend: BackendProtocol | BackendFactory,
        subagent_specs: list[_SubagentSpec],
    ) -> None:
        """Initialize the ``SwarmMiddleware``."""
        super().__init__()
        self.tools = [_build_swarm_tool(subagent_specs, backend)]
        self.system_prompt = SWARM_SYSTEM_PROMPT

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Inject the swarm system prompt."""
        new_system_message = append_to_system_message(request.system_message, self.system_prompt)
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """(async) Inject the swarm system prompt."""
        new_system_message = append_to_system_message(request.system_message, self.system_prompt)
        return await handler(request.override(system_message=new_system_message))
