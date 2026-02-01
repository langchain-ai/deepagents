"""Middleware for task board and swarm execution tools."""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool

from deepagents_cli.swarm.executor import SwarmExecutor, get_default_output_dir
from deepagents_cli.swarm.graph import CycleError
from deepagents_cli.swarm.parser import TaskFileError, parse_task_file
from deepagents_cli.swarm.task_store import TaskStore
from deepagents_cli.swarm.types import SwarmProgress, TaskStatus


class TaskBoardMiddleware(AgentMiddleware):
    """Middleware that provides task board tools.

    The task board tools are scoped to a session_id which is extracted from
    the runtime configurable at invocation time.

    Usage:
        agent = create_cli_agent(...)
        result = agent.invoke(
            {"messages": [...]},
            config={"configurable": {"thread_id": "my-session-123"}}
        )
    """

    def __init__(self, task_store: TaskStore | None = None):
        super().__init__()
        self.task_store = task_store or TaskStore()

    @property
    def tools(self) -> list[StructuredTool]:
        """Return task board tools that extract session_id from runtime."""
        return _create_task_board_tools_with_runtime(self.task_store)


def _create_task_board_tools_with_runtime(task_store: TaskStore) -> list[StructuredTool]:
    """Create task board tools that get session_id from runtime config."""

    def _get_session_id(runtime: ToolRuntime) -> str:
        """Extract session_id from runtime configurable."""
        configurable = runtime.config.get("configurable", {})
        session_id = configurable.get("thread_id")
        if not session_id:
            return "default-session"
        return session_id

    async def task_create(
        subject: str,
        description: str,
        active_form: str | None = None,
        metadata: dict[str, Any] | None = None,
        runtime: ToolRuntime = None,
    ) -> str:
        """Create a new task on the task board. All tasks start as 'pending'.

        Args:
            subject: Brief title in imperative form (e.g., 'Fix auth bug')
            description: Detailed requirements for the task
            active_form: Present continuous form for display (e.g., 'Fixing auth bug')
            metadata: Arbitrary key-value data
        """
        session_id = _get_session_id(runtime)
        task = await task_store.create_task(
            session_id=session_id,
            subject=subject,
            description=description,
            active_form=active_form,
            metadata=metadata,
        )
        return f"Task #{task['id']} created successfully"

    async def task_get(
        task_id: str,
        runtime: ToolRuntime = None,
    ) -> str:
        """Get full details of a task.

        Args:
            task_id: The task ID to retrieve
        """
        session_id = _get_session_id(runtime)
        task = await task_store.get_task(session_id, task_id)
        if not task:
            return f"Task #{task_id} not found"

        lines = [
            f"Task #{task['id']}: {task['subject']}",
            f"Status: {task['status'].value}",
            f"Description: {task['description']}",
        ]
        if task["blocked_by"]:
            lines.append(
                f"Blocked by: {', '.join(f'#{t}' for t in task['blocked_by'])}"
            )
        if task["owner"]:
            lines.append(f"Owner: {task['owner']}")
        return "\n".join(lines)

    async def task_update(
        task_id: str,
        status: str | None = None,
        subject: str | None = None,
        description: str | None = None,
        active_form: str | None = None,
        owner: str | None = None,
        metadata: dict[str, Any] | None = None,
        add_blocks: list[str] | None = None,
        add_blocked_by: list[str] | None = None,
        runtime: ToolRuntime = None,
    ) -> str:
        """Update task status, details, or dependencies.

        Args:
            task_id: The task ID to update
            status: 'pending', 'in_progress', or 'completed'
            subject: New title
            description: New description
            active_form: New display text
            owner: Agent name to assign
            metadata: Metadata to merge
            add_blocks: Task IDs this task blocks
            add_blocked_by: Task IDs that block this task
        """
        session_id = _get_session_id(runtime)
        status_enum = TaskStatus(status) if status else None
        task = await task_store.update_task(
            session_id,
            task_id,
            status=status_enum,
            subject=subject,
            description=description,
            active_form=active_form,
            owner=owner,
            metadata=metadata,
            add_blocks=add_blocks,
            add_blocked_by=add_blocked_by,
        )
        return f"Task #{task_id} updated" if task else f"Task #{task_id} not found"

    async def task_list(
        runtime: ToolRuntime = None,
    ) -> str:
        """List all tasks with status."""
        session_id = _get_session_id(runtime)
        tasks = await task_store.list_tasks(session_id)
        if not tasks:
            return "No tasks"

        lines = []
        for task in tasks:
            blocked = ""
            if task["blocked_by"]:
                active_blockers = []
                for b_id in task["blocked_by"]:
                    b = await task_store.get_task(session_id, b_id)
                    if b and b["status"] != TaskStatus.COMPLETED:
                        active_blockers.append(b_id)
                if active_blockers:
                    blocked = (
                        f" [blocked by {', '.join(f'#{t}' for t in active_blockers)}]"
                    )

            lines.append(
                f"#{task['id']} [{task['status'].value}] {task['subject']}{blocked}"
            )

        return "\n".join(lines)

    return [
        StructuredTool.from_function(
            name="TaskCreate",
            coroutine=task_create,
            description="Create a new task on the task board for tracking work with dependencies.",
        ),
        StructuredTool.from_function(
            name="TaskGet",
            coroutine=task_get,
            description="Get full details of a task by ID.",
        ),
        StructuredTool.from_function(
            name="TaskUpdate",
            coroutine=task_update,
            description="Update task status/details. Workflow: pending → in_progress → completed",
        ),
        StructuredTool.from_function(
            name="TaskList",
            coroutine=task_list,
            description="List all tasks with status and blockers.",
        ),
    ]


# =============================================================================
# Swarm Middleware
# =============================================================================

SWARM_SYSTEM_PROMPT = """## `swarm_execute` (batch task execution)

You have access to a `swarm_execute` tool for running multiple tasks in parallel using subagents.

When to use swarm_execute:
- Processing many similar items (files, data records, etc.) in parallel
- Running batch analysis or transformation tasks
- Any scenario where you need to execute many independent (or dependency-ordered) tasks

How to use:
1. Create a JSONL or CSV file with task definitions
2. Call swarm_execute with the file path and desired concurrency
3. Results are written to output files; summary is returned

Task file format (JSONL):
```
{"id": "1", "description": "Analyze file A"}
{"id": "2", "description": "Analyze file B"}
{"id": "3", "description": "Compare results", "blocked_by": ["1", "2"]}
```

The tool handles dependencies automatically - tasks with `blocked_by` wait for their
dependencies to complete first.
"""


class SwarmMiddleware(AgentMiddleware):
    """Middleware that provides the swarm_execute tool for batch task execution.

    The swarm_execute tool allows the agent to run multiple tasks in parallel
    using subagents, with optional dependency management.

    This middleware can be initialized in two ways:

    1. With pre-built subagent_graphs (for reusing existing subagent infrastructure):
        SwarmMiddleware(subagent_graphs={"general-purpose": agent_graph})

    2. With a subagent_factory callable (for lazy initialization):
        SwarmMiddleware(subagent_factory=lambda: build_subagent_graphs())

    Usage:
        agent = create_cli_agent(
            model="...",
            middleware=[
                SwarmMiddleware(subagent_graphs=subagent_graphs),
                ...
            ]
        )
    """

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
        """Initialize the SwarmMiddleware.

        Args:
            subagent_graphs: Pre-built map of subagent type names to runnable instances.
                            Either this or subagent_factory must be provided.
            subagent_factory: Callable that returns subagent_graphs when invoked.
                             Called lazily on first tool use if subagent_graphs not provided.
            default_concurrency: Default number of parallel workers.
            max_concurrency: Maximum allowed concurrency.
            timeout_seconds: Timeout for individual task execution.
            progress_callback: Optional callback for progress updates.

        Raises:
            ValueError: If neither subagent_graphs nor subagent_factory is provided.
        """
        super().__init__()

        if subagent_graphs is None and subagent_factory is None:
            raise ValueError("Either subagent_graphs or subagent_factory must be provided")

        self._subagent_graphs = subagent_graphs
        self._subagent_factory = subagent_factory
        self.default_concurrency = default_concurrency
        self.max_concurrency = max_concurrency
        self.timeout_seconds = timeout_seconds
        self.progress_callback = progress_callback

    @property
    def subagent_graphs(self) -> dict[str, Runnable]:
        """Get or lazily initialize subagent graphs."""
        if self._subagent_graphs is None:
            if self._subagent_factory is None:
                raise RuntimeError("No subagent_graphs or subagent_factory configured")
            self._subagent_graphs = self._subagent_factory()
        return self._subagent_graphs

    @property
    def tools(self) -> list[StructuredTool]:
        """Return swarm execution tools."""
        return _create_swarm_tools(
            subagent_graphs_getter=lambda: self.subagent_graphs,
            default_concurrency=self.default_concurrency,
            max_concurrency=self.max_concurrency,
            timeout_seconds=self.timeout_seconds,
            progress_callback=self.progress_callback,
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Add swarm instructions to system message."""
        from deepagents.middleware._utils import append_to_system_message

        new_system_message = append_to_system_message(request.system_message, SWARM_SYSTEM_PROMPT)
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Add swarm instructions to system message."""
        from deepagents.middleware._utils import append_to_system_message

        new_system_message = append_to_system_message(request.system_message, SWARM_SYSTEM_PROMPT)
        return await handler(request.override(system_message=new_system_message))


def _create_swarm_tools(
    subagent_graphs_getter: Callable[[], dict[str, Runnable]],
    default_concurrency: int,
    max_concurrency: int,
    timeout_seconds: float,
    progress_callback: Callable[[SwarmProgress], None] | None,
) -> list[StructuredTool]:
    """Create swarm execution tools.

    Args:
        subagent_graphs_getter: Callable that returns subagent_graphs when invoked.
                                Used for lazy initialization.
    """

    async def swarm_execute(
        source: Annotated[
            str,
            "Path to a JSONL or CSV file with task definitions. Required fields: 'id', "
            "'description'. Optional: 'type' (subagent type), 'blocked_by', 'metadata'.",
        ],
        concurrency: Annotated[
            int,
            "Maximum number of parallel subagent executions. Higher = faster but more load.",
        ] = default_concurrency,
        output_dir: Annotated[
            str | None,
            "Directory to write results. Defaults to ./batch_results/<timestamp>/",
        ] = None,
        runtime: ToolRuntime = None,
    ) -> str:
        """Execute a batch of tasks in parallel using subagents.

        Tasks are defined in a JSONL or CSV file. Each task is executed by a subagent,
        respecting any dependency ordering (blocked_by).

        Results are written to:
        - summary.json: Overview and statistics
        - results.jsonl: All task outputs
        - failures.jsonl: Failed and skipped tasks

        Args:
            source: Path to task file (JSONL or CSV)
            concurrency: Max parallel workers (default: 10)
            output_dir: Results directory (default: ./batch_results/<timestamp>/)

        Returns:
            Summary string with statistics and file paths.
        """
        # Validate concurrency
        actual_concurrency = min(max(1, concurrency), max_concurrency)

        # Parse task file
        try:
            tasks = parse_task_file(source)
        except FileNotFoundError as e:
            return f"Error: {e}"
        except TaskFileError as e:
            return f"Error parsing task file: {e}"

        # Determine output directory
        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = get_default_output_dir()

        # Get subagent graphs (lazy initialization)
        subagent_graphs = subagent_graphs_getter()

        # Create executor
        executor = SwarmExecutor(
            subagent_graphs=subagent_graphs,
            timeout_seconds=timeout_seconds,
        )

        # Execute
        try:
            summary = await executor.execute(
                tasks=tasks,
                concurrency=actual_concurrency,
                output_dir=out_path,
                progress_callback=progress_callback,
            )
        except CycleError as e:
            return f"Error: {e}"

        # Format response
        lines = [
            f"Batch execution complete: {summary['total']} tasks in {summary['duration_seconds']}s",
            f"  Succeeded: {summary['succeeded']}",
            f"  Failed: {summary['failed']}",
            f"  Skipped: {summary['skipped']}",
            "",
            "Results written to:",
            f"  {out_path}/",
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
                "Execute a batch of tasks in parallel using subagents. "
                "Tasks are defined in a JSONL/CSV file with optional dependency ordering."
            ),
        ),
    ]
