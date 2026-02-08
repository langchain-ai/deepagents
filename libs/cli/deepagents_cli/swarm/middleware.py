"""Middleware for minimal JSONL swarm execution."""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool

from deepagents_cli.swarm.executor import (
    SwarmExecutor,
    generate_swarm_run_id,
    get_default_output_dir,
)
from deepagents_cli.swarm.parser import TaskFileError, parse_task_file
from deepagents_cli.swarm.types import SwarmProgress

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
