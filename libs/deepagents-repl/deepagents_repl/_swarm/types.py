"""Swarm task/result/summary dataclasses and constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

DEFAULT_CONCURRENCY = 10
"""Default number of concurrent subagent invocations per swarm call."""

MAX_CONCURRENCY = 50
"""Maximum allowed concurrency. Higher values risk rate limits."""

TASK_TIMEOUT_SECONDS = 300.0
"""Per-task timeout (300 seconds)."""


@dataclass
class SwarmTaskSpec:
    """A single row from tasks.jsonl — one unit of work to dispatch."""

    id: str
    """Unique identifier. Used to correlate with results."""

    description: str
    """Complete, self-contained prompt for the subagent."""

    subagent_type: str | None = None
    """Which subagent type to dispatch to. Defaults to ``general-purpose``."""

    response_schema: dict[str, Any] | None = None
    """JSON Schema enforcing structured output from the subagent.

    Must be a top-level object schema with a non-empty ``properties`` map
    (array schemas must be wrapped in an object). Requires a subagent
    factory to be available for the target subagent type; a task with a
    schema but no factory falls back to the default graph.
    """


@dataclass
class SwarmTaskResult:
    """A single row from results.jsonl — outcome of one dispatched task."""

    id: str
    subagent_type: str
    status: Literal["completed", "failed"]
    result: str | None = None
    error: str | None = None


@dataclass
class FailedTaskInfo:
    """Compact error summary for a failed task, included in the swarm tool response."""

    id: str
    error: str


@dataclass
class SwarmExecutionSummary:
    """Structured response returned by the executor."""

    total: int
    completed: int
    failed: int
    results_dir: str
    results: list[SwarmTaskResult]
    """Per-task results including both completed and failed entries.

    Present so callers can aggregate in-memory without reading
    ``results.jsonl`` back off the backend.
    """
    failed_tasks: list[FailedTaskInfo]
