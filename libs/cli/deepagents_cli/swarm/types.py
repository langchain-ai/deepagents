"""Type definitions for JSONL swarm execution."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, NotRequired, TypedDict


class SwarmTask(TypedDict):
    """Task definition for batch swarm execution.

    Required fields:
        id: Unique identifier for the task within the batch.
        description: Instructions for the subagent to execute.

    Optional fields:
        type: Subagent type to use (default: ``general-purpose``).
        metadata: User-defined metadata passed through to results.
    """

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
