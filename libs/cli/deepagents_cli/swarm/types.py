"""Type definitions for the task board and swarm execution systems."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NotRequired, TypedDict


class TaskStatus(str, Enum):
    """Task status values matching Claude Code's workflow."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class Task(TypedDict):
    """Task board task structure."""

    id: str
    subject: str
    description: str
    active_form: str | None
    status: TaskStatus
    metadata: dict[str, Any] | None
    blocks: list[str]
    blocked_by: list[str]
    owner: str | None
    created_at: str
    updated_at: str


# =============================================================================
# Swarm Execution Types
# =============================================================================


class SwarmTask(TypedDict):
    """Task definition for swarm batch execution.

    Required fields:
        id: Unique identifier for the task within the batch.
        description: Instructions for the subagent to execute.

    Optional fields:
        type: Subagent type to use (default: "general-purpose").
        blocked_by: List of task IDs that must complete before this task runs.
        metadata: User-defined metadata passed through to results.
    """

    id: str
    description: str
    type: NotRequired[str]
    blocked_by: NotRequired[list[str]]
    metadata: NotRequired[dict[str, Any]]


class SwarmResultStatus(str, Enum):
    """Status values for swarm task results."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class SwarmResult(TypedDict):
    """Result of a single swarm task execution.

    Fields:
        task_id: ID of the task this result belongs to.
        status: Execution status (success, failed, skipped).
        output: Subagent output (for successful tasks).
        error: Error type (for failed tasks).
        message: Error message or skip reason.
        duration_ms: Execution time in milliseconds (0 for skipped).
        metadata: User-defined metadata passed through from task.
    """

    task_id: str
    status: SwarmResultStatus
    output: NotRequired[str]
    error: NotRequired[str]
    message: NotRequired[str]
    duration_ms: int
    metadata: NotRequired[dict[str, Any]]


@dataclass
class SwarmProgress:
    """Progress information for swarm execution.

    Used for real-time progress reporting during batch execution.
    """

    total: int
    """Total number of tasks in the batch."""

    completed: int = 0
    """Number of tasks that have finished (success + failed)."""

    succeeded: int = 0
    """Number of tasks that succeeded."""

    failed: int = 0
    """Number of tasks that failed."""

    skipped: int = 0
    """Number of tasks skipped due to dependency failures."""

    running: list[str] = field(default_factory=list)
    """Task IDs currently being executed."""

    blocked: int = 0
    """Number of tasks waiting on dependencies."""

    @property
    def pending(self) -> int:
        """Number of tasks not yet started (excluding blocked and running)."""
        return self.total - self.completed - self.skipped - len(self.running) - self.blocked


class SwarmSummary(TypedDict):
    """Summary of a completed swarm batch execution.

    Written to summary.json in the output directory.
    """

    run_id: str
    started_at: str
    total: int
    succeeded: int
    failed: int
    skipped: int
    duration_seconds: float
    concurrency: int
    results_path: str
    failures_path: str
