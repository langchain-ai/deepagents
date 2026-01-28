"""Type definitions for the task board system."""

from enum import Enum
from typing import Any, TypedDict


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
