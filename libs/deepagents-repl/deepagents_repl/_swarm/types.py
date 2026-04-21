"""Swarm task/result dataclasses and constants."""

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
    """A single unit of work to dispatch to a subagent.

    Built internally by the executor from rows in the target table; the
    description is the interpolated instruction and the response_schema
    is pulled from the execute options.
    """

    id: str
    description: str
    subagent_type: str | None = None
    response_schema: dict[str, Any] | None = None


@dataclass
class SwarmTaskResult:
    """The outcome of one dispatched task."""

    id: str
    subagent_type: str
    status: Literal["completed", "failed"]
    result: str | None = None
    error: str | None = None
