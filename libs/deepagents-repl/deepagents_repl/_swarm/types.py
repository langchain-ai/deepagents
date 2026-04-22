"""Swarm task/result dataclasses and constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from deepagents_repl._swarm.filter import SwarmFilter

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


@dataclass
class CreateTableSource:
    """Source definition for ``swarm.create``.

    Exactly one of ``glob`` / ``file_paths`` / ``tasks`` should be
    populated. ``glob`` and ``file_paths`` produce ``{id, file}`` rows;
    ``tasks`` passes rows through as-is (each must have ``id: str``).
    """

    glob: str | list[str] | None = None
    file_paths: list[str] | None = None
    tasks: list[dict[str, Any]] | None = None


@dataclass
class SwarmExecuteOptions:
    """Options for ``swarm.execute``."""

    instruction: str
    """Template with ``{column}`` / ``{dotted.path}`` placeholders."""

    column: str = "result"
    """Column name to write results into."""

    filter: SwarmFilter | None = None
    """Only dispatch rows matching this clause; others pass through."""

    subagent_type: str | None = None
    """Subagent type for all dispatched rows. Defaults to ``general-purpose``."""

    response_schema: dict[str, Any] | None = None
    """JSON Schema for structured output. Parsed into the column value when set."""

    concurrency: int | None = None
    """Max concurrent subagent dispatches. Defaults to ``DEFAULT_CONCURRENCY``."""

    context: str | None = None
    """Free-form prose prepended to every subagent prompt.

    Use for dataset-wide rules, domain terms, classification criteria, or
    examples that apply across all rows — anything you'd otherwise have
    to duplicate inside each per-row ``instruction`` template. Kept
    separate so per-row context windows are smaller and the shared
    material is written once.
    """

    batch_size: int | None = None
    """Number of rows to group into a single subagent call.

    Defaults to 1 (one row per call). When ``> 1``, requires
    ``response_schema`` — the executor wraps it into a
    ``{results: [{id, ...}]}`` envelope, dispatches one subagent per
    batch with a combined prompt, and unpacks results back to rows by
    id.
    """


@dataclass
class SwarmResultEntry:
    """Per-row outcome included in the :class:`SwarmSummary`."""

    id: str
    subagent_type: str
    status: Literal["completed", "failed"]
    result: str | None = None
    error: str | None = None


@dataclass
class SwarmSummary:
    """In-memory summary returned by ``swarm.execute``.

    Results are also written as columns on the source table rows, so the
    summary is a convenience view — not the canonical store.
    """

    total: int
    completed: int
    failed: int
    skipped: int
    file: str
    column: str
    results: list[SwarmResultEntry] = field(default_factory=list)
    failed_tasks: list[dict[str, str]] = field(default_factory=list)
