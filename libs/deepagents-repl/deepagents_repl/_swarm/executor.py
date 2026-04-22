"""Parallel subagent dispatch against a JSONL table."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage

from deepagents_repl._swarm.filter import SwarmFilter, evaluate_filter
from deepagents_repl._swarm.interpolate import interpolate_instruction
from deepagents_repl._swarm.parse import parse_table_jsonl, serialize_table_jsonl
from deepagents_repl._swarm.table import WriteCallback
from deepagents_repl._swarm.types import (
    DEFAULT_CONCURRENCY,
    MAX_CONCURRENCY,
    TASK_TIMEOUT_SECONDS,
    SwarmResultEntry,
    SwarmSummary,
    SwarmTaskResult,
    SwarmTaskSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langchain_core.runnables import Runnable


SubagentFactory = Callable[[Any], "Runnable"]
"""Compile a subagent variant bound to a given ``response_format``.

Passed ``None`` to produce the default (no structured output) graph.
Used by the swarm executor to lazily compile per-schema variants.
"""


# State keys excluded when passing state to subagents.
# Mirrors ``_EXCLUDED_STATE_KEYS`` in ``deepagents/middleware/subagents.py``
# — we accept both the Python names (``structured_response``,
# ``skills_metadata``, ``memory_contents``) and the camelCase variants
# (``structuredResponse``, ``skillsMetadata``, ``memoryContents``) so
# the exclusion works regardless of which naming the caller uses.
_EXCLUDED_STATE_KEYS = frozenset(
    {
        "messages",
        "todos",
        "structured_response",
        "structuredResponse",
        "skills_metadata",
        "skillsMetadata",
        "memory_contents",
        "memoryContents",
    }
)

# Content block types filtered out when extracting subagent output text.
_INVALID_CONTENT_BLOCK_TYPES = frozenset(
    {
        "tool_use",
        "thinking",
        "redacted_thinking",
    }
)


def _filter_state_for_subagent(state: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in state.items() if k not in _EXCLUDED_STATE_KEYS}


def _extract_result_text(result: Mapping[str, Any]) -> str:
    """Extract the text content from a subagent's final message.

    Handles string content, list-of-content-blocks (dicts or objects with
    a ``text`` attribute), and falls back to a structured response if
    present. Filters out tool_use/thinking/redacted_thinking blocks.
    """
    structured = result.get("structured_response")
    if structured is None:
        structured = result.get("structuredResponse")
    if structured is not None:
        if hasattr(structured, "model_dump_json"):
            return structured.model_dump_json()
        try:
            return json.dumps(structured, default=str)
        except (TypeError, ValueError):
            return str(structured)

    messages = result.get("messages") or []
    if not messages:
        return "Task completed"
    last = messages[-1]
    content = getattr(last, "content", last)

    if isinstance(content, str):
        return content or "Task completed"

    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            block_type = _block_type(block)
            if block_type in _INVALID_CONTENT_BLOCK_TYPES:
                continue
            text = _block_text(block)
            if text is not None:
                texts.append(text)
            else:
                try:
                    texts.append(json.dumps(block, default=str))
                except (TypeError, ValueError):
                    texts.append(str(block))
        if not texts:
            return "Task completed"
        return "\n".join(texts)

    return "Task completed"


def _block_type(block: Any) -> str | None:
    if isinstance(block, dict):
        value = block.get("type")
        return value if isinstance(value, str) else None
    return getattr(block, "type", None)


def _block_text(block: Any) -> str | None:
    if isinstance(block, dict):
        value = block.get("text")
        return value if isinstance(value, str) else None
    return getattr(block, "text", None)


def _prepend_context(prompt: str, context: str | None) -> str:
    """Prepend orchestrator-supplied ``context`` to a subagent prompt.

    Returns the prompt unchanged when no context is supplied or when
    it's whitespace-only.
    """
    if context is None:
        return prompt
    trimmed = context.strip()
    if not trimmed:
        return prompt
    return f"{trimmed}\n\n---\n\n{prompt}"


async def _dispatch_task(
    task: SwarmTaskSpec,
    subagent: Runnable,
    filtered_state: dict[str, Any],
    task_timeout_seconds: float,
    cancel_event: asyncio.Event | None = None,
    context: str | None = None,
) -> SwarmTaskResult:
    subagent_type = task.subagent_type or "general-purpose"
    # Short-circuit: if the outer eval has already aborted, don't start a
    # new ainvoke. In-flight ainvoke calls are unwound via asyncio task
    # cancellation propagating from the outer wait_for — we don't need a
    # separate cancel-race here.
    if cancel_event is not None and cancel_event.is_set():
        return SwarmTaskResult(
            id=task.id,
            subagent_type=subagent_type,
            status="failed",
            error="Aborted",
        )
    prompt = _prepend_context(task.description, context)
    subagent_state = {
        **filtered_state,
        "messages": [HumanMessage(content=prompt)],
    }
    try:
        result = await asyncio.wait_for(
            subagent.ainvoke(subagent_state),
            timeout=task_timeout_seconds,
        )
    except TimeoutError:
        return SwarmTaskResult(
            id=task.id,
            subagent_type=subagent_type,
            status="failed",
            error=f"Timed out after {task_timeout_seconds:g}s",
        )
    except Exception as exc:  # noqa: BLE001 — surface subagent errors as failed tasks
        return SwarmTaskResult(
            id=task.id,
            subagent_type=subagent_type,
            status="failed",
            error=str(exc) if str(exc) else type(exc).__name__,
        )

    if not isinstance(result, dict):
        # Runnables sometimes hand back non-dict shapes (e.g. Pydantic
        # models). Coerce defensively so the extractor can still do its
        # job; anything unparseable falls through to "Task completed".
        result = getattr(result, "__dict__", {}) or {}

    return SwarmTaskResult(
        id=task.id,
        subagent_type=subagent_type,
        status="completed",
        result=_extract_result_text(result),
    )


def _validate_subagent_types(
    tasks: list[SwarmTaskSpec], subagent_graphs: Mapping[str, Runnable]
) -> None:
    unknown: set[str] = set()
    for task in tasks:
        subagent_type = task.subagent_type or "general-purpose"
        if subagent_type not in subagent_graphs:
            unknown.add(subagent_type)
    if unknown:
        available = ", ".join(subagent_graphs.keys())
        unknown_str = ", ".join(sorted(unknown))
        msg = f"Unknown subagent type(s): {unknown_str}. Available: {available}"
        raise ValueError(msg)


def _validate_response_schema(task_id: str, schema: dict[str, Any]) -> None:
    """Enforce top-level ``type: object`` with non-empty ``properties``.

    Open schemas (no explicit properties) are rejected — they aren't
    supported reliably by structured-output runtimes across providers.
    """
    if schema.get("type") != "object":
        msg = (
            'responseSchema must have type "object" at the top level. '
            'Wrap array schemas in an object. Invalid task: '
            f'"{task_id}" has type "{schema.get("type")}".'
        )
        raise ValueError(msg)
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        msg = (
            'responseSchema must define "properties" with at least one field. '
            f'Invalid task: "{task_id}".'
        )
        raise ValueError(msg)


def _schema_cache_key(subagent_type: str, schema: dict[str, Any]) -> str:
    """Stable cache key for a (subagent_type, schema) pair.

    The ``::`` separator guarantees no collision with the plain type
    names used as default-graph keys.
    """
    digest = hashlib.sha256(
        json.dumps(schema, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"{subagent_type}::{digest}"


def _build_subagent_resolver(
    subagent_graphs: Mapping[str, Runnable],
    subagent_factories: Mapping[str, SubagentFactory] | None,
) -> Callable[[str, dict[str, Any] | None], Runnable]:
    """Build a resolver that maps (subagent_type, response_schema?) to a graph.

    Default (no-schema) graphs are pre-seeded from ``subagent_graphs``.
    When a ``response_schema`` is provided, the resolver compiles a
    variant via the corresponding :class:`SubagentFactory` and caches it
    under a ``"type::sha256hash"`` key so identical schemas are compiled
    only once per swarm call.
    """
    cache: dict[str, Runnable] = dict(subagent_graphs)
    factories = subagent_factories or {}

    def resolve(
        subagent_type: str, response_schema: dict[str, Any] | None
    ) -> Runnable:
        if not response_schema:
            return cache[subagent_type]
        key = _schema_cache_key(subagent_type, response_schema)
        cached = cache.get(key)
        if cached is not None:
            return cached
        factory = factories.get(subagent_type)
        if factory is None:
            # No factory available — fall back to the default graph.
            return cache[subagent_type]
        variant = factory(response_schema)
        cache[key] = variant
        return variant

    return resolve


ReadCallback = Callable[[str], Awaitable[str]]
"""Async callback that returns file contents, checking the pending-writes
buffer before the backend so a ``swarm.create`` in the same eval is
visible to the subsequent ``swarm.execute``."""


@dataclass
class SwarmExecutionOptions:
    """Everything the executor needs to run a swarm against a table."""

    file: str
    """Path to the JSONL table to dispatch against."""

    instruction: str
    """Template with ``{column}`` placeholders, interpolated per row."""

    subagent_graphs: Mapping[str, Runnable]
    """Default compiled subagent graphs keyed by type name."""

    read: ReadCallback
    """Read-through callback: pending buffer first, then backend."""

    write: WriteCallback
    """Write callback that streams updated rows back to the table."""

    column: str = "result"
    filter: SwarmFilter | None = None
    subagent_type: str | None = None
    response_schema: dict[str, Any] | None = None
    concurrency: int | None = None
    context: str | None = None
    """Prose prepended to every subagent prompt. See ``SwarmExecuteOptions.context``."""
    current_state: dict[str, Any] = field(default_factory=dict)
    subagent_factories: Mapping[str, SubagentFactory] | None = None
    task_timeout_seconds: float = TASK_TIMEOUT_SECONDS
    cancel_event: asyncio.Event | None = None


@dataclass
class _PreparedSwarm:
    tasks: list[SwarmTaskSpec]
    row_index_by_id: dict[str, int]
    rows: list[dict[str, Any]]
    skipped: int
    interpolation_errors: list[dict[str, str]]


async def _prepare_swarm(
    read: ReadCallback,
    file: str,
    instruction: str,
    filter_clause: SwarmFilter | None,
    subagent_type: str | None,
    response_schema: dict[str, Any] | None,
) -> _PreparedSwarm:
    """Read the table, partition by filter, interpolate instructions.

    Rows that fail interpolation are recorded but not dispatched; matched
    rows become task specs. Non-matching rows pass through unchanged.
    """
    content = await read(file)
    rows = parse_table_jsonl(content)

    tasks: list[SwarmTaskSpec] = []
    row_index_by_id: dict[str, int] = {}
    interpolation_errors: list[dict[str, str]] = []
    skipped = 0

    for idx, row in enumerate(rows):
        if filter_clause is not None and not evaluate_filter(filter_clause, row):
            skipped += 1
            continue
        task_id = row["id"] if isinstance(row.get("id"), str) else f"row-{idx}"
        try:
            description = interpolate_instruction(instruction, row)
        except ValueError as exc:
            interpolation_errors.append({"id": task_id, "error": str(exc)})
            continue
        tasks.append(
            SwarmTaskSpec(
                id=task_id,
                description=description,
                subagent_type=subagent_type,
                response_schema=response_schema,
            )
        )
        row_index_by_id[task_id] = idx

    return _PreparedSwarm(
        tasks=tasks,
        row_index_by_id=row_index_by_id,
        rows=rows,
        skipped=skipped,
        interpolation_errors=interpolation_errors,
    )


def _try_parse_json(value: str) -> Any:
    """Parse JSON if possible, return the raw string otherwise."""
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _merge_result_into_row(
    row: dict[str, Any], column: str, value: Any
) -> dict[str, Any]:
    """Write ``value`` onto ``row``. Structured objects (from a
    ``response_schema``) are spread — each property becomes a top-level
    column on the row. Plain text and arrays land under ``column``.
    """
    if isinstance(value, dict):
        return {**row, **value}
    return {**row, column: value}


def _build_summary(
    results: list[SwarmTaskResult],
    interpolation_errors: list[dict[str, str]],
    file: str,
    column: str,
    skipped: int,
) -> SwarmSummary:
    completed = sum(1 for r in results if r.status == "completed")
    dispatch_failed = sum(1 for r in results if r.status == "failed")
    entries = [
        SwarmResultEntry(
            id=r.id,
            subagent_type=r.subagent_type,
            status=r.status,
            result=r.result,
            error=r.error,
        )
        for r in results
    ]
    failed_tasks: list[dict[str, str]] = [
        {"id": r.id, "error": r.error or ""} for r in results if r.status == "failed"
    ]
    failed_tasks.extend(
        {"id": entry["id"], "error": f"Interpolation: {entry['error']}"}
        for entry in interpolation_errors
    )
    return SwarmSummary(
        total=len(results),
        completed=completed,
        failed=dispatch_failed + len(interpolation_errors),
        skipped=skipped,
        file=file,
        column=column,
        results=entries,
        failed_tasks=failed_tasks,
    )


async def execute_swarm(options: SwarmExecutionOptions) -> SwarmSummary:
    """Execute a swarm against a JSONL table.

    Reads the table, partitions rows by ``filter``, interpolates the
    instruction template per matched row, dispatches subagents with
    bounded concurrency, and streams each result back as a column on
    the source row via ``options.write``.

    Raises:
        ValueError: when a task references an unknown subagent type,
            when ``response_schema`` is malformed, or when the table
            cannot be read / parsed.
    """
    prepared = await _prepare_swarm(
        options.read,
        options.file,
        options.instruction,
        options.filter,
        options.subagent_type,
        options.response_schema,
    )

    effective_concurrency = min(
        options.concurrency if options.concurrency is not None else DEFAULT_CONCURRENCY,
        MAX_CONCURRENCY,
    )
    if effective_concurrency < 1:
        effective_concurrency = 1

    _validate_subagent_types(prepared.tasks, options.subagent_graphs)
    for task in prepared.tasks:
        if task.response_schema is not None:
            _validate_response_schema(task.id, task.response_schema)

    resolve = _build_subagent_resolver(options.subagent_graphs, options.subagent_factories)
    filtered_state = _filter_state_for_subagent(options.current_state)

    semaphore = asyncio.Semaphore(effective_concurrency)
    rows = prepared.rows
    write_lock = asyncio.Lock()

    async def _run(task: SwarmTaskSpec) -> SwarmTaskResult:
        async with semaphore:
            subagent_type = task.subagent_type or "general-purpose"
            if options.cancel_event is not None and options.cancel_event.is_set():
                return SwarmTaskResult(
                    id=task.id,
                    subagent_type=subagent_type,
                    status="failed",
                    error="Aborted",
                )
            subagent = resolve(subagent_type, task.response_schema)
            result = await _dispatch_task(
                task,
                subagent,
                filtered_state,
                options.task_timeout_seconds,
                options.cancel_event,
                options.context,
            )
            # Merge the result into the row and re-serialise the full
            # table. The lock keeps concurrent completions from
            # interleaving write calls (rows is mutable, shared).
            if result.status == "completed" and result.result is not None:
                row_idx = prepared.row_index_by_id.get(result.id)
                if row_idx is not None:
                    async with write_lock:
                        value: Any = (
                            _try_parse_json(result.result)
                            if options.response_schema is not None
                            else result.result
                        )
                        rows[row_idx] = _merge_result_into_row(
                            rows[row_idx], options.column, value
                        )
                        options.write(options.file, serialize_table_jsonl(rows))
            return result

    results = await asyncio.gather(*[_run(t) for t in prepared.tasks])

    return _build_summary(
        list(results),
        prepared.interpolation_errors,
        options.file,
        options.column,
        prepared.skipped,
    )
