"""Parallel subagent dispatch.

Ported from ``libs/deepagents/src/swarm/executor.ts``.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage

from deepagents_repl._swarm.parse import serialize_results_jsonl
from deepagents_repl._swarm.types import (
    DEFAULT_CONCURRENCY,
    MAX_CONCURRENCY,
    TASK_TIMEOUT_SECONDS,
    FailedTaskInfo,
    SwarmExecutionSummary,
    SwarmTaskResult,
    SwarmTaskSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from deepagents.backends.protocol import BackendProtocol
    from langchain_core.runnables import Runnable


# State keys excluded when passing state to subagents.
# Mirrors ``_EXCLUDED_STATE_KEYS`` in ``deepagents/middleware/subagents.py``
# — we accept both the Python names (``structured_response``,
# ``skills_metadata``, ``memory_contents``) and the JS-style camelCase
# names (``structuredResponse``, ``skillsMetadata``, ``memoryContents``)
# so the exclusion works regardless of which naming the caller uses.
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


@dataclass
class SwarmExecutionOptions:
    """Everything the executor needs to run a swarm."""

    tasks: list[SwarmTaskSpec]
    subagent_graphs: Mapping[str, Runnable]
    backend: BackendProtocol
    current_state: dict[str, Any] = field(default_factory=dict)
    concurrency: int | None = None
    synthesized_tasks_jsonl: str | None = None
    task_timeout_seconds: float = TASK_TIMEOUT_SECONDS
    """Per-subagent-task wall-clock timeout."""
    cancel_event: asyncio.Event | None = None
    """When set, pending dispatches short-circuit and in-flight ainvoke calls are cancelled."""


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


async def _dispatch_task(
    task: SwarmTaskSpec,
    subagent: Runnable,
    filtered_state: dict[str, Any],
    task_timeout_seconds: float,
    cancel_event: asyncio.Event | None = None,
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
    subagent_state = {
        **filtered_state,
        "messages": [HumanMessage(content=task.description)],
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


async def execute_swarm(options: SwarmExecutionOptions) -> SwarmExecutionSummary:
    """Dispatch tasks to subagents in parallel, write results.jsonl, return summary.

    Raises:
        ValueError: when any task references an unknown subagent type.
    """
    resolved_concurrency = min(
        options.concurrency if options.concurrency is not None else DEFAULT_CONCURRENCY,
        MAX_CONCURRENCY,
    )
    if resolved_concurrency < 1:
        resolved_concurrency = 1

    unknown: set[str] = set()
    for task in options.tasks:
        subagent_type = task.subagent_type or "general-purpose"
        if subagent_type not in options.subagent_graphs:
            unknown.add(subagent_type)
    if unknown:
        available = ", ".join(options.subagent_graphs.keys())
        unknown_str = ", ".join(sorted(unknown))
        msg = f"Unknown subagent type(s): {unknown_str}. Available: {available}"
        raise ValueError(msg)

    filtered_state = _filter_state_for_subagent(options.current_state)

    semaphore = asyncio.Semaphore(resolved_concurrency)

    async def _run(task: SwarmTaskSpec) -> SwarmTaskResult:
        async with semaphore:
            subagent_type = task.subagent_type or "general-purpose"
            return await _dispatch_task(
                task,
                options.subagent_graphs[subagent_type],
                filtered_state,
                options.task_timeout_seconds,
                options.cancel_event,
            )

    results = await asyncio.gather(*[_run(task) for task in options.tasks])

    results_dir = f"/swarm_runs/{uuid.uuid4()}"
    await options.backend.awrite(
        f"{results_dir}/results.jsonl",
        serialize_results_jsonl(results),
    )
    if options.synthesized_tasks_jsonl:
        await options.backend.awrite(
            f"{results_dir}/tasks.jsonl",
            options.synthesized_tasks_jsonl,
        )

    completed = sum(1 for r in results if r.status == "completed")
    failed = sum(1 for r in results if r.status == "failed")
    failed_tasks = [
        FailedTaskInfo(id=r.id, error=r.error or "") for r in results if r.status == "failed"
    ]
    return SwarmExecutionSummary(
        total=len(options.tasks),
        completed=completed,
        failed=failed,
        results_dir=results_dir,
        results=list(results),
        failed_tasks=failed_tasks,
    )
