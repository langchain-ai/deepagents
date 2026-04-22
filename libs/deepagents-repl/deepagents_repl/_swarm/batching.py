"""Batched subagent dispatch for swarm execution.

When ``batch_size > 1`` rows are grouped into a single subagent call.
The subagent receives a combined prompt and a wrapped array schema;
results are unpacked per-row and merged back into the table once per
batch.

Kept separate from the single-row dispatch path in ``executor.py`` so
both paths stay independently readable.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage

from deepagents_repl._swarm.parse import serialize_table_jsonl
from deepagents_repl._swarm.types import (
    SwarmTaskResult,
    SwarmTaskSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.runnables import Runnable


def _as_object(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return value


def _is_pre_wrapped_batch_schema(schema: dict[str, Any]) -> bool:
    """Detect a schema that already has the ``{results: [{id, ...}]}`` shape.

    When the orchestrator hand-authors the batch envelope (e.g. to add
    descriptions on the wrapper), we preserve their shape and only stamp
    the per-batch ``minItems``/``maxItems`` on dispatch.
    """
    props = _as_object(schema.get("properties"))
    if props is None:
        return False
    results = _as_object(props.get("results"))
    if results is None or results.get("type") != "array":
        return False
    items = _as_object(results.get("items"))
    if items is None or items.get("type") != "object":
        return False
    item_props = _as_object(items.get("properties"))
    if item_props is None:
        return False
    return "id" in item_props


def _enforce_batch_count(schema: dict[str, Any], count: int) -> dict[str, Any]:
    """Stamp ``minItems``/``maxItems`` on ``results`` without disturbing
    anything else the orchestrator authored."""
    props = _as_object(schema.get("properties"))
    if props is None:
        return schema
    results = _as_object(props.get("results"))
    if results is None:
        return schema
    return {
        **schema,
        "properties": {
            **props,
            "results": {**results, "minItems": count, "maxItems": count},
        },
    }


def _wrap_batch_schema(item_schema: dict[str, Any], count: int) -> dict[str, Any]:
    """Wrap a per-item ``response_schema`` into a batch envelope.

    Pre-wrapped schemas (orchestrator-authored) are passed through with
    only the count constraints stamped on.
    """
    if _is_pre_wrapped_batch_schema(item_schema):
        return _enforce_batch_count(item_schema, count)

    user_props = item_schema.get("properties") or {}
    user_required = item_schema.get("required") or []
    if not isinstance(user_required, list):
        user_required = []

    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        **user_props,
                    },
                    "required": ["id", *user_required],
                },
                "minItems": count,
                "maxItems": count,
            },
        },
        "required": ["results"],
    }


def _group_into_batches(items: list[Any], size: int) -> list[list[Any]]:
    """Partition ``items`` into chunks of ``size``. Last chunk may be smaller."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def _compose_batch_instruction(
    batch: list[SwarmTaskSpec], context: str | None
) -> str:
    """Build a single combined prompt listing each task with its id."""
    # Local import to avoid a cycle with executor.py.
    from deepagents_repl._swarm.executor import _prepend_context  # noqa: PLC0415

    items = "\n".join(f"[{task.id}] {task.description}" for task in batch)
    body = (
        f'Process {len(batch)} items. Return a JSON "results" array '
        f"with exactly {len(batch)} entries. Each entry must include "
        f"the item's id exactly as shown.\n\n"
        f"Items:\n{items}"
    )
    return _prepend_context(body, context)


def _unpack_batch_result(
    batch: list[SwarmTaskSpec], raw_results: list[dict[str, Any]]
) -> list[SwarmTaskResult]:
    """Map the subagent's response array back to per-task results.

    Items are matched by ``id``; tasks whose id is missing from the
    response are marked failed.
    """
    subagent_type = (batch[0].subagent_type if batch else None) or "general-purpose"
    by_id: dict[str, dict[str, Any]] = {}
    for item in raw_results:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            by_id[item["id"]] = item

    out: list[SwarmTaskResult] = []
    for task in batch:
        match = by_id.get(task.id)
        if match is None:
            out.append(
                SwarmTaskResult(
                    id=task.id,
                    subagent_type=subagent_type,
                    status="failed",
                    error=f'No result returned for id "{task.id}"',
                )
            )
            continue
        rest = {k: v for k, v in match.items() if k != "id"}
        out.append(
            SwarmTaskResult(
                id=task.id,
                subagent_type=subagent_type,
                status="completed",
                result=json.dumps(rest, default=str),
            )
        )
    return out


async def _dispatch_batch(
    batch: list[SwarmTaskSpec],
    subagent: Runnable,
    filtered_state: dict[str, Any],
    context: str | None,
    task_timeout_seconds: float,
) -> list[SwarmTaskResult]:
    """Dispatch one batch as a single subagent call. Never raises —
    failures map to per-task failed results with a shared error."""
    # Local import to avoid a cycle with executor.py.
    from deepagents_repl._swarm.executor import _extract_result_text  # noqa: PLC0415

    subagent_type = (batch[0].subagent_type if batch else None) or "general-purpose"
    prompt = _compose_batch_instruction(batch, context)
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
        return [
            SwarmTaskResult(
                id=task.id,
                subagent_type=subagent_type,
                status="failed",
                error=f"Timed out after {task_timeout_seconds:g}s",
            )
            for task in batch
        ]
    except Exception as exc:  # noqa: BLE001 — surface as failed results
        msg = str(exc) or type(exc).__name__
        return [
            SwarmTaskResult(
                id=task.id,
                subagent_type=subagent_type,
                status="failed",
                error=msg,
            )
            for task in batch
        ]

    if not isinstance(result, dict):
        result = getattr(result, "__dict__", {}) or {}

    structured = result.get("structured_response") or result.get("structuredResponse")
    if isinstance(structured, dict):
        results_list = structured.get("results")
        if isinstance(results_list, list):
            return _unpack_batch_result(batch, results_list)

    # Fallback: parse JSON out of the final text content.
    from deepagents_repl._swarm.executor import _try_parse_json  # noqa: PLC0415

    parsed = _try_parse_json(_extract_result_text(result))
    if (
        isinstance(parsed, dict)
        and isinstance(parsed.get("results"), list)
    ):
        return _unpack_batch_result(batch, parsed["results"])

    return [
        SwarmTaskResult(
            id=task.id,
            subagent_type=subagent_type,
            status="failed",
            error="Could not parse batch response as results array",
        )
        for task in batch
    ]


async def dispatch_batched(
    *,
    tasks: list[SwarmTaskSpec],
    rows: list[dict[str, Any]],
    row_index_by_id: dict[str, int],
    column: str,
    file: str,
    write: Callable[[str, str], None],
    resolve_subagent: Callable[[str, dict[str, Any] | None], Runnable],
    filtered_state: dict[str, Any],
    context: str | None,
    response_schema: dict[str, Any],
    subagent_type: str | None,
    batch_size: int,
    concurrency: int,
    task_timeout_seconds: float,
    cancel_event: asyncio.Event | None,
) -> list[SwarmTaskResult]:
    """Run all tasks via batched subagent calls with bounded concurrency.

    Returns per-task results in dispatch order so the caller can build a
    summary the same way the single-row path does. Each completed batch
    streams its results back to the table via ``write``.
    """
    # Local import to avoid a cycle with executor.py.
    from deepagents_repl._swarm.executor import _merge_result_into_row  # noqa: PLC0415

    batches = _group_into_batches(tasks, batch_size)
    batch_subagent_type = subagent_type or "general-purpose"
    semaphore = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()
    results: list[SwarmTaskResult] = [None] * len(tasks)  # type: ignore[list-item]

    async def _run(offset: int, batch: list[SwarmTaskSpec]) -> None:
        async with semaphore:
            if cancel_event is not None and cancel_event.is_set():
                for i, task in enumerate(batch):
                    results[offset + i] = SwarmTaskResult(
                        id=task.id,
                        subagent_type=batch_subagent_type,
                        status="failed",
                        error="Aborted",
                    )
                return

            wrapped = _wrap_batch_schema(response_schema, len(batch))
            subagent = resolve_subagent(batch_subagent_type, wrapped)
            batch_results = await _dispatch_batch(
                batch,
                subagent,
                filtered_state,
                context,
                task_timeout_seconds,
            )

            async with write_lock:
                for i, res in enumerate(batch_results):
                    results[offset + i] = res
                    row_idx = row_index_by_id.get(res.id)
                    if (
                        res.status == "completed"
                        and res.result is not None
                        and row_idx is not None
                    ):
                        from deepagents_repl._swarm.executor import (  # noqa: PLC0415
                            _try_parse_json,
                        )

                        value = _try_parse_json(res.result)
                        rows[row_idx] = _merge_result_into_row(
                            rows[row_idx], column, value
                        )
                write(file, serialize_table_jsonl(rows))

    offset = 0
    coros = []
    for batch in batches:
        coros.append(_run(offset, batch))
        offset += len(batch)
    await asyncio.gather(*coros)
    return results
