"""Parse and serialize tasks.jsonl / results.jsonl.

Ported from ``libs/deepagents/src/swarm/parse.ts``.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

from deepagents_repl._swarm.types import SwarmTaskSpec

if TYPE_CHECKING:
    from collections.abc import Iterable

    from deepagents_repl._swarm.types import SwarmTaskResult


def parse_tasks_jsonl(content: str) -> list[SwarmTaskSpec]:
    """Parse and validate a tasks.jsonl string into a list of task specs.

    Validates each line is valid JSON, each task has a non-empty ``id``
    and ``description``, all task IDs are unique, and at least one task
    is present. Accumulates errors line-by-line so the caller sees every
    issue before raising.
    """
    lines = [line for line in content.split("\n") if line.strip() != ""]
    if not lines:
        msg = "tasks.jsonl is empty. The generation script must write at least one task."
        raise ValueError(msg)

    tasks: list[SwarmTaskSpec] = []
    seen_ids: set[str] = set()
    errors: list[str] = []

    for idx, raw in enumerate(lines, start=1):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            errors.append(f"Line {idx}: invalid JSON")
            continue

        task, validation_errors = _validate_task(parsed)
        if validation_errors:
            errors.append(f"Line {idx}: {', '.join(validation_errors)}")
            continue

        assert task is not None  # noqa: S101 — refined above
        if task.id in seen_ids:
            errors.append(f'Line {idx}: duplicate task id "{task.id}"')
            continue

        seen_ids.add(task.id)
        tasks.append(task)

    if errors:
        joined = "\n".join(errors)
        msg = f"tasks.jsonl validation failed:\n{joined}"
        raise ValueError(msg)

    return tasks


def _validate_task(parsed: Any) -> tuple[SwarmTaskSpec | None, list[str]]:
    """Validate one parsed JSON row. Returns (spec, errors) — never both truthy.

    Mirrors the zod schema in the JS port: strips unknown keys, rejects
    non-string / empty required fields, allows optional ``subagentType``.
    """
    errors: list[str] = []
    if not isinstance(parsed, dict):
        return None, ["Expected object"]

    raw_id = parsed.get("id")
    if not isinstance(raw_id, str) or not raw_id:
        errors.append("id must be a non-empty string")
    raw_description = parsed.get("description")
    if not isinstance(raw_description, str) or not raw_description:
        errors.append("description must be a non-empty string")
    raw_subagent = parsed.get("subagentType")
    if raw_subagent is not None and not isinstance(raw_subagent, str):
        errors.append("subagentType must be a string when provided")

    if errors:
        return None, errors

    return (
        SwarmTaskSpec(
            id=raw_id,  # type: ignore[arg-type]
            description=raw_description,  # type: ignore[arg-type]
            subagent_type=raw_subagent,
        ),
        [],
    )


def serialize_tasks_jsonl(tasks: Iterable[SwarmTaskSpec]) -> str:
    """Serialize a list of task specs to JSONL with a trailing newline.

    Wire format uses the JS-style ``subagentType`` key so JSONL produced
    by Python and JS round-trips across both runtimes.
    """
    lines = [json.dumps(_spec_to_wire(t), separators=(",", ":")) for t in tasks]
    return "\n".join(lines) + "\n"


def serialize_results_jsonl(results: Iterable[SwarmTaskResult]) -> str:
    """Serialize results to JSONL with a trailing newline."""
    lines = [json.dumps(_result_to_wire(r), separators=(",", ":")) for r in results]
    return "\n".join(lines) + "\n"


def _spec_to_wire(spec: SwarmTaskSpec) -> dict[str, Any]:
    out: dict[str, Any] = {"id": spec.id, "description": spec.description}
    if spec.subagent_type is not None:
        out["subagentType"] = spec.subagent_type
    return out


def _result_to_wire(result: SwarmTaskResult) -> dict[str, Any]:
    # Use JS-side key names on the wire so JSONL files are portable.
    payload: dict[str, Any] = {
        "id": result.id,
        "subagentType": result.subagent_type,
        "status": result.status,
    }
    for field in dataclasses.fields(result):
        if field.name in {"id", "subagent_type", "status"}:
            continue
        value = getattr(result, field.name)
        if value is not None:
            payload[field.name] = value
    return payload
