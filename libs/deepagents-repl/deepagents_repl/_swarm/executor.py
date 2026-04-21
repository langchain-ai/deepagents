"""Parallel subagent dispatch — helpers shared by the table-oriented executor."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage

from deepagents_repl._swarm.types import SwarmTaskResult, SwarmTaskSpec

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
