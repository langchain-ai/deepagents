"""QuickJS adapter for the Deep Agents `task` subagent tool."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Final, Literal, NotRequired, TypedDict

from langchain.agents.structured_output import AutoStrategy

from langchain_quickjs._format import coerce_tool_output_for_ptc

try:
    from deepagents.middleware.subagents import SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY
except ImportError:  # pragma: no cover - compatibility with older deepagents
    SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY = "__deepagents_subagent_response_format"


if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

SUBAGENT_STREAM_EVENT_TYPE: Final = "subagent"
"""Discriminator value for subagent events on the custom stream."""

_SCHEMA_MAX_BYTES = 4096
"""Maximum serialized size of an accepted `response_schema`."""

_SCHEMA_MAX_DEPTH = 5
"""Maximum nesting depth allowed in a `response_schema`."""

_SCHEMA_MAX_PROPERTIES = 32
"""Maximum total property count allowed across a `response_schema`."""

_SUBAGENT_TASK_TOOL_FIELDS = frozenset({"description", "subagent_type"})
"""Input field names that identify the Deep Agents task tool."""

_EVENT_DESCRIPTION_MAX_CHARS = 200
"""Character cap on the `description` carried in a start event."""

_EVENT_LABEL_MAX_CHARS = 120
"""Character cap on an explicit `label` carried in a start event."""

_EVENT_LABEL_FALLBACK_MAX_CHARS = 60
"""Character cap on a label derived from the description fallback."""


class SubagentStartEvent(TypedDict):
    """A subagent began running inside a `js_eval` call."""

    id: str
    """Per-dispatch id, stable across this subagent's start/complete/error."""

    type: Literal["subagent"]
    """Stream event discriminator; always `subagent`."""

    phase: Literal["start"]
    """Lifecycle phase for this event."""

    eval_id: NotRequired[str]
    """Parent `js_eval` tool-call id, used to group a fan-out by batch.

    Omitted when the runtime exposes no `tool_call_id`.
    """

    subagent_type: str
    """The dispatched subagent type (the `subagentType` argument)."""

    label: str
    """Short row label; falls back to a compact description when unset."""

    description: str
    """The task description, truncated for display."""


class SubagentCompleteEvent(TypedDict):
    """A subagent finished successfully inside a `js_eval` call."""

    id: str
    """Per-dispatch id, matching the corresponding `start` event."""

    type: Literal["subagent"]
    """Stream event discriminator; always `subagent`."""

    phase: Literal["complete"]
    """Lifecycle phase for this event."""

    eval_id: NotRequired[str]
    """Parent `js_eval` tool-call id; omitted when the runtime exposes none."""

    duration_ms: int
    """Wall-clock duration of the subagent, in milliseconds."""


class SubagentErrorEvent(TypedDict):
    """A subagent raised before returning inside a `js_eval` call."""

    id: str
    """Per-dispatch id, matching the corresponding `start` event."""

    type: Literal["subagent"]
    """Stream event discriminator; always `subagent`."""

    phase: Literal["error"]
    """Lifecycle phase for this event."""

    eval_id: NotRequired[str]
    """Parent `js_eval` tool-call id; omitted when the runtime exposes none."""

    duration_ms: int
    """Wall-clock duration before the failure, in milliseconds."""

    error: str
    """The failure string (`str(exc)` of the raised exception)."""


SubagentStreamEvent = SubagentStartEvent | SubagentCompleteEvent | SubagentErrorEvent
"""One lifecycle event for a subagent dispatched from inside `js_eval`.

Emitted on LangGraph's `custom` stream so UIs can render a live fan-out panel.
A `phase`-discriminated union: `start` carries the descriptive fields,
`complete`/`error` carry the measured `duration_ms`, and `error` carries the
failure string. `type`/`phase`/`id` are always present.

Consumers should tolerate unrecognized `phase` values rather than assume the
union is closed, so a future phase can be added without breaking them.
"""


def _emit_subagent_event(stream_writer: Any, event: SubagentStreamEvent) -> None:
    """Emit a subagent lifecycle event on the custom stream.

    Any failure is swallowed so observability never breaks dispatch.
    """
    if stream_writer is None:
        return
    try:
        stream_writer(event)
    except Exception:  # noqa: BLE001 — observability must not break dispatch
        # Use `.get` rather than subscripting: this handler must never raise,
        # regardless of how well-formed the event that reached it was.
        logger.debug(
            "Failed to emit subagent stream event (id=%s, phase=%s)",
            event.get("id"),
            event.get("phase"),
            exc_info=True,
        )


def _event_label(label: str | None, description: str) -> str:
    """Return the explicit label or a compact description fallback."""
    explicit = " ".join(label.split()) if label else ""
    if explicit:
        return explicit[:_EVENT_LABEL_MAX_CHARS]
    return " ".join(description.split())[:_EVENT_LABEL_FALLBACK_MAX_CHARS]


def find_subagent_task_tool(tools: Sequence[BaseTool]) -> BaseTool | None:
    """Return the Deep Agents task tool that backs top-level `task()`."""
    for tool in tools:
        if (
            getattr(tool, "name", None) == "task"
            and _tool_input_field_names(tool) >= _SUBAGENT_TASK_TOOL_FIELDS
        ):
            return tool
    return None


def _tool_input_field_names(tool: BaseTool) -> frozenset[str]:
    """Return input field names from a LangChain tool's public schema."""
    schema = getattr(tool, "args_schema", None)
    fields = getattr(schema, "model_fields", None)
    if isinstance(fields, dict):
        return frozenset(str(name) for name in fields)
    fields = getattr(schema, "__fields__", None)
    if isinstance(fields, dict):
        return frozenset(str(name) for name in fields)
    args = getattr(tool, "args", None)
    if isinstance(args, dict):
        return frozenset(str(name) for name in args)
    return frozenset()


async def call_subagent_task_tool(
    task_tool: BaseTool,
    *,
    description: str,
    subagent_type: str,
    response_schema: dict[str, Any] | None,
    runtime: Any,
    label: str | None = None,
) -> Any:
    """Call the Deep Agents task tool and return a JavaScript-friendly value.

    This also emits `start` then `complete`/`error` subagent lifecycle
    events on the custom stream.
    """
    if runtime is None:
        msg = "task() requires an active ToolRuntime"
        raise RuntimeError(msg)

    parse_json_output = response_schema is not None
    if response_schema is not None:
        _validate_response_schema(response_schema)
        response_schema = _ensure_schema_title(response_schema)
        runtime = _runtime_with_response_format(runtime, response_schema)

    eval_id = getattr(runtime, "tool_call_id", None)
    stream_writer = getattr(runtime, "stream_writer", None)
    subagent_id = f"ptc_{task_tool.name}_{uuid.uuid4().hex[:8]}"

    runtime = _runtime_with_tool_call_id(runtime, subagent_id)

    start_event: SubagentStartEvent = {
        "type": SUBAGENT_STREAM_EVENT_TYPE,
        "phase": "start",
        "id": subagent_id,
        "subagent_type": subagent_type,
        "label": _event_label(label, description),
        "description": description[:_EVENT_DESCRIPTION_MAX_CHARS],
    }
    # Only carry `eval_id` when the runtime exposes a parent tool-call id;
    # omitting it (rather than sending None) keeps the wire type tight and lets
    # consumers distinguish "no parent batch" from a real id.
    if eval_id is not None:
        start_event["eval_id"] = eval_id
    _emit_subagent_event(stream_writer, start_event)

    started_at = time.monotonic()
    try:
        result = await task_tool.arun(
            {
                "description": description,
                "subagent_type": subagent_type,
                "runtime": runtime,
            }
        )
    except Exception as e:
        error_event: SubagentErrorEvent = {
            "type": SUBAGENT_STREAM_EVENT_TYPE,
            "phase": "error",
            "id": subagent_id,
            "duration_ms": int((time.monotonic() - started_at) * 1000),
            "error": str(e),
        }
        if eval_id is not None:
            error_event["eval_id"] = eval_id
        _emit_subagent_event(stream_writer, error_event)
        raise

    output = _extract_task_tool_output(result, parse_json_output=parse_json_output)
    complete_event: SubagentCompleteEvent = {
        "type": SUBAGENT_STREAM_EVENT_TYPE,
        "phase": "complete",
        "id": subagent_id,
        "duration_ms": int((time.monotonic() - started_at) * 1000),
    }
    if eval_id is not None:
        complete_event["eval_id"] = eval_id
    _emit_subagent_event(stream_writer, complete_event)
    return output


def _validate_response_schema(schema: dict[str, Any]) -> None:
    """Reject schemas that exceed size, depth, or property-count limits."""
    serialized = json.dumps(schema)
    if len(serialized) > _SCHEMA_MAX_BYTES:
        msg = (
            f"response_schema exceeds {_SCHEMA_MAX_BYTES}"
            f" byte limit ({len(serialized)} bytes)"
        )
        raise ValueError(msg)

    def _check(node: dict[str, Any], depth: int, prop_count: list[int]) -> None:
        if depth > _SCHEMA_MAX_DEPTH:
            msg = (
                f"response_schema exceeds maximum nesting depth of {_SCHEMA_MAX_DEPTH}"
            )
            raise ValueError(msg)
        props = node.get("properties")
        if isinstance(props, dict):
            prop_count[0] += len(props)
            if prop_count[0] > _SCHEMA_MAX_PROPERTIES:
                msg = (
                    "response_schema exceeds maximum of"
                    f" {_SCHEMA_MAX_PROPERTIES} properties"
                )
                raise ValueError(msg)
            for value in props.values():
                if isinstance(value, dict):
                    _check(value, depth + 1, prop_count)
        items = node.get("items")
        if isinstance(items, dict):
            _check(items, depth + 1, prop_count)

    _check(schema, 0, [0])


_DEFAULT_SCHEMA_TITLE = "subagent_response"


def _ensure_schema_title(schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure the response schema carries a non-empty top-level ``title``.

    Structured output backends that treat a JSON schema as a function (for
    example, the OpenAI function-calling path) require a top-level ``title`` to
    use as the function name. Agent-generated ``response_schema`` values often
    omit it, so inject a default when it is missing or blank.
    """
    existing = schema.get("title")
    if isinstance(existing, str) and existing.strip():
        return schema
    return {**schema, "title": _DEFAULT_SCHEMA_TITLE}


def _runtime_with_response_format(
    runtime: Any,
    response_schema: dict[str, Any],
) -> Any:
    """Return a per-dispatch runtime carrying response format in configurable."""
    config = getattr(runtime, "config", None)
    updated_config = dict(config) if isinstance(config, dict) else {}
    configurable = updated_config.get("configurable")
    if not isinstance(configurable, dict):
        configurable = {}
    updated_config["configurable"] = {
        **configurable,
        SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY: AutoStrategy(response_schema),
    }
    return replace(runtime, config=updated_config)


def _runtime_with_tool_call_id(runtime: Any, tool_call_id: str) -> Any:
    """Return a per-dispatch runtime for the nested task tool call."""
    return replace(runtime, tool_call_id=tool_call_id)


def _extract_task_tool_output(result: Any, *, parse_json_output: bool) -> Any:
    output = coerce_tool_output_for_ptc(result)
    if not parse_json_output or not isinstance(output, str):
        return output
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return output
