"""QuickJS adapter for the Deep Agents `task` subagent tool."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

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

SUBAGENT_STREAM_EVENT_TYPE = "subagent"

_SCHEMA_MAX_BYTES = 4096
_SCHEMA_MAX_DEPTH = 5
_SCHEMA_MAX_PROPERTIES = 32
_SUBAGENT_TASK_TOOL_FIELDS = frozenset({"description", "subagent_type"})
_EVENT_DESCRIPTION_MAX_CHARS = 200
_EVENT_LABEL_MAX_CHARS = 120


class SubagentStreamEvent(TypedDict):
    """One lifecycle event for a subagent dispatched from inside `js_eval`.

    Emitted on LangGraph's ``custom`` stream so UIs can render a live fan-out
    panel. ``type``/``phase``/``id`` are always present; the remaining fields
    depend on ``phase`` (start vs complete vs error).
    """

    id: str
    type: Literal["subagent"]
    phase: Literal["start", "complete", "error"]
    eval_id: NotRequired[str | None]
    subagent_type: NotRequired[str]
    label: NotRequired[str]
    description: NotRequired[str]
    duration_ms: NotRequired[int]
    error: NotRequired[str]


def _emit_subagent_event(stream_writer: Any, payload: dict[str, Any]) -> None:
    """Emits a subagent lifecycle event on the custom stream.

    Any failure is swallowed.
    """
    if stream_writer is None:
        return
    try:
        stream_writer(
            {
                "type": SUBAGENT_STREAM_EVENT_TYPE,
                **payload,
            }
        )
    except Exception:  # noqa: BLE001 — observability must not break dispatch
        logger.debug("Failed to emit subagent stream event", exc_info=True)


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
    label: str,
    response_schema: dict[str, Any] | None,
    runtime: Any,
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

    _emit_subagent_event(
        stream_writer,
        {
            "phase": "start",
            "id": subagent_id,
            "eval_id": eval_id,
            "subagent_type": subagent_type,
            "label": label[:_EVENT_LABEL_MAX_CHARS],
            "description": description[:_EVENT_DESCRIPTION_MAX_CHARS],
        },
    )
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
        _emit_subagent_event(
            stream_writer,
            {
                "phase": "error",
                "id": subagent_id,
                "eval_id": eval_id,
                "duration_ms": int((time.monotonic() - started_at) * 1000),
                "error": str(e),
            },
        )
        raise

    output = _extract_task_tool_output(result, parse_json_output=parse_json_output)
    _emit_subagent_event(
        stream_writer,
        {
            "phase": "complete",
            "id": subagent_id,
            "eval_id": eval_id,
            "duration_ms": int((time.monotonic() - started_at) * 1000),
        },
    )
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
