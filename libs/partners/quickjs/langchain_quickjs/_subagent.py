"""QuickJS adapter for the Deep Agents `task` subagent tool."""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from langchain.agents.structured_output import AutoStrategy

from langchain_quickjs._format import coerce_tool_output_for_ptc

try:
    from deepagents.middleware.subagents import SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY
except ImportError:  # pragma: no cover - compatibility with older deepagents
    SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY = "__deepagents_subagent_response_format"


if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tools import BaseTool

_SCHEMA_MAX_BYTES = 4096
_SCHEMA_MAX_DEPTH = 5
_SCHEMA_MAX_PROPERTIES = 32
_SUBAGENT_TASK_TOOL_FIELDS = frozenset({"description", "subagent_type"})


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
) -> Any:
    """Call the Deep Agents task tool and return a JavaScript-friendly value."""
    if runtime is None:
        msg = "task() requires an active ToolRuntime"
        raise RuntimeError(msg)

    parse_json_output = response_schema is not None
    if response_schema is not None:
        _validate_response_schema(response_schema)
        runtime = _runtime_with_response_format(runtime, response_schema)

    runtime = _runtime_with_tool_call_id(
        runtime,
        f"ptc_{task_tool.name}_{uuid.uuid4().hex[:8]}",
    )
    result = await task_tool.arun(
        {
            "description": description,
            "subagent_type": subagent_type,
            "runtime": runtime,
        }
    )
    return _extract_task_tool_output(result, parse_json_output=parse_json_output)


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
