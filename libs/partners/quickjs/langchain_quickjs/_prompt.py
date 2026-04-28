"""Prompt/rendering helpers for REPL and PTC system prompts."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tools import BaseTool


_CAMEL_SEP = re.compile(r"[-_]([a-z])")
_JS_IDENTIFIER = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_REPL_SYSTEM_PROMPT_TEMPLATE = (
    "### Interpreter\n\n"
    "An `{tool_name}` tool is available. It runs JavaScript in a persistent "
    "REPL.\n"
    "- State (variables, functions) persists across tool calls within a single turn of "
    "conversation. They DO NOT persist across multiple turns.\n"
    "- Top-level `await` works; Promises resolve before the call returns.\n"
    "- Sandboxed: no filesystem, no stdlib, no network, no real clock, "
    "no `fetch`, no `require`.\n"
    "- Timeout: {timeout}s per call. Memory: {memory_limit_mb} MB total.\n"
    "- `console.log` output is captured and returned alongside the result."
)


def render_repl_system_prompt(
    *,
    tool_name: str,
    timeout: float,
    memory_limit_mb: int,
) -> str:
    """Render the base REPL system prompt text for ``REPLMiddleware``."""
    return _REPL_SYSTEM_PROMPT_TEMPLATE.format(
        tool_name=tool_name,
        timeout=timeout,
        memory_limit_mb=memory_limit_mb,
    )


def to_camel_case(name: str) -> str:
    """Convert ``snake_case`` / ``kebab-case`` → ``camelCase``."""
    return _CAMEL_SEP.sub(lambda m: m.group(1).upper(), name)


def is_valid_js_identifier(name: str) -> bool:
    """Return whether `name` is a valid JavaScript identifier."""
    return _JS_IDENTIFIER.fullmatch(name) is not None


def is_valid_ptc_tool_name(name: str) -> bool:
    """Return whether a tool can be exposed as `tools.<camelCaseName>`."""
    return is_valid_js_identifier(to_camel_case(name))


def render_ptc_prompt(tools: Sequence[BaseTool]) -> str:
    """Build the `tools` namespace section of the system prompt."""
    if not tools:
        return ""
    blocks: list[str] = []
    for tool in tools:
        camel = to_camel_case(tool.name)
        schema = _safe_json_schema(tool)
        signature = _render_signature(camel, schema)
        description = (
            (tool.description or "").strip().splitlines()[0] if tool.description else ""
        )
        blocks.append(f"/** {description} */\n{signature}")
    body = "\n\n".join(blocks)
    return (
        "\n\n"
        "### API Reference — `tools` namespace\n\n"
        "The agent tools listed below are exposed on the global object at "
        "`globalThis.tools` (also reachable as `tools`). Each takes a single object "
        "argument and returns a Promise that resolves to a string.\n\n"
        "Invocation pattern: `await tools.<name>({ ... })`).\n\n"
        "Use `await`; combine with `Promise.all` for concurrent calls.\n\n"
        "```typescript\n"
        f"{body}\n"
        "```"
    )


def _safe_json_schema(tool: BaseTool) -> dict[str, Any] | None:
    try:
        if tool.args_schema is None:
            return None
        model_json_schema = getattr(tool.args_schema, "model_json_schema", None)
        if callable(model_json_schema):
            return model_json_schema()
    except Exception:  # noqa: BLE001 — prompt rendering is best-effort
        return None
    return None


def _render_signature(fn_name: str, schema: dict[str, Any] | None) -> str:
    if not schema or not isinstance(schema.get("properties"), dict):
        return (
            f"async function {fn_name}(input: Record<string, unknown>): Promise<string>"
        )
    props: dict[str, Any] = schema["properties"]
    required = set(schema.get("required", []))
    fields = []
    for key, prop in props.items():
        optional = "" if key in required else "?"
        type_str = _json_schema_to_ts(prop)
        desc = prop.get("description")
        prefix = f"/**\n *{desc}\n */ " if desc else ""
        fields.append(f"  {prefix}{key}{optional}: {type_str};")
    body = "\n".join(fields) if fields else ""
    return (
        f"async function {fn_name}(input: {{\n{body}\n}}): Promise<string>"
        if body
        else f"async function {fn_name}(input: Record<string, unknown>): Promise<string>"
    )


def _json_schema_to_ts(prop: dict[str, Any]) -> str:
    """Shallow JSON-Schema → TS type renderer."""
    if "enum" in prop:
        return " | ".join(json.dumps(v) for v in prop["enum"])
    if "anyOf" in prop:
        parts = [_json_schema_to_ts(part) for part in prop["anyOf"]]
        return " | ".join(dict.fromkeys(parts))
    t = prop.get("type")
    if t == "string":
        return "string"
    if t in {"integer", "number"}:
        return "number"
    if t == "boolean":
        return "boolean"
    if t == "null":
        return "null"
    if t == "array":
        items = prop.get("items")
        inner = _json_schema_to_ts(items) if isinstance(items, dict) else "unknown"
        return f"{inner}[]"
    if t == "object":
        sub_props = prop.get("properties")
        if isinstance(sub_props, dict) and sub_props:
            required = set(prop.get("required", []))
            fields = [
                f"{k}{'' if k in required else '?'}: {_json_schema_to_ts(v)}"
                for k, v in sub_props.items()
            ]
            return "{ " + "; ".join(fields) + " }"
        return "Record<string, unknown>"
    return "unknown"
