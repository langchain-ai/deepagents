"""Programmatic tool calling (PTC) support for ``REPLMiddleware``.

PTC exposes the agent's LangChain tools inside the JavaScript REPL as
``tools.<camelCaseName>(input)`` async functions. Instead of issuing
N serial tool calls, the model writes one ``eval`` that loops / parallelises
/ chains tools in-code:

    const [a, b] = await Promise.all([
        tools.search({query: "foo"}),
        tools.search({query: "bar"}),
    ]);

Two pieces live here:

- filtering — turn the live agent toolset into the subset exposed to PTC
- prompt rendering — render a short TS-ish API-reference block describing
  each exposed tool, so the model knows the call shape

The host-function bridge that actually invokes each tool lives in
``_repl.py`` next to the rest of the context wiring.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from collections.abc import Sequence


PTCOption = list[str | BaseTool]


def filter_tools_for_ptc(
    tools: Sequence[BaseTool],
    config: PTCOption,
    *,
    self_tool_name: str,
) -> list[BaseTool]:
    """Return the subset of ``tools`` exposed inside the REPL.

    ``self_tool_name`` is the REPL's own tool name; it is *always* excluded
    to prevent the model from recursing ``tools.eval("tools.eval(...)")``.
    If the model wants a nested eval, it can just write nested code in one
    call — that's the whole point of PTC.

    ``config`` is allowlist-only:

    - ``str`` entries: expose matching tool names from ``tools``.
    - ``BaseTool`` entries: expose those tools directly (minus
      ``self_tool_name``).

    Mixed lists are supported and merged. Explicit ``BaseTool`` entries
    are included first, then name-matched agent tools are appended.
    Duplicate tool names are deduplicated.

    Warning:
        PTC tool calls execute through the REPL bridge and currently do
        not respect `interrupt_on` / HITL approval hooks for each
        individual tool invocation.
    """
    if isinstance(config, list):
        explicit_tools: list[BaseTool] = []
        allow_names: set[str] = set()
        for entry in config:
            if isinstance(entry, BaseTool):
                if entry.name != self_tool_name:
                    explicit_tools.append(entry)
                continue
            if isinstance(entry, str):
                allow_names.add(entry)
                continue
            msg = "ptc list entries must be str or BaseTool"
            raise TypeError(msg)
        selected = [
            *explicit_tools,
            *[
                t
                for t in tools
                if t.name != self_tool_name and t.name in allow_names
            ],
        ]
        deduped: list[BaseTool] = []
        seen_names: set[str] = set()
        for tool in selected:
            if tool.name in seen_names:
                continue
            seen_names.add(tool.name)
            deduped.append(tool)
        selected = deduped
        _raise_on_invalid_ptc_tools(selected)
        return selected
    msg = (
        "Unsupported `ptc` config type. "
        "Use a list of tool names, list of BaseTool instances, or disable PTC."
    )
    raise TypeError(msg)


_CAMEL_SEP = re.compile(r"[-_]([a-z])")
_JS_IDENTIFIER = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")


def to_camel_case(name: str) -> str:
    """Convert ``snake_case`` / ``kebab-case`` → ``camelCase``.

    Matches the TS package's convention so PTC-savvy users get the same
    identifier shape across Python and TS backends. ``my_tool`` → ``myTool``.
    """
    return _CAMEL_SEP.sub(lambda m: m.group(1).upper(), name)


def is_valid_js_identifier(name: str) -> bool:
    """Return whether `name` is a valid JavaScript identifier."""
    return _JS_IDENTIFIER.fullmatch(name) is not None


def is_valid_ptc_tool_name(name: str) -> bool:
    """Return whether a tool can be exposed as `tools.<camelCaseName>`."""
    return is_valid_js_identifier(to_camel_case(name))


def _raise_on_invalid_ptc_tools(tools: Sequence[BaseTool]) -> None:
    for tool in tools:
        camel = to_camel_case(tool.name)
        if is_valid_js_identifier(camel):
            continue
        msg = (
            f"PTC tool name {tool.name!r} cannot be exposed as JavaScript "
            f"identifier {camel!r}. Tool names must map to "
            "`/^[A-Za-z_$][A-Za-z0-9_$]*$/`."
        )
        raise ValueError(msg)


def render_ptc_prompt(tools: Sequence[BaseTool]) -> str:
    """Build the `tools` namespace section of the system prompt.

    One block per tool: name, description (first line), and a TS-ish
    signature derived from the Pydantic args_schema. Falls back to
    ``input: Record<string, unknown>`` when a tool has no schema or the
    schema can't be walked.
    """
    if not tools:
        return ""
    _raise_on_invalid_ptc_tools(tools)
    blocks: list[str] = []
    for t in tools:
        camel = to_camel_case(t.name)
        schema = _safe_json_schema(t)
        signature = _render_signature(camel, schema)
        description = (
            (t.description or "").strip().splitlines()[0] if t.description else ""
        )
        blocks.append(f"/** {description} */\n{signature}")
    body = "\n\n".join(blocks)
    return (
        "\n\n"
        "### API Reference — `tools` namespace\n\n"
        "The agent tools listed below are callable as async functions inside the REPL "
        "under the `tools` namespace. Each takes a single object argument and returns "
        "a Promise that resolves to a string. Use `await`; combine with `Promise.all` "
        "for concurrent calls.\n\n"
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
        return f"async tools.{fn_name}(input: Record<string, unknown>): Promise<string>"
    props: dict[str, Any] = schema["properties"]
    required = set(schema.get("required", []))
    # Keep the interface inline (same arg) rather than defining a named
    # interface — fewer lines in the prompt, and JS-side naming doesn't
    # matter since the model is just reading the shape.
    fields = []
    for key, prop in props.items():
        optional = "" if key in required else "?"
        type_str = _json_schema_to_ts(prop)
        desc = prop.get("description")
        prefix = f"/**\n *{desc}\n */ " if desc else ""
        fields.append(f"  {prefix}{key}{optional}: {type_str};")
    body = "\n".join(fields) if fields else ""
    return (
        f"async tools.{fn_name}(input: {{\n{body}\n}}): Promise<string>"
        if body
        else f"async tools.{fn_name}(input: Record<string, unknown>): Promise<string>"
    )


def _json_schema_to_ts(prop: dict[str, Any]) -> str:
    """Shallow JSON-Schema → TS type renderer.

    Handles the cases Pydantic v2 actually emits for simple tool schemas:
    primitives, arrays, nested objects, enums, unions via ``anyOf``.
    ``$ref`` and ``allOf`` fall back to ``unknown`` — the description is
    usually enough for the model to understand intent, and the call still
    works because we pass input through unchanged.
    """
    if "enum" in prop:
        return " | ".join(json.dumps(v) for v in prop["enum"])
    if "anyOf" in prop:
        parts = [_json_schema_to_ts(p) for p in prop["anyOf"]]
        return " | ".join(dict.fromkeys(parts))  # dedupe while preserving order
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
