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
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tools import BaseTool


class PTCConfig(TypedDict, total=False):
    """Filter for which agent tools are exposed inside the REPL.

    Use ``include`` **or** ``exclude``, not both. ``exclude`` is always
    additive with PTC's own hard-coded exclusions (the REPL's own eval
    tool, for example) — users cannot force the REPL tool into its own
    namespace.
    """

    include: list[str]
    exclude: list[str]


PTCOption = bool | list[str] | PTCConfig


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
    """
    if config is False:
        return []
    candidates = [t for t in tools if t.name != self_tool_name]
    if config is True:
        return candidates
    if isinstance(config, list):
        allow = set(config)
        return [t for t in candidates if t.name in allow]
    if isinstance(config, dict):
        if "include" in config and "exclude" in config:
            msg = "ptc config cannot specify both include and exclude"
            raise ValueError(msg)
        if "include" in config:
            allow = set(config["include"])
            return [t for t in candidates if t.name in allow]
        if "exclude" in config:
            deny = set(config["exclude"])
            return [t for t in candidates if t.name not in deny]
    return []


_CAMEL_SEP = re.compile(r"[-_]([a-z])")


def to_camel_case(name: str) -> str:
    """Convert ``snake_case`` / ``kebab-case`` → ``camelCase``.

    Matches the TS package's convention so PTC-savvy users get the same
    identifier shape across Python and TS backends. ``my_tool`` → ``myTool``.
    """
    return _CAMEL_SEP.sub(lambda m: m.group(1).upper(), name)


def render_ptc_prompt(tools: Sequence[BaseTool]) -> str:
    """Build the `tools` namespace section of the system prompt.

    One block per tool: name, description (first line), and a TS-ish
    signature derived from the Pydantic args_schema. Falls back to
    ``input: Record<string, unknown>`` when a tool has no schema or the
    schema can't be walked.
    """
    if not tools:
        return ""
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
        if hasattr(tool.args_schema, "model_json_schema"):
            return tool.args_schema.model_json_schema()
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
        prefix = f"/** {desc} */ " if desc else ""
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
    if t == "integer" or t == "number":
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
