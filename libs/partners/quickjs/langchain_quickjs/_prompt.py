"""Prompt/rendering helpers for REPL and PTC system prompts."""

from __future__ import annotations

import contextlib
import inspect
import json
import re
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.tools import BaseTool

_CAMEL_SEP = re.compile(r"[-_]([a-z])")
_JS_IDENTIFIER = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_REPL_SYSTEM_PROMPT_TEMPLATE = (
    "### Interpreter\n\n"
    "An `{tool_name}` tool is available. It runs JavaScript in a persistent "
    "REPL.\n"
    "{state_persistence_line}\n"
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
    snapshot_between_turns: bool,
) -> str:
    """Render the base REPL system prompt text for ``REPLMiddleware``."""
    state_persistence_line = (
        "- State (variables, functions) persists across tool calls and across "
        "multiple turns for this conversation thread."
        if snapshot_between_turns
        else "- State (variables, functions) persists across tool calls within "
        "a single turn of conversation. They DO NOT persist across multiple turns."
    )
    return _REPL_SYSTEM_PROMPT_TEMPLATE.format(
        tool_name=tool_name,
        state_persistence_line=state_persistence_line,
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


def render_ptc_prompt(tools: Sequence[BaseTool], *, tool_name: str = "eval") -> str:
    """Build the `tools` namespace section of the system prompt."""
    if not tools:
        return ""
    blocks: list[str] = []
    for tool in tools:
        camel = to_camel_case(tool.name)
        schema = _safe_json_schema(tool)
        return_type = _render_return_type(tool)
        signature = _render_signature(camel, schema, return_type=return_type)
        description = (
            (tool.description or "").strip().splitlines()[0] if tool.description else ""
        )
        blocks.append(f"/** {description} */\n{signature}")
    body = "\n\n".join(blocks)
    referenced_types = _collect_referenced_typed_dicts(tools)
    referenced_block = ""
    if referenced_types:
        type_definitions = "\n\n".join(
            _render_typed_dict_definition(t) for t in referenced_types
        )
        referenced_block = (
            f"\n\nReferenced types:\n\n```typescript\n{type_definitions}\n```"
        )
    return (
        "\n\n"
        "### API Reference — `tools` namespace\n\n"
        "The agent tools listed below are exposed on the global object at "
        "`globalThis.tools` (also reachable as `tools`). Each takes a single "
        "object argument and returns a Promise that resolves to the tool's "
        "native value: strings as strings, numbers as numbers, lists as "
        "arrays, dicts as objects, and `None` as `null`. You do NOT need to "
        "`JSON.parse` results — they are already typed.\n\n"
        "Invocation pattern: `await tools.<name>({ ... })`.\n\n"
        "- Use `await` to get tool results; combine with `Promise.all` for "
        "independent calls so they run concurrently.\n"
        f"- If the task needs multiple tool calls, prefer one `{tool_name}` "
        "invocation that performs all of them rather than splitting the work "
        f"across multiple `{tool_name}` calls — each round-trip costs a model "
        "turn.\n"
        "- Pipeline dependent calls within a single program. If a result from "
        "one tool is needed as input to a later tool, chain them in one "
        "program instead of returning the intermediate value to the model.\n"
        "- If a tool returns an ID or other value that can be passed directly "
        "into the next tool, trust it and chain the calls instead of stopping "
        "to double-check it.\n"
        "- To inspect an intermediate value, `console.log` it inside the same "
        "program; otherwise, fetch as much information as possible in one "
        "call.\n"
        f"- Only split work across multiple `{tool_name}` invocations when "
        "you genuinely cannot determine what to do next without additional "
        "model reasoning or user input.\n\n"
        "Example shape — substitute real tool names:\n\n"
        "```typescript\n"
        'const users = await tools.findUsers({ name: "Ada" });\n'
        "const userId = users[0].id;\n"
        "const [city, normalized] = await Promise.all([\n"
        "  tools.cityForUser({ user_id: userId }),\n"
        '  tools.normalize({ name: "Ada" }),\n'
        "]);\n"
        "console.log({ city, normalized });\n"
        "```\n\n"
        "```typescript\n"
        f"{body}\n"
        "```"
        f"{referenced_block}"
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


def _render_signature(
    fn_name: str,
    schema: dict[str, Any] | None,
    *,
    return_type: str = "unknown",
) -> str:
    return_clause = f"Promise<{return_type}>"
    default_signature = (
        f"async function {fn_name}(input: Record<string, unknown>): {return_clause}"
    )
    if not schema or not isinstance(schema.get("properties"), dict):
        return default_signature
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
    if not body:
        return default_signature
    return f"async function {fn_name}(input: {{\n{body}\n}}): {return_clause}"


# ---------------------------------------------------------------------------
# Return-type rendering
#
# Inputs come from the tool's args_schema (Pydantic JSON Schema) so we get
# Field descriptions and required/optional handling. Return types do NOT have
# a JSON Schema source — only the function's Python annotation — so we render
# them directly. Intentionally narrow scope: primitives, list[T], dict[str, V],
# Optional[T], Literal[...], TypedDict (via referenced-type block), and bare
# class names. Anything richer (Union, generics, BaseModel field expansion)
# renders as ``unknown``.
# ---------------------------------------------------------------------------


def _is_typed_dict(annotation: Any) -> bool:
    return (
        isinstance(annotation, type)
        and hasattr(annotation, "__annotations__")
        and hasattr(annotation, "__required_keys__")
    )


def _is_optional_union(origin: Any, args: tuple[Any, ...]) -> bool:
    """Return whether *origin*/*args* describe ``T | None`` exactly."""
    origin_name = getattr(origin, "__name__", None)
    if origin_name not in {"Union", "UnionType"}:
        return False
    non_none = [a for a in args if a is not type(None)]
    return len(non_none) == 1 and len(args) == len(non_none) + 1


def _format_return_annotation(annotation: Any) -> str:  # noqa: C901, PLR0912 — flat dispatch over a small fixed set of annotation shapes; splitting hurts readability
    """Render a Python return annotation as a compact TS-ish string."""
    dict_arg_count = 2
    # Empty / Any → unknown.
    if annotation is inspect.Signature.empty or annotation is Any:
        return "unknown"

    # Primitives.
    if annotation is type(None):
        return "null"
    if annotation in (int, float):
        return "number"
    if annotation is str:
        return "string"
    if annotation is bool:
        return "boolean"
    # Bare ``dict`` / ``list`` (no parameters) render as their parameterised
    # equivalents so the model sees standard TS types instead of "dict"/"list".
    if annotation is dict:
        return "Record<string, unknown>"
    if annotation is list:
        return "unknown[]"

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[T] / T | None — render as ``T | null``. Other unions → unknown.
    if _is_optional_union(origin, args):
        non_none = next(a for a in args if a is not type(None))
        return f"{_format_return_annotation(non_none)} | null"

    # Literal["a", "b", 1] — quoted union of literal values.
    if origin is Literal:
        return " | ".join(json.dumps(arg) for arg in args)

    # Containers: list[T], dict[str, V]. Other generics → unknown.
    if origin in (list, set, frozenset):
        inner = _format_return_annotation(args[0]) if args else "unknown"
        return f"{inner}[]"
    if origin is dict:
        if len(args) == dict_arg_count and args[0] is str:
            return f"Record<string, {_format_return_annotation(args[1])}>"
        return "unknown"

    # TypedDict / bare class → render its name. TypedDicts get a definition
    # block emitted separately by `_collect_referenced_typed_dicts`.
    if isinstance(annotation, type):
        return annotation.__name__

    return "unknown"


def _get_tool_doc_target(tool: BaseTool) -> Callable[..., Any] | None:
    target = getattr(tool, "func", None)
    if callable(target):
        return target
    target = getattr(tool, "coroutine", None)
    if callable(target):
        return target
    return None


def _get_return_annotation(target: Callable[..., Any]) -> Any:
    """Resolve the return annotation for a callable, ``Signature.empty`` if absent."""
    with contextlib.suppress(TypeError, ValueError, NameError):
        signature = inspect.signature(target)
        resolved = get_type_hints(target)
        return resolved.get("return", signature.return_annotation)
    return inspect.Signature.empty


def _render_return_type(tool: BaseTool) -> str:
    target = _get_tool_doc_target(tool)
    if target is None:
        return "unknown"
    return _format_return_annotation(_get_return_annotation(target))


def _render_typed_dict_definition(annotation: type[Any]) -> str:
    optional_keys = frozenset(getattr(annotation, "__optional_keys__", frozenset()))
    try:
        field_types = get_type_hints(annotation)
    except (TypeError, NameError):
        field_types = getattr(annotation, "__annotations__", {})
    lines = [f"type {annotation.__name__} = {{"]
    for key, value in field_types.items():
        field_name = f"{key}?" if key in optional_keys else key
        lines.append(f"  {field_name}: {_format_return_annotation(value)};")
    lines.append("}")
    return "\n".join(lines)


def _collect_referenced_typed_dicts(tools: Sequence[BaseTool]) -> list[type[Any]]:
    """Return TypedDicts referenced by tools' return types, first-seen order."""
    collected: list[type[Any]] = []
    seen: set[type[Any]] = set()
    for tool in tools:
        target = _get_tool_doc_target(tool)
        if target is None:
            continue
        annotation = _get_return_annotation(target)
        # Surface the inner element of a one-level container: e.g.
        # ``list[ServiceSearchResult]`` should pull ServiceSearchResult.
        origin = get_origin(annotation)
        if origin in (list, set, frozenset):
            inner = get_args(annotation)
            if inner:
                annotation = inner[0]
        if not _is_typed_dict(annotation) or annotation in seen:
            continue
        seen.add(annotation)
        collected.append(annotation)
    return collected


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
