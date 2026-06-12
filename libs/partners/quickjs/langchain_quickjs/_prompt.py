"""Prompt/rendering helpers for REPL and PTC system prompts."""

from __future__ import annotations

import contextlib
import inspect
import json
import re
from typing import TYPE_CHECKING, Any, Literal, get_type_hints

from pydantic import TypeAdapter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tools import BaseTool

_CAMEL_SEP = re.compile(r"[-_]([a-z])")
_JS_IDENTIFIER = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_REPL_SYSTEM_PROMPT_TEMPLATE = (
    "### Interpreter\n\n"
    "{repl_intro_line}\n\n"
    "{state_persistence_line}\n"
    "- Top-level `await` works; Promises resolve before the call returns.\n"
    "- Runtime sandbox: no built-in filesystem, network, stdlib, or wall-clock "
    "APIs (`fetch`, `require`, `fs`, `process`, real `Date.now()` are "
    "unavailable or stubbed). External side effects from inside the REPL are "
    "only reachable via the `tools.*` namespace when it is exposed (see "
    "below); without it, the REPL is pure computation.\n"
    "- Timeout: {timeout}s per call. Memory: {memory_limit_mb} MB total.\n"
    "- `console.log` output is captured and returned alongside the result."
)
_SUBAGENT_SYSTEM_PROMPT_TEMPLATE = """

### Dispatching Subagents with `task`

`task` is your primitive for running configured subagents from inside the
JavaScript REPL. You orchestrate everything else - fan-out, filtering,
deduplication, multi-stage flow, and synthesis - in plain JavaScript.

#### The primitive

```javascript
await task({
  description,      // full autonomous task prompt
  subagent_type,    // configured subagent name
  response_schema,  // optional JSON Schema for structured output
}); // -> Promise<unknown>
```

`task` runs a full agentic loop for the selected configured subagent. The
subagent can use whatever tools it was configured with, iterate, inspect
context, and return one final result. `subagent_type` is required; use one of
the configured subagent names.

`description` is the only prompt the subagent receives for this dispatch. Make
it complete: include the goal, constraints, relevant context, what to inspect,
and the exact shape or level of detail you expect back. Each dispatch is
stateless from the caller's perspective; you cannot send follow-up messages to
the same subagent run.

`response_schema` is optional. When provided, the resolved value is already a
typed JavaScript value matching the schema. Do not call `JSON.parse` unless the
subagent intentionally returned a JSON string. Dynamic schemas work for
declarative subagents; runnable-backed subagents reject dynamic schemas because
their runnable is already compiled.

#### Approval model

`task` dispatches from inside the already-running `{tool_name}` call. It
does not route through the parent agent's `ToolNode`-managed `task` tool and
does not trigger parent-level `interrupt_on` / HITL approval for each dispatch.
Declarative subagents still honor approval middleware configured inside their
own spec. If you need approval before launching a subagent from the parent, use
the normal `task` tool outside JavaScript or ensure the `{tool_name}` call
itself is approval-gated.

#### Mental model

Hold your work in JS: an array of items in, an array of results out. Merge each
dispatch result back onto its item. Multi-stage analysis means: run a pass,
filter or regroup the array in JS, then run another pass over the survivors.

Prefer one `{tool_name}` call that performs the whole workflow. Splitting the
workflow across multiple `{tool_name}` calls costs model turns and forces you to
re-establish state.

#### Fan out with bounded concurrency

Dispatch independent work in parallel with `Promise.all`, but in explicit
batches around 10 so you do not launch hundreds of subagents at once. The bridge
enforces a hard per-REPL cap of 32 concurrent `task` calls.

```javascript
const batchSize = 10;
const reviewed = [];
for (let i = 0; i < items.length; i += batchSize) {
  const batch = items.slice(i, i + batchSize);
  reviewed.push(...(await Promise.all(batch.map(async (it) => {
    const result = await task({
      description: "Review " + it.file + " for SQL injection. Cite line numbers.",
      subagent_type: "reviewer",
      response_schema: {
        type: "object",
        properties: {
          vulnerabilities: {
            type: "array",
            items: {
              type: "object",
              properties: {
                type: { type: "string" },
                line: { type: "number" },
                evidence: { type: "string" },
              },
              required: ["type", "line", "evidence"],
            },
          },
        },
        required: ["vulnerabilities"],
      },
    });
    return { ...it, ...result };
  }))));
}
```

#### Use parent JS for cheap work; use subagents for agentic work

Use JavaScript in the parent REPL for deterministic orchestration: joining
arrays, deduping, sorting, filtering, grouping, batching, and merging results.
If the `tools.*` namespace is exposed, also use it to pre-read files or collect
shared data once, then pass only the relevant content to each subagent in
`description`.

Use `task` for work that benefits from an autonomous agentic loop: reading
or searching with the subagent's own tools, inspecting multiple files, following
leads, making judgment calls, or producing a final synthesized report.

#### Pre-read shared context in the parent when useful

If many subagents need the same source list or file content and `tools.*` is
available, gather that context once in the parent REPL before dispatching:

```javascript
const files = (await tools.glob({ pattern: "src/**/*.ts" }))
  .split("\\n")
  .filter(Boolean);

const items = await Promise.all(files.map(async (file) => {
  const content = await tools.readFile({ file_path: file });
  return { file, content };
}));

const batchSize = 10;
const results = [];
for (let i = 0; i < items.length; i += batchSize) {
  const batch = items.slice(i, i + batchSize);
  results.push(...(await Promise.all(batch.map(async (it) => {
    const finding = await task({
      description:
        "Review this file for auth bypasses. Return concrete findings only.\\n\\n" +
        "File: " + it.file + "\\n\\n" +
        it.content,
      subagent_type: "reviewer",
      response_schema: {
        type: "object",
        properties: {
          findings: { type: "array", items: { type: "object" } },
        },
        required: ["findings"],
      },
    });
    return { ...it, ...finding };
  }))));
}
```

#### Compose multiple stages

Filter the array in JS between passes. For example: first ask subagents for a
cheap classification, filter to the risky items, then dispatch deeper reviews
only for those items.

```javascript
const tagged = [];
for (let i = 0; i < items.length; i += 10) {
  const batch = items.slice(i, i + 10);
  tagged.push(...(await Promise.all(batch.map(async (it) => {
    const tag = await task({
      description: "Classify " + it.file + " as handler, util, test, or config.",
      subagent_type: "reviewer",
      response_schema: {
        type: "object",
        properties: { kind: { type: "string" }, risky: { type: "boolean" } },
        required: ["kind", "risky"],
      },
    });
    return { ...it, ...tag };
  }))));
}

const riskyHandlers = tagged.filter((it) => it.kind === "handler" && it.risky);
const deepReviews = [];
for (let i = 0; i < riskyHandlers.length; i += 10) {
  const batch = riskyHandlers.slice(i, i + 10);
  deepReviews.push(...(await Promise.all(batch.map(async (it) => {
    const review = await task({
      description: "Deep security review of " + it.file + ". Cite line numbers.",
      subagent_type: "reviewer",
    });
    return { ...it, review };
  }))));
}
```

#### Get results out without flooding your context

Keep large result sets in JS variables. Do not `console.log` the full result set.
If `tools.writeFile` is exposed, persist structured output from inside the eval:

```javascript
await tools.writeFile({
  file_path: "/results/subagent-output.json",
  content: JSON.stringify(deepReviews),
});
```

Otherwise return a compact summary or a small slice of the results, not the
entire intermediate dataset.

#### Across evals

Variables persist according to the interpreter persistence mode above, but
re-establish what you need in each eval. Doing the whole workflow in one
`{tool_name}` call is usually simplest.
"""


def render_repl_system_prompt(
    *,
    tool_name: str,
    timeout: float,
    memory_limit_mb: int,
    mode: Literal["thread", "turn", "call"],
) -> str:
    """Render the base REPL system prompt text for ``CodeInterpreterMiddleware``."""
    if mode == "call":
        repl_intro_line = (
            f"An `{tool_name}` tool is available. It runs JavaScript in a fresh "
            "sandboxed REPL for each invocation."
        )
        state_persistence_line = (
            "- State (variables, functions) does not persist across tool calls. "
            "Each invocation starts from a blank environment."
        )
    elif mode == "thread":
        repl_intro_line = (
            f"An `{tool_name}` tool is available. It runs JavaScript in a persistent "
            "REPL."
        )
        state_persistence_line = (
            "- State (variables, functions) persists across tool calls and across "
            "multiple turns for this conversation thread."
        )
    else:
        repl_intro_line = (
            f"An `{tool_name}` tool is available. It runs JavaScript in a persistent "
            "REPL."
        )
        state_persistence_line = (
            "- State (variables, functions) persists across tool calls within "
            "a single turn of conversation. They DO NOT persist across multiple turns."
        )
    return _REPL_SYSTEM_PROMPT_TEMPLATE.format(
        repl_intro_line=repl_intro_line,
        state_persistence_line=state_persistence_line,
        timeout=timeout,
        memory_limit_mb=memory_limit_mb,
    )


def render_subagent_system_prompt(*, tool_name: str = "eval") -> str:
    """Render guidance for the top-level QuickJS `task` global."""
    return _SUBAGENT_SYSTEM_PROMPT_TEMPLATE.replace("{tool_name}", tool_name)


def render_eval_tool_code_doc(*, mode: Literal["thread", "turn", "call"]) -> str:
    """Render the eval tool's `code` argument description."""
    if mode == "call":
        persistence = (
            "Each call runs in a fresh REPL environment (no cross-call state)."
        )
    elif mode == "thread":
        persistence = (
            "State persists across calls and across turns in this conversation."
        )
    else:
        persistence = (
            "State persists across calls within a turn, but resets between turns."
        )
    return (
        "JavaScript expression or statement(s) to evaluate in the sandboxed REPL. "
        f"{persistence}"
    )


def render_eval_tool_description(*, mode: Literal["thread", "turn", "call"]) -> str:
    """Render the public eval tool description."""
    if mode == "call":
        state_line = (
            "Each call runs in a fresh sandboxed REPL with no state carried over."
        )
    elif mode == "thread":
        state_line = (
            "Persistent state is enabled: variables and functions defined in one "
            "call are visible to subsequent calls in this conversation."
        )
    else:
        state_line = (
            "Persistent state is enabled within a single turn: variables and "
            "functions defined in one call are visible to later calls within "
            "the same turn, but reset between turns."
        )
    return (
        "Execute JavaScript in a sandboxed REPL. "
        f"{state_line} No filesystem, network, or real clock. "
        "Synchronous only — top-level `await` will not resolve."
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


# Return types come from the tool's underlying function annotation. We feed
# the annotation through ``pydantic.TypeAdapter`` to get a JSON Schema and
# render it through the same ``_json_schema_to_ts`` we use for input args.
# Compound shapes (TypedDict, BaseModel, recursive types) end up as ``$ref``
# in the schema and currently render as ``unknown`` — same behaviour as
# nested-model input args. Until that path resolves ``$ref`` / ``$defs``,
# the simpler unified renderer is the right trade-off here.


def _render_return_type(tool: BaseTool) -> str:
    """Render the return annotation as a TS type, defaulting to ``unknown``."""
    target = getattr(tool, "func", None) or getattr(tool, "coroutine", None)
    if target is None:
        return "unknown"
    annotation = inspect.Signature.empty
    with contextlib.suppress(TypeError, ValueError, NameError):
        signature = inspect.signature(target)
        resolved = get_type_hints(target)
        annotation = resolved.get("return", signature.return_annotation)
    if annotation is inspect.Signature.empty or annotation is Any:
        return "unknown"
    try:
        schema = TypeAdapter(annotation).json_schema()
    except Exception:  # noqa: BLE001 — schema generation is best-effort
        return "unknown"
    return _json_schema_to_ts(schema)


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
