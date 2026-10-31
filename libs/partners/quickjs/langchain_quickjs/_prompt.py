"""Prompt/rendering helpers for REPL and PTC system prompts."""

from __future__ import annotations

import contextlib
import inspect
import json
import re
from typing import TYPE_CHECKING, Any, Literal, get_type_hints

from pydantic import BaseModel, TypeAdapter

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from langchain_core.tools import BaseTool
    from pydantic.json_schema import JsonSchemaMode

_CAMEL_SEP = re.compile(r"[-_]([a-z])")
_JS_IDENTIFIER = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
# Names a declared alias must not shadow: the TS helpers the prompt itself uses.
_RESERVED_TYPE_NAMES = frozenset({"Array", "Promise", "Record"})
_REPL_SYSTEM_PROMPT_TEMPLATE = (
    "### Interpreter\n\n"
    "{repl_intro_line}\n\n"
    "{state_persistence_line}\n"
    "- Top-level `await` works; Promises resolve before the call returns.\n"
    "- Runtime sandbox: no built-in filesystem, network, stdlib, or wall-clock "
    "APIs (`fetch`, `require`, `fs`, `process`, real `Date.now()` are "
    "unavailable or stubbed).\n"
    "{side_effects_line}\n"
    "- {timeout_line}Memory: {memory_limit_mb} MB total.\n"
    "- `console.log` output is captured and returned alongside the result."
)
_SUBAGENT_SYSTEM_PROMPT_TEMPLATE = """

### Dispatching Subagents with `task`

`task` is your primitive for running configured subagents from inside the
JavaScript REPL. Your job here is to DISTRIBUTE work, not to do it yourself:
write JavaScript that fans work out to subagents and assembles their results.
You handle the orchestration - fan-out, filtering, deduplication, multi-stage
flow, and synthesis - in plain JavaScript.

#### The primitive

```javascript
await task({
  description,      // full autonomous task prompt
  subagentType,     // configured subagent name
  label,            // optional short UI label for this dispatch
  responseSchema,   // optional JSON Schema for structured output
}); // -> Promise<unknown>
```

`task` runs a full agentic loop for the selected configured subagent. The
subagent can use whatever tools it was configured with, iterate, inspect
context, and return one final result. `subagentType` is required; use one of
the configured subagent names.

`description` is the only prompt the subagent receives for this dispatch. Make
it complete: the goal, the constraints, what to inspect, and the exact shape
or level of detail you expect back. Give context as locators — file paths and
symbol names — not as pasted file contents. If you already read a file while
exploring, still pass its path and let the subagent read it; do not paste back
what you read. Each dispatch is stateless from the caller's perspective; you
cannot send follow-up messages to the same subagent run.

`label` is optional: when provided, it is shown in the live progress UI
instead of the default description-derived fallback. It is not sent to the
subagent and does not affect execution.

`responseSchema` is optional, but set it on any dispatch whose result feeds
later code. A deterministic, typed shape is what lets you compose the next
stage reliably — index it, sort it, compare fields, branch on it, merge it —
instead of parsing free-form text. This is what makes a whole workflow
composable as one script. When provided, the resolved value is already a typed
JavaScript value matching the schema; do not call `JSON.parse` unless the
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

You can run the whole workflow in one `{tool_name}` call or split it across
several — both are fine. A single end-to-end script (generate, compare, pick a
winner; or review every item, then synthesize) is clean when you can write it
in one go; splitting is also fine when you want to inspect results between
stages. Either way, don't redo work across calls — reuse what is already in
scope (see "Reuse what earlier evals left in scope" below).

#### Fan out with bounded concurrency

Dispatch independent work in parallel with `Promise.all`. Batches around 10 are
a conservative recommendation, not a required limit; use larger batches when
appropriate for the workload and its admission controls. The bridge enforces
a per-REPL cap of {max_subagent_concurrency} concurrent subagent calls; excess
calls wait for capacity.

```javascript
const files = ["/src/a.ts", "/src/b.ts", "/src/c.ts"]; // found while exploring
const batchSize = 10;
const reviewed = [];
for (let i = 0; i < files.length; i += batchSize) {
  const batch = files.slice(i, i + batchSize);
  reviewed.push(...(await Promise.all(batch.map(async (file) => {
    const result = await task({
      description: "Read " + file + " and review it for SQL injection. " +
        "Cite line numbers.",
      subagentType: "reviewer",
      responseSchema: {
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
    return { file, ...result };
  }))));
}
```

#### Explore with your own tools first, then distribute

You already have your normal tools for reading, listing, globbing, and
grepping files. Use them to explore and understand the task BEFORE you write
the orchestration script. These are ordinary tool calls, separate from the
`{tool_name}` tool: read the data file, list or glob the directory, grep for
what matters, then decide how to split the work.

Never write `{tool_name}` code that spawns a subagent just to read or parse a
file or list a directory. That is a deterministic step you do yourself with a
direct tool call; spending a whole agent loop on it is wasteful.

Once you understand the shape of the work, you have creative freedom in how
you split it:

- One dispatch per file or per record, when the items are already separate.
- Chunk a large input yourself — read it, split it, optionally write a small
  input file per chunk — and dispatch one subagent per chunk.
- A cheap classification pass first, then deeper dispatches only for the items
  that warrant them.

Then write JavaScript in the `{tool_name}` tool that distributes the heavy,
agentic work to subagents with `task()`: analyzing file contents, exploring a
codebase, making judgment calls, rewriting code, or synthesizing a report.

Hand each subagent a locator, not a payload. Subagents have their own file
tools, so for anything that lives in a file — a file to review, rewrite, or
audit — pass the path and let the subagent read it. Do NOT read a whole file
just to paste its contents into the description; that bloats every dispatch
and duplicates the file across them. Reserve inline content for small or
derived data that has no path of its own: a single parsed record, or a chunk
you split out of a larger input (write the chunk to its own file and pass that
path if it is large). Assemble the results in JS.

#### Compose multiple stages

Filter the array in JS between passes. For example: first ask subagents for a
cheap classification, filter to the risky items, then dispatch deeper reviews
only for those items.

```javascript
const tagged = await Promise.all(files.map((file) =>
  task({
    description: "Read " + file + " and classify it as handler, util, " +
      "test, or config.",
    subagentType: "reviewer",
    responseSchema: {
      type: "object",
      properties: { kind: { type: "string" }, risky: { type: "boolean" } },
      required: ["kind", "risky"],
    },
  }).then((tag) => ({ file, ...tag }))
));

const riskyHandlers = tagged.filter((it) => it.kind === "handler" && it.risky);
const deepReviews = await Promise.all(riskyHandlers.map((it) =>
  task({
    description: "Deep security review of " + it.file + ". Cite line numbers.",
    subagentType: "reviewer",
  }).then((review) => ({ ...it, review }))
));
```

#### Return results via the last expression, not `console.log`

The value of the last expression in an `{tool_name}` call (or a resolved
top-level `await`) is returned to you as the result. Make that final
expression the variable holding your result and read it from there.
`console.log` is only for incidental debugging: its output is capped and
truncated, while the returned value is not, so never `console.log` your
actual results.

Keep large intermediate sets in JS variables and return only a compact
summary or a small slice, not the entire dataset. To persist full output,
have a subagent write it, or write it with your own file tool outside the
`{tool_name}` call.

#### Reuse what earlier evals left in scope

The REPL is persistent within a turn: every top-level variable, function, and
class you declare is kept and is available in your next `{tool_name}` call
(each is hoisted to global scope). So if a later step needs something an
earlier eval produced or bound, **reference that variable by name** — do not
write a new literal that re-types data a previous eval already returned or
computed.

If you catch yourself pasting a big array or object of values you produced in
an earlier call, that is the tell: the variable is still in scope, so use it.
Re-typing prior results as a fresh literal wastes tokens and drifts from what
actually ran.

```javascript
// An earlier eval bound this:
//   const auditResults = await Promise.all(files.map(/* ...audit... */));

// A later eval — reference it; do NOT paste the findings back in as a literal:
const findings = auditResults.flatMap((r) =>
  r.findings.map((f) => ({ ...f, file: r.file }))
);
const verified = await Promise.all(findings.map((f) =>
  task({
    description: "Verify this finding: " + f.evidence,
    subagentType: "verifier",
  }).then((v) => ({ ...f, ...v }))
));
```

#### When the user asks for a "workflow"

If the user's request mentions running a "workflow" (or otherwise uses the
word "workflow"), fan the work out to subagents rather than doing it all
yourself. Explore with your own tools first as needed, then write JavaScript
in the `{tool_name}` tool that dispatches subagents with `task()` and
assembles their results. The point is to distribute the heavy work in
parallel, not to grind through it one tool call at a time.
"""


def render_repl_system_prompt(
    *,
    tool_name: str,
    timeout: float | None,
    memory_limit_mb: int,
    mode: Literal["thread", "turn", "call"],
    ptc_attached: bool = False,
) -> str:
    """Render the base REPL system prompt text for `CodeInterpreterMiddleware`.

    `ptc_attached` controls the "external side effects" bullet: when host
    tools are exposed as the `tools.*` namespace it points the model at the
    API reference; otherwise it states the REPL is pure computation.
    """
    if ptc_attached:
        side_effects_line = (
            "- External side effects from inside the REPL are only reachable "
            "via the `tools.*` namespace documented in the API reference below."
        )
    else:
        side_effects_line = (
            "- The REPL has no access to host tools, files, or the network: it "
            "is pure computation. Return values to communicate results."
        )
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
        side_effects_line=side_effects_line,
        timeout_line=("" if timeout is None else f"Timeout: {timeout}s per call. "),
        memory_limit_mb=memory_limit_mb,
    )


def render_subagent_system_prompt(
    *, tool_name: str = "eval", max_subagent_concurrency: int = 32
) -> str:
    """Render guidance for the top-level QuickJS `task` global."""
    return _SUBAGENT_SYSTEM_PROMPT_TEMPLATE.replace("{tool_name}", tool_name).replace(
        "{max_subagent_concurrency}", str(max_subagent_concurrency)
    )


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
        "Top-level `await` is supported; a final-expression Promise resolves "
        "before the call returns."
    )


def to_camel_case(name: str) -> str:
    """Convert `snake_case` / `kebab-case` → `camelCase`."""
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
    schemas, scope = _collect_pydantic_schemas(tools)
    blocks: list[str] = []
    for index, tool in enumerate(tools):
        camel = to_camel_case(tool.name)
        input_schema = schemas.get((index, "input"))
        output_schema = schemas.get((index, "output"))
        signature = _render_signature(
            camel,
            input_schema if input_schema is not None else _safe_json_schema(tool),
            return_type=(
                _render_root_type(output_schema, scope)
                if output_schema is not None
                else _render_return_type(tool)
            ),
            scope=scope if input_schema is not None else None,
        )
        description = (
            (tool.description or "").strip().splitlines()[0] if tool.description else ""
        )
        blocks.append(f"/** {description} */\n{signature}")
    if scope.declarations:
        blocks.insert(0, "\n".join(scope.declarations))
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
    scope: _RefScope | None = None,
) -> str:
    return_clause = f"Promise<{return_type}>"
    default_signature = (
        f"tools.{fn_name}(input: Record<string, unknown>): {return_clause}"
    )
    root = scope.root(schema) if scope and schema else schema
    if not root or not isinstance(root.get("properties"), dict):
        return default_signature
    props: dict[str, Any] = root["properties"]
    required = set(root.get("required", []))
    fields = []
    for key, prop in props.items():
        optional = "" if key in required else "?"
        type_str = _json_schema_to_ts(prop, scope)
        desc = prop.get("description")
        prefix = f"/**\n *{desc}\n */ " if desc else ""
        fields.append(f"  {prefix}{key}{optional}: {type_str};")
    body = "\n".join(fields) if fields else ""
    if not body:
        return default_signature
    return f"tools.{fn_name}(input: {{\n{body}\n}}): {return_clause}"


_SchemaKey = tuple[int, Literal["input", "output"]]


def _collect_pydantic_schemas(
    tools: Sequence[BaseTool],
) -> tuple[dict[_SchemaKey, dict[str, Any]], _RefScope]:
    """Generate every tool's Pydantic input/return schema in one batch.

    `TypeAdapter.json_schemas` puts all nested models in one shared `$defs`,
    deduplicated by class and with clashing class names qualified by Pydantic.
    Schemas that can't join the batch are left out and rendered per tool.
    """
    adapters: list[tuple[_SchemaKey, JsonSchemaMode, TypeAdapter[Any]]] = []
    for index, tool in enumerate(tools):
        for key, adapter in (
            ((index, "input"), _input_adapter(tool)),
            ((index, "output"), _return_adapter(tool)),
        ):
            if adapter is not None:
                adapters.append((key, "validation", adapter))
    if not adapters:
        return {}, _RefScope({})
    try:
        roots, definitions = TypeAdapter.json_schemas(adapters)
    except Exception:  # noqa: BLE001 — fall back to per-tool rendering
        return {}, _RefScope({})
    schemas = {key: schema for (key, _mode), schema in roots.items()}
    return schemas, _RefScope(definitions.get("$defs", {}))


def _input_adapter(tool: BaseTool) -> TypeAdapter[Any] | None:
    """Return an adapter for a plain Pydantic `args_schema`, else `None`.

    A model that customizes `model_json_schema` is left to the per-tool path,
    which renders the schema it declares.
    """
    args_schema = tool.args_schema
    if not (isinstance(args_schema, type) and issubclass(args_schema, BaseModel)):
        return None
    try:
        adapter = TypeAdapter(args_schema)
        if adapter.json_schema() != args_schema.model_json_schema():
            return None
    except Exception:  # noqa: BLE001 — prompt rendering is best-effort
        return None
    return adapter


def _return_adapter(tool: BaseTool) -> TypeAdapter[Any] | None:
    annotation = _return_annotation(tool)
    if annotation is None:
        return None
    try:
        adapter = TypeAdapter(annotation)
        adapter.json_schema()
    except Exception:  # noqa: BLE001 — schema generation is best-effort
        return None
    return adapter


def _return_annotation(tool: BaseTool) -> Any:
    """Return the tool function's return annotation, or `None` if unusable."""
    target = getattr(tool, "func", None) or getattr(tool, "coroutine", None)
    if target is None:
        return None
    annotation = inspect.Signature.empty
    with contextlib.suppress(TypeError, ValueError, NameError):
        signature = inspect.signature(target)
        resolved = get_type_hints(target)
        annotation = resolved.get("return", signature.return_annotation)
    if annotation is inspect.Signature.empty or annotation is Any:
        return None
    return annotation


def _render_return_type(tool: BaseTool) -> str:
    """Render a return annotation outside the batch, defaulting to `unknown`."""
    annotation = _return_annotation(tool)
    if annotation is None:
        return "unknown"
    try:
        schema = TypeAdapter(annotation).json_schema()
    except Exception:  # noqa: BLE001 — schema generation is best-effort
        return "unknown"
    return _json_schema_to_ts(schema)


def _render_root_type(schema: dict[str, Any], scope: _RefScope) -> str:
    """Render a root schema, inlining a top-level model unless it is recursive."""
    ref = schema.get("$ref")
    if scope.is_recursive(ref):
        return scope.resolve(ref)
    return _json_schema_to_ts(scope.root(schema), scope)


class _RefScope:
    """Renders the shared Pydantic `$defs` as named TS aliases.

    Each definition is declared once as `type Name = ...`, on first use, so a
    model shared by several fields or tools isn't repeated and a recursive
    model refers to its own alias.
    """

    def __init__(self, definitions: dict[str, Any]) -> None:
        self.definitions = definitions
        self.declarations: list[str] = []
        self._declared: set[str] = set()
        self._aliases: dict[str, str] = {}
        used: set[str] = set(_RESERVED_TYPE_NAMES)
        for name in definitions:
            alias = base = _to_type_name(name)
            suffix = 1
            while alias in used:
                suffix += 1
                alias = f"{base}{suffix}"
            used.add(alias)
            self._aliases[name] = alias

    def root(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Resolve a top-level `$ref` so the root model renders inline."""
        definition = self.definitions.get(_ref_name(schema.get("$ref")) or "")
        return definition if isinstance(definition, dict) else schema

    def is_recursive(self, ref: Any) -> bool:
        """Return whether the definition `ref` points at can reach itself."""
        start = _ref_name(ref)
        pending, seen = [start], set()
        while pending:
            definition = self.definitions.get(pending.pop() or "")
            for found in _local_refs(definition):
                if found == start:
                    return True
                if found not in seen:
                    seen.add(found)
                    pending.append(found)
        return False

    def resolve(self, ref: Any) -> str:
        """Render a `$ref` as its alias, declaring the alias on first use."""
        name = _ref_name(ref)
        definition = self.definitions.get(name) if name is not None else None
        if name is None or not isinstance(definition, dict):
            return "unknown"
        alias = self._aliases[name]
        if name not in self._declared:
            # Mark before rendering so a recursive reference resolves to the name.
            self._declared.add(name)
            self.declarations.append(
                f"type {alias} = {_json_schema_to_ts(definition, self)};"
            )
        return alias


def _local_refs(node: Any) -> Iterator[str | None]:
    """Yield the definition names referenced anywhere inside `node`."""
    if isinstance(node, dict):
        if "$ref" in node:
            yield _ref_name(node["$ref"])
        for value in node.values():
            yield from _local_refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from _local_refs(value)


def _ref_name(ref: Any) -> str | None:
    """Return the definition name of a local `#/$defs/...` reference."""
    if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
        return None
    return ref.removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")


def _to_type_name(name: str) -> str:
    """Turn a definition name such as `mod__Address` into a TS identifier."""
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", name) if part]
    type_name = "".join(part[:1].upper() + part[1:] for part in parts)
    return re.sub(r"^\d+", "", type_name) or "Model"


def _json_schema_to_ts(prop: dict[str, Any], scope: _RefScope | None = None) -> str:
    """JSON-Schema → TS type renderer; nested definitions render by name."""
    if not isinstance(prop, dict):
        return "unknown"
    if "$ref" in prop:
        return scope.resolve(prop["$ref"]) if scope else "unknown"
    if "const" in prop:
        return json.dumps(prop["const"])
    if "enum" in prop:
        return " | ".join(json.dumps(v) for v in prop["enum"])
    for key, separator in (("anyOf", " | "), ("oneOf", " | "), ("allOf", " & ")):
        if isinstance(prop.get(key), list):
            parts = [
                _json_schema_to_ts(part, scope)
                for part in prop[key]
                if isinstance(part, dict)
            ]
            return separator.join(dict.fromkeys(parts)) or "unknown"
    return _render_typed_schema(prop, scope)


def _render_typed_schema(prop: dict[str, Any], scope: _RefScope | None) -> str:
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
        inner = (
            _json_schema_to_ts(items, scope) if isinstance(items, dict) else "unknown"
        )
        return f"({inner})[]" if " | " in inner or " & " in inner else f"{inner}[]"
    if t == "object":
        return _render_object(prop, scope)
    return "unknown"


def _render_object(prop: dict[str, Any], scope: _RefScope | None) -> str:
    sub_props = prop.get("properties")
    if isinstance(sub_props, dict) and sub_props:
        required = set(prop.get("required", []))
        fields = [
            f"{k}{'' if k in required else '?'}: {_json_schema_to_ts(v, scope)}"
            for k, v in sub_props.items()
        ]
        return "{ " + "; ".join(fields) + " }"
    additional = prop.get("additionalProperties")
    if isinstance(additional, dict) and additional:
        return f"Record<string, {_json_schema_to_ts(additional, scope)}>"
    return "Record<string, unknown>"
