"""Prompt/rendering helpers for REPL and PTC system prompts."""

from __future__ import annotations

import contextlib
import inspect
import itertools
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin, get_type_hints

from langchain_core.messages.tool import ToolOutputMixin
from pydantic import TypeAdapter

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from langchain_core.tools import BaseTool

_CAMEL_SEP = re.compile(r"[-_]([a-z])")
_JS_IDENTIFIER = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_NON_IDENTIFIER_CHAR = re.compile(r"[^A-Za-z0-9_$]")
_DEFS_REF_PREFIX = "#/$defs/"
# Names a declared type must not shadow: types the renderer itself emits.
_RESERVED_TYPE_NAMES = frozenset(
    {
        "Promise",
        "Record",
        "Uint8Array",
        "boolean",
        "null",
        "number",
        "string",
        "unknown",
    }
)
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
    body = _render_api_reference(tools)
    return (
        "\n\n"
        "### API Reference — `tools` namespace\n\n"
        "The agent tools listed below are exposed on the global object at "
        "`globalThis.tools` (also reachable as `tools`). Each takes a single "
        "object argument and returns a Promise that resolves to the tool's "
        "data, not the text the tool would show you: strings as strings, "
        "numbers as numbers, lists as arrays, dicts, models, and dataclasses "
        "as objects, bytes as `Uint8Array`, and `None` as `null`. A tool that "
        "returns separate content and an artifact resolves to the artifact. "
        "You do NOT need to `JSON.parse` results — they are already typed.\n\n"
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


def _render_api_reference(tools: Sequence[BaseTool]) -> str:
    """Render the named type declarations followed by each tool's signature."""
    declarations = _TypeDeclarations()
    blocks: list[str] = []
    for tool in tools:
        camel = to_camel_case(tool.name)
        owner = camel[:1].upper() + camel[1:]
        schema = _safe_json_schema(tool)
        refs = declarations.declare(schema, owner=owner) if schema else {}
        return_type = declarations.render(_return_schema(tool), owner=owner)
        signature = _render_signature(camel, schema, refs, return_type=return_type)
        description = (
            (tool.description or "").strip().splitlines()[0] if tool.description else ""
        )
        blocks.append(f"/** {description} */\n{signature}")
    return "\n\n".join([*declarations.blocks, *blocks])


@dataclass
class _TypeDeclarations:
    """Named TS types shared by every rendered `tools.*` signature.

    Each JSON Schema `$defs` entry becomes a `type` alias and each `$ref`
    renders as that alias's name, so recursive types terminate. Schemas are
    generated independently per tool, so a def is identified by its own
    schema plus every def it reaches: the same model used by two tools gets
    one declaration, and a different model under a taken name is prefixed
    with the tool's name.
    """

    blocks: list[str] = field(default_factory=list)
    _names: dict[str, str] = field(default_factory=dict)
    _taken: set[str] = field(default_factory=lambda: set(_RESERVED_TYPE_NAMES))

    def render(self, schema: dict[str, Any] | None, *, owner: str) -> str:
        """Render `schema` as a TS type, declaring the types it references."""
        if schema is None:
            return "unknown"
        return _json_schema_to_ts(schema, self.declare(schema, owner=owner))

    def declare(self, schema: dict[str, Any], *, owner: str) -> dict[str, str]:
        """Declare `schema`'s `$defs` and map each `$ref` to its TS name."""
        defs: dict[str, Any] = schema.get("$defs") or {}
        refs: dict[str, str] = {}
        new: list[str] = []
        for def_name in defs:
            identity = _def_identity(def_name, defs)
            if identity not in self._names:
                self._names[identity] = self._claim(def_name, owner)
                new.append(def_name)
            refs[_DEFS_REF_PREFIX + def_name] = self._names[identity]
        for def_name in new:
            name = refs[_DEFS_REF_PREFIX + def_name]
            self.blocks.append(_render_declaration(name, defs[def_name], refs))
        return refs

    def _claim(self, def_name: str, owner: str) -> str:
        """Reserve a free TS name for `def_name`, prefixing `owner` on conflict."""
        base = _NON_IDENTIFIER_CHAR.sub("_", def_name)
        if not is_valid_js_identifier(base):
            base = f"_{base}"
        candidates = itertools.chain(
            (base, f"{owner}{base}"),
            (f"{owner}{base}{i}" for i in itertools.count(2)),
        )
        name = next(c for c in candidates if c not in self._taken)
        self._taken.add(name)
        return name


def _def_identity(def_name: str, defs: Mapping[str, Any]) -> str:
    """Return a key identifying `def_name` together with every def it reaches."""
    reached: dict[str, Any] = {}
    pending = [def_name]
    while pending:
        name = pending.pop()
        if name in reached or name not in defs:
            continue
        reached[name] = defs[name]
        pending.extend(
            ref.removeprefix(_DEFS_REF_PREFIX)
            for ref in _refs_in(defs[name])
            if ref.startswith(_DEFS_REF_PREFIX)
        )
    return json.dumps([def_name, reached], sort_keys=True, default=str)


def _refs_in(node: object) -> Iterator[str]:
    """Yield every `$ref` string nested anywhere in a JSON Schema node."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            yield ref
        for value in node.values():
            yield from _refs_in(value)
    elif isinstance(node, list):
        for item in node:
            yield from _refs_in(item)


def _render_declaration(
    name: str, schema: dict[str, Any], refs: Mapping[str, str]
) -> str:
    """Render one `$defs` entry as a TS `type` alias."""
    description = (schema.get("description") or "").strip().splitlines()
    doc = f"/** {description[0]} */\n" if description else ""
    if schema.get("type") == "object" and schema.get("properties"):
        body = "{\n" + "\n".join(_render_fields(schema, refs)) + "\n}"
    else:
        body = _json_schema_to_ts(schema, refs)
    return f"{doc}type {name} = {body};"


def _safe_json_schema(tool: BaseTool) -> dict[str, Any] | None:
    """Return the JSON Schema of `tool`'s input, if it has one.

    MCP tools carry the server's JSON Schema as a dict; other tools carry a
    Pydantic model.
    """
    if isinstance(tool.args_schema, dict):
        return tool.args_schema
    try:
        model_json_schema = getattr(tool.args_schema, "model_json_schema", None)
        if callable(model_json_schema):
            return model_json_schema()
    except Exception:  # noqa: BLE001 — prompt rendering is best-effort
        return None
    return None


def _render_signature(
    fn_name: str,
    schema: dict[str, Any] | None,
    refs: Mapping[str, str],
    *,
    return_type: str = "unknown",
) -> str:
    return_clause = f"Promise<{return_type}>"
    default_signature = (
        f"tools.{fn_name}(input: Record<string, unknown>): {return_clause}"
    )
    if not schema or not isinstance(schema.get("properties"), dict):
        return default_signature
    body = "\n".join(_render_fields(schema, refs))
    if not body:
        return default_signature
    return f"tools.{fn_name}(input: {{\n{body}\n}}): {return_clause}"


def _render_fields(schema: dict[str, Any], refs: Mapping[str, str]) -> list[str]:
    """Render an object schema's properties as one TS field per line."""
    required = set(schema.get("required", []))
    fields = []
    for key, prop in schema["properties"].items():
        optional = "" if key in required else "?"
        type_str = _json_schema_to_ts(prop, refs)
        desc = prop.get("description") if isinstance(prop, dict) else None
        prefix = f"/**\n *{desc}\n */ " if desc else ""
        fields.append(f"  {prefix}{key}{optional}: {type_str};")
    return fields


# Return types come from the tool's underlying function annotation, narrowed
# to the value JS gets: a `content_and_artifact` tool's artifact element, and
# `unknown` for `ToolMessage` / `Command` envelopes, whose data is only known
# at runtime. We feed the annotation through `pydantic.TypeAdapter` in
# serialization mode, matching how the bridge dumps the value, and render it
# through the same `_json_schema_to_ts` we use for input args. MCP tools have
# no useful annotation, so their type comes from the server's declared
# `outputSchema` instead, when the tool's metadata carries one.


def _return_schema(tool: BaseTool) -> dict[str, Any] | None:
    """Return the JSON Schema of the value JS gets from `tool`, if known."""
    mcp_schema = _mcp_artifact_schema(tool)
    if mcp_schema is not None:
        return mcp_schema
    annotation = _return_annotation(tool)
    if tool.response_format == "content_and_artifact":
        annotation = _artifact_annotation(annotation)
    if (
        annotation is inspect.Signature.empty
        or annotation is Any
        or _mentions_tool_envelope(annotation)
    ):
        return None
    try:
        return TypeAdapter(annotation).json_schema(mode="serialization")
    except Exception:  # noqa: BLE001 — schema generation is best-effort
        return None


def _mcp_artifact_schema(tool: BaseTool) -> dict[str, Any] | None:
    """Return the schema of an MCP tool's artifact when its server declares one.

    The declared schema is read from `metadata["mcp"]["tool"]["output_schema"]`.
    A server that declares an `outputSchema` must return matching
    `structuredContent`, which reaches JS as the artifact
    `{structured_content: ...}`.
    """
    if tool.response_format != "content_and_artifact":
        return None
    mcp = (tool.metadata or {}).get("mcp")
    mcp_tool = mcp.get("tool") if isinstance(mcp, dict) else None
    output = mcp_tool.get("output_schema") if isinstance(mcp_tool, dict) else None
    if not isinstance(output, dict):
        return None
    return {
        "type": "object",
        "properties": {
            "structured_content": {k: v for k, v in output.items() if k != "$defs"}
        },
        "required": ["structured_content"],
        "$defs": output.get("$defs") or {},
    }


def _return_annotation(tool: BaseTool) -> Any:
    """Return the tool function's resolved return annotation, if it has one."""
    target = getattr(tool, "func", None) or getattr(tool, "coroutine", None)
    if target is None:
        return inspect.Signature.empty
    annotation = inspect.Signature.empty
    with contextlib.suppress(TypeError, ValueError, NameError):
        signature = inspect.signature(target)
        resolved = get_type_hints(target)
        annotation = resolved.get("return", signature.return_annotation)
    return annotation


def _artifact_annotation(annotation: Any) -> Any:
    """Return `A` from a `tuple[C, A]` annotation, else no annotation.

    A `None` artifact falls back to the content, so an `A` that admits `None`
    has no single type to render.
    """
    args = get_args(annotation)
    if get_origin(annotation) is not tuple or len(args) != 2:  # noqa: PLR2004
        return inspect.Signature.empty
    artifact = args[1]
    admits_none = artifact is None or type(None) in (artifact, *get_args(artifact))
    if admits_none or artifact is ...:
        return inspect.Signature.empty
    return artifact


def _mentions_tool_envelope(annotation: Any) -> bool:
    """Return whether the annotation names a `ToolMessage` or `Command`."""
    origin = get_origin(annotation) or annotation
    if isinstance(origin, type) and issubclass(origin, ToolOutputMixin):
        return True
    return any(_mentions_tool_envelope(arg) for arg in get_args(annotation))


def _json_schema_to_ts(
    prop: dict[str, Any] | bool,  # noqa: FBT001 — a JSON Schema node, not a flag
    refs: Mapping[str, str],
) -> str:
    """Render a JSON Schema node as a TS type; `refs` names each `$ref` target."""
    if not isinstance(prop, dict):
        # Raw JSON Schema allows `true` / `false` as a subschema.
        return "unknown"
    if "$ref" in prop:
        return refs.get(prop["$ref"], "unknown")
    if len(prop.get("allOf", ())) == 1:
        return _json_schema_to_ts(prop["allOf"][0], refs)
    if "const" in prop:
        return json.dumps(prop["const"])
    if "enum" in prop:
        return " | ".join(json.dumps(v) for v in prop["enum"])
    for union in ("anyOf", "oneOf"):
        if union in prop:
            parts = [_json_schema_to_ts(part, refs) for part in prop[union]]
            return " | ".join(dict.fromkeys(parts))
    return _typed_schema_to_ts(prop, refs)


def _typed_schema_to_ts(prop: dict[str, Any], refs: Mapping[str, str]) -> str:
    """Render a JSON Schema node by its `type` keyword."""
    t = prop.get("type")
    if isinstance(t, list):
        # Raw JSON Schema may list several types, as in `["string", "null"]`.
        parts = [_typed_schema_to_ts({**prop, "type": member}, refs) for member in t]
        return " | ".join(dict.fromkeys(parts))
    if t == "string":
        # quickjs_rs marshals `bytes` to and from `Uint8Array`.
        return "Uint8Array" if prop.get("format") == "binary" else "string"
    if t in {"integer", "number"}:
        return "number"
    if t in {"boolean", "null"}:
        return t
    if t == "array":
        items = prop.get("items")
        inner = (
            _json_schema_to_ts(items, refs) if isinstance(items, dict) else "unknown"
        )
        return f"({inner})[]" if " | " in inner else f"{inner}[]"
    if t == "object":
        return _object_schema_to_ts(prop, refs)
    return "unknown"


def _object_schema_to_ts(prop: dict[str, Any], refs: Mapping[str, str]) -> str:
    """Render an inline object, or a `Record` when only values are typed."""
    sub_props = prop.get("properties")
    if isinstance(sub_props, dict) and sub_props:
        required = set(prop.get("required", []))
        fields = [
            f"{k}{'' if k in required else '?'}: {_json_schema_to_ts(v, refs)}"
            for k, v in sub_props.items()
        ]
        return "{ " + "; ".join(fields) + " }"
    values = prop.get("additionalProperties")
    value_type = (
        _json_schema_to_ts(values, refs) if isinstance(values, dict) else "unknown"
    )
    return f"Record<string, {value_type}>"
