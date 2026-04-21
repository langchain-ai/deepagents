"""``REPLMiddleware``: exposes a persistent JavaScript REPL as an agent tool.

State persists across tool calls within a LangGraph thread (each thread
gets its own QuickJS context). See the spec at
``libs/deepagents/deepagents/middleware/JS_EVAL_MIDDLEWARE_SPEC.md`` for
the full design rationale.
"""

# NOTE: Deliberately no ``from __future__ import annotations`` here.
# LangChain's StructuredTool machinery reads ``ToolRuntime`` from tool
# function signatures at tool-build time to discover injected args. With
# the future import on, those annotations are strings and the
# introspection fails silently — the runtime arg drops out of the
# invoke path and the tool call raises ``TypeError: missing 1 required
# positional argument: 'runtime'`` at execution time. Keeping real
# annotations here matches what FilesystemMiddleware does.

import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Annotated, Any

from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.config import get_config
from pydantic import BaseModel, Field

from deepagents_repl._ptc import (
    PTCOption,
    filter_tools_for_ptc,
    render_ptc_prompt,
)
from deepagents_repl._repl import SwarmBinding, _Registry, format_outcome
from deepagents_repl._swarm.executor import SubagentFactory
from deepagents_repl._swarm.types import DEFAULT_CONCURRENCY

logger = logging.getLogger(__name__)

_DEFAULT_MEMORY_LIMIT = 64 * 1024 * 1024
_DEFAULT_TIMEOUT = 5.0
_DEFAULT_MAX_RESULT_CHARS = 4_000
_DEFAULT_TOOL_NAME = "eval"

_LARGE_FILE_RULE_SWARM = (
    "- **Check inputs before processing** — for any file, use `ls` or "
    "`read_file` with offset/limit to understand its size and shape. "
    "If the work involves many independent items, multiple entities, "
    "or data that exceeds a single context, use "
    "`swarm.create`/`swarm.execute` — see \"Parallel fan-out\" below."
)

_LARGE_FILE_RULE_NO_SWARM = (
    "- **Check file size before processing** — before working with any "
    "input file, check its size. If it exceeds ~50,000 characters, "
    "decompose the work into chunks and process them separately."
)

_BASE_PROMPT_TEMPLATE = """\
## TypeScript/JavaScript REPL (`{tool_name}`)

You have access to a sandboxed TypeScript/JavaScript REPL running in an isolated interpreter.
TypeScript syntax (type annotations, interfaces, generics, `as` casts) is supported and stripped at evaluation time.
Variables, functions, and closures persist across calls within the same session.

### Hard rules

{large_file_rule}
- **No network, no imports** — do not attempt `fetch`, `require`, or `import` inside `{tool_name}`. Use your file tools (`read_file`, `grep`, `ls`, etc.) for exploration and `readFile`/`writeFile` for direct REPL file I/O.
- **Cite your sources** — when reporting values from files, include the path and key/index so the user can verify.
- **Use console.log()** for output — it is captured and returned. `console.warn()` and `console.error()` are also available.
- **Reuse state from previous cells** — variables, functions, and results from earlier `{tool_name}` calls persist across calls. Reference them by name in follow-up cells instead of re-embedding data as inline JSON literals.

### First-time usage

```typescript
// Read a file from the agent's virtual filesystem
const raw: string = await readFile("/data.json");
const data = JSON.parse(raw) as {{ n: number }};
console.log(data);

// Write results back
await writeFile("/output.txt", JSON.stringify({{ result: data.n }}));
```

### API Reference — built-in globals

```typescript
/**
 * Read a file from the agent's virtual filesystem. Throws if the file does not exist.
 */
async readFile(path: string): Promise<string>

/**
 * Write a file to the agent's virtual filesystem.
 */
async writeFile(path: string, content: string): Promise<void>
```

### Limitations

- ES2023+ syntax with TypeScript support. No Node.js APIs, no `require`, no `import`.
- Output is truncated beyond a fixed character limit — be selective about what you log.
- Execution timeout per call: {timeout:g} s.
"""


_SWARM_PROMPT_TEMPLATE = """

## Parallel fan-out (`swarm.create` + `swarm.execute` inside `{tool_name}`)

Use `swarm.create` and `swarm.execute` inside `{tool_name}` to dispatch many independent subagent calls in parallel against a JSONL table. Each subagent runs in an isolated context — it sees only the interpolated instruction you write for it.

### When to use swarm

Reach for swarm when any of these apply:
- A dataset has many items needing the same operation (classification, extraction, transformation)
- A collection of entities each needs its own analysis (per-document, per-PR, per-entity)
- The same input benefits from multiple independent perspectives
- The work exceeds what a single subagent's context can hold

Don't use swarm when:
- Fewer than ~5 independent units — use inline tool calls or the `task` tool
- Tasks depend on each other's output
- One end-to-end analysis with no natural decomposition

### Flow

1. **Explore.** Sample the input with your file tools (`read_file` with offset/limit, `grep`, `ls`) outside `{tool_name}` to learn its shape. Finish in 2–3 tool calls.
2. **Create a table.** In `{tool_name}`, call `swarm.create(file, source)` to materialise a JSONL table from a glob, explicit file paths, or inline task rows.
3. **Execute against the table.** Call `swarm.execute(file, options)` with an instruction template. Results stream back as a new column on each row.
4. **Aggregate.** In the same or a follow-up `{tool_name}`, read the table from the backend to combine results programmatically. For qualitative output (summaries, research, narrative), work from the table — don't pull every result string back into the orchestrator's context.

### Hard rules

- **Never read the full input that triggers swarm.** If the data is too large for one context, it reaches subagents via interpolated instruction templates, not through you.
- **Results are final.** Do not dispatch recheck, verify, or cross-check tasks for completed results. Re-dispatching the same data with different ids is still rechecking.
- **One retry for failures, then move on.** Fix the root cause (instruction, schema) and re-dispatch only the failed rows using a filter. Don't retry twice.

### `swarm.create(file, source)`

Materialises a JSONL table at ``file``. Overwrites if it exists.

```typescript
// From a glob pattern
await swarm.create("/analysis.jsonl", {{ glob: "src/**/*.ts" }});

// From explicit file paths
await swarm.create("/analysis.jsonl", {{ filePaths: ["a.ts", "b.ts"] }});

// From inline task rows (each must have id: string)
await swarm.create("/analysis.jsonl", {{
  tasks: lines.map((line, i) => ({{ id: `row-${{i}}`, text: line }}))
}});
```

Glob and filePaths sources produce rows with `{{id, file}}`. Inline tasks can have any shape.

### `swarm.execute(file, options)`

Dispatches subagents against an existing table. Returns a JSON string — use `JSON.parse()`.

```typescript
const summary = JSON.parse(await swarm.execute("/analysis.jsonl", {{
  instruction: "Review this file for security issues.\\n\\nFile: {{file}}",
  column: "review",              // column to write results into (default: "result")
  subagentType: "general-purpose",
  concurrency: 25,
}}));
console.log("Completed:", summary.completed, "Failed:", summary.failed);
```

### Instruction templates

`{{column}}` / `{{dotted.path}}` placeholders interpolate per-row from the table.

```typescript
// Row: {{ id: "utils.ts", file: "src/utils.ts" }}
await swarm.execute("/analysis.jsonl", {{
  instruction: "Analyze {{file}} for code complexity.",
  column: "complexity",
}});
```

### Filtering rows

Use `filter` to dispatch only matching rows; others pass through unchanged.

```typescript
// Only rows where the column doesn't exist yet
await swarm.execute("/analysis.jsonl", {{
  instruction: "...",
  filter: {{ column: "review", exists: false }},
}});

// Retry failed rows
await swarm.execute("/analysis.jsonl", {{
  instruction: "...",
  filter: {{ column: "review", equals: null }},
}});

// Combine conditions
await swarm.execute("/analysis.jsonl", {{
  instruction: "...",
  filter: {{ and: [
    {{ column: "status", equals: "pending" }},
    {{ column: "priority", in: ["high", "critical"] }},
  ]}},
}});
```

Operators: `equals`, `notEquals`, `in`, `exists` (boolean), `and`, `or`.

### Multi-pass enrichment

Run multiple `swarm.execute` calls against the same table, each writing a different column. Later passes can reference earlier columns.

```typescript
await swarm.create("/docs.jsonl", {{ glob: "docs/**/*.md" }});

// Pass 1: extract summary
await swarm.execute("/docs.jsonl", {{
  instruction: "Summarize this document.\\n\\nFile: {{file}}",
  column: "summary",
}});

// Pass 2: classify based on summary
await swarm.execute("/docs.jsonl", {{
  instruction: "Classify: {{file}}\\nSummary: {{summary}}",
  column: "category",
  responseSchema: {{
    type: "object",
    properties: {{ category: {{ type: "string" }} }},
    required: ["category"],
  }},
}});
```

### Structured output (`responseSchema`)

Use `responseSchema` when results will be aggregated programmatically. The column value is the parsed JSON, not a string.

```typescript
await swarm.execute("/analysis.jsonl", {{
  instruction: "Classify the complexity of {{file}}.",
  column: "metrics",
  responseSchema: {{
    type: "object",
    properties: {{
      complexity: {{ type: "string", enum: ["low", "medium", "high"] }},
      reason: {{ type: "string" }},
    }},
    required: ["complexity", "reason"],
  }},
}});
```

Schema rules (enforced at dispatch time — violations throw before any subagent runs):
- Top-level `type` must be `"object"`. Wrap arrays under a named field.
- `properties` must be defined with at least one explicit field.

### API Reference

```typescript
async function swarm.create(file: string, source: {{
  glob?: string | string[];
  filePaths?: string[];
  tasks?: Array<{{id: string, [key: string]: any}}>;
}}): Promise<void>

async function swarm.execute(file: string, options: {{
  instruction: string;
  column?: string;               // default: "result"
  filter?: SwarmFilter;
  subagentType?: string;         // default: "general-purpose"
  responseSchema?: object;       // top-level must be type: "object"
  concurrency?: number;          // default: {default_concurrency}
}}): Promise<string>  // JSON string of SwarmSummary
```

Available subagent types: {available_subagents}
"""


class EvalSchema(BaseModel):
    """Input schema for the `eval` tool."""

    code: str = Field(
        description=(
            "JavaScript expression or statement(s) to evaluate. "
            "State persists across calls. No fs/network/real-clock access."
        ),
    )


def _resolve_thread_id(fallback: str) -> str:
    """Extract ``thread_id`` from langgraph config or use ``fallback``.

    The fallback is a middleware-instance-scoped id: when the caller
    didn't configure a ``thread_id`` (common for ad-hoc
    ``agent.invoke(...)`` in tests or single-shot scripts), we still need
    all resolver calls within one REPLMiddleware lifetime to return the
    same id — otherwise ``wrap_model_call`` installs tools on one REPL
    and the eval tool looks up a different one, and the model sees
    ``ReferenceError: tools is not defined``.
    """
    try:
        config = get_config()
    except RuntimeError:
        # Not running inside a Runnable — test / bare-call path.
        return fallback
    thread_id = config.get("configurable", {}).get("thread_id") if config else None
    if thread_id is not None:
        return str(thread_id)
    return fallback


class REPLMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware exposing a persistent JS REPL to the agent.

    One ``quickjs_rs.Runtime`` is created lazily per middleware instance
    and shared across threads; each LangGraph thread gets its own
    ``Context`` so globals from one conversation cannot leak into another.

    Args:
        memory_limit: Bytes the QuickJS heap may use. Shared across all
            contexts under the same Runtime. Default 64 MiB.
        timeout: Per-call wall-clock timeout in seconds. Applied to every
            ``eval`` on every context. Default 5.
        tool_name: Name of the tool exposed to the model. Default ``eval``.
        max_result_chars: Result and stdout blocks are independently
            truncated to this many characters before being sent back to
            the model. Default 4000.
        capture_console: If ``True``, install a ``console`` object that
            buffers ``console.log/warn/error`` calls and emits them in
            ``<stdout>`` blocks alongside the result. Default ``True``.
        ptc: Programmatic tool calling — expose agent tools inside the
            REPL as ``tools.<camelCase>(input) => Promise<string>``. One
            ``eval`` call can then orchestrate many tool calls (loops,
            ``Promise.all``, conditional branching). Accepts:

            - ``False`` (default) — disabled.
            - ``True`` — expose every agent tool except the REPL itself.
            - ``list[str]`` — expose only the listed tools.
            - ``{"include": [...]}`` — equivalent to ``list[str]``.
            - ``{"exclude": [...]}`` — expose all except the listed tools.

            The REPL's own tool is always excluded; a model asking for
            ``tools.eval("...")`` would recurse pointlessly.

    Example:
        ```python
        from deepagents import create_deep_agent
        from deepagents_repl import REPLMiddleware

        agent = create_deep_agent(
            model="claude-sonnet-4-6",
            middleware=[REPLMiddleware()],
        )
        ```
    """

    def __init__(
        self,
        *,
        memory_limit: int = _DEFAULT_MEMORY_LIMIT,
        timeout: float = _DEFAULT_TIMEOUT,
        tool_name: str = _DEFAULT_TOOL_NAME,
        max_result_chars: int = _DEFAULT_MAX_RESULT_CHARS,
        capture_console: bool = True,
        ptc: PTCOption = False,
        backend: BackendProtocol | None = None,
        subagents: Sequence[SubAgent | CompiledSubAgent] | None = None,
        subagent_factories: Mapping[str, SubagentFactory] | None = None,
        swarm_task_timeout: float | None = None,
    ) -> None:
        super().__init__()
        self._memory_limit = memory_limit
        self._timeout = timeout
        self._tool_name = tool_name
        self._max_result_chars = max_result_chars
        self._capture_console = capture_console
        self._ptc = ptc
        self._backend = backend

        if subagents and backend is None:
            msg = (
                "REPLMiddleware: `subagents` requires `backend` — swarm needs "
                "somewhere to persist its JSONL table. Pass a BackendProtocol "
                "instance (e.g. `StateBackend()`) alongside `subagents`."
            )
            raise ValueError(msg)

        swarm_binding: SwarmBinding | None = None
        subagent_descriptions: list[dict[str, str]] = []
        if subagents and backend is not None:
            subagent_graphs: dict[str, Runnable] = {}
            for spec in subagents:
                if "runnable" not in spec:
                    msg = (
                        f"REPLMiddleware: subagent '{spec['name']}' lacks a "
                        "pre-compiled `runnable`. Compile declarative "
                        "SubAgent specs via `deepagents.graph.build_subagents` "
                        "before passing them here."
                    )
                    raise ValueError(msg)
                subagent_graphs[spec["name"]] = spec["runnable"]
                subagent_descriptions.append(
                    {"name": spec["name"], "description": spec["description"]}
                )
            swarm_binding = SwarmBinding(
                backend=backend,
                subagent_graphs=subagent_graphs,
                subagent_factories=subagent_factories,
                task_timeout_seconds=swarm_task_timeout,
            )
        self._swarm_binding = swarm_binding
        self._swarm_subagent_descriptions = subagent_descriptions

        self._registry = _Registry(
            memory_limit=memory_limit,
            timeout=timeout,
            capture_console=capture_console,
            swarm_binding=swarm_binding,
        )
        has_swarm = swarm_binding is not None
        base_prompt = _BASE_PROMPT_TEMPLATE.format(
            tool_name=tool_name,
            timeout=timeout,
            large_file_rule=(
                _LARGE_FILE_RULE_SWARM if has_swarm else _LARGE_FILE_RULE_NO_SWARM
            ),
        )
        if swarm_binding is not None:
            available = ", ".join(s["name"] for s in subagent_descriptions)
            base_prompt += _SWARM_PROMPT_TEMPLATE.format(
                tool_name=tool_name,
                available_subagents=available,
                default_concurrency=DEFAULT_CONCURRENCY,
            )
        self._base_system_prompt = base_prompt
        # Backwards-compatible alias used in tests / external introspection.
        self.system_prompt = self._base_system_prompt
        self._ptc_prompt_cache: tuple[frozenset[str], str] | None = None
        # Stable fallback thread id — used when ``thread_id`` isn't in
        # langgraph config. Must be instance-scoped so ``wrap_model_call``
        # and ``eval`` invocations within one conversation resolve to the
        # same REPL; otherwise the PTC install happens on one REPL and the
        # eval runs on another (and sees ``tools`` undefined).
        self._fallback_thread_id = f"session_{uuid.uuid4().hex[:8]}"
        self.tools: list[BaseTool] = [self._build_tool()]

    def _build_tool(self) -> BaseTool:
        tool_name = self._tool_name
        registry = self._registry
        max_chars = self._max_result_chars
        fallback_id = self._fallback_thread_id

        def _run(outcome_fn: Any, code: str, tool_call_id: str) -> ToolMessage:
            content = format_outcome(outcome_fn(code), max_result_chars=max_chars)
            return ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)

        def sync_eval(
            runtime: ToolRuntime[None, Any],
            code: Annotated[
                str,
                "JavaScript expression or statement(s) to evaluate in the persistent REPL.",
            ],
        ) -> ToolMessage:
            repl = registry.get(_resolve_thread_id(fallback_id))
            # The sync path doesn't support PTC (host-fn bridges are
            # async); set_outer_runtime is a no-op here.
            repl.set_outer_runtime(runtime)
            try:
                return _run(repl.eval_sync, code, runtime.tool_call_id)
            finally:
                repl.set_outer_runtime(None)

        backend = self._backend

        async def async_eval(
            runtime: ToolRuntime[None, Any],
            code: Annotated[
                str,
                "JavaScript expression or statement(s) to evaluate in the persistent REPL.",
            ],
        ) -> ToolMessage:
            repl = registry.get(_resolve_thread_id(fallback_id))
            # Capture the outer runtime so PTC bridges can forward
            # state/store/context into tool calls during this eval. Clear
            # after so a stale runtime can't bleed into a later call on
            # the same thread — the lock serialises, but the closure
            # would otherwise retain a reference past the call.
            repl.set_outer_runtime(runtime)
            try:
                content = format_outcome(
                    await repl.eval_async(code),
                    max_result_chars=max_chars,
                )
                # Flush any table writes the eval accumulated via
                # `swarm.create` / `swarm.execute`. Done outside the
                # eval so a crash mid-flush doesn't leave the REPL in
                # an inconsistent state relative to its pending buffer.
                if backend is not None:
                    for path, pending_content in repl._drain_pending_writes():
                        await backend.awrite(path, pending_content)
            finally:
                repl.set_outer_runtime(None)
            return ToolMessage(content=content, tool_call_id=runtime.tool_call_id, name=tool_name)

        return StructuredTool.from_function(
            name=tool_name,
            description=(
                "Evaluate TypeScript/JavaScript code in a sandboxed REPL. "
                "State persists across calls.\n"
                "Use readFile(path) and writeFile(path, content) for file access.\n"
                "Use swarm.create(file, source) and swarm.execute(file, options) "
                "for parallel fan-out.\n"
                "Use console.log() for output. Returns the result of the last expression."
            ),
            func=sync_eval,
            coroutine=async_eval,
            infer_schema=False,
            args_schema=EvalSchema,
        )

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Inject the REPL's system-prompt snippet on every model call."""
        prompt = self._prepare_for_call(request)
        return handler(
            request.override(system_message=self._extend(request.system_message, prompt)),
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """(async) Inject the REPL's system-prompt snippet on every model call."""
        prompt = self._prepare_for_call(request)
        return await handler(
            request.override(system_message=self._extend(request.system_message, prompt)),
        )

    def _prepare_for_call(self, request: ModelRequest[ContextT]) -> str:
        """Install PTC bindings for this turn and return the full system-prompt addendum.

        Called from both sync and async model-call wrappers. Reads the
        live tool list off the request (middlewares upstream may have
        filtered it), decides what PTC exposes this turn, registers any
        missing host-function bridges on the current thread's REPL, and
        rebuilds ``globalThis.tools`` if the exposed name set changed.
        """
        if self._ptc is False:
            return self._base_system_prompt
        request_tools: list[BaseTool] = list(getattr(request, "tools", []) or [])
        exposed = filter_tools_for_ptc(
            request_tools,
            self._ptc,
            self_tool_name=self._tool_name,
        )
        # Install on the current thread's REPL. If the thread hasn't
        # evaluated anything yet, this creates the context lazily — which
        # is fine: PTC bindings must be in place *before* the first eval
        # that references them, and the next eval on this thread is the
        # earliest that could matter.
        thread_id = _resolve_thread_id(self._fallback_thread_id)
        repl = self._registry.get(thread_id)
        repl.install_tools(exposed)
        # Rendering the TS-ish signature block is cheap but not free;
        # cache by the set of exposed names. The set doesn't encode tool
        # *identity* — if a tool keeps its name but its schema changes
        # between turns, the cached prompt staleness is on the caller.
        # Same tradeoff the TS package accepts; see the module docstring.
        exposed_names = frozenset(t.name for t in exposed)
        if self._ptc_prompt_cache is None or self._ptc_prompt_cache[0] != exposed_names:
            self._ptc_prompt_cache = (exposed_names, render_ptc_prompt(exposed))
        return self._base_system_prompt + self._ptc_prompt_cache[1]

    def _extend(self, system_message: SystemMessage | None, prompt: str) -> SystemMessage:
        return append_to_system_message(system_message, prompt)

    def __del__(self) -> None:
        # Best-effort cleanup. If the Runtime was never built (no tool
        # calls happened) this is a no-op. Wrapped in a bare except because
        # __del__ must not raise during interpreter shutdown, when wasmtime
        # or its dependencies may already be half-unloaded.
        try:
            self._registry.close()
        except Exception:  # noqa: BLE001 — GC path, never raise
            pass
