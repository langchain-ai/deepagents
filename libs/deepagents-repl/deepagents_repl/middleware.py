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
from langchain_core.tools import StructuredTool
from langgraph.config import get_config
from pydantic import BaseModel, Field

from deepagents_repl._ptc import (
    PTCOption,
    filter_tools_for_ptc,
    render_ptc_prompt,
)
from deepagents_repl._repl import SwarmBinding, _Registry, format_outcome
from deepagents_repl._swarm import DEFAULT_CONCURRENCY, compile_subagents
from deepagents_repl._swarm.executor import SubagentFactory

logger = logging.getLogger(__name__)

_DEFAULT_MEMORY_LIMIT = 64 * 1024 * 1024
_DEFAULT_TIMEOUT = 5.0
_DEFAULT_MAX_RESULT_CHARS = 4_000
_DEFAULT_TOOL_NAME = "eval"

_SYSTEM_PROMPT_TEMPLATE = (
    "An `{tool_name}` tool is available. It runs JavaScript in a persistent "
    "REPL backed by QuickJS.\n"
    "- State (variables, functions) persists across calls within this conversation.\n"
    "- Top-level `await` works; Promises resolve before the call returns.\n"
    "- Sandboxed: no filesystem, no stdlib, no network, no real clock, no `fetch`, no `require`.\n"
    "- Timeout: {timeout}s per call. Memory: {memory_limit_mb} MB total.\n"
    "- `console.log` output is captured and returned alongside the result."
)


_SWARM_PROMPT_TEMPLATE = """

## Parallel fan-out (`swarm()` inside `{tool_name}`)

Use `swarm()` inside `{tool_name}` to dispatch many independent subagent calls in parallel and aggregate the results. Each subagent runs in an isolated context — it sees only the description you write for it.

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

1. **Explore.** Sample — don't read in full. Use your file tools (`read_file` with offset/limit, `grep`, `ls`) outside `{tool_name}` to learn the input's shape. Finish in 2–3 tool calls.
2. **Dispatch.** In `{tool_name}`, build task descriptions and call `swarm()`. Each description should tell its subagent exactly which slice to read.
3. **Aggregate.** In the same or a follow-up `{tool_name}` call, combine results programmatically from `summary.results`. For qualitative output (summaries, research, narrative), read `resultsDir + "/results.jsonl"` with your file tools instead of pulling every result string into the orchestrator's context.

### Hard rules

- **Never read the full input that triggers swarm.** If the data is too large for one context, it reaches subagents via chunked descriptions or via per-subagent file reads, not through you.
- **Results are final.** Do not dispatch recheck, verify, or cross-check tasks for completed results. Re-dispatching the same data with different ids is still rechecking.
- **One retry for failures, then move on.** Fix the root cause (description, slice bounds) and re-dispatch only the failed ids. Don't retry twice.

### Dispatch example

```typescript
// After inspecting /data.txt you know it has ~400 lines.
const chunkSize = 50;
const totalLines = 400;

const tasks = [];
for (let start = 0; start < totalLines; start += chunkSize) {{
  tasks.push({{
    id: `chunk_${{start}}`,
    description:
      `Read /data.txt lines ${{start}}-${{start + chunkSize - 1}} ` +
      `(use read_file with offset=${{start}} and limit=${{chunkSize}}). ` +
      `Count occurrences of each label.\\n\\n` +
      `Respond with ONLY a raw JSON object — no markdown fences, no ` +
      `explanation, no other text.\\nOutput schema: {{ "label": count }}`,
  }});
}}

const summary = await swarm({{
  tasks,
  concurrency: Math.min(25, tasks.length),
}});
console.log("completed:", summary.completed, "failed:", summary.failed);

// Aggregate in-script from summary.results — no backend round-trip needed.
const merged = {{}};
for (const r of summary.results) {{
  if (r.status === "completed") {{
    try {{
      const partial = JSON.parse(r.result);
      for (const k of Object.keys(partial)) {{
        merged[k] = (merged[k] || 0) + partial[k];
      }}
    }} catch (e) {{ /* skip unparseable */ }}
  }}
}}
console.log(JSON.stringify(merged));
```

### Writing task descriptions

Each subagent sees only its description. A good description lets the subagent work mechanically, with no judgment required. Include:

- What the input is and how it's structured (delimiters, format, encoding)
- What the subagent should produce (format, fields, allowed values)
- The rules — including edge cases and examples you found during exploration
- The actual slice of data for this task (or the file path + range to read)

Anything you discovered during exploration must be written into every description that needs it. Subagents cannot see your notes.

When results will be aggregated programmatically, prefer `responseSchema` over prompt-only JSON directives (see next section). For qualitative output (summaries, research, narrative), free-form text aggregates better when read from `results.jsonl`.

### Structured output (`responseSchema`)

Use `responseSchema` when results will be aggregated programmatically. It enforces the schema at the model API level — stricter than asking for JSON in prose. The subagent remains fully agentic (tools, reasoning) — only its final response is constrained.

```typescript
{{
  id: "t1",
  description: "...",
  responseSchema: {{
    type: "object",
    properties: {{
      results: {{
        type: "array",
        items: {{
          type: "object",
          properties: {{
            id:    {{ type: "string" }},
            label: {{ type: "string", enum: ["a", "b", "c"] }}
          }},
          required: ["id", "label"]
        }}
      }}
    }},
    required: ["results"]
  }}
}}
```

Schema tips:
- `enum` on categorical fields prevents label drift across subagents.
- `description` on properties — models read them during generation.
- `minItems` / `maxItems` on arrays — ensures the expected count.

Schema rules (enforced at dispatch time — violations throw before any subagent runs):
- Top-level `type` must be `"object"`. Wrap arrays under a `results` field.
- `properties` must be defined with at least one explicit field. Open schemas (`additionalProperties` alone, no `properties`) are rejected.
- Declare every field you expect. If you don't know keys ahead of time, use `results: {{ type: "array", items: {{...}} }}` instead of an open object.

### Chunk sizing and concurrency

Aim for 10–25 tasks per swarm call. Fewer, and parallelism doesn't pay for overhead. More, and a bad description affects many tasks at once.

Per-task sizing depends on item size:
- Short items (labels, one-line entries): 30–60 per task
- Medium items (reviews, paragraphs): 10–20 per task
- Long items (documents, articles): 1–5 per task, or one-per-task

For runs with >10 tasks, set `concurrency` explicitly. Good rule: `Math.min(25, tasks.length)`.

### Decomposition patterns

- **One-per-item** — one task per discrete unit (file, document, entity). Use when items are naturally discrete and each needs its own analysis.
- **Flat fan-out** — split a collection into equal chunks; all tasks have the same shape. Use when applying the same operation to many items.
- **Dimensional** — multiple tasks examine the same input from different angles. Use for multi-criteria evaluation (code review, red-team analysis).

### API Reference — `swarm()`

```typescript
/**
 * Dispatch tasks to subagents in parallel.
 */
async function swarm(input: {{
  // Pre-built tasks form
  tasks?: Array<{{
    id: string;              // unique task identifier
    description: string;     // complete, self-contained prompt for the subagent
    subagentType?: string;   // which subagent to use (default: "general-purpose")
    responseSchema?: object; // JSON Schema for structured output (top-level must be `type: "object"`)
  }}>;
  // Virtual-table form (alternative to `tasks`): one task synthesised per file.
  glob?: string | string[]; // glob pattern(s) to match files on the VFS
  filePaths?: string[];     // explicit file paths
  instruction?: string;     // shared instruction prepended to each file task
  subagentType?: string;    // subagent type for all synthesised tasks
  concurrency?: number;     // max concurrent subagents (default: {default_concurrency})
}}): Promise<{{
  total: number;
  completed: number;
  failed: number;
  resultsDir: string;        // VFS path — results.jsonl is also persisted here
  results: Array<{{
    id: string;
    subagentType: string;
    status: "completed" | "failed";
    result?: string;         // present when status is "completed"
    error?: string;          // present when status is "failed"
  }}>;
  failedTasks: Array<{{ id: string; error: string }}>;
}}>
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
        # Swarm is active only when both a backend and at least one
        # subagent are configured. Either alone is a user error — swarm
        # needs a VFS to persist results.jsonl and at least one subagent
        # to dispatch to — but we treat "neither configured" as the
        # common case (swarm disabled) and don't raise.
        if subagents and backend is not None:
            subagent_graphs, subagent_descriptions = compile_subagents(subagents)
            swarm_binding: SwarmBinding | None = SwarmBinding(
                backend=backend,
                subagent_graphs=subagent_graphs,
                task_timeout_seconds=swarm_task_timeout,
                subagent_factories=subagent_factories,
            )
            self._swarm_subagent_descriptions = subagent_descriptions
        elif subagents and backend is None:
            msg = (
                "REPLMiddleware: `subagents` requires `backend` (results.jsonl must "
                "be written somewhere). Pass a BackendProtocol instance (e.g. "
                "`StateBackend()`) alongside `subagents`."
            )
            raise ValueError(msg)
        else:
            swarm_binding = None
            self._swarm_subagent_descriptions = []
        self._registry = _Registry(
            memory_limit=memory_limit,
            timeout=timeout,
            capture_console=capture_console,
            swarm_binding=swarm_binding,
        )
        base_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            tool_name=tool_name,
            timeout=timeout,
            memory_limit_mb=memory_limit // (1024 * 1024),
        )
        if swarm_binding is not None:
            available = ", ".join(s["name"] for s in self._swarm_subagent_descriptions)
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
            finally:
                repl.set_outer_runtime(None)
            return ToolMessage(content=content, tool_call_id=runtime.tool_call_id, name=tool_name)

        return StructuredTool.from_function(
            name=tool_name,
            description=(
                "Execute JavaScript in a persistent sandboxed REPL. "
                "Variables and functions defined in one call are visible to "
                "subsequent calls in this conversation. No filesystem, "
                "network, or real clock. Synchronous only — top-level `await` "
                "will not resolve."
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
