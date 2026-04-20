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
from collections.abc import Awaitable, Callable, Sequence
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

### API Reference — `swarm()` global

Fan out tasks to subagents and collect all results. Returns a summary object.

```typescript
// Virtual-table form — one task per file resolved from the VFS
const summary = await swarm({{
  glob: "feedback/*.txt",         // glob pattern(s), or filePaths: ["/a.txt", "/b.txt"]
  instruction: "Classify as bug, feature, or praise. Return JSON: {{category, confidence}}.",
  subagentType: "{default_subagent}",        // optional
  concurrency: 5,                 // optional, default {default_concurrency}
}});

// Pre-built tasks form — explicit task list
const summary = await swarm({{
  tasks: [
    {{ id: "q1", description: "Summarize /reports/q1.txt in one sentence." }},
    {{ id: "q2", description: "Summarize /reports/q2.txt in one sentence." }},
  ],
}});

// summary shape:
//   {{
//     total, completed, failed,
//     resultsDir,      // VFS path — results.jsonl is also written here for durability
//     results,         // SwarmTaskResult[] — full per-task outputs, use this to aggregate
//     failedTasks,     // [{{ id, error }}] for quick failure inspection
//   }}
// Each SwarmTaskResult: {{ id, subagentType, status: "completed"|"failed", result?, error? }}
```

Available subagent types: {available_subagents}

Use `swarm()` for large batches; it handles concurrency limits, timeouts, and result persistence.
Aggregate per-task outputs from `summary.results` in the same `js_eval` call."""


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
            default = self._swarm_subagent_descriptions[0]["name"]
            base_prompt += _SWARM_PROMPT_TEMPLATE.format(
                available_subagents=available,
                default_subagent=default,
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
