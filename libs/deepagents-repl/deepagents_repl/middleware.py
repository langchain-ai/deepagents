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
import math
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any

from deepagents.middleware._utils import append_to_system_message
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

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from deepagents.middleware.skills import SkillMetadata

from deepagents_repl._ptc import (
    PTCOption,
    filter_tools_for_ptc,
    render_ptc_prompt,
)
from deepagents_repl._repl import _Registry, format_outcome
from deepagents_repl._skills import scan_skill_references

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
    "- Timeout: {timeout} per call. Memory: {memory_limit_mb} MB total.\n"
    "- `console.log` output is captured and returned alongside the result."
)


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
            ``eval`` on every context. Default 5. Pass ``None`` to
            disable the deadline entirely — useful for REPL code that
            awaits tool calls (e.g. ``tools.task``) whose latency is
            dominated by model round-trips, not JS work. ``None`` is
            coerced to ``math.inf`` before reaching the QuickJS
            context, which treats it as "no deadline."
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
        timeout: float | None = _DEFAULT_TIMEOUT,
        tool_name: str = _DEFAULT_TOOL_NAME,
        max_result_chars: int = _DEFAULT_MAX_RESULT_CHARS,
        capture_console: bool = True,
        ptc: PTCOption = False,
        skills_backend: "BackendProtocol | None" = None,
    ) -> None:
        """See the class docstring for parameter details.

        Args:
            skills_backend: Optional ``BackendProtocol`` the REPL reads
                skill source files from. When set and a paired
                ``SkillsMiddleware`` populates ``skills_metadata`` in
                state, skills with a ``module`` frontmatter key become
                dynamic-importable from the REPL as
                ``await import("@/skills/<name>")``. When ``None``,
                skill modules are not installed (``import(...)`` fails
                at the resolver). This must be the same backend
                ``SkillsMiddleware`` uses — the REPL treats skill file
                paths relative to SKILL.md as coming from the same
                store.
        """
        super().__init__()
        # ``None`` means "no deadline." Coerce once here so every downstream
        # layer (Registry, _ThreadREPL, quickjs_rs.Context) keeps the strict
        # ``float`` contract it's written against.
        effective_timeout = math.inf if timeout is None else timeout
        self._memory_limit = memory_limit
        self._timeout = effective_timeout
        self._tool_name = tool_name
        self._max_result_chars = max_result_chars
        self._capture_console = capture_console
        self._ptc = ptc
        self._skills_backend = skills_backend
        self._registry = _Registry(
            memory_limit=memory_limit,
            timeout=effective_timeout,
            capture_console=capture_console,
        )
        timeout_desc = (
            "disabled"
            if not math.isfinite(effective_timeout)
            else f"{effective_timeout}s"
        )
        self._base_system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            tool_name=tool_name,
            timeout=timeout_desc,
            memory_limit_mb=memory_limit // (1024 * 1024),
        )
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
        middleware = self

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
            # Install any referenced skills before eval. If install
            # fails (bad skill, unknown name, backend error) we short-
            # circuit with a formatted error so the model sees a clean
            # "skill unavailable" message instead of the raw
            # ReferenceError quickjs-rs's resolver would produce.
            install_error = await middleware._ensure_skills_for_eval(runtime, code, repl)
            if install_error is not None:
                return ToolMessage(
                    content=install_error,
                    tool_call_id=runtime.tool_call_id,
                    name=tool_name,
                )
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

    async def _ensure_skills_for_eval(
        self,
        runtime: ToolRuntime[None, Any],
        code: str,
        repl: Any,  # _ThreadREPL — Any to avoid a cross-module type cycle
    ) -> str | None:
        """Install any ``@/skills/<name>`` specifiers the code references.

        Returns a preformatted error string if install failed for any
        referenced skill; ``None`` if everything is ready to eval.
        Skill install is a no-op when ``skills_backend`` is ``None`` or
        no literal specifiers appear in the source.
        """
        if self._skills_backend is None:
            return None
        referenced = scan_skill_references(code)
        if not referenced:
            return None
        metadata_list = runtime.state.get("skills_metadata", []) if runtime.state else []
        metadata: dict[str, SkillMetadata] = {m["name"]: m for m in metadata_list}
        errors = await self._registry.aensure_skills_installed(
            referenced,
            metadata,
            self._skills_backend,
            repl._ctx,
        )
        if not errors:
            return None
        # Each error is a SkillLoadError subclass with a readable
        # message. Surface all of them joined so the model sees every
        # broken skill, not just the first.
        rendered = "; ".join(str(e) for e in errors)
        return f'<error type="SkillNotAvailable">{rendered}</error>'

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
