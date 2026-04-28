"""``REPLMiddleware``: exposes a persistent JavaScript REPL as an agent tool.

State persists across tool calls within a LangGraph thread (each thread
gets its own QuickJS context).
"""

import contextlib
import logging
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
from langgraph.types import Command
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from deepagents.middleware.skills import SkillMetadata

from langchain_quickjs._format import format_outcome
from langchain_quickjs._prompt import render_repl_system_prompt
from langchain_quickjs._ptc import (
    PTCOption,
    filter_tools_for_ptc,
    render_ptc_prompt,
)
from langchain_quickjs._repl import _Registry

logger = logging.getLogger(__name__)

_DEFAULT_MEMORY_LIMIT = 64 * 1024 * 1024
_DEFAULT_TIMEOUT = 5.0
_DEFAULT_MAX_RESULT_CHARS = 4_000
_DEFAULT_TOOL_NAME = "eval"
_EvalToolResult = ToolMessage | list[Command | ToolMessage]


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
        skills_backend: Optional ``BackendProtocol`` the REPL reads skill
            source files from. When set and a paired
            ``SkillsMiddleware`` populates ``skills_metadata`` in state,
            skills with a ``module`` frontmatter key become dynamic-
            importable from the REPL as ``await import("@/skills/<name>")``.
            When ``None``, skill modules are not installed
            (``import(...)`` fails at the resolver). This must be the
            same backend ``SkillsMiddleware`` uses.
        ptc: Programmatic tool calling — expose agent tools inside the
            REPL as ``tools.<camelCase>(input) => Promise<string>``. One
            ``eval`` call can then orchestrate many tool calls (loops,
            ``Promise.all``, conditional branching). Accepts:

            - ``None`` (default) — disabled.
            - ``list[str | BaseTool]`` — allowlist entries may be:
              - ``str`` tool names, matched against the agent's toolset.
              - ``BaseTool`` instances, exposed directly even if not on
                the agent's tool list.

            Mixed lists are supported. Explicit ``BaseTool`` entries are
            considered first; then name-matched agent tools are added.
            Duplicate names are deduplicated.

            !!! warning
                PTC calls currently execute through the REPL bridge and
                do **not** go through the normal `ToolNode` path. As a
                result, `interrupt_on` / HITL approval workflows are not
                enforced per PTC-invoked tool call.

            The REPL's own tool is always excluded; a model asking for
            ``tools.eval("...")`` would recurse pointlessly.
        idle_ttl_sec: Close a LangGraph thread's Runtime after this many
            seconds of inactivity. Eviction is lazy: the next ``get()``
            on any thread scans and closes stale slots. On return, the
            user's ``globalThis`` scratchpad is reset; referenced skills
            are reloaded from the backend on demand.
        max_active_threads: Optional hard cap on concurrent slots. When
            exceeded, least-recently-used slots are evicted regardless
            of TTL. ``None`` (default) = TTL only.

    Example:
        ```python
        from deepagents import create_deep_agent
        from langchain_quickjs import REPLMiddleware

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
        ptc: PTCOption | None = None,
        skills_backend: "BackendProtocol | None" = None,
        idle_ttl_sec: float = 3600.0,
        max_active_threads: int | None = None,
    ) -> None:
        """Initialize REPL middleware state and build the exposed eval tool."""
        super().__init__()
        self._memory_limit = memory_limit
        self._timeout = timeout
        self._tool_name = tool_name
        self._max_result_chars = max_result_chars
        self._capture_console = capture_console
        self._ptc = ptc
        self._skills_backend = skills_backend
        self._registry = _Registry(
            memory_limit=memory_limit,
            timeout=timeout,
            capture_console=capture_console,
            idle_ttl_sec=idle_ttl_sec,
            max_active_threads=max_active_threads,
        )
        self._base_system_prompt = render_repl_system_prompt(
            tool_name=tool_name,
            timeout=timeout,
            memory_limit_mb=memory_limit // (1024 * 1024),
        )
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

        def _run(
            outcome_fn: Any, code: str, tool_call_id: str | None
        ) -> _EvalToolResult:
            outcome = outcome_fn(code)
            message = ToolMessage(
                content=format_outcome(outcome, max_result_chars=max_chars),
                tool_call_id=tool_call_id,
                name=tool_name,
            )
            if outcome.commands:
                return [*outcome.commands, message]
            return message

        code_doc = (
            "JavaScript expression or statement(s) to evaluate in the persistent REPL."
        )

        def sync_eval(
            runtime: ToolRuntime[None, Any],
            code: Annotated[str, code_doc],
        ) -> _EvalToolResult:
            repl = registry.get(_resolve_thread_id(fallback_id))
            skills = middleware._skills_for_eval(runtime)
            # The sync path doesn't support PTC (host-fn bridges are
            # async); set_outer_runtime is a no-op here.
            repl.set_outer_runtime(runtime)
            try:
                return _run(
                    lambda c: repl.eval_sync(
                        c,
                        skills=skills,
                        skills_backend=middleware._skills_backend,
                    ),
                    code,
                    runtime.tool_call_id,
                )
            finally:
                repl.set_outer_runtime(None)

        async def async_eval(
            runtime: ToolRuntime[None, Any],
            code: Annotated[str, code_doc],
        ) -> _EvalToolResult:
            repl = registry.get(_resolve_thread_id(fallback_id))
            skills = middleware._skills_for_eval(runtime)
            # Capture the outer runtime so PTC bridges can forward
            # state/store/context into tool calls during this eval. Clear
            # after so a stale runtime can't bleed into a later call on
            # the same thread — the lock serialises, but the closure
            # would otherwise retain a reference past the call.
            repl.set_outer_runtime(runtime)
            try:
                outcome = await repl.eval_async(
                    code,
                    skills=skills,
                    skills_backend=middleware._skills_backend,
                )
            finally:
                repl.set_outer_runtime(None)
            message = ToolMessage(
                content=format_outcome(outcome, max_result_chars=max_chars),
                tool_call_id=runtime.tool_call_id,
                name=tool_name,
            )
            if outcome.commands:
                return [*outcome.commands, message]
            return message

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

    def _skills_for_eval(
        self,
        runtime: ToolRuntime[None, Any],
    ) -> dict[str, "SkillMetadata"] | None:
        """Return per-eval skill metadata map."""
        if self._skills_backend is None:
            return None
        metadata_list = (
            runtime.state.get("skills_metadata", []) if runtime.state else []
        )
        return {m["name"]: m for m in metadata_list}

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Inject the REPL's system-prompt snippet on every model call."""
        prompt = self._prepare_for_call(request)
        return handler(
            request.override(
                system_message=self._extend(request.system_message, prompt)
            ),
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        """(async) Inject the REPL's system-prompt snippet on every model call."""
        prompt = self._prepare_for_call(request)
        return await handler(
            request.override(
                system_message=self._extend(request.system_message, prompt)
            ),
        )

    def _prepare_for_call(self, request: ModelRequest[ContextT]) -> str:
        """Install PTC bindings for this turn and return the system-prompt addendum.

        Called from both sync and async model-call wrappers. Reads the
        live tool list off the request (middlewares upstream may have
        filtered it), decides what PTC exposes this turn, registers any
        missing host-function bridges on the current thread's REPL, and
        rebuilds ``globalThis.tools`` if the exposed name set changed.
        """
        if self._ptc is None:
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

    def _extend(
        self, system_message: SystemMessage | None, prompt: str
    ) -> SystemMessage:
        return append_to_system_message(system_message, prompt)

    def __del__(self) -> None:
        """Best-effort Runtime cleanup on GC; never raises at shutdown."""
        # Wrapped in ``contextlib.suppress`` because __del__ must not raise
        # during interpreter shutdown, when dependencies may already be
        # half-unloaded.
        with contextlib.suppress(Exception):
            self._registry.close()
