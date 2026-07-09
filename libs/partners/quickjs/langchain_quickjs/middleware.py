"""`CodeInterpreterMiddleware`: exposes a sandboxed JavaScript REPL tool."""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal, NotRequired

from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
)
from langchain.tools import BaseTool, ToolRuntime
from langchain_core._api import beta
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.channels import DeltaChannel
from langgraph.config import get_config
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

from langchain_quickjs._format import format_outcome
from langchain_quickjs._prompt import (
    render_eval_tool_code_doc,
    render_eval_tool_description,
    render_repl_system_prompt,
    render_subagent_system_prompt,
)
from langchain_quickjs._ptc import (
    PTCOption,
    filter_tools_for_ptc,
    render_ptc_prompt,
)
from langchain_quickjs._repl import _DEFAULT_MAX_ACTIVE_THREADS, _Registry
from langchain_quickjs._snapshot import encode_snapshot, replay_snapshot_chain
from langchain_quickjs._subagent import find_subagent_task_tool

logger = logging.getLogger(__name__)

_DEFAULT_MEMORY_LIMIT = 64 * 1024 * 1024
_DEFAULT_TIMEOUT = 5.0
_DEFAULT_MAX_PTC_CALLS = 256
_DEFAULT_MAX_RESULT_CHARS = 4_000
_DEFAULT_TOOL_NAME = "eval"

PersistenceMode = Literal["thread", "turn", "call"]


class REPLState(AgentState):
    """State schema for `CodeInterpreterMiddleware`."""

    _quickjs_snapshot_payload: NotRequired[
        Annotated[
            bytes,
            DeltaChannel(replay_snapshot_chain),
            PrivateStateAttr,
        ]
    ]


class EvalSchema(BaseModel):
    """Input schema for the `eval` tool."""

    code: str = Field(
        description=(
            "JavaScript expression or statement(s) to evaluate. "
            "No fs/network/real-clock access."
        ),
    )


def _resolve_mode(
    *,
    mode: str | None,
) -> PersistenceMode:
    """Normalize persistence mode and enforce invariant constraints."""
    match mode:
        case None | "thread":
            return "thread"
        case "turn":
            return "turn"
        case "call":
            return "call"
        case _:
            msg = "`mode` must be one of 'thread', 'turn', or 'call'."
            raise ValueError(msg)


def _resolve_thread_id(fallback: str) -> str:
    """Extract `thread_id` from langgraph config or use `fallback`.

    The fallback is a middleware-instance-scoped id: when the caller
    didn't configure a `thread_id` (common for ad-hoc
    `agent.invoke(...)` in tests or single-shot scripts), we still need
    all resolver calls within one CodeInterpreterMiddleware lifetime to return the
    same id — otherwise `wrap_model_call` installs tools on one REPL
    and the eval tool looks up a different one, and the model sees
    `ReferenceError: tools is not defined`.
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


@beta()
class CodeInterpreterMiddleware(AgentMiddleware[REPLState, ContextT, ResponseT]):
    """Middleware exposing a JS REPL to the agent.

    Each LangGraph thread gets its own QuickJS slot (worker + runtime +
    context), so globals from one conversation cannot leak into another.

    Args:
        memory_limit: Bytes the QuickJS heap may use. Shared across all
            contexts under the same Runtime. Default 64 MiB.
        timeout: Per-call wall-clock timeout in seconds. Applied to every
            `eval` on every context. Default 5.
        max_ptc_calls: Maximum number of `tools.*` bridge calls allowed
            during one `eval` execution. Exceeding this budget throws
            from the host-function bridge before invoking the tool.
            Uncaught overflows surface as `PTCCallBudgetExceeded`.
            `None` disables the budget (unsafe for untrusted prompts;
            enables PTC-call DoS patterns). Default 256.

            !!! warning

                Setting `max_ptc_calls=None` disables the call budget and can allow
                unbounded PTC host-call loops (DoS risk). Only disable in trusted
                environments.

        tool_name: Name of the tool exposed to the model. Default `eval`.
        max_result_chars: Result and stdout blocks are independently
            truncated to this many characters before being sent back to
            the model. Console buffering is also bounded to this value
            during collection. Default 4000.
        capture_console: If `True`, install a `console` object that
            buffers `console.log/warn/error` calls and emits them in
            `<stdout>` blocks alongside the result. Default `True`.
        subagents: If `True`, expose the top-level `task(...)`
            JavaScript API when the current agent has a Deep Agents `task`
            tool. Set to `False` to require subagent dispatch through the
            normal parent `task` tool path instead.

            !!! warning
                `task(...)` calls run inside an already-approved `eval`
                invocation and do not trigger parent-level `interrupt_on` /
                HITL approval per dispatch. Gate the `eval` tool itself, add
                approval middleware inside subagent specs, or set
                `subagents=False` if per-dispatch parent approval is required.
        ptc: Programmatic tool calling — expose agent tools inside the
            REPL as `tools.<camelCase>(input) => Promise<string>`. One
            `eval` call can then orchestrate many tool calls (loops,
            `Promise.all`, conditional branching). Accepts:

            - `None` (default) — disabled.
            - `list[str | BaseTool]` — allowlist entries may be:
              - `str` tool names, matched against the agent's toolset.
              - `BaseTool` instances, exposed directly even if not on
                the agent's tool list.

            Mixed lists are supported. Explicit `BaseTool` entries are
            considered first; then name-matched agent tools are added.
            Duplicate names are deduplicated.

            !!! warning
                PTC calls currently execute through the REPL bridge and
                do **not** go through the normal `ToolNode` path. As a
                result, `interrupt_on` / HITL approval workflows are not
                enforced per PTC-invoked tool call.

            The REPL's own tool is always excluded; a model asking for
            `tools.eval("...")` would recurse pointlessly.
        mode: REPL state persistence mode.
            - `"thread"`: state persists across calls and across turns.
            - `"turn"`: state persists across calls within a turn only.
            - `"call"`: each eval call runs in a fresh REPL.
            If omitted, defaults to `"thread"`
        max_snapshot_bytes: Maximum serialized snapshot payload size allowed
            in middleware state. If a snapshot exceeds this size, it is
            dropped (`_quickjs_snapshot_payload=None`). Defaults to
            `memory_limit`.

    Example:
        ```python
        from deepagents import create_deep_agent
        from langchain_quickjs import CodeInterpreterMiddleware

        agent = create_deep_agent(
            model="claude-sonnet-4-6",
            middleware=[CodeInterpreterMiddleware()],
        )
        ```
    """

    state_schema = REPLState

    def __init__(
        self,
        *,
        memory_limit: int = _DEFAULT_MEMORY_LIMIT,
        timeout: float = _DEFAULT_TIMEOUT,
        max_ptc_calls: int | None = _DEFAULT_MAX_PTC_CALLS,
        tool_name: str = _DEFAULT_TOOL_NAME,
        max_result_chars: int = _DEFAULT_MAX_RESULT_CHARS,
        capture_console: bool = True,
        max_active_threads: int | None = _DEFAULT_MAX_ACTIVE_THREADS,
        subagents: bool = True,
        ptc: PTCOption | None = None,
        mode: PersistenceMode | None = None,
        max_snapshot_bytes: int | None = None,
    ) -> None:
        """Initialize REPL middleware state and build the exposed eval tool."""
        super().__init__()
        if max_ptc_calls is not None and max_ptc_calls < 1:
            msg = "`max_ptc_calls` must be >= 1 or None"
            raise ValueError(msg)
        if max_snapshot_bytes is not None and max_snapshot_bytes < 1:
            msg = "`max_snapshot_bytes` must be >= 1 or None"
            raise ValueError(msg)
        if max_active_threads is not None and max_active_threads < 1:
            msg = "`max_active_threads` must be >= 1 or None"
            raise ValueError(msg)
        self._memory_limit = memory_limit
        self._timeout = timeout
        self._max_ptc_calls = max_ptc_calls
        self._tool_name = tool_name
        self._max_result_chars = max_result_chars
        self._capture_console = capture_console
        self._subagents = subagents
        self._ptc = ptc
        self._mode = _resolve_mode(mode=mode)
        self._max_snapshot_bytes = (
            memory_limit if max_snapshot_bytes is None else max_snapshot_bytes
        )
        self._registry = _Registry(
            memory_limit=memory_limit,
            timeout=timeout,
            capture_console=capture_console,
            max_stdout_chars=max_result_chars,
            max_ptc_calls=max_ptc_calls,
            subagents_enabled=subagents,
            max_active_threads=max_active_threads,
        )
        self._memory_limit_mb = memory_limit // (1024 * 1024)
        self._base_prompt_cache: dict[bool, str] = {}
        self._ptc_prompt_cache: tuple[frozenset[str], str] | None = None
        self._ptc_tools_by_thread: dict[str, tuple[BaseTool, ...]] = {}
        # Stable fallback thread id — used when `thread_id` isn't in
        # langgraph config. Must be instance-scoped so `wrap_model_call`
        # and `eval` invocations within one conversation resolve to the
        # same REPL; otherwise the PTC install happens on one REPL and the
        # eval runs on another (and sees `tools` undefined).
        self._fallback_thread_id = f"session_{uuid.uuid4().hex[:8]}"
        self.tools: list[BaseTool] = [self._build_tool()]

    def _build_tool(self) -> BaseTool:
        tool_name = self._tool_name
        max_chars = self._max_result_chars
        fallback_id = self._fallback_thread_id
        middleware = self
        code_doc = render_eval_tool_code_doc(mode=self._mode)
        tool_description = render_eval_tool_description(mode=self._mode)

        def _make_tool_message(
            outcome: Any,
            tool_call_id: str | None,
        ) -> ToolMessage:
            return ToolMessage(
                content=format_outcome(outcome, max_result_chars=max_chars),
                tool_call_id=tool_call_id,
                name=tool_name,
            )

        def sync_eval(
            runtime: ToolRuntime[None, Any],
            code: Annotated[str, code_doc],
        ) -> ToolMessage:
            thread_id = _resolve_thread_id(fallback_id)
            repl = middleware._repl_for_eval(thread_id)
            try:
                outcome = repl.eval_sync(
                    code,
                    outer_runtime=runtime,
                )
            finally:
                if middleware._mode == "call":
                    middleware._registry.reset_repl(thread_id)
            return _make_tool_message(outcome, runtime.tool_call_id)

        async def async_eval(
            runtime: ToolRuntime[None, Any],
            code: Annotated[str, code_doc],
        ) -> ToolMessage:
            thread_id = _resolve_thread_id(fallback_id)
            repl = middleware._repl_for_eval(thread_id)
            try:
                outcome = await repl.eval_async(
                    code,
                    outer_runtime=runtime,
                    outer_loop=asyncio.get_running_loop(),
                )
            finally:
                if middleware._mode == "call":
                    middleware._registry.reset_repl(thread_id)
            return _make_tool_message(outcome, runtime.tool_call_id)

        return StructuredTool.from_function(
            name=tool_name,
            description=tool_description,
            func=sync_eval,
            coroutine=async_eval,
            infer_schema=False,
            args_schema=EvalSchema,
            metadata={"ls_code_input_language": "javascript"},
        )

    def _ptc_tool_names(self) -> set[str]:
        """Collect tool names from the PTC configuration."""
        names: set[str] = set()
        for entry in self._ptc or []:
            if isinstance(entry, str):
                names.add(entry)
            elif isinstance(entry, BaseTool):
                names.add(entry.name)
        return names

    def _repl_for_eval(self, thread_id: str) -> Any:
        """Return the REPL slot for one eval invocation."""
        repl = self._registry.get(thread_id)
        if self._mode == "call" and self._ptc is not None:
            repl.install_tools(list(self._ptc_tools_by_thread.get(thread_id, ())))
        return repl

    def before_agent(
        self,
        state: REPLState,
        runtime: "Runtime[ContextT]",  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Restore REPL snapshot bytes into the current thread slot."""
        if self._mode != "thread":
            return None
        payload = state.get("_quickjs_snapshot_payload")
        if not payload:
            return None
        thread_id = _resolve_thread_id(self._fallback_thread_id)
        repl = self._registry.get(thread_id)
        try:
            repl.restore_snapshot(payload, inject_globals=True)
        except Exception:  # noqa: BLE001  # best-effort restore path
            logger.warning(
                "Failed to restore QuickJS snapshot for thread_id=%s",
                thread_id,
                exc_info=True,
            )
            return {"_quickjs_snapshot_payload": None}
        return None

    async def abefore_agent(
        self,
        state: REPLState,
        runtime: "Runtime[ContextT]",  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Async variant of `before_agent` snapshot restore."""
        if self._mode != "thread":
            return None
        payload = state.get("_quickjs_snapshot_payload")
        if not payload:
            return None
        thread_id = _resolve_thread_id(self._fallback_thread_id)
        repl = self._registry.get(thread_id)
        try:
            await repl.arestore_snapshot(payload, inject_globals=True)
        except Exception:  # noqa: BLE001  # best-effort restore path
            logger.warning(
                "Failed to restore QuickJS snapshot for thread_id=%s",
                thread_id,
                exc_info=True,
            )
            return {"_quickjs_snapshot_payload": None}
        return None

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

    def _base_prompt(self, *, ptc_attached: bool) -> str:
        """Return the base REPL system prompt, rendered lazily and memoized.

        The text depends only on construction-time config and `ptc_attached`,
        so it's computed on first use per boolean and cached. Avoids rendering
        the `tools.*` variant at all when PTC is disabled.
        """
        cached = self._base_prompt_cache.get(ptc_attached)
        if cached is None:
            cached = render_repl_system_prompt(
                tool_name=self._tool_name,
                timeout=self._timeout,
                memory_limit_mb=self._memory_limit_mb,
                mode=self._mode,
                ptc_attached=ptc_attached,
            )
            self._base_prompt_cache[ptc_attached] = cached
        return cached

    def _prepare_for_call(self, request: ModelRequest[ContextT]) -> str:
        """Install PTC bindings for this turn and return the prompt addendum.

        Called from both sync and async model-call wrappers. Reads the
        live tool list off the request (middlewares upstream may have
        filtered it), installs PTC bridges on the current thread's REPL,
        and renders matching API-reference text.
        """
        request_tools: list[BaseTool] = list(getattr(request, "tools", []) or [])

        subagent_section = ""
        if self._subagents and find_subagent_task_tool(request_tools) is not None:
            subagent_section = render_subagent_system_prompt(tool_name=self._tool_name)

        if self._ptc is None:
            return self._base_prompt(ptc_attached=False) + subagent_section

        exposed = filter_tools_for_ptc(
            request_tools,
            self._ptc,
            self_tool_name=self._tool_name,
        )
        prompt = self._base_prompt(ptc_attached=bool(exposed)) + subagent_section
        thread_id = _resolve_thread_id(self._fallback_thread_id)
        repl = self._registry.get(thread_id)
        repl.install_tools(exposed)
        self._ptc_tools_by_thread[thread_id] = tuple(exposed)
        # Rendering the TS-ish signature block is cheap but not free;
        # cache by the set of exposed names. The set doesn't encode tool
        # *identity* — if a tool keeps its name but its schema changes
        # between turns, the cached prompt staleness is on the caller.
        # Same tradeoff the TS package accepts; see the module docstring.
        exposed_names = frozenset(t.name for t in exposed)
        if self._ptc_prompt_cache is None or self._ptc_prompt_cache[0] != exposed_names:
            self._ptc_prompt_cache = (
                exposed_names,
                render_ptc_prompt(exposed, tool_name=self._tool_name),
            )
        return prompt + self._ptc_prompt_cache[1]

    def _extend(
        self, system_message: SystemMessage | None, prompt: str
    ) -> SystemMessage:
        return append_to_system_message(system_message, prompt)

    def _snapshot_update(
        self, *, payload: bytes, prior: bytes, thread_id: str
    ) -> dict[str, Any]:
        """Build a patch-chain state update for a fresh snapshot ``payload``."""
        size = len(payload)
        if size > self._max_snapshot_bytes:
            logger.warning(
                (
                    "Dropping QuickJS snapshot for thread_id=%s "
                    "(size=%d bytes exceeds max_snapshot_bytes=%d)"
                ),
                thread_id,
                size,
                self._max_snapshot_bytes,
            )
            return {"_quickjs_snapshot_payload": None}
        return {"_quickjs_snapshot_payload": encode_snapshot(payload, prior)}

    def after_agent(
        self,
        state: REPLState,
        runtime: "Runtime[ContextT]",  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Snapshot REPL state (optional) and evict this turn's REPL slot."""
        thread_id = _resolve_thread_id(self._fallback_thread_id)
        self._ptc_tools_by_thread.pop(thread_id, None)
        if self._mode != "thread":
            self._registry.evict(thread_id)
            return None

        repl = self._registry.get_if_exists(thread_id)
        if repl is None:
            return None
        prior = state.get("_quickjs_snapshot_payload") or b""
        update: dict[str, Any]
        try:
            update = self._snapshot_update(
                payload=repl.create_snapshot(),
                prior=prior,
                thread_id=thread_id,
            )
        except Exception:  # noqa: BLE001  # best-effort snapshot path
            logger.warning(
                "Failed to create QuickJS snapshot for thread_id=%s",
                thread_id,
                exc_info=True,
            )
            update = {"_quickjs_snapshot_payload": None}
        finally:
            self._registry.evict(thread_id)
        return update

    async def aafter_agent(
        self,
        state: REPLState,
        runtime: "Runtime[ContextT]",  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Async variant of `after_agent` snapshot+evict behavior."""
        thread_id = _resolve_thread_id(self._fallback_thread_id)
        self._ptc_tools_by_thread.pop(thread_id, None)
        if self._mode != "thread":
            await self._registry.aevict(thread_id)
            return None

        repl = self._registry.get_if_exists(thread_id)
        if repl is None:
            return None
        prior = state.get("_quickjs_snapshot_payload") or b""
        update: dict[str, Any]
        try:
            update = self._snapshot_update(
                payload=await repl.acreate_snapshot(),
                prior=prior,
                thread_id=thread_id,
            )
        except Exception:  # noqa: BLE001  # best-effort snapshot path
            logger.warning(
                "Failed to create QuickJS snapshot for thread_id=%s",
                thread_id,
                exc_info=True,
            )
            update = {"_quickjs_snapshot_payload": None}
        finally:
            await self._registry.aevict(thread_id)
        return update

    def __del__(self) -> None:
        """Best-effort Runtime cleanup on GC; never raises at shutdown."""
        # Wrapped in `contextlib.suppress` because __del__ must not raise
        # during interpreter shutdown, when dependencies may already be
        # half-unloaded.
        with contextlib.suppress(Exception):
            self._registry.close()
