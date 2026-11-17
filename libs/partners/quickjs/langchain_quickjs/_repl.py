"""Thread-keyed QuickJS REPL registry, console bridge, and result formatter.

Kept separate from `middleware.py` so the REPL mechanics stay testable
without constructing an agent or wiring up LangGraph state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, TypedDict, get_type_hints

from langchain_core.runnables import RunnableLambda
from langchain_core.tools.base import (
    InjectedToolCallId,
    _is_injected_arg_type,
    get_all_basemodel_annotations,
)
from langgraph.errors import GraphInterrupt
from langgraph.prebuilt.tool_node import _get_all_injected_args
from quickjs_rs import (
    UNDEFINED,
    ConcurrentEvalError,
    Context,
    DeadlockError,
    HostCancellationError,
    JSError,
    MarshalError,
    MemoryLimitError,
    Runtime,
    Snapshot,
    SourceTransform,
    ThreadWorker,
)
from quickjs_rs import (
    TimeoutError as QJSTimeoutError,
)

from langchain_quickjs import _fs
from langchain_quickjs._format import (
    coerce_tool_output_for_ptc,
    format_handle,
    stringify,
)
from langchain_quickjs._ptc import is_valid_js_identifier, to_camel_case
from langchain_quickjs._subagent import (
    call_subagent_task_tool,
    find_subagent_task_tool,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Sequence

    from deepagents.backends.protocol import BackendProtocol
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SUBAGENT_CONCURRENCY = 32
# Warm slots kept per middleware. A slot is closed once `_MAX_SLOTS` other
# threads have evaled since its last eval, even mid-turn; its next eval then
# restores the last persisted snapshot and loses that turn's earlier state.
# That takes a busy process: langgraph-api runs 10 concurrent runs per worker
# by default.
_MAX_SLOTS = 32
_TASK_FUNCTION_NAME = "task"


def _clear_exception_references(exc: BaseException) -> None:
    """Drop traceback links to avoid cross-thread GC finalizing QJS handles.

    quickjs_rs exceptions may keep traceback frames that hold temporary
    `QjsHandle` objects. If those cycles are collected on a different
    thread, quickjs_rs raises "unsendable ... dropped on another thread".
    """
    exc.__traceback__ = None
    exc.__context__ = None
    exc.__cause__ = None


@dataclass
class EvalOutcome:
    """Normalized result of a single REPL eval.

    Exactly one of `result` / `error` is meaningful per call; `stdout`
    is collected from `console.*` regardless.
    """

    stdout: str = ""
    stdout_truncated_chars: int = 0
    result: str | None = None
    result_kind: str | None = None  # "handle" when marshaling fell back
    error_type: str | None = None
    error_message: str = ""
    error_stack: str | None = None


class _SnapshotRestoreError(RuntimeError):
    """Raised when persisted state cannot be restored before evaluation."""


class _PTCCallBudgetExceededError(RuntimeError):
    """Raised when one eval exceeds its configured PTC call budget."""

    def __init__(self, *, limit: int, attempted: int, function_name: str) -> None:
        self.limit = limit
        self.attempted = attempted
        self.function_name = function_name
        msg = (
            "PTC call budget exceeded "
            f"(limit={limit}, attempted={attempted}, "
            f"function={function_name})"
        )
        super().__init__(msg)

    def render_message(self) -> str:
        return (
            "PTC call budget exceeded "
            f"(limit={self.limit}, attempted={self.attempted}, "
            f"function={self.function_name})"
        )


class _TaskBridgeError(RuntimeError):
    """Wrap errors from the top-level `task()` host function."""

    def __init__(self, exc: Exception) -> None:
        self.error_type = type(exc).__name__
        self.error_message = str(exc)
        super().__init__(self.error_message)


@dataclass(frozen=True, slots=True)
class _PTCState:
    """Per-eval PTC state (reset on each eval call)."""

    remaining_calls: int | None
    outer_runtime: ToolRuntime | None = None
    outer_loop: asyncio.AbstractEventLoop | None = None
    fs: BackendProtocol | None = None
    task_dispatch_count: int = 0
    """Ordinal for the next `task()` dispatch in this eval.

    Starts at 0 for every eval, so a replayed eval that dispatches in the
    same order reproduces the same ordinals and therefore the same derived
    dispatch ids.
    """

    def consume_call_budget(
        self, *, function_name: str, max_ptc_calls: int | None
    ) -> _PTCState:
        """Count one PTC bridge call and enforce the per-eval limit."""
        if self.remaining_calls is None:
            return self
        if self.remaining_calls > 0:
            return replace(self, remaining_calls=self.remaining_calls - 1)

        normalized_limit = max_ptc_calls if max_ptc_calls is not None else 0
        raise _PTCCallBudgetExceededError(
            limit=normalized_limit,
            attempted=normalized_limit + 1,
            function_name=function_name,
        )


class _ConsoleBuffer:
    """Accumulates `console.*` output between evals.

    Shared by the three host functions we install on each context. We don't
    bother distinguishing log/warn/error in the output format — the model
    does not care about the level, and flattening keeps the returned string
    smaller.
    """

    def __init__(self, max_chars: int) -> None:
        self._max_chars = max(0, max_chars)
        self._stdout = ""
        self._dropped_chars = 0

    def append(self, level: str, args: tuple[Any, ...]) -> None:
        del level  # flattened; see class docstring
        line = " ".join(stringify(a) for a in args)
        chunk = line if not self._stdout else f"\n{line}"
        remaining = self._max_chars - len(self._stdout)
        if remaining <= 0:
            self._dropped_chars += len(chunk)
            return
        kept = chunk[:remaining]
        self._stdout += kept
        self._dropped_chars += len(chunk) - len(kept)

    def drain(self) -> tuple[str, int]:
        if not self._stdout and self._dropped_chars == 0:
            return "", 0
        out = self._stdout
        dropped = self._dropped_chars
        self._stdout = ""
        self._dropped_chars = 0
        return out, dropped


_UNDEFINED_TYPE = type(UNDEFINED)


def _is_undefined(value: Any) -> bool:
    """Return whether ``value`` is a QuickJS ``undefined`` marshal result.

    Marshaling produces fresh ``Undefined`` instances rather than the
    ``UNDEFINED`` singleton, so identity comparison is unreliable.
    """
    return isinstance(value, _UNDEFINED_TYPE)


def _strip_undefined(value: Any) -> Any:
    """Recursively drop ``undefined`` object properties so defaults apply.

    JS ``undefined`` means "absent", so a dict key whose value is
    ``undefined`` is omitted, letting the tool's schema defaults take over.
    Array entries are converted to ``None`` (JS ``null``) instead of being
    dropped, since dropping would shift indices and corrupt the array.
    """
    if isinstance(value, dict):
        return {
            key: _strip_undefined(val)
            for key, val in value.items()
            if not _is_undefined(val)
        }
    if isinstance(value, list):
        return [
            None if _is_undefined(item) else _strip_undefined(item) for item in value
        ]
    return value


def _normalize_tool_input(raw: Any) -> dict[str, Any]:
    """Coerce whatever JS passed into `tools.X(...)` to a dict.

    LangChain tools accept a dict. QuickJS marshals JS objects to dicts
    already; we just want to guard against the model passing `null`,
    `undefined`, a bare string, or a number (none of which a well-
    formed tool call should produce, but the model is the model).
    """
    if raw is None or _is_undefined(raw):
        return {}
    if isinstance(raw, dict):
        return _strip_undefined(raw)
    # Bare scalar / list — wrap under a conventional key so the tool's
    # schema validation produces an informative error rather than a
    # silent miss.
    return {"input": _strip_undefined(raw)}


def _synth_tool_call_id(tool_name: str) -> str:
    """Mint a synthetic tool_call_id for a PTC-driven tool invocation."""
    return f"ptc_{tool_name}_{uuid.uuid4().hex[:8]}"


async def _on_outer_loop(
    call: Callable[[], Coroutine[Any, Any, Any]],
    outer_loop: asyncio.AbstractEventLoop | None,
) -> Any:
    """Await `call()` on the agent's event loop when one is known."""
    if outer_loop is None or asyncio.get_running_loop() is outer_loop:
        return await call()
    future = asyncio.run_coroutine_threadsafe(call(), outer_loop)
    try:
        return await asyncio.wrap_future(future)
    except asyncio.CancelledError:
        future.cancel()
        raise


def _inject_tool_args_for_ptc(
    tool: Any,
    payload: dict[str, Any],
    outer_runtime: Any,
    tool_call_id: str,
) -> dict[str, Any]:
    """Mirror LangGraph's `ToolNode._inject_tool_args` for PTC calls.

    LangChain tools that declare `ToolRuntime` / `InjectedState` /
    `InjectedStore` only see those values when a real `ToolNode` wires
    them in. PTC calls bypass it, so we replicate the detection logic here.
    The outer runtime (captured from the active `eval` tool invocation)
    provides state/store/context/config; `tool_call_id` is freshly minted
    per sub-call. `InjectedToolCallId` is handled separately via
    `BaseTool.arun(..., tool_call_id=...)` at the bridge site.
    """
    enriched = dict(payload)

    injected = _get_all_injected_args(tool)
    if not injected or outer_runtime is None:
        return enriched

    # Build a ToolRuntime matching the outer one but with a fresh
    # tool_call_id. `type(outer_runtime)` rather than a literal import
    # so the shape stays in lockstep with whatever langgraph ships.
    derived = type(outer_runtime)(
        state=outer_runtime.state,
        tool_call_id=tool_call_id,
        config=outer_runtime.config,
        context=outer_runtime.context,
        store=outer_runtime.store,
        stream_writer=outer_runtime.stream_writer,
        tools=outer_runtime.tools,
        execution_info=getattr(outer_runtime, "execution_info", None),
        server_info=getattr(outer_runtime, "server_info", None),
    )

    if injected.runtime:
        enriched[injected.runtime] = derived
    # InjectedState: state can be injected under one or more arg names.
    if injected.state:
        for arg_name, state_field in injected.state.items():
            if state_field:
                enriched[arg_name] = (
                    outer_runtime.state.get(state_field)
                    if isinstance(outer_runtime.state, dict)
                    else getattr(outer_runtime.state, state_field, None)
                )
            else:
                enriched[arg_name] = outer_runtime.state
    if injected.store and outer_runtime.store is not None:
        enriched[injected.store] = outer_runtime.store
    return enriched


def _tool_uses_injected_tool_call_id(tool: Any) -> bool:
    """Return whether a tool declares an injected tool call ID."""
    schema_annotations = get_all_basemodel_annotations(tool.get_input_schema())
    func = getattr(tool, "func", None) or getattr(tool, "coroutine", None)
    func_annotations = get_type_hints(func, include_extras=True) if func else {}
    return any(
        _is_injected_arg_type(type_, injected_type=InjectedToolCallId)
        for type_ in {**func_annotations, **schema_annotations}.values()
    )


_TOOLS_CALL = "__tools_call"
_TOOLS_NAMES = "__tools_names"


def _render_tools_namespace() -> str:
    """Return JS that defines `globalThis.tools` as a proxy over the host.

    Python owns the tool map: a call is `__tools_call(name, input)` and the
    installed names come from `__tools_names()`, so changing the tool set
    never touches the guest. Any other name, including inherited
    `Object.prototype` members, reads from a plain object.
    """
    call = json.dumps(_TOOLS_CALL)
    names = json.dumps(_TOOLS_NAMES)
    return (
        f"{{ const fn = (name) => (input) => globalThis[{call}](name, input);"
        " const has = (name) => typeof name === 'string'"
        f" && globalThis[{names}]().includes(name);"
        " globalThis.tools = new Proxy({}, {"
        " get(target, name) {"
        " return has(name) ? fn(name) : Reflect.get(target, name); },"
        " has(_, name) { return has(name); },"
        f" ownKeys() {{ return globalThis[{names}](); }},"
        " getOwnPropertyDescriptor(_, name) {"
        " return has(name) ? { value: fn(name), enumerable: true,"
        " configurable: true, writable: false } : undefined; },"
        " }); } undefined"
    )


class _REPLOptions(TypedDict):
    timeout: float | None
    capture_console: bool
    max_stdout_chars: int
    max_ptc_calls: int | None
    subagents_enabled: bool
    max_subagent_concurrency: int


class _ThreadREPL:
    """One QuickJS context + console buffer, per LangGraph thread.

    All `ctx.*` operations are marshalled onto the worker's dedicated
    thread because `quickjs_rs` objects are `!Send`. The public
    methods are safe to call from any thread/loop.
    """

    def __init__(
        self,
        worker: ThreadWorker,
        runtime: Runtime,
        *,
        timeout: float | None,
        capture_console: bool,
        max_stdout_chars: int,
        max_ptc_calls: int | None = 256,
        subagents_enabled: bool = True,
        max_subagent_concurrency: int = _DEFAULT_MAX_SUBAGENT_CONCURRENCY,
    ) -> None:
        """Build the REPL's Context. Must run on `worker`'s own thread."""
        self._worker = worker
        self._runtime = runtime
        # The Context-level `timeout` is used as the cumulative budget
        # for sync evals. Async evals pass `timeout=` per call so each
        # call gets a fresh budget — matches what a REPL user expects,
        # and what we describe in the system prompt.
        self._per_call_timeout = float("inf") if timeout is None else timeout
        self._capture_console = capture_console
        # Static budget config; mutable counters live in `_ptc_state`.
        self._max_ptc_calls = max_ptc_calls
        self._subagents_enabled = subagents_enabled
        self._max_subagent_concurrency = max_subagent_concurrency
        self._console = _ConsoleBuffer(max_stdout_chars)
        self._ctx: Context | None = None
        self._prepared_for_eval = False
        # PTC tools by camel-case name. `install_tools` swaps this map; the
        # guest reaches it through `__tools_call` / `__tools_names`.
        self._tools: dict[str, BaseTool] = {}
        # Whether `globalThis.tools` has been defined. Without PTC it stays
        # undefined.
        self._tools_installed: bool = False
        # Mutable per-eval PTC state. Tracks call budget plus outer
        # runtime/loop dispatch context for bridge invocations. Allocated
        # at eval start and cleared in finally so bridge calls can't run
        # outside the current eval.
        self._ptc_state: _PTCState | None = None
        self._task_calls: asyncio.Semaphore | None = None

        self._init()

    def _init(self) -> None:
        self._ctx = self._runtime.new_context(timeout=self._per_call_timeout)
        if self._capture_console:
            self._install_console()
        self._register_tools_bridge()
        if self._subagents_enabled:
            self._task_calls = asyncio.Semaphore(self._max_subagent_concurrency)
            self._register_task_bridge()
        self._register_fs_bridge()

    def _require_ctx(self) -> Context:
        """Return the live QuickJS context or raise if this REPL is closed."""
        if self._ctx is None:
            msg = "QuickJS context is closed"
            raise RuntimeError(msg)
        return self._ctx

    def _install_console(self) -> None:
        ctx = self._require_ctx()
        buf = self._console

        @ctx.function(name="__console_log")
        def _log(*args: Any) -> None:
            buf.append("log", args)

        @ctx.function(name="__console_warn")
        def _warn(*args: Any) -> None:
            buf.append("warn", args)

        @ctx.function(name="__console_error")
        def _error(*args: Any) -> None:
            buf.append("error", args)

        # Install the JS-level console object. We do this via a separate
        # eval because register_host_function only puts the callable on the
        # global object under its given name; `globalThis.console` needs
        # to exist as a normal object for idiomatic JS. Trailing primitive
        # keeps the eval's result marshalable — assigning an object would
        # bubble a MarshalError we'd have to special-case.
        ctx.eval(
            "globalThis.console = {"
            " log: __console_log,"
            " warn: __console_warn,"
            " error: __console_error,"
            "}; undefined"
        )

    def install_tools(self, tools: Sequence[BaseTool]) -> None:
        """Expose `tools` as `globalThis.tools.<camelCase>` in the REPL.

        Swaps the Python-side tool map; the guest's `tools` proxy is
        defined once and reads the map on every access.
        """
        self._worker.run_sync(self._ainstall_tools(tools))

    async def _ainstall_tools(self, tools: Sequence[BaseTool]) -> None:
        name_to_tool: dict[str, BaseTool] = {}
        for tool in tools:
            camel = to_camel_case(tool.name)
            if not is_valid_js_identifier(camel):
                logger.warning(
                    "Skipping PTC tool %r: %r is not a valid JS identifier",
                    tool.name,
                    camel,
                )
                continue
            name_to_tool[camel] = tool
        self._tools = name_to_tool
        if not self._tools_installed:
            self._require_ctx().eval(_render_tools_namespace())
            self._tools_installed = True

    @staticmethod
    def _validate_task_payload(
        payload: dict[str, Any],
    ) -> tuple[str, str, str | None, dict[str, Any] | None]:
        """Validate JS `task()` input and return its typed fields.

        JS callers pass camelCase keys (`subagentType`, `responseSchema`) as
        documented in the system prompt; the returned tuple is snake_case for
        the Python dispatch path.
        """
        description = payload.get("description")
        if not isinstance(description, str) or not description:
            msg = "task() requires non-empty string field `description`"
            raise ValueError(msg)

        subagent_type = payload.get("subagentType")
        if not isinstance(subagent_type, str) or not subagent_type:
            msg = "task() requires non-empty string field `subagentType`"
            raise ValueError(msg)

        raw_label = payload.get("label")
        if raw_label is not None and not isinstance(raw_label, str):
            msg = "task() field `label` must be a string when provided"
            raise ValueError(msg)
        label = raw_label.strip() if isinstance(raw_label, str) else None
        if label == "":
            label = None

        response_schema = payload.get("responseSchema")
        if response_schema is not None and not isinstance(response_schema, dict):
            msg = "task() field `responseSchema` must be an object when provided"
            raise ValueError(msg)

        return description, subagent_type, label, response_schema

    async def _ainvoke_task_on_outer_loop(
        self,
        payload: dict[str, Any],
        *,
        state: _PTCState,
        dispatch_ordinal: int,
    ) -> Any:
        """Validate JS `task()` input and invoke the runner on the right loop.

        The QuickJS host call runs on the REPL worker loop, but subagent runnables
        should execute on the parent LangGraph loop when one exists so callbacks,
        context, and async loop affinity match normal tool execution.
        """
        validated = self._validate_task_payload(payload)
        description, subagent_type, label, response_schema = validated

        async def _call() -> Any:
            runtime = state.outer_runtime
            if runtime is None:
                msg = "task() requires an active ToolRuntime"
                raise RuntimeError(msg)
            task_tool = find_subagent_task_tool(getattr(runtime, "tools", ()) or ())
            if task_tool is None:
                msg = "task tool not configured for this eval"
                raise RuntimeError(msg)
            return await call_subagent_task_tool(
                task_tool,
                description=description,
                subagent_type=subagent_type,
                response_schema=response_schema,
                runtime=runtime,
                label=label,
                dispatch_ordinal=dispatch_ordinal,
            )

        return await _on_outer_loop(_call, state.outer_loop)

    def _register_task_bridge(self) -> None:
        """Install the async host function backing top-level `task()`."""
        ctx = self._require_ctx()

        async def _bridge(raw_input: Any = None) -> Any:
            state = self._ptc_state
            if state is None:
                msg = "task bridge called outside active eval"
                raise ConcurrentEvalError(msg)
            task_calls = self._task_calls
            if task_calls is None:
                msg = "task call limiter not initialized"
                raise RuntimeError(msg)

            payload = _normalize_tool_input(raw_input)
            dispatch_ordinal = state.task_dispatch_count
            self._ptc_state = replace(state, task_dispatch_count=dispatch_ordinal + 1)
            async with task_calls:
                try:
                    result = await self._ainvoke_task_on_outer_loop(
                        payload,
                        state=state,
                        dispatch_ordinal=dispatch_ordinal,
                    )
                except GraphInterrupt:
                    raise
                except Exception as e:
                    # Subagent dispatches are part of the eval language, not
                    # PTC calls. Surface their validation/runtime failures as
                    # eval errors without changing normal `tools.*` semantics.
                    raise _TaskBridgeError(e) from e
            return coerce_tool_output_for_ptc(result)

        ctx.register(_TASK_FUNCTION_NAME, _bridge, is_async=True)
        ctx.eval(
            "Object.freeze(globalThis.task);"
            "Object.defineProperty(globalThis, 'task', {"
            " value: globalThis.task,"
            " writable: false,"
            " configurable: false,"
            "}); undefined"
        )

    async def _ainvoke_tool_on_outer_loop(
        self,
        tool: BaseTool,
        tool_call: dict[str, Any],
        *,
        outer_runtime: ToolRuntime | None,
        outer_loop: asyncio.AbstractEventLoop | None,
    ) -> Any:
        """Run the tool with the outer runtime's loop and config when available."""
        args = tool_call["args"]
        tool_call_id = tool_call.get("id")

        async def _call() -> Any:
            return await tool.arun(
                args,
                callbacks=(
                    outer_runtime.config.get("callbacks")
                    if outer_runtime is not None
                    else None
                ),
                run_id=uuid.uuid4(),
                config=outer_runtime.config if outer_runtime is not None else None,
                tool_call_id=(
                    tool_call_id if _tool_uses_injected_tool_call_id(tool) else None
                ),
            )

        return await _on_outer_loop(_call, outer_loop)

    def _register_tools_bridge(self) -> None:
        """Install the host functions behind the guest's `tools` proxy.

        `_call` looks the tool up in `self._tools` on every call, so
        `install_tools` swapping the map is picked up without touching
        the guest.
        """
        ctx = self._require_ctx()

        def _names() -> list[str]:
            return list(self._tools)

        async def _call(camel: str, raw_input: Any = None) -> Any:
            tool = self._tools.get(camel)
            if tool is None:
                # A function captured before `install_tools` dropped it.
                msg = f"tool '{camel}' not registered"
                raise RuntimeError(msg)
            if self._ptc_state is None:
                msg = "PTC bridge called outside active eval"
                raise ConcurrentEvalError(msg)
            state = self._ptc_state.consume_call_budget(
                function_name=f"tools.{camel}",
                max_ptc_calls=self._max_ptc_calls,
            )
            self._ptc_state = state
            payload = _normalize_tool_input(raw_input)
            call_id = _synth_tool_call_id(tool.name)
            # Inject runtime/state/store ourselves. The bridge uses `arun`
            # rather than the tool-call envelope path so the result can be
            # unwrapped back to native JS-visible values after the standard
            # tool lifecycle callbacks have fired.
            args = _inject_tool_args_for_ptc(
                tool, payload, state.outer_runtime, call_id
            )
            result = await self._ainvoke_tool_on_outer_loop(
                tool,
                {"name": tool.name, "args": args, "id": call_id, "type": "tool_call"},
                outer_runtime=state.outer_runtime,
                outer_loop=state.outer_loop,
            )
            return coerce_tool_output_for_ptc(result)

        ctx.register(_TOOLS_NAMES, _names)
        ctx.register(_TOOLS_CALL, _call, is_async=True)

    def _register_fs_bridge(self) -> None:
        ctx = self._require_ctx()
        ctx.register(_fs.HOST_FUNCTION, self._fs_call, is_async=True)
        ctx.eval(
            f"Object.defineProperty(globalThis, '{_fs.HOST_FUNCTION}',"
            " { writable: false, configurable: false }); undefined"
        )

    async def _fs_call(self, op: str, args: dict[str, Any]) -> dict[str, Any]:
        state = self._ptc_state
        if state is None or state.fs is None:
            msg = "fs called outside active eval"
            raise ConcurrentEvalError(msg)
        name = _fs.public_name(op)
        self._ptc_state = state.consume_call_budget(
            function_name=name, max_ptc_calls=self._max_ptc_calls
        )
        backend = state.fs
        reply: dict[str, Any] = {}

        async def call(_: None) -> None:
            # The value skips the run's output so file text stays out of traces.
            reply["value"] = await _fs.run(backend, op, args)

        # Running under the eval call's config lets backends that keep files in
        # graph state, such as `StateBackend`, read and write them.
        config = state.outer_runtime.config if state.outer_runtime else None
        runnable = RunnableLambda(call, name=name)
        try:
            await _on_outer_loop(
                lambda: runnable.ainvoke(None, config), state.outer_loop
            )
        except _fs.FsError as e:
            return e.reply
        return reply

    def eval_sync(
        self,
        code: str,
        *,
        outer_runtime: ToolRuntime | None = None,
        snapshot_payload: bytes | None = None,
        tools: Sequence[BaseTool] | None = None,
        fs: BackendProtocol | None = None,
    ) -> EvalOutcome:
        # Both sync and async entry points funnel through ctx.eval_async on
        # the worker loop. Sync ctx.eval can't dispatch async host functions
        # (PTC bridges are is_async=True), so routing sync callers through
        # the async path is required for PTC to work under sync invocation.
        return self._worker.run_sync(
            self._aeval_async(
                code,
                outer_runtime=outer_runtime,
                snapshot_payload=snapshot_payload,
                tools=tools,
                fs=fs,
            )
        )

    async def eval_async(
        self,
        code: str,
        *,
        outer_runtime: ToolRuntime | None = None,
        snapshot_payload: bytes | None = None,
        tools: Sequence[BaseTool] | None = None,
        outer_loop: asyncio.AbstractEventLoop | None = None,
        fs: BackendProtocol | None = None,
    ) -> EvalOutcome:
        return await self._worker.run_async(
            self._aeval_async(
                code,
                outer_runtime=outer_runtime,
                snapshot_payload=snapshot_payload,
                tools=tools,
                outer_loop=outer_loop,
                fs=fs,
            )
        )

    def create_snapshot(self) -> bytes:
        """Capture the current context snapshot as bytes."""
        return self._worker.run_sync(self._acreate_snapshot())

    async def acreate_snapshot(self) -> bytes:
        """Async variant of `create_snapshot`."""
        return await self._worker.run_async(self._acreate_snapshot())

    async def _acreate_snapshot(self) -> bytes:
        ctx = self._require_ctx()
        snapshot = ctx.create_snapshot()
        return snapshot.to_bytes()

    def restore_snapshot(self, payload: bytes, *, inject_globals: bool = True) -> None:
        """Restore snapshot bytes into this REPL's context."""
        self._worker.run_sync(
            self._arestore_snapshot(payload, inject_globals=inject_globals)
        )

    async def arestore_snapshot(
        self, payload: bytes, *, inject_globals: bool = True
    ) -> None:
        """Async variant of `restore_snapshot`."""
        await self._worker.run_async(
            self._arestore_snapshot(payload, inject_globals=inject_globals)
        )

    async def _arestore_snapshot(self, payload: bytes, *, inject_globals: bool) -> None:
        ctx = self._require_ctx()
        snapshot = Snapshot.from_bytes(payload)
        self._runtime.restore_snapshot(
            snapshot,
            ctx,
            inject_globals=inject_globals,
        )

    async def _aprepare_for_eval(
        self,
        payload: bytes | None,
        tools: Sequence[BaseTool] | None,
        fs: BackendProtocol | None,
    ) -> None:
        """Restore once and bind current tools on the owning worker."""
        if not self._prepared_for_eval:
            self._prepared_for_eval = True
            if payload:
                try:
                    await self._arestore_snapshot(payload, inject_globals=True)
                except Exception as exc:  # best-effort restore path
                    logger.warning("Failed to restore QuickJS snapshot", exc_info=True)
                    msg = "Failed to restore QuickJS snapshot"
                    raise _SnapshotRestoreError(msg) from exc
        if tools is not None:
            await self._ainstall_tools(tools)
        if fs is not None:
            self._require_ctx().eval(_fs.install_source())

    async def _aeval_async(  # noqa: C901, PLR0912, PLR0915
        self,
        code: str,
        *,
        outer_runtime: ToolRuntime | None = None,
        snapshot_payload: bytes | None = None,
        tools: Sequence[BaseTool] | None = None,
        outer_loop: asyncio.AbstractEventLoop | None = None,
        fs: BackendProtocol | None = None,
    ) -> EvalOutcome:
        """Uses `ctx.eval_async` directly.

        Overlapping evals on the same context surface as
        `ConcurrentEvalError` (recorded in `EvalOutcome.error_type`).
        We intentionally do not queue: a model dispatching overlapping
        evals against shared state is almost always a prompting bug,
        and a loud failure is a better signal than silent serialisation.
        """
        ctx = self._require_ctx()
        outcome = EvalOutcome()
        # Save/restore rather than clear-on-exit: a second eval that hits
        # ConcurrentEvalError would otherwise null out the in-flight
        # eval's state and orphan its bridge calls.
        prev_ptc_state = self._ptc_state
        self._ptc_state = _PTCState(
            remaining_calls=self._max_ptc_calls,
            outer_runtime=outer_runtime,
            outer_loop=outer_loop,
            fs=fs,
        )
        try:
            await self._aprepare_for_eval(snapshot_payload, tools, fs)
            # Drive any final-expression Promise (e.g. a bare async
            # IIFE) to its resolved value before marshaling. Without
            # this the Promise object itself fails to marshal and
            # the result surfaces as `[object]` rather than the
            # awaited value.
            handle = await ctx.eval_handle_async(code, timeout=self._per_call_timeout)
            try:
                if handle.is_promise():
                    resolved = await handle.await_promise(
                        timeout=self._per_call_timeout
                    )
                else:
                    resolved = handle
                try:
                    try:
                        value = resolved.to_python()
                    except MarshalError as me:
                        outcome.result_kind = "handle"
                        outcome.result = format_handle(resolved)
                        _clear_exception_references(me)
                    else:
                        outcome.result = stringify(value)
                finally:
                    if resolved is not handle:
                        resolved.dispose()
            finally:
                handle.dispose()
        except _PTCCallBudgetExceededError as e:
            # Raised from inside the PTC bridge; quickjs-rs propagates the
            # original exception out of eval_handle_async / await_promise.
            # Surface it as a distinct, model-recoverable error so the
            # agent can shorten its script rather than crash.
            outcome.error_type = "PTCCallBudgetExceeded"
            outcome.error_message = e.render_message()
            _clear_exception_references(e)
        except QJSTimeoutError as e:
            outcome.error_type = "Timeout"
            outcome.error_message = str(e)
            _clear_exception_references(e)
        except DeadlockError as e:
            # Top-level Promise never resolved and no async host work in
            # flight. Surface as a distinct error type because the fix
            # is user-level (their JS has an un-resolvable Promise or a
            # sync host fn that should be async); a plain error-type
            # message without context would make this hard to diagnose.
            outcome.error_type = "Deadlock"
            outcome.error_message = str(e)
            _clear_exception_references(e)
        except HostCancellationError:
            # JS declined to catch a cancellation — re-raise as
            # CancelledError so asyncio unwinds the caller's task.
            # Do not record anything in `outcome`; the call is dead.
            raise asyncio.CancelledError from None
        except JSError as e:
            self._record_js_error(outcome, e)
            _clear_exception_references(e)
        except ConcurrentEvalError as e:
            outcome.error_type = "ConcurrentEval"
            outcome.error_message = str(e)
            _clear_exception_references(e)
        except MemoryLimitError as e:
            outcome.error_type = "OutOfMemory"
            outcome.error_message = str(e)
            _clear_exception_references(e)
        except MarshalError as e:
            # A host result, such as a large `fs.readFile`, could not be
            # converted to a JS value; without this the agent run would crash.
            outcome.error_type = "MarshalError"
            outcome.error_message = (
                f"a host result could not be passed to JavaScript ({e}); "
                "it may be too large or too deeply nested."
            )
            _clear_exception_references(e)
        except _TaskBridgeError as e:
            outcome.error_type = e.error_type
            outcome.error_message = e.error_message
            _clear_exception_references(e)
        finally:
            self._ptc_state = prev_ptc_state
            outcome.stdout, outcome.stdout_truncated_chars = self._console.drain()
        return outcome

    def _record_js_error(self, outcome: EvalOutcome, e: JSError) -> None:
        outcome.error_type = e.name
        outcome.error_message = e.message
        outcome.error_stack = e.stack

    def close(self) -> None:
        self._worker.run_sync(self._aclose())

    async def aclose(self) -> None:
        await self._worker.run_async(self._aclose())

    async def _aclose(self) -> None:
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None


class _Slot:
    """One LangGraph thread's QuickJS worker, Runtime and Contexts.

    Creates nothing until the first eval. All QuickJS work, including building
    and closing the Runtime and Contexts, runs on the slot's own worker thread,
    so async callers only await it. Evals share one Context per turn, closed by
    `end_turn`; with `fresh=True` (`call` mode) each eval gets its own Context,
    closed when it finishes. The worker and Runtime outlive turns.
    """

    def __init__(self, name: str, *, memory_limit: int, options: _REPLOptions) -> None:
        self.worker = ThreadWorker(name=name)  # starts its thread on first use
        self.runtime: Runtime | None = None
        self.repl: _ThreadREPL | None = None
        self._memory_limit = memory_limit
        self._options = options
        # Evals (or snapshots) currently using the slot. A leased slot is never
        # closed; `evict` marks it `retired` and the last release closes it.
        self.leases = 0
        self.retired = False

    def eval_sync(
        self, code: str, *, fresh: bool = False, **kwargs: Any
    ) -> EvalOutcome:
        return self.worker.run_sync(self._aeval(code, fresh=fresh, **kwargs))

    async def eval_async(
        self, code: str, *, fresh: bool = False, **kwargs: Any
    ) -> EvalOutcome:
        return await self.worker.run_async(self._aeval(code, fresh=fresh, **kwargs))

    async def _aeval(self, code: str, *, fresh: bool, **kwargs: Any) -> EvalOutcome:
        # Runs on the worker. Building the Runtime and Context never awaits, so
        # concurrent first evals on this single-threaded loop can't both build.
        if fresh:
            repl = self._new_repl()
            try:
                return await repl._aeval_async(code, **kwargs)  # noqa: SLF001
            finally:
                await repl._aclose()  # noqa: SLF001
        if self.repl is None:
            self.repl = self._new_repl()
        return await self.repl._aeval_async(code, **kwargs)  # noqa: SLF001

    def _new_repl(self) -> _ThreadREPL:
        if self.runtime is None:
            self.runtime = self._create_runtime()
        return _ThreadREPL(self.worker, self.runtime, **self._options)

    def _create_runtime(self) -> Runtime:
        return Runtime(
            memory_limit=self._memory_limit,
            transform_flags=SourceTransform.TOP_LEVEL_CONST_TO_VAR,
        )

    def create_snapshot(self) -> bytes | None:
        """Snapshot the turn's Context, or `None` if no eval ran this turn."""
        return self.repl.create_snapshot() if self.repl is not None else None

    async def acreate_snapshot(self) -> bytes | None:
        """Async variant of `create_snapshot`."""
        return await self.repl.acreate_snapshot() if self.repl is not None else None

    def end_turn(self) -> None:
        """Close the turn's Context; keep the worker and Runtime."""
        if self.repl is not None:
            self.worker.run_sync(self._aend_turn())

    async def aend_turn(self) -> None:
        """Async variant of `end_turn`."""
        if self.repl is not None:
            await self.worker.run_async(self._aend_turn())

    async def _aend_turn(self) -> None:
        repl, self.repl = self.repl, None
        if repl is not None:
            await repl._aclose()  # noqa: SLF001

    def close(self) -> None:
        """Close the Context and Runtime on the worker, then stop the worker."""
        try:
            if self.runtime is not None:
                self.worker.run_sync(self._aclose_runtime())
        finally:
            self.worker.close()

    async def aclose(self) -> None:
        """Async variant of `close`."""
        try:
            if self.runtime is not None:
                await self.worker.run_async(self._aclose_runtime())
        finally:
            await self.worker.aclose()

    async def _aclose_runtime(self) -> None:
        await self._aend_turn()
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None


@dataclass
class _Registry:
    """Per-thread slot registry: an LRU of up to `max_slots` `_Slot`s.

    Slots stay warm across turns. Over the cap, the least recently used slot
    with no eval running is closed, mid-turn or not, so slots left by failed
    or abandoned runs are reclaimed too. Nothing is built or closed while
    holding the lock.
    """

    memory_limit: int
    timeout: float | None
    capture_console: bool
    max_stdout_chars: int
    max_ptc_calls: int | None = 256
    subagents_enabled: bool = True
    max_subagent_concurrency: int = _DEFAULT_MAX_SUBAGENT_CONCURRENCY
    max_slots: int = _MAX_SLOTS
    _slots: OrderedDict[str, _Slot] = field(default_factory=OrderedDict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, thread_id: str) -> _Slot:
        """Return the slot for `thread_id`, creating it if needed."""
        with self._lock:
            slot, stale = self._get_locked(thread_id)
        for old in stale:
            old.close()
        return slot

    def acquire(self, thread_id: str) -> _Slot:
        """Lease the slot for `thread_id`, creating it if needed.

        A leased slot is never closed. Pair with `release`.
        """
        with self._lock:
            slot, stale = self._get_locked(thread_id)
            slot.leases += 1
        for old in stale:
            old.close()
        return slot

    async def aacquire(self, thread_id: str) -> _Slot:
        """Async variant of `acquire`."""
        with self._lock:
            slot, stale = self._get_locked(thread_id)
            slot.leases += 1
        for old in stale:
            await old.aclose()
        return slot

    def _get_locked(self, thread_id: str) -> tuple[_Slot, list[_Slot]]:
        slot = self._slots.get(thread_id)
        if slot is None:
            slot = self._slots[thread_id] = _Slot(
                f"quickjs-worker-{thread_id[:8]}",
                memory_limit=self.memory_limit,
                options=self._repl_options(),
            )
        self._slots.move_to_end(thread_id)
        stale = []
        excess = len(self._slots) - self.max_slots
        for key, old in list(self._slots.items()):  # least recently used first
            if excess <= 0:
                break
            # Never close a slot with an eval running on it.
            if key == thread_id or old.leases:
                continue
            del self._slots[key]
            stale.append(old)
            excess -= 1
        return slot, stale

    def lease_if_exists(self, thread_id: str) -> _Slot | None:
        """Lease the slot for `thread_id` without creating one."""
        with self._lock:
            slot = self._slots.get(thread_id)
            if slot is not None:
                slot.leases += 1
            return slot

    def release(self, slot: _Slot) -> None:
        """Return a lease; closes the slot if it was evicted meanwhile."""
        if self._release_locked(slot):
            slot.close()

    async def arelease(self, slot: _Slot) -> None:
        """Async variant of `release`."""
        if self._release_locked(slot):
            await slot.aclose()

    def _release_locked(self, slot: _Slot) -> bool:
        with self._lock:
            slot.leases -= 1
            return slot.retired and slot.leases == 0

    def end_turn(self, thread_id: str) -> None:
        """Close `thread_id`'s turn Context, keeping its worker and Runtime."""
        if (slot := self.lease_if_exists(thread_id)) is not None:
            try:
                slot.end_turn()
            finally:
                self.release(slot)

    async def aend_turn(self, thread_id: str) -> None:
        """Async variant of `end_turn`."""
        if (slot := self.lease_if_exists(thread_id)) is not None:
            try:
                await slot.aend_turn()
            finally:
                await self.arelease(slot)

    def get_if_exists(self, thread_id: str) -> _Slot | None:
        """Return the slot for `thread_id` without creating one."""
        with self._lock:
            return self._slots.get(thread_id)

    def evict(self, thread_id: str) -> None:
        """Close and remove the slot for `thread_id`.

        A leased slot is removed now and closed when its last lease ends.
        """
        with self._lock:
            slot = self._pop_for_close_locked(thread_id)
        if slot is not None:
            slot.close()

    async def aevict(self, thread_id: str) -> None:
        """Async variant of `evict`."""
        with self._lock:
            slot = self._pop_for_close_locked(thread_id)
        if slot is not None:
            await slot.aclose()

    def _pop_for_close_locked(self, thread_id: str) -> _Slot | None:
        """Remove a slot; return it if it can close now, else retire it."""
        slot = self._slots.pop(thread_id, None)
        if slot is None:
            return None
        if slot.leases:
            slot.retired = True
            return None
        return slot

    def _repl_options(self) -> _REPLOptions:
        return {
            "timeout": self.timeout,
            "capture_console": self.capture_console,
            "max_stdout_chars": self.max_stdout_chars,
            "max_ptc_calls": self.max_ptc_calls,
            "subagents_enabled": self.subagents_enabled,
            "max_subagent_concurrency": self.max_subagent_concurrency,
        }

    def close(self) -> None:
        with self._lock:
            slots = list(self._slots.values())
            self._slots.clear()
        for slot in slots:
            slot.close()
