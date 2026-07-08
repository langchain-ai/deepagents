"""Thread-keyed QuickJS REPL registry, console bridge, and result formatter.

Kept separate from `middleware.py` so the REPL mechanics stay testable
without constructing an agent or wiring up LangGraph state.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

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
    from collections.abc import Sequence

    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

logger = logging.getLogger(__name__)

_MAX_TASK_CALLS_PER_THREAD = 32
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
    """Mint a synthetic tool_call_id for a PTC-driven tool invocation.

    Tools like `task` require a non-empty `tool_call_id` to stamp
    into their emitted `ToolMessage`. The real call_id lives on the
    outer `eval` tool call; we synthesise a child id so downstream
    state (checkpointer, tracing) can correlate the PTC sub-call back
    to the REPL cell that issued it.
    """
    return f"ptc_{tool_name}_{uuid.uuid4().hex[:8]}"


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

    try:
        from langgraph.prebuilt.tool_node import (  # noqa: PLC0415 — optional dep, imported here so ImportError is catchable
            _get_all_injected_args,
        )
    except ImportError:  # pragma: no cover — langgraph always present
        return enriched

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
    """Return whether *tool* declares an `InjectedToolCallId` parameter.

    PTC invokes tools with an args dict via `BaseTool.arun`. Tools that
    declare `InjectedToolCallId` need `tool_call_id` passed as a kwarg
    so `BaseTool._parse_input`'s built-in injection runs. Detect via the
    same combination of schema annotations and `get_type_hints` that
    langgraph's `_get_all_injected_args` uses.

    Trade-off: passing `tool_call_id` as a kwarg makes
    `BaseTool._format_output` wrap the result in a `ToolMessage` with
    string-coerced `.content` (unless the tool returns a `ToolOutputMixin`
    such as `Command`). For tools without this annotation we pass
    `tool_call_id=None` and recover the native return value.
    """
    try:
        from typing import get_type_hints  # noqa: PLC0415

        from langchain_core.tools.base import (  # noqa: PLC0415
            InjectedToolCallId,
            _is_injected_arg_type,
            get_all_basemodel_annotations,
        )
    except ImportError:  # pragma: no cover — both deps are required at runtime
        return False

    try:
        schema_annotations = get_all_basemodel_annotations(tool.get_input_schema())
    except Exception:  # noqa: BLE001 — schema introspection is best-effort
        schema_annotations = {}
    func = getattr(tool, "func", None) or getattr(tool, "coroutine", None)
    try:
        func_annotations = (
            get_type_hints(func, include_extras=True) if func is not None else {}
        )
    except Exception:  # noqa: BLE001 — type-hint resolution is best-effort
        func_annotations = {}

    # Match langgraph's merge order: schema annotations override func ones.
    all_annotations = {**func_annotations, **schema_annotations}
    return any(
        _is_injected_arg_type(type_, injected_type=InjectedToolCallId)
        for type_ in all_annotations.values()
    )


def _bridge_symbol_name(tool_name: str) -> str:
    """Build a stable, JS-safe global symbol for one PTC bridge."""
    # Keep only identifier-safe characters and salt with a short hash to
    # avoid collisions when different source names sanitize similarly.
    normalized = "".join(
        c if c.isalnum() or c in {"_", "$"} else "_" for c in tool_name
    )
    if not normalized or normalized[0].isdigit():
        normalized = f"_{normalized}"
    digest = hashlib.sha256(tool_name.encode("utf-8")).hexdigest()[:8]
    return f"__tools_{normalized}_{digest}"


def _render_tools_namespace_assignment(bridges: dict[str, str]) -> str:
    """Return JS that atomically rebuilds `globalThis.tools` from bridges."""
    statements = ["globalThis.tools = {};"]
    for tool_name, bridge_symbol in sorted(bridges.items()):
        quoted_tool_name = json.dumps(tool_name)
        quoted_bridge_symbol = json.dumps(bridge_symbol)
        statements.append(
            "globalThis.tools"
            f"[{quoted_tool_name}] = globalThis[{quoted_bridge_symbol}];"
        )
    statements.append("undefined")
    return " ".join(statements)


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
        timeout: float,
        capture_console: bool,
        max_stdout_chars: int,
        max_ptc_calls: int | None = 256,
        subagents_enabled: bool = True,
    ) -> None:
        self._worker = worker
        self._runtime = runtime
        # The Context-level `timeout` is used as the cumulative budget
        # for sync evals. Async evals pass `timeout=` per call so each
        # call gets a fresh budget — matches what a REPL user expects,
        # and what we describe in the system prompt.
        self._per_call_timeout = timeout
        self._capture_console = capture_console
        # Static budget config; mutable counters live in `_ptc_state`.
        self._max_ptc_calls = max_ptc_calls
        self._subagents_enabled = subagents_enabled
        self._console = _ConsoleBuffer(max_stdout_chars)
        self._ctx: Context | None = None
        # PTC state. `_registered_tools` tracks which camel-case names
        # have already had their host-function bridge installed on the
        # QuickJS context. Host functions cannot be un-registered, so we
        # never remove entries from here — changes to the exposed set
        # are reflected by rewriting `globalThis.tools` (see
        # install_tools) to include only the currently-active subset.
        self._registered_tools: dict[str, BaseTool] = {}
        self._bridge_symbols: dict[str, str] = {}
        self._active_tool_names: frozenset[str] = frozenset()
        # Tracks whether `globalThis.tools` has been assigned at least
        # once. Distinct from `_active_tool_names` so the first call
        # with an empty tool set still installs `tools = {}` (otherwise
        # `typeof tools.X` throws ReferenceError instead of returning
        # `"undefined"`).
        self._tools_installed: bool = False
        # Mutable per-eval PTC state. Tracks call budget plus outer
        # runtime/loop dispatch context for bridge invocations. Allocated
        # at eval start and cleared in finally so bridge calls can't run
        # outside the current eval.
        self._ptc_state: _PTCState | None = None
        self._task_calls: asyncio.Semaphore | None = None
        # Context creation + console install must happen on the worker
        # thread. Block caller here so the REPL is ready to use when
        # __init__ returns.
        worker.run_sync(self._ainit())

    async def _ainit(self) -> None:
        self._ctx = self._runtime.new_context(timeout=self._per_call_timeout)
        if self._capture_console:
            self._install_console()
        if self._subagents_enabled:
            self._task_calls = asyncio.Semaphore(_MAX_TASK_CALLS_PER_THREAD)
            self._register_task_bridge()

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

        Idempotent per (camelName, tool identity). Safe to call on every
        model-call turn; we diff against the current active set and only
        (a) register new host-function bridges for tools we haven't seen
        before and (b) rewrite `globalThis.tools` when the active-name
        set changes. Hot path cost when nothing changes: one frozenset
        equality check.
        """
        self._worker.run_sync(self._ainstall_tools(tools))

    async def _ainstall_tools(self, tools: Sequence[BaseTool]) -> None:
        ctx = self._require_ctx()
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
        target_names = frozenset(name_to_tool)
        if target_names == self._active_tool_names and self._tools_installed:
            # Fast path: stable toolset, nothing to do. Keep the bridge's
            # dispatch target pointer current in case tool objects rotate
            # while keeping the same names. Guard with `_tools_installed`
            # so the empty → empty transition on first call still installs
            # a `tools = {}` global — otherwise `typeof tools.x` hits a
            # ReferenceError instead of returning "undefined".
            self._registered_tools.update(name_to_tool)
            return

        # Register host-function bridges for tools we haven't seen before.
        for camel, tool in name_to_tool.items():
            if camel not in self._registered_tools:
                self._bridge_symbols[camel] = self._register_tool_bridge(camel)
            self._registered_tools[camel] = tool

        # Rewrite globalThis.tools. Building the object inside a single
        # eval keeps assignments atomic from the model's point of view —
        # there's no moment where tools is half-populated. The trailing
        # `undefined` sidesteps the MarshalError on object returns
        # (same trick as the console install).
        bridges = {camel: self._bridge_symbols[camel] for camel in target_names}
        ctx.eval(_render_tools_namespace_assignment(bridges))
        self._active_tool_names = target_names
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
            )

        outer_loop = state.outer_loop
        if outer_loop is None:
            return await _call()
        current_loop = asyncio.get_running_loop()
        if current_loop is outer_loop:
            return await _call()
        future = asyncio.run_coroutine_threadsafe(_call(), outer_loop)
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            future.cancel()
            raise

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
            async with task_calls:
                try:
                    result = await self._ainvoke_task_on_outer_loop(
                        payload,
                        state=state,
                    )
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
        outer_loop: asyncio.AbstractEventLoop | None,
    ) -> Any:
        """Run the tool on the outer runtime's loop when available.

        Uses `BaseTool.arun(args, tool_call_id=...)` rather than
        `ainvoke(envelope)` so the result is the tool's native return
        value rather than a string-coerced `ToolMessage`. We only pass
        `tool_call_id` when the tool declares `InjectedToolCallId` —
        otherwise `_format_output` would wrap the result anyway.
        """
        args = tool_call["args"]
        tool_call_id = (
            tool_call.get("id") if _tool_uses_injected_tool_call_id(tool) else None
        )

        async def _call() -> Any:
            return await tool.arun(args, tool_call_id=tool_call_id)

        if outer_loop is None:
            return await _call()
        current_loop = asyncio.get_running_loop()
        if current_loop is outer_loop:
            return await _call()
        future = asyncio.run_coroutine_threadsafe(_call(), outer_loop)
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            future.cancel()
            raise

    def _register_tool_bridge(self, camel: str) -> str:
        """Install a host-function bridge for one camel-cased tool name.

        The bridge is async so `eval_async`'s driving loop can await
        `tool.ainvoke` without blocking the event loop. We look the
        tool up through `self._registered_tools` on every call so a
        later `install_tools` that swaps the underlying object (same
        name, different instance) is picked up without re-registration.
        """
        ctx = self._require_ctx()
        registered = self._registered_tools

        async def _bridge(raw_input: Any = None) -> Any:
            tool = registered.get(camel)
            if tool is None:
                # Shouldn't happen — we only rewrite `globalThis.tools`
                # with names currently in the map — but if a race causes
                # it, fail loud.
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
            # Inject runtime/state/store ourselves; `InjectedToolCallId`
            # is handled inside `_ainvoke_tool_on_outer_loop` via
            # `tool.arun(..., tool_call_id=...)`. The bridge intentionally
            # avoids the tool-call envelope path because it wraps the
            # result in a `ToolMessage` and string-coerces `.content`,
            # destroying native return types (lists, dicts, numbers).
            args = _inject_tool_args_for_ptc(
                tool, payload, state.outer_runtime, call_id
            )
            result = await self._ainvoke_tool_on_outer_loop(
                tool,
                {"name": tool.name, "args": args, "id": call_id, "type": "tool_call"},
                outer_loop=state.outer_loop,
            )
            return coerce_tool_output_for_ptc(result)

        bridge_symbol = _bridge_symbol_name(camel)
        ctx.register(bridge_symbol, _bridge, is_async=True)
        return bridge_symbol

    def eval_sync(
        self,
        code: str,
        *,
        outer_runtime: ToolRuntime | None = None,
    ) -> EvalOutcome:
        # Both sync and async entry points funnel through ctx.eval_async on
        # the worker loop. Sync ctx.eval can't dispatch async host functions
        # (PTC bridges are is_async=True), so routing sync callers through
        # the async path is required for PTC to work under sync invocation.
        return self._worker.run_sync(
            self._aeval_async(
                code,
                outer_runtime=outer_runtime,
            )
        )

    async def eval_async(
        self,
        code: str,
        *,
        outer_runtime: ToolRuntime | None = None,
        outer_loop: asyncio.AbstractEventLoop | None = None,
    ) -> EvalOutcome:
        return await self._worker.run_async(
            self._aeval_async(
                code,
                outer_runtime=outer_runtime,
                outer_loop=outer_loop,
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

    async def _aeval_async(  # noqa: C901, PLR0912, PLR0915
        self,
        code: str,
        *,
        outer_runtime: ToolRuntime | None = None,
        outer_loop: asyncio.AbstractEventLoop | None = None,
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
        )
        try:
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


@dataclass
class _Slot:
    """One LangGraph thread's private QuickJS stack: worker + Runtime + REPL.

    Each slot owns an OS thread (via `ThreadWorker`) and a Runtime. This
    keeps per-conversation JS execution on its own event loop so one
    user's slow computation can't block others.
    """

    worker: ThreadWorker
    runtime: Runtime
    repl: _ThreadREPL


@dataclass
class _Registry:
    """Per-thread Runtime registry.

    Each LangGraph `thread_id` gets its own `_Slot` (worker + Runtime
    + Context). Eviction is driven externally via `evict(thread_id)` —
    typically from the middleware's `after_agent` hook.
    """

    memory_limit: int
    timeout: float
    capture_console: bool
    max_stdout_chars: int
    max_ptc_calls: int | None = 256
    subagents_enabled: bool = True
    _slots: dict[str, _Slot] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, thread_id: str) -> _ThreadREPL:
        with self._lock:
            slot = self._slots.get(thread_id)
            if slot is None:
                slot = self._build_slot_locked(thread_id)
                self._slots[thread_id] = slot
            return slot.repl

    def get_if_exists(self, thread_id: str) -> _ThreadREPL | None:
        """Return existing REPL for `thread_id` without creating a new slot."""
        with self._lock:
            slot = self._slots.get(thread_id)
            return slot.repl if slot is not None else None

    def evict(self, thread_id: str) -> None:
        """Close and remove the slot for `thread_id`. No-op if absent."""
        with self._lock:
            slot = self._slots.pop(thread_id, None)
        if slot is not None:
            self._close_slot(slot)

    async def aevict(self, thread_id: str) -> None:
        """Async variant of `evict`: closes the runtime via the worker loop."""
        with self._lock:
            slot = self._slots.pop(thread_id, None)
        if slot is not None:
            await self._aclose_slot(slot)

    def reset_repl(self, thread_id: str) -> None:
        """Replace the slot REPL while keeping its worker and runtime alive."""
        with self._lock:
            slot = self._slots.get(thread_id)
        if slot is None:
            return

        with contextlib.suppress(Exception):
            slot.repl.close()
        new_repl = _ThreadREPL(
            slot.worker,
            slot.runtime,
            timeout=self.timeout,
            capture_console=self.capture_console,
            max_stdout_chars=self.max_stdout_chars,
            max_ptc_calls=self.max_ptc_calls,
            subagents_enabled=self.subagents_enabled,
        )

        with self._lock:
            current = self._slots.get(thread_id)
            if current is slot:
                slot.repl = new_repl
                return
        # Slot was removed/replaced while rebuilding.
        with contextlib.suppress(Exception):
            new_repl.close()

    def _build_slot_locked(self, thread_id: str) -> _Slot:
        name = f"quickjs-worker-{thread_id[:8]}"
        worker = ThreadWorker(name=name)
        runtime = worker.run_sync(self._acreate_runtime())
        repl = _ThreadREPL(
            worker,
            runtime,
            timeout=self.timeout,
            capture_console=self.capture_console,
            max_stdout_chars=self.max_stdout_chars,
            max_ptc_calls=self.max_ptc_calls,
            subagents_enabled=self.subagents_enabled,
        )
        return _Slot(worker=worker, runtime=runtime, repl=repl)

    def _close_slot(self, slot: _Slot) -> None:
        # Close the context on its owning worker thread before closing the
        # runtime. This avoids unsendable handle wrappers being finalized on
        # a non-owner thread during later GC.
        with contextlib.suppress(Exception):
            slot.repl.close()
        # Best-effort; never block shutdown on a misbehaving runtime.
        with contextlib.suppress(Exception):
            slot.worker.run_sync(_aclose_runtime(slot.runtime))
        slot.worker.close()

    async def _aclose_slot(self, slot: _Slot) -> None:
        with contextlib.suppress(Exception):
            await slot.repl.aclose()
        with contextlib.suppress(Exception):
            await slot.worker.run_async(_aclose_runtime(slot.runtime))
        slot.worker.close()

    async def _acreate_runtime(self) -> Runtime:
        return Runtime(
            memory_limit=self.memory_limit,
            transform_flags=SourceTransform.TOP_LEVEL_CONST_TO_VAR,
        )

    def close(self) -> None:
        with self._lock:
            slots = list(self._slots.values())
            self._slots.clear()
        for slot in slots:
            self._close_slot(slot)


async def _aclose_runtime(runtime: Runtime) -> None:
    runtime.close()
