"""Slot-keyed QuickJS REPL registry, console bridge, and result formatter.

Kept separate from `middleware.py` so the REPL mechanics stay testable
without constructing an agent or wiring up LangGraph state.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.runnables import RunnableLambda
from langgraph.errors import GraphInterrupt
from quickjs_rs import (
    ConcurrentEvalError,
    Context as QjsContext,
    DeadlockError,
    HostCancellationError,
    JSError,
    MarshalError,
    MemoryLimitError,
    Runtime as QjsRuntime,
    Snapshot as QjsSnapshot,
    SourceTransform,
    ThreadWorker,
    TimeoutError as QjsTimeoutError,
)

from langchain_quickjs import _fs
from langchain_quickjs._format import (
    coerce_tool_output_for_ptc,
    format_handle,
    stringify,
)
from langchain_quickjs._ptc import (
    TOOLS_CALL,
    TOOLS_NAMES,
    PTCCallBudgetExceededError,
    PTCState,
    UnsupportedCommandError,
    inject_tool_args,
    is_valid_js_identifier,
    normalize_tool_input,
    render_tools_namespace,
    send_state_updates,
    synth_tool_call_id,
    to_camel_case,
    uses_injected_tool_call_id,
)
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
# Warm slots kept per process. A slot is closed once `_MAX_SLOTS` other
# slots have evaled since its last eval, even mid-turn; its next eval then
# restores the last persisted snapshot and loses that turn's earlier state.
# That takes a busy process: langgraph-api runs 10 concurrent runs per worker
# by default.
_MAX_SLOTS = 32

PersistenceMode = Literal["thread", "turn", "call"]
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


class _TaskBridgeError(RuntimeError):
    """Wrap errors from the top-level `task()` host function."""

    def __init__(self, exc: Exception) -> None:
        self.error_type = type(exc).__name__
        self.error_message = str(exc)
        super().__init__(self.error_message)


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


@dataclass(frozen=True)
class _SlotConfig:
    """Settings for the interpreters one middleware creates."""

    memory_limit: int | None = 64 * 1024 * 1024
    timeout: float | None = 5.0
    capture_console: bool = True
    max_stdout_chars: int = 4000
    max_ptc_calls: int | None = 256
    subagents_enabled: bool = True
    max_subagent_concurrency: int = _DEFAULT_MAX_SUBAGENT_CONCURRENCY


class _Slot:
    """One LangGraph thread's QuickJS worker, Runtime and Context.

    Builds nothing until the first eval. `quickjs_rs` objects are `!Send`, so
    all QuickJS work runs on the slot's own worker thread; callers on any
    thread or loop only wait on it. Evals run one at a time: one that starts
    while another is running fails with `ConcurrentEval`. The worker and
    Runtime outlive turns; the Context lives until `close_context`, and stays
    warm across turns while it matches the thread's last snapshot. Persistence
    modes are the middleware's business.
    """

    def __init__(self, name: str, config: _SlotConfig) -> None:
        self._worker = ThreadWorker(name=name)  # starts its thread on first use
        self._config = config
        self._runtime: QjsRuntime | None = None
        self._context: QjsContext | None = None
        # The Context-level `timeout` is used as the cumulative budget
        # for sync evals. Async evals pass `timeout=` per call so each
        # call gets a fresh budget — matches what a REPL user expects,
        # and what we describe in the system prompt.
        self._per_call_timeout = (
            float("inf") if config.timeout is None else config.timeout
        )
        self._task_calls = asyncio.Semaphore(config.max_subagent_concurrency)
        # Whether an eval is running. Evals run one at a time.
        self._eval_running = False
        # The snapshot the live Context still matches: set when one is taken,
        # cleared by the next eval. Kept whole (not hashed): comparing it is
        # far cheaper than hashing a multi-MB payload every turn.
        self._snapshot: bytes | None = None
        # Mutable per-eval PTC state. Tracks call budget plus outer
        # runtime/loop dispatch context for bridge invocations. Set while an
        # eval runs, so bridge calls can't run outside one.
        self._ptc_state: PTCState | None = None

    @property
    def eval_running(self) -> bool:
        """Whether an eval is running, even one whose caller was cancelled."""
        return self._eval_running

    def _ensure_runtime(self) -> QjsRuntime:
        """Return the Runtime, building it if needed. Runs on the worker."""
        if self._runtime is None:
            self._runtime = QjsRuntime(
                memory_limit=self._config.memory_limit,
                transform_flags=SourceTransform.TOP_LEVEL_CONST_TO_VAR,
            )
        return self._runtime

    def _ensure_context(self) -> QjsContext:
        """Return the live Context, building it (and the Runtime) if needed.

        Runs on the worker and never awaits, so concurrent first evals on its
        single-threaded loop can't both build.
        """
        if self._context is not None:
            return self._context
        self._context = self._ensure_runtime().new_context(
            timeout=self._per_call_timeout
        )
        self._console = _ConsoleBuffer(self._config.max_stdout_chars)
        # A Context restores a snapshot at most once.
        self._snapshot_restored = False
        # PTC tools by camel-case name. `install_tools` swaps this map; the
        # guest reaches it through `__tools_call` / `__tools_names`.
        self._tools: dict[str, BaseTool] = {}
        # Whether `globalThis.tools` has been defined. Without PTC it stays
        # undefined.
        self._tools_installed = False
        if self._config.capture_console:
            self._install_console()
        self._register_tools_bridge()
        if self._config.subagents_enabled:
            self._register_task_bridge()
        self._register_fs_bridge()
        return self._context

    def _require_context(self) -> QjsContext:
        """Return the live QuickJS context or raise if this REPL is closed."""
        if self._context is None:
            msg = "QuickJS context is closed"
            raise RuntimeError(msg)
        return self._context

    def _install_console(self) -> None:
        ctx = self._require_context()
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
            self._require_context().eval(render_tools_namespace())
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
        state: PTCState,
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
        ctx = self._require_context()

        async def _bridge(raw_input: Any = None) -> Any:
            state = self._ptc_state
            if state is None:
                msg = "task bridge called outside active eval"
                raise ConcurrentEvalError(msg)
            task_calls = self._task_calls

            payload = normalize_tool_input(raw_input)
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
                    tool_call_id if uses_injected_tool_call_id(tool) else None
                ),
            )

        return await _on_outer_loop(_call, outer_loop)

    def _register_tools_bridge(self) -> None:
        """Install the host functions behind the guest's `tools` proxy.

        `_call` looks the tool up in `self._tools` on every call, so
        `install_tools` swapping the map is picked up without touching
        the guest.
        """
        ctx = self._require_context()

        def _names() -> list[str]:
            return list(self._tools)

        async def _call(camel: str, raw_input: Any = None) -> Any:
            tool = self._tools.get(camel)
            if tool is None:
                # A function captured before `install_tools` dropped it. Fail
                # the eval as calling a missing `tools.*` would, not the run.
                msg = f"tools.{camel} is no longer available"
                raise _TaskBridgeError(TypeError(msg))
            if self._ptc_state is None:
                msg = "PTC bridge called outside active eval"
                raise ConcurrentEvalError(msg)
            state = self._ptc_state.consume_call_budget(
                function_name=f"tools.{camel}",
                max_ptc_calls=self._config.max_ptc_calls,
            )
            self._ptc_state = state
            payload = normalize_tool_input(raw_input)
            call_id = synth_tool_call_id(tool.name)
            # Inject runtime/state/store ourselves. The bridge uses `arun`
            # rather than the tool-call envelope path so the result can be
            # unwrapped back to native JS-visible values after the standard
            # tool lifecycle callbacks have fired.
            args = inject_tool_args(tool, payload, state.outer_runtime, call_id)
            result = await self._ainvoke_tool_on_outer_loop(
                tool,
                {"name": tool.name, "args": args, "id": call_id, "type": "tool_call"},
                outer_runtime=state.outer_runtime,
                outer_loop=state.outer_loop,
            )
            send_state_updates(state.outer_runtime, f"tools.{camel}", result)
            return coerce_tool_output_for_ptc(result)

        ctx.register(TOOLS_NAMES, _names)
        ctx.register(TOOLS_CALL, _call, is_async=True)

    def _register_fs_bridge(self) -> None:
        ctx = self._require_context()
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
            function_name=name, max_ptc_calls=self._config.max_ptc_calls
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

    async def _aprepare_for_eval(
        self, tools: Sequence[BaseTool] | None, fs: BackendProtocol | None
    ) -> None:
        """Bind the current tools and `fs` on the owning worker."""
        if tools is not None:
            await self._ainstall_tools(tools)
        if fs is not None:
            self._require_context().eval(_fs.install_source())

    async def _aeval_async(  # noqa: C901, PLR0912, PLR0915
        self,
        code: str,
        *,
        outer_runtime: ToolRuntime | None = None,
        tools: Sequence[BaseTool] | None = None,
        outer_loop: asyncio.AbstractEventLoop | None = None,
        fs: BackendProtocol | None = None,
        close_context_after: bool = False,
    ) -> EvalOutcome:
        """Run `code` in the Context. Runs on the worker.

        With `close_context_after`, the Context is closed on every exit before
        the next eval can start, so that eval gets a fresh one.

        An eval that starts while another is running fails with
        `ConcurrentEval`. We intentionally do not queue: a model dispatching
        overlapping evals is almost always a prompting bug, and a loud
        failure is a better signal than silent serialisation.
        """
        if self._eval_running:
            return EvalOutcome(
                error_type="ConcurrentEval",
                error_message=(
                    "another eval call is still running; make one call at a "
                    "time and use Promise.allSettled inside it for parallel work"
                ),
            )
        ctx = self._ensure_context()
        outcome = EvalOutcome()
        self._eval_running = True
        self._snapshot = None
        self._ptc_state = PTCState(
            remaining_calls=self._config.max_ptc_calls,
            outer_runtime=outer_runtime,
            outer_loop=outer_loop,
            fs=fs,
        )
        try:
            await self._aprepare_for_eval(tools, fs)
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
        except PTCCallBudgetExceededError as e:
            # Raised from inside the PTC bridge; quickjs-rs propagates the
            # original exception out of eval_handle_async / await_promise.
            # Surface it as a distinct, model-recoverable error so the
            # agent can shorten its script rather than crash.
            outcome.error_type = "PTCCallBudgetExceeded"
            outcome.error_message = e.render_message()
            _clear_exception_references(e)
        except UnsupportedCommandError as e:
            outcome.error_type = "UnsupportedCommand"
            outcome.error_message = str(e)
            _clear_exception_references(e)
        except QjsTimeoutError as e:
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
            self._ptc_state = None
            outcome.stdout, outcome.stdout_truncated_chars = self._console.drain()
            if close_context_after:
                self._dispose_context()
            self._eval_running = False
        return outcome

    def _record_js_error(self, outcome: EvalOutcome, e: JSError) -> None:
        outcome.error_type = e.name
        outcome.error_message = e.message
        outcome.error_stack = e.stack

    def _dispose_context(self) -> None:
        self._snapshot = None
        ctx, self._context = self._context, None
        if ctx is not None:
            ctx.close()

    def eval_sync(self, code: str, **kwargs: Any) -> EvalOutcome:
        # Both sync and async entry points funnel through ctx.eval_async on
        # the worker loop. Sync ctx.eval can't dispatch async host functions
        # (PTC bridges are is_async=True), so routing sync callers through
        # the async path is required for PTC to work under sync invocation.
        return self._worker.run_sync(self._aeval_async(code, **kwargs))

    async def eval_async(self, code: str, **kwargs: Any) -> EvalOutcome:
        return await self._worker.run_async(self._aeval_async(code, **kwargs))

    def restore_snapshot(self, payload: bytes) -> None:
        """Restore a snapshot into the Context, once per Context.

        Raises:
            _SnapshotRestoreError: If the payload can't be restored.
        """
        self._worker.run_sync(self._arestore_snapshot(payload))

    async def arestore_snapshot(self, payload: bytes) -> None:
        """Async variant of `restore_snapshot`."""
        await self._worker.run_async(self._arestore_snapshot(payload))

    async def _arestore_snapshot(self, payload: bytes) -> None:
        context = self._ensure_context()
        if self._snapshot_restored:
            return
        self._snapshot_restored = True
        try:
            self._ensure_runtime().restore_snapshot(
                QjsSnapshot.from_bytes(payload), context, inject_globals=True
            )
        except Exception as exc:  # best-effort restore path
            logger.warning("Failed to restore QuickJS snapshot", exc_info=True)
            msg = "Failed to restore QuickJS snapshot"
            raise _SnapshotRestoreError(msg) from exc

    def create_snapshot(self) -> bytes | None:
        """Snapshot the Context, or `None` if no eval changed it since the last."""
        if self._context is None or self._snapshot is not None:
            return None
        return self._worker.run_sync(self._acreate_snapshot())

    async def acreate_snapshot(self) -> bytes | None:
        """Async variant of `create_snapshot`."""
        if self._context is None or self._snapshot is not None:
            return None
        return await self._worker.run_async(self._acreate_snapshot())

    async def _acreate_snapshot(self) -> bytes | None:
        if self._context is None or self._snapshot is not None:
            return None
        payload = self._context.create_snapshot().to_bytes()
        # The Context now is this snapshot: it needs no restore to match it.
        self._snapshot = payload
        self._snapshot_restored = True
        return payload

    def close_context(self, *, keep_snapshot: bytes | None = None) -> None:
        """Close the Context and its REPL state; keep the worker and Runtime.

        A Context that still matches `keep_snapshot` (no eval since it was
        taken) stays open, so the next turn skips the restore.
        """
        if self._context is not None:
            self._worker.run_sync(self._aclose_context(keep_snapshot))

    async def aclose_context(self, *, keep_snapshot: bytes | None = None) -> None:
        """Async variant of `close_context`."""
        if self._context is not None:
            await self._worker.run_async(self._aclose_context(keep_snapshot))

    async def _aclose_context(self, keep_snapshot: bytes | None) -> None:
        # A forked subagent shares this slot; it must not close the Context
        # under a parent eval that is still running.
        if self._eval_running:
            return
        if keep_snapshot is not None and keep_snapshot == self._snapshot:
            return
        self._dispose_context()

    def close(self) -> None:
        """Close the Context and Runtime on the worker, then stop the worker."""
        try:
            if self._runtime is not None:
                self._worker.run_sync(self._aclose())
        finally:
            self._worker.close()

    async def aclose(self) -> None:
        """Async variant of `close`."""
        try:
            if self._runtime is not None:
                await self._worker.run_async(self._aclose())
        finally:
            await self._worker.aclose()

    async def _aclose(self) -> None:
        self._dispose_context()
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None


class _Registry:
    """Process-wide LRU of up to `max_slots` warm slots.

    Slots are keyed by slot id and config, so middlewares with different
    settings never share one. Over the cap, the least recently used slot with
    no eval running is closed, mid-turn or not, so slots left by failed or
    abandoned runs are reclaimed too. Nothing is built or closed while holding
    the lock.
    """

    def __init__(self, max_slots: int = _MAX_SLOTS) -> None:
        self._max_slots = max_slots
        self._slots: OrderedDict[tuple[str, _SlotConfig], _Slot] = OrderedDict()
        # Evals (or snapshots) running on each slot. A leased slot is never
        # evicted.
        self._leases: Counter[_Slot] = Counter()
        self._lock = threading.Lock()

    def acquire(self, slot_id: str, config: _SlotConfig) -> _Slot:
        """Lease the slot for `slot_id`, creating it if needed.

        A leased slot is never evicted. Pair with `release`.
        """
        slot, stale = self._acquire_locked(slot_id, config)
        for old in stale:
            old.close()
        return slot

    async def aacquire(self, slot_id: str, config: _SlotConfig) -> _Slot:
        """Async variant of `acquire`."""
        slot, stale = self._acquire_locked(slot_id, config)
        for old in stale:
            await old.aclose()
        return slot

    def _acquire_locked(
        self, slot_id: str, config: _SlotConfig
    ) -> tuple[_Slot, list[_Slot]]:
        key = (slot_id, config)
        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                slot = _Slot(f"quickjs-worker-{slot_id[:8]}", config)
                self._slots[key] = slot
            self._slots.move_to_end(key)
            self._leases[slot] += 1
            stale = []
            excess = len(self._slots) - self._max_slots
            for other, old in list(self._slots.items()):  # least recent first
                if excess <= 0:
                    break
                # Never close a slot with an eval running on it. A cancelled
                # caller drops its lease before the JS on the worker stops.
                if self._leases[old] or old.eval_running:
                    continue
                del self._slots[other]
                stale.append(old)
                excess -= 1
            return slot, stale

    def lease_if_exists(self, slot_id: str, config: _SlotConfig) -> _Slot | None:
        """Lease the slot for `slot_id` without creating one."""
        with self._lock:
            slot = self._slots.get((slot_id, config))
            if slot is not None:
                self._leases[slot] += 1
            return slot

    def release(self, slot: _Slot) -> None:
        """Return a lease taken by `acquire` or `lease_if_exists`."""
        with self._lock:
            self._leases[slot] -= 1
            if not self._leases[slot]:
                del self._leases[slot]

    def close_context(
        self, slot_id: str, config: _SlotConfig, *, keep_snapshot: bytes | None = None
    ) -> None:
        """Close `slot_id`'s Context, keeping the slot warm (see `_Slot`)."""
        if (slot := self.lease_if_exists(slot_id, config)) is not None:
            try:
                slot.close_context(keep_snapshot=keep_snapshot)
            finally:
                self.release(slot)

    async def aclose_context(
        self, slot_id: str, config: _SlotConfig, *, keep_snapshot: bytes | None = None
    ) -> None:
        """Async variant of `close_context`."""
        if (slot := self.lease_if_exists(slot_id, config)) is not None:
            try:
                await slot.aclose_context(keep_snapshot=keep_snapshot)
            finally:
                self.release(slot)


_REGISTRY = _Registry()
