"""Thread-keyed QuickJS REPL registry, console bridge, and result formatter.

Kept separate from ``middleware.py`` so the REPL mechanics stay testable
without constructing an agent or wiring up LangGraph state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from quickjs_rs import (
    UNDEFINED,
    ConcurrentEvalError,
    Context,
    DeadlockError,
    HostCancellationError,
    HostError,
    JSError,
    MarshalError,
    MemoryLimitError,
    Runtime,
)
from quickjs_rs import (
    TimeoutError as QJSTimeoutError,
)

from deepagents_repl._ptc import to_camel_case

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepagents.backends.protocol import BackendProtocol
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

logger = logging.getLogger(__name__)

# Sentinel returned by the formatter when the underlying value was a
# function/circular ref that couldn't be auto-marshaled. We format it as
# a handle-shaped result so the model sees "you got back a function" rather
# than nothing.
_HANDLE_PLACEHOLDER = "[unmarshalable value]"

_TRUNCATE_MARKER = "… [truncated {n} chars]"


@dataclass
class EvalOutcome:
    """Normalized result of a single REPL eval.

    Exactly one of ``result`` / ``error`` is meaningful per call; ``stdout``
    is collected from ``console.*`` regardless.
    """

    stdout: str = ""
    result: str | None = None
    result_kind: str | None = None  # "handle" when marshaling fell back
    error_type: str | None = None
    error_message: str = ""
    error_stack: str | None = None


class _ConsoleBuffer:
    """Accumulates ``console.*`` output between evals.

    Shared by the three host functions we install on each context. We don't
    bother distinguishing log/warn/error in the output format — the model
    does not care about the level, and flattening keeps the returned string
    smaller.
    """

    def __init__(self) -> None:
        self._lines: list[str] = []

    def append(self, level: str, args: tuple[Any, ...]) -> None:
        del level  # flattened; see class docstring
        self._lines.append(" ".join(_stringify(a) for a in args))

    def drain(self) -> str:
        if not self._lines:
            return ""
        out = "\n".join(self._lines)
        self._lines.clear()
        return out


def _format_handle(handle: Any) -> str:
    """Describe a ``Handle`` value in REPL-style shorthand.

    Caller owns the handle's lifetime; we only read from it.
    """
    kind = handle.type_of
    if kind == "function":
        # Arity is convenient context when the model wants to call the
        # thing back. Fall back gracefully if .length is absent.
        try:
            arity_h = handle.get("length")
            try:
                arity = arity_h.to_python()
            finally:
                arity_h.dispose()
            return f"[Function] arity={arity}"
        except Exception:  # noqa: BLE001 — best-effort
            return "[Function]"
    return f"[{kind}]"


def _stringify(value: Any) -> str:
    """Best-effort string form for a console arg or eval result.

    QuickJS auto-marshals primitives and plain objects through msgpack, so
    everything we see here is already a Python value. Formatting choices
    match Node's REPL rather than Python's ``repr``:

    - ``None`` → ``"null"`` (the model expects JS-shaped output)
    - ``UNDEFINED`` → ``"undefined"``
    - Booleans → ``"true"`` / ``"false"``
    - Whole-valued floats → integer form (``42.0`` → ``"42"``). JS has no
      integer type, so every ``1 + 1`` comes back as a float; without this
      the model sees ``42.0`` where a human would expect ``42``. Applied
      recursively inside lists and dicts.
    """
    return _format_jsvalue(value)


def _format_jsvalue(value: Any) -> str:
    if value is None:
        return "null"
    if value is UNDEFINED:
        return "undefined"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # Preserve ±inf / NaN: .is_integer() returns False for NaN and inf,
        # so they fall through to str().
        if value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, str):
        # Top-level strings render bare (matches what a REPL user expects
        # when they eval a string expression); nested strings get quoted
        # so ``[1, "a"]`` renders as ``[1, "a"]`` not ``[1, a]``.
        return value
    if isinstance(value, list):
        return "[" + ", ".join(_format_nested(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_format_nested(v)}" for k, v in value.items()) + "}"
    return repr(value)


def _format_nested(value: Any) -> str:
    """Like ``_format_jsvalue`` but quotes nested strings."""
    if isinstance(value, str):
        return f'"{value}"'
    return _format_jsvalue(value)


def _normalize_tool_input(raw: Any) -> dict[str, Any]:
    """Coerce whatever JS passed into ``tools.X(...)`` to a dict.

    LangChain tools accept a dict. QuickJS marshals JS objects to dicts
    already; we just want to guard against the model passing ``null``,
    ``undefined``, a bare string, or a number (none of which a well-
    formed tool call should produce, but the model is the model).
    """
    if raw is None or raw is UNDEFINED:
        return {}
    if isinstance(raw, dict):
        return raw
    # Bare scalar / list — wrap under a conventional key so the tool's
    # schema validation produces an informative error rather than a
    # silent miss.
    return {"input": raw}


def _coerce_tool_output(value: Any) -> str:
    """Tools return arbitrary Python; JS-side users expect a string.

    Handles three shapes:

    - ``str`` — pass through unchanged.
    - ``langgraph.types.Command`` — the shape ``task`` / subagent tools
      return. Extract the last ``ToolMessage`` content from
      ``command.update["messages"]`` since that's what the parent agent
      would normally see; the state update itself is intentionally
      dropped — PTC calls happen inside a JS ``await`` and we have
      nowhere to funnel a state mutation back into the parent graph.
    - everything else — ``json.dumps`` for faithful JSON → JS parseable
      round-tripping, falling back to ``str`` on non-serialisable
      values.
    """
    if isinstance(value, str):
        return value
    # Delayed import: langgraph is always present as a deepagents-repl
    # transitive dep, but keeping the import local means this file can
    # still be imported in environments where the type isn't needed.
    try:
        from langgraph.types import Command as _Command
    except ImportError:  # pragma: no cover — we always ship langgraph
        _Command = None  # type: ignore[assignment]
    if _Command is not None and isinstance(value, _Command):
        update = value.update
        if isinstance(update, dict):
            messages = update.get("messages")
            if messages:
                last = messages[-1]
                content = getattr(last, "content", None)
                if isinstance(content, str):
                    return content
        # No extractable message — stringify the update for debuggability.
        return str(update)
    # When we invoke with a ToolCall-shaped input, BaseTool wraps the
    # return value in a ToolMessage. Unwrap its content so the JS side
    # sees the raw tool output, not a Python repr of the envelope.
    try:
        from langchain_core.messages import ToolMessage as _ToolMessage
    except ImportError:  # pragma: no cover — langchain always present
        _ToolMessage = None  # type: ignore[assignment]
    if _ToolMessage is not None and isinstance(value, _ToolMessage):
        content = value.content
        if isinstance(content, str):
            return content
        try:
            return json.dumps(content, default=str)
        except (TypeError, ValueError):
            return str(content)
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _synth_tool_call_id(tool_name: str) -> str:
    """Mint a synthetic tool_call_id for a PTC-driven tool invocation.

    Tools like ``task`` require a non-empty ``tool_call_id`` to stamp
    into their emitted ``ToolMessage``. The real call_id lives on the
    outer ``eval`` tool call; we synthesise a child id so downstream
    state (checkpointer, tracing) can correlate the PTC sub-call back
    to the REPL cell that issued it.
    """
    import uuid

    return f"ptc_{tool_name}_{uuid.uuid4().hex[:8]}"


def _inject_tool_args_for_ptc(
    tool: Any,
    payload: dict[str, Any],
    outer_runtime: Any,
    tool_call_id: str,
) -> dict[str, Any]:
    """Mirror LangGraph's ``ToolNode._inject_tool_args`` for PTC calls.

    LangChain tools that declare ``ToolRuntime`` / ``InjectedState`` /
    ``InjectedStore`` only see those values when a real ``ToolNode``
    wires them in. PTC calls bypass the ToolNode, so we replicate the
    detection logic here. The outer runtime (captured from the active
    ``eval`` tool invocation) provides state/store/context/config;
    ``tool_call_id`` is freshly minted per sub-call.
    """
    try:
        from langgraph.prebuilt.tool_node import _get_all_injected_args
    except ImportError:  # pragma: no cover — langgraph always present
        return payload

    injected = _get_all_injected_args(tool)
    if not injected or outer_runtime is None:
        return payload

    # Build a ToolRuntime matching the outer one but with a fresh
    # tool_call_id. ``type(outer_runtime)`` rather than a literal import
    # so the shape stays in lockstep with whatever langgraph ships.
    derived = type(outer_runtime)(
        state=outer_runtime.state,
        tool_call_id=tool_call_id,
        config=outer_runtime.config,
        context=outer_runtime.context,
        store=outer_runtime.store,
        stream_writer=outer_runtime.stream_writer,
        execution_info=getattr(outer_runtime, "execution_info", None),
        server_info=getattr(outer_runtime, "server_info", None),
    )

    enriched = dict(payload)
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


class _ThreadREPL:
    """One QuickJS context + console buffer + lock, per LangGraph thread."""

    def __init__(
        self,
        runtime: Runtime,
        *,
        timeout: float,
        capture_console: bool,
    ) -> None:
        # The Context-level ``timeout`` is used as the cumulative budget
        # for sync evals. Async evals pass ``timeout=`` per call so each
        # call gets a fresh budget — matches what a REPL user expects,
        # and what we describe in the system prompt.
        self._ctx: Context = runtime.new_context(timeout=timeout)
        self._per_call_timeout = timeout
        self._console = _ConsoleBuffer()
        # Two locks because async and sync entry points both exist:
        #   - ``_async_lock`` serializes queued concurrent async calls on
        #     the same context. QuickJS raises ``ConcurrentEvalError``
        #     rather than queueing; we prefer to queue, so hold a lock
        #     around ``ctx.eval_async`` to guarantee only one is in
        #     flight at a time.
        #   - ``_sync_lock`` exists for the same reason on the sync path,
        #     which will usually be used from a worker thread.
        # A call from the sync path *can* race a call from the async
        # path. Contexts aren't reentrant; we rely on the natural
        # ordering of "the agent's tool handler picks one path or the
        # other per call" and don't try to coordinate across both
        # locks. If that assumption breaks, add a single threading.Lock
        # that both paths acquire.
        self._async_lock = asyncio.Lock()
        self._sync_lock = threading.Lock()
        # PTC state. ``_registered_tools`` tracks which camel-case names
        # have already had their host-function bridge installed on the
        # QuickJS context. Host functions cannot be un-registered, so we
        # never remove entries from here — changes to the exposed set
        # are reflected by rewriting ``globalThis.tools`` (see
        # install_tools) to include only the currently-active subset.
        self._registered_tools: dict[str, BaseTool] = {}
        self._active_tool_names: frozenset[str] = frozenset()
        # Tracks whether ``globalThis.tools`` has been assigned at least
        # once. Distinct from ``_active_tool_names`` so the first call
        # with an empty tool set still installs ``tools = {}`` (otherwise
        # ``typeof tools.X`` throws ReferenceError instead of returning
        # ``"undefined"``).
        self._tools_installed: bool = False
        # Outer ToolRuntime captured for the current eval. PTC bridges
        # forward it into their tool calls so `task`/subagent tools see
        # graph state, store, context, etc. Set via ``set_outer_runtime``
        # from the middleware's tool handler immediately before eval.
        self._outer_runtime: ToolRuntime | None = None
        # Fired when the current eval_async times out or is cancelled;
        # read by host bridges so pending work can short-circuit cleanly
        # (in-flight ainvoke calls are unwound via asyncio task
        # cancellation propagating from the outer wait_for).
        self._eval_cancel_event: asyncio.Event | None = None
        # Path → content buffer for file writes initiated inside an eval.
        # Swarm bridges that need read-after-write semantics inside a
        # single eval (e.g. `swarm.create` followed by `swarm.execute`)
        # push here, then the middleware flushes after the eval returns.
        self._pending_writes: dict[str, str] = {}
        if capture_console:
            self._install_console()

    def _install_console(self) -> None:
        ctx = self._ctx
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
        # global object under its given name; ``globalThis.console`` needs
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
        """Expose ``tools`` as ``globalThis.tools.<camelCase>`` in the REPL.

        Idempotent per (camelName, tool identity). Safe to call on every
        model-call turn; we diff against the current active set and only
        (a) register new host-function bridges for tools we haven't seen
        before and (b) rewrite ``globalThis.tools`` when the active-name
        set changes. Hot path cost when nothing changes: one frozenset
        equality check.
        """
        ctx = self._ctx
        name_to_tool: dict[str, BaseTool] = {to_camel_case(t.name): t for t in tools}
        target_names = frozenset(name_to_tool)
        if target_names == self._active_tool_names and self._tools_installed:
            # Fast path: stable toolset, nothing to do. Keep the bridge's
            # dispatch target pointer current in case tool objects rotate
            # while keeping the same names. Guard with ``_tools_installed``
            # so the empty → empty transition on first call still installs
            # a ``tools = {}`` global — otherwise ``typeof tools.x`` hits a
            # ReferenceError instead of returning "undefined".
            self._registered_tools.update(name_to_tool)
            return

        # Register host-function bridges for tools we haven't seen before.
        for camel, tool in name_to_tool.items():
            if camel not in self._registered_tools:
                self._register_tool_bridge(camel)
            self._registered_tools[camel] = tool

        # Rewrite globalThis.tools. Building the object inside a single
        # eval keeps assignments atomic from the model's point of view —
        # there's no moment where tools is half-populated. The trailing
        # ``undefined`` sidesteps the MarshalError on object returns
        # (same trick as the console install).
        if target_names:
            pairs = ", ".join(
                f"{camel}: __tools_{camel}" for camel in target_names
            )
            ctx.eval(f"globalThis.tools = {{ {pairs} }}; undefined")
        else:
            ctx.eval("globalThis.tools = {}; undefined")
        self._active_tool_names = target_names
        self._tools_installed = True

    def set_outer_runtime(self, runtime: ToolRuntime | None) -> None:
        """Record the outer ``ToolRuntime`` for the current eval.

        PTC bridges forward this into their ``tool.ainvoke`` calls so
        tools that depend on ``state`` / ``store`` / ``tool_call_id``
        (notably subagent ``task`` tools) see the orchestrator's graph
        context. The middleware calls this immediately before each eval
        and again with ``None`` after.
        """
        self._outer_runtime = runtime

    def _stage_write(self, path: str, content: str) -> None:
        """Buffer a file write for flush after the current eval returns.

        Replaces any prior pending write to the same path so only the
        latest version is flushed. Used by in-eval host bridges that
        need read-after-write semantics before the backend has been
        touched (e.g. ``swarm.create`` → ``swarm.execute``).
        """
        self._pending_writes[path] = content

    async def _read_through_pending(
        self,
        path: str,
        backend: BackendProtocol,
    ) -> str:
        """Read ``path``, preferring the pending buffer over the backend.

        Raises ``FileNotFoundError`` if the path is absent from both.
        """
        if path in self._pending_writes:
            return self._pending_writes[path]
        result = await backend.aread(path)
        if result.error is not None or result.file_data is None:
            msg = f"Failed to read {path!r}: {result.error or 'empty'}"
            raise FileNotFoundError(msg)
        content = result.file_data.get("content")
        if not isinstance(content, str):
            msg = f"Unexpected content shape for {path!r}: {type(content).__name__}"
            raise ValueError(msg)
        return content

    def _drain_pending_writes(self) -> list[tuple[str, str]]:
        """Return and clear all buffered writes."""
        drained = list(self._pending_writes.items())
        self._pending_writes.clear()
        return drained

    def _register_tool_bridge(self, camel: str) -> None:
        """Install a host-function bridge for one camel-cased tool name.

        The bridge is async so ``eval_async``'s driving loop can await
        ``tool.ainvoke`` without blocking the event loop. We look the
        tool up through ``self._registered_tools`` on every call so a
        later ``install_tools`` that swaps the underlying object (same
        name, different instance) is picked up without re-registration.
        """
        registered = self._registered_tools

        async def _bridge(raw_input: Any = None) -> str:
            tool = registered.get(camel)
            if tool is None:
                # Shouldn't happen — we only rewrite ``globalThis.tools``
                # with names currently in the map — but if a race causes
                # it, fail loud.
                msg = f"tool '{camel}' not registered"
                raise RuntimeError(msg)
            payload = _normalize_tool_input(raw_input)
            call_id = _synth_tool_call_id(tool.name)
            # Build a ToolCall-shaped input so InjectedToolCallId and the
            # runtime-arg injection in _inject_tool_args_for_ptc fire.
            args = _inject_tool_args_for_ptc(tool, payload, self._outer_runtime, call_id)
            result = await tool.ainvoke(
                {"name": tool.name, "args": args, "id": call_id, "type": "tool_call"},
            )
            return _coerce_tool_output(result)

        self._ctx.register(f"__tools_{camel}", _bridge, is_async=True)

    def eval_sync(self, code: str) -> EvalOutcome:
        with self._sync_lock:
            return self._eval_sync_locked(code)

    async def eval_async(self, code: str) -> EvalOutcome:
        # Hold the lock around the whole eval_async call so concurrent
        # tool calls queue rather than racing into ConcurrentEvalError.
        async with self._async_lock:
            return await self._eval_async_locked(code)

    def _eval_sync_locked(self, code: str) -> EvalOutcome:
        outcome = EvalOutcome()
        try:
            value = self._ctx.eval(code)
            outcome.result = _stringify(value)
        except MarshalError:
            outcome.result_kind = "handle"
            outcome.result = self._describe_via_handle_sync(code)
        except QJSTimeoutError as e:
            outcome.error_type = "Timeout"
            outcome.error_message = str(e)
        except JSError as e:
            self._record_js_error(outcome, e)
        except ConcurrentEvalError as e:
            # Shouldn't happen given the locks, but map it defensively so
            # it's a clean tool error rather than a propagated exception.
            outcome.error_type = "ConcurrentEval"
            outcome.error_message = str(e)
        except MemoryLimitError as e:
            outcome.error_type = "OutOfMemory"
            outcome.error_message = str(e)
        outcome.stdout = self._console.drain()
        return outcome

    async def _eval_async_locked(self, code: str) -> EvalOutcome:
        """v0.2 async path. Uses ``ctx.eval_async`` directly.

        Differences from the sync path the model should know about:
        - Top-level ``await`` works; Promises settle before returning.
        - ``timeout=`` is per-call (fresh budget each invocation) instead
          of cumulative across a Context's lifetime.
        - ``asyncio.CancelledError`` propagates out if JS doesn't absorb
          the ``HostCancellationError``. We let it through untouched so
          LangGraph's cancellation semantics work end-to-end.
        """
        outcome = EvalOutcome()
        # Fresh cancel event for this eval. When set, host bridges can
        # short-circuit pending work. In-flight asyncio tasks are
        # cancelled the usual way via task cancellation (propagated
        # below by wait_for).
        cancel_event = asyncio.Event()
        self._eval_cancel_event = cancel_event
        try:
            try:
                # External timeout via wait_for: cancelling the wrapped
                # task propagates CancelledError to in-flight host
                # coroutines. Internal timeout is the outer budget plus a
                # small slack as a backstop for runaway JS that never
                # yields back to the asyncio loop.
                value = await asyncio.wait_for(
                    self._ctx.eval_async(
                        code, timeout=self._per_call_timeout + 60
                    ),
                    timeout=self._per_call_timeout,
                )
                outcome.result = _stringify(value)
            except MarshalError:
                outcome.result_kind = "handle"
                outcome.result = await self._describe_via_handle_async(code)
            except TimeoutError:
                cancel_event.set()
                outcome.error_type = "Timeout"
                outcome.error_message = (
                    f"eval_async exceeded {self._per_call_timeout}s"
                )
            except QJSTimeoutError as e:
                cancel_event.set()
                outcome.error_type = "Timeout"
                outcome.error_message = str(e)
            except DeadlockError as e:
                # Top-level Promise never resolved and no async host work in
                # flight. Surface as a distinct error type because the fix
                # is user-level (their JS has an un-resolvable Promise or a
                # sync host fn that should be async); a plain error-type
                # message without context would make this hard to diagnose.
                cancel_event.set()
                outcome.error_type = "Deadlock"
                outcome.error_message = str(e)
            except HostCancellationError:
                # JS declined to catch a cancellation — re-raise as
                # CancelledError so asyncio unwinds the caller's task.
                # Do not record anything in ``outcome``; the call is dead.
                cancel_event.set()
                raise asyncio.CancelledError from None
            except JSError as e:
                self._record_js_error(outcome, e)
            except ConcurrentEvalError as e:
                outcome.error_type = "ConcurrentEval"
                outcome.error_message = str(e)
            except MemoryLimitError as e:
                outcome.error_type = "OutOfMemory"
                outcome.error_message = str(e)
        finally:
            self._eval_cancel_event = None
        outcome.stdout = self._console.drain()
        return outcome

    def _record_js_error(self, outcome: EvalOutcome, e: JSError) -> None:
        # HostError is a JSError subclass; surface it as "HostError"
        # so operators can distinguish a bug in our console bridge
        # from a user-code error.
        if isinstance(e, HostError):
            logger.warning("console-bridge host error", exc_info=e.__cause__)
            outcome.error_type = "HostError"
        else:
            outcome.error_type = e.name
        outcome.error_message = e.message
        outcome.error_stack = e.stack

    def _describe_via_handle_sync(self, code: str) -> str:
        try:
            handle = self._ctx.eval_handle(code)
        except Exception:  # noqa: BLE001 — describe-only path; swallow to placeholder
            return _HANDLE_PLACEHOLDER
        try:
            return _format_handle(handle)
        finally:
            handle.dispose()

    async def _describe_via_handle_async(self, code: str) -> str:
        try:
            handle = await self._ctx.eval_handle_async(
                code, timeout=self._per_call_timeout
            )
        except Exception:  # noqa: BLE001 — describe-only path; swallow to placeholder
            return _HANDLE_PLACEHOLDER
        try:
            return _format_handle(handle)
        finally:
            handle.dispose()

    def close(self) -> None:
        self._ctx.close()


@dataclass
class _Registry:
    """Shared Runtime + per-thread contexts, lazily created on first use.

    One Runtime per middleware instance is enough for common deployments
    (single-process agent server). Each LangGraph thread gets its own
    Context, so globals never leak between conversations.
    """

    memory_limit: int
    timeout: float
    capture_console: bool
    _runtime: Runtime | None = None
    _repls: dict[str, _ThreadREPL] = field(default_factory=dict)
    _init_lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, thread_id: str) -> _ThreadREPL:
        # Double-checked lock on the runtime; then a bare check on the per-
        # thread entry because contexts are cheap and we'd rather create a
        # throwaway on a race than serialize every eval through a global
        # lock.
        if self._runtime is None:
            with self._init_lock:
                if self._runtime is None:
                    self._runtime = Runtime(memory_limit=self.memory_limit)
        repl = self._repls.get(thread_id)
        if repl is None:
            with self._init_lock:
                repl = self._repls.get(thread_id)
                if repl is None:
                    repl = _ThreadREPL(
                        self._runtime,
                        timeout=self.timeout,
                        capture_console=self.capture_console,
                    )
                    self._repls[thread_id] = repl
        return repl

    def close(self) -> None:
        # Runtime.close() walks its own context list, so we don't need to
        # close each _ThreadREPL individually. Drop our handles first so we
        # don't touch already-freed contexts.
        self._repls.clear()
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None


def format_outcome(
    outcome: EvalOutcome,
    *,
    max_result_chars: int,
) -> str:
    """Render an EvalOutcome as the tool's wire format (see spec §8)."""
    parts: list[str] = []
    if outcome.stdout:
        parts.append(f"<stdout>\n{_truncate(outcome.stdout, max_result_chars)}\n</stdout>")
    if outcome.error_type is not None:
        inner = outcome.error_message
        if outcome.error_stack:
            inner = f"{inner}\n{outcome.error_stack}"
        parts.append(
            f'<error type="{_xml_escape(outcome.error_type)}">'
            f"{_xml_escape(_truncate(inner, max_result_chars))}"
            f"</error>"
        )
    else:
        body = outcome.result if outcome.result is not None else "undefined"
        kind_attr = f' kind="{outcome.result_kind}"' if outcome.result_kind else ""
        parts.append(
            f"<result{kind_attr}>{_xml_escape(_truncate(body, max_result_chars))}</result>"
        )
    return "\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(_TRUNCATE_MARKER.format(n=0)))
    dropped = len(text) - keep
    return text[:keep] + _TRUNCATE_MARKER.format(n=dropped)


def _xml_escape(text: str) -> str:
    # Minimal escape — we emit the tag set we control, so we only need to
    # keep angle brackets from closing our wrapper tags early.
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
