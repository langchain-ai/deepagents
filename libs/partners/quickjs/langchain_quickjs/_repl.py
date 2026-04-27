"""Thread-keyed QuickJS REPL registry, console bridge, and result formatter.

Kept separate from ``middleware.py`` so the REPL mechanics stay testable
without constructing an agent or wiring up LangGraph state.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from langgraph.types import Command
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
    ModuleScope,
    Runtime,
    ThreadWorker,
)
from quickjs_rs import (
    TimeoutError as QJSTimeoutError,
)

from langchain_quickjs._format import (
    coerce_tool_output,
    format_handle,
    stringify,
)
from langchain_quickjs._ptc import is_valid_js_identifier, to_camel_case
from langchain_quickjs._skills import LoadedSkill, SkillLoadError, aload_skill

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepagents.backends.protocol import BackendProtocol
    from deepagents.middleware.skills import SkillMetadata
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

logger = logging.getLogger(__name__)

# Sentinel returned by the formatter when the underlying value was a
# function/circular ref that couldn't be auto-marshaled. We format it as
# a handle-shaped result so the model sees "you got back a function" rather
# than nothing.
_HANDLE_PLACEHOLDER = "[unmarshalable value]"


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
    commands: list[Command] = field(default_factory=list)


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
        self._lines.append(" ".join(stringify(a) for a in args))

    def drain(self) -> str:
        if not self._lines:
            return ""
        out = "\n".join(self._lines)
        self._lines.clear()
        return out


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


def _extract_commands(value: Any) -> list[Command]:
    """Collect LangGraph ``Command`` values from a tool return payload."""
    if isinstance(value, Command):
        return [value]
    if isinstance(value, list):
        return [entry for entry in value if isinstance(entry, Command)]
    return []


def _synth_tool_call_id(tool_name: str) -> str:
    """Mint a synthetic tool_call_id for a PTC-driven tool invocation.

    Tools like ``task`` require a non-empty ``tool_call_id`` to stamp
    into their emitted ``ToolMessage``. The real call_id lives on the
    outer ``eval`` tool call; we synthesise a child id so downstream
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
    """Mirror LangGraph's ``ToolNode._inject_tool_args`` for PTC calls.

    LangChain tools that declare ``ToolRuntime`` / ``InjectedState`` /
    ``InjectedStore`` only see those values when a real ``ToolNode``
    wires them in. PTC calls bypass the ToolNode, so we replicate the
    detection logic here. The outer runtime (captured from the active
    ``eval`` tool invocation) provides state/store/context/config;
    ``tool_call_id`` is freshly minted per sub-call.
    """
    try:
        from langgraph.prebuilt.tool_node import (  # noqa: PLC0415 — optional dep, imported here so ImportError is catchable
            _get_all_injected_args,
        )
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
        tools=outer_runtime.tools,
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
    """Return JS that atomically rebuilds ``globalThis.tools`` from bridges."""
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

    All ``ctx.*`` operations are marshalled onto the worker's dedicated
    thread because ``quickjs_rs`` objects are ``!Send``. The public
    methods are safe to call from any thread/loop.
    """

    def __init__(
        self,
        worker: ThreadWorker,
        runtime: Runtime,
        *,
        timeout: float,
        capture_console: bool,
    ) -> None:
        self._worker = worker
        self._runtime = runtime
        # The Context-level ``timeout`` is used as the cumulative budget
        # for sync evals. Async evals pass ``timeout=`` per call so each
        # call gets a fresh budget — matches what a REPL user expects,
        # and what we describe in the system prompt.
        self._per_call_timeout = timeout
        self._capture_console = capture_console
        self._console = _ConsoleBuffer()
        self._ctx: Context | None = None
        # PTC state. ``_registered_tools`` tracks which camel-case names
        # have already had their host-function bridge installed on the
        # QuickJS context. Host functions cannot be un-registered, so we
        # never remove entries from here — changes to the exposed set
        # are reflected by rewriting ``globalThis.tools`` (see
        # install_tools) to include only the currently-active subset.
        self._registered_tools: dict[str, BaseTool] = {}
        self._bridge_symbols: dict[str, str] = {}
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
        # ``Command`` values emitted by PTC-invoked tools during the
        # currently-running eval call. Drained into ``EvalOutcome``.
        self._ptc_command_buffer: list[Command] = []
        # Context creation + console install must happen on the worker
        # thread. Block caller here so the REPL is ready to use when
        # __init__ returns.
        worker.run_sync(self._ainit())

    async def _ainit(self) -> None:
        self._ctx = self._runtime.new_context(timeout=self._per_call_timeout)
        if self._capture_console:
            self._install_console()

    def _install_console(self) -> None:
        ctx = cast("Context", self._ctx)
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
        self._worker.run_sync(self._ainstall_tools(tools))

    async def _ainstall_tools(self, tools: Sequence[BaseTool]) -> None:
        ctx = cast("Context", self._ctx)
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
            # while keeping the same names. Guard with ``_tools_installed``
            # so the empty → empty transition on first call still installs
            # a ``tools = {}`` global — otherwise ``typeof tools.x`` hits a
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
        # ``undefined`` sidesteps the MarshalError on object returns
        # (same trick as the console install).
        bridges = {camel: self._bridge_symbols[camel] for camel in target_names}
        ctx.eval(_render_tools_namespace_assignment(bridges))
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

    def _register_tool_bridge(self, camel: str) -> str:
        """Install a host-function bridge for one camel-cased tool name.

        The bridge is async so ``eval_async``'s driving loop can await
        ``tool.ainvoke`` without blocking the event loop. We look the
        tool up through ``self._registered_tools`` on every call so a
        later ``install_tools`` that swaps the underlying object (same
        name, different instance) is picked up without re-registration.
        """
        ctx = cast("Context", self._ctx)
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
            args = _inject_tool_args_for_ptc(
                tool, payload, self._outer_runtime, call_id
            )
            result = await tool.ainvoke(
                {"name": tool.name, "args": args, "id": call_id, "type": "tool_call"},
            )
            self._ptc_command_buffer.extend(_extract_commands(result))
            return coerce_tool_output(result)

        bridge_symbol = _bridge_symbol_name(camel)
        ctx.register(bridge_symbol, _bridge, is_async=True)
        return bridge_symbol

    def eval_sync(self, code: str) -> EvalOutcome:
        # Both sync and async entry points funnel through ctx.eval_async on
        # the worker loop. Sync ctx.eval can't dispatch async host functions
        # (PTC bridges are is_async=True), so routing sync callers through
        # the async path is required for PTC to work under sync invocation.
        return self._worker.run_sync(self._aeval_async(code))

    async def eval_async(self, code: str) -> EvalOutcome:
        return await self._worker.run_async(self._aeval_async(code))

    async def ainstall_module_scope(self, scope: ModuleScope) -> None:
        """Install a ``ModuleScope`` on the context from the caller's loop."""
        await self._worker.run_async(self._ainstall_module_scope(scope))

    async def _ainstall_module_scope(self, scope: ModuleScope) -> None:
        self._runtime.install(scope)

    async def _aeval_async(self, code: str) -> EvalOutcome:
        """Uses ``ctx.eval_async`` directly.

        Overlapping evals on the same context surface as
        ``ConcurrentEvalError`` (recorded in ``EvalOutcome.error_type``).
        We intentionally do not queue: a model dispatching overlapping
        evals against shared state is almost always a prompting bug,
        and a loud failure is a better signal than silent serialisation.
        """
        ctx = cast("Context", self._ctx)
        outcome = EvalOutcome()
        self._ptc_command_buffer.clear()
        try:
            value = await ctx.eval_async(code, timeout=self._per_call_timeout)
            outcome.result = stringify(value)
        except MarshalError:
            outcome.result_kind = "handle"
            outcome.result = await self._describe_via_handle_async(code)
        except QJSTimeoutError as e:
            outcome.error_type = "Timeout"
            outcome.error_message = str(e)
        except DeadlockError as e:
            # Top-level Promise never resolved and no async host work in
            # flight. Surface as a distinct error type because the fix
            # is user-level (their JS has an un-resolvable Promise or a
            # sync host fn that should be async); a plain error-type
            # message without context would make this hard to diagnose.
            outcome.error_type = "Deadlock"
            outcome.error_message = str(e)
        except HostCancellationError:
            # JS declined to catch a cancellation — re-raise as
            # CancelledError so asyncio unwinds the caller's task.
            # Do not record anything in ``outcome``; the call is dead.
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
            outcome.commands.extend(self._ptc_command_buffer)
            self._ptc_command_buffer.clear()
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

    async def _describe_via_handle_async(self, code: str) -> str:
        ctx = cast("Context", self._ctx)
        try:
            handle = await ctx.eval_handle_async(code, timeout=self._per_call_timeout)
        except Exception:  # noqa: BLE001 — describe-only path; swallow to placeholder
            return _HANDLE_PLACEHOLDER
        try:
            return format_handle(handle)
        finally:
            handle.dispose()

    def close(self) -> None:
        self._worker.run_sync(self._aclose())

    async def _aclose(self) -> None:
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None


@dataclass
class _SkillInstall:
    """Install-cache entry for one skill on a Runtime.

    Either ``loaded`` is set (install succeeded) or ``error`` is set
    (install failed; subsequent references fail fast without re-hitting
    the backend). Never both.
    """

    loaded: LoadedSkill | None = None
    error: SkillLoadError | None = None


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
    _worker: ThreadWorker = field(default_factory=ThreadWorker)
    _runtime: Runtime | None = None
    _repls: dict[str, _ThreadREPL] = field(default_factory=dict)
    _init_lock: threading.Lock = field(default_factory=threading.Lock)
    # Per-Runtime skill install cache. `ModuleScope` storage is per-
    # Runtime in quickjs-rs v0.4 (see src/runtime.rs:132-146), and
    # ``runtime.install`` is additive — re-install of a bare specifier
    # after first import is a silent no-op. So we only need to install
    # each skill *once* per Runtime, even if multiple threads reference
    # it. Successes go under ``loaded``; failures are cached too so a
    # broken skill doesn't re-hit the backend every eval.
    _skill_installs: dict[str, _SkillInstall] = field(default_factory=dict)
    _skill_install_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def get(self, thread_id: str) -> _ThreadREPL:
        # Double-checked lock on the runtime; then a bare check on the per-
        # thread entry because contexts are cheap and we'd rather create a
        # throwaway on a race than serialize every eval through a global
        # lock. Runtime + Context are !Send so creation is dispatched onto
        # the worker thread.
        if self._runtime is None:
            with self._init_lock:
                if self._runtime is None:
                    self._runtime = self._worker.run_sync(self._acreate_runtime())
        repl = self._repls.get(thread_id)
        if repl is None:
            with self._init_lock:
                repl = self._repls.get(thread_id)
                if repl is None:
                    repl = _ThreadREPL(
                        self._worker,
                        self._runtime,
                        timeout=self.timeout,
                        capture_console=self.capture_console,
                    )
                    self._repls[thread_id] = repl
        return repl

    async def _acreate_runtime(self) -> Runtime:
        return Runtime(memory_limit=self.memory_limit)

    async def aensure_skills_installed(
        self,
        referenced: frozenset[str],
        metadata: dict[str, SkillMetadata],
        backend: BackendProtocol,
        repl: _ThreadREPL,
    ) -> list[SkillLoadError]:
        """Install any referenced skills not yet on this Runtime.

        Returns a list of errors for skills that either (a) were in the
        referenced set but have no metadata entry, or (b) failed to
        load — present or newly-cached. The caller surfaces these as a
        pre-eval failure.

        The lock is held around the whole install pass rather than per
        skill so we don't cold-fetch the same skill twice on a race.
        Install is additive and idempotent, so cross-thread racing is
        safe; serialising here just avoids wasted I/O.
        """
        errors: list[SkillLoadError] = []
        async with self._skill_install_lock:
            for name in referenced:
                cached = self._skill_installs.get(name)
                if cached is not None:
                    if cached.error is not None:
                        errors.append(cached.error)
                    continue
                meta = metadata.get(name)
                if meta is None:
                    err = SkillLoadError(
                        f"skill {name!r} referenced but not available on this agent"
                    )
                    # Don't cache unavailability — the set of installed
                    # SkillsMiddleware sources may differ per call, and
                    # a skill that's absent now could be present later.
                    errors.append(err)
                    continue
                try:
                    loaded = await aload_skill(meta, backend)
                except SkillLoadError as exc:
                    self._skill_installs[name] = _SkillInstall(error=exc)
                    errors.append(exc)
                    continue
                # Install under the bare specifier. Additive — any
                # previously-installed scope for a different specifier
                # is untouched. Re-install of the same specifier after
                # first import is a silent no-op (spec §5).
                await repl.ainstall_module_scope(
                    ModuleScope({loaded.specifier: loaded.scope})
                )
                self._skill_installs[name] = _SkillInstall(loaded=loaded)
        return errors

    def close(self) -> None:
        # Runtime.close() walks its own context list, so we don't need to
        # close each _ThreadREPL individually. Drop our handles first so we
        # don't touch already-freed contexts.
        self._repls.clear()
        self._skill_installs.clear()
        if self._runtime is not None:
            self._worker.run_sync(self._aclose_runtime())
            self._runtime = None
        self._worker.close()

    async def _aclose_runtime(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
