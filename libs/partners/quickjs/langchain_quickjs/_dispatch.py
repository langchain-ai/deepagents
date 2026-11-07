"""Run `tools.*` calls from an eval through the agent's tool-call chain.

Each call goes through the `wrap_tool_call` middleware below
`CodeInterpreterMiddleware` inside its own `langgraph.func.task`. The task
journals a finished call, including one that raised, so a re-run after a resume
returns its record instead of running it again. It scopes an interrupt, and so
the human's decision, to that call. Its name carries a digest of the tool name
and arguments, so a re-run that issues a different call at the same position
never receives another call's record or decision.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import threading
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tools import StructuredTool, Tool
from langchain_core.tools.base import _format_output
from langgraph._internal._constants import CONFIG_KEY_READ, CONFIG_KEY_SEND
from langgraph.config import get_config
from langgraph.errors import GraphBubbleUp, GraphInterrupt
from langgraph.func import task

from langchain_quickjs._format import coerce_tool_output_for_ptc
from langchain_quickjs._fs import request_raw_result

if TYPE_CHECKING:
    import concurrent.futures
    from collections.abc import Awaitable, Callable, Iterable

    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from quickjs_rs import Snapshot


class EvalSuspendedError(RuntimeError):
    """Raised to JavaScript for calls issued after the eval started pausing."""


@dataclass(frozen=True, slots=True)
class ToolChain:
    """The middleware chain below `CodeInterpreterMiddleware` for one eval call."""

    request: ToolCallRequest
    handler: Callable[[ToolCallRequest], Any]
    loop: asyncio.AbstractEventLoop | None
    """The graph's event loop for an async chain; `None` for a sync one."""


@dataclass(frozen=True, slots=True)
class CallResult:
    """What a nested tool call returned, as recorded in the journal."""

    status: Literal["success", "error"]
    value: Any
    artifact: Any = None
    """The tool's raw result, kept only for a call that asked for one."""
    rejected: bool = False
    """Middleware refused the call before the tool ran, as a reviewer does."""


@dataclass(slots=True)
class _Returned:
    """What the tool function itself returned, before the tool node formats it."""

    called: bool = False
    """`run` or `arun` was entered; they check arguments, so no middleware refused."""
    ran: bool = False
    value: Any = None


def _watch(
    fn: Callable[..., Any], seen: _Returned, config: RunnableConfig
) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        token = var_child_runnable_config.set(config)
        try:
            seen.value = fn(*args, **kwargs)
        finally:
            var_child_runnable_config.reset(token)
        seen.ran = True
        return seen.value

    return wrapper


def _awatch(
    fn: Callable[..., Awaitable[Any]], seen: _Returned, config: RunnableConfig
) -> Callable[..., Any]:
    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        token = var_child_runnable_config.set(config)
        try:
            seen.value = await fn(*args, **kwargs)
        finally:
            var_child_runnable_config.reset(token)
        seen.ran = True
        return seen.value

    return wrapper


def _entered(fn: Callable[..., Any], seen: _Returned) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        seen.called = True
        return fn(*args, **kwargs)

    return wrapper


def _run_coroutine(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        async def run() -> Any:
            return await fn(*args, **kwargs)

        return asyncio.run(run())

    return wrapper


def _watched_tool(
    tool: BaseTool, seen: _Returned, *, sync: bool, config: RunnableConfig
) -> BaseTool:
    """Return a copy of `tool` that runs under `config` and records its return value.

    The call's own config keeps an `interrupt()` inside the tool scoped to the
    call, and `seen` keeps the value JavaScript wants rather than the tool
    node's model text. Only `StructuredTool` and `Tool` with a `content`
    response format expose that value; other tools run as they are, under the
    eval's config, so a pause drops their graph-state writes. An async-only tool
    gets a sync entry point so a sync graph can run it.
    """
    if (
        not isinstance(tool, (StructuredTool, Tool))
        or tool.response_format != "content"
    ):
        # Unobservable, so its errors are never reported as refusals.
        seen.called = True
        return tool
    func = tool.func
    if func is None and sync and tool.coroutine is not None:
        func = _run_coroutine(tool.coroutine)
    update: dict[str, Any] = {}
    if func is not None:
        update["func"] = _watch(func, seen, config)
    if tool.coroutine is not None:
        update["coroutine"] = _awatch(tool.coroutine, seen, config)
    watched = tool.model_copy(update=update)
    for name in ("run", "arun"):
        object.__setattr__(watched, name, _entered(getattr(watched, name), seen))
    return watched


def _model_text(value: Any, message: ToolMessage) -> Any:
    """The content the tool node gives the model for `value`, before middleware."""
    if isinstance(value, ToolMessage):
        return value.content
    plain = _format_output(
        value, None, message.tool_call_id, message.name or "", "success"
    )
    return plain.content if isinstance(plain, ToolMessage) else None


def _record(
    result: Any, seen: _Returned, writes: list[tuple[str, Any]], *, raw: bool
) -> dict[str, Any]:
    """Journal record of the tool's own return value; changed model text is a note."""
    messages = result if isinstance(result, list) else [result]
    message = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)
    status, value, note = "success", result, None
    if message is not None and message.status == "error":
        status = "error"
    elif seen.ran:
        value = seen.value
        if message is not None and message.content != _model_text(value, message):
            note = message.text
    return {
        "status": status,
        "value": coerce_tool_output_for_ptc(value),
        "note": note,
        "artifact": message.artifact if raw and message is not None else None,
        "rejected": status == "error" and not seen.called,
        "writes": writes,
    }


@dataclass
class EvalJournal:
    """Dispatcher and journal for the calls one run of one `eval` makes.

    A re-run after a resume builds a new instance; determinism across runs
    comes from the LangGraph task journal, not from state kept here.
    """

    chain: ToolChain
    next_seq: int = 0
    suspended: list[tuple[int, GraphBubbleUp]] = field(default_factory=list)
    """Bubble-ups from paused calls, with the issue order of the call."""
    inflight: dict[concurrent.futures.Future[Any], int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    heap: Snapshot | None = None
    """The heap before the eval ran, restored if a call pauses; `None` if busy."""

    @property
    def runtime(self) -> ToolRuntime:
        """The eval call's runtime."""
        return self.chain.request.runtime

    def suspend(self, exc: GraphBubbleUp) -> None:
        """Record a bubble-up that did not come from a journaled call."""
        self.suspended.append((self.next_seq, exc))

    def check_running(self) -> None:
        """Refuse new work once a call in this eval has paused.

        Raises:
            EvalSuspendedError: If a call in this eval paused.
        """
        if self.suspended:
            msg = "eval is pausing for human review; this call was not run"
            raise EvalSuspendedError(msg)

    async def dispatch(
        self, tool: BaseTool, args: dict[str, Any], *, raw: bool = False
    ) -> CallResult:
        """Run one `tools.*` or `fs` call through the chain inside its own task.

        Raises:
            EvalSuspendedError: If an earlier call in this eval paused, so this
                call must wait for the re-run instead of running now.
        """
        seq = self.next_seq
        self.next_seq += 1
        call = ToolCall(
            name=tool.name,
            args=args,
            id=f"{self.runtime.tool_call_id}.ptc{seq}",
            type="tool_call",
        )
        ran: list[bool] = []
        failure: list[Exception] = []

        def failed(exc: Exception, writes: list[tuple[str, Any]]) -> dict[str, Any]:
            if isinstance(exc, GraphBubbleUp):
                raise exc
            failure.append(exc)
            return _raised_record(exc, writes)

        def run(config: RunnableConfig) -> dict[str, Any]:
            ran.append(True)
            request, seen, writes = self._request(tool, call, config, raw=raw)
            try:
                return _record(self.chain.handler(request), seen, writes, raw=raw)
            except Exception as exc:  # noqa: BLE001  # journaled, re-raised by dispatch
                return failed(exc, writes)

        async def arun(config: RunnableConfig) -> dict[str, Any]:
            ran.append(True)
            request, seen, writes = self._request(tool, call, config, raw=raw)
            try:
                return _record(await self.chain.handler(request), seen, writes, raw=raw)
            except Exception as exc:  # noqa: BLE001  # journaled, re-raised by dispatch
                return failed(exc, writes)

        # A raw-result call, as from `fs`, never shares a record with `tools.*`.
        key = [tool.name, args, "raw"] if raw else [tool.name, args]
        blob = json.dumps(key, sort_keys=True, default=str)
        digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
        # The record may not be serialized yet, so it is read, never changed.
        body = run if self.chain.loop is None else arun
        record = await self._journaled(f"ptc.{tool.name}.{digest}", body, seq=seq)
        if not ran and record.get("writes"):
            # A saved call did not run, so its state writes were not sent.
            with self.state_lock:
                self.runtime.config["configurable"][CONFIG_KEY_SEND](
                    [(channel, value) for channel, value in record["writes"]]
                )
        if raised := record.get("raised"):
            # The original exception on the run that raised it, so an uncaught
            # one ends the agent run as it would for a model tool call.
            raise failure[0] if failure else RuntimeError(raised)
        if record.get("note"):
            self.notes.append(f"tools.{tool.name} call {seq}: {record['note']}")
        # A record saved before fs has no `artifact`.
        return CallResult(
            status=record["status"],
            value=record["value"],
            artifact=record.get("artifact"),
            rejected=record["rejected"],
        )

    def _request(
        self, tool: BaseTool, call: ToolCall, config: RunnableConfig, *, raw: bool
    ) -> tuple[ToolCallRequest, _Returned, list[tuple[str, Any]]]:
        """Build the nested call's request under its own task's config."""
        seen, writes = _Returned(), []
        config = _state_from_eval_node(
            config, self.runtime.config, writes, self.state_lock
        )
        if raw:
            config = request_raw_result(config, call["id"] or "")
        runtime = replace(self.runtime, tool_call_id=call["id"], config=config)
        watched = _watched_tool(tool, seen, sync=self.chain.loop is None, config=config)
        request = self.chain.request.override(
            tool_call=call, tool=watched, runtime=runtime
        )
        return request, seen, writes

    async def _journaled(
        self, name: str, body: Callable[[RunnableConfig], Any], *, seq: int
    ) -> Any:
        """Run `body` as a LangGraph task under the eval's task."""
        self.check_running()
        future = self._schedule(name, body)
        self.inflight[future] = seq
        future.add_done_callback(self._settled)
        waiter = asyncio.wrap_future(future)
        try:
            return await asyncio.shield(waiter)
        except asyncio.CancelledError:
            # A pausing eval keeps its in-flight calls running so their own
            # interrupts are collected and their results journaled.
            if not self.suspended:
                future.cancel()
            raise

    def _schedule(
        self, name: str, body: Callable[[RunnableConfig], Any]
    ) -> concurrent.futures.Future[Any]:
        """Start the task on the outer loop (async graph) or this thread (sync)."""
        outer = self.runtime.config
        loop = self.chain.loop
        if loop is None:

            def run() -> Any:
                return body(get_config())

            token = var_child_runnable_config.set(outer)
            try:
                return task(name=name)(run)()
            finally:
                var_child_runnable_config.reset(token)

        async def arun() -> Any:
            return await body(get_config())

        async def call() -> Any:
            token = var_child_runnable_config.set(outer)
            try:
                return await task(name=name)(arun)()
            finally:
                var_child_runnable_config.reset(token)

        return asyncio.run_coroutine_threadsafe(call(), loop)

    def _settled(self, future: concurrent.futures.Future[Any]) -> None:
        seq = self.inflight.pop(future)
        if future.cancelled():
            return
        exc = future.exception()
        if isinstance(exc, GraphBubbleUp):
            self.suspended.append((seq, exc))

    async def settle(self) -> GraphBubbleUp:
        """Wait for in-flight calls; merge their interrupts in issue order."""
        while self.inflight:
            pending = list(self.inflight)
            await asyncio.gather(
                *(asyncio.wrap_future(f) for f in pending), return_exceptions=True
            )
        ordered = [e for _, e in sorted(self.suspended, key=lambda item: item[0])]
        interrupts = [
            i for e in ordered if isinstance(e, GraphInterrupt) for i in e.args[0]
        ]
        others = [e for e in ordered if not isinstance(e, GraphInterrupt)]
        if others or not interrupts:
            return others[0] if others else ordered[0]
        return GraphInterrupt(tuple(interrupts))


def _state_from_eval_node(
    config: RunnableConfig,
    eval_config: RunnableConfig,
    writes: list[tuple[str, Any]],
    lock: threading.Lock,
) -> RunnableConfig:
    """Return a call's task config that reads and writes graph state as the eval.

    A task reads the state as of the step's start plus its own writes, so one
    call's `StateBackend` write would be invisible to the next. State writes go
    through the `eval` node instead and are kept in `writes`, which `dispatch`
    resends when it returns a record without running the call; LangGraph drops
    a paused node's writes, so they land once. Channels starting with `__`,
    such as the resume value `interrupt()` writes, stay with the task. `lock`
    serializes concurrent calls' use of the node's shared write list, but not
    a parallel model tool call's, as before.
    """
    configurable = config["configurable"]
    task_send = configurable[CONFIG_KEY_SEND]
    eval_send = eval_config["configurable"][CONFIG_KEY_SEND]
    eval_read = eval_config["configurable"][CONFIG_KEY_READ]

    def send(items: Iterable[tuple[str, Any]]) -> None:
        items = list(items)
        own = [(c, v) for c, v in items if c.startswith("__")]
        state = [(c, v) for c, v in items if not c.startswith("__")]
        if state:
            writes.extend(state)
            with lock:
                eval_send(state)
        if own:
            task_send(own)

    def read(*args: Any, **kwargs: Any) -> Any:
        with lock:
            return eval_read(*args, **kwargs)

    return {
        **config,
        "configurable": {**configurable, CONFIG_KEY_SEND: send, CONFIG_KEY_READ: read},
    }


def _raised_record(exc: Exception, writes: list[tuple[str, Any]]) -> dict[str, Any]:
    """Record a raise, since LangGraph re-runs a task that raised."""
    return {"raised": f"{type(exc).__name__}: {exc}", "writes": writes}
