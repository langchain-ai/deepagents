"""Run `tools.*` calls from an eval through the agent's tool-call chain.

A call made from JavaScript is dispatched the way a model tool call is: as a
`ToolCallRequest` through the `wrap_tool_call` middleware below
`CodeInterpreterMiddleware`, so `HumanInTheLoopMiddleware` and other policy
middleware see it. Each call runs inside its own `langgraph.func.task`, which
gives it three properties the eval needs when a call pauses for approval:

- A finished call, including one whose tool raised, is journaled in the
  checkpoint. When the eval re-runs after the human answers, the call returns
  its recorded result instead of running again, so side effects happen once.
- An interrupt raised by the call is scoped to the call's own task, so its
  interrupt id, and therefore the human's decision, belongs to that call.
- The task name carries a digest of the tool name and arguments. A re-run that
  issues a different call at the same position gets a different task, so a
  recorded result, or a decision keyed by interrupt id, is never handed to a
  call it was not made for. A resume without an interrupt id goes to whichever
  call asks first on the re-run.
"""

from __future__ import annotations

import asyncio
import contextlib
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

if TYPE_CHECKING:
    import concurrent.futures
    from collections.abc import Awaitable, Callable, Iterable, Iterator

    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from quickjs_rs import Snapshot

ToolStatus = Literal["success", "error"]

# Block types a direct call would show the model and JS cannot pass along as
# text: standard image, audio, video, and file blocks, plus OpenAI's legacy
# `image_url`.
_MEDIA_BLOCK_TYPES = frozenset({"image", "audio", "video", "file", "image_url"})


class EvalSuspendedError(RuntimeError):
    """Raised to JavaScript for calls issued after the eval started pausing."""


@dataclass(frozen=True, slots=True)
class ToolChain:
    """The middleware chain below `CodeInterpreterMiddleware` for one eval call.

    Captured in `wrap_tool_call` while the `eval` tool call runs. `handler` is
    sync for `invoke` and async for `ainvoke`, matching `is_async`.
    """

    request: ToolCallRequest
    handler: Callable[[ToolCallRequest], Any]
    is_async: bool


@dataclass(frozen=True, slots=True)
class CallResult:
    """What a nested tool call returned, as recorded in the journal."""

    status: ToolStatus
    value: Any
    note: str | None = None
    """Text middleware addressed to the model, such as a reviewer-edit notice."""
    media: tuple[dict[str, Any], ...] = ()
    """Media blocks the model would have seen had it made the call itself."""


@dataclass(frozen=True, slots=True)
class ToolMedia:
    """Media blocks from one `tools.*` call, for the eval's result."""

    seq: int
    """The call's issue order within the eval."""
    tool: str
    blocks: tuple[dict[str, Any], ...]


def call_digest(name: str, args: dict[str, Any]) -> str:
    """Return a short digest identifying a call by tool name and arguments."""
    blob = json.dumps([name, args], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass(slots=True)
class ToolReturn:
    """What a tool's `_run` or `_arun` returned, before model formatting."""

    ran: bool = False
    value: Any = None


def recording_copy(
    tool: BaseTool, record: ToolReturn, *, config: RunnableConfig | None = None
) -> BaseTool:
    """Return a copy of `tool` that keeps its own return value in `record`.

    `run`/`arun` and the tool node format a return value for the model: they
    drop a `content_and_artifact` tool's artifact without a tool call id and
    turn a dict into JSON text with one. JavaScript wants the value itself.
    Every `BaseTool` runs through `_run` or `_arun`, so recording there works
    for any tool while callbacks, validation, and error handling still run.

    With `config`, the tool body runs under it instead of the config its
    caller set, which keeps an `interrupt()` inside the tool scoped to the
    call's own task.
    """
    copy = tool.model_copy()
    run, arun = copy._run, copy._arun  # noqa: SLF001 — the entry points every tool implements

    # `wraps` keeps the signatures `run`/`arun` inspect for `config` / `run_manager`.
    @functools.wraps(run)
    def _run(*args: Any, **kwargs: Any) -> Any:
        with _config_scope(config):
            record.value = run(*args, **kwargs)
        record.ran = True
        return record.value

    @functools.wraps(arun)
    async def _arun(*args: Any, **kwargs: Any) -> Any:
        with _config_scope(config):
            record.value = await arun(*args, **kwargs)
        record.ran = True
        return record.value

    # Instance attributes shadow the methods without touching the class.
    object.__setattr__(copy, "_run", _run)
    object.__setattr__(copy, "_arun", _arun)
    return copy


@contextlib.contextmanager
def _config_scope(config: RunnableConfig | None) -> Iterator[None]:
    if config is None:
        yield
        return
    token = var_child_runnable_config.set(config)
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


def _run_coroutine(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        async def run() -> Any:
            return await fn(*args, **kwargs)

        return asyncio.run(run())

    return wrapper


def _with_sync_entry(tool: BaseTool) -> BaseTool:
    """Give an async-only function tool a sync entry point for a sync graph.

    The REPL bridge could always run such a tool; the tool node runs it
    synchronously under `invoke`.
    """
    if (
        isinstance(tool, (StructuredTool, Tool))
        and tool.func is None
        and tool.coroutine is not None
    ):
        return tool.model_copy(update={"func": _run_coroutine(tool.coroutine)})
    return tool


def _model_text(value: Any, message: ToolMessage, response_format: str) -> Any:
    """The content the tool node gives the model for `value`, before middleware."""
    content, artifact = value, None
    if (
        response_format == "content_and_artifact"
        and isinstance(value, tuple)
        and len(value) == 2  # noqa: PLR2004 — a (content, artifact) pair
    ):
        content, artifact = value
    plain = _format_output(
        content, artifact, message.tool_call_id, message.name or "", "success"
    )
    if isinstance(plain, list):
        plain = next((m for m in reversed(plain) if isinstance(m, ToolMessage)), None)
    return plain.content if isinstance(plain, ToolMessage) else None


def _result_of(result: Any, seen: ToolReturn, tool: BaseTool) -> CallResult:
    """Reduce the chain's return value to the status and JS-facing value.

    JavaScript gets the data in what the tool itself returned. If middleware
    changed the text the model would have read for it, that text is kept as a
    note. Media blocks in the message are kept for the model too.
    """
    messages = result if isinstance(result, list) else [result]
    message = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)
    if message is not None and message.status == "error":
        return CallResult(status="error", value=coerce_tool_output_for_ptc(result))
    media = _media_of(message)
    if not seen.ran:
        value = coerce_tool_output_for_ptc(result)
        return CallResult(status="success", value=value, media=media)
    note = None
    model_text = (
        _model_text(seen.value, message, tool.response_format) if message else None
    )
    if message is not None and message.content != model_text:
        note = message.text
    value = coerce_tool_output_for_ptc(seen.value, response_format=tool.response_format)
    return CallResult(status="success", value=value, note=note, media=media)


def _media_of(message: ToolMessage | None) -> tuple[dict[str, Any], ...]:
    """Return the media blocks in a tool message's content."""
    if message is None or isinstance(message.content, str):
        return ()
    return tuple(
        block
        for block in message.content
        if isinstance(block, dict) and block.get("type") in _MEDIA_BLOCK_TYPES
    )


@dataclass
class EvalJournal:
    """Per-eval dispatcher and journal for calls JavaScript makes.

    One instance lives for one run of one `eval` tool call. A re-run after a
    resume builds a new instance; determinism across the two comes from the
    LangGraph task journal, not from state kept here.
    """

    chain: ToolChain
    outer_loop: asyncio.AbstractEventLoop | None
    next_seq: int = 0
    suspended: list[tuple[int, GraphBubbleUp]] = field(default_factory=list)
    """Bubble-ups from paused calls, with the issue order of the call."""
    inflight: dict[concurrent.futures.Future[Any], int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    """Serializes calls' reads and writes of the eval node's state across threads."""
    media: list[ToolMedia] = field(default_factory=list)
    """Media from finished calls, in completion order."""
    heap: Snapshot | None = None
    """The heap before the eval ran, restored if a call pauses.

    `None` when the context could not be snapshotted, for example while a call
    from an earlier eval is still running; a re-run then starts from the heap
    as it is.
    """

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

    async def dispatch(self, tool: BaseTool, args: dict[str, Any]) -> CallResult:
        """Run one `tools.*` call through the chain inside its own task.

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

        def run(config: RunnableConfig) -> dict[str, Any]:
            ran.append(True)
            seen, writes = ToolReturn(), []
            request = self._request(tool, call, config, seen, writes)
            try:
                return _as_record(
                    _result_of(self.chain.handler(request), seen, tool), writes
                )
            except GraphBubbleUp:
                raise
            except Exception as exc:  # noqa: BLE001  # journaled, re-raised by dispatch
                failure.append(exc)
                return _raised_record(exc, writes)

        async def arun(config: RunnableConfig) -> dict[str, Any]:
            ran.append(True)
            seen, writes = ToolReturn(), []
            request = self._request(tool, call, config, seen, writes)
            try:
                result = await self.chain.handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:  # noqa: BLE001  # journaled, re-raised by dispatch
                failure.append(exc)
                return _raised_record(exc, writes)
            return _as_record(_result_of(result, seen, tool), writes)

        name = f"ptc.{tool.name}.{call_digest(tool.name, args)}"
        # The record may not be serialized yet, so it is read, never changed.
        record = await self._journaled(name, run, arun, seq=seq)
        if not ran:
            self._replay_writes(record)
        if raised := record.get("raised"):
            # The original exception on the run that raised it, so an uncaught
            # one ends the agent run as it would for a model tool call.
            raise failure[0] if failure else RuntimeError(raised)
        return self._call_result(record, tool.name, seq)

    def _call_result(self, record: dict[str, Any], tool: str, seq: int) -> CallResult:
        """Read a call's record, keeping its note and media for the eval's output."""
        if record.get("note"):
            self.notes.append(f"tools.{tool} call {seq}: {record['note']}")
        # Records journaled before media was kept have no `media` key.
        media = tuple(record.get("media", ()))
        if media:
            self.media.append(ToolMedia(seq=seq, tool=tool, blocks=media))
        return CallResult(
            status=record["status"],
            value=record["value"],
            note=record.get("note"),
            media=media,
        )

    def _replay_writes(self, record: dict[str, Any]) -> None:
        """Send the state writes of a saved call that did not run this time."""
        if writes := record.get("writes"):
            with self.state_lock:
                self.runtime.config["configurable"][CONFIG_KEY_SEND](
                    [(channel, value) for channel, value in writes]
                )

    def _request(
        self,
        tool: BaseTool,
        call: ToolCall,
        config: RunnableConfig,
        seen: ToolReturn,
        writes: list[tuple[str, Any]],
    ) -> ToolCallRequest:
        """Build the nested call's request under its own task's config."""
        config = _state_from_eval_node(
            config, self.runtime.config, writes, self.state_lock
        )
        runtime = replace(self.runtime, tool_call_id=call["id"], config=config)
        runnable = tool if self.chain.is_async else _with_sync_entry(tool)
        watched = recording_copy(runnable, seen, config=config)
        return self.chain.request.override(
            tool_call=call, tool=watched, runtime=runtime
        )

    async def _journaled(
        self,
        name: str,
        run: Callable[[RunnableConfig], Any],
        arun: Callable[[RunnableConfig], Awaitable[Any]],
        *,
        seq: int,
    ) -> Any:
        """Run `run`/`arun` as a LangGraph task under the eval's task.

        The task is called on the thread and loop LangGraph expects: the outer
        event loop for an async graph, the calling worker thread for a sync one.
        """
        self.check_running()
        future = self._schedule(name, run, arun)
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
        self,
        name: str,
        run: Callable[[RunnableConfig], Any],
        arun: Callable[[RunnableConfig], Awaitable[Any]],
    ) -> concurrent.futures.Future[Any]:
        outer = self.runtime.config
        if not self.chain.is_async:

            def body() -> Any:
                return run(get_config())

            token = var_child_runnable_config.set(outer)
            try:
                return task(name=name)(body)()
            finally:
                var_child_runnable_config.reset(token)

        async def abody() -> Any:
            return await arun(get_config())

        async def call() -> Any:
            token = var_child_runnable_config.set(outer)
            try:
                return await task(name=name)(abody)()
            finally:
                var_child_runnable_config.reset(token)

        loop = self.outer_loop
        if loop is None:
            msg = "async tool chain requires the outer event loop"
            raise RuntimeError(msg)
        return asyncio.run_coroutine_threadsafe(call(), loop)

    def _settled(self, future: concurrent.futures.Future[Any]) -> None:
        seq = self.inflight.pop(future, self.next_seq)
        if future.cancelled():
            return
        exc = future.exception()
        if isinstance(exc, GraphBubbleUp):
            self.suspended.append((seq, exc))

    async def settle(self) -> GraphBubbleUp:
        """Wait for in-flight calls and return the bubble-up to raise.

        Interrupts from every call that paused are merged into one
        `GraphInterrupt`, in issue order, so concurrent gated calls are
        answered in one resume and listed the same way on every run.
        """
        while self.inflight:
            pending = list(self.inflight)
            await asyncio.gather(
                *(asyncio.wrap_future(f) for f in pending), return_exceptions=True
            )
            for f in pending:
                self.inflight.pop(f, None)
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

    A LangGraph task reads the state as of the step's start plus its own
    writes, so a file one call writes through `StateBackend` would be invisible
    to the next call. The call reads and writes through the `eval` node
    instead, so later calls and the call itself see its writes. Its state
    writes are also kept in `writes` for the journal record, and `dispatch`
    sends them again on a re-run that returns the record without running the
    call; LangGraph drops a node's writes when it pauses, so they land once.
    LangGraph's own channels, such as the resume value `interrupt()` writes,
    all start with `__` and stay with the call's task.

    Concurrent calls from one eval run on other threads, and the node's writes
    are one shared list that a read iterates, so `lock` serializes their reads
    and sends. It does not cover a model tool call running in parallel in the
    same step, which writes to that list without it, as it already could.
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


def _as_record(result: CallResult, writes: list[tuple[str, Any]]) -> dict[str, Any]:
    """Plain-dict form that the checkpointer serializes."""
    return {
        "status": result.status,
        "value": result.value,
        "note": result.note,
        "media": list(result.media),
        "writes": writes,
    }


def _raised_record(exc: Exception, writes: list[tuple[str, Any]]) -> dict[str, Any]:
    """Record a tool that raised, so a re-run raises again instead of re-running it.

    LangGraph re-runs a task that raised. Recording the failure keeps the call's
    side effects to one run and gives JavaScript the same outcome both times.
    """
    return {"raised": f"{type(exc).__name__}: {exc}", "writes": writes}
