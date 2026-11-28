"""Host side: a `Slot` for `CodeInterpreterMiddleware` backed by worker processes.

The middleware, its PTC tools and their credentials stay in this process. A
worker only ever sees code, tool names, JSON inputs and JSON results.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import pwd
import shutil
import sys
import tempfile
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_quickjs import EvalOutcome, SlotConfig

from qjs_worker import protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain.tools import ToolRuntime
    from langchain_core.tools import BaseTool

# Errors after which a worker is not trusted to run anything else.
RECYCLE_ON = {"Timeout", "OutOfMemory", "WorkerDied", "HostTimeout"}


@dataclass
class WorkerConfig:
    """How workers are started and when they are thrown away."""

    host_timeout: float = 10.0  # wall clock after which the host kills the worker
    cpu_seconds: int = 60  # RLIMIT_CPU over the worker's whole life
    max_files: int = 16
    max_evals: int = 100  # recycle after this many evals
    seccomp: bool = sys.platform == "linux"
    user: str = "nobody"  # applied only when the host runs as root
    # Prefix for the worker command, e.g. ["runsc", "--network=none", "do"] or
    # ["unshare", "--user", "--net", "--"]. The protocol only needs stdin/stdout.
    launcher: list[str] = field(default_factory=list)
    allow_probe: bool = False  # demo only: lets escape_check.py act as an escaped guest


class WorkerProcess:
    """One spawned worker; `spawn` returns once it has locked itself down."""

    def __init__(self, proc: asyncio.subprocess.Process, cwd: str, ready: dict) -> None:
        self.proc = proc
        self.cwd = cwd
        self.ready = ready
        self.evals = 0

    @classmethod
    async def spawn(cls, config: WorkerConfig) -> WorkerProcess:
        cwd = tempfile.mkdtemp(prefix="qjs-worker-")
        account = pwd.getpwnam(config.user) if os.geteuid() == 0 else None
        if account:
            os.chown(cwd, account.pw_uid, account.pw_gid)
        worker_config = {
            "cpu_seconds": config.cpu_seconds,
            "max_files": config.max_files,
            "seccomp": config.seccomp,
            "allow_probe": config.allow_probe,
        }
        proc = await asyncio.create_subprocess_exec(
            *config.launcher, sys.executable, "-I", "-m", "qjs_worker.worker", json.dumps(worker_config),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            env={},  # nothing inherited: no API keys, no cloud credentials, no PATH
            cwd=cwd,
            close_fds=True,
            start_new_session=True,
            user=account.pw_uid if account else None,
            group=account.pw_gid if account else None,
            extra_groups=[] if account else None,
            limit=protocol.MAX_FRAME,
        )  # fmt: skip
        ready = await asyncio.wait_for(protocol.read(proc.stdout), config.host_timeout)
        if ready is None:
            msg = f"worker failed to start (exit code {await proc.wait()})"
            raise RuntimeError(msg)
        return cls(proc, cwd, ready)

    async def send(self, msg: dict) -> None:
        self.proc.stdin.write(protocol.encode(msg))
        await self.proc.stdin.drain()

    async def recv(self) -> dict | None:
        return await protocol.read(self.proc.stdout)

    async def request(self, msg: dict) -> dict | None:
        await self.send(msg)
        return await self.recv()

    async def kill(self) -> None:
        if self.proc.returncode is None:
            self.proc.kill()
            await self.proc.wait()
        shutil.rmtree(self.cwd, ignore_errors=True)


class WorkerPool:
    """Keeps `size` workers starting or started, so an eval rarely waits on a spawn."""

    def __init__(self, config: WorkerConfig, size: int = 2) -> None:
        self.config = config
        self.size = size
        self.spawning: deque[asyncio.Task[WorkerProcess]] = deque()
        self.slots: list[ProcessSlot] = []

    async def start(self) -> None:
        while len(self.spawning) < self.size:
            self._spawn()

    def _spawn(self) -> None:
        self.spawning.append(asyncio.ensure_future(WorkerProcess.spawn(self.config)))

    async def take(self) -> WorkerProcess:
        """Return the oldest spawn, raising its error if it failed, and start a replacement."""
        if not self.spawning:
            self._spawn()
        task = self.spawning.popleft()
        self._spawn()
        return await task

    def slot(self, name: str, config: SlotConfig) -> ProcessSlot:
        """`slot_factory` for `CodeInterpreterMiddleware`."""
        del name  # workers are named by pid
        self.slots.append(ProcessSlot(self, config))
        return self.slots[-1]

    async def aclose(self) -> None:
        """Kill every worker: the slots' and the spares."""
        for slot in self.slots:
            await slot.aclose()
        for worker in await asyncio.gather(*self.spawning, return_exceptions=True):
            if isinstance(worker, WorkerProcess):
                await worker.kill()


class ProcessSlot:
    """A `Slot` whose QuickJS Context lives in a pooled worker process.

    The worker is the Context: closing the Context kills it, and the next eval
    takes a fresh one from the pool. Async only, so run the agent with
    `ainvoke`/`astream`. `fs` and `task()` are not bridged; create the
    middleware with `subagents=False` and no `fs`.
    """

    def __init__(self, pool: WorkerPool, config: SlotConfig) -> None:
        self.pool = pool
        self.config = config
        self.worker: WorkerProcess | None = None
        self.tools: dict[str, BaseTool] = {}
        self.eval_running = False
        self.snapshot: bytes | None = None  # the snapshot the live worker still matches
        self.restored = False  # a worker restores a snapshot at most once

    async def _live_worker(self) -> WorkerProcess:
        if self.worker is None or self.worker.proc.returncode is not None:
            self.worker = await self.pool.take()
            self.snapshot, self.restored = None, False
            await self.worker.send({"op": "init", "memory_limit": self.config.memory_limit})
        return self.worker

    async def _close_worker(self) -> None:
        if self.worker is not None:
            await self.worker.kill()
        self.worker, self.snapshot = None, None

    async def eval_async(
        self,
        code: str,
        *,
        outer_runtime: ToolRuntime | None = None,
        tools: Sequence[BaseTool] | None = None,
        outer_loop: asyncio.AbstractEventLoop | None = None,
        fs: object = None,
        close_context_after: bool = False,
    ) -> EvalOutcome:
        del outer_loop  # tools run on the caller's loop, which is this one
        if fs is not None:
            return EvalOutcome(error_type="UnsupportedCommand", error_message="fs is not available in this sandbox")
        if self.eval_running:
            return EvalOutcome(error_type="ConcurrentEval", error_message="another eval call is still running")
        if tools is not None:
            self.tools = {to_camel_case(tool.name): tool for tool in tools}
        worker = await self._live_worker()
        self.eval_running = True
        worker.evals += 1
        self.snapshot = None
        outcome = None
        try:
            outcome = await asyncio.wait_for(self._run(worker, code, outer_runtime), self.pool.config.host_timeout)
        except TimeoutError:
            outcome = EvalOutcome(error_type="HostTimeout", error_message="worker killed after host_timeout")
        finally:
            self.eval_running = False
            # No outcome means the eval was cancelled with its script still
            # running: the worker can't be reused.
            recycle = outcome is None or outcome.error_type in RECYCLE_ON
            if recycle or close_context_after or worker.evals >= self.pool.config.max_evals:
                await self._close_worker()
        return outcome

    async def _run(self, worker: WorkerProcess, code: str, outer_runtime: ToolRuntime | None) -> EvalOutcome:
        timeout = self.config.timeout if self.config.timeout is not None else float("inf")
        await worker.send({"op": "eval", "code": code, "timeout": timeout})
        # Frames from the worker are untrusted: size-capped by `protocol`, and
        # all they can do is name a tool in `self.tools` and pass it JSON.
        config = outer_runtime.config if outer_runtime is not None else None
        calls: set[asyncio.Task] = set()
        limit, made = self.config.max_ptc_calls, 0
        try:
            while (msg := await worker.recv()) is not None:
                if msg["op"] == "tool_call":
                    made += 1
                    over_budget = limit is not None and made > limit
                    task = asyncio.ensure_future(self._call_tool(worker, msg, config, over_budget=over_budget))
                    calls.add(task)
                    task.add_done_callback(calls.discard)
                elif msg["op"] == "done":
                    return self._outcome(msg)
            return EvalOutcome(error_type="WorkerDied", error_message=f"worker exited ({await worker.proc.wait()})")
        finally:
            for task in calls:
                task.cancel()

    def _outcome(self, done: dict) -> EvalOutcome:
        limit = self.config.max_stdout_chars
        console = done["console"] if self.config.capture_console else ""
        result = done.get("result")
        return EvalOutcome(
            stdout=console[:limit],
            stdout_truncated_chars=max(0, len(console) - limit),
            result=None if "error_type" in done else result if isinstance(result, str) else json.dumps(result),
            error_type=done.get("error_type"),
            error_message=done.get("error_message", ""),
        )

    async def _call_tool(self, worker: WorkerProcess, msg: dict, config: object, *, over_budget: bool) -> None:
        tool = self.tools.get(msg["name"])
        reply = {"op": "tool_result", "id": msg["id"], "ok": False}
        if over_budget:
            reply["error"] = f"PTCCallBudgetExceeded: more than {self.config.max_ptc_calls} tool calls in one eval"
        elif tool is None:
            reply["error"] = f"tools.{msg['name']} is not available"
        else:
            try:
                reply |= {"ok": True, "value": await tool.ainvoke(msg["input"], config=config)}
            except Exception as e:  # noqa: BLE001 - a tool failure becomes a JS exception
                reply["error"] = f"{type(e).__name__}: {e}"
        await worker.send(reply)

    async def arestore_snapshot(self, payload: bytes) -> None:
        worker = await self._live_worker()
        if self.restored:
            return
        self.restored = True
        await worker.request({"op": "restore", "data": base64.b64encode(payload).decode()})

    async def acreate_snapshot(self) -> bytes | None:
        if self.worker is None or self.snapshot is not None:
            return None
        reply = await self.worker.request({"op": "snapshot"})
        self.snapshot, self.restored = base64.b64decode(reply["data"]), True
        return self.snapshot

    async def aclose_context(self, *, keep_snapshot: bytes | None = None) -> None:
        if keep_snapshot is None or keep_snapshot != self.snapshot:
            await self._close_worker()

    async def aclose(self) -> None:
        await self._close_worker()

    def eval_sync(self, code: str, **kwargs: object) -> EvalOutcome:
        raise NotImplementedError(ASYNC_ONLY)

    def restore_snapshot(self, payload: bytes) -> None:
        raise NotImplementedError(ASYNC_ONLY)

    def create_snapshot(self) -> bytes | None:
        raise NotImplementedError(ASYNC_ONLY)

    def close_context(self, *, keep_snapshot: bytes | None = None) -> None:
        raise NotImplementedError(ASYNC_ONLY)

    def close(self) -> None:
        raise NotImplementedError(ASYNC_ONLY)


ASYNC_ONLY = "ProcessSlot is async only; run the agent with ainvoke/astream"


def to_camel_case(name: str) -> str:
    """`list_tickets` -> `listTickets`, the name the middleware's prompt uses."""
    head, *rest = name.replace("-", "_").split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)
