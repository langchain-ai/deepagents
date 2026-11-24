"""Host side: spawn credential-less QuickJS workers and run their tools here.

Tools run in this process, with its credentials, config and callbacks. A
worker only ever sees tool names, JSON inputs and JSON results.
"""

from __future__ import annotations

import asyncio
import json
import os
import pwd
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from qjs_worker import protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool

# Errors after which a worker is not trusted to run anything else.
RECYCLE_ON = {"Timeout", "OutOfMemory", "WorkerDied", "HostTimeout"}


@dataclass
class WorkerConfig:
    """How workers are started and when they are thrown away."""

    memory_limit: int = 64 * 1024 * 1024  # QuickJS heap, enforced inside wasm
    eval_timeout: float = 5.0  # QuickJS's own per-eval timeout
    host_timeout: float = 10.0  # wall clock after which the host kills the worker
    cpu_seconds: int = 60  # RLIMIT_CPU over the worker's whole life
    max_files: int = 16
    max_evals: int = 100  # recycle after this many evals
    seccomp: bool = sys.platform == "linux"
    user: str = "nobody"  # applied only when the host runs as root
    # Prefix for the worker command, e.g. ["runsc", "--network=none", "do"] or
    # ["unshare", "--user", "--net", "--"]. The protocol only needs stdin/stdout.
    launcher: list[str] = field(default_factory=list)
    allow_probe: bool = False  # demo only: lets demo.py act as an escaped guest


@dataclass
class Outcome:
    """Result of one eval."""

    result: object = None
    console: str = ""
    error_type: str | None = None
    error_message: str = ""


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
            "memory_limit": config.memory_limit,
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

    async def kill(self) -> None:
        if self.proc.returncode is None:
            self.proc.kill()
            await self.proc.wait()
        shutil.rmtree(self.cwd, ignore_errors=True)


class WorkerPool:
    """Keeps `size` workers started, so an eval never waits on a spawn."""

    def __init__(self, config: WorkerConfig, size: int = 2) -> None:
        self.config = config
        self.size = size
        self.ready: asyncio.Queue[WorkerProcess] = asyncio.Queue()
        self.spawning: set[asyncio.Task] = set()

    async def start(self) -> None:
        for _ in range(self.size):
            self._replace()

    def _replace(self) -> None:
        task = asyncio.ensure_future(self._spawn())
        self.spawning.add(task)
        task.add_done_callback(self.spawning.discard)

    async def _spawn(self) -> None:
        self.ready.put_nowait(await WorkerProcess.spawn(self.config))

    async def take(self) -> WorkerProcess:
        """Return a started worker and begin spawning its replacement."""
        worker = await self.ready.get()
        self._replace()
        return worker

    async def aclose(self) -> None:
        await asyncio.gather(*self.spawning)
        while not self.ready.empty():
            await self.ready.get_nowait().kill()


class Interpreter:
    """One conversation's interpreter: evals run one at a time on a pooled worker."""

    def __init__(self, pool: WorkerPool, tools: Sequence[BaseTool]) -> None:
        self.pool = pool
        self.config = pool.config
        self.tools = {tool.name: tool for tool in tools}
        self.worker: WorkerProcess | None = None
        self.lock = asyncio.Lock()

    async def eval(self, code: str, config: RunnableConfig | None = None) -> Outcome:
        """Run `code`; host tools it calls get `config` (the caller's user, callbacks, tracing)."""
        async with self.lock:
            if self.worker is None or self.worker.proc.returncode is not None:
                self.worker = await self.pool.take()
            self.worker.evals += 1
            try:
                outcome = await asyncio.wait_for(self._run(code, config), self.config.host_timeout)
            except TimeoutError:
                outcome = Outcome(error_type="HostTimeout", error_message="worker killed after host_timeout")
            if outcome.error_type in RECYCLE_ON or self.worker.evals >= self.config.max_evals:
                await self.worker.kill()
                self.worker = None
            return outcome

    async def _run(self, code: str, config: RunnableConfig | None) -> Outcome:
        worker = self.worker
        await worker.send({"op": "eval", "code": code, "timeout": self.config.eval_timeout})
        # Frames from the worker are untrusted: size-capped by `protocol`, and
        # all they can do is name a tool in `self.tools` and pass it JSON.
        calls: set[asyncio.Task] = set()
        try:
            while (msg := await worker.recv()) is not None:
                if msg["op"] == "tool_call":
                    task = asyncio.ensure_future(self._call_tool(worker, msg, config))
                    calls.add(task)
                    task.add_done_callback(calls.discard)
                elif msg["op"] == "done":
                    return Outcome(
                        msg.get("result"), msg["console"], msg.get("error_type"), msg.get("error_message", "")
                    )
            return Outcome(error_type="WorkerDied", error_message=f"worker exited ({await worker.proc.wait()})")
        finally:
            for task in calls:
                task.cancel()

    async def _call_tool(self, worker: WorkerProcess, msg: dict, config: RunnableConfig | None) -> None:
        tool = self.tools.get(msg["name"])
        reply = {"op": "tool_result", "id": msg["id"], "ok": False}
        if tool is None:
            reply["error"] = f"tools.{msg['name']} is not available"
        else:
            try:
                reply |= {"ok": True, "value": await tool.ainvoke(msg["input"], config=config)}
            except Exception as e:  # noqa: BLE001 - a tool failure becomes a JS exception
                reply["error"] = f"{type(e).__name__}: {e}"
        await worker.send(reply)

    async def aclose(self) -> None:
        if self.worker is not None:
            await self.worker.kill()
