"""Worker process: runs untrusted JavaScript and holds no credentials.

The host starts it with an empty environment, as `nobody` when the host is
root, in an empty working directory. Startup order matters: rlimits, then
load QuickJS (wasm compile, event loop, pipes), then seccomp, then `ready`.
Untrusted code only arrives after the filter is in place.

`tools.*` calls leave as `tool_call` frames on stdout; the host runs the tool
and answers with a `tool_result` frame on stdin.
"""

from __future__ import annotations

import asyncio
import errno
import itertools
import json
import os
import resource
import socket
import subprocess
import sys

from quickjs_rs import JSError, MemoryLimitError, Runtime
from quickjs_rs import TimeoutError as QjsTimeoutError

from qjs_worker import protocol

TOOLS_NAMESPACE = """
globalThis.tools = new Proxy({}, {
  get: (_, name) => async (input) => {
    const reply = await __tools_call(name, input ?? null);
    if (!reply.ok) throw new Error(reply.error);
    return reply.value;
  },
});
globalThis.console = { log: __console_log };
undefined
"""

# Syscalls an escaped guest must not make. A denylist keeps the example
# short; production should use an allowlist built from tracing a real worker.
DENIED_SYSCALLS = (
    "socket", "socketpair", "connect", "bind", "listen", "accept", "accept4",
    "open", "openat", "openat2", "creat", "execve", "execveat", "fork", "vfork",
    "ptrace", "process_vm_readv", "process_vm_writev", "kill", "mount", "unshare", "setns",
)  # fmt: skip
CLONE_THREAD = 0x00010000


def apply_rlimits(cpu_seconds: int, max_files: int) -> None:
    """Cap CPU time, open files and core dumps for this process."""
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_NOFILE, (max_files, max_files))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def apply_seccomp() -> None:
    """Deny network, file opens, exec and new processes (Linux only)."""
    import pyseccomp as seccomp

    syscall_filter = seccomp.SyscallFilter(defaction=seccomp.ALLOW)  # sets NO_NEW_PRIVS
    for name in DENIED_SYSCALLS:
        syscall_filter.add_rule(seccomp.ERRNO(errno.EPERM), name)
    # Threads are fine (wasmtime runs a thread pool); new processes are not.
    # clone3 gets ENOSYS so libc falls back to clone, where flags are visible.
    no_thread_flag = seccomp.Arg(0, seccomp.MASKED_EQ, CLONE_THREAD, 0)
    syscall_filter.add_rule(seccomp.ERRNO(errno.EPERM), "clone", no_thread_flag)
    syscall_filter.add_rule(seccomp.ERRNO(errno.ENOSYS), "clone3")
    syscall_filter.load()


class Worker:
    """One QuickJS context whose host functions are frames to the host."""

    def __init__(self, memory_limit: int, *, allow_probe: bool) -> None:
        self.runtime = Runtime(memory_limit=memory_limit)
        self.ctx = self.runtime.new_context()
        self.allow_probe = allow_probe
        self.console: list[str] = []
        self.pending: dict[int, asyncio.Future[dict]] = {}
        self.call_ids = itertools.count(1)
        self.ctx.register("__tools_call", self.tools_call, is_async=True)
        self.ctx.register("__console_log", self.console_log)
        self.ctx.eval(TOOLS_NAMESPACE)

    def console_log(self, *args: object) -> None:
        self.console.append(" ".join(a if isinstance(a, str) else json.dumps(a) for a in args))

    async def tools_call(self, name: str, payload: object) -> dict:
        call_id = next(self.call_ids)
        self.pending[call_id] = asyncio.get_running_loop().create_future()
        send({"op": "tool_call", "id": call_id, "name": name, "input": payload})
        return await self.pending[call_id]

    async def evaluate(self, code: str, timeout: float) -> object:
        # As in langchain-quickjs, a final-expression Promise resolves to its value.
        handle = await self.ctx.eval_handle_async(code, timeout=timeout)
        resolved = await handle.await_promise(timeout=timeout) if handle.is_promise() else handle
        try:
            return resolved.to_python()
        finally:
            if resolved is not handle:
                resolved.dispose()
            handle.dispose()

    async def run_eval(self, msg: dict) -> None:
        done: dict = {"op": "done"}
        try:
            done["result"] = await self.evaluate(msg["code"], msg["timeout"])
        except JSError as e:
            done |= {"error_type": e.name, "error_message": e.message}
        except QjsTimeoutError as e:
            done |= {"error_type": "Timeout", "error_message": str(e)}
        except MemoryLimitError as e:
            done |= {"error_type": "OutOfMemory", "error_message": str(e)}
        except Exception as e:  # noqa: BLE001 - always answer so the host isn't left waiting
            done |= {"error_type": type(e).__name__, "error_message": str(e)}
        done["console"], self.console = "\n".join(self.console), []
        send(done)

    async def serve(self, reader: asyncio.StreamReader) -> None:
        while (msg := await protocol.read(reader)) is not None:
            if msg["op"] == "eval":
                asyncio.ensure_future(self.run_eval(msg))
            elif msg["op"] == "tool_result":
                self.pending.pop(msg["id"]).set_result(msg)
            elif msg["op"] == "probe" and self.allow_probe:
                send({"op": "probe", "results": probe(msg["paths"])})


def send(msg: dict) -> None:
    protocol.write_sync(sys.stdout.buffer, msg)


def probe(paths: list[str]) -> dict[str, str]:
    """Demo only: act like code that escaped wasm and runs as this process."""

    def attempt(action: object) -> str:
        try:
            return f"ALLOWED: {action()!r}"[:100]
        except OSError as e:
            return f"denied: {e.strerror}"

    results = {"environment": attempt(lambda: dict(os.environ))}
    for path in paths:
        results[f"read {path}"] = attempt(lambda path=path: open(path).read())  # noqa: SIM115
    results["open a socket"] = attempt(lambda: socket.socket().connect(("1.1.1.1", 443)))
    results["run /bin/sh"] = attempt(
        lambda: subprocess.run(["/bin/sh", "-c", "id"], capture_output=True, check=False).stdout
    )
    return results


async def main(config: dict) -> None:
    apply_rlimits(config["cpu_seconds"], config["max_files"])
    worker = Worker(config["memory_limit"], allow_probe=config["allow_probe"])
    reader = await protocol.stdin_reader()
    if config["seccomp"]:
        apply_seccomp()
    send({"op": "ready", "pid": os.getpid(), "uid": os.getuid(), "seccomp": config["seccomp"]})
    await worker.serve(reader)


if __name__ == "__main__":
    asyncio.run(main(json.loads(sys.argv[1])))
