"""Inline execution under a host that interrupts busy threads.

Durable-execution hosts such as Temporal treat a workflow thread that does not
yield within a deadline (2s by default) as deadlocked and inject an exception
into it with `PyThreadState_SetAsyncExc`. With `execution="inline"` that thread
is running the QuickJS guest, so the exception lands inside a wasmtime host
callback. The scenario runs in a subprocess because the failure mode is a
process crash.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

_SCENARIO = textwrap.dedent(
    """
    import asyncio, ctypes, threading, time

    from langchain_quickjs import CodeInterpreterMiddleware

    class HostInterrupt(BaseException):
        pass

    def workflow() -> None:
        middleware = CodeInterpreterMiddleware(execution="inline", subagents=False)

        async def run() -> None:
            repl = middleware._registry.get("slot")
            try:
                outcome = await repl.eval_async(
                    "const end = Date.now() + 3000; while (Date.now() < end) {} 1"
                )
                print("OUTCOME", outcome.error_type, flush=True)
            finally:
                await middleware._registry.aevict("slot")

        try:
            asyncio.run(run())
        except HostInterrupt:
            print("INTERRUPTED", flush=True)

    thread = threading.Thread(target=workflow)
    thread.start()
    time.sleep(2.0)  # the host's deadlock timeout
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread.ident), ctypes.py_object(HostInterrupt)
    )
    thread.join(10)
    print("DONE", flush=True)
    """
)


def test_default_inline_eval_survives_a_host_thread_interrupt() -> None:
    result = subprocess.run(  # noqa: S603 — fixed interpreter and inline script
        [sys.executable, "-c", _SCENARIO],
        capture_output=True,
        text=True,
        timeout=25,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stderr[-2000:])
    assert "OUTCOME Timeout" in result.stdout, result.stdout
    assert "DONE" in result.stdout, result.stdout
