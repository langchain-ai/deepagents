"""Check what a worker can reach if code breaks out of wasm.

The host has an API key in its environment and a token file on disk. A
demo-only `probe` frame runs Python inside a worker, standing in for code
that escaped QuickJS's wasm sandbox. Then runaway scripts show recycling.
"""

import asyncio
import os
import tempfile

from langchain_quickjs import SlotConfig

from qjs_worker.host import WorkerConfig, WorkerPool

os.environ["TICKETS_API_TOKEN"] = "fake-secret"
# Stands in for a mounted service-account token. World-readable on purpose:
# projected tokens often are, so file permissions alone don't protect it.
with tempfile.NamedTemporaryFile("w", suffix="-token", delete=False) as token_file:
    token_file.write("fake-service-account-token")
os.chmod(token_file.name, 0o644)


async def main() -> None:
    pool = WorkerPool(WorkerConfig(allow_probe=True), size=2)
    await pool.start()

    print("Python running inside a worker, as if it had escaped wasm:")
    worker = await pool.take()
    print(f"   worker: {worker.ready}")
    await worker.send({"op": "probe", "paths": [token_file.name, "/proc/1/environ"]})
    for check, result in (await worker.recv())["results"].items():
        print(f"   {check:45} {result}")
    await worker.kill()

    print("Runaway scripts recycle the worker:")
    slot = pool.slot("escape-check", SlotConfig())
    await slot.eval_async("1")
    for code in ("while (true) {}", "const a = []; while (true) a.push(new Array(1e5).fill(1));"):
        pid = slot.worker.ready["pid"]
        outcome = await slot.eval_async(code)
        after = await slot.eval_async("1 + 1")
        print(
            f"   {outcome.error_type}: worker {pid} replaced by {slot.worker.ready['pid']}, next eval = {after.result}"
        )

    await pool.aclose()
    os.unlink(token_file.name)


if __name__ == "__main__":
    asyncio.run(main())
