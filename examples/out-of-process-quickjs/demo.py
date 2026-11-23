"""Run model-written JavaScript in a worker that holds no credentials.

The host has an API key in its environment and a token file on disk, the way
an agent server does. Tools called from JavaScript run here and can use them;
the worker running the JavaScript cannot reach either.
"""

import asyncio
import os
import tempfile

from langchain_core.tools import ToolException, tool

from qjs_worker.host import Interpreter, WorkerConfig, WorkerPool

os.environ["CLOUD_API_KEY"] = "fake-secret"
# Stands in for a mounted service-account token. World-readable on purpose:
# projected tokens often are, so file permissions alone don't protect it.
with tempfile.NamedTemporaryFile("w", suffix="-token", delete=False) as token_file:
    token_file.write("fake-service-account-token")
os.chmod(token_file.name, 0o644)


@tool
async def lookup_asset(asset_id: int) -> dict:
    """Look up an asset's severity."""
    assert os.environ["CLOUD_API_KEY"] == "fake-secret"
    return {"id": asset_id, "severity": "high" if asset_id % 2 else "low"}


@tool
async def delete_asset(asset_id: int) -> str:
    """Delete an asset."""
    msg = f"not authorized to delete asset {asset_id}"
    raise ToolException(msg)


SCRIPT = """
const assets = await Promise.all([1, 2, 3, 4].map((id) => tools.lookup_asset({ asset_id: id })));
let refused;
try { await tools.delete_asset({ asset_id: 1 }); } catch (e) { refused = e.message; }
console.log(`${assets.filter((a) => a.severity === "high").length} high-severity assets`);
({ assets, refused })
"""


async def main() -> None:
    pool = WorkerPool(WorkerConfig(allow_probe=True), size=2)
    await pool.start()
    interpreter = Interpreter(pool, [lookup_asset, delete_asset])
    eval_tool = interpreter.as_tool()

    print("1. tools.* calls run in the host:")
    print("  ", await eval_tool.ainvoke({"code": SCRIPT}))
    print(f"   worker: {interpreter.worker.ready}")

    print("2. Python running inside a worker, as if it had escaped wasm:")
    worker = await pool.take()
    await worker.send({"op": "probe", "paths": [token_file.name, "/proc/1/environ"]})
    for check, result in (await worker.recv())["results"].items():
        print(f"   {check:45} {result}")
    await worker.kill()

    print("3. Runaway scripts recycle the worker:")
    for code in ("while (true) {}", "const a = []; while (true) a.push(new Array(1e5).fill(1));"):
        pid = interpreter.worker and interpreter.worker.ready["pid"]
        outcome = await interpreter.eval(code)
        after = await interpreter.eval("1 + 1")
        print(
            f"   {outcome.error_type}: worker {pid} replaced by {interpreter.worker.ready['pid']}, next eval = {after.result}"
        )

    await interpreter.aclose()
    await pool.aclose()
    os.unlink(token_file.name)


if __name__ == "__main__":
    asyncio.run(main())
