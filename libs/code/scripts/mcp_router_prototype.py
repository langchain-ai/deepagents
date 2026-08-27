"""Can dcode build its own composite router: prefixes + per-server status + middleware?"""
import asyncio, contextlib, sys
from pathlib import Path

from fastmcp import Client, FastMCP
from fastmcp.client.transports import StdioTransport
from fastmcp.server.providers.proxy import StatefulProxyClient
from fastmcp.server.server import create_proxy
from fastmcp.server.middleware import Middleware, MiddlewareContext

DEMO = str(Path(__file__).parent / "server_demo.py")


class Recorder(Middleware):
    """Client-side interception, via the router that fronts every backend."""
    def __init__(self): self.calls = []
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        self.calls.append(context.message.name)
        return await call_next(context)


async def main() -> None:
    servers = {
        "good": {"command": sys.executable, "args": [DEMO]},
        "broken": {"command": sys.executable, "args": ["/nonexistent.py"]},
    }
    recorder = Recorder()
    router = FastMCP(name="dcode-router")
    router.add_middleware(recorder)

    statuses = {}
    async with contextlib.AsyncExitStack() as stack:
        for name, cfg in servers.items():
            try:
                transport = StdioTransport(command=cfg["command"], args=cfg["args"])
                backend = StatefulProxyClient(transport=transport)
                await stack.enter_async_context(backend)
                proxy = create_proxy(backend)
                router.mount(proxy, namespace=name)
                statuses[name] = ("ok", None)
            except Exception as exc:
                statuses[name] = ("error", f"{type(exc).__name__}: {exc}"[:70])

        async with Client(router) as client:
            tools = await client.list_tools()
            print("TOOLS:", sorted(t.name for t in tools))
            res = await client.call_tool("good_add", {"a": 7, "b": 8})
            print("CALL:", res.content[0].text)

    print("STATUS:", statuses)
    print("MIDDLEWARE SAW:", recorder.calls)

asyncio.run(main())
