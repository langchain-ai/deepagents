"""MCP loads own backend lifetimes through failure, cancellation, and reuse."""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest
from fastmcp import FastMCP
from fastmcp.client.transports import FastMCPTransport
from fastmcp.server.middleware import Middleware

from deepagents_code import mcp_tools

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import Path

    import mcp_types
    from fastmcp.server.middleware import CallNext, MiddlewareContext
    from fastmcp.tools import Tool


class Backend:
    """A real server with observable session lifetimes and discovery barriers."""

    def __init__(self, name: str) -> None:
        self.active = 0
        self.started = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.failure = False
        owner = self

        @asynccontextmanager
        async def lifespan(_server: FastMCP) -> AsyncIterator[dict[str, object]]:
            owner.active += 1
            owner.started += 1
            try:
                yield {}
            finally:
                owner.active -= 1

        class Discovery(Middleware):
            async def on_list_tools(
                self,
                context: MiddlewareContext[mcp_types.ListToolsRequest],
                call_next: CallNext[mcp_types.ListToolsRequest, Sequence[Tool]],
            ) -> Sequence[Tool]:
                owner.entered.set()
                await owner.release.wait()
                if owner.failure:
                    msg = "discovery refused"
                    raise RuntimeError(msg)
                return await call_next(context)

        self.server = FastMCP(name, lifespan=lifespan)
        self.server.add_middleware(Discovery())

        @self.server.tool
        def echo() -> str:
            """Report the backend name."""
            return name


@pytest.fixture
def backends(monkeypatch: pytest.MonkeyPatch) -> dict[str, Backend]:
    registry: dict[str, Backend] = {}
    monkeypatch.setattr(mcp_tools, "_check_stdio_server", lambda *_: None)
    monkeypatch.setattr(
        mcp_tools,
        "_build_transport",
        lambda name, *_args, **_kwargs: FastMCPTransport(registry[name].server),
    )
    return registry


def config(*names: str) -> dict[str, object]:
    return {"mcpServers": {name: {"command": "unused"} for name in names}}


async def test_repeated_adoption_keeps_all_loads(backends: dict[str, Backend]) -> None:
    manager = mcp_tools.MCPSessionManager()
    backends.update({name: Backend(name) for name in ("first", "second")})
    first, _, _ = await mcp_tools._load_tools_from_config(
        config("first"), session_manager=manager, stateless=True
    )
    second, _, _ = await mcp_tools._load_tools_from_config(
        config("second"), session_manager=manager
    )
    try:
        assert "first" in str(await first[0].ainvoke({}))
        assert "second" in str(await second[0].ainvoke({}))
        assert all(backend.active for backend in backends.values())
    finally:
        await manager.cleanup()
    assert all(backend.active == 0 for backend in backends.values())


@pytest.mark.parametrize("race", [False, True])
async def test_rejected_adoption_closes_load(
    backends: dict[str, Backend], race: bool
) -> None:
    backend = backends["server"] = Backend("server")
    manager = mcp_tools.MCPSessionManager()
    if not race:
        await manager.cleanup()
    backend.release.clear()
    task = asyncio.create_task(
        mcp_tools._load_tools_from_config(config("server"), session_manager=manager)
    )
    await asyncio.wait_for(backend.entered.wait(), 5)
    await manager.cleanup()
    backend.release.set()
    with pytest.raises(RuntimeError, match="closed MCP session manager"):
        await task
    assert backend.started > 0
    assert backend.active == 0


async def test_stateless_calls_close_every_session(
    backends: dict[str, Backend],
) -> None:
    backend = backends["server"] = Backend("server")
    tools, manager, infos = await mcp_tools._load_tools_from_config(
        config("server"), stateless=True
    )
    assert infos[0].status == "ok"
    assert manager is None
    assert backend.active == 0
    for _ in range(2):
        started = backend.started
        assert "server" in str(await tools[0].ainvoke({}))
        assert backend.started > started
        assert backend.active == 0


async def test_failed_discovery_closes_before_return(
    backends: dict[str, Backend],
) -> None:
    backend = backends["bad"] = Backend("bad")
    backend.failure = True
    backends["good"] = Backend("good")
    tools, manager, infos = await mcp_tools._load_tools_from_config(
        config("bad", "good")
    )
    try:
        assert [info.status for info in infos] == ["error", "ok"]
        assert backend.started > 0
        assert backend.active == 0
        assert "good" in str(await tools[0].ainvoke({}))
    finally:
        assert manager is not None
        await manager.cleanup()


async def test_cancelled_discovery_closes_backends(
    backends: dict[str, Backend],
) -> None:
    backend = backends["server"] = Backend("server")
    backend.release.clear()
    task = asyncio.create_task(mcp_tools._load_tools_from_config(config("server")))
    await asyncio.wait_for(backend.entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend.active == 0


async def test_cancelled_router_cleanup_finishes_backends(
    backends: dict[str, Backend], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = backends["server"] = Backend("server")
    _, manager, _ = await mcp_tools._load_tools_from_config(config("server"))
    assert manager is not None
    assert manager.client is not None
    entered, release = asyncio.Event(), asyncio.Event()
    close = manager.client.close

    async def delayed_close() -> None:
        entered.set()
        await release.wait()
        await close()

    monkeypatch.setattr(manager.client, "close", delayed_close)
    task = asyncio.create_task(manager.cleanup())
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert backend.active > 0
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend.active == 0
    await manager.cleanup()


async def test_discovery_overlaps_with_bounded_connections(
    backends: dict[str, Backend], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_tools, "_MCP_LOAD_CONCURRENCY", 2)
    for name in ("first", "second", "third"):
        backends[name] = Backend(name)
        backends[name].release.clear()
    task = asyncio.create_task(mcp_tools._load_tools_from_config(config(*backends)))
    try:
        await asyncio.wait_for(
            asyncio.gather(
                *(backends[name].entered.wait() for name in ("first", "second"))
            ),
            5,
        )
        assert backends["third"].started == 0
        backends["second"].release.set()
        await asyncio.wait_for(backends["third"].entered.wait(), 5)
    finally:
        for backend in backends.values():
            backend.release.set()
        _, manager, infos = await task
        assert manager is not None
        await manager.cleanup()
    assert [info.name for info in infos] == list(backends)
    assert all(backend.active == 0 for backend in backends.values())


@pytest.mark.parametrize("stateless", [False, True])
async def test_stdio_startup_under_blockbuster(tmp_path: Path, stateless: bool) -> None:
    from blockbuster import blockbuster_ctx

    script = tmp_path / "server.py"
    script.write_text(
        "import os\nfrom fastmcp import FastMCP\n"
        "server = FastMCP('stdio')\n"
        "@server.tool\nasync def pid() -> int:\n"
        "    return os.getpid()\n"
        "server.run()\n",
        encoding="utf-8",
    )
    await asyncio.to_thread(mcp_tools._warm_mcp_adapter_imports)
    with blockbuster_ctx():
        tools, manager, infos = await mcp_tools._load_tools_from_config(
            {
                "mcpServers": {
                    "stdio": {"command": sys.executable, "args": [str(script)]}
                }
            },
            stateless=stateless,
        )
        assert (manager is None) == stateless
        try:
            assert infos[0].status == "ok", infos[0].error
            result = await tools[0].ainvoke({})
            pid = int(result[0]["text"])
        finally:
            if manager is not None:
                await manager.cleanup()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("phase", ["initialize", "list_tools"])
async def test_failed_stdio_startup_terminates_process(
    tmp_path: Path, phase: str
) -> None:
    script, pid_file = tmp_path / "broken.py", tmp_path / "pid"
    script.write_text(
        "import os, sys\nfrom pathlib import Path\n"
        "from fastmcp import FastMCP\n"
        "from fastmcp.server.middleware import Middleware\n"
        "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "class Broken(Middleware):\n"
        f"    async def on_{phase}(self, context, call_next):\n"
        "        raise RuntimeError('startup refused')\n"
        "server = FastMCP('broken')\n"
        "server.add_middleware(Broken())\n"
        "server.run()\n",
        encoding="utf-8",
    )
    tools, manager, infos = await mcp_tools._load_tools_from_config(
        {
            "mcpServers": {
                "broken": {
                    "command": sys.executable,
                    "args": [str(script), str(pid_file)],
                }
            }
        }
    )
    try:
        assert tools == []
        assert infos[0].status == "error"
        pid = int(pid_file.read_text(encoding="utf-8"))
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        assert manager is not None
        await manager.cleanup()
