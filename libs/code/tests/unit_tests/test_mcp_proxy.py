"""Runtime backend failures survive the proxy and LangChain adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx2
from fastmcp import FastMCP
from fastmcp.client.transports import StreamableHttpTransport

from deepagents_code import mcp_tools
from deepagents_code.mcp_auth import MCPReauthRequiredError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    import pytest


class ExpiringAuth(httpx2.Auth):
    """Fail in the HTTP transport's background task after discovery."""

    expired = False
    failures = 0

    async def async_auth_flow(
        self, request: httpx2.Request
    ) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        if self.expired and request.method == "POST":
            self.failures += 1
            server_name = "remote"
            raise MCPReauthRequiredError(server_name)
        yield request


async def test_runtime_reauth_preserves_login_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = FastMCP("remote")

    @server.tool
    def echo() -> str:
        return "success"

    app = server.http_app(json_response=True, stateless_http=True)
    auth = ExpiringAuth()
    clients: list[httpx2.AsyncClient] = []

    def client_factory(
        headers: dict[str, str] | None = None,
        timeout: httpx2.Timeout | None = None,
        auth: httpx2.Auth | None = None,
        *,
        follow_redirects: bool = True,
    ) -> httpx2.AsyncClient:
        client = httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app),
            headers=headers,
            timeout=timeout,
            auth=auth,
            follow_redirects=follow_redirects,
        )
        clients.append(client)
        return client

    transport = StreamableHttpTransport(
        "http://test/mcp", auth=auth, httpx_client_factory=client_factory
    )

    async def preflight(*_args: object) -> None:
        pass

    monkeypatch.setattr(mcp_tools, "_check_remote_server", preflight)
    monkeypatch.setattr(mcp_tools, "_build_transport", lambda *_a, **_kw: transport)
    async with app.router.lifespan_context(app):
        tools, manager, infos = await mcp_tools._load_tools_from_config(
            {"mcpServers": {"remote": {"url": "http://test/mcp"}}}
        )
        assert manager is not None
        try:
            assert infos[0].status == "ok", infos[0].error
            assert "success" in str(await tools[0].ainvoke({}))
            auth.expired = True
            for _ in range(2):
                result = await tools[0].ainvoke(
                    {
                        "type": "tool_call",
                        "id": "auth",
                        "name": tools[0].name,
                        "args": {},
                    }
                )
                assert result.status == "error"
                assert "/mcp login remote" in str(result.content)
                assert "dcode mcp login remote" in str(result.content)
                assert "Connection closed" not in str(result.content)
            assert auth.failures == 2
            auth.expired = False
            assert "success" in str(await tools[0].ainvoke({}))
        finally:
            await manager.cleanup()
    assert clients
    assert all(client.is_closed for client in clients)
