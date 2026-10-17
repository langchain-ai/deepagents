from __future__ import annotations

import socket
import ssl

import httpcore
import httpx2
import pytest
from anyio._core._sockets import idna2008_resolve
from httpcore._backends.auto import AutoBackend
from langchain_core._security import SSRFBlockedError
from mcp.shared.auth import OAuthClientMetadata

from deepagents_talon._version import __version__
from deepagents_talon.mcp_auth import (
    FileTokenStorage,
    _oauth_http_client,
    _PersistedExpiryOAuthProvider,
)


@pytest.mark.parametrize("hostname", ["mcp.slack.com", "auth.slack.com"])
@pytest.mark.parametrize("user_agent", [None, "", "custom-talon/1.0"])
@pytest.mark.parametrize("sdk_request", [False, True])
async def test_oauth_tls_preserves_hostname_and_pinned_address(
    monkeypatch, tmp_path, hostname, user_agent, sdk_request
):
    if user_agent is None:
        monkeypatch.delenv("DEEPAGENTS_TALON_MCP_USER_AGENT", raising=False)
    else:
        monkeypatch.setenv("DEEPAGENTS_TALON_MCP_USER_AGENT", user_agent)
    connected = []
    written = []
    tls_names = []
    closed = []

    class Stream(httpcore.AsyncNetworkStream):
        async def read(self, _max_bytes: int, **_kwargs: object) -> bytes:
            return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}"

        async def write(self, buffer: bytes, **_kwargs: object) -> None:
            written.append(buffer)

        async def aclose(self) -> None:
            closed.append(True)

        async def start_tls(
            self, ssl_context: ssl.SSLContext, server_hostname: str, **_kwargs: object
        ) -> Stream:
            tls_names.append(idna2008_resolve(server_hostname))
            assert ssl_context.check_hostname
            assert ssl_context.verify_mode == ssl.CERT_REQUIRED
            return self

    async def connect_tcp(_self, host: str, port: int, **_kwargs: object) -> Stream:
        connected.append((host, port))
        return Stream()

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    monkeypatch.setattr(AutoBackend, "connect_tcp", connect_tcp)

    url = f"https://{hostname}/metadata"
    if sdk_request:
        provider = _PersistedExpiryOAuthProvider(
            server_url=url,
            client_metadata=OAuthClientMetadata(redirect_uris=["http://localhost:3000/callback"]),
            storage=FileTokenStorage("test", server_url=url, agent_root=tmp_path),
        )
        response = await provider._send_oauth_request(httpx2.Request("GET", url))
    else:
        async with _oauth_http_client() as client:
            response = await client.get(url)

    expected_user_agent = user_agent or f"deepagents-talon/{__version__}"
    assert f"User-Agent: {expected_user_agent}\r\n".encode() in b"".join(written)
    assert response.json() == {}
    assert connected == [("93.184.216.34", 443)]
    assert tls_names == [hostname.encode("ascii")]
    assert f"Host: {hostname}\r\n".encode() in b"".join(written)
    assert closed


async def test_oauth_transport_rejects_mixed_public_private_dns(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))
            for address in ("93.184.216.34", "127.0.0.1")
        ],
    )
    async with _oauth_http_client() as client:
        with pytest.raises(SSRFBlockedError):
            await client.get("https://mcp.slack.com/metadata")
