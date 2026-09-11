from __future__ import annotations

import socket
import ssl

import httpcore
import pytest
from anyio._core._sockets import idna2008_resolve
from httpcore._backends.auto import AutoBackend
from langchain_core._security import SSRFBlockedError

from deepagents_talon.mcp_auth import _oauth_http_client


@pytest.mark.parametrize("hostname", ["mcp.slack.com", "auth.slack.com"])
async def test_oauth_tls_preserves_hostname_and_pinned_address(monkeypatch, hostname):
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

    async with _oauth_http_client() as client:
        response = await client.get(f"https://{hostname}/metadata")

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
