from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import socket
import stat
import tempfile
import threading
import time
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import SecretStr

from deepagents_talon.mcp_auth import (
    DeviceAuthorizationCompletedError,
    FileTokenStorage,
    MCPAuthorizationError,
    _AuthorizationServerMetadata,
    _authorize_discovered_device,
    _DeviceCodeResponse,
    _discover_device_metadata,
    _issuer_endpoint,
    _OAuthSafeTransport,
    _present_device_code,
    _register_device_client,
    build_oauth_provider,
    extract_oauth_callback_url,
    format_login_error,
    prepare_device_client,
    prepare_oauth_login,
)
from deepagents_talon.mcp_config import WORKSPACE_ENV, locked_path

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _public_dns(
    host: str, port: int, *_args: object, **_kwargs: object
) -> list[tuple[int, int, int, str, tuple[str, int]]]:
    try:
        address = str(ipaddress.ip_address(host))
    except ValueError:
        address = "93.184.216.34"
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]


@pytest.fixture
def oauth_network(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Callable[[httpx.Request], httpx.Response]], httpx.MockTransport]:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)

    def install(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(httpx, "AsyncHTTPTransport", lambda **_kwargs: transport)
        monkeypatch.setattr(
            "deepagents_talon.mcp_auth._OAuthHTTPTransport", lambda **_kwargs: transport
        )
        monkeypatch.setattr("httpx._client.AsyncHTTPTransport", lambda **_kwargs: transport)
        return transport

    return install


async def test_file_token_storage_round_trip_and_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    storage = FileTokenStorage("notion", server_url="https://example.com/mcp")
    tokens = OAuthToken(
        access_token="secret",  # noqa: S106
        refresh_token="refresh",  # noqa: S106
    )
    client = OAuthClientInformationFull(
        redirect_uris=["http://localhost:3000/callback"],
        client_id="client-id",
    )

    await storage.set_tokens(tokens)
    await storage.set_client_info(client)

    assert await storage.get_tokens() == tokens
    assert await storage.get_client_info() == client
    assert stat.S_IMODE(storage.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(storage.path.parent.stat().st_mode) == 0o700


async def test_file_token_storage_persists_absolute_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    monkeypatch.setattr("deepagents_talon.mcp_auth.time.time", lambda: 1_000.0)
    storage = FileTokenStorage("notion", server_url="https://example.com/mcp")

    await storage.set_tokens(
        OAuthToken(
            access_token="secret",  # noqa: S106
            refresh_token="refresh",  # noqa: S106
            expires_in=60,
        )
    )

    assert await storage.get_token_expiry() == 1_060.0


async def test_provider_restores_persisted_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    monkeypatch.setattr("deepagents_talon.mcp_auth.time.time", lambda: 1_000.0)
    storage = FileTokenStorage("notion", server_url="https://example.com/mcp")
    await storage.set_tokens(
        OAuthToken(
            access_token="secret",  # noqa: S106
            refresh_token="refresh",  # noqa: S106
            expires_in=60,
        )
    )
    await storage.set_client_info(
        OAuthClientInformationFull(
            redirect_uris=["http://localhost:3000/callback"],
            client_id="client-id",
        )
    )
    provider = build_oauth_provider(
        server_name="notion",
        server_url="https://example.com/mcp",
        storage=storage,
        interactive=True,
    )

    await provider._initialize()

    assert provider.context.token_expiry_time == 1_060.0
    assert provider.context.can_refresh_token()


async def test_provider_refreshes_expired_token_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, oauth_network
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    storage = FileTokenStorage("notion", server_url="https://example.com/mcp")
    await storage.set_tokens(
        OAuthToken(
            access_token="expired",  # noqa: S106
            refresh_token="refresh",  # noqa: S106
            expires_in=-1,
        )
    )
    await storage.set_client_info(
        OAuthClientInformationFull(
            redirect_uris=["http://localhost:3000/callback"],
            client_id="client-id",
        )
    )
    provider = build_oauth_provider(
        server_name="notion",
        server_url="https://example.com/mcp",
        storage=storage,
        interactive=True,
    )
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/mcp":
            if request.headers.get("Authorization") == "Bearer renewed":
                return httpx.Response(200)
            return httpx.Response(
                401,
                headers={
                    "WWW-Authenticate": 'Bearer resource_metadata="https://example.com/resource"'
                },
            )
        if request.url.path == "/resource":
            return httpx.Response(
                200,
                json={
                    "resource": "https://example.com/mcp",
                    "authorization_servers": ["https://auth.example/tenant"],
                },
            )
        if request.url.path == "/.well-known/oauth-authorization-server/tenant":
            return httpx.Response(200, json=_oauth_metadata())
        if request.url.path == "/tenant/access-token":
            assert request.headers["Host"] == "auth.example"
            assert parse_qs(request.content.decode())["refresh_token"] == ["refresh"]
            return httpx.Response(
                200,
                json={
                    "access_token": "renewed",
                    "refresh_token": "next-refresh",
                    "expires_in": 3600,
                },
            )
        return httpx.Response(404)

    transport = oauth_network(handle)
    async with httpx.AsyncClient(transport=transport, auth=provider) as client:
        response = await client.get("https://example.com/mcp")

    assert response.status_code == 200
    assert [request.url.path for request in requests if request.method == "POST"] == [
        "/tenant/access-token"
    ]
    assert all("Authorization" not in request.headers for request in requests[:-1])
    assert requests[-1].headers["Authorization"] == "Bearer renewed"
    tokens = await storage.get_tokens()
    assert tokens is not None
    assert tokens.access_token == "renewed"  # noqa: S105


async def test_file_token_storage_accepts_legacy_file_without_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    storage = FileTokenStorage("notion", server_url="https://example.com/mcp")
    storage.path.parent.mkdir(parents=True)
    storage.path.write_text(
        json.dumps({"tokens": {"access_token": "secret"}}),
        encoding="utf-8",
    )

    assert await storage.get_token_expiry() is None


async def test_file_token_storage_rejects_invalid_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    storage = FileTokenStorage("notion", server_url="https://example.com/mcp")
    storage.path.parent.mkdir(parents=True)
    storage.path.write_text(json.dumps({"expires_at": "tomorrow"}), encoding="utf-8")

    with pytest.raises(TypeError, match="Invalid MCP credential file"):
        await storage.get_token_expiry()


async def test_file_token_storage_writes_tokens_and_client_info_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    storage = FileTokenStorage("github", server_url="https://api.githubcopilot.com/mcp")
    tokens = OAuthToken(access_token="secret")  # noqa: S106
    client = OAuthClientInformationFull(
        redirect_uris=["http://localhost/callback"],
        client_id="client-id",
    )

    await storage.set_tokens_and_client_info(tokens, client)

    assert await storage.get_tokens() == tokens
    assert await storage.get_client_info() == client
    assert set(json.loads(storage.path.read_text(encoding="utf-8"))) == {
        "tokens",
        "expires_at",
        "client_info",
    }


async def test_forced_authorization_preserves_stored_tokens_until_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    stored = OAuthToken(access_token="stored")  # noqa: S106
    replacement = OAuthToken(access_token="replacement")  # noqa: S106
    storage = FileTokenStorage("notion", server_url="https://example.com/mcp")
    await storage.set_tokens(stored)

    forced = FileTokenStorage(
        "notion",
        server_url="https://example.com/mcp",
        force_authorization=True,
    )

    assert await forced.get_tokens() is None
    assert await forced.get_tokens() == stored

    await forced.set_tokens(replacement)

    assert await storage.get_tokens() == replacement


def test_file_token_storage_binds_path_to_server_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)

    first = FileTokenStorage("remote", server_url="https://one.example/mcp")
    second = FileTokenStorage("remote", server_url="https://two.example/mcp")

    assert first.path != second.path
    assert first.path.parent == second.path.parent


async def test_interactive_provider_validates_callback_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FileTokenStorage("remote", server_url="https://example.com/mcp")
    provider = build_oauth_provider(
        server_name="remote",
        server_url="https://example.com/mcp",
        storage=storage,
        interactive=True,
    )
    monkeypatch.setattr(
        "builtins.input", lambda _prompt: "http://localhost:3000/callback?code=abc&state=state"
    )

    code, state = await provider.context.callback_handler()

    assert (code, state) == ("abc", "state")


def test_slack_provider_selection_requires_slack_hostname() -> None:
    storage = FileTokenStorage("slack", server_url="https://slack.com/mcp")
    slack = build_oauth_provider(
        server_name="slack",
        server_url="https://slack.com/mcp",
        storage=storage,
        interactive=True,
    )
    lookalike = build_oauth_provider(
        server_name="lookalike",
        server_url="https://slack.com.attacker.example/mcp",
        storage=storage,
        interactive=True,
    )

    assert [str(uri) for uri in slack.context.client_metadata.redirect_uris or []] == [
        "http://localhost:3118/callback"
    ]
    assert [str(uri) for uri in lookalike.context.client_metadata.redirect_uris or []] == [
        "http://localhost:3000/callback"
    ]


async def test_slack_login_preseeds_public_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    storage = FileTokenStorage("slack", server_url="https://slack.com/mcp")

    await prepare_oauth_login(server_url="https://slack.com/mcp", storage=storage)

    client = await storage.get_client_info()
    assert client is not None
    assert client.client_id == "4518649543379.10944517634130"
    assert [str(uri) for uri in client.redirect_uris or []] == ["http://localhost:3118/callback"]


async def test_slack_provider_validates_registered_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FileTokenStorage("slack", server_url="https://slack.com/mcp")
    provider = build_oauth_provider(
        server_name="slack",
        server_url="https://slack.com/mcp",
        storage=storage,
        interactive=True,
    )
    callback = provider.context.callback_handler
    assert callback is not None
    callbacks = iter(
        [
            "http://localhost:3118/callback?code=abc&state=state",
            "http://localhost:3000/callback?code=abc&state=state",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(callbacks))

    assert await callback() == ("abc", "state")
    with pytest.raises(MCPAuthorizationError, match="callback is invalid"):
        await callback()


@pytest.mark.parametrize("port", [3000, 3118])
def test_extract_oauth_callback_url_accepts_registered_ports(port: int) -> None:
    callback = f"http://localhost:{port}/callback?code=secret&state=opaque"

    assert extract_oauth_callback_url(f"<{callback}>") == callback


@pytest.mark.parametrize(
    "callback",
    [
        "http://localhost:3119/callback?code=secret&state=opaque",
        "http://localhost:3118/other?code=secret&state=opaque",
        "http://attacker.example/callback?code=secret&state=opaque",
        "http://localhost:3118/callback?code=secret&state=opaque\nextra",
    ],
)
def test_extract_oauth_callback_url_rejects_unregistered_or_unsafe_urls(
    callback: str,
) -> None:
    assert extract_oauth_callback_url(callback) is None


async def test_malformed_credential_file_does_not_expose_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    storage = FileTokenStorage("remote", server_url="https://example.com/mcp")
    storage.path.parent.mkdir(parents=True)
    storage.path.write_text(json.dumps(["secret"]), encoding="utf-8")

    with pytest.raises(TypeError, match="Invalid MCP credential file") as caught:
        await storage.get_tokens()

    assert "secret" not in str(caught.value)


async def test_provider_switches_to_discovered_device_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    oauth_network: Callable[[Callable[[httpx.Request], httpx.Response]], httpx.MockTransport],
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    grant = "urn:ietf:params:oauth:grant-type:device_code"
    polls = iter(
        [
            {"error": "authorization_pending"},
            {"error": "slow_down"},
            {"access_token": "access-secret", "token_type": "bearer"},
        ]
    )
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "issuer": "https://auth.example",
                    "token_endpoint": "https://auth.example/token",
                    "registration_endpoint": "https://auth.example/register",
                    "grant_types_supported": [grant],
                    "device_authorization_endpoint": "https://auth.example/device",
                },
            )
        if request.url.path == "/register":
            return httpx.Response(
                201,
                json={
                    "client_id": "dynamic-client",
                    "redirect_uris": ["http://localhost:3000/callback"],
                    "grant_types": [grant],
                    "token_endpoint_auth_method": "none",
                },
            )
        if request.url.path == "/device":
            return httpx.Response(
                200,
                json={
                    "device_code": "device-secret",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://auth.example/activate",
                    "expires_in": 120,
                    "interval": 1,
                },
            )
        body = next(polls)
        return httpx.Response(
            400 if body.get("error") == "authorization_pending" else 200, json=body
        )

    oauth_network(handle)
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("deepagents_talon.mcp_auth.asyncio.sleep", sleep)
    storage = FileTokenStorage("remote", server_url="https://mcp.example/mcp")
    provider = build_oauth_provider(
        server_name="remote",
        server_url="https://mcp.example/mcp",
        storage=storage,
        interactive=True,
    )
    provider.context.auth_server_url = "https://auth.example"
    provider.context.client_info = OAuthClientInformationFull(
        client_id="authorization-code-client",
        redirect_uris=["http://localhost:3000/callback"],
        grant_types=["authorization_code"],
        token_endpoint_auth_method="none",  # noqa: S106
    )
    redirect = provider.context.redirect_handler
    assert redirect is not None

    with pytest.raises(DeviceAuthorizationCompletedError):
        await redirect("https://auth.example/authorize")

    tokens = await storage.get_tokens()
    client_info = await storage.get_client_info()
    assert tokens is not None
    assert client_info is not None
    assert tokens.access_token == "access-secret"  # noqa: S105
    assert client_info.client_id == "dynamic-client"
    assert sleeps == [1, 1, 6]
    assert [request.url.path for request in requests][-5:] == [
        "/register",
        "/device",
        "/token",
        "/token",
        "/token",
    ]
    output = capsys.readouterr().out
    assert "https://auth.example/activate" in output
    assert "ABCD-1234" in output
    assert "device-secret" not in output


@pytest.mark.parametrize(
    "server_url",
    [
        "https://api.githubcopilot.com.evil.example/mcp",
        "https://githubcopilot.com/mcp",
        "http://api.githubcopilot.com/mcp",
        "https://api.githubcopilot.com:8443/mcp",
    ],
)
async def test_github_public_client_requires_exact_mcp_host(
    server_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    storage = FileTokenStorage("remote", server_url=server_url)

    await prepare_device_client(server_url, storage)

    assert await storage.get_client_info() is None


async def test_github_endpoint_seeds_public_device_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    server_url = "https://api.githubcopilot.com/mcp/"
    storage = FileTokenStorage("github", server_url=server_url)

    await prepare_device_client(server_url, storage)

    client_info = await storage.get_client_info()
    assert client_info is not None
    assert client_info.client_id == "Iv23libxz8qOApH0WQL3"


async def test_provider_keeps_authorization_code_when_device_grant_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    oauth_network: Callable[[Callable[[httpx.Request], httpx.Response]], httpx.MockTransport],
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "issuer": "https://auth.example",
                "token_endpoint": "https://auth.example/token",
                "grant_types_supported": ["authorization_code"],
            },
        )

    oauth_network(handle)
    storage = FileTokenStorage("remote", server_url="https://mcp.example/mcp")
    provider = build_oauth_provider(
        server_name="remote",
        server_url="https://mcp.example/mcp",
        storage=storage,
        interactive=True,
    )
    provider.context.auth_server_url = "https://auth.example"
    provider.context.client_info = OAuthClientInformationFull(
        client_id="client-id",
        redirect_uris=["http://localhost:3000/callback"],
    )
    redirect = provider.context.redirect_handler
    assert redirect is not None

    await redirect("https://auth.example/authorize")

    assert "https://auth.example/authorize" in capsys.readouterr().out


def test_device_response_credentials_are_redacted() -> None:
    response = _DeviceCodeResponse(
        device_code=SecretStr("device-secret"),
        user_code=SecretStr("user-secret"),
        verification_uri="https://github.com/login/device",
        expires_in=120,
    )

    assert "device-secret" not in repr(response)
    assert "user-secret" not in repr(response)


async def test_device_endpoint_must_match_authorization_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "deepagents_talon.mcp_auth.validate_safe_url",
        lambda url, **_kwargs: url,
    )
    metadata = _AuthorizationServerMetadata(
        issuer="https://auth.example",
        token_endpoint="https://evil.example/token",  # noqa: S106  # URL, not credential.
        grant_types_supported=["urn:ietf:params:oauth:grant-type:device_code"],
        device_authorization_endpoint="https://auth.example/device",
    )

    with pytest.raises(MCPAuthorizationError, match="does not match"):
        await _issuer_endpoint(metadata.token_endpoint, metadata)


def _oauth_metadata(**overrides: object) -> dict[str, object]:
    return {
        "issuer": "https://auth.example/tenant",
        "authorization_endpoint": "https://auth.example/authorize",
        "token_endpoint": "https://auth.example/tenant/access-token",
        "registration_endpoint": "https://auth.example/register",
        **overrides,
    }


@pytest.fixture
async def oauth_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, oauth_network):
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    storage = FileTokenStorage("remote", server_url="https://example.com/mcp")
    provider = build_oauth_provider(
        server_name="remote",
        server_url="https://example.com/mcp",
        storage=storage,
        interactive=True,
    )
    states: list[str] = []
    requests: list[httpx.Request] = []
    responses = {
        "/mcp": httpx.Response(
            401,
            headers={"WWW-Authenticate": 'Bearer resource_metadata="https://example.com/resource"'},
        ),
        "/resource": httpx.Response(
            200,
            json={
                "resource": "https://example.com/mcp",
                "authorization_servers": ["https://auth.example/tenant"],
            },
        ),
        "/.well-known/oauth-authorization-server/tenant": httpx.Response(
            200, json=_oauth_metadata()
        ),
        "/register": httpx.Response(
            201,
            json={"client_id": "client-id", "redirect_uris": ["http://localhost:3000/callback"]},
        ),
        "/tenant/access-token": httpx.Response(
            200, json={"access_token": "renewed", "expires_in": 3600}
        ),
    }

    async def redirect(url: str) -> None:
        states.append(parse_qs(urlparse(url).query)["state"][0])

    async def callback() -> tuple[str, str]:
        return "authorization-code", states[-1]

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.headers.get("Authorization") == "Bearer renewed":
            return httpx.Response(200)
        return responses.get(request.url.path, httpx.Response(404))

    provider.context.redirect_handler = redirect
    provider.context.callback_handler = callback
    async with httpx.AsyncClient(
        transport=oauth_network(handle), auth=provider, follow_redirects=True
    ) as client:
        yield client, requests, responses, storage


async def test_authorization_code_flow_uses_guarded_discovery(oauth_client) -> None:
    client, requests, _, storage = oauth_client
    assert (await client.get("https://example.com/mcp")).status_code == 200
    exchange = next(request for request in requests if request.url.path == "/tenant/access-token")
    assert exchange.headers["Host"] == "auth.example"
    assert exchange.url.host == "93.184.216.34"
    assert exchange.extensions["sni_hostname"] == b"auth.example"
    assert parse_qs(exchange.content.decode())["code"] == ["authorization-code"]
    assert "code_verifier" in parse_qs(exchange.content.decode())
    assert (await storage.get_tokens()).access_token == "renewed"  # noqa: S105 - Mock token.


@pytest.mark.parametrize("resource", ["https://example.com/mcp", "http://127.0.0.1:8080/mcp"])
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/private",
        "https://127.0.0.1/private",
        "https://[::1]/private",
        "https://169.254.169.254/latest/meta-data",
        "https://10.0.0.1/private",
        "https://localhost/private",
        "http://public.example/resource",
        "https://user:password@public.example/resource",
    ],
)
async def test_oauth_rejects_unsafe_resource_discovery(
    oauth_client, url: str, resource: str
) -> None:
    client, requests, responses, _ = oauth_client
    responses["/mcp"] = httpx.Response(
        401, headers={"WWW-Authenticate": f'Bearer resource_metadata="{url}"'}
    )
    with pytest.raises(MCPAuthorizationError, match="safe public HTTPS"):
        await client.get(resource)
    assert [str(request.url) for request in requests] == [resource]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("authorization_endpoint", "https://127.0.0.1/private", "safe public HTTPS"),
        ("token_endpoint", "https://127.0.0.1/private", "safe public HTTPS"),
        ("registration_endpoint", "https://127.0.0.1/private", "safe public HTTPS"),
        ("issuer", "https://other.example", "issuer does not match"),
        ("issuer", "https://auth.example/other-tenant", "issuer does not match"),
    ],
)
async def test_oauth_rejects_unsafe_metadata(
    oauth_client, field: str, value: str, error: str
) -> None:
    client, requests, responses, _ = oauth_client
    responses["/.well-known/oauth-authorization-server/tenant"] = httpx.Response(
        200, json=_oauth_metadata(**{field: value})
    )
    with pytest.raises(MCPAuthorizationError, match=error):
        await client.get("https://example.com/mcp")
    assert all(request.method == "GET" for request in requests)


@pytest.mark.parametrize(
    "path",
    [
        "/resource",
        "/.well-known/oauth-authorization-server/tenant",
        "/register",
        "/tenant/access-token",
    ],
)
async def test_oauth_never_follows_discovered_redirects(oauth_client, path: str) -> None:
    client, requests, responses, storage = oauth_client
    responses[path] = httpx.Response(307, headers={"Location": "https://127.0.0.1/private"})
    with pytest.raises(MCPAuthorizationError, match="redirects are not allowed"):
        await client.get("https://example.com/mcp")
    assert all(request.url.path != "/private" for request in requests)
    assert await storage.get_tokens() is None


async def test_refresh_without_metadata_does_not_send_credentials(oauth_client) -> None:
    client, requests, responses, storage = oauth_client
    await storage.set_tokens(
        OAuthToken.model_validate(
            {"access_token": "expired", "refresh_token": "refresh", "expires_in": -1}
        )
    )
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="client-id", redirect_uris=["http://localhost:3000/callback"]
        )
    )
    responses.clear()
    with pytest.raises(MCPAuthorizationError, match="Could not discover"):
        await client.get("https://example.com/mcp")
    assert all(
        request.method == "GET" and "Authorization" not in request.headers for request in requests
    )
    assert (await storage.get_tokens()).refresh_token == "refresh"  # noqa: S105 - Mock token.


@pytest.mark.parametrize("rebind", [False, True])
async def test_oauth_blocks_private_dns_before_connecting(
    oauth_client, monkeypatch: pytest.MonkeyPatch, *, rebind: bool
) -> None:
    client, requests, _, _ = oauth_client
    lookups = 0

    def resolve(
        _host: str, port: int, *_args: object, **_kwargs: object
    ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        nonlocal lookups
        lookups += 1
        return _public_dns("93.184.216.34" if rebind and lookups == 1 else "127.0.0.1", port)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    with pytest.raises(MCPAuthorizationError, match="safe public HTTPS"):
        await client.get("https://example.com/mcp")
    assert [request.url.path for request in requests] == ["/mcp"]


async def test_device_flow_uses_only_pinned_clients_that_ignore_ambient_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    oauth_network: Callable[[Callable[[httpx.Request], httpx.Response]], httpx.MockTransport],
) -> None:
    """Discovery, registration, the device code and polling all carry credentials."""
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "attacker-ca.pem"))
    grant = "urn:ietf:params:oauth:grant-type:device_code"

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "issuer": "https://auth.example",
                    "token_endpoint": "https://auth.example/token",
                    "registration_endpoint": "https://auth.example/register",
                    "grant_types_supported": [grant],
                    "device_authorization_endpoint": "https://auth.example/device",
                },
            )
        if request.url.path == "/register":
            return httpx.Response(
                201,
                json={
                    "client_id": "dynamic-client",
                    "redirect_uris": ["http://localhost:3000/callback"],
                    "grant_types": [grant],
                    "token_endpoint_auth_method": "none",
                },
            )
        if request.url.path == "/device":
            return httpx.Response(
                200,
                json={
                    "device_code": "device-secret",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://auth.example/activate",
                    "expires_in": 120,
                    "interval": 1,
                },
            )
        return httpx.Response(200, json={"access_token": "access-secret", "token_type": "bearer"})

    oauth_network(handle)
    clients: list[dict[str, object]] = []
    async_client = httpx.AsyncClient

    def record(**kwargs: object) -> httpx.AsyncClient:
        clients.append(kwargs)
        return async_client(**kwargs)

    monkeypatch.setattr("deepagents_talon.mcp_auth.httpx.AsyncClient", record)

    async def sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("deepagents_talon.mcp_auth.asyncio.sleep", sleep)
    storage = FileTokenStorage("remote", server_url="https://mcp.example/mcp")
    provider = build_oauth_provider(
        server_name="remote",
        server_url="https://mcp.example/mcp",
        storage=storage,
        interactive=True,
    )
    provider.context.auth_server_url = "https://auth.example"
    provider.context.client_info = OAuthClientInformationFull(
        client_id="authorization-code-client",
        redirect_uris=["http://localhost:3000/callback"],
        grant_types=["authorization_code"],
        token_endpoint_auth_method="none",  # noqa: S106
    )

    authorized = await _authorize_discovered_device(
        "remote", storage, provider.context, interactive=True
    )

    assert authorized is True
    tokens = await storage.get_tokens()
    assert tokens is not None
    assert tokens.access_token == "access-secret"  # noqa: S105
    assert len(clients) == 3
    assert all(isinstance(kwargs["transport"], _OAuthSafeTransport) for kwargs in clients)
    assert all(kwargs["trust_env"] is False for kwargs in clients)
    assert "device-secret" not in capsys.readouterr().out


async def test_device_metadata_discovery_bounds_slow_name_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 10s discovery bound is unenforceable while getaddrinfo blocks the loop."""
    monkeypatch.setattr("deepagents_talon.mcp_auth._DISCOVERY_TIMEOUT_SECONDS", 0.05)
    released = threading.Event()

    def slow_resolve(url: str, **_kwargs: object) -> str:
        released.wait(5.0)
        return url

    monkeypatch.setattr("deepagents_talon.mcp_auth.validate_safe_url", slow_resolve)
    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.005)
            ticks += 1

    ticker = asyncio.create_task(tick())
    started = time.monotonic()
    try:
        assert await _discover_device_metadata("https://auth.example") is None
    finally:
        released.set()
        ticker.cancel()
    assert time.monotonic() - started < 1.0
    assert ticks > 0


def test_token_storage_warns_about_the_agent_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Placement is reported, not enforced, until the deny rules make it real."""
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path / "home")

    with caplog.at_level(logging.WARNING, logger="deepagents_talon.mcp_config"):
        storage = FileTokenStorage(
            "remote",
            server_url="https://example.com/mcp",
            agent_root=tmp_path,
        )

    assert storage.path.parent == tmp_path / "home" / ".deepagents" / "mcp-tokens"
    assert "MCP credential storage" in caplog.text
    assert str(tmp_path) in caplog.text
    assert WORKSPACE_ENV in caplog.text


async def test_token_storage_works_when_talon_runs_from_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression guard: the default paths sit under CWD when launched from $HOME."""
    monkeypatch.delenv(WORKSPACE_ENV, raising=False)
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    storage = FileTokenStorage("notion", server_url="https://example.com/mcp")

    await storage.set_tokens(OAuthToken(access_token="secret"))  # noqa: S106

    stored = await storage.get_tokens()
    assert stored is not None
    assert stored.access_token == "secret"  # noqa: S105


async def test_token_storage_tightens_intermediate_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mkdir(mode=...) sets the mode on the leaf only, leaving ~/.deepagents open."""
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    state = tmp_path / ".deepagents"
    state.mkdir()
    state.chmod(0o755)
    storage = FileTokenStorage("notion", server_url="https://example.com/mcp")

    await storage.set_tokens(OAuthToken(access_token="secret"))  # noqa: S106

    assert stat.S_IMODE(state.stat().st_mode) == 0o700
    assert stat.S_IMODE(storage.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(storage.path.stat().st_mode) == 0o600


async def test_refresh_without_a_refresh_token_keeps_the_stored_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RFC 6749 section 6 lets a refresh response omit refresh_token."""
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    storage = FileTokenStorage("notion", server_url="https://example.com/mcp")
    await storage.set_tokens(
        OAuthToken(access_token="first", refresh_token="long-lived")  # noqa: S106
    )

    await storage.set_tokens(OAuthToken(access_token="second"))  # noqa: S106

    stored = await storage.get_tokens()
    assert stored is not None
    assert stored.access_token == "second"  # noqa: S105
    assert stored.refresh_token == "long-lived"  # noqa: S105


async def test_refresh_with_a_new_refresh_token_replaces_the_stored_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    storage = FileTokenStorage("notion", server_url="https://example.com/mcp")
    await storage.set_tokens(
        OAuthToken(access_token="first", refresh_token="rotated-away")  # noqa: S106
    )

    await storage.set_tokens(
        OAuthToken(access_token="second", refresh_token="rotated-in")  # noqa: S106
    )

    stored = await storage.get_tokens()
    assert stored is not None
    assert stored.refresh_token == "rotated-in"  # noqa: S105


async def test_failed_token_write_closes_its_descriptor_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second close on a worker thread can land on an unrelated reused fd."""
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    storage = FileTokenStorage("notion", server_url="https://example.com/mcp")
    descriptors: list[int] = []
    mkstemp = tempfile.mkstemp

    def record_mkstemp(**kwargs: object) -> tuple[int, str]:
        descriptor, name = mkstemp(**kwargs)
        descriptors.append(descriptor)
        return descriptor, name

    closed: list[int] = []
    real_close = os.close

    def record_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    def explode(*_args: object, **_kwargs: object) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr("deepagents_talon.mcp_auth.tempfile.mkstemp", record_mkstemp)
    monkeypatch.setattr("deepagents_talon.mcp_auth.os.close", record_close)
    monkeypatch.setattr("deepagents_talon.mcp_auth.json.dump", explode)

    with pytest.raises(OSError, match="disk full"):
        await storage.set_tokens(OAuthToken(access_token="secret"))  # noqa: S106

    assert descriptors
    # The stream owns the descriptor once fdopen succeeds and closes it on exit.
    assert [descriptor for descriptor in closed if descriptor in descriptors] == []
    assert list(storage.path.parent.glob(".tokens-*")) == []


async def test_token_write_is_flushed_and_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent refreshes must not interleave a read-modify-write."""
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    monkeypatch.setattr("deepagents_talon.mcp_config._LOCK_TIMEOUT_SECONDS", 0.1)
    synced: list[int] = []
    real_fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        synced.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr("deepagents_talon.mcp_auth.os.fsync", record_fsync)
    storage = FileTokenStorage("notion", server_url="https://example.com/mcp")
    await storage.set_tokens(OAuthToken(access_token="first"))  # noqa: S106
    assert synced

    holding = threading.Event()
    release = threading.Event()

    def hold_the_lock() -> None:
        with locked_path(storage.path):
            holding.set()
            release.wait(5.0)

    holder = threading.Thread(target=hold_the_lock)
    holder.start()
    try:
        assert holding.wait(5.0)
        with pytest.raises(TimeoutError):
            await storage.set_tokens(OAuthToken(access_token="second"))  # noqa: S106
    finally:
        release.set()
        holder.join(5.0)

    stored = await storage.get_tokens()
    assert stored is not None
    assert stored.access_token == "first"  # noqa: S105


def test_format_login_error_survives_an_empty_message() -> None:
    """Both call sites are error handlers; str(OSError()) is empty."""
    assert format_login_error(OSError()) == "OSError"
    assert format_login_error(ValueError("first line\nsecond")) == "first line"


async def test_authorization_without_a_conversation_names_the_remedy() -> None:
    """Scheduled jobs and background subagents run without an authorization handler."""
    device = _DeviceCodeResponse(
        device_code=SecretStr("device-secret"),
        user_code=SecretStr("ABCD-1234"),
        verification_uri="https://auth.example/activate",
        expires_in=120,
    )

    with pytest.raises(MCPAuthorizationError) as caught:
        await _present_device_code(
            "notion",
            device,
            deadline=0.0,
            interactive=False,
        )

    message = str(caught.value)
    assert "scheduled job or a background subagent" in message
    assert "deepagents-talon mcp login notion" in message
    assert "device-secret" not in message


async def test_fresh_grant_without_a_refresh_token_clears_the_stored_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new grant is not a refresh response: the old refresh token may be revoked."""
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    storage = FileTokenStorage("notion", server_url="https://example.com/mcp")
    await storage.set_tokens(
        OAuthToken(access_token="first", refresh_token="from-old-grant")  # noqa: S106
    )
    client = OAuthClientInformationFull(
        redirect_uris=["http://localhost:3000/callback"],
        client_id="client-id",
    )

    await storage.set_tokens_and_client_info(OAuthToken(access_token="fresh"), client)  # noqa: S106

    stored = await storage.get_tokens()
    assert stored is not None
    assert stored.access_token == "fresh"  # noqa: S105
    assert stored.refresh_token is None


async def test_device_registration_bounds_slow_name_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """to_thread keeps the loop free, but nothing else bounded this resolution."""
    monkeypatch.setattr("deepagents_talon.mcp_auth._RESOLVE_TIMEOUT_SECONDS", 0.05)
    released = threading.Event()

    def slow_resolve(url: str, **_kwargs: object) -> str:
        released.wait(5.0)
        return url

    monkeypatch.setattr("deepagents_talon.mcp_auth.validate_safe_url", slow_resolve)
    metadata = _AuthorizationServerMetadata(
        issuer="https://auth.example",
        token_endpoint="https://auth.example/token",  # noqa: S106
        registration_endpoint="https://auth.example/register",
        grant_types_supported=["urn:ietf:params:oauth:grant-type:device_code"],
        device_authorization_endpoint="https://auth.example/device",
    )
    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.005)
            ticks += 1

    ticker = asyncio.create_task(tick())
    started = time.monotonic()
    try:
        with pytest.raises(MCPAuthorizationError, match="Timed out resolving"):
            await _register_device_client(metadata)
    finally:
        released.set()
        ticker.cancel()
    assert time.monotonic() - started < 1.0
    assert ticks > 0
