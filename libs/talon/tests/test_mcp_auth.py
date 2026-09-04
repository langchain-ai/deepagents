from __future__ import annotations

import json
import stat
from typing import TYPE_CHECKING

import httpx
import pytest
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import SecretStr

from deepagents_talon.mcp_auth import (
    DeviceAuthorizationCompletedError,
    FileTokenStorage,
    MCPAuthorizationError,
    _AuthorizationServerMetadata,
    _DeviceCodeResponse,
    _issuer_endpoint,
    build_oauth_provider,
    extract_oauth_callback_url,
    prepare_device_client,
    prepare_oauth_login,
)

if TYPE_CHECKING:
    from pathlib import Path


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

    transport = httpx.MockTransport(handle)
    async_client = httpx.AsyncClient
    monkeypatch.setattr(
        "deepagents_talon.mcp_auth.httpx.AsyncClient",
        lambda **kwargs: async_client(transport=transport, **kwargs),
    )
    monkeypatch.setattr(
        "deepagents_talon.mcp_auth.validate_safe_url",
        lambda url, **_kwargs: url,
    )
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
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    monkeypatch.setattr(
        "deepagents_talon.mcp_auth.validate_safe_url",
        lambda url, **_kwargs: url,
    )

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "issuer": "https://auth.example",
                "token_endpoint": "https://auth.example/token",
                "grant_types_supported": ["authorization_code"],
            },
        )

    transport = httpx.MockTransport(handle)
    async_client = httpx.AsyncClient
    monkeypatch.setattr(
        "deepagents_talon.mcp_auth.httpx.AsyncClient",
        lambda **kwargs: async_client(transport=transport, **kwargs),
    )
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


def test_device_endpoint_must_match_authorization_server(
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
        _issuer_endpoint(metadata.token_endpoint, metadata)
