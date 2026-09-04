from __future__ import annotations

import json
import stat
from typing import TYPE_CHECKING

import httpx
import pytest
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import SecretStr

from deepagents_talon.mcp_auth import (
    FileTokenStorage,
    _DeviceCodeResponse,
    _run_github_device_flow,
    build_oauth_provider,
    is_github_mcp_url,
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


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.githubcopilot.com/mcp", True),
        ("https://API.GITHUBCOPILOT.COM/mcp", True),
        ("https://api.githubcopilot.com.evil.example/mcp", False),
        ("https://githubcopilot.com/mcp", False),
    ],
)
def test_github_mcp_url_matching_is_host_exact(url: str, expected: object) -> None:
    assert is_github_mcp_url(url) is expected


async def test_github_device_flow_polls_pending_and_slow_down(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses = iter(
        [
            httpx.Response(
                200,
                json={
                    "device_code": "device-secret",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://github.com/login/device",
                    "expires_in": 120,
                    "interval": 1,
                },
            ),
            httpx.Response(400, json={"error": "authorization_pending"}),
            httpx.Response(200, json={"error": "slow_down"}),
            httpx.Response(200, json={"access_token": "access-secret", "token_type": "bearer"}),
        ]
    )
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response = next(responses)
        response.request = request
        return response

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle), timeout=30.0)
    monkeypatch.setattr("deepagents_talon.mcp_auth.httpx.AsyncClient", lambda **_kwargs: client)
    monkeypatch.setattr(
        "deepagents_talon.mcp_auth.validate_safe_url",
        lambda url, **_kwargs: url,
    )
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("deepagents_talon.mcp_auth.asyncio.sleep", sleep)

    token = await _run_github_device_flow("github", interactive=True)

    assert token.access_token == "access-secret"  # noqa: S105
    assert sleeps == [1, 1, 6]
    assert [request.url.path for request in requests] == [
        "/login/device/code",
        "/login/oauth/access_token",
        "/login/oauth/access_token",
        "/login/oauth/access_token",
    ]
    output = capsys.readouterr().out
    assert "https://github.com/login/device" in output
    assert "ABCD-1234" in output
    assert "device-secret" not in output


def test_device_response_credentials_are_redacted() -> None:
    response = _DeviceCodeResponse(
        device_code=SecretStr("device-secret"),
        user_code=SecretStr("user-secret"),
        verification_uri="https://github.com/login/device",
        expires_in=120,
    )

    assert "device-secret" not in repr(response)
    assert "user-secret" not in repr(response)
