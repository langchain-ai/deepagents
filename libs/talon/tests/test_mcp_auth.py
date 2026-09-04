from __future__ import annotations

import json
import stat
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

import pytest
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from deepagents_talon.mcp_auth import (
    FileTokenStorage,
    MCPAuthorizationError,
    _with_slack_team,
    build_oauth_provider,
    extract_oauth_callback_url,
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


async def test_slack_login_preseeds_public_client_and_persists_team_with_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "  T01234567  ")
    storage = FileTokenStorage("slack", server_url="https://slack.com/mcp")

    team_id = await prepare_oauth_login(
        server_url="https://slack.com/mcp",
        storage=storage,
        interactive=True,
    )

    client = await storage.get_client_info()
    assert client is not None
    assert client.client_id == "4518649543379.10944517634130"
    assert [str(uri) for uri in client.redirect_uris or []] == ["http://localhost:3118/callback"]
    assert team_id == "T01234567"
    assert await storage.get_slack_team_id() is None
    assert "slack_team_id" not in json.loads(storage.path.read_text(encoding="utf-8"))

    await storage.set_tokens(OAuthToken(access_token="secret"))  # noqa: S106

    assert await storage.get_slack_team_id() == "T01234567"


async def test_slack_login_reuses_team_without_prompt_or_rewriting_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    storage = FileTokenStorage("slack", server_url="https://slack.com/mcp")
    storage.store_slack_team_id_with_tokens("T01234567")
    await storage.set_tokens(OAuthToken(access_token="secret"))  # noqa: S106
    await prepare_oauth_login(
        server_url="https://slack.com/mcp",
        storage=storage,
        interactive=False,
    )
    modified_at = storage.path.stat().st_mtime_ns
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: pytest.fail("cached Slack team ID should skip the prompt"),
    )

    team_id = await prepare_oauth_login(
        server_url="https://slack.com/mcp",
        storage=storage,
        interactive=True,
    )

    assert team_id == "T01234567"
    assert storage.path.stat().st_mtime_ns == modified_at


async def test_slack_team_query_is_encoded_and_replaces_existing_value() -> None:
    urls: list[str] = []

    async def capture(url: str) -> None:
        urls.append(url)

    redirect = _with_slack_team(capture, "T01 &")
    await redirect("https://slack.com/oauth?state=opaque&team=old")

    assert len(urls) == 1
    assert parse_qs(urlparse(urls[0]).query) == {"state": ["opaque"], "team": ["T01 &"]}


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
