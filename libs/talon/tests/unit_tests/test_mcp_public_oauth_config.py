from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.mcp_auth import FileTokenStorage
from deepagents_talon.mcp_config import MCPConfigStore
from deepagents_talon.mcp_oauth import MCPOAuthConfigError, parse_oauth_config

if TYPE_CHECKING:
    from pathlib import Path


def test_public_oauth_configuration_keeps_tools_disabled(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    updates: list[bool] = []
    store = MCPConfigStore(path, lambda: updates.append(True), auto_approve=False)
    view, update = store.tools()
    server = {
        "url": "https://openapi.doordash.com/mcp/consumer",
        "auth": "oauth",
        "oauth": {
            "client_id": "assigned-client",
            "callback_url": "http://127.0.0.1:6359/callback",
            "callback_port": 6359,
            "scopes": ["mcp:consumer:write"],
        },
        "disabledTools": ["*"],
    }
    result = update.invoke(
        {
            "server_name": "doordash",
            "server": server,
            "expected_revision": view.invoke({})["revision"],
        }
    )
    assert result["status"] == "updated"
    assert json.loads(path.read_text())["mcpServers"]["doordash"] == server
    assert updates == [True]
    redacted = view.invoke({})
    assert "assigned-client" not in json.dumps(redacted)
    assert "oauth" in redacted["mcpServers"]["doordash"]


@pytest.mark.parametrize(
    ("oauth", "remedy"),
    [
        ("sensitive-invalid", "object"),
        ({"client_id": "\nsensitive-invalid"}, "client_id"),
        ({"client_secret": "sensitive-invalid"}, "Supported MCP oauth fields"),
        ({"callback_port": True}, "integer from 1 to 65535"),
        ({"callback_url": "http://sensitive-invalid.example:6359/callback"}, "HTTP loopback"),
        ({"callback_url": "http://127.0.0.1:6359/callback", "callback_port": 6360}, "must match"),
        ({"scopes": ["sensitive-invalid scope"]}, "scope tokens"),
        ({"scopes": ["read"]}, "requires an explicit client_id"),
        ({"scopes": []}, "requires an explicit client_id"),
        ({"callback_url": "http://localhost:80/callback"}, "canonical spelling"),
        ({"callback_port": 80}, "canonical spelling"),
        ({"callback_url": "http://LOCALHOST:6359/callback"}, "canonical spelling"),
        ({"callback_url": "http://localhost:6359/a/../callback"}, "canonical spelling"),
        ({"callback_url": "http://localhost:6359/caf\u00e9"}, "canonical spelling"),
    ],
)
def test_invalid_public_oauth_update_is_actionable_and_safe(
    tmp_path: Path, oauth: object, remedy: str
) -> None:
    path = tmp_path / "config.json"
    updates: list[bool] = []
    store = MCPConfigStore(path, lambda: updates.append(True), auto_approve=False)
    view, update = store.tools()
    result = update.invoke(
        {
            "server_name": "doordash",
            "server": {"url": "https://example.com/mcp", "auth": "oauth", "oauth": oauth},
            "expected_revision": view.invoke({})["revision"],
        }
    )
    assert result["status"] == "error"
    assert remedy in result["message"]
    assert "sensitive-invalid" not in result["message"]
    assert not path.exists()
    assert updates == []


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
def test_public_oauth_accepts_loopback_hosts(host: str) -> None:
    url = f"http://{host}:6359/callback"
    assert parse_oauth_config({"callback_url": url, "callback_port": 6359}).callback_url == url


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:6359/callback",
        "http://127.0.0.1/callback",
        "http://127.0.0.1:0/callback",
        "http://127.0.0.1:65536/callback",
        "http://127.0.0.1:bad/callback",
        "http://user:secret@127.0.0.1:6359/callback",
        "http://127.0.0.1:6359/callback?code=secret",
        "http://127.0.0.1:6359/callback#secret",
        "http://127.0.0.1:6359/callback\n",
        "http://localhost.evil.example:6359/callback",
        "http://127.0.0.1:6359",
    ],
)
def test_public_oauth_rejects_unsafe_callback_urls(url: str) -> None:
    with pytest.raises(MCPOAuthConfigError, match="HTTP loopback"):
        parse_oauth_config({"callback_url": url})


@pytest.mark.parametrize("port", [False, 0, 65536, "6359", None])
def test_public_oauth_rejects_invalid_ports(port: object) -> None:
    with pytest.raises(MCPOAuthConfigError, match="integer from 1 to 65535"):
        parse_oauth_config({"callback_port": port})


def test_callback_port_alone_selects_localhost() -> None:
    assert (
        parse_oauth_config({"callback_port": 6359}).callback_url == "http://localhost:6359/callback"
    )


def test_public_oauth_storage_isolated_from_other_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    configurations = [
        None,
        {"client_id": "first"},
        {"client_id": "second"},
        {"client_id": "first", "callback_url": "http://127.0.0.1:6359/callback"},
        {"client_id": "first", "scopes": ["mcp:consumer:write"]},
    ]
    paths = {
        FileTokenStorage(
            "doordash", server_url="https://example.com/mcp", oauth=parse_oauth_config(config)
        ).path
        for config in configurations
    }
    assert len(paths) == len(configurations)
    default = FileTokenStorage("doordash", server_url="https://example.com/mcp")
    explicit_default = FileTokenStorage(
        "doordash", server_url="https://example.com/mcp", oauth=parse_oauth_config(None)
    )
    assert default.path == explicit_default.path
