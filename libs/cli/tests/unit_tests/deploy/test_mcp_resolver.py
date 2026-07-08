"""Tests for resolve_referenced_servers."""

from __future__ import annotations

import httpx
import pytest

from deepagents_cli.deploy.api_client import ApiClient
from deepagents_cli.deploy.mcp_resolver import (
    UninvokableServersError,
    UnresolvedServersError,
    resolve_referenced_servers,
)


def _client(monkeypatch: pytest.MonkeyPatch, handler) -> ApiClient:
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    return ApiClient.from_env(transport=httpx.MockTransport(handler))


def test_url_normalization_strips_trailing_slash_and_lowercases() -> None:
    from deepagents_cli.deploy.mcp_resolver import _normalize_url

    assert (
        _normalize_url("https://tools.langchain.com/") == "https://tools.langchain.com"
    )
    assert (
        _normalize_url("HTTPS://Tools.LangChain.com") == "https://tools.langchain.com"
    )


def test_all_referenced_servers_present(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"id": "s1", "url": "https://tools.langchain.com"},
                {"id": "s2", "url": "https://other.example/"},
            ],
        )

    client = _client(monkeypatch, handler)
    payload = {
        "tools": {
            "tools": [{"name": "x", "mcp_server_url": "https://tools.langchain.com/"}]
        },
        "subagents": [
            {
                "tools": {
                    "tools": [{"name": "y", "mcp_server_url": "https://other.example"}]
                }
            }
        ],
    }
    cache = resolve_referenced_servers(client, payload, cache={})
    assert cache["https://tools.langchain.com"] == "s1"
    assert cache["https://other.example"] == "s2"


def test_unresolved_server_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    client = _client(monkeypatch, handler)
    payload = {
        "tools": {"tools": [{"name": "x", "mcp_server_url": "https://missing.example"}]}
    }
    with pytest.raises(UnresolvedServersError) as excinfo:
        resolve_referenced_servers(client, payload, cache={})
    assert excinfo.value.urls == ("https://missing.example",)


def test_uninvokable_server_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "id": "s1",
                    "name": "GitHub",
                    "url": "https://tools.example",
                    "auth_type": "oauth",
                    "can_invoke": False,
                }
            ],
        )

    client = _client(monkeypatch, handler)
    payload = {
        "tools": {"tools": [{"name": "x", "mcp_server_url": "https://tools.example"}]}
    }
    with pytest.raises(UninvokableServersError) as excinfo:
        resolve_referenced_servers(client, payload, cache={})
    assert excinfo.value.urls == ("https://tools.example",)
    assert "cannot invoke" in str(excinfo.value)
    assert "deepagents mcp-servers connect s1" in str(excinfo.value)


def test_cached_ids_are_refreshed_from_list_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200,
            json=[{"id": "fresh", "url": "https://tools.example"}],
        )

    client = _client(monkeypatch, handler)
    payload = {
        "tools": {"tools": [{"name": "x", "mcp_server_url": "https://tools.example"}]}
    }
    cache = {"https://tools.example": "s1"}
    out = resolve_referenced_servers(client, payload, cache=cache)
    assert calls["n"] == 1
    assert out["https://tools.example"] == "fresh"
