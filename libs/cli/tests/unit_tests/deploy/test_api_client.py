"""Tests for the /v1/deepagents/* HTTP client."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from deepagents_cli.deploy.api_client import ApiClient, ApiError

Handler = Callable[[httpx.Request], httpx.Response]


def _transport(handler: Handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_from_env_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        ApiClient.from_env()
    assert excinfo.value.code != 0


def test_from_env_prefers_langsmith_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_a")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_pt_b")
    client = ApiClient.from_env(
        transport=_transport(lambda _request: httpx.Response(200, json={}))
    )
    assert client.api_key == "lsv2_pt_a"


def test_client_ignores_environment_proxy_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.example.invalid")
    client = ApiClient.from_env(
        transport=_transport(lambda _request: httpx.Response(200, json={}))
    )
    assert client._client.trust_env is False


def test_endpoint_resolution_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://eu.example.invalid/")
    client = ApiClient.from_env(
        transport=_transport(lambda _request: httpx.Response(200, json={}))
    )
    assert client.endpoint == "https://eu.example.invalid"


def test_endpoint_defaults_when_env_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGCHAIN_ENDPOINT", raising=False)
    client = ApiClient.from_env(
        transport=_transport(lambda _request: httpx.Response(200, json={}))
    )
    assert client.endpoint == "https://api.smith.langchain.com"


def test_endpoint_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://env.example.invalid")
    client = ApiClient.from_env(
        transport=_transport(lambda _request: httpx.Response(200, json={}))
    )
    assert client.endpoint == "https://env.example.invalid"


def test_endpoint_rejects_non_https(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "http://env.example.invalid")
    with pytest.raises(SystemExit):
        ApiClient.from_env(
            transport=_transport(lambda _request: httpx.Response(200, json={}))
        )


def test_endpoint_rejects_userinfo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://user:pass@env.example.invalid")
    with pytest.raises(SystemExit):
        ApiClient.from_env(
            transport=_transport(lambda _request: httpx.Response(200, json={}))
        )


def test_request_sends_x_api_key_header(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["x-api-key"] = request.headers.get("x-api-key", "")
        return httpx.Response(200, json={})

    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_xyz")
    client = ApiClient.from_env(transport=_transport(handler))
    client._request("GET", "/v1/deepagents/agents")
    assert seen["x-api-key"] == "lsv2_pt_xyz"


def test_4xx_parses_error_response(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "type": "https://errors.langchain.com/bad-request",
        "code": "invalid_request",
        "detail": "tools.tools[0].mcp_server_url is required",
        "status": 400,
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=body)

    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    client = ApiClient.from_env(transport=_transport(handler))
    with pytest.raises(ApiError) as excinfo:
        client._request("POST", "/v1/deepagents/agents", json={"name": "x"})
    assert excinfo.value.detail == body["detail"]
    assert excinfo.value.status == 400
    assert excinfo.value.code == "invalid_request"


def test_5xx_retries_once_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="upstream")

    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    client = ApiClient.from_env(transport=_transport(handler))
    with pytest.raises(ApiError) as excinfo:
        client._request("GET", "/v1/deepagents/agents")
    assert calls["n"] == 2
    assert excinfo.value.status == 503


def test_5xx_retry_succeeds_on_second_try(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(502, text="bad gw")
        return httpx.Response(200, json={"items": []})

    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    client = ApiClient.from_env(transport=_transport(handler))
    body = client._request("GET", "/v1/deepagents/agents")
    assert calls["n"] == 2
    assert body == {"items": []}


def test_create_agent_posts_to_v1_deepagents_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            201,
            json={"id": "agent-1", "revision": "rev-1", "name": "x"},
        )

    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    client = ApiClient.from_env(transport=_transport(handler))
    agent = client.create_agent({"name": "x", "system_prompt": "hi"})
    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/deepagents/agents"
    assert captured["body"] == {"name": "x", "system_prompt": "hi"}
    assert agent == {"id": "agent-1", "revision": "rev-1", "name": "x"}


def test_get_agent_passes_include_files(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["query"] = str(request.url.query)
        return httpx.Response(200, json={"id": "a", "revision": "r"})

    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    client = ApiClient.from_env(transport=_transport(handler))
    client.get_agent("a", include_files=True)
    assert seen["path"] == "/v1/deepagents/agents/a"
    assert "include_files=true" in seen["query"]


def test_list_agents_paginates(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [
        {"items": [{"id": "1"}], "next_cursor": "c2"},
        {"items": [{"id": "2"}], "next_cursor": None},
    ]
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        body = pages[calls["n"]]
        calls["n"] += 1
        return httpx.Response(200, json=body)

    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    client = ApiClient.from_env(transport=_transport(handler))
    out = list(client.iter_agents(page_size=50))
    assert [a["id"] for a in out] == ["1", "2"]
    assert calls["n"] == 2


def test_patch_agent_passes_body(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "a", "revision": "r2"})

    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    client = ApiClient.from_env(transport=_transport(handler))
    client.patch_agent("a", {"description": "new"})
    assert captured["method"] == "PATCH"
    assert captured["body"] == {"description": "new"}


def test_delete_agent_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        return httpx.Response(204)

    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    client = ApiClient.from_env(transport=_transport(handler))
    assert client.delete_agent("a") is None


def test_list_mcp_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"servers": [{"id": "s1", "url": "https://tools.langchain.com"}]}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/deepagents/mcp-servers"
        return httpx.Response(200, json=body)

    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    client = ApiClient.from_env(transport=_transport(handler))
    assert client.list_mcp_servers() == body["servers"]


def test_list_mcp_servers_accepts_bare_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = [{"id": "s1", "url": "https://tools.langchain.com"}]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/deepagents/mcp-servers"
        return httpx.Response(200, json=body)

    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    client = ApiClient.from_env(transport=_transport(handler))
    assert client.list_mcp_servers() == body


def test_create_mcp_server(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            201,
            json={"id": "s1", "name": "Fleet", "url": "https://tools.langchain.com"},
        )

    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    client = ApiClient.from_env(transport=_transport(handler))
    out = client.create_mcp_server(
        name="Fleet",
        url="https://tools.langchain.com",
        headers=[{"key": "X-Api-Key", "value": "secret"}],
        auth_type="headers",
    )
    assert out["id"] == "s1"
    assert captured["body"] == {
        "name": "Fleet",
        "url": "https://tools.langchain.com",
        "headers": [{"key": "X-Api-Key", "value": "secret"}],
        "auth_type": "headers",
    }


def test_create_oauth_mcp_server(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            201,
            json={"id": "s1", "name": "GitHub", "url": "https://tools.example/mcp"},
        )

    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    client = ApiClient.from_env(transport=_transport(handler))
    out = client.create_mcp_server(
        name="GitHub",
        url="https://tools.example/mcp",
        auth_type="oauth",
        oauth_mode="per_user_dynamic_client",
    )
    assert out["id"] == "s1"
    assert captured["body"] == {
        "name": "GitHub",
        "url": "https://tools.example/mcp",
        "auth_type": "oauth",
        "oauth_mode": "per_user_dynamic_client",
    }


def test_update_mcp_server_patches_body(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"id": "s1", "url": "https://new.example/mcp"},
        )

    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    client = ApiClient.from_env(transport=_transport(handler))
    out = client.update_mcp_server(
        "s1",
        url="https://new.example/mcp",
        headers=[],
        auth_type="headers",
    )
    assert out["id"] == "s1"
    assert captured == {
        "method": "PATCH",
        "path": "/v1/deepagents/mcp-servers/s1",
        "body": {
            "url": "https://new.example/mcp",
            "headers": [],
            "auth_type": "headers",
        },
    }


def test_delete_mcp_server(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/v1/deepagents/mcp-servers/s1"
        return httpx.Response(204)

    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    client = ApiClient.from_env(transport=_transport(handler))
    assert client.delete_mcp_server("s1") is None


def test_mcp_oauth_provider_and_auth_session_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, str, dict[str, object], str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        query = request.url.query.decode()
        requests.append((request.method, request.url.path, body, query))
        if request.url.path.endswith("/oauth-provider"):
            return httpx.Response(200, json={"oauth_provider_id": "provider-1"})
        if request.method == "POST" and request.url.path.endswith("/auth-sessions"):
            return httpx.Response(
                201,
                json={
                    "id": "session-1",
                    "provider_id": "provider-1",
                    "status": "PENDING",
                    "verification_url": "https://auth.example/authorize",
                },
            )
        return httpx.Response(200, json={"id": "session-1", "status": "COMPLETED"})

    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    client = ApiClient.from_env(transport=_transport(handler))
    assert client.register_mcp_oauth_provider("s1") == {
        "oauth_provider_id": "provider-1"
    }
    assert (
        client.create_auth_session(
            provider_id="provider-1",
            scopes=[],
            strategy="REUSE",
        )["status"]
        == "PENDING"
    )
    assert client.get_auth_session("session-1", wait_seconds=5)["status"] == (
        "COMPLETED"
    )
    assert requests == [
        ("POST", "/v1/deepagents/mcp-servers/s1/oauth-provider", {}, ""),
        (
            "POST",
            "/v1/deepagents/auth-sessions",
            {"provider_id": "provider-1", "scopes": [], "strategy": "REUSE"},
            "",
        ),
        ("GET", "/v1/deepagents/auth-sessions/session-1", {}, "wait_seconds=5"),
    ]


def test_agent_directory_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "files": {"AGENTS.md": {"type": "file", "content": "hi"}},
                    "commit_hash": "c1",
                },
            )
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"commit": {"commit_hash": "c2"}})

    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    client = ApiClient.from_env(transport=_transport(handler))
    assert client.get_agent_directory("a")["commit_hash"] == "c1"
    assert captured["path"] == "/v1/platform/hub/repos/-/a/directories"

    out = client.commit_agent_directory(
        "a",
        files={"AGENTS.md": {"type": "file", "content": "new"}},
        parent_commit="c1",
    )

    assert captured["path"] == "/v1/platform/hub/repos/-/a/directories/commits"
    assert captured["body"] == {
        "files": {"AGENTS.md": {"type": "file", "content": "new"}},
        "parent_commit": "c1",
    }
    assert out == {"commit": {"commit_hash": "c2"}}
