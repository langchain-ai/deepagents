"""Tests for the /v1/deepagents/* HTTP client."""

from __future__ import annotations

import json

import httpx
import pytest

from deepagents_cli.deploy.api_client import ApiClient, ApiError


def _transport(handler):
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
    client = ApiClient.from_env(transport=_transport(lambda r: httpx.Response(200, json={})))
    assert client.api_key == "lsv2_pt_a"


def test_endpoint_resolution_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://eu.example.invalid/")
    client = ApiClient.from_env(transport=_transport(lambda r: httpx.Response(200, json={})))
    assert client.endpoint == "https://eu.example.invalid"


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

    def handler(request: httpx.Request) -> httpx.Response:
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

    def handler(request: httpx.Request) -> httpx.Response:
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

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(502, text="bad gw")
        return httpx.Response(200, json={"items": []})

    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    client = ApiClient.from_env(transport=_transport(handler))
    body = client._request("GET", "/v1/deepagents/agents")
    assert calls["n"] == 2
    assert body == {"items": []}
