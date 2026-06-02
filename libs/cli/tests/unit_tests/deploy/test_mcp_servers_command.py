"""Tests for `deepagents mcp-servers {list,add,get,update,delete}`."""

from __future__ import annotations

import argparse
import json
import webbrowser
from collections.abc import Callable
from typing import cast

import httpx
import pytest

import deepagents_cli.config as config_module
import deepagents_cli.deploy.api_client as api_client_module
from deepagents_cli.deploy.commands import execute_mcp_servers_command

Handler = Callable[[httpx.Request], httpx.Response]

# A UUID-shaped id keeps get/update/delete/connect on the resolver fast path
# (no list lookup), matching how real server ids look.
_SID = "11111111-1111-4111-8111-111111111111"


def _patch_client(
    monkeypatch: pytest.MonkeyPatch,
    handler: Handler,
    *,
    dotenv_calls: list[str] | None = None,
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")

    def load_dotenv(*, start_path: object) -> bool:
        if dotenv_calls is not None:
            dotenv_calls.append(str(start_path))
        return True

    def from_env(
        cls: type[api_client_module.ApiClient],
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> api_client_module.ApiClient:
        _ = transport
        return cls(
            endpoint="https://api.invalid",
            api_key="k",
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(config_module, "_load_dotenv", load_dotenv)
    monkeypatch.setattr(
        api_client_module.ApiClient,
        "from_env",
        classmethod(from_env),
    )


def test_mcp_servers_add_parses_header_pairs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, dict[str, object]] = {}
    dotenv_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = cast("dict[str, object]", json.loads(request.content))
        return httpx.Response(
            201,
            json={"id": "s1", "name": "Fleet", "url": "https://tools.langchain.com"},
        )

    _patch_client(monkeypatch, handler, dotenv_calls=dotenv_calls)
    execute_mcp_servers_command(
        argparse.Namespace(
            mcp_cmd="add",
            url="https://tools.langchain.com",
            name="Fleet",
            header=["X-Api-Key=secret-value"],
            auth_type="headers",
            no_tools=True,
        )
    )
    assert dotenv_calls
    assert captured["body"]["headers"] == [
        {"key": "X-Api-Key", "value": "secret-value"}
    ]
    assert captured["body"]["name"] == "Fleet"
    out = capsys.readouterr().out
    assert "s1" in out


def test_mcp_servers_add_defaults_name_to_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = cast("dict[str, object]", json.loads(request.content))
        return httpx.Response(201, json={"id": "s1"})

    _patch_client(monkeypatch, handler)
    execute_mcp_servers_command(
        argparse.Namespace(
            mcp_cmd="add",
            url="https://tools.langchain.com",
            name=None,
            header=[],
            auth_type="headers",
            no_tools=True,
        )
    )
    assert captured["body"]["name"] == "tools.langchain.com"


def test_mcp_servers_add_oauth_sends_per_user_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = cast("dict[str, object]", json.loads(request.content))
        return httpx.Response(201, json={"id": "s1"})

    _patch_client(monkeypatch, handler)
    execute_mcp_servers_command(
        argparse.Namespace(
            mcp_cmd="add",
            url="https://tools.langchain.com",
            name="GitHub",
            header=[],
            auth_type="oauth",
            connect=False,
            scope=[],
            force_new=False,
            timeout=300,
            no_browser=False,
            no_tools=True,
        )
    )
    assert captured["body"] == {
        "name": "GitHub",
        "url": "https://tools.langchain.com",
        "auth_type": "oauth",
        "oauth_mode": "per_user_dynamic_client",
    }


def test_mcp_servers_add_oauth_rejects_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        msg = "should not call"
        raise AssertionError(msg)

    _patch_client(monkeypatch, handler)
    with pytest.raises(SystemExit):
        execute_mcp_servers_command(
            argparse.Namespace(
                mcp_cmd="add",
                url="https://tools.langchain.com",
                name="GitHub",
                header=["X-Api-Key=secret-value"],
                auth_type="oauth",
                connect=False,
                scope=[],
                force_new=False,
                timeout=300,
                no_browser=False,
            )
        )


def test_mcp_servers_add_connect_runs_oauth_flow(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[tuple[str, str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        requests.append((request.method, request.url.path, body))
        if request.method == "POST" and request.url.path.endswith("/mcp-servers"):
            return httpx.Response(
                201,
                json={"id": "s1", "name": "GitHub", "url": "https://tools.example"},
            )
        if request.url.path.endswith("/oauth-provider"):
            return httpx.Response(200, json={"oauth_provider_id": "provider-1"})
        return httpx.Response(
            200,
            json={
                "id": "session-1",
                "provider_id": "provider-1",
                "status": "COMPLETED",
            },
        )

    _patch_client(monkeypatch, handler)
    execute_mcp_servers_command(
        argparse.Namespace(
            mcp_cmd="add",
            url="https://tools.example",
            name="GitHub",
            header=[],
            auth_type="oauth",
            connect=True,
            scope=[],
            force_new=False,
            timeout=300,
            no_browser=True,
            no_tools=True,
        )
    )
    assert requests == [
        (
            "POST",
            "/v1/deepagents/mcp-servers",
            {
                "name": "GitHub",
                "url": "https://tools.example",
                "auth_type": "oauth",
                "oauth_mode": "per_user_dynamic_client",
            },
        ),
        ("POST", "/v1/deepagents/mcp-servers/s1/oauth-provider", {}),
        (
            "POST",
            "/v1/deepagents/auth-sessions",
            {"provider_id": "provider-1", "scopes": [], "strategy": "REUSE"},
        ),
    ]
    assert "MCP OAuth connection is ready." in capsys.readouterr().out


def test_mcp_servers_connect_opens_url_and_polls(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    opened: list[str] = []
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

    monkeypatch.setattr(
        webbrowser,
        "open",
        lambda url: opened.append(url) or True,
    )
    _patch_client(monkeypatch, handler)
    execute_mcp_servers_command(
        argparse.Namespace(
            mcp_cmd="connect",
            mcp_server_id=_SID,
            scope=["repo", "read:user"],
            force_new=True,
            timeout=30,
            no_browser=False,
        )
    )
    assert opened == ["https://auth.example/authorize"]
    assert requests == [
        ("POST", f"/v1/deepagents/mcp-servers/{_SID}/oauth-provider", {}, ""),
        (
            "POST",
            "/v1/deepagents/auth-sessions",
            {
                "provider_id": "provider-1",
                "scopes": ["repo", "read:user"],
                "strategy": "CREATE",
            },
            "",
        ),
        ("GET", "/v1/deepagents/auth-sessions/session-1", {}, "wait_seconds=5"),
    ]
    out = capsys.readouterr().out
    assert "https://auth.example/authorize" in out
    assert "MCP OAuth connection is ready." in out


def test_mcp_servers_add_bad_header_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        msg = "should not call"
        raise AssertionError(msg)

    _patch_client(monkeypatch, handler)
    with pytest.raises(SystemExit):
        execute_mcp_servers_command(
            argparse.Namespace(
                mcp_cmd="add",
                url="https://x",
                name=None,
                header=["no-equals-here"],
                auth_type="headers",
            )
        )


def test_mcp_servers_add_headers_lists_tools(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/mcp-servers"):
            return httpx.Response(
                201,
                json={"id": _SID, "name": "Fleet", "url": "https://tools.example"},
            )
        if request.url.path == "/v1/deepagents/mcp/tools":
            assert request.url.params.get("url") == "https://tools.example"
            return httpx.Response(
                200,
                json={"tools": [{"name": "read_url", "description": "Read a URL."}]},
            )
        msg = f"unexpected path {request.url.path}"
        raise AssertionError(msg)

    _patch_client(monkeypatch, handler)
    execute_mcp_servers_command(
        argparse.Namespace(
            mcp_cmd="add",
            url="https://tools.example",
            name="Fleet",
            header=["X-Api-Key=secret"],
            auth_type="headers",
            no_tools=False,
        )
    )
    out = capsys.readouterr().out
    assert "Created mcp_server" in out
    assert "read_url" in out
    assert '"mcp_server_url": "https://tools.example"' in out


def test_mcp_servers_add_oauth_without_connect_hints(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(
            201,
            json={"id": _SID, "name": "GitHub", "url": "https://tools.example"},
        )

    _patch_client(monkeypatch, handler)
    execute_mcp_servers_command(
        argparse.Namespace(
            mcp_cmd="add",
            url="https://tools.example",
            name="GitHub",
            header=[],
            auth_type="oauth",
            connect=False,
            scope=[],
            force_new=False,
            timeout=300,
            no_browser=False,
            no_tools=False,
        )
    )
    out = capsys.readouterr().out
    assert "After connecting, run" in out
    assert "/v1/deepagents/mcp/tools" not in paths


def test_mcp_servers_add_tool_listing_error_degrades(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/mcp-servers"):
            return httpx.Response(
                201,
                json={"id": _SID, "name": "Fleet", "url": "https://tools.example"},
            )
        if request.url.path == "/v1/deepagents/mcp/tools":
            return httpx.Response(404, json={"detail": "not found"})
        msg = f"unexpected path {request.url.path}"
        raise AssertionError(msg)

    _patch_client(monkeypatch, handler)
    # `add` must still succeed (exit 0) even though tool discovery 404s.
    execute_mcp_servers_command(
        argparse.Namespace(
            mcp_cmd="add",
            url="https://tools.example",
            name="Fleet",
            header=[],
            auth_type="headers",
            no_tools=False,
        )
    )
    out = capsys.readouterr().out
    assert "Created mcp_server" in out
    assert "mcp-servers tools" in out


def test_mcp_servers_add_no_tools_skips_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(
            201,
            json={"id": _SID, "name": "Fleet", "url": "https://tools.example"},
        )

    _patch_client(monkeypatch, handler)
    execute_mcp_servers_command(
        argparse.Namespace(
            mcp_cmd="add",
            url="https://tools.example",
            name="Fleet",
            header=[],
            auth_type="headers",
            no_tools=True,
        )
    )
    assert "/v1/deepagents/mcp/tools" not in paths


def test_mcp_servers_list(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"id": "s1", "url": "https://x"}])

    _patch_client(monkeypatch, handler)
    execute_mcp_servers_command(argparse.Namespace(mcp_cmd="list"))
    assert "s1" in capsys.readouterr().out


def test_mcp_servers_get_redacts_header_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/v1/deepagents/mcp-servers/{_SID}"
        return httpx.Response(
            200,
            json={
                "id": _SID,
                "headers": [{"key": "X-Api-Key", "value": "secret-value"}],
            },
        )

    _patch_client(monkeypatch, handler)
    execute_mcp_servers_command(argparse.Namespace(mcp_cmd="get", mcp_server_id=_SID))
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["headers"] == [{"key": "X-Api-Key", "value": "***"}]
    assert "secret-value" not in out


def test_mcp_servers_update_sends_patch_body(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"id": "s1", "name": "Fleet", "url": "https://new.example/mcp"},
        )

    _patch_client(monkeypatch, handler)
    execute_mcp_servers_command(
        argparse.Namespace(
            mcp_cmd="update",
            mcp_server_id=_SID,
            url="https://new.example/mcp",
            header=["X-Api-Key=new-value"],
            clear_headers=False,
            auth_type="headers",
        )
    )
    assert captured == {
        "method": "PATCH",
        "path": f"/v1/deepagents/mcp-servers/{_SID}",
        "body": {
            "url": "https://new.example/mcp",
            "headers": [{"key": "X-Api-Key", "value": "new-value"}],
            "auth_type": "headers",
        },
    }
    assert "Updated mcp_server" in capsys.readouterr().out


def test_mcp_servers_update_clear_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "s1"})

    _patch_client(monkeypatch, handler)
    execute_mcp_servers_command(
        argparse.Namespace(
            mcp_cmd="update",
            mcp_server_id=_SID,
            url=None,
            header=None,
            clear_headers=True,
            auth_type=None,
        )
    )
    assert captured["body"] == {"headers": []}


def test_mcp_servers_update_requires_a_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        msg = "should not call"
        raise AssertionError(msg)

    _patch_client(monkeypatch, handler)
    with pytest.raises(SystemExit):
        execute_mcp_servers_command(
            argparse.Namespace(
                mcp_cmd="update",
                mcp_server_id="s1",
                url=None,
                header=None,
                clear_headers=False,
                auth_type=None,
            )
        )


def test_mcp_servers_update_rejects_header_and_clear_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        msg = "should not call"
        raise AssertionError(msg)

    _patch_client(monkeypatch, handler)
    with pytest.raises(SystemExit):
        execute_mcp_servers_command(
            argparse.Namespace(
                mcp_cmd="update",
                mcp_server_id="s1",
                url=None,
                header=["X-Api-Key=value"],
                clear_headers=True,
                auth_type=None,
            )
        )


def test_mcp_servers_delete(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(204)

    _patch_client(monkeypatch, handler)
    execute_mcp_servers_command(
        argparse.Namespace(mcp_cmd="delete", mcp_server_id=_SID, yes=True)
    )
    assert methods == ["DELETE"]
    assert "Deleted" in capsys.readouterr().out


def test_mcp_servers_tools_lists_and_prints_snippet(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/v1/deepagents/mcp-servers/{_SID}":
            return httpx.Response(
                200,
                json={
                    "id": _SID,
                    "name": "deep-wiki",
                    "url": "https://mcp.deepwiki.com/mcp",
                },
            )
        if request.url.path == "/v1/deepagents/mcp/tools":
            assert request.url.params.get("url") == "https://mcp.deepwiki.com/mcp"
            return httpx.Response(
                200,
                json={
                    "tools": [
                        {"name": "read_wiki", "description": "Read a page.\nMore."},
                        {"name": "search", "description": "Search."},
                    ],
                    "cached": True,
                },
            )
        msg = f"unexpected path {request.url.path}"
        raise AssertionError(msg)

    _patch_client(monkeypatch, handler)
    execute_mcp_servers_command(argparse.Namespace(mcp_cmd="tools", mcp_server_id=_SID))
    out = capsys.readouterr().out
    assert "read_wiki" in out
    assert "Read a page." in out  # only the first description line
    assert '"mcp_server_url": "https://mcp.deepwiki.com/mcp"' in out
    assert '"mcp_server_name": "deep-wiki"' in out
    snippet = json.loads(out[out.index("{") :])
    assert [t["name"] for t in snippet["tools"]] == ["read_wiki", "search"]


def test_mcp_servers_delete_resolves_name(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(
                200,
                json=[
                    {"id": _SID, "name": "vic-notion", "url": "https://notion/mcp"},
                    {"id": "other", "name": "deep-wiki", "url": "https://wiki/mcp"},
                ],
            )
        return httpx.Response(204)

    _patch_client(monkeypatch, handler)
    execute_mcp_servers_command(
        argparse.Namespace(mcp_cmd="delete", mcp_server_id="vic-notion", yes=True)
    )
    assert requests == [
        ("GET", "/v1/deepagents/mcp-servers"),
        ("DELETE", f"/v1/deepagents/mcp-servers/{_SID}"),
    ]
    assert _SID in capsys.readouterr().out


def test_mcp_servers_delete_resolves_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(
                200,
                json=[
                    {"id": _SID, "name": "notion", "url": "https://mcp.notion.com/mcp"}
                ],
            )
        return httpx.Response(204)

    _patch_client(monkeypatch, handler)
    # Trailing slash differs from the registered URL; normalization should match.
    execute_mcp_servers_command(
        argparse.Namespace(
            mcp_cmd="delete",
            mcp_server_id="https://mcp.notion.com/mcp/",
            yes=True,
        )
    )
    assert ("DELETE", f"/v1/deepagents/mcp-servers/{_SID}") in requests


def test_mcp_servers_resolve_not_found_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(200, json=[{"id": _SID, "name": "other"}])

    _patch_client(monkeypatch, handler)
    with pytest.raises(SystemExit):
        execute_mcp_servers_command(
            argparse.Namespace(mcp_cmd="delete", mcp_server_id="missing", yes=True)
        )
    assert "no MCP server matches" in capsys.readouterr().out


def test_mcp_servers_resolve_ambiguous_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(
            200,
            json=[
                {"id": "id-a", "name": "dupe", "url": "https://a"},
                {"id": "id-b", "name": "dupe", "url": "https://b"},
            ],
        )

    _patch_client(monkeypatch, handler)
    with pytest.raises(SystemExit):
        execute_mcp_servers_command(
            argparse.Namespace(mcp_cmd="delete", mcp_server_id="dupe", yes=True)
        )
    assert "matches multiple MCP servers" in capsys.readouterr().out
