"""Tests for `deepagents mcp-servers {list,add,get,update,delete}`."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import cast

import httpx
import pytest

import deepagents_cli.config as config_module
import deepagents_cli.deploy.api_client as api_client_module
from deepagents_cli.deploy.commands import execute_mcp_servers_command

Handler = Callable[[httpx.Request], httpx.Response]


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
        )
    )
    assert captured["body"]["name"] == "tools.langchain.com"


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
        assert request.url.path == "/v1/deepagents/mcp-servers/s1"
        return httpx.Response(
            200,
            json={
                "id": "s1",
                "headers": [{"key": "X-Api-Key", "value": "secret-value"}],
            },
        )

    _patch_client(monkeypatch, handler)
    execute_mcp_servers_command(argparse.Namespace(mcp_cmd="get", mcp_server_id="s1"))
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
            mcp_server_id="s1",
            url="https://new.example/mcp",
            header=["X-Api-Key=new-value"],
            clear_headers=False,
            auth_type="headers",
        )
    )
    assert captured == {
        "method": "PATCH",
        "path": "/v1/deepagents/mcp-servers/s1",
        "body": {
            "url": "https://new.example/mcp",
            "headers": [{"key": "X-Api-Key", "value": "new-value"}],
            "auth_type": "headers",
        },
    }
    assert "Updated mcp_server s1" in capsys.readouterr().out


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
            mcp_server_id="s1",
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
        argparse.Namespace(mcp_cmd="delete", mcp_server_id="s1", yes=True)
    )
    assert methods == ["DELETE"]
    assert "Deleted" in capsys.readouterr().out
