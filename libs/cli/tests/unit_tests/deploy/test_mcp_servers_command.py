"""Tests for `deepagents mcp-servers {list,add,get,delete}`."""

from __future__ import annotations

import argparse
import json

import httpx
import pytest

import deepagents_cli.deploy.api_client as api_client_module
from deepagents_cli.deploy.commands import execute_mcp_servers_command


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.setattr(
        api_client_module.ApiClient, "from_env",
        classmethod(lambda cls, transport=None: cls(
            endpoint="https://api.invalid", api_key="k",
            transport=httpx.MockTransport(handler))),
    )


def test_mcp_servers_add_parses_header_pairs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "s1", "name": "Fleet", "url": "https://tools.langchain.com"})

    _patch_client(monkeypatch, handler)
    execute_mcp_servers_command(argparse.Namespace(
        mcp_cmd="add",
        url="https://tools.langchain.com",
        name="Fleet",
        header=["X-Api-Key=secret-value"],
        auth_type="headers",
    ))
    assert captured["body"]["headers"] == [{"key": "X-Api-Key", "value": "secret-value"}]
    assert captured["body"]["name"] == "Fleet"
    out = capsys.readouterr().out
    assert "s1" in out


def test_mcp_servers_add_defaults_name_to_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "s1"})

    _patch_client(monkeypatch, handler)
    execute_mcp_servers_command(argparse.Namespace(
        mcp_cmd="add", url="https://tools.langchain.com",
        name=None, header=[], auth_type="headers",
    ))
    assert captured["body"]["name"] == "tools.langchain.com"


def test_mcp_servers_add_bad_header_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not call")

    _patch_client(monkeypatch, handler)
    with pytest.raises(SystemExit):
        execute_mcp_servers_command(argparse.Namespace(
            mcp_cmd="add", url="https://x", name=None,
            header=["no-equals-here"], auth_type="headers",
        ))


def test_mcp_servers_list(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"servers": [{"id": "s1", "url": "https://x"}]})
    _patch_client(monkeypatch, handler)
    execute_mcp_servers_command(argparse.Namespace(mcp_cmd="list"))
    assert "s1" in capsys.readouterr().out


def test_mcp_servers_delete(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    methods: list[str] = []
    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(204)
    _patch_client(monkeypatch, handler)
    execute_mcp_servers_command(argparse.Namespace(mcp_cmd="delete", mcp_server_id="s1", yes=True))
    assert methods == ["DELETE"]
    assert "Deleted" in capsys.readouterr().out
