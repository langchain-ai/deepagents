"""Tests for `deepagents agents {list,get,delete}`."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import TYPE_CHECKING

import httpx

import deepagents_cli.config as config_module
import deepagents_cli.deploy.api_client as api_client_module
from deepagents_cli.deploy.commands import execute_agents_command

if TYPE_CHECKING:
    import pytest


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


def test_agents_list(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"items": [{"id": "a1", "name": "x"}], "next_cursor": None}
        )

    _patch_client(monkeypatch, handler, dotenv_calls=calls)
    execute_agents_command(argparse.Namespace(agents_cmd="list"))
    out = capsys.readouterr().out
    assert calls
    assert "a1" in out
    assert "x" in out


def test_agents_get(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/deepagents/agents/a1"
        return httpx.Response(200, json={"id": "a1", "name": "x", "revision": "r1"})

    _patch_client(monkeypatch, handler)
    execute_agents_command(
        argparse.Namespace(agents_cmd="get", agent_id="a1", include_files=False)
    )
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["id"] == "a1"


def test_agents_delete_requires_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "n")

    def handler(_request: httpx.Request) -> httpx.Response:
        msg = "should not be called"
        raise AssertionError(msg)

    _patch_client(monkeypatch, handler)
    execute_agents_command(
        argparse.Namespace(agents_cmd="delete", agent_id="a1", yes=False)
    )
    assert "Aborted" in capsys.readouterr().out


def test_agents_delete_with_yes_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        return httpx.Response(204)

    _patch_client(monkeypatch, handler)
    execute_agents_command(
        argparse.Namespace(agents_cmd="delete", agent_id="a1", yes=True)
    )
    assert calls == ["DELETE"]
    assert "Deleted" in capsys.readouterr().out
