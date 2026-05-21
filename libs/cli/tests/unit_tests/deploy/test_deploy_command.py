"""End-to-end tests for `deepagents deploy` against a mocked HTTP transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
import pytest

import deepagents_cli.deploy.api_client as api_client_module
from deepagents_cli.deploy.commands import execute_deploy_command


def _make_transport(handler):
    return httpx.MockTransport(handler)


def _ns(dir_: Path, **overrides):  # type: ignore[no-untyped-def]
    base = {"dir": str(dir_), "dry_run": False, "detach": True, "reset": False}
    base.update({k.replace("-", "_"): v for k, v in overrides.items()})
    return argparse.Namespace(**base)


def _seed_project(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "agent.json").write_text(
        '{"name": "test-agent", "description": "test",'
        '"runtime": {"model": {"model_id": "anthropic:claude-sonnet-4-6"}}}'
    )
    (root / "AGENTS.md").write_text("You are a test agent.\n")


def test_deploy_dry_run_prints_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_project(tmp_path)
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    execute_deploy_command(_ns(tmp_path, dry_run=True))
    out = capsys.readouterr().out
    payload = json.loads(_extract_json(out))
    assert payload["name"] == "test-agent"
    assert "system_prompt" in payload


def test_deploy_creates_agent_and_writes_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_project(tmp_path)
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path.endswith("/agents"):
            return httpx.Response(201, json={"id": "a-1", "revision": "r-1", "name": "test-agent"})
        return httpx.Response(500)

    monkeypatch.setattr(
        api_client_module.ApiClient, "from_env",
        classmethod(lambda cls, transport=None: cls(
            endpoint="https://api.invalid", api_key="k",
            transport=_make_transport(handler))),
    )
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")

    execute_deploy_command(_ns(tmp_path))

    state = json.loads((tmp_path / ".deepagents" / "state.json").read_text())
    assert state["agent_id"] == "a-1"
    assert state["revision"] == "r-1"
    assert any(method == "POST" and path.endswith("/agents") for method, path in calls)


def test_second_deploy_patches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_project(tmp_path)
    (tmp_path / ".deepagents").mkdir()
    (tmp_path / ".deepagents" / "state.json").write_text(json.dumps({
        "schema_version": 1,
        "agent_id": "a-1",
        "revision": "r-1",
        "endpoint": "https://api.invalid",
        "last_deployed_at": "2026-05-20T00:00:00+00:00",
        "mcp_servers": {},
    }))

    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(200, json={"id": "a-1", "revision": "r-2"})

    monkeypatch.setattr(
        api_client_module.ApiClient, "from_env",
        classmethod(lambda cls, transport=None: cls(
            endpoint="https://api.invalid", api_key="k",
            transport=_make_transport(handler))),
    )
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")

    execute_deploy_command(_ns(tmp_path))
    assert "PATCH" in methods
    state = json.loads((tmp_path / ".deepagents" / "state.json").read_text())
    assert state["revision"] == "r-2"


def test_deploy_404_falls_back_to_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_project(tmp_path)
    (tmp_path / ".deepagents").mkdir()
    (tmp_path / ".deepagents" / "state.json").write_text(json.dumps({
        "schema_version": 1, "agent_id": "stale", "revision": "old",
        "endpoint": None, "last_deployed_at": "0",
        "mcp_servers": {},
    }))

    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "PATCH":
            return httpx.Response(404, json={"code": "not_found", "detail": "gone", "status": 404})
        return httpx.Response(201, json={"id": "new", "revision": "r-x"})

    monkeypatch.setattr(
        api_client_module.ApiClient, "from_env",
        classmethod(lambda cls, transport=None: cls(
            endpoint="https://api.invalid", api_key="k",
            transport=_make_transport(handler))),
    )
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")

    execute_deploy_command(_ns(tmp_path))
    assert methods == ["PATCH", "POST"]
    state = json.loads((tmp_path / ".deepagents" / "state.json").read_text())
    assert state["agent_id"] == "new"


def _extract_json(stdout: str) -> str:
    """Extract the first {...} block from stdout."""
    start = stdout.index("{")
    depth = 0
    for i, ch in enumerate(stdout[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stdout[start : i + 1]
    raise AssertionError("no JSON object found in stdout")


def test_deploy_fails_when_tools_reference_unregistered_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_project(tmp_path)
    (tmp_path / "tools.json").write_text(json.dumps({
        "tools": [{"name": "x", "mcp_server_url": "https://missing.example"}],
        "interrupt_config": {},
    }))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/mcp-servers"):
            return httpx.Response(200, json={"servers": []})
        return httpx.Response(500)

    monkeypatch.setattr(
        api_client_module.ApiClient, "from_env",
        classmethod(lambda cls, transport=None: cls(
            endpoint="https://api.invalid", api_key="k",
            transport=_make_transport(handler))),
    )
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")

    with pytest.raises(SystemExit):
        execute_deploy_command(_ns(tmp_path))
    err = capsys.readouterr().out
    assert "https://missing.example" in err
    assert "deepagents mcp-servers add" in err
