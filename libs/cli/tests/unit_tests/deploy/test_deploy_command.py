"""End-to-end tests for `deepagents deploy` against a mocked HTTP transport."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import TYPE_CHECKING

import httpx
import pytest

import deepagents_cli.deploy.api_client as api_client_module
import deepagents_cli.deploy.state as state_module
from deepagents_cli.deploy.commands import execute_deploy_command
from deepagents_cli.deploy.state import State

if TYPE_CHECKING:
    from pathlib import Path


Handler = Callable[[httpx.Request], httpx.Response]


def _make_transport(handler: Handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _ns(dir_: Path, **overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "dir": str(dir_),
        "dry_run": False,
        "detach": True,
        "reset": False,
        "yes": False,
    }
    base.update({k.replace("-", "_"): v for k, v in overrides.items()})
    return argparse.Namespace(**base)


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> None:
    def from_env(
        cls: type[api_client_module.ApiClient],
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> api_client_module.ApiClient:
        _ = transport
        return cls(
            endpoint="https://api.invalid",
            api_key="k",
            transport=_make_transport(handler),
        )

    monkeypatch.setattr(
        api_client_module.ApiClient,
        "from_env",
        classmethod(from_env),
    )


def _seed_project(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "agent.json").write_text(
        '{"name": "test-agent", "description": "test",'
        '"runtime": {"model": {"model_id": "anthropic:claude-sonnet-4-6"}}}'
    )
    (root / "AGENTS.md").write_text("You are a test agent.\n")


@pytest.fixture(autouse=True)
def _state_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(state_module, "_STATE_ROOT", tmp_path / "deploy-state")


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
    assert payload["agent_payload"]["name"] == "test-agent"
    assert "system_prompt" in payload["agent_payload"]
    assert (
        payload["directory_files"]["AGENTS.md"]["content"] == "You are a test agent.\n"
    )


def test_deploy_dry_run_normalizes_legacy_sandbox_backend(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_project(tmp_path)
    (tmp_path / "agent.json").write_text(
        '{"name": "test-agent", "backend": {"type": "sandbox"}}'
    )
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    execute_deploy_command(_ns(tmp_path, dry_run=True))
    out = capsys.readouterr().out
    payload = json.loads(_extract_json(out))
    assert payload["agent_payload"]["backend"] == {
        "type": "sandbox",
        "sandbox_config": {"scope": "thread"},
    }


def test_deploy_creates_agent_and_writes_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_project(tmp_path)
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path.endswith("/agents"):
            return httpx.Response(
                201, json={"id": "a-1", "revision": "r-1", "name": "test-agent"}
            )
        return httpx.Response(500)

    _patch_client(monkeypatch, handler)
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")

    execute_deploy_command(_ns(tmp_path))

    state = State.load(tmp_path, endpoint="https://api.invalid")
    assert state.agent_id == "a-1"
    assert state.revision == "r-1"
    assert any(method == "POST" and path.endswith("/agents") for method, path in calls)


def test_deploy_ignores_project_local_state_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_project(tmp_path)
    (tmp_path / ".deepagents").mkdir()
    (tmp_path / ".deepagents" / "state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "agent_id": "attacker-controlled",
                "revision": "old",
                "endpoint": "https://api.invalid",
                "last_deployed_at": "2026-05-20T00:00:00+00:00",
                "mcp_servers": {},
            }
        )
    )
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path.endswith("/agents"):
            return httpx.Response(
                201, json={"id": "a-safe", "revision": "r-1", "name": "test-agent"}
            )
        return httpx.Response(500)

    _patch_client(monkeypatch, handler)
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")

    execute_deploy_command(_ns(tmp_path))

    assert requests == [("POST", "/v1/deepagents/agents")]
    state = State.load(tmp_path, endpoint="https://api.invalid")
    assert state.agent_id == "a-safe"


def test_deploy_uses_agent_json_agent_id_with_confirmation_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_project(tmp_path)
    (tmp_path / "agent.json").write_text(
        '{"name": "test-agent", "agent_id": "a-1", "description": "test"}'
    )
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path.endswith("/agents/a-1"):
            return httpx.Response(200, json={"id": "a-1", "name": "remote-agent"})
        if request.method == "PATCH" and request.url.path.endswith("/agents/a-1"):
            return httpx.Response(
                200,
                json={"id": "a-1", "name": "test-agent", "revision": "agent-r2"},
            )
        if request.method == "GET" and request.url.path.endswith("/directories"):
            return httpx.Response(
                200,
                json={
                    "commit_hash": "c1",
                    "files": {
                        "AGENTS.md": {
                            "type": "file",
                            "content": "You are a test agent.\n",
                        },
                    },
                },
            )
        return httpx.Response(500)

    _patch_client(monkeypatch, handler)
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")

    execute_deploy_command(_ns(tmp_path, yes=True))

    assert requests == [
        ("GET", "/v1/deepagents/agents/a-1"),
        ("PATCH", "/v1/deepagents/agents/a-1"),
        ("GET", "/v1/platform/hub/repos/-/a-1/directories"),
    ]
    state = State.load(tmp_path, endpoint="https://api.invalid")
    assert state.agent_id == "a-1"
    assert state.revision == "c1"


def test_deploy_agent_json_agent_id_404_does_not_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_project(tmp_path)
    (tmp_path / "agent.json").write_text(
        '{"name": "test-agent", "agent_id": "missing", "description": "test"}'
    )
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path.endswith("/agents/missing"):
            return httpx.Response(
                404, json={"code": "not_found", "detail": "gone", "status": 404}
            )
        return httpx.Response(500)

    _patch_client(monkeypatch, handler)
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")

    with pytest.raises(SystemExit):
        execute_deploy_command(_ns(tmp_path, yes=True))

    assert requests == [("GET", "/v1/deepagents/agents/missing")]
    state = State.load(tmp_path, endpoint="https://api.invalid")
    assert state.agent_id is None


def test_deploy_reset_with_agent_json_agent_id_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_project(tmp_path)
    (tmp_path / "agent.json").write_text(
        '{"name": "test-agent", "agent_id": "a-1", "description": "test"}'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        msg = f"unexpected request: {request.method} {request.url.path}"
        raise AssertionError(msg)

    _patch_client(monkeypatch, handler)
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")

    with pytest.raises(SystemExit):
        execute_deploy_command(_ns(tmp_path, reset=True, yes=True))


def test_second_deploy_patches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_project(tmp_path)
    State.load(tmp_path, endpoint="https://api.invalid").save(
        agent_id="a-1", revision="r-1"
    )

    requests: list[tuple[str, str]] = []
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.content:
            bodies.append(json.loads(request.content))
        if request.method == "PATCH":
            return httpx.Response(
                200,
                json={"id": "a-1", "name": "test-agent", "revision": "agent-r2"},
            )
        if request.method == "GET" and request.url.path.endswith("/directories"):
            return httpx.Response(
                200,
                json={
                    "commit_hash": "c1",
                    "files": {
                        "AGENTS.md": {"type": "file", "content": "old"},
                        "skills/old/SKILL.md": {
                            "type": "file",
                            "content": "old",
                        },
                    },
                },
            )
        if request.method == "POST" and request.url.path.endswith("/commits"):
            return httpx.Response(201, json={"commit": {"commit_hash": "c2"}})
        return httpx.Response(500)

    _patch_client(monkeypatch, handler)
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")

    execute_deploy_command(_ns(tmp_path))
    assert requests == [
        ("PATCH", "/v1/deepagents/agents/a-1"),
        ("GET", "/v1/platform/hub/repos/-/a-1/directories"),
        ("POST", "/v1/platform/hub/repos/-/a-1/directories/commits"),
    ]
    assert "system_prompt" not in bodies[0]
    assert bodies[1] == {
        "files": {
            "AGENTS.md": {"type": "file", "content": "You are a test agent.\n"},
            "skills/old/SKILL.md": None,
        },
        "parent_commit": "c1",
    }
    state = State.load(tmp_path, endpoint="https://api.invalid")
    assert state.revision == "c2"


def test_deploy_404_falls_back_to_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_project(tmp_path)
    State.load(tmp_path, endpoint="https://api.invalid").save(
        agent_id="stale", revision="old"
    )

    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "PATCH":
            return httpx.Response(
                404, json={"code": "not_found", "detail": "gone", "status": 404}
            )
        return httpx.Response(201, json={"id": "new", "revision": "r-x"})

    _patch_client(monkeypatch, handler)
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")

    execute_deploy_command(_ns(tmp_path))
    assert methods == ["PATCH", "POST"]
    state = State.load(tmp_path, endpoint="https://api.invalid")
    assert state.agent_id == "new"


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
    msg = "no JSON object found in stdout"
    raise AssertionError(msg)


def test_deploy_fails_when_tools_reference_unregistered_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_project(tmp_path)
    (tmp_path / "tools.json").write_text(
        json.dumps(
            {
                "tools": [{"name": "x", "mcp_server_url": "https://missing.example"}],
                "interrupt_config": {},
            }
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/mcp-servers"):
            return httpx.Response(200, json=[])
        return httpx.Response(500)

    _patch_client(monkeypatch, handler)
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")

    with pytest.raises(SystemExit):
        execute_deploy_command(_ns(tmp_path))
    err = capsys.readouterr().out
    assert "  - https://missing.example" in err.splitlines()
    assert "deepagents mcp-servers add" in err


def test_deploy_fails_when_tools_reference_uninvokable_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_project(tmp_path)
    (tmp_path / "tools.json").write_text(
        json.dumps(
            {
                "tools": [{"name": "x", "mcp_server_url": "https://tools.example"}],
                "interrupt_config": {},
            }
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/mcp-servers"):
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "s1",
                        "url": "https://tools.example",
                        "can_invoke": False,
                    }
                ],
            )
        return httpx.Response(500)

    _patch_client(monkeypatch, handler)
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")

    with pytest.raises(SystemExit):
        execute_deploy_command(_ns(tmp_path))
    err = capsys.readouterr().out
    assert "  - https://tools.example" in err.splitlines()
    assert "cannot invoke" in err
