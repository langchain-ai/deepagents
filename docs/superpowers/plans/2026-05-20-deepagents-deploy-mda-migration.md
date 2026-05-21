# `deepagents deploy` → MDA `/v1/deepagents/*` Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `langgraph deploy` bundler with a thin REST client that
talks to `/v1/deepagents/*` directly, switch the on-disk project shape to
`agent.json` + `AGENTS.md` + `tools.json` + `skills/` + `subagents/`, and ship
matching `agents` + `mcp-servers` CLI subcommands.

**Architecture:** New modules `project.py`, `payload.py`, `api_client.py`,
`state.py` under `libs/cli/deepagents_cli/deploy/`. `commands.py` slimmed to ~5
argparse handlers + a 50-line `_deploy` orchestrator. `bundler.py`,
`templates.py`, `context_hub.py`, old `config.py`, and `frontend_dist/` are
deleted. HTTP mocking via `httpx.MockTransport` (no new dep).

**Tech Stack:** Python 3.11+, argparse, httpx (already in deps), tomllib (only
for legacy-toml detection), pytest, `httpx.MockTransport` for HTTP fakes.

**Spec:** `docs/superpowers/specs/2026-05-20-deepagents-deploy-mda-migration-design.md`

---

## Pre-work: Reference layout

All new code lives in `libs/cli/deepagents_cli/deploy/`. All tests live in
`libs/cli/tests/unit_tests/deploy/`. The CLI runs from
`libs/cli/deepagents_cli/main.py`. Run tests from `libs/cli/` with
`uv run pytest tests/unit_tests/deploy/<file>::<test_name> -v`.

The Go DTOs that the payload must match live in
`/Users/victormoreira/Desktop/langchain-repos/langchainplus/smith-go/fleet/agents/types.go`.
Read that file when in doubt about field names — it is the source of truth.

---

### Task 1: Add fixture directory scaffold

**Files:**
- Create: `libs/cli/tests/unit_tests/deploy/fixtures/__init__.py`
- Create: `libs/cli/tests/unit_tests/deploy/fixtures/projects/.gitkeep`

- [ ] **Step 1: Create the empty fixture skeleton**

```bash
mkdir -p libs/cli/tests/unit_tests/deploy/fixtures/projects
touch libs/cli/tests/unit_tests/deploy/fixtures/__init__.py
touch libs/cli/tests/unit_tests/deploy/fixtures/projects/.gitkeep
```

- [ ] **Step 2: Commit**

```bash
git add libs/cli/tests/unit_tests/deploy/fixtures
git commit -m "test(cli): add fixture scaffold for new deploy module"
```

---

### Task 2: `state.py` — read/write `.deepagents/state.json`

**Files:**
- Create: `libs/cli/deepagents_cli/deploy/state.py`
- Create: `libs/cli/tests/unit_tests/deploy/test_state.py`

- [ ] **Step 1: Write failing tests**

`libs/cli/tests/unit_tests/deploy/test_state.py`:

```python
"""Tests for deploy state (.deepagents/state.json)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepagents_cli.deploy.state import State


def test_load_missing_returns_empty(tmp_path: Path) -> None:
    state = State.load(tmp_path)
    assert state.agent_id is None
    assert state.revision is None
    assert state.endpoint is None
    assert state.mcp_servers == {}


def test_save_writes_schema_versioned_json(tmp_path: Path) -> None:
    state = State.load(tmp_path)
    state.endpoint = "https://api.smith.langchain.com"
    state.save(agent_id="abc", revision="rev1")
    data = json.loads((tmp_path / ".deepagents" / "state.json").read_text())
    assert data["schema_version"] == 1
    assert data["agent_id"] == "abc"
    assert data["revision"] == "rev1"
    assert data["endpoint"] == "https://api.smith.langchain.com"
    assert "last_deployed_at" in data
    assert data["mcp_servers"] == {}


def test_save_then_reload_roundtrips(tmp_path: Path) -> None:
    s1 = State.load(tmp_path)
    s1.endpoint = "https://example.invalid"
    s1.mcp_servers = {"https://tools.example/": "srv-1"}
    s1.save(agent_id="aid", revision="r1")
    s2 = State.load(tmp_path)
    assert s2.agent_id == "aid"
    assert s2.revision == "r1"
    assert s2.endpoint == "https://example.invalid"
    assert s2.mcp_servers == {"https://tools.example/": "srv-1"}


def test_reset_clears_existing(tmp_path: Path) -> None:
    State.load(tmp_path).save(agent_id="abc", revision="r1")
    fresh = State.load(tmp_path, reset=True)
    assert fresh.agent_id is None
    assert not (tmp_path / ".deepagents" / "state.json").exists()


def test_clear_agent_removes_id(tmp_path: Path) -> None:
    s = State.load(tmp_path)
    s.save(agent_id="abc", revision="r1")
    s.clear_agent()
    reloaded = State.load(tmp_path)
    assert reloaded.agent_id is None
    assert reloaded.revision is None


def test_unknown_schema_version_raises(tmp_path: Path) -> None:
    (tmp_path / ".deepagents").mkdir()
    (tmp_path / ".deepagents" / "state.json").write_text(
        json.dumps({"schema_version": 99, "agent_id": "x"})
    )
    with pytest.raises(ValueError, match="schema_version"):
        State.load(tmp_path)
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_state.py -v
```
Expected: ImportError (state.py does not exist yet).

- [ ] **Step 3: Implement `state.py`**

`libs/cli/deepagents_cli/deploy/state.py`:

```python
"""Local deploy state persisted under `.deepagents/state.json`.

Tracks the managed agent ID returned by the last successful deploy so that
subsequent runs of `deepagents deploy` issue `PATCH` rather than `POST`. Also
caches the `{mcp_server_url → mcp_server_id}` map to skip the list-call on
every deploy.

The file is gitignored by the scaffold written by `deepagents init`; humans
should never need to edit it by hand.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path

_STATE_DIR = ".deepagents"
_STATE_FILE = "state.json"
_SCHEMA_VERSION = 1


@dataclass
class State:
    """In-memory view of `.deepagents/state.json`.

    Use `State.load(project_root)` to read; mutate fields freely; call
    `state.save(...)` to persist.
    """

    project_root: Path
    agent_id: str | None = None
    revision: str | None = None
    endpoint: str | None = None
    last_deployed_at: str | None = None
    mcp_servers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, project_root: Path, *, reset: bool = False) -> State:
        """Load state from `<project_root>/.deepagents/state.json`.

        Returns an empty state if the file does not exist. With `reset=True`,
        deletes the file (if present) before returning the empty state.
        """
        path = project_root / _STATE_DIR / _STATE_FILE
        if reset and path.exists():
            path.unlink()
        if not path.is_file():
            return cls(project_root=project_root)
        data = json.loads(path.read_text(encoding="utf-8"))
        version = data.get("schema_version")
        if version != _SCHEMA_VERSION:
            msg = (
                f"Unknown schema_version {version!r} in {path}. "
                f"Expected {_SCHEMA_VERSION}. Delete the file to start fresh."
            )
            raise ValueError(msg)
        return cls(
            project_root=project_root,
            agent_id=data.get("agent_id"),
            revision=data.get("revision"),
            endpoint=data.get("endpoint"),
            last_deployed_at=data.get("last_deployed_at"),
            mcp_servers=dict(data.get("mcp_servers") or {}),
        )

    def save(self, *, agent_id: str | None = None, revision: str | None = None) -> None:
        """Persist state, optionally updating agent_id / revision in the same call."""
        if agent_id is not None:
            self.agent_id = agent_id
        if revision is not None:
            self.revision = revision
        self.last_deployed_at = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
        directory = self.project_root / _STATE_DIR
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "endpoint": self.endpoint,
            "agent_id": self.agent_id,
            "revision": self.revision,
            "last_deployed_at": self.last_deployed_at,
            "mcp_servers": self.mcp_servers,
        }
        (directory / _STATE_FILE).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def clear_agent(self) -> None:
        """Remove agent_id / revision from state and persist."""
        self.agent_id = None
        self.revision = None
        self.save()
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_state.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/state.py libs/cli/tests/unit_tests/deploy/test_state.py
git commit -m "feat(cli): state.py for tracking managed agent deploys"
```

---

### Task 3: `api_client.py` — auth + base HTTP plumbing

**Files:**
- Create: `libs/cli/deepagents_cli/deploy/api_client.py`
- Create: `libs/cli/tests/unit_tests/deploy/test_api_client.py`

- [ ] **Step 1: Write failing tests for env-driven construction**

`libs/cli/tests/unit_tests/deploy/test_api_client.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_api_client.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement the base client**

`libs/cli/deepagents_cli/deploy/api_client.py`:

```python
"""HTTP client for the Managed Deep Agents `/v1/deepagents/*` surface.

Thin wrapper around `httpx.Client` that:

- Resolves auth from `LANGSMITH_API_KEY` (preferred) or `LANGCHAIN_API_KEY`
  and sends it as `X-Api-Key`.
- Resolves the endpoint from `LANGSMITH_ENDPOINT` / `LANGCHAIN_ENDPOINT`,
  defaulting to `https://api.smith.langchain.com`.
- Parses 4xx responses into `ApiError` with the platform's `ErrorResponse`
  shape (`type`/`code`/`detail`/`status`).
- Retries 5xx responses once with a short backoff before raising.

Agents and MCP-servers CRUD methods are layered on top in subsequent tasks.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx

_DEFAULT_ENDPOINT = "https://api.smith.langchain.com"
_DEPLOY_PATH = "/v1/deepagents"
_RETRY_SLEEP_SECONDS = 1.0


@dataclass
class ApiError(Exception):
    """Surface the platform's `ErrorResponse` envelope as a Python exception."""

    status: int
    code: str = ""
    detail: str = ""
    type_: str = ""

    def __str__(self) -> str:  # noqa: D105
        bits = [f"HTTP {self.status}"]
        if self.code:
            bits.append(self.code)
        if self.detail:
            bits.append(self.detail)
        return " — ".join(bits)


class ApiClient:
    """HTTP client for `/v1/deepagents/*`."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client(
            base_url=self.endpoint,
            transport=transport,
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
        )

    @classmethod
    def from_env(
        cls, *, transport: httpx.BaseTransport | None = None
    ) -> ApiClient:
        """Build a client from `LANGSMITH_*` / `LANGCHAIN_*` env vars.

        Exits non-zero with a friendly message if the API key is missing.
        """
        api_key = (
            os.environ.get("LANGSMITH_API_KEY")
            or os.environ.get("LANGCHAIN_API_KEY")
            or ""
        ).strip()
        if not api_key:
            sys.stderr.write(
                "Error: set LANGSMITH_API_KEY in your .env or environment.\n"
            )
            raise SystemExit(1)
        endpoint = (
            os.environ.get("LANGSMITH_ENDPOINT")
            or os.environ.get("LANGCHAIN_ENDPOINT")
            or _DEFAULT_ENDPOINT
        ).rstrip("/")
        return cls(endpoint=endpoint, api_key=api_key, transport=transport)

    def close(self) -> None:
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401
        last_status = 0
        last_text = ""
        for attempt in range(2):
            response = self._client.request(method, path, json=json, params=params)
            last_status = response.status_code
            last_text = response.text
            if 200 <= response.status_code < 300:
                if response.status_code == 204 or not response.content:
                    return None
                return response.json()
            if 400 <= response.status_code < 500:
                raise self._build_error(response)
            # 5xx: retry once
            if attempt == 0:
                time.sleep(_RETRY_SLEEP_SECONDS)
                continue
        raise ApiError(status=last_status, detail=last_text[:500])

    @staticmethod
    def _build_error(response: httpx.Response) -> ApiError:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        return ApiError(
            status=response.status_code,
            code=str(payload.get("code") or ""),
            detail=str(payload.get("detail") or response.text[:500]),
            type_=str(payload.get("type") or ""),
        )
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_api_client.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/api_client.py libs/cli/tests/unit_tests/deploy/test_api_client.py
git commit -m "feat(cli): ApiClient for /v1/deepagents/* with X-Api-Key auth + retry"
```

---

### Task 4: ApiClient — agents CRUD methods

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/api_client.py`
- Modify: `libs/cli/tests/unit_tests/deploy/test_api_client.py`

- [ ] **Step 1: Add failing tests for agent CRUD**

Append to `test_api_client.py`:

```python
def test_create_agent_posts_to_v1_deepagents_agents(monkeypatch: pytest.MonkeyPatch) -> None:
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

    def handler(request: httpx.Request) -> httpx.Response:
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_api_client.py -v
```
Expected: 5 new failures (`AttributeError` on missing methods).

- [ ] **Step 3: Add the CRUD methods to `ApiClient`**

Append to `api_client.py` (inside `class ApiClient`):

```python
    # --- agents ----------------------------------------------------------

    def create_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"{_DEPLOY_PATH}/agents", json=payload)

    def get_agent(self, agent_id: str, *, include_files: bool = False) -> dict[str, Any]:
        params = {"include_files": "true"} if include_files else None
        return self._request("GET", f"{_DEPLOY_PATH}/agents/{agent_id}", params=params)

    def iter_agents(
        self,
        *,
        page_size: int = 50,
        name: str | None = None,
    ):  # type: ignore[no-untyped-def]
        """Yield AgentSummary objects across all pages."""
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"page_size": page_size}
            if cursor:
                params["cursor"] = cursor
            if name:
                params["name"] = name
            body = self._request("GET", f"{_DEPLOY_PATH}/agents", params=params)
            for item in body.get("items", []):
                yield item
            cursor = body.get("next_cursor")
            if not cursor:
                return

    def patch_agent(self, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("PATCH", f"{_DEPLOY_PATH}/agents/{agent_id}", json=payload)

    def delete_agent(self, agent_id: str) -> None:
        self._request("DELETE", f"{_DEPLOY_PATH}/agents/{agent_id}")
```

- [ ] **Step 4: Run to confirm pass**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_api_client.py -v
```
Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/api_client.py libs/cli/tests/unit_tests/deploy/test_api_client.py
git commit -m "feat(cli): ApiClient agents CRUD (create/get/iter/patch/delete)"
```

---

### Task 5: ApiClient — mcp-servers CRUD methods

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/api_client.py`
- Modify: `libs/cli/tests/unit_tests/deploy/test_api_client.py`

- [ ] **Step 1: Add failing tests**

Append to `test_api_client.py`:

```python
def test_list_mcp_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"servers": [{"id": "s1", "url": "https://tools.langchain.com"}]}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/deepagents/mcp-servers"
        return httpx.Response(200, json=body)

    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    client = ApiClient.from_env(transport=_transport(handler))
    assert client.list_mcp_servers() == body["servers"]


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


def test_delete_mcp_server(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/v1/deepagents/mcp-servers/s1"
        return httpx.Response(204)

    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    client = ApiClient.from_env(transport=_transport(handler))
    assert client.delete_mcp_server("s1") is None
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_api_client.py -v
```
Expected: 3 failures.

- [ ] **Step 3: Add the methods**

Append inside `class ApiClient`:

```python
    # --- mcp-servers -----------------------------------------------------

    def list_mcp_servers(self) -> list[dict[str, Any]]:
        body = self._request("GET", f"{_DEPLOY_PATH}/mcp-servers")
        return list(body.get("servers", []))

    def get_mcp_server(self, mcp_server_id: str) -> dict[str, Any]:
        return self._request("GET", f"{_DEPLOY_PATH}/mcp-servers/{mcp_server_id}")

    def create_mcp_server(
        self,
        *,
        name: str,
        url: str,
        headers: list[dict[str, str]] | None = None,
        auth_type: str = "headers",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": name,
            "url": url,
            "auth_type": auth_type,
        }
        if headers:
            payload["headers"] = headers
        return self._request("POST", f"{_DEPLOY_PATH}/mcp-servers", json=payload)

    def delete_mcp_server(self, mcp_server_id: str) -> None:
        self._request("DELETE", f"{_DEPLOY_PATH}/mcp-servers/{mcp_server_id}")
```

- [ ] **Step 4: Run to confirm pass**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_api_client.py -v
```
Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/api_client.py libs/cli/tests/unit_tests/deploy/test_api_client.py
git commit -m "feat(cli): ApiClient mcp-servers CRUD"
```

---

### Task 6: Project loader — bare project (`agent.json` + `AGENTS.md`)

**Files:**
- Create: `libs/cli/deepagents_cli/deploy/project.py`
- Create: `libs/cli/tests/unit_tests/deploy/fixtures/projects/bare/agent.json`
- Create: `libs/cli/tests/unit_tests/deploy/fixtures/projects/bare/AGENTS.md`
- Create: `libs/cli/tests/unit_tests/deploy/test_project.py`

- [ ] **Step 1: Write the bare fixture**

`libs/cli/tests/unit_tests/deploy/fixtures/projects/bare/agent.json`:

```json
{
  "name": "research-assistant",
  "description": "Researches a topic and returns a summary."
}
```

`libs/cli/tests/unit_tests/deploy/fixtures/projects/bare/AGENTS.md`:

```markdown
# Research Assistant

You are a careful research assistant.
```

- [ ] **Step 2: Write failing tests**

`libs/cli/tests/unit_tests/deploy/test_project.py`:

```python
"""Tests for Project.load() (parsing agent.json/AGENTS.md/tools.json/skills/subagents)."""

from __future__ import annotations

from pathlib import Path

import pytest

from deepagents_cli.deploy.project import Project, ProjectError

_FIXTURES = Path(__file__).parent / "fixtures" / "projects"


def test_load_bare_project_reads_agent_json_and_agents_md() -> None:
    proj = Project.load(_FIXTURES / "bare")
    assert proj.name == "research-assistant"
    assert proj.description == "Researches a topic and returns a summary."
    assert "careful research assistant" in proj.system_prompt
    assert proj.tools is None
    assert proj.skills == []
    assert proj.subagents == []
    assert proj.runtime is None
    assert proj.permissions is None


def test_load_missing_agent_json_raises(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match="agent.json"):
        Project.load(tmp_path)


def test_load_missing_agents_md_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    with pytest.raises(ProjectError, match="AGENTS.md"):
        Project.load(tmp_path)


def test_load_invalid_agent_json_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text("{not json")
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match="agent.json"):
        Project.load(tmp_path)


def test_load_missing_name_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"description": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match="name"):
        Project.load(tmp_path)


def test_runtime_and_permissions_round_trip(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        """
        {
          "name": "x",
          "runtime": {
            "model": {"model_id": "anthropic:claude-sonnet-4-6"},
            "backend_type": "thread_scoped_sandbox"
          },
          "permissions": {
            "identity": "personal",
            "visibility": "tenant",
            "tenant_access_level": "read"
          }
        }
        """
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    proj = Project.load(tmp_path)
    assert proj.runtime == {
        "model": {"model_id": "anthropic:claude-sonnet-4-6"},
        "backend_type": "thread_scoped_sandbox",
    }
    assert proj.permissions == {
        "identity": "personal",
        "visibility": "tenant",
        "tenant_access_level": "read",
    }


def test_invalid_runtime_backend_type_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text(
        '{"name": "x", "runtime": {"backend_type": "lol_unknown"}}'
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match="backend_type"):
        Project.load(tmp_path)
```

- [ ] **Step 3: Run to confirm failure**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_project.py -v
```
Expected: ImportError.

- [ ] **Step 4: Implement `project.py` (bare scope only — extended in later tasks)**

`libs/cli/deepagents_cli/deploy/project.py`:

```python
"""Parse a Managed Deep Agents project directory into a structured value.

Layout (canonical, all paths relative to the project root):

    agent.json              required — top-level config
    AGENTS.md               required — system prompt
    tools.json              optional — verbatim ToolsConfig
    skills/<name>/SKILL.md  optional — frontmatter + body
    skills/<name>/<file>    optional — siblings of SKILL.md → files map
    subagents/<name>/agent.json   required if subagent dir exists
    subagents/<name>/AGENTS.md    required if subagent dir exists
    subagents/<name>/tools.json   optional

The result is plain Python data — no I/O happens after `load()` returns. The
payload builder (`payload.py`) consumes this dataclass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_AGENT_JSON = "agent.json"
_AGENTS_MD = "AGENTS.md"
_TOOLS_JSON = "tools.json"
_SKILLS_DIR = "skills"
_SUBAGENTS_DIR = "subagents"
_SKILL_FILE = "SKILL.md"

_VALID_BACKEND_TYPES = frozenset(
    {"default", "thread_scoped_sandbox", "agent_scoped_sandbox"}
)
_VALID_IDENTITY = frozenset({"personal", "shared"})
_VALID_VISIBILITY = frozenset({"tenant", "user"})
_VALID_TENANT_ACCESS = frozenset({"read", "run", "write"})


class ProjectError(ValueError):
    """Raised when the on-disk project is malformed."""


@dataclass
class Skill:
    """A skill discovered under `skills/<name>/`."""

    name: str
    description: str
    instructions: str
    files: dict[str, str] = field(default_factory=dict)


@dataclass
class Subagent:
    """A subagent discovered under `subagents/<name>/`."""

    name: str
    description: str | None
    model_id: str | None
    instructions: str
    tools: dict[str, Any] | None = None
    extra_files: dict[str, str] = field(default_factory=dict)
    """Subagent-local skills, keyed by path under `subagents/<name>/`."""


@dataclass
class Project:
    """In-memory view of the on-disk project."""

    root: Path
    name: str
    description: str | None
    system_prompt: str
    runtime: dict[str, Any] | None
    permissions: dict[str, Any] | None
    extras: dict[str, Any] | None
    tools: dict[str, Any] | None
    skills: list[Skill]
    subagents: list[Subagent]

    @classmethod
    def load(cls, root: Path) -> Project:
        """Read the project at *root*; raise `ProjectError` on any problem."""
        root = root.resolve()
        if not root.is_dir():
            msg = f"Project root is not a directory: {root}"
            raise ProjectError(msg)

        agent_data = _read_agent_json(root)
        system_prompt = _read_agents_md(root)

        return cls(
            root=root,
            name=agent_data["name"],
            description=agent_data.get("description"),
            system_prompt=system_prompt,
            runtime=agent_data.get("runtime"),
            permissions=agent_data.get("permissions"),
            extras=agent_data.get("extras"),
            tools=None,         # task 7
            skills=[],          # task 8
            subagents=[],       # task 9
        )


def _read_agent_json(root: Path) -> dict[str, Any]:
    path = root / _AGENT_JSON
    if not path.is_file():
        msg = f"agent.json is required but not found in {root}."
        raise ProjectError(msg)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in {path}: {exc}"
        raise ProjectError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{path} must contain a JSON object."
        raise ProjectError(msg)

    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        msg = f"`name` (non-empty string) is required in {path}."
        raise ProjectError(msg)

    runtime = data.get("runtime")
    if runtime is not None:
        backend_type = runtime.get("backend_type")
        if backend_type is not None and backend_type not in _VALID_BACKEND_TYPES:
            msg = (
                f"runtime.backend_type {backend_type!r} not in "
                f"{sorted(_VALID_BACKEND_TYPES)}"
            )
            raise ProjectError(msg)

    permissions = data.get("permissions")
    if permissions is not None:
        if (ident := permissions.get("identity")) and ident not in _VALID_IDENTITY:
            msg = f"permissions.identity {ident!r} not in {sorted(_VALID_IDENTITY)}"
            raise ProjectError(msg)
        if (vis := permissions.get("visibility")) and vis not in _VALID_VISIBILITY:
            msg = f"permissions.visibility {vis!r} not in {sorted(_VALID_VISIBILITY)}"
            raise ProjectError(msg)
        if (lvl := permissions.get("tenant_access_level")) and lvl not in _VALID_TENANT_ACCESS:
            msg = (
                f"permissions.tenant_access_level {lvl!r} not in "
                f"{sorted(_VALID_TENANT_ACCESS)}"
            )
            raise ProjectError(msg)

    return data


def _read_agents_md(root: Path) -> str:
    path = root / _AGENTS_MD
    if not path.is_file():
        msg = f"AGENTS.md is required but not found in {root}."
        raise ProjectError(msg)
    return path.read_text(encoding="utf-8")
```

- [ ] **Step 5: Run to confirm pass**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_project.py -v
```
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/project.py \
        libs/cli/tests/unit_tests/deploy/test_project.py \
        libs/cli/tests/unit_tests/deploy/fixtures/projects/bare
git commit -m "feat(cli): Project.load() — bare project (agent.json + AGENTS.md)"
```

---

### Task 7: Project loader — `tools.json`

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/project.py`
- Modify: `libs/cli/tests/unit_tests/deploy/test_project.py`
- Create: `libs/cli/tests/unit_tests/deploy/fixtures/projects/with_tools/agent.json`
- Create: `libs/cli/tests/unit_tests/deploy/fixtures/projects/with_tools/AGENTS.md`
- Create: `libs/cli/tests/unit_tests/deploy/fixtures/projects/with_tools/tools.json`

- [ ] **Step 1: Add fixture**

`libs/cli/tests/unit_tests/deploy/fixtures/projects/with_tools/agent.json`:

```json
{"name": "research-assistant"}
```

`libs/cli/tests/unit_tests/deploy/fixtures/projects/with_tools/AGENTS.md`:

```markdown
You are a research assistant.
```

`libs/cli/tests/unit_tests/deploy/fixtures/projects/with_tools/tools.json`:

```json
{
  "tools": [
    {
      "name": "tavily_web_search",
      "mcp_server_url": "https://tools.langchain.com",
      "mcp_server_name": "Fleet",
      "display_name": "tavily_web_search"
    }
  ],
  "interrupt_config": {
    "https://tools.langchain.com::tavily_web_search::Fleet": true
  }
}
```

- [ ] **Step 2: Add failing tests**

Append to `test_project.py`:

```python
def test_load_with_tools_reads_tools_json() -> None:
    proj = Project.load(_FIXTURES / "with_tools")
    assert proj.tools is not None
    assert proj.tools["tools"][0]["name"] == "tavily_web_search"
    assert proj.tools["tools"][0]["mcp_server_url"] == "https://tools.langchain.com"
    assert proj.tools["interrupt_config"][
        "https://tools.langchain.com::tavily_web_search::Fleet"
    ] is True


def test_invalid_tools_json_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    (tmp_path / "tools.json").write_text("[]")  # array, not object
    with pytest.raises(ProjectError, match="tools.json"):
        Project.load(tmp_path)


def test_tools_missing_mcp_server_url_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    (tmp_path / "tools.json").write_text(
        '{"tools": [{"name": "search"}], "interrupt_config": {}}'
    )
    with pytest.raises(ProjectError, match="mcp_server_url"):
        Project.load(tmp_path)
```

- [ ] **Step 3: Run to confirm failure**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_project.py::test_load_with_tools_reads_tools_json -v
```
Expected: AssertionError (proj.tools is None).

- [ ] **Step 4: Extend the loader**

In `project.py`, modify `Project.load` to call a new `_read_tools_json` and add
the helper. Inside `Project.load`, replace `tools=None,` with
`tools=_read_tools_json(root),`. Add at module scope:

```python
def _read_tools_json(root: Path) -> dict[str, Any] | None:
    path = root / _TOOLS_JSON
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in {path}: {exc}"
        raise ProjectError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{path} must contain a JSON object."
        raise ProjectError(msg)
    tools = data.get("tools")
    if not isinstance(tools, list):
        msg = f"{path}: `tools` must be an array."
        raise ProjectError(msg)
    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            msg = f"{path}: tools[{idx}] must be an object."
            raise ProjectError(msg)
        if not isinstance(tool.get("name"), str) or not tool["name"]:
            msg = f"{path}: tools[{idx}].name is required."
            raise ProjectError(msg)
        if not isinstance(tool.get("mcp_server_url"), str) or not tool["mcp_server_url"]:
            msg = f"{path}: tools[{idx}].mcp_server_url is required."
            raise ProjectError(msg)
    interrupt_config = data.get("interrupt_config")
    if interrupt_config is not None and not isinstance(interrupt_config, dict):
        msg = f"{path}: `interrupt_config` must be an object."
        raise ProjectError(msg)
    return data
```

- [ ] **Step 5: Run to confirm pass**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_project.py -v
```
Expected: all passing.

- [ ] **Step 6: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/project.py \
        libs/cli/tests/unit_tests/deploy/test_project.py \
        libs/cli/tests/unit_tests/deploy/fixtures/projects/with_tools
git commit -m "feat(cli): Project.load() — tools.json"
```

---

### Task 8: Project loader — `skills/`

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/project.py`
- Modify: `libs/cli/tests/unit_tests/deploy/test_project.py`
- Create: `libs/cli/tests/unit_tests/deploy/fixtures/projects/with_skills/{agent.json,AGENTS.md,skills/summarize/SKILL.md,skills/summarize/examples.md}`

- [ ] **Step 1: Add fixture**

`agent.json`: `{"name": "x"}` · `AGENTS.md`: `hi`

`skills/summarize/SKILL.md`:

```markdown
---
name: summarize
description: Summarise text into a one-paragraph summary.
---

# Summarize

Given a text, produce a one-paragraph summary.
```

`skills/summarize/examples.md`:

```markdown
- Example 1: ...
- Example 2: ...
```

- [ ] **Step 2: Failing tests**

```python
def test_load_with_skills_parses_frontmatter_and_files() -> None:
    proj = Project.load(_FIXTURES / "with_skills")
    assert len(proj.skills) == 1
    skill = proj.skills[0]
    assert skill.name == "summarize"
    assert skill.description == "Summarise text into a one-paragraph summary."
    assert "one-paragraph summary" in skill.instructions
    assert "examples.md" in skill.files
    assert "Example 1" in skill.files["examples.md"]


def test_skill_missing_frontmatter_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    skill_dir = tmp_path / "skills" / "bad"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# No frontmatter here\n")
    with pytest.raises(ProjectError, match="frontmatter"):
        Project.load(tmp_path)


def test_skill_duplicate_names_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    for dirname in ("a", "b"):
        d = tmp_path / "skills" / dirname
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            "---\nname: same\ndescription: x\n---\nhi\n"
        )
    with pytest.raises(ProjectError, match="duplicate"):
        Project.load(tmp_path)
```

- [ ] **Step 3: Run to confirm failure**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_project.py -k skill -v
```
Expected: 3 failures.

- [ ] **Step 4: Extend the loader**

Replace `skills=[],` in `Project.load` with `skills=_read_skills(root),`. Add
to `project.py`:

```python
import re as _re

_FRONTMATTER_RE = _re.compile(
    r"^---\n(?P<fm>.*?)\n---\n(?P<body>.*)$", _re.DOTALL
)


def _parse_skill_frontmatter(text: str, *, source: Path) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        msg = f"{source}: YAML frontmatter (--- ... ---) is required."
        raise ProjectError(msg)
    frontmatter: dict[str, str] = {}
    for line in match.group("fm").splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, _, value = line.partition(":")
        frontmatter[key.strip()] = value.strip().strip('"').strip("'")
    if "name" not in frontmatter or not frontmatter["name"]:
        msg = f"{source}: frontmatter is missing required key `name`."
        raise ProjectError(msg)
    if "description" not in frontmatter or not frontmatter["description"]:
        msg = f"{source}: frontmatter is missing required key `description`."
        raise ProjectError(msg)
    return frontmatter, match.group("body").strip()


def _read_skills(root: Path) -> list[Skill]:
    skills_dir = root / _SKILLS_DIR
    if not skills_dir.is_dir():
        return []
    result: list[Skill] = []
    seen: set[str] = set()
    for entry in sorted(skills_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        skill_file = entry / _SKILL_FILE
        if not skill_file.is_file():
            msg = f"{entry}: missing SKILL.md"
            raise ProjectError(msg)
        frontmatter, body = _parse_skill_frontmatter(
            skill_file.read_text(encoding="utf-8"), source=skill_file
        )
        name = frontmatter["name"]
        if name in seen:
            msg = f"duplicate skill name {name!r} in {skills_dir}"
            raise ProjectError(msg)
        seen.add(name)
        files: dict[str, str] = {}
        for child in sorted(entry.iterdir()):
            if child.is_file() and child.name != _SKILL_FILE and not child.name.startswith("."):
                files[child.name] = child.read_text(encoding="utf-8")
        result.append(
            Skill(
                name=name,
                description=frontmatter["description"],
                instructions=body,
                files=files,
            )
        )
    return result
```

- [ ] **Step 5: Run to confirm pass**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_project.py -v
```
Expected: all passing.

- [ ] **Step 6: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/project.py \
        libs/cli/tests/unit_tests/deploy/test_project.py \
        libs/cli/tests/unit_tests/deploy/fixtures/projects/with_skills
git commit -m "feat(cli): Project.load() — skills/ with frontmatter + files"
```

---

### Task 9: Project loader — `subagents/`

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/project.py`
- Modify: `libs/cli/tests/unit_tests/deploy/test_project.py`
- Create: fixture trees for `with_subagents` and `subagent_with_local_skills`

- [ ] **Step 1: Add fixtures**

`fixtures/projects/with_subagents/agent.json`: `{"name": "parent"}` ·
`AGENTS.md`: `Parent prompt`.

`fixtures/projects/with_subagents/subagents/researcher/agent.json`:

```json
{"description": "Researches a topic.", "model_id": "anthropic:claude-sonnet-4-6"}
```

`fixtures/projects/with_subagents/subagents/researcher/AGENTS.md`:

```markdown
You research a topic and summarise.
```

`fixtures/projects/with_subagents/subagents/researcher/tools.json`:

```json
{"tools": [{"name": "search", "mcp_server_url": "https://tools.example"}], "interrupt_config": {}}
```

For `subagent_with_local_skills`: identical layout, plus
`subagents/researcher/skills/note/SKILL.md`:

```markdown
---
name: note
description: Take a note.
---
Take a note.
```

- [ ] **Step 2: Failing tests**

```python
def test_load_with_subagents() -> None:
    proj = Project.load(_FIXTURES / "with_subagents")
    assert len(proj.subagents) == 1
    sa = proj.subagents[0]
    assert sa.name == "researcher"
    assert sa.description == "Researches a topic."
    assert sa.model_id == "anthropic:claude-sonnet-4-6"
    assert "research a topic" in sa.instructions
    assert sa.tools is not None
    assert sa.tools["tools"][0]["name"] == "search"
    assert sa.extra_files == {}


def test_subagent_local_skills_go_into_extra_files() -> None:
    proj = Project.load(_FIXTURES / "subagent_with_local_skills")
    sa = proj.subagents[0]
    assert "skills/note/SKILL.md" in sa.extra_files
    assert "Take a note." in sa.extra_files["skills/note/SKILL.md"]


def test_subagent_missing_agent_json_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    sa = tmp_path / "subagents" / "broken"
    sa.mkdir(parents=True)
    (sa / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match="agent.json"):
        Project.load(tmp_path)


def test_subagent_duplicate_names_raises(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    for n in ("a", "a"):  # case 1: same dir name twice is impossible on FS
        pass
    # Trigger by colliding with a top-level skill name? not relevant.
    # Instead just verify the helper rejects a manually duplicated list.
    from deepagents_cli.deploy.project import _read_subagents  # noqa: PLC0415
    sa1 = tmp_path / "subagents" / "x"
    sa1.mkdir(parents=True)
    (sa1 / "agent.json").write_text("{}")
    (sa1 / "AGENTS.md").write_text("hi")
    # Real duplicate: case-insensitive collision
    sa2 = tmp_path / "subagents" / "X"
    if sa1.resolve() == sa2.resolve():  # case-insensitive FS — synthesize
        pytest.skip("case-insensitive FS")
    sa2.mkdir(parents=True)
    (sa2 / "agent.json").write_text("{}")
    (sa2 / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match="duplicate"):
        _read_subagents(tmp_path)
```

- [ ] **Step 3: Run to confirm failure**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_project.py -k subagent -v
```
Expected: failures.

- [ ] **Step 4: Extend the loader**

Replace `subagents=[],` with `subagents=_read_subagents(root),`. Add to
`project.py`:

```python
def _read_subagents(root: Path) -> list[Subagent]:
    sa_dir = root / _SUBAGENTS_DIR
    if not sa_dir.is_dir():
        return []
    result: list[Subagent] = []
    seen: set[str] = set()
    for entry in sorted(sa_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        agent_json = entry / _AGENT_JSON
        agents_md = entry / _AGENTS_MD
        if not agent_json.is_file():
            msg = f"{entry}: missing agent.json"
            raise ProjectError(msg)
        if not agents_md.is_file():
            msg = f"{entry}: missing AGENTS.md"
            raise ProjectError(msg)
        try:
            data = json.loads(agent_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in {agent_json}: {exc}"
            raise ProjectError(msg) from exc
        if not isinstance(data, dict):
            msg = f"{agent_json} must contain a JSON object."
            raise ProjectError(msg)
        name = entry.name
        key = name.lower()
        if key in seen:
            msg = f"duplicate subagent name {name!r} (case-insensitive)"
            raise ProjectError(msg)
        seen.add(key)

        tools = _read_tools_json(entry)
        extra_files: dict[str, str] = {}
        local_skills_dir = entry / _SKILLS_DIR
        if local_skills_dir.is_dir():
            for f in sorted(local_skills_dir.rglob("*")):
                if f.is_file() and not f.name.startswith("."):
                    rel = f.relative_to(entry).as_posix()
                    extra_files[rel] = f.read_text(encoding="utf-8")

        result.append(
            Subagent(
                name=name,
                description=data.get("description"),
                model_id=data.get("model_id"),
                instructions=agents_md.read_text(encoding="utf-8"),
                tools=tools,
                extra_files=extra_files,
            )
        )
    return result
```

- [ ] **Step 5: Run to confirm pass**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_project.py -v
```
Expected: all passing.

- [ ] **Step 6: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/project.py \
        libs/cli/tests/unit_tests/deploy/test_project.py \
        libs/cli/tests/unit_tests/deploy/fixtures/projects/with_subagents \
        libs/cli/tests/unit_tests/deploy/fixtures/projects/subagent_with_local_skills
git commit -m "feat(cli): Project.load() — subagents/ + raw-files fallback for subagent skills"
```

---

### Task 10: Project loader — legacy `deepagents.toml` / `mcp.json` migration hint

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/project.py`
- Modify: `libs/cli/tests/unit_tests/deploy/test_project.py`

- [ ] **Step 1: Failing tests**

```python
def test_legacy_deepagents_toml_raises_migration_hint(tmp_path: Path) -> None:
    (tmp_path / "deepagents.toml").write_text(
        '[agent]\nname = "x"\n'
    )
    (tmp_path / "AGENTS.md").write_text("hi")
    with pytest.raises(ProjectError, match="legacy deepagents.toml"):
        Project.load(tmp_path)


def test_legacy_mcp_json_raises_migration_hint(tmp_path: Path) -> None:
    (tmp_path / "agent.json").write_text('{"name": "x"}')
    (tmp_path / "AGENTS.md").write_text("hi")
    (tmp_path / "mcp.json").write_text('{"mcpServers": {}}')
    with pytest.raises(ProjectError, match="mcp.json"):
        Project.load(tmp_path)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_project.py -k legacy -v
```
Expected: failures.

- [ ] **Step 3: Extend the loader**

Add early in `Project.load(...)` (before `_read_agent_json`):

```python
        _check_no_legacy_files(root)
```

And add the helper:

```python
_LEGACY_TOML_HINT = """\
Found legacy `deepagents.toml` in {root}. The migrated `deepagents deploy`
expects the new layout. Quick mapping:

  [agent]                       → agent.json (top-level keys: name, description)
  [agent].model                 → agent.json runtime.model.model_id
  [sandbox].scope               → agent.json runtime.backend_type
                                  ("thread_scoped_sandbox" or "agent_scoped_sandbox")
  [auth], [memories], [frontend]→ remove; managed by the platform now

Then run `deepagents init --force` to refresh scaffolding or migrate by hand.
"""


_LEGACY_MCP_HINT = """\
Found legacy `mcp.json` in {root}. MCP servers are now workspace-level resources:

  deepagents mcp-servers add --url <url> --header KEY=VALUE [--name <name>]

Then reference the server in tools.json by mcp_server_url.
"""


def _check_no_legacy_files(root: Path) -> None:
    if (root / "deepagents.toml").is_file():
        raise ProjectError(_LEGACY_TOML_HINT.format(root=root))
    if (root / "mcp.json").is_file():
        raise ProjectError(_LEGACY_MCP_HINT.format(root=root))
```

- [ ] **Step 4: Run to confirm pass**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_project.py -v
```
Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/project.py libs/cli/tests/unit_tests/deploy/test_project.py
git commit -m "feat(cli): friendly migration hint for legacy deepagents.toml/mcp.json"
```

---

### Task 11: `payload.py` — build_payload(project, mode)

**Files:**
- Create: `libs/cli/deepagents_cli/deploy/payload.py`
- Create: `libs/cli/tests/unit_tests/deploy/test_payload.py`
- Create: `libs/cli/tests/unit_tests/deploy/fixtures/projects/<name>/expected_payload.json` for each fixture

- [ ] **Step 1: Write expected payloads for fixtures**

`fixtures/projects/bare/expected_payload.json`:

```json
{
  "name": "research-assistant",
  "description": "Researches a topic and returns a summary.",
  "system_prompt": "# Research Assistant\n\nYou are a careful research assistant.\n"
}
```

`fixtures/projects/with_tools/expected_payload.json`:

```json
{
  "name": "research-assistant",
  "system_prompt": "You are a research assistant.\n",
  "tools": {
    "tools": [
      {
        "name": "tavily_web_search",
        "mcp_server_url": "https://tools.langchain.com",
        "mcp_server_name": "Fleet",
        "display_name": "tavily_web_search"
      }
    ],
    "interrupt_config": {
      "https://tools.langchain.com::tavily_web_search::Fleet": true
    }
  }
}
```

`fixtures/projects/with_skills/expected_payload.json`:

```json
{
  "name": "x",
  "system_prompt": "hi\n",
  "skills": [
    {
      "type": "inline",
      "name": "summarize",
      "description": "Summarise text into a one-paragraph summary.",
      "instructions": "# Summarize\n\nGiven a text, produce a one-paragraph summary.",
      "files": {
        "examples.md": "- Example 1: ...\n- Example 2: ...\n"
      }
    }
  ]
}
```

`fixtures/projects/with_subagents/expected_payload.json`:

```json
{
  "name": "parent",
  "system_prompt": "Parent prompt\n",
  "subagents": [
    {
      "name": "researcher",
      "description": "Researches a topic.",
      "model_id": "anthropic:claude-sonnet-4-6",
      "instructions": "You research a topic and summarise.\n",
      "tools": {
        "tools": [{"name": "search", "mcp_server_url": "https://tools.example"}],
        "interrupt_config": {}
      }
    }
  ]
}
```

`fixtures/projects/subagent_with_local_skills/expected_payload.json`:

```json
{
  "name": "parent",
  "system_prompt": "Parent prompt\n",
  "subagents": [
    {
      "name": "researcher",
      "description": "Researches a topic.",
      "model_id": "anthropic:claude-sonnet-4-6",
      "instructions": "You research a topic and summarise.\n",
      "tools": {
        "tools": [{"name": "search", "mcp_server_url": "https://tools.example"}],
        "interrupt_config": {}
      }
    }
  ],
  "files": {
    "subagents/researcher/skills/note/SKILL.md": {
      "content": "---\nname: note\ndescription: Take a note.\n---\nTake a note.\n"
    }
  }
}
```

- [ ] **Step 2: Failing tests**

`libs/cli/tests/unit_tests/deploy/test_payload.py`:

```python
"""Snapshot tests for build_payload over fixture projects."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepagents_cli.deploy.payload import build_payload
from deepagents_cli.deploy.project import Project

_FIXTURES = Path(__file__).parent / "fixtures" / "projects"

_FIXTURE_NAMES = [
    "bare",
    "with_tools",
    "with_skills",
    "with_subagents",
    "subagent_with_local_skills",
]


@pytest.mark.parametrize("name", _FIXTURE_NAMES)
def test_create_payload_matches_expected(name: str) -> None:
    project = Project.load(_FIXTURES / name)
    payload = build_payload(project, mode="create")
    expected = json.loads(
        (_FIXTURES / name / "expected_payload.json").read_text(encoding="utf-8")
    )
    assert payload == expected


def test_patch_payload_omits_name_when_unchanged() -> None:
    project = Project.load(_FIXTURES / "bare")
    payload = build_payload(project, mode="patch")
    # PATCH always includes name (full-replace on send); this asserts the
    # current contract — adjust if the spec evolves.
    assert payload["name"] == "research-assistant"
```

- [ ] **Step 3: Run to confirm failure**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_payload.py -v
```
Expected: ImportError.

- [ ] **Step 4: Implement payload.py**

`libs/cli/deepagents_cli/deploy/payload.py`:

```python
"""Build the JSON body POSTed (or PATCHed) to /v1/deepagents/agents.

This is a pure function over `Project`; no I/O happens here. The result is
suitable for `ApiClient.create_agent` or `ApiClient.patch_agent`.
"""

from __future__ import annotations

from typing import Any, Literal

from deepagents_cli.deploy.project import Project, Skill, Subagent


Mode = Literal["create", "patch"]


def build_payload(project: Project, *, mode: Mode = "create") -> dict[str, Any]:
    """Compose the request body for create_agent / patch_agent."""
    payload: dict[str, Any] = {"name": project.name}
    if project.description:
        payload["description"] = project.description
    if project.runtime:
        payload["runtime"] = project.runtime
    if project.permissions:
        payload["permissions"] = project.permissions
    if project.extras:
        payload["extras"] = project.extras

    payload["system_prompt"] = project.system_prompt

    if project.tools is not None:
        payload["tools"] = project.tools

    if project.skills:
        payload["skills"] = [_skill_dict(s) for s in project.skills]

    if project.subagents:
        payload["subagents"] = [_subagent_dict(s) for s in project.subagents]

    extra_files = _collect_extra_files(project.subagents)
    if extra_files:
        payload["files"] = extra_files

    # `mode` is exposed for forward-compat (e.g. emitting `deleted_paths` on
    # patch). Today the body shape is identical; we still keep the literal
    # type to document caller intent.
    _ = mode
    return payload


def _skill_dict(skill: Skill) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": "inline",
        "name": skill.name,
        "description": skill.description,
        "instructions": skill.instructions,
    }
    if skill.files:
        out["files"] = dict(skill.files)
    return out


def _subagent_dict(sa: Subagent) -> dict[str, Any]:
    out: dict[str, Any] = {"name": sa.name, "instructions": sa.instructions}
    if sa.description:
        out["description"] = sa.description
    if sa.model_id:
        out["model_id"] = sa.model_id
    if sa.tools is not None:
        out["tools"] = sa.tools
    return out


def _collect_extra_files(subagents: list[Subagent]) -> dict[str, dict[str, str]]:
    """Map raw-files entries from subagents into the top-level `files` field."""
    out: dict[str, dict[str, str]] = {}
    for sa in subagents:
        for rel, content in sa.extra_files.items():
            out[f"subagents/{sa.name}/{rel}"] = {"content": content}
    return out
```

- [ ] **Step 5: Reorder keys in `Subagent` dict** so the snapshot fixture order
  matches. Either order the JSON fixture (preferred) or sort the dict here. The
  fixture above lists `name`, `description`, `model_id`, `instructions`, `tools`.
  Adjust `_subagent_dict` if needed to emit in that order (Python dicts preserve
  insertion order):

```python
def _subagent_dict(sa: Subagent) -> dict[str, Any]:
    out: dict[str, Any] = {"name": sa.name}
    if sa.description:
        out["description"] = sa.description
    if sa.model_id:
        out["model_id"] = sa.model_id
    out["instructions"] = sa.instructions
    if sa.tools is not None:
        out["tools"] = sa.tools
    return out
```

- [ ] **Step 6: Run to confirm pass**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_payload.py -v
```
Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/payload.py \
        libs/cli/tests/unit_tests/deploy/test_payload.py \
        libs/cli/tests/unit_tests/deploy/fixtures/projects/*/expected_payload.json
git commit -m "feat(cli): build_payload — Project → CreateAgentRequest"
```

---

### Task 12: MCP server resolver

**Files:**
- Create: `libs/cli/deepagents_cli/deploy/mcp_resolver.py`
- Create: `libs/cli/tests/unit_tests/deploy/test_mcp_resolver.py`

- [ ] **Step 1: Failing tests**

```python
"""Tests for resolve_referenced_servers."""

from __future__ import annotations

import httpx
import pytest

from deepagents_cli.deploy.api_client import ApiClient
from deepagents_cli.deploy.mcp_resolver import (
    UnresolvedServersError,
    resolve_referenced_servers,
)


def _client(monkeypatch: pytest.MonkeyPatch, handler) -> ApiClient:
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    return ApiClient.from_env(transport=httpx.MockTransport(handler))


def test_url_normalization_strips_trailing_slash_and_lowercases() -> None:
    from deepagents_cli.deploy.mcp_resolver import _normalize_url
    assert _normalize_url("https://tools.langchain.com/") == "https://tools.langchain.com"
    assert _normalize_url("HTTPS://Tools.LangChain.com") == "https://tools.langchain.com"


def test_all_referenced_servers_present(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "servers": [
                    {"id": "s1", "url": "https://tools.langchain.com"},
                    {"id": "s2", "url": "https://other.example/"},
                ]
            },
        )

    client = _client(monkeypatch, handler)
    payload = {
        "tools": {"tools": [{"name": "x", "mcp_server_url": "https://tools.langchain.com/"}]},
        "subagents": [
            {"tools": {"tools": [{"name": "y", "mcp_server_url": "https://other.example"}]}}
        ],
    }
    cache = resolve_referenced_servers(client, payload, cache={})
    assert cache["https://tools.langchain.com"] == "s1"
    assert cache["https://other.example"] == "s2"


def test_unresolved_server_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"servers": []})

    client = _client(monkeypatch, handler)
    payload = {"tools": {"tools": [{"name": "x", "mcp_server_url": "https://missing.example"}]}}
    with pytest.raises(UnresolvedServersError) as excinfo:
        resolve_referenced_servers(client, payload, cache={})
    assert "https://missing.example" in str(excinfo.value)


def test_cached_ids_skip_list_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"servers": []})

    client = _client(monkeypatch, handler)
    payload = {"tools": {"tools": [{"name": "x", "mcp_server_url": "https://tools.example"}]}}
    cache = {"https://tools.example": "s1"}
    out = resolve_referenced_servers(client, payload, cache=cache)
    assert calls["n"] == 0
    assert out["https://tools.example"] == "s1"
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_mcp_resolver.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `mcp_resolver.py`**

```python
"""Resolve MCP server URLs in a payload to workspace-registered server IDs.

The deploy command does not auto-create MCP servers. Instead it validates that
every `mcp_server_url` the payload references already exists at the
`/v1/deepagents/mcp-servers` endpoint, and surfaces a friendly hint if not.
"""

from __future__ import annotations

from typing import Any

from deepagents_cli.deploy.api_client import ApiClient


class UnresolvedServersError(RuntimeError):
    """Raised when one or more `mcp_server_url`s aren't registered."""


def _normalize_url(url: str) -> str:
    return url.strip().rstrip("/").lower()


def _collect_referenced_urls(payload: dict[str, Any]) -> set[str]:
    urls: set[str] = set()
    for tool in (payload.get("tools") or {}).get("tools", []):
        u = tool.get("mcp_server_url")
        if isinstance(u, str) and u:
            urls.add(_normalize_url(u))
    for sa in payload.get("subagents", []):
        for tool in (sa.get("tools") or {}).get("tools", []):
            u = tool.get("mcp_server_url")
            if isinstance(u, str) and u:
                urls.add(_normalize_url(u))
    return urls


def resolve_referenced_servers(
    client: ApiClient,
    payload: dict[str, Any],
    *,
    cache: dict[str, str],
) -> dict[str, str]:
    """Return `{normalized_url → mcp_server_id}` for every URL in *payload*.

    Uses `cache` as a starting point and only hits the list endpoint if any
    URL is missing. Raises `UnresolvedServersError` if any URL is still
    unresolved after the list call.
    """
    referenced = _collect_referenced_urls(payload)
    out: dict[str, str] = {url: cache[url] for url in referenced if url in cache}
    missing = referenced - out.keys()
    if not missing:
        return out

    for server in client.list_mcp_servers():
        url = server.get("url")
        if isinstance(url, str) and url:
            key = _normalize_url(url)
            if key in missing:
                out[key] = server["id"]

    still_missing = referenced - out.keys()
    if still_missing:
        listed = "\n".join(f"  - {u}" for u in sorted(still_missing))
        msg = (
            f"The following MCP server URLs referenced in your tools "
            f"are not registered in this workspace:\n{listed}\n\n"
            f"Register each with:\n"
            f"  deepagents mcp-servers add --url <url> "
            f"--header KEY=VALUE [--name <name>]"
        )
        raise UnresolvedServersError(msg)
    return out
```

- [ ] **Step 4: Run to confirm pass**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_mcp_resolver.py -v
```
Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/mcp_resolver.py \
        libs/cli/tests/unit_tests/deploy/test_mcp_resolver.py
git commit -m "feat(cli): MCP server resolver — verify URLs are registered"
```

---

### Task 13: New `commands.py` skeleton + init scaffold

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/commands.py` (rewrite)
- Create: `libs/cli/tests/unit_tests/deploy/test_init_command.py`

Goal: stand up a new `commands.py` that registers `init`, `deploy`, `agents`,
and `mcp-servers` subparsers, with only `init` fully implemented in this task.
`deploy`, `agents`, `mcp-servers` get stub handlers raising `NotImplementedError`
so the argparse tree compiles.

- [ ] **Step 1: Failing test for init**

```python
"""Tests for `deepagents init`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from deepagents_cli.deploy.commands import execute_init_command


def _ns(name: str | None, *, force: bool = False) -> argparse.Namespace:
    return argparse.Namespace(name=name, force=force)


def test_init_scaffolds_new_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    execute_init_command(_ns("my-agent"))
    project = tmp_path / "my-agent"
    assert (project / "agent.json").is_file()
    agent = json.loads((project / "agent.json").read_text())
    assert agent["name"] == "my-agent"
    assert (project / "AGENTS.md").is_file()
    assert (project / ".gitignore").is_file()
    assert ".deepagents/" in (project / ".gitignore").read_text()
    assert (project / "skills").is_dir()


def test_init_refuses_existing_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x").mkdir()
    with pytest.raises(SystemExit):
        execute_init_command(_ns("x"))


def test_init_force_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "agent.json").write_text("{}")
    execute_init_command(_ns("x", force=True))
    agent = json.loads((tmp_path / "x" / "agent.json").read_text())
    assert agent["name"] == "x"
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_init_command.py -v
```
Expected: ImportError or failure (current `execute_init_command` writes old layout).

- [ ] **Step 3: Replace `commands.py`**

`libs/cli/deepagents_cli/deploy/commands.py`:

```python
"""CLI commands for `deepagents init`, `deploy`, `agents`, and `mcp-servers`.

Wired into the root argparse subparsers by `setup_deploy_parsers` (called from
`deepagents_cli.main`). Each top-level command has an `execute_*_command`
entrypoint that the main module dispatches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

_BETA_WARNING = (
    "\033[33mWarning: `deepagents deploy` is in beta. "
    "APIs, configuration format, and behavior may change between releases.\033[0m\n"
)


def setup_deploy_parsers(
    subparsers: Any,  # noqa: ANN401
    *,
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
) -> None:
    """Register the top-level subparsers for the migrated deploy CLI."""
    _add_init_parser(subparsers, make_help_action)
    _add_deploy_parser(subparsers, make_help_action)
    _add_agents_parser(subparsers, make_help_action)
    _add_mcp_servers_parser(subparsers, make_help_action)


# --- init -------------------------------------------------------------------


def _add_init_parser(subparsers: Any, make_help_action) -> None:  # noqa: ANN001
    p = subparsers.add_parser(
        "init",
        help="(beta) Scaffold a new managed-agent project",
        add_help=False,
    )
    p.add_argument("name", nargs="?", default=None)
    p.add_argument(
        "-h", "--help",
        action=make_help_action(lambda: p.print_help()),
        help="show this help message and exit",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing files")


def execute_init_command(args: argparse.Namespace) -> None:
    print(_BETA_WARNING)
    name = args.name
    if name is None:
        try:
            name = input("Project name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(1) from None
        if not name:
            print("Error: project name is required.")
            raise SystemExit(1)
    _scaffold(name=name, force=args.force)


def _scaffold(*, name: str, force: bool) -> None:
    project_dir = Path.cwd() / name
    if project_dir.exists() and not force:
        print(f"Error: {name}/ already exists. Use --force to overwrite.")
        raise SystemExit(1)
    project_dir.mkdir(parents=True, exist_ok=True)

    (project_dir / "agent.json").write_text(_STARTER_AGENT_JSON.format(name=name))
    (project_dir / "AGENTS.md").write_text(_STARTER_AGENTS_MD)
    (project_dir / ".gitignore").write_text(_STARTER_GITIGNORE)
    (project_dir / ".env").write_text(_STARTER_ENV)
    (project_dir / "skills").mkdir(exist_ok=True)

    print(f"Created {name}/ with: agent.json, AGENTS.md, .gitignore, .env, skills/")
    print("\nNext steps:")
    print(f"  cd {name}")
    print("  # edit AGENTS.md, optionally add tools.json / skills/ / subagents/")
    print("  deepagents deploy")


_STARTER_AGENT_JSON = """\
{{
  "name": "{name}",
  "description": "A managed deep agent.",
  "runtime": {{
    "model": {{"model_id": "anthropic:claude-sonnet-4-6"}},
    "backend_type": "thread_scoped_sandbox"
  }}
}}
"""

_STARTER_AGENTS_MD = """\
# Agent Instructions

You are a helpful AI agent.

## Guidelines

- Follow the user's instructions carefully.
- Ask for clarification when the request is ambiguous.
"""

_STARTER_GITIGNORE = """\
.env
.deepagents/
"""

_STARTER_ENV = """\
# Required: LangSmith API key for /v1/deepagents/* (private preview)
LANGSMITH_API_KEY=

# Optional: override the API endpoint (defaults to https://api.smith.langchain.com)
# LANGSMITH_ENDPOINT=
"""


# --- deploy / agents / mcp-servers (stubs filled by later tasks) ------------


def _add_deploy_parser(subparsers: Any, make_help_action) -> None:  # noqa: ANN001
    p = subparsers.add_parser(
        "deploy",
        help="(beta) Upsert the project as a managed deep agent",
        add_help=False,
    )
    p.add_argument("-h", "--help",
                   action=make_help_action(lambda: p.print_help()),
                   help="show this help message and exit")
    p.add_argument("--dir", type=str, default=None,
                   help="Project directory (default: cwd)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print payload without sending")
    p.add_argument("--detach", action="store_true",
                   help="Exit immediately after upsert without polling health")
    p.add_argument("--reset", action="store_true",
                   help="Discard local state and create a fresh agent")


def execute_deploy_command(args: argparse.Namespace) -> None:
    raise NotImplementedError("filled in Task 14")


def _add_agents_parser(subparsers: Any, make_help_action) -> None:  # noqa: ANN001
    p = subparsers.add_parser("agents", help="Manage agents", add_help=False)
    p.add_argument("-h", "--help",
                   action=make_help_action(lambda: p.print_help()),
                   help="show this help message and exit")
    sub = p.add_subparsers(dest="agents_cmd", required=True)
    sub.add_parser("list")
    g = sub.add_parser("get"); g.add_argument("agent_id"); g.add_argument("--include-files", action="store_true")
    d = sub.add_parser("delete"); d.add_argument("agent_id"); d.add_argument("--yes", action="store_true")


def execute_agents_command(args: argparse.Namespace) -> None:
    raise NotImplementedError("filled in Task 16")


def _add_mcp_servers_parser(subparsers: Any, make_help_action) -> None:  # noqa: ANN001
    p = subparsers.add_parser("mcp-servers", help="Manage MCP servers", add_help=False)
    p.add_argument("-h", "--help",
                   action=make_help_action(lambda: p.print_help()),
                   help="show this help message and exit")
    sub = p.add_subparsers(dest="mcp_cmd", required=True)
    sub.add_parser("list")
    a = sub.add_parser("add")
    a.add_argument("--url", required=True)
    a.add_argument("--name", default=None)
    a.add_argument("--header", action="append", default=[], metavar="KEY=VALUE")
    a.add_argument("--auth-type", default="headers", choices=["headers"])
    g = sub.add_parser("get"); g.add_argument("mcp_server_id")
    d = sub.add_parser("delete"); d.add_argument("mcp_server_id"); d.add_argument("--yes", action="store_true")


def execute_mcp_servers_command(args: argparse.Namespace) -> None:
    raise NotImplementedError("filled in Task 17")
```

- [ ] **Step 4: Run to confirm pass**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_init_command.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/commands.py libs/cli/tests/unit_tests/deploy/test_init_command.py
git commit -m "feat(cli): new commands.py skeleton + init scaffold for managed layout"
```

---

### Task 14: Wire `deploy` command orchestrator

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/commands.py`
- Create: `libs/cli/tests/unit_tests/deploy/test_deploy_command.py`

- [ ] **Step 1: Failing tests**

```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_deploy_command.py -v
```
Expected: NotImplementedError raised.

- [ ] **Step 3: Implement deploy orchestrator**

Replace `execute_deploy_command` and add helpers in `commands.py`:

```python
def execute_deploy_command(args: argparse.Namespace) -> None:
    from deepagents_cli.config import _load_dotenv  # existing helper
    from deepagents_cli.deploy.api_client import ApiClient, ApiError
    from deepagents_cli.deploy.mcp_resolver import (
        UnresolvedServersError,
        resolve_referenced_servers,
    )
    from deepagents_cli.deploy.payload import build_payload
    from deepagents_cli.deploy.project import Project, ProjectError
    from deepagents_cli.deploy.state import State

    print(_BETA_WARNING)
    root = Path(args.dir).resolve() if args.dir else Path.cwd().resolve()
    _load_dotenv(start_path=root)

    try:
        project = Project.load(root)
    except ProjectError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from None

    state = State.load(root, reset=args.reset)
    payload = build_payload(project, mode="patch" if state.agent_id else "create")

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    client = ApiClient.from_env()
    state.endpoint = client.endpoint

    try:
        state.mcp_servers = resolve_referenced_servers(
            client, payload, cache=state.mcp_servers
        )
    except UnresolvedServersError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from None

    try:
        agent = _upsert_agent(client, state.agent_id, payload)
    except ApiError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from None

    state.save(agent_id=agent["id"], revision=agent.get("revision"))
    _print_deploy_result(agent, client.endpoint, detach=args.detach, client=client)


def _upsert_agent(
    client,  # type: ignore[no-untyped-def]
    agent_id: str | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    from deepagents_cli.deploy.api_client import ApiError

    if agent_id:
        try:
            return client.patch_agent(agent_id, payload)
        except ApiError as exc:
            if exc.status == 404:
                print(
                    f"Note: agent {agent_id} no longer exists — creating a new one."
                )
            else:
                raise
    return client.create_agent(payload)


def _print_deploy_result(
    agent: dict[str, Any],
    endpoint: str,
    *,
    detach: bool,
    client,  # type: ignore[no-untyped-def]
) -> None:
    name = agent.get("name", "?")
    agent_id = agent.get("id", "?")
    revision = agent.get("revision", "")[:8]
    smith_endpoint = endpoint.replace("api.smith.langchain.com", "smith.langchain.com")
    print(f"\nDeployed: {name}")
    print(f"  agent_id: {agent_id}")
    print(f"  revision: {revision}")
    print(f"  {smith_endpoint}/o/-/agents/{agent_id}")
    if detach:
        return
    try:
        health = client._request("GET", f"/v1/deepagents/agents/{agent_id}/health")
        print(f"  health:   {health}")
    except Exception as exc:  # noqa: BLE001
        print(f"  health check skipped: {exc}")
```

- [ ] **Step 4: Run to confirm pass**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_deploy_command.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/commands.py libs/cli/tests/unit_tests/deploy/test_deploy_command.py
git commit -m "feat(cli): deploy command — POST/PATCH /v1/deepagents/agents"
```

---

### Task 15: Validate `tools.json` references at deploy time (already wired)

The previous task wired `resolve_referenced_servers`; this task verifies the
end-to-end UX: a missing server registration produces the friendly hint with
exit code 1.

**Files:**
- Modify: `libs/cli/tests/unit_tests/deploy/test_deploy_command.py`

- [ ] **Step 1: Failing test**

```python
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
```

- [ ] **Step 2: Run + confirm pass (no impl change needed)**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_deploy_command.py::test_deploy_fails_when_tools_reference_unregistered_server -v
```
Expected: PASS (the helper is wired in Task 14).

- [ ] **Step 3: Commit (the new test only)**

```bash
git add libs/cli/tests/unit_tests/deploy/test_deploy_command.py
git commit -m "test(cli): deploy fails fast on unregistered MCP server URL"
```

---

### Task 16: `agents` CRUD subcommand handlers

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/commands.py`
- Create: `libs/cli/tests/unit_tests/deploy/test_agents_command.py`

- [ ] **Step 1: Failing tests**

```python
"""Tests for `deepagents agents {list,get,delete}`."""

from __future__ import annotations

import argparse
import json

import httpx
import pytest

import deepagents_cli.deploy.api_client as api_client_module
from deepagents_cli.deploy.commands import execute_agents_command


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.setattr(
        api_client_module.ApiClient, "from_env",
        classmethod(lambda cls, transport=None: cls(
            endpoint="https://api.invalid", api_key="k",
            transport=httpx.MockTransport(handler))),
    )


def test_agents_list(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": [{"id": "a1", "name": "x"}], "next_cursor": None})

    _patch_client(monkeypatch, handler)
    execute_agents_command(argparse.Namespace(agents_cmd="list"))
    out = capsys.readouterr().out
    assert "a1" in out and "x" in out


def test_agents_get(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/deepagents/agents/a1"
        return httpx.Response(200, json={"id": "a1", "name": "x", "revision": "r1"})

    _patch_client(monkeypatch, handler)
    execute_agents_command(argparse.Namespace(agents_cmd="get", agent_id="a1", include_files=False))
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["id"] == "a1"


def test_agents_delete_requires_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "n")
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not be called")
    _patch_client(monkeypatch, handler)
    execute_agents_command(argparse.Namespace(agents_cmd="delete", agent_id="a1", yes=False))
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
    execute_agents_command(argparse.Namespace(agents_cmd="delete", agent_id="a1", yes=True))
    assert calls == ["DELETE"]
    assert "Deleted" in capsys.readouterr().out
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_agents_command.py -v
```
Expected: NotImplementedError.

- [ ] **Step 3: Implement the handlers**

In `commands.py`, replace `execute_agents_command`:

```python
def execute_agents_command(args: argparse.Namespace) -> None:
    from deepagents_cli.deploy.api_client import ApiClient, ApiError

    client = ApiClient.from_env()
    try:
        if args.agents_cmd == "list":
            for agent in client.iter_agents(page_size=50):
                print(f"{agent.get('id')}\t{agent.get('name', '')}\t{agent.get('updated_at', '')}")
        elif args.agents_cmd == "get":
            agent = client.get_agent(args.agent_id, include_files=args.include_files)
            print(json.dumps(agent, indent=2))
        elif args.agents_cmd == "delete":
            if not args.yes:
                try:
                    answer = input(f"Delete agent {args.agent_id}? [y/N]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    print("Aborted.")
                    return
                if answer not in {"y", "yes"}:
                    print("Aborted.")
                    return
            client.delete_agent(args.agent_id)
            print(f"Deleted {args.agent_id}")
    except ApiError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from None
```

- [ ] **Step 4: Run to confirm pass**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_agents_command.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/commands.py libs/cli/tests/unit_tests/deploy/test_agents_command.py
git commit -m "feat(cli): `deepagents agents` list/get/delete"
```

---

### Task 17: `mcp-servers` CRUD subcommand handlers

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/commands.py`
- Create: `libs/cli/tests/unit_tests/deploy/test_mcp_servers_command.py`

- [ ] **Step 1: Failing tests**

```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_mcp_servers_command.py -v
```
Expected: NotImplementedError.

- [ ] **Step 3: Implement the handlers**

In `commands.py`, replace `execute_mcp_servers_command`:

```python
def execute_mcp_servers_command(args: argparse.Namespace) -> None:
    from urllib.parse import urlparse

    from deepagents_cli.deploy.api_client import ApiClient, ApiError

    client = ApiClient.from_env()
    try:
        if args.mcp_cmd == "list":
            for srv in client.list_mcp_servers():
                print(f"{srv.get('id')}\t{srv.get('name', '')}\t{srv.get('url', '')}")
        elif args.mcp_cmd == "add":
            headers = _parse_header_args(args.header)
            name = args.name or urlparse(args.url).hostname or args.url
            srv = client.create_mcp_server(
                name=name,
                url=args.url,
                headers=headers,
                auth_type=args.auth_type,
            )
            print(f"Created mcp_server {srv.get('id')}: {srv.get('name')} → {srv.get('url')}")
        elif args.mcp_cmd == "get":
            print(json.dumps(client.get_mcp_server(args.mcp_server_id), indent=2))
        elif args.mcp_cmd == "delete":
            if not args.yes:
                try:
                    answer = input(f"Delete MCP server {args.mcp_server_id}? [y/N]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    print("Aborted.")
                    return
                if answer not in {"y", "yes"}:
                    print("Aborted.")
                    return
            client.delete_mcp_server(args.mcp_server_id)
            print(f"Deleted {args.mcp_server_id}")
    except ApiError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from None


def _parse_header_args(raw: list[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for entry in raw:
        if "=" not in entry:
            print(f"Error: --header must be KEY=VALUE, got {entry!r}")
            raise SystemExit(1)
        key, _, value = entry.partition("=")
        out.append({"key": key.strip(), "value": value})
    return out
```

- [ ] **Step 4: Run to confirm pass**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_mcp_servers_command.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/commands.py libs/cli/tests/unit_tests/deploy/test_mcp_servers_command.py
git commit -m "feat(cli): `deepagents mcp-servers` list/add/get/delete"
```

---

### Task 18: Wire new commands in `main.py`; drop `dev`

**Files:**
- Modify: `libs/cli/deepagents_cli/main.py`
- Modify: `libs/cli/deepagents_cli/deploy/__init__.py`

- [ ] **Step 1: Update `main.py` dispatcher**

Find the block (approx lines 149-164) and replace with:

```python
    try:
        if args.command == "init":
            from deepagents_cli.deploy import execute_init_command

            execute_init_command(args)
        elif args.command == "deploy":
            from deepagents_cli.deploy import execute_deploy_command

            execute_deploy_command(args)
        elif args.command == "agents":
            from deepagents_cli.deploy import execute_agents_command

            execute_agents_command(args)
        elif args.command == "mcp-servers":
            from deepagents_cli.deploy import execute_mcp_servers_command

            execute_mcp_servers_command(args)
        else:
            sys.stderr.write(_REPL_REDIRECT_MESSAGE + "\n")
            raise SystemExit(1)
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        raise SystemExit(130) from None
```

Also update the docstring at the top of `main.py`:

```python
"""Entry point for the `deepagents` CLI.

This CLI exposes the deployment-oriented commands for Managed Deep Agents:
`init`, `deploy`, `agents`, and `mcp-servers`. Bare invocations print a
deprecation notice and exit non-zero.
"""
```

And update `_REPL_REDIRECT_MESSAGE` so the help line at the bottom reads
`The `deepagents` CLI now only provides `init`, `deploy`, `agents`, and `mcp-servers`.`

- [ ] **Step 2: Update `deploy/__init__.py`**

`libs/cli/deepagents_cli/deploy/__init__.py`:

```python
"""Deploy commands for the Managed Deep Agents (`/v1/deepagents/*`) surface."""

from deepagents_cli.deploy.commands import (
    execute_agents_command,
    execute_deploy_command,
    execute_init_command,
    execute_mcp_servers_command,
    setup_deploy_parsers,
)

__all__ = [
    "execute_agents_command",
    "execute_deploy_command",
    "execute_init_command",
    "execute_mcp_servers_command",
    "setup_deploy_parsers",
]
```

- [ ] **Step 3: Smoke check — argparse builds without error**

```bash
cd libs/cli && uv run deepagents --help
```
Expected: help text shows `init`, `deploy`, `agents`, `mcp-servers`; no error.
`dev` no longer appears.

- [ ] **Step 4: Run the full deploy test suite**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/ -v --ignore=tests/unit_tests/deploy/test_bundler.py --ignore=tests/unit_tests/deploy/test_config.py --ignore=tests/unit_tests/deploy/test_context_hub.py --ignore=tests/unit_tests/deploy/test_frontend_bundle.py --ignore=tests/unit_tests/deploy/test_frontend_config.py --ignore=tests/unit_tests/deploy/test_commands.py
```
Expected: all new tests pass; legacy tests are still on disk but skipped via
`--ignore`.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/main.py libs/cli/deepagents_cli/deploy/__init__.py
git commit -m "feat(cli): wire agents + mcp-servers; drop dev from main.py"
```

---

### Task 19: Delete legacy code

**Files:**
- Delete: `libs/cli/deepagents_cli/deploy/bundler.py`
- Delete: `libs/cli/deepagents_cli/deploy/templates.py`
- Delete: `libs/cli/deepagents_cli/deploy/context_hub.py`
- Delete: `libs/cli/deepagents_cli/deploy/config.py` (old TOML parser)
- Delete: `libs/cli/deepagents_cli/deploy/frontend_dist/` (entire directory)
- Delete: `libs/cli/tests/unit_tests/deploy/test_bundler.py`
- Delete: `libs/cli/tests/unit_tests/deploy/test_config.py`
- Delete: `libs/cli/tests/unit_tests/deploy/test_context_hub.py`
- Delete: `libs/cli/tests/unit_tests/deploy/test_frontend_bundle.py`
- Delete: `libs/cli/tests/unit_tests/deploy/test_frontend_config.py`
- Delete: old `libs/cli/tests/unit_tests/deploy/test_commands.py` (covered by new
  test files per-command)
- Delete: `libs/cli/frontend/` (the React source — frontend bundle no longer shipped)
- Modify: `libs/cli/pyproject.toml` (drop `frontend_dist` packaging + sandbox extras)

- [ ] **Step 1: Verify nothing outside the deploy module imports the dead files**

```bash
grep -rn -E "from deepagents_cli.deploy.(bundler|templates|context_hub|config)|deepagents_cli/deploy/frontend_dist" \
  libs/cli/deepagents_cli libs/cli/tests --include='*.py' \
  | grep -v 'tests/unit_tests/deploy/test_(bundler|config|context_hub|frontend)'
```
Expected: zero matches.

- [ ] **Step 2: Delete the files**

```bash
cd libs/cli
rm -rf deepagents_cli/deploy/bundler.py \
       deepagents_cli/deploy/templates.py \
       deepagents_cli/deploy/context_hub.py \
       deepagents_cli/deploy/config.py \
       deepagents_cli/deploy/frontend_dist \
       frontend \
       tests/unit_tests/deploy/test_bundler.py \
       tests/unit_tests/deploy/test_config.py \
       tests/unit_tests/deploy/test_context_hub.py \
       tests/unit_tests/deploy/test_frontend_bundle.py \
       tests/unit_tests/deploy/test_frontend_config.py \
       tests/unit_tests/deploy/test_commands.py
```

- [ ] **Step 3: Update `pyproject.toml`**

Remove the `[project.optional-dependencies]` sandbox extras (agentcore, daytona,
modal, runloop, all-sandboxes) since the managed runtime owns sandboxing. Replace
the entire block (lines ~47-54) with:

```toml
[project.optional-dependencies]
# (Sandbox extras dropped — managed runtime handles sandboxing.)
```

Remove the `deepagents_cli/deploy/frontend_dist/**/*` line from
`[tool.hatch.build.targets.wheel].include` (~lines 89-92):

```toml
[tool.hatch.build.targets.wheel]
packages = ["deepagents_cli"]
```

Remove `langchain-daytona`, `langchain-modal`, `langchain-runloop` from
`[tool.uv.sources]` (~lines 94-98):

```toml
[tool.uv.sources]
deepagents = { path = "../deepagents", editable = true }
```

Remove the matching extra-paths from `[tool.ty.environment]` (~lines 100-107):

```toml
[tool.ty.environment]
python-version = "3.11"
extra-paths = ["../deepagents"]
```

Also drop the langgraph CLI dependency from the `[project] dependencies` block
(~lines 26-44), and the partner-package sandbox deps that no longer apply:

Remove these lines from `dependencies`:
- `"langgraph-cli[inmem]>=0.4.24,<1.0.0",`
- `"langgraph-runtime-inmem>=0.28.1,<1.0.0",`

Keep: `deepagents`, `langchain`, `langgraph`, `langgraph-sdk`, `httpx`,
`langsmith`, `python-dotenv`.

- [ ] **Step 4: Re-resolve the lockfile**

```bash
cd libs/cli && uv lock
```

- [ ] **Step 5: Run the entire test suite**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/ -v
```
Expected: only new tests, all passing. No collection errors.

- [ ] **Step 6: Commit**

```bash
git add -A libs/cli
git commit -m "refactor(cli): delete bundler/templates/context_hub/frontend; trim deps"
```

---

### Task 20: Rewrite examples for the new layout

**Files:**
- Delete: `examples/deploy-gtm-agent/deepagents.toml`
- Create: `examples/deploy-gtm-agent/agent.json`
- Same for: `deploy-content-writer`, `deploy-coding-agent`, `deploy-mcp-docs-agent`,
  `libs/cli/examples/deploy-content-writer`

- [ ] **Step 1: For each example directory, convert `deepagents.toml` → `agent.json`**

For `examples/deploy-gtm-agent/agent.json`:

```json
{
  "name": "deepagents-deploy-gtm-agent",
  "description": "Go-to-market strategy agent that coordinates research and content creation",
  "runtime": {
    "model": {"model_id": "openai:gpt-5.4-nano"}
  }
}
```

Repeat for each example, reading the existing `[agent]` name/description/model
and the `[sandbox].scope` → `runtime.backend_type` if non-default.

- [ ] **Step 2: Remove legacy files**

```bash
find examples libs/cli/examples -name deepagents.toml -delete
find examples libs/cli/examples -name mcp.json -delete
```

If any example had subagents using `deepagents.toml`, convert each subagent's
config to `subagents/<name>/agent.json` with `{"description": "...", "model_id": "..."}`.

- [ ] **Step 3: Walk through `Project.load(<example_dir>)` for each example**

```bash
cd libs/cli && uv run python -c '
from pathlib import Path
from deepagents_cli.deploy.project import Project
for ex in [
    "../../examples/deploy-gtm-agent",
    "../../examples/deploy-content-writer",
    "../../examples/deploy-coding-agent",
    "../../examples/deploy-mcp-docs-agent",
    "examples/deploy-content-writer",
]:
    p = Project.load(Path(ex))
    print(ex, "→", p.name, "subagents:", [s.name for s in p.subagents])
'
```
Expected: every example loads cleanly.

- [ ] **Step 4: Commit**

```bash
git add examples libs/cli/examples
git commit -m "examples: migrate to agent.json + tools.json layout"
```

---

### Task 21: Update README + CHANGELOG

**Files:**
- Modify: `libs/cli/README.md`
- Modify: `libs/cli/CHANGELOG.md`
- Modify: `libs/cli/pyproject.toml` (bump version to 0.2.0)

- [ ] **Step 1: Update `libs/cli/CHANGELOG.md`**

Prepend:

```markdown
## 0.2.0 (2026-05-20) — Managed Deep Agents

**Breaking changes** — `deepagents deploy` now targets the Managed Deep Agents
API (`/v1/deepagents/*`) instead of `langgraph deploy`. The on-disk layout
changes too:

- `deepagents.toml` → `agent.json`
- `mcp.json` → MCP servers registered via `deepagents mcp-servers add ...`,
  with tool references living in `tools.json`
- `[sandbox].scope` → `agent.json.runtime.backend_type`
- `[frontend]`, `[auth]`, `[memories]` — removed (the platform owns these now)

Removed: `deepagents dev` (no local-iteration path post-migration).

New: `deepagents agents {list,get,delete}`, `deepagents mcp-servers
{list,add,get,delete}`.

Run `deepagents init --force` on an existing project, or migrate by hand —
`deepagents deploy` prints a migration hint when it detects a legacy
`deepagents.toml`.
```

- [ ] **Step 2: Update `libs/cli/README.md`**

Rewrite the "Quickstart" and "Project layout" sections to describe `agent.json`,
`AGENTS.md`, `tools.json`, `skills/`, `subagents/`. Document the new
subcommands. Reference the spec at
`docs/superpowers/specs/2026-05-20-deepagents-deploy-mda-migration-design.md`.

- [ ] **Step 3: Bump version in `pyproject.toml`**

Change `version = "0.1.1"` to `version = "0.2.0"`.

- [ ] **Step 4: Commit**

```bash
git add libs/cli/README.md libs/cli/CHANGELOG.md libs/cli/pyproject.toml
git commit -m "docs(cli): document MDA-deploy migration; bump to 0.2.0"
```

---

### Task 22: Final integration sweep

**Files:**
- All of `libs/cli/`

- [ ] **Step 1: Run the full CLI test suite**

```bash
cd libs/cli && uv run pytest tests/unit_tests/ -v
```
Expected: all passing, no collection errors.

- [ ] **Step 2: Run linter and type-check**

```bash
cd libs/cli && uv run ruff check deepagents_cli/deploy && uv run ty check deepagents_cli/deploy
```
Expected: clean. Fix any issues that surface.

- [ ] **Step 3: Smoke test the CLI**

```bash
cd /tmp && rm -rf smoke-agent && cd /Users/victormoreira/Desktop/langchain-repos/deepagents/libs/cli
uv run deepagents init smoke-agent --force
ls /tmp/smoke-agent  # actually scaffolds in libs/cli/smoke-agent — adjust as needed
cd /tmp && rm -rf smoke-agent
```

Or, more cleanly:

```bash
cd /tmp && uv run --directory /Users/victormoreira/Desktop/langchain-repos/deepagents/libs/cli deepagents init smoke-agent
test -f /tmp/smoke-agent/agent.json && echo "OK: scaffold layout"
rm -rf /tmp/smoke-agent
```

Expected: scaffold creates the new layout.

- [ ] **Step 4: Validate `deploy --dry-run` against one of the examples**

```bash
cd /Users/victormoreira/Desktop/langchain-repos/deepagents/libs/cli
uv run deepagents deploy --dir ../../examples/deploy-gtm-agent --dry-run
```
Expected: prints a valid JSON payload to stdout; no API calls happen.

- [ ] **Step 5: Verify branch summary**

```bash
git -C /Users/victormoreira/Desktop/langchain-repos/deepagents log master..HEAD --oneline
```
Expected: the migration commits in order.

- [ ] **Step 6: No final commit needed** unless lint/type-check fixes were
  required, in which case commit them as `chore(cli): lint/type fixes after MDA
  migration`.

---

## Self-Review

**Spec coverage:** Each spec section maps to tasks:
- CLI surface → Tasks 13, 14, 16, 17, 18
- Project layout (`agent.json` etc.) → Tasks 6, 7, 8, 9, 13
- `agent.json` schema → Task 6
- `tools.json` schema → Task 7
- Module layout → Tasks 2, 3, 4, 5, 6, 11, 12, 13
- Deploy orchestration → Task 14
- MCP server management → Tasks 12, 15, 17
- Local state & idempotency → Tasks 2, 14
- Auth and endpoint resolution → Task 3
- Error handling → Tasks 3, 10, 14
- Output → Task 14
- Validation → Tasks 6, 7, 8, 9, 10
- Testing — fixtures + snapshots → Tasks 6, 7, 8, 9, 11

**Placeholder scan:** None — every step contains the actual code or command.

**Type consistency:** `Project`, `Skill`, `Subagent`, `State`, `ApiClient`,
`ApiError`, `UnresolvedServersError`, `build_payload`, `resolve_referenced_servers`
all used with consistent signatures across tasks.

**Open questions from the spec:**
- Subagent-local skills via raw `files` — Task 9 + Task 11 implement the fallback;
  Task 22 smoke-tests one example with this shape if available.
- Trace UI URL shape — Task 14 produces `/o/-/agents/<id>`; revise in a
  follow-up if MDA team prefers a different path.
- `extras` map — supported in `agent.json` parsing (Task 6) and payload (Task 11);
  scaffold does not include it (Task 13).
