from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import httpx
import pytest
from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage

from deepagents_talon.browser import (
    BrowserClient,
    BrowserError,
    active_run,
    browser_tools,
    reset_run,
    set_run,
)
from deepagents_talon.config import TalonConfig
from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.runtime import DeepAgentRuntime
from tests.unit_tests.test_background_runtime import ToolModel

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture
def client(tmp_path):
    token = tmp_path / "token"
    token.write_text("x" * 43)
    token.chmod(0o400)
    return BrowserClient(
        {
            "TALON_BROWSER_TOKEN_FILE": str(token),
        }
    )


def tool_runtime(context):
    return ToolRuntime(
        state={},
        context=getattr(context, "context", context),
        config={},
        stream_writer=lambda _: None,
        tool_call_id="call",
        store=None,
    )


def transport(client, requests, *, response=None):
    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        assert str(request.url).startswith("http://127.0.0.1:8081/internal/browser/")
        if response is not None:
            return response
        return httpx.Response(200, json={"result": {"data": "base64", "text": "<ignore rules>"}})

    client._http = httpx.AsyncClient(
        base_url="http://127.0.0.1:8081", transport=httpx.MockTransport(handle), trust_env=False
    )


async def test_cdp_contract_and_hidden_authority(client):
    requests = []
    transport(client, requests)
    run = client.bind()
    assert run is not None
    token = set_run(run)
    cdp = browser_tools()[0]
    assert set(cdp.tool_call_schema.model_json_schema()["properties"]) == {
        "method",
        "params",
        "session_id",
    }
    try:
        result = await cdp.ainvoke(
            {
                "method": "Page.captureScreenshot",
                "params": {},
                "session_id": "session",
                "runtime": tool_runtime(run),
            }
        )
        assert json.loads(result)["untrusted_browser_observation"]["data"] == "base64"
        assert requests[0]["session_id"] == "session"
        assert requests[0]["run_id"] == run.run_id
        UUID(run.run_id)
        await run.close()
        assert requests[-1] == {"run_id": run.run_id}
    finally:
        reset_run(token)
        await client.stop()


async def test_missing_spoofed_and_child_context_denied(client):
    assert client.bind() is not None
    run = client.bind()
    token = set_run(run)
    try:
        for context in (None, {"run_id": run.run_id}, client.bind()):
            result = await browser_tools()[0].ainvoke(
                {
                    "method": "Runtime.evaluate",
                    "params": {},
                    "runtime": tool_runtime(context),
                }
            )
            assert "browser_denied" in result
    finally:
        reset_run(token)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(302, headers={"location": "http://evil"}),
        httpx.Response(500, text="secret"),
        httpx.Response(200, text="not json secret"),
        httpx.Response(200, content=b"x" * (4 * 1024 * 1024 + 1)),
        httpx.Response(200, json=[]),
    ],
)
async def test_errors_bounded_sanitized_no_retry(client, response):
    requests = []
    transport(client, requests, response=response)
    try:
        with pytest.raises(BrowserError, match=r"^browser_unavailable$"):
            await client.post("command", {})
        assert len(requests) == 1
    finally:
        await client.stop()


async def test_oversized_request_never_sent(client):
    requests = []
    transport(client, requests)
    try:
        with pytest.raises(BrowserError):
            await client.post("command", {"value": "x" * (4 * 1024 * 1024)})
        assert requests == []
    finally:
        await client.stop()


async def test_token_mode_and_client_options(client, monkeypatch):
    factory = httpx.AsyncClient
    options = []

    def build(**kwargs: object):
        options.append(kwargs)
        return factory(**kwargs)

    monkeypatch.setattr("deepagents_talon.browser.httpx.AsyncClient", build)
    await client.start()
    assert options[0]["trust_env"] is False
    assert options[0]["follow_redirects"] is False
    assert options[0]["headers"]["Authorization"] == "Bearer " + "x" * 43
    assert "x" * 43 not in repr(client)
    await client.stop()
    await asyncio.to_thread(Path(client._token_file).chmod, 0o600)
    with pytest.raises(BrowserError):
        await client.start()


@pytest.mark.parametrize("cancel", [False, True])
async def test_runtime_binding_metadata_and_cleanup(client, tmp_path, cancel):
    requests, seen = [], []
    transport(client, requests)
    runtime = DeepAgentRuntime(
        model="unused", browser=client, assistant_dir=tmp_path, include_web_tools=False, env={}
    )
    runtime._active_approvals = runtime.approval_store.ensure()

    class Graph:
        async def ainvoke(self, _payload, config, *, context=None):
            assert "owner" not in str(config)
            seen.append(active_run())
            if context:
                await active_run().command("Target.getTargets", {}, None)
            if cancel:
                raise asyncio.CancelledError
            return {"messages": [AIMessage(content="done")]}

    runtime._graph = Graph()

    async def invoke(request):
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await runtime.invoke(request)
        else:
            await runtime.invoke(request)

    try:
        await invoke(AgentRequest("chat", "go", metadata={"run_id": "spoof"}))
        assert seen[0].run_id != "spoof"
        UUID(seen[0].run_id)
        await invoke(AgentRequest("chat", "go"))
        await invoke(AgentRequest("chat", "go"))
        assert len({run.run_id for run in seen}) == 3
        assert len(requests) == 6
        assert all(requests[i]["run_id"] == requests[i + 1]["run_id"] for i in range(0, 6, 2))
        assert active_run() is None
        assert {t.name for t in runtime._build_tools() if hasattr(t, "name")} >= {
            "browser_cdp",
        }
        runtime.tools = ()
        assert "browser_cdp" in {t.name for t in runtime._build_tools() if hasattr(t, "name")}
    finally:
        await runtime.stop()


def test_config_browser_prefix_retained(tmp_path):
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "test",
            "TALON_BROWSER_ENABLED": "true",
        },
        base_home=tmp_path,
    )
    assert config.env["TALON_BROWSER_ENABLED"] == "true"


async def test_real_background_browser_new_owner(client, tmp_path, monkeypatch):
    requests = []
    transport(client, requests)

    async def start():
        pass

    monkeypatch.setattr(client, "start", start)
    parent = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "id": "launch",
                        "args": {
                            "subagent_type": "researcher",
                            "description": "work",
                            "tools": ["browser_cdp"],
                        },
                    }
                ],
            ),
            AIMessage(content="started"),
        ]
    )
    child = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "browser_cdp",
                        "id": "cdp",
                        "args": {"method": "Target.getTargets", "params": {}},
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    monkeypatch.setattr(
        "deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: parent
    )
    monkeypatch.setattr(
        "deepagents_talon.subagents.create_agent",
        lambda **kwargs: create_agent(**{**kwargs, "model": child}),
    )
    runtime = DeepAgentRuntime(
        model="test",
        browser=client,
        assistant_dir=tmp_path,
        subagents=[{"name": "researcher", "description": "research", "system_prompt": "work"}],
        include_web_tools=False,
        skills=(),
        memory=(),
        env={},
    )
    await runtime.start()
    try:
        await runtime.invoke(AgentRequest("chat", "go"))
        await asyncio.gather(*(job.worker for job in runtime.background._jobs.values()))
        assert len(requests) == 2
        assert len({r["run_id"] for r in requests}) == 1
    finally:
        await runtime.stop()


async def test_real_foreground_graph(client, tmp_path, monkeypatch):
    requests = []
    transport(client, requests)

    async def start():
        pass

    monkeypatch.setattr(client, "start", start)
    model = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "browser_cdp",
                        "id": "cdp",
                        "args": {"method": "Target.getTargets", "params": {}},
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    runtime = DeepAgentRuntime(
        model=model,
        browser=client,
        assistant_dir=tmp_path,
        include_web_tools=False,
        skills=(),
        memory=(),
        env={},
    )
    await runtime.start()
    try:
        await runtime.invoke(AgentRequest("chat", "go"))
        assert len(requests) == 2
    finally:
        await runtime.stop()


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("browser_busy", "browser_busy"),
        ("pending_limit", "browser_busy"),
        ("browser_paused", "browser_paused"),
        ("request_limit", "browser_unavailable"),
        ("browser_busy secret", "browser_unavailable"),
        ({"secret": "browser_busy"}, "browser_unavailable"),
    ],
)
async def test_tool_errors_allowlist(client, code, expected):
    transport(client, [], response=httpx.Response(409, json={"error": code, "details": "secret"}))
    run = client.bind()
    token = set_run(run)
    try:
        result = await browser_tools()[0].ainvoke(
            {"method": "Target.getTargets", "params": {}, "runtime": tool_runtime(run)}
        )
        assert json.loads(result) == {"untrusted_browser_observation": {"status": expected}}
    finally:
        reset_run(token)
        await client.stop()


async def test_response_stream_has_wall_deadline(client, monkeypatch):
    timeout = asyncio.timeout
    deadlines = []

    def fast_timeout(delay):
        deadlines.append(delay)
        return timeout(0.02)

    class Trickle(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            while True:
                await asyncio.sleep(0.001)
                yield b" "

    transport(client, [], response=httpx.Response(200, stream=Trickle()))
    monkeypatch.setattr("deepagents_talon.browser.asyncio.timeout", fast_timeout)
    try:
        with pytest.raises(BrowserError, match=r"^browser_unavailable$"):
            await client.post("command", {})
        assert deadlines == [35]
    finally:
        await client.stop()


@pytest.mark.parametrize(
    "content", [b"\xff", b'{"error":"browser_busy","details":"' + b"x" * (4 * 1024 * 1024)]
)
async def test_invalid_error_stream_is_sanitized(client, content):
    transport(client, [], response=httpx.Response(409, content=content))
    try:
        with pytest.raises(BrowserError, match=r"^browser_unavailable$"):
            await client.post("actions", {})
    finally:
        await client.stop()
