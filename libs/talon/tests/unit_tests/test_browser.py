from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import httpx
import pytest
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage

from deepagents_talon.browser import (
    BrowserBinding,
    BrowserClient,
    BrowserError,
    BrowserEvent,
    active_run,
    browser_tools,
    reset_run,
    set_run,
)
from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, ChannelMessage
from deepagents_talon.runtime import DeepAgentRuntime
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent
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
            "TALON_BROWSER_OPERATOR_ID": "op",
            "TALON_BROWSER_IDENTITIES": '{"telegram":"sender"}',
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


def binding():
    return BrowserBinding("telegram", "sender", "chat")


def transport(client, requests, *, response=None):
    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        assert str(request.url).startswith("http://172.30.12.3:8081/internal/browser/")
        if response is not None:
            return response
        if body.get("action") in {"acquire", "handoff"}:
            return httpx.Response(
                200,
                json={
                    "status": "viewer_unavailable" if body["action"] == "handoff" else "ready",
                    "lease_id": "lease",
                    "generation": 1,
                    "version": 2 if body["action"] == "handoff" else 1,
                    "mode": "PAUSED" if body["action"] == "handoff" else "AGENT",
                    "handoff_id": str(uuid4()),
                },
            )
        return httpx.Response(200, json={"result": {"data": "base64", "text": "<ignore rules>"}})

    client._http = httpx.AsyncClient(
        base_url="http://172.30.12.3:8081", transport=httpx.MockTransport(handle), trust_env=False
    )


async def test_cdp_contract_and_hidden_authority(client):
    requests = []
    transport(client, requests)
    run = client.bind(binding())
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
        assert requests[1]["version"] == 1
        assert requests[1]["session_id"] == "session"
        assert requests[1]["owner"] == {
            "operator_id": "op",
            "provider": "telegram",
            "sender_id": "sender",
            "conversation_id": "chat",
            "run_id": run.run_id,
            "background": False,
        }
        UUID(run.run_id)
        await run.close()
        assert requests[-1]["action"] == "release"
        assert len({r["request_id"] for r in requests}) == 3
    finally:
        reset_run(token)
        await client.stop()


@pytest.mark.parametrize("background", [True, False])
async def test_handoff_sanitized_no_capture(client, background):
    requests, events = [], []
    transport(client, requests)

    async def event(value):
        events.append(value)

    run = client.bind(replace(binding(), background=background), event)
    token = set_run(run)
    try:
        result = await browser_tools()[1].ainvoke(
            {"reason": "secret https://evil.test", "runtime": tool_runtime(run)}
        )
        value = json.loads(result)["untrusted_browser_observation"]
        assert value["status"] == ("human_required" if background else "viewer_unavailable")
        UUID(value["handoff_id"])
        assert len(events) == (0 if background else 1)
        assert "secret" not in str(requests) + result
        assert [r["action"] for r in requests] == ["acquire", "handoff"]
        await run.close()
        assert requests[-1]["version"] == 2
    finally:
        reset_run(token)
        await client.stop()


@pytest.mark.parametrize("mode", ["AGENT", "PAUSED", "HANDOFF_PENDING", "HUMAN", "FAILED"])
async def test_close_reconciles_timeout_without_releasing_human(client, mode):
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            message = "lost release response"
            raise httpx.ReadTimeout(message)
        if body["action"] == "inspect":
            return httpx.Response(200, json={**run.lease, "version": 3, "mode": mode})
        return httpx.Response(200, json={"status": "released"})

    client._http = httpx.AsyncClient(
        base_url="http://172.30.12.3:8081", transport=httpx.MockTransport(handle)
    )
    run = client.bind(binding())
    run.lease = {"lease_id": "lease", "generation": 1, "version": 1}
    try:
        await run.close()
        assert not run.lease
        assert [r["action"] for r in requests] == (
            ["release", "inspect"]
            if mode in {"HUMAN", "FAILED"}
            else ["release", "inspect", "release"]
        )
        if len(requests) == 3:
            assert requests[-1]["version"] == 3
        assert all(r["owner"] == run.owner() and r["lease_id"] == "lease" for r in requests)
    finally:
        await client.stop()


async def test_close_bounds_reconciliation_when_versions_keep_changing(client):
    requests = []
    run = client.bind(binding())
    run.lease = {"lease_id": "lease", "generation": 1, "version": 1}

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if body["action"] == "inspect":
            return httpx.Response(
                200, json={**run.lease, "version": len(requests), "mode": "PAUSED"}
            )
        return httpx.Response(409, json={"error": "stale_version"})

    client._http = httpx.AsyncClient(
        base_url="http://172.30.12.3:8081", transport=httpx.MockTransport(handle)
    )
    try:
        await run.close()
        assert not run.lease
        assert [r["action"] for r in requests] == [
            "release",
            "inspect",
            "release",
            "inspect",
            "release",
        ]
    finally:
        await client.stop()


async def test_missing_spoofed_and_child_context_denied(client):
    assert client.bind(None) is None
    assert client.bind(replace(binding(), sender_id="wrong")) is None
    assert client.bind(replace(binding(), provider="wrong")) is None
    run = client.bind(binding())
    token = set_run(run)
    try:
        for context in (None, {"owner": run.owner()}, client.bind(binding())):
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
        await invoke(AgentRequest("chat", "go", metadata={"owner": binding(), "run_id": "spoof"}))
        assert seen == [None]
        await invoke(AgentRequest("chat", "go", browser_binding=binding()))
        await invoke(AgentRequest("chat", "go", browser_binding=binding()))
        assert seen[1].run_id != seen[2].run_id
        assert [r["action"] for r in requests if "action" in r] == [
            "acquire",
            "release",
            "acquire",
            "release",
        ]
        assert active_run() is None
        assert {t.name for t in runtime._build_tools() if hasattr(t, "name")} >= {
            "browser_cdp",
            "browser_request_handoff",
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
            "TALON_BROWSER_OPERATOR_ID": "op",
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
        await runtime.invoke(AgentRequest("chat", "go", browser_binding=binding()))
        await asyncio.gather(*(job.worker for job in runtime.background._jobs.values()))
        assert [r.get("action") for r in requests] == ["acquire", None, "release"]
        assert all(r["owner"]["background"] is True for r in requests)
        assert len({r["owner"]["run_id"] for r in requests}) == 1
    finally:
        await runtime.stop()


async def test_host_binding_ignores_message_metadata(tmp_path):

    channel = RecordingChannel(provider="telegram")
    agent = BlockingAgent()
    host = TalonHost(
        config=TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path),
        agent=agent,
        channels=[channel],
    )
    await host.start()
    try:
        await host.receive_message(
            channel,
            ChannelMessage(
                "chat",
                "go",
                sender_id="sender",
                metadata={
                    "sender_id": "attacker",
                    "channel": "evil",
                    "run_id": "spoof",
                    "browser_binding": {"provider": "evil"},
                },
            ),
        )
        await asyncio.gather(*host._tasks.values())
        actual = agent.requests[0].browser_binding
        assert actual.provider == "telegram"
        assert actual.sender_id == "sender"
        assert actual.conversation_id == agent.requests[0].conversation_id
        assert actual.background is False
        assert "sender" not in repr(actual)
    finally:
        await host.stop()


async def test_host_event_bound_and_scheduled_owner_explicit(tmp_path):

    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "test",
            "TALON_BROWSER_SCHEDULED_OWNERS": json.dumps(
                {"job": {"provider": "telegram", "sender_id": "sender"}}
            ),
        },
        base_home=tmp_path,
    )
    host = TalonHost(config=config, agent=BlockingAgent())
    calls = []

    async def event(owner, value):
        calls.append((owner, value))

    host.browser_event_handler = event
    owner = host._scheduled_browser_binding("job", "scheduled-chat")
    assert owner.background is True
    assert owner.sender_id == "sender"
    assert host._scheduled_browser_binding("missing", "chat") is None
    assert host._browser_handler(owner) is None
    foreground = replace(owner, background=False)
    value = BrowserEvent("viewer_unavailable", str(uuid4()))
    await host._browser_handler(foreground)(value)
    assert calls == [(foreground, value)]


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
        await runtime.invoke(AgentRequest("chat", "go", browser_binding=binding()))
        assert [r.get("action") for r in requests] == ["acquire", None, "release"]
        assert all(r["owner"]["background"] is False for r in requests)
    finally:
        await runtime.stop()


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("lease_busy", "browser_busy"),
        ("transport_busy", "browser_busy"),
        ("pending_limit", "browser_busy"),
        ("request_limit", "browser_unavailable"),
        ("lease_busy secret", "browser_unavailable"),
        ({"secret": "lease_busy"}, "browser_unavailable"),
    ],
)
@pytest.mark.parametrize("tool_index", [0, 1])
async def test_tool_errors_allowlist(client, code, expected, tool_index):
    transport(client, [], response=httpx.Response(409, json={"error": code, "details": "secret"}))
    run = client.bind(binding())
    token = set_run(run)
    try:
        args = (
            {"method": "Target.getTargets", "params": {}}
            if tool_index == 0
            else {"reason": "login"}
        )
        result = await browser_tools()[tool_index].ainvoke({**args, "runtime": tool_runtime(run)})
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
    "content", [b"\xff", b'{"error":"lease_busy","details":"' + b"x" * (4 * 1024 * 1024)]
)
async def test_invalid_error_stream_is_sanitized(client, content):
    transport(client, [], response=httpx.Response(409, content=content))
    try:
        with pytest.raises(BrowserError, match=r"^browser_unavailable$"):
            await client.post("actions", {})
    finally:
        await client.stop()


@pytest.mark.parametrize("enabled", [False, True])
async def test_graph_context_schema_opt_in(client, tmp_path, monkeypatch, enabled):
    captured = []

    def build(**kwargs: object):
        captured.append(kwargs)
        return create_deep_agent(**kwargs)

    async def start():
        pass

    monkeypatch.setattr(client, "start", start)
    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", build)
    runtime = DeepAgentRuntime(
        model=ToolModel(responses=[AIMessage(content="done")]),
        browser=client if enabled else None,
        assistant_dir=tmp_path,
        include_web_tools=False,
        skills=(),
        memory=(),
        env={},
    )
    await runtime.start()
    try:
        await runtime.invoke(AgentRequest("chat", "go"))
        assert ("context_schema" in captured[0]) is enabled
        assert ("browser_cdp" in {tool.name for tool in captured[0]["tools"]}) is enabled
    finally:
        await runtime.stop()
