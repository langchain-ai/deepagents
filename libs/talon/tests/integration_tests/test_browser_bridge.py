from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage

from deepagents_talon.browser import BrowserBinding, BrowserClient, BrowserError
from deepagents_talon.config import TalonConfig
from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.runtime import DeepAgentRuntime
from tests.unit_tests.test_background_runtime import ToolModel

_SOURCE = Path(__file__).resolve().parents[4] / "examples/talon/browser"
_SERVER = """
const { createBridge } = await import(process.argv[1]);
const { Coordinator } = await import(process.argv[2]);
const coordinator = new Coordinator({ operator: 'op', identities: { telegram: 'sender' } });
const create = coordinator.create.bind(coordinator);
coordinator.create = (...args) => {
  create(...args);
  coordinator.lease.transport = {
    command: async (method, params, session) => ({ method, params, session }),
    close: async () => {}, abort: () => {},
  };
};
class WebSocketServer { close() {} }
const bridge = createBridge({ token: 'x'.repeat(43), coordinator, WebSocketServer,
  controlHost: '127.0.0.1', controlPort: 0, viewerHost: '127.0.0.1', viewerPort: 0 });
await bridge.start();
console.log(`http://127.0.0.1:${bridge.control.address().port}`);
process.stdin.resume();
process.stdin.on('end', async () => { await bridge.close(); });
"""


@pytest.fixture
async def bridge_client(tmp_path, monkeypatch):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the native bridge contract test")
    process = await asyncio.create_subprocess_exec(
        node,
        "--input-type=module",
        "-e",
        _SERVER,
        (_SOURCE / "bridge.mjs").as_uri(),
        (_SOURCE / "coordinator.mjs").as_uri(),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    client = None
    try:
        async with asyncio.timeout(10):
            address = (await process.stdout.readline()).decode().strip()
        assert address.startswith("http://127.0.0.1:"), (await process.stderr.read()).decode()
        monkeypatch.setattr("deepagents_talon.browser._CONTROL", address)
        token = tmp_path / "token"
        token.write_text("x" * 43)
        token.chmod(0o400)
        config = TalonConfig.from_env(
            {
                "AGENT_ASSISTANT_ID": "test",
                "TALON_BROWSER_ENABLED": "true",
                "TALON_BROWSER_OPERATOR_ID": "op",
                "TALON_BROWSER_IDENTITIES": '{"telegram":"sender"}',
                "TALON_BROWSER_TOKEN_FILE": str(token),
            },
            base_home=tmp_path,
        )
        client = BrowserClient(config.env)
        yield client
    finally:
        if client is not None:
            await client.stop()
        process.stdin.close()
        try:
            async with asyncio.timeout(5):
                await process.wait()
        except TimeoutError:
            process.kill()
            await process.wait()


@pytest.mark.parametrize("background", [False, True])
async def test_runtime_handoff_against_node(bridge_client, tmp_path, background):
    events = []

    async def event(value):
        events.append(value)

    model = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "browser_cdp",
                        "id": "cdp",
                        "args": {
                            "method": "Runtime.evaluate",
                            "params": {"expression": "1 + 1"},
                            "session_id": "attached-session",
                        },
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "browser_request_handoff",
                        "id": "handoff",
                        "args": {"reason": "secret login reason"},
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    runtime = DeepAgentRuntime(
        model=model,
        browser=bridge_client,
        assistant_dir=tmp_path,
        include_web_tools=False,
        skills=(),
        memory=(),
        env={},
    )
    await runtime.start()
    try:
        result = await runtime.invoke(
            AgentRequest(
                "chat",
                "go",
                browser_binding=BrowserBinding("telegram", "sender", "chat", background),
                browser_event_handler=event,
            )
        )
        assert result is not None
        state = await runtime._graph.aget_state({"configurable": {"thread_id": "chat"}})
        observations = [
            json.loads(message.content)["untrusted_browser_observation"]
            for message in state.values["messages"]
            if message.type == "tool" and message.name in {"browser_cdp", "browser_request_handoff"}
        ]
        assert observations[0] == {
            "method": "Runtime.evaluate",
            "params": {"expression": "1 + 1"},
            "session": "attached-session",
        }
        handoff = observations[1]
        assert handoff["status"] == ("human_required" if background else "viewer_unavailable")
        assert handoff["mode"] == "PAUSED"
        UUID(handoff["handoff_id"])
        assert len(events) == (0 if background else 1)
        assert "secret" not in json.dumps(observations)
        next_run = bridge_client.bind(BrowserBinding("telegram", "sender", "next"))
        await next_run.action("acquire")
        await next_run.close()
    finally:
        await runtime.stop()


@pytest.mark.parametrize("cancel_cleanup", [False, True])
async def test_cancel_handoff_before_response_parsed(bridge_client, cancel_cleanup):
    await bridge_client.start()
    run = bridge_client.bind(BrowserBinding("telegram", "sender", "chat"))
    await run.action("acquire")
    original = dict(run.lease)
    received, releasing = asyncio.Event(), asyncio.Event()
    resume = asyncio.Event()

    async def intercept(response):
        action = json.loads(response.request.content)["action"]
        if action == "handoff":
            assert response.status_code == 200
            received.set()
            await asyncio.Event().wait()
        if action == "release" and cancel_cleanup:
            releasing.set()
            await resume.wait()

    bridge_client._http.event_hooks["response"] = [intercept]

    async def invoke():
        try:
            await run.action("handoff")
        finally:
            await run.close()

    task = asyncio.create_task(invoke())
    async with asyncio.timeout(10):
        await received.wait()
        assert run.lease == original
        task.cancel()
        if cancel_cleanup:
            await releasing.wait()
            task.cancel()
            resume.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not run.lease
        next_run = bridge_client.bind(BrowserBinding("telegram", "sender", "next"))
        await next_run.action("acquire")
        assert next_run.lease["generation"] > original["generation"]
        await next_run.close()


async def test_node_busy_and_task_command_quota(bridge_client):
    await bridge_client.start()
    run = bridge_client.bind(BrowserBinding("telegram", "sender", "chat"))
    contender = bridge_client.bind(BrowserBinding("telegram", "sender", "other"))
    try:
        await run.action("acquire")
        with pytest.raises(BrowserError, match=r"^browser_busy$"):
            await contender.action("acquire")
        for _ in range(256):
            await run.command("Target.getTargets", {}, None)
        with pytest.raises(BrowserError, match=r"^browser_unavailable$"):
            await run.command("Target.getTargets", {}, None)
    finally:
        await run.close()
