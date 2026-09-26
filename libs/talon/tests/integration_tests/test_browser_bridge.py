from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from deepagents_talon.browser import BrowserClient, BrowserError
from deepagents_talon.config import TalonConfig
from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.runtime import DeepAgentRuntime
from tests.unit_tests.test_background_runtime import ToolModel

pytestmark = pytest.mark.allow_hosts(["127.0.0.1"])

_SOURCE = Path(__file__).resolve().parents[2] / "deepagents_talon/steel_runtime"
_SERVER = """
const { createBridge } = await import(process.argv[1]);
const { Coordinator } = await import(process.argv[2]);
const coordinator = new Coordinator();
const acquire = coordinator.acquire.bind(coordinator);
coordinator.acquire = (...args) => {
  const run = acquire(...args);
  run.transport ??= {
    command: async (method, params, session) => ({ method, params, session }),
    close: async () => {}, abort: () => {},
  };
  return run;
};
const bridge = createBridge({ token: 'x'.repeat(43), coordinator,
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


async def test_runtime_commands_and_cleanup_against_node(bridge_client, tmp_path):
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
        await runtime.invoke(AgentRequest("chat", "go"))
        state = await runtime._graph.aget_state({"configurable": {"thread_id": "chat"}})
        observation = next(
            json.loads(message.content)["untrusted_browser_observation"]
            for message in state.values["messages"]
            if message.type == "tool"
        )
        assert observation == {
            "method": "Runtime.evaluate",
            "params": {"expression": "1 + 1"},
            "session": "attached-session",
        }
        run = bridge_client.bind()
        await run.command("Target.getTargets", {}, None)
        await run.close()
    finally:
        await runtime.stop()


async def test_competing_runs_cannot_release_each_other(bridge_client):
    await bridge_client.start()
    run, contender = bridge_client.bind(), bridge_client.bind()
    try:
        await run.command("Target.getTargets", {}, None)
        with pytest.raises(BrowserError, match="browser_busy"):
            await contender.command("Target.getTargets", {}, None)
        await contender.close()
        with pytest.raises(BrowserError, match="browser_busy"):
            await contender.command("Target.getTargets", {}, None)
        await run.close()
        await contender.command("Target.getTargets", {}, None)
    finally:
        await run.close()
        await contender.close()


@pytest.mark.parametrize("cancel_cleanup", [False, True])
async def test_cancel_command_before_response_parsed(bridge_client, cancel_cleanup):
    await bridge_client.start()
    run = bridge_client.bind()
    received, releasing, resume = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def intercept(response):
        if response.request.url.path.endswith("/command"):
            received.set()
            await asyncio.Event().wait()
        elif cancel_cleanup:
            releasing.set()
            await resume.wait()

    bridge_client._http.event_hooks["response"] = [intercept]

    async def invoke():
        try:
            await run.command("Target.getTargets", {}, None)
        finally:
            await run.close()

    task = asyncio.create_task(invoke())
    await asyncio.wait_for(received.wait(), 5)
    task.cancel()
    if cancel_cleanup:
        await asyncio.wait_for(releasing.wait(), 5)
        task.cancel()
        resume.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    bridge_client._http.event_hooks["response"] = []
    next_run = bridge_client.bind()
    await next_run.command("Target.getTargets", {}, None)
    await next_run.close()
