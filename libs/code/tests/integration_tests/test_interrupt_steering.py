"""Exercise native steering against a local server without model credentials."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from deepagents_code.client.launch.server import ServerProcess, generate_langgraph_json
from deepagents_code.client.steering import (
    SteeredError,
    SteeringControl,
    steerable_stream,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from langchain_core.runnables import RunnableConfig

_GRAPH = '''
import asyncio
from pathlib import Path
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware

class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

@tool
async def blocked_tool() -> str:
    """Wait until cancellation without external side effects."""
    Path(__file__).with_suffix(".started").touch()
    try:
        await asyncio.sleep(120)
    except asyncio.CancelledError:
        Path(__file__).with_suffix(".cancelled").touch()
        raise
    return "unexpected completion"

graph = create_agent(
    Model(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "blocked_tool", "args": {}, "id": "blocked-call",
            "type": "tool_call",
        }]),
        AIMessage(content="replacement complete"),
    ]),
    tools=[blocked_tool],
    middleware=[PatchToolCallsMiddleware()],
)
'''


@pytest.fixture
async def steering_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[ServerProcess, Path]]:
    """Start an isolated loopback server with a deterministic tool-calling model."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "false")
    for name in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "LANGGRAPH_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    from deepagents_code import config

    monkeypatch.setattr(config, "_bootstrap_state", config._BootstrapState())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    config._ensure_bootstrap()
    module = tmp_path / "steering_graph.py"
    module.write_text(_GRAPH)
    generate_langgraph_json(tmp_path, graph_ref="./steering_graph.py:graph")
    server = ServerProcess(config_dir=tmp_path, scaffold=None)
    try:
        await server.start()
        yield server, module
    finally:
        server.stop()


def _wait_for(path: Path) -> None:
    """Wait for the subprocess to reach its tool boundary."""
    deadline = time.monotonic() + 30
    while not path.exists():
        if time.monotonic() >= deadline:
            msg = f"Server did not reach {path.name}"
            raise TimeoutError(msg)
        time.sleep(0.02)


@pytest.mark.timeout(90)
async def test_native_steer_interrupts_tool_and_repairs_history(
    steering_server: tuple[ServerProcess, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real replacement cancels the tool and retains valid conversation history."""
    from deepagents_code.client.remote_client import RemoteAgent

    server, module = steering_server
    agent = RemoteAgent(server.url)
    control = SteeringControl()
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
    original = {"role": "user", "content": "original task", "id": "original"}
    replacement = {"role": "user", "content": "do this instead", "id": "steer"}

    async def workspace(_: object) -> dict[str, object]:
        await asyncio.sleep(0)
        return {}

    monkeypatch.setattr(agent, "_workspace_for_thread", workspace)

    async def consume_original() -> None:
        stream = agent.astream(
            {"messages": [original]}, config=config, steering=control
        )
        assert isinstance(stream, AsyncGenerator)
        async for _ in steerable_stream(stream, control):
            pass

    first = asyncio.create_task(consume_original())
    try:
        await asyncio.to_thread(_wait_for, module.with_suffix(".started"))
        assert control.registered.is_set()
        assert control.submit("do this instead")
        with pytest.raises(SteeredError, match="do this instead"):
            await first
        assert control.detached
        async for _ in agent.astream(
            {"messages": [original, replacement]},
            config=config,
            steering=control,
            multitask_strategy="interrupt",
        ):
            pass
        await asyncio.to_thread(_wait_for, module.with_suffix(".cancelled"))
        state = await agent.aget_state(dict(config))
        messages = state.values["messages"]
        assert [m["content"] for m in messages if m["type"] == "human"] == [
            "original task",
            "do this instead",
        ]
        calls = [call["id"] for m in messages for call in m.get("tool_calls", [])]
        results = [m["tool_call_id"] for m in messages if m["type"] == "tool"]
        assert calls == results == ["blocked-call"]
        assert messages[-1]["content"] == "replacement complete"
    finally:
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        await agent._get_graph().client.http.client.aclose()


@pytest.mark.timeout(90)
async def test_native_steer_before_original_run_executes(
    steering_server: tuple[ServerProcess, Path],
) -> None:
    """A steer before registration preserves the original not-yet-checkpointed input."""
    from langgraph.pregel.remote import RemoteGraph

    server, module = steering_server
    graph = RemoteGraph("agent", url=server.url)
    assert graph.client is not None
    control = SteeringControl()
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
    original = {"role": "user", "content": "original task", "id": "original"}
    replacement = {"role": "user", "content": "do this instead", "id": "steer"}
    assert control.submit("do this instead")

    try:
        stream = graph.astream(
            {"messages": [original]},
            config=config,
            after_seconds=30,
            on_disconnect="continue",
            on_run_created=lambda _: control.registered.set(),
        )
        assert isinstance(stream, AsyncGenerator)
        with pytest.raises(SteeredError, match="do this instead"):
            async for _ in steerable_stream(stream, control):
                pass
        assert control.registered.is_set()
        assert not module.with_suffix(".started").exists()
        before = await graph.client.threads.get_state(
            config["configurable"]["thread_id"]
        )
        assert before["checkpoint"] is None
        assert not before["values"]
        async for _ in graph.astream(
            {"messages": [original, replacement]},
            config=config,
            multitask_strategy="interrupt",
            interrupt_before=["tools"],
        ):
            pass
        after = await graph.aget_state(config)
        assert [
            m["content"] for m in after.values["messages"] if m["type"] == "human"
        ] == [
            "original task",
            "do this instead",
        ]
    finally:
        await graph.client.http.client.aclose()
