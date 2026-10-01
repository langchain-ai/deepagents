"""Incoming work survives replacement and cannot cross cancellation boundaries."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, AgentResult, ChannelMessage
from deepagents_talon.runtime import DeepAgentRuntime
from tests.conftest import RecordingChannel

if TYPE_CHECKING:
    from pathlib import Path


class RecordingAgent:
    def __init__(self) -> None:
        self.requests: asyncio.Queue[AgentRequest] = asyncio.Queue()

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def recover_interrupted(self, conversation_id: str) -> None:
        pass

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.put_nowait(request)
        return AgentResult(text="done")

    @property
    def history_enabled(self) -> bool:
        return True

    async def clear_history(self, channel: str, chat: str) -> None:
        pass


def make_host(tmp_path: Path) -> tuple[TalonHost, RecordingChannel, RecordingAgent]:
    channel = RecordingChannel()
    agent = RecordingAgent()
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path)
    return TalonHost(config=config, agent=agent, channels=[channel]), channel, agent


@pytest.mark.parametrize("command", ["/stop", "/new", "/reset-all-history"])
async def test_control_invalidates_preparing_input(
    tmp_path: Path, command: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    host, channel, agent = make_host(tmp_path)
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def prepare(_transcriber: object, message: ChannelMessage) -> ChannelMessage:
        if message.text != "older attachment":
            return message

        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            # A download in a worker thread can finish even after its waiter is cancelled.
            await release.wait()
        finished.set()
        return message

    monkeypatch.setattr("deepagents_talon.host.transcribe_voice_message", prepare)
    await host.start()
    try:
        await channel.receive("older attachment")
        await entered.wait()
        await channel.receive("independent", conversation_id="other")
        assert (await agent.requests.get()).text == "independent"
        await channel.receive(command)
        release.set()
        await finished.wait()
        await channel.receive("later")
        assert (await agent.requests.get()).text == "later"
        assert agent.requests.empty()
    finally:
        release.set()
        await host.stop()


@pytest.mark.parametrize("stalled", [False, True])
async def test_replacement_preserves_input_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, stalled: bool
) -> None:
    host, channel, agent = make_host(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def prepare(_transcriber: object, message: ChannelMessage) -> ChannelMessage:
        nonlocal calls
        if message.text != "Only work in staging.":
            return message
        calls += 1
        entered.set()
        await release.wait()
        return message

    monkeypatch.setattr("deepagents_talon.host.transcribe_voice_message", prepare)
    await host.start()
    try:
        await channel.receive("Only work in staging.")
        if stalled:
            await entered.wait()
        await channel.receive("Now run cleanup.")
        release.set()
        request = await agent.requests.get()
        assert request.text == "Only work in staging.\n\nNow run cleanup."
        inputs = request.metadata["talon_inputs"]
        assert isinstance(inputs, list)
        assert [item["content"] for item in inputs] == [
            "Only work in staging.",
            "Now run cleanup.",
        ]
        assert len({item["id"] for item in inputs}) == 2
        assert calls == 1
    finally:
        release.set()
        await host.stop()


async def test_capacity_rejection_leaves_control_responsive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deepagents_talon.host._MAX_CONVERSATION_INPUTS", 1)
    host, channel, agent = make_host(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare(_transcriber: object, message: ChannelMessage) -> ChannelMessage:
        if message.text != "first":
            return message

        entered.set()
        await release.wait()
        return message

    monkeypatch.setattr("deepagents_talon.host.transcribe_voice_message", prepare)
    await host.start()
    try:
        await channel.receive("first")
        await entered.wait()
        await channel.receive("rejected")
        assert "not accepted" in channel.sent[-1][1]
        await channel.receive("/stop")
        await channel.receive("later")
        assert (await agent.requests.get()).text == "later"
    finally:
        release.set()
        await host.stop()


@pytest.mark.parametrize("with_callback", [False, True])
async def test_admitted_inputs_without_checkpointing(
    monkeypatch: pytest.MonkeyPatch, *, with_callback: bool
) -> None:
    seen: list[tuple[str | None, object]] = []

    async def model(state: MessagesState) -> dict[str, list[AIMessage]]:
        seen.extend((message.id, message.content) for message in state["messages"])
        return {"messages": [AIMessage(content="done")]}

    builder = StateGraph(MessagesState)
    builder.add_node("model", model)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)
    graph = builder.compile(checkpointer=False)
    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", lambda **_kwargs: graph)
    runtime = DeepAgentRuntime(
        model="test:model", checkpointer=False, include_web_tools=False, skills=(), memory=()
    )
    committed = asyncio.Event()
    await runtime.start()
    try:
        result = await runtime.invoke(
            AgentRequest(
                "chat",
                "staging only\n\ncleanup",
                metadata={
                    "talon_inputs": [
                        {"role": "user", "id": "first", "content": "staging only"},
                        {"role": "user", "id": "second", "content": "cleanup"},
                    ]
                },
                inputs_committed=committed.set if with_callback else None,
            )
        )
        assert result.text == "done"
        assert seen == [("first", "staging only"), ("second", "cleanup")]
        assert not committed.is_set()
    finally:
        await runtime.stop()


async def test_checkpointed_input_is_not_replayed(monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def model(_state: MessagesState) -> dict[str, list[AIMessage]]:
        entered.set()
        await release.wait()
        return {"messages": [AIMessage(content="done")]}

    builder = StateGraph(MessagesState)
    builder.add_node("model", model)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", lambda **_kwargs: graph)
    runtime = DeepAgentRuntime(model="test:model", include_web_tools=False, skills=(), memory=())
    await runtime.start()
    committed = asyncio.Event()
    first = {"role": "user", "id": "first", "content": "staging only"}
    second = {"role": "user", "id": "second", "content": "cleanup"}
    task = asyncio.create_task(
        runtime.invoke(
            AgentRequest(
                "chat",
                "staging only",
                metadata={"talon_inputs": [first]},
                inputs_committed=committed.set,
            )
        )
    )
    try:
        await entered.wait()
        assert committed.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await runtime.recover_interrupted("chat")
        release.set()
        await runtime.invoke(
            AgentRequest(
                "chat",
                "staging only\n\ncleanup",
                metadata={"talon_inputs": [first, second]},
            )
        )
        snapshot = await graph.aget_state({"configurable": {"thread_id": "chat"}})
        messages = snapshot.values["messages"]
        assert [message.content for message in messages if message.id in {"first", "second"}] == [
            "staging only",
            "cleanup",
        ]
        assert (
            sum(isinstance(message, HumanMessage) and message.id == "first" for message in messages)
            == 1
        )
        # Committed history stays ahead of the interruption marker, not moved to the new turn.
        assert messages[0].id == "first"
    finally:
        release.set()
        await runtime.stop()


async def test_committed_input_releases_capacity_before_model_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deepagents_talon.host._MAX_CONVERSATION_INPUTS", 1)
    seen: asyncio.Queue[list[str]] = asyncio.Queue()
    release = asyncio.Event()

    async def model(state: MessagesState) -> dict[str, list[AIMessage]]:
        seen.put_nowait(
            [
                str(message.content)
                for message in state["messages"]
                if isinstance(message, HumanMessage)
                and message.content in {"staging only", "cleanup"}
            ]
        )
        await release.wait()
        return {"messages": [AIMessage(content="done")]}

    builder = StateGraph(MessagesState)
    builder.add_node("model", model)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", lambda **_kwargs: graph)
    runtime = DeepAgentRuntime(model="test:model", include_web_tools=False, skills=(), memory=())
    channel = RecordingChannel()
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path)
    host = TalonHost(config=config, agent=runtime, channels=[channel])
    await host.start()
    try:
        await channel.receive("staging only")
        assert await seen.get() == ["staging only"]
        await channel.receive("cleanup")
        assert await seen.get() == ["staging only", "cleanup"]
        assert not any("not accepted" in text for _, text in channel.sent)
    finally:
        release.set()
        await host.stop()
