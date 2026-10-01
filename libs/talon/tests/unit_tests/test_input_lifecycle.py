"""Incoming work survives replacement and cannot cross cancellation boundaries."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from deepagents_talon.channels.base import dispatch_message
from deepagents_talon.channels.slack import _SlackInboundMessage
from deepagents_talon.channels.whatsapp import WhatsAppChannel, WhatsAppChannelConfig
from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, AgentResult, ChannelMessage
from deepagents_talon.runtime import DeepAgentRuntime
from tests.channels.test_slack import OPERATOR, _channel
from tests.channels.test_whatsapp import RecordingTransport
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
async def test_control_invalidates_preparing_input(tmp_path: Path, command: str) -> None:
    host, channel, agent = make_host(tmp_path)
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def prepare(message: ChannelMessage) -> ChannelMessage:
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            # A download in a worker thread can finish even after its waiter is cancelled.
            await release.wait()
        finished.set()
        return message

    await host.start()
    try:
        await dispatch_message(
            channel.handler,
            ChannelMessage("chat", "older attachment"),
            provider="test",
            prepare=prepare,
        )
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
async def test_replacement_preserves_input_order(tmp_path: Path, *, stalled: bool) -> None:
    host, channel, agent = make_host(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def prepare(message: ChannelMessage) -> ChannelMessage:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return message

    await host.start()
    try:
        await dispatch_message(
            channel.handler,
            ChannelMessage("chat", "Only work in staging."),
            provider="test",
            prepare=prepare,
        )
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

    async def prepare(message: ChannelMessage) -> ChannelMessage:
        entered.set()
        await release.wait()
        return message

    await host.start()
    try:
        await dispatch_message(
            channel.handler,
            ChannelMessage("chat", "first"),
            provider="test",
            prepare=prepare,
        )
        await entered.wait()
        await channel.receive("rejected")
        assert "not accepted" in channel.sent[-1][1]
        await channel.receive("/stop")
        await channel.receive("later")
        assert (await agent.requests.get()).text == "later"
    finally:
        release.set()
        await host.stop()


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


async def test_slack_stop_bypasses_stalled_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    channel, gateway, _, _ = _channel(tmp_path)
    agent = RecordingAgent()
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path)
    host = TalonHost(config=config, agent=agent, channels=[channel])
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare(message: ChannelMessage, _files: object) -> ChannelMessage:
        entered.set()
        await release.wait()
        return message

    monkeypatch.setattr(channel, "_prepare_inbound_media", prepare)
    await host.start()
    try:
        await gateway.handle_message(
            _SlackInboundMessage(
                channel_id="DCHAT",
                sender_id=OPERATOR,
                text="older attachment",
                ts="1",
                is_dm=True,
                thread_ts=None,
            )
        )
        await entered.wait()
        await gateway.handle_message(
            _SlackInboundMessage(
                channel_id="DCHAT",
                sender_id=OPERATOR,
                text="/stop",
                ts="2",
                is_dm=True,
                thread_ts=None,
            )
        )
        release.set()
        await gateway.handle_message(
            _SlackInboundMessage(
                channel_id="DCHAT",
                sender_id=OPERATOR,
                text="later",
                ts="3",
                is_dm=True,
                thread_ts=None,
            )
        )
        assert (await agent.requests.get()).text == "later"
        assert agent.requests.empty()
    finally:
        release.set()
        await host.stop()


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


async def test_whatsapp_prepares_only_authorized_envelopes(tmp_path: Path) -> None:
    admitted: dict[str, object] = {
        "chat_id": "self",
        "message_id": "allowed",
        "user_id": "self",
        "text": "allowed",
        "from_self": True,
        "self_chat": True,
        "has_media": True,
        "preparation_token": "allowed-token",
    }
    rejected = {
        **admitted,
        "message_id": "rejected",
        "from_self": False,
        "preparation_token": "rejected-token",
    }

    class Transport(RecordingTransport):
        async def post(self, path: str, payload: dict[str, object]) -> object:
            if path in {"/claim", "/release"}:
                return await super().post(path, payload)
            assert path == "/prepare"
            assert payload == {
                "preparation_token": "allowed-token",
                "chat_id": "self",
                "message_id": "allowed",
            }
            self.posts.append((path, payload))
            return {**admitted, "preparation_token": None}

    transport = Transport([rejected, admitted])
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(session_dir=tmp_path, poll_interval_seconds=60),
        transport=transport,
    )
    received: asyncio.Queue[ChannelMessage] = asyncio.Queue()

    async def receive(message: ChannelMessage) -> None:
        received.put_nowait(message)

    channel.set_message_handler(receive)
    await channel.start()
    try:
        assert (await received.get()).text == "allowed"
        assert [path for path, _ in transport.posts].count("/prepare") == 1
        assert received.empty()
    finally:
        await channel.stop()
    released = [
        entry["preparation_token"]
        for path, payload in transport.posts
        if path == "/release"
        for entry in payload["inputs"]
    ]
    assert set(released) == {"allowed-token", "rejected-token"}


class OwnedPreparation:
    def __init__(self) -> None:
        self.ready = asyncio.Event()
        self.released = asyncio.Event()

    async def __call__(self, message: ChannelMessage) -> ChannelMessage:
        await self.ready.wait()
        return message

    def release(self) -> None:
        self.released.set()


@pytest.mark.parametrize("command", ["/stop", "/new", "/reset-all-history"])
async def test_discard_releases_preparing_and_queued_inputs(tmp_path: Path, command: str) -> None:
    host, channel, agent = make_host(tmp_path)
    preparations = [OwnedPreparation(), OwnedPreparation()]
    await host.start()
    try:
        for index, prepare in enumerate(preparations):
            await dispatch_message(
                channel.handler,
                ChannelMessage("chat", str(index)),
                provider="test",
                prepare=prepare,
            )
        await channel.receive(command)
        assert all(prepare.released.is_set() for prepare in preparations)
        assert agent.requests.empty()
    finally:
        await host.stop()


@pytest.mark.parametrize("reason", ["help", "capacity", "completed", "shutdown"])
async def test_host_releases_owned_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    host, channel, agent = make_host(tmp_path)
    prepare = OwnedPreparation()
    if reason == "capacity":
        monkeypatch.setattr("deepagents_talon.host._MAX_CONVERSATION_INPUTS", 0)
    if reason == "completed":
        prepare.ready.set()
    await host.start()
    try:
        await dispatch_message(
            channel.handler,
            ChannelMessage("chat", "/help" if reason == "help" else "hello"),
            provider="test",
            prepare=prepare,
        )
        if reason == "completed":
            assert (await agent.requests.get()).text == "hello"
        if reason == "shutdown":
            await host.stop()
        await prepare.released.wait()
    finally:
        await host.stop()


@pytest.mark.parametrize("stage", ["claim", "dispatch"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_whatsapp_releases_untransferred_batch_on_interruption(
    tmp_path: Path, stage: str, *, cancel: bool
) -> None:
    envelopes = [
        {
            "chat_id": "self",
            "message_id": str(index),
            "user_id": "self",
            "text": "hello",
            "from_self": True,
            "self_chat": True,
            "preparation_token": f"token-{index}",
        }
        for index in range(3)
    ]
    entered = asyncio.Event()

    async def interrupt() -> None:
        entered.set()
        if cancel:
            await asyncio.Event().wait()
        msg = "interrupted batch"
        raise RuntimeError(msg)

    class Transport(RecordingTransport):
        async def post(self, path: str, payload: dict[str, object]) -> object:
            self.posts.append((path, payload))
            if path == "/claim" and stage == "claim":
                await interrupt()
            if path == "/prepare":
                return {**envelopes[int(payload["message_id"])], "preparation_token": None}
            return {"success": True}

    transport = Transport(envelopes)
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(session_dir=tmp_path, poll_interval_seconds=60), transport=transport
    )

    async def receive(_message: ChannelMessage) -> None:
        await interrupt()

    channel.set_message_handler(receive)
    await channel.start()
    try:
        await entered.wait()
        if not cancel:
            with pytest.raises(RuntimeError, match="interrupted batch"):
                await channel._poll
    finally:
        await channel.stop()
    released = {
        entry["preparation_token"]
        for path, payload in transport.posts
        if path == "/release"
        for entry in payload["inputs"]
    }
    assert released == {"token-0", "token-1", "token-2"}
