from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from deepagents_talon.channels.base import ChannelExposure, ExposureMode
from deepagents_talon.channels.slack import (
    SlackChannel,
    SlackChannelConfig,
    _SlackInboundCommand,
    _SlackInboundMessage,
)
from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, AgentResult
from tests.archive_helpers import make_runtime, make_saver

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class EchoAgent:
    """Minimal runtime: echoes request text so replies are identifiable."""

    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        return AgentResult(text=f"echo:{request.text}")

    async def recover_interrupted(self, conversation_id: str) -> None:
        del conversation_id


class StubGateway:
    """Delivers inbound Slack events without a Socket Mode connection."""

    def __init__(self) -> None:
        self.bot_id = "UBOT"
        self.posts: list[tuple[str, str, str | None]] = []
        self.handle_message = None
        self.handle_command = None

    async def start(self, *, handle_message, handle_reaction, handle_connection, handle_command):
        del handle_reaction, handle_connection
        self.handle_message = handle_message
        self.handle_command = handle_command

    async def stop(self):
        pass

    async def post_message(self, channel_id, text, *, thread_ts):
        self.posts.append((channel_id, text, thread_ts))
        return f"1700000100.{len(self.posts):06d}"

    async def upload_file(self, channel_id, file_path, *, thread_ts, comment):
        del channel_id, file_path, thread_ts, comment

    async def update_message(self, channel_id, ts, text):
        del channel_id, ts, text


class CapturingResponder:
    def __init__(self) -> None:
        self.rejects: list[str] = []
        self.sends: list[str] = []

    async def reject(self, text):
        self.rejects.append(text)

    async def send(self, text):
        self.sends.append(text)


async def _drain() -> None:
    for _ in range(80):
        await asyncio.sleep(0)


def _host(tmp_path: Path) -> tuple[TalonHost, StubGateway]:
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "assistant"}, base_home=tmp_path)
    gateway = StubGateway()
    channel = SlackChannel(
        SlackChannelConfig(
            bot_token="xoxb-test",  # noqa: S106  # inert test token
            app_token="xapp-test",  # noqa: S106  # inert test token
            inbound_media_dir=tmp_path / "inbound",
            exposure=ChannelExposure(mode=ExposureMode.SELF, operator_ids=frozenset({"UOP"})),
        ),
        gateway=gateway,
    )
    return TalonHost(config=config, agent=EchoAgent(), channels=[channel]), gateway


def _mention(ts: str, *, thread_ts: str | None = None, text: str = "hi") -> _SlackInboundMessage:
    return _SlackInboundMessage(
        channel_id="C1",
        ts=ts,
        thread_ts=thread_ts,
        sender_id="UOP",
        text=text,
        is_dm=False,
    )


async def test_channel_threads_share_archive_without_sharing_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listed: list[list[str]] = []
    found: list[str] = []

    def factory(**kwargs: object):
        tools = {tool.name: tool for tool in kwargs["tools"]}

        async def reply(state):
            summaries = await tools["list_conversations"].ainvoke({"limit": 20})
            listed.append([item["preview"] for item in summaries])
            if state["messages"][-1].text == "second":
                hits = await tools["search_conversations"].ainvoke({"query": "first"})
                found.extend(item["text"] for item in hits["results"])
            return {"messages": [AIMessage("noted")]}

        graph = StateGraph(MessagesState)
        graph.add_node("reply", reply)
        graph.add_edge(START, "reply")
        graph.add_edge("reply", END)
        return graph.compile(checkpointer=kwargs["checkpointer"])

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", factory)
    async with make_saver(tmp_path / "history.sqlite") as saver:
        runtime = make_runtime(saver, tmp_path)
        config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "assistant"}, base_home=tmp_path)
        gateway = StubGateway()
        channel = SlackChannel(
            SlackChannelConfig(
                bot_token="xoxb-test",  # noqa: S106  # inert test token
                app_token="xapp-test",  # noqa: S106  # inert test token
                exposure=ChannelExposure(mode=ExposureMode.SELF, operator_ids=frozenset({"UOP"})),
            ),
            gateway=gateway,
        )
        host = TalonHost(config=config, agent=runtime, channels=[channel])
        await host.start()
        try:

            async def send(
                channel_id: str,
                ts: str,
                text: str,
                *,
                dm: bool = False,
                thread_ts: str | None = None,
            ) -> None:
                await gateway.handle_message(
                    _SlackInboundMessage(
                        channel_id=channel_id,
                        ts=ts,
                        thread_ts=thread_ts,
                        sender_id="UOP",
                        text=text,
                        is_dm=dm,
                    )
                )
                await asyncio.gather(*host._tasks.values())

            await send("C1", "1700000000.000100", "first")
            await send("C1", "1700000000.000200", "second")
            await send("C2", "1700000000.000300", "other")
            await send("D1", "1700000000.000400", "dm", dm=True)
            sessions = await saver.archive.conversations(
                {"talon_history_channel": "slack", "talon_history_chat": "C1"}, limit=20
            )
            assert {entry["preview"] for entry in sessions} == {"first", "second"}
            assert listed[1] == ["first"]
            assert "first" in found
            assert listed[2] == []
            assert listed[3] == []
            await send("C1", "1700000000.000201", "follow up", thread_ts="1700000000.000200")
            assert set(listed[4]) == {"first", "second"}
            checkpoints = [
                await runtime._graph.aget_state(
                    {"configurable": {"thread_id": entry["session_id"]}}
                )
                for entry in sessions
            ]
            assert sorted(
                sum(isinstance(message, HumanMessage) for message in state.values["messages"])
                for state in checkpoints
            ) == [1, 2]
            assert [thread for _, _, thread in gateway.posts] == [
                "1700000000.000100",
                "1700000000.000200",
                "1700000000.000300",
                None,
                "1700000000.000200",
            ]
        finally:
            await host.stop()


async def test_channel_reset_clears_shared_archive_but_new_keeps_siblings(tmp_path: Path) -> None:
    class HistoryAgent(EchoAgent):
        history_enabled = True

        def __init__(self) -> None:
            super().__init__()
            self.cleared: list[tuple[str, str]] = []

        async def clear_history(self, channel: str, chat: str) -> None:
            self.cleared.append((channel, chat))

    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "assistant"}, base_home=tmp_path)
    gateway = StubGateway()
    channel = SlackChannel(
        SlackChannelConfig(
            bot_token="xoxb-test",  # noqa: S106  # inert test token
            app_token="xapp-test",  # noqa: S106  # inert test token
            exposure=ChannelExposure(mode=ExposureMode.SELF, operator_ids=frozenset({"UOP"})),
        ),
        gateway=gateway,
    )
    agent = HistoryAgent()
    host = TalonHost(config=config, agent=agent, channels=[channel])
    await host.start()
    try:
        await gateway.handle_message(_mention("1700000000.000100", text="/new"))
        assert agent.cleared == []
        await gateway.handle_message(_mention("1700000000.000200", text="/reset-all-history"))
        assert agent.cleared == [("slack", "C1")]
        assert gateway.posts[-1][2] == "1700000000.000200"
    finally:
        await host.stop()


async def test_channel_mention_is_answered_in_its_thread(tmp_path: Path) -> None:
    host, gateway = _host(tmp_path)
    await host.start()
    await gateway.handle_message(_mention("1700000000.000100"))
    await _drain()
    await host.stop()

    assert gateway.posts == [("C1", "echo:hi", "1700000000.000100")]


async def test_thread_follow_up_keeps_the_thread_conversation(tmp_path: Path) -> None:
    host, gateway = _host(tmp_path)
    agent = host.agent
    await host.start()
    await gateway.handle_message(_mention("1700000000.000100", text="first"))
    await _drain()
    await gateway.handle_message(
        _mention("1700000000.000200", thread_ts="1700000000.000100", text="second"),
    )
    await _drain()
    await gateway.handle_message(_mention("1700000000.000300", text="other thread"))
    await _drain()
    await host.stop()

    threads = [request.conversation_id for request in agent.requests]
    assert threads[0] == threads[1]
    assert threads[2] != threads[0]
    assert [thread_ts for _, _, thread_ts in gateway.posts] == [
        "1700000000.000100",
        "1700000000.000100",
        "1700000000.000300",
    ]


async def test_talon_help_is_answered_through_the_command(tmp_path: Path) -> None:
    host, gateway = _host(tmp_path)
    responder = CapturingResponder()
    await host.start()
    await gateway.handle_command(
        _SlackInboundCommand(
            command="help",
            channel_id="D1",
            sender_id="UOP",
            trigger_id="trig-1",
            responder=responder,
        ),
    )
    await _drain()
    await host.stop()

    assert len(responder.sends) == 1
    assert "/new" in responder.sends[0]
    assert gateway.posts == []
