from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import AIMessage

from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, AgentResult, SendResult
from deepagents_talon.messaging import MESSAGE_HANDLER, send_message
from deepagents_talon.runtime import DeepAgentRuntime
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent, _config

if TYPE_CHECKING:
    from pathlib import Path


async def test_message_requires_channel() -> None:
    assert "unavailable" in await send_message.ainvoke({"text": "update"})


@pytest.mark.parametrize("failure", [False, True])
async def test_message_validation_and_delivery(*, failure: bool) -> None:
    sent: list[str] = []

    async def deliver(text: str) -> SendResult:
        sent.append(text)
        return SendResult(success=not failure, error="private transport detail")

    token = MESSAGE_HANDLER.set(deliver)
    try:
        assert "blank" in await send_message.ainvoke({"text": "  "})
        assert sent == []
        result = await send_message.ainvoke({"text": "update"})
        assert ("Message sent." in result) is not failure
        assert "private" not in result
        assert sent == ["update"]
    finally:
        MESSAGE_HANDLER.reset(token)


async def test_runtime_scopes_progress_to_concurrent_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = asyncio.Barrier(2)
    registered = {}

    class ProgressGraph:
        async def ainvoke(self, _payload: object, config: dict) -> dict:
            await barrier.wait()
            name = config["configurable"]["thread_id"]
            assert "Message sent." in await registered["send_message"].ainvoke({"text": name})
            return {"messages": [AIMessage(content="done")]}

    def create_graph(**kwargs: object) -> ProgressGraph:
        registered.update({item.name: item for item in kwargs["tools"]})
        return ProgressGraph()

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", create_graph)
    runtime = DeepAgentRuntime(model="test:model", include_web_tools=False, skills=(), memory=())
    await runtime.start()
    channels = [RecordingChannel(), RecordingChannel()]
    try:
        results = await asyncio.gather(
            *(
                runtime.invoke(
                    AgentRequest(
                        conversation_id=str(index),
                        text="work",
                        message_handler=lambda text, channel=channel: channel.send_message(
                            "chat", text
                        ),
                    )
                )
                for index, channel in enumerate(channels)
            )
        )
        assert [result.text for result in results] == ["done", "done"]
        assert [channel.sent for channel in channels] == [[("chat", "0")], [("chat", "1")]]
        assert MESSAGE_HANDLER.get() is None
    finally:
        await runtime.stop()


@pytest.mark.parametrize("cancel", [False, True])
async def test_host_progress_precedes_final_and_expires(tmp_path: Path, *, cancel: bool) -> None:
    entered = asyncio.Event()

    class ProgressAgent(BlockingAgent):
        async def invoke(self, request: AgentRequest) -> AgentResult:
            entered.set()
            return await super().invoke(request)

    agent = ProgressAgent()
    channel = RecordingChannel()
    other = RecordingChannel("other")
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel, other])
    await host.start()
    try:
        await channel.receive("block", conversation_id="origin")
        await entered.wait()
        handler = agent.requests[0].message_handler
        assert handler is not None
        assert (await handler("working")).success
        assert channel.sent == [("origin", "working")]
        assert other.sent == []
        if cancel:
            await channel.receive("/stop", conversation_id="origin")
        else:
            agent.released.set()
            await asyncio.gather(*host._tasks.values())
            assert channel.sent[-1] == ("origin", "reply:block")
        assert not (await handler("too late")).success
        assert len(channel.sent) == 2
    finally:
        await host.stop()
