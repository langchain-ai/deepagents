import asyncio
from pathlib import Path

import pytest

from deepagents_talon.channels.base import ChannelExposure, ExposureMode
from deepagents_talon.channels.whatsapp import WhatsAppChannel, WhatsAppChannelConfig
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import ChannelMessage, ChannelReaction, ReactionChannelAdapter
from tests.channels.test_whatsapp import RecordingTransport
from tests.test_host import ApprovalAgent, _config


@pytest.mark.parametrize(
    ("sender", "emoji", "self_state", "expected"),
    [
        ("operator", "\U0001f44d", (False, False), 1),
        ("operator", "\U0001f44e", (False, False), 1),
        ("stranger", "\U0001f44d", (False, False), 0),
        (None, "\U0001f44d", (False, False), 0),
        ("operator", "", (False, False), 0),
        ("operator", "\U0001f44d", (True, False), 0),
        ("operator", "\U0001f44d", (True, True), 1),
    ],
)
async def test_poll_dispatches_only_authorized_reactions(
    tmp_path: Path,
    sender: str | None,
    emoji: str,
    self_state: tuple[bool, bool],
    expected: int,
) -> None:
    transport = RecordingTransport(
        messages=[
            {
                "event_type": "reaction",
                "chat_id": "chat",
                "user_id": sender,
                "message_id": "approval-prompt",
                "text": emoji,
                "from_self": self_state[0],
                "self_chat": self_state[1],
            }
        ]
    )
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(
            session_dir=tmp_path,
            exposure=ChannelExposure(mode=ExposureMode.OPEN, operator_ids=frozenset({"operator"})),
            poll_interval_seconds=0,
        ),
        transport=transport,
    )
    assert isinstance(channel, ReactionChannelAdapter)
    reactions: list[ChannelReaction] = []
    messages: list[ChannelMessage] = []

    async def receive(reaction: ChannelReaction) -> None:
        reactions.append(reaction)

    async def receive_message(message: ChannelMessage) -> None:
        messages.append(message)

    async def get(path: str) -> object:
        channel._stopped.set()
        return await RecordingTransport.get(transport, path)

    transport.get = get
    channel.set_reaction_handler(receive)
    channel.set_message_handler(receive_message)
    await channel._poll_messages()
    assert messages == []
    assert len(reactions) == expected
    if expected:
        assert reactions[0].message_id == "approval-prompt"
        assert reactions[0].sender_id == "operator"
        assert reactions[0].emoji == emoji


class ApprovalTransport(RecordingTransport):
    def __init__(self) -> None:
        super().__init__()
        self.sent: asyncio.Queue[str] = asyncio.Queue()

    async def post(self, path: str, payload: dict[str, object]) -> object:
        result = await super().post(path, payload)
        if path == "/send":
            self.sent.put_nowait(str(payload["text"]))
        return result


@pytest.mark.parametrize(
    ("emoji", "decision"), [("\U0001f44d\U0001f3fd", "approve"), ("\U0001f44e", "reject")]
)
async def test_whatsapp_reaction_resolves_host_approval(
    tmp_path: Path,
    emoji: str,
    decision: str,
) -> None:
    transport = ApprovalTransport()
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(
            session_dir=tmp_path / "whatsapp",
            exposure=ChannelExposure(operator_ids=frozenset({"operator"})),
            poll_interval_seconds=0.001,
        ),
        transport=transport,
    )
    agent = ApprovalAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(
            channel,
            ChannelMessage(conversation_id="chat", text="run", sender_id="operator"),
        )
        prompt = await asyncio.wait_for(transport.sent.get(), timeout=2)
        assert "Tool approval required." in prompt
        transport.messages.append(
            {
                "event_type": "reaction",
                "chat_id": "chat",
                "user_id": "operator",
                "message_id": "sent",
                "text": emoji,
            }
        )
        result = await asyncio.wait_for(transport.sent.get(), timeout=2)
        assert f"decision:{decision}" in result
        assert len(agent.requests) == 1
    finally:
        await host.stop()
