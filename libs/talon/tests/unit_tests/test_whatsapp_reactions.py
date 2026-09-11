import asyncio
import json
import logging
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


@pytest.mark.parametrize(
    ("from_self", "self_chat", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
        ("true", True, False),
        (True, "true", False),
    ],
)
async def test_self_reactions_without_configured_operators(
    tmp_path: Path, *, from_self: bool | str, self_chat: bool | str, expected: bool
) -> None:
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(session_dir=tmp_path, exposure=ChannelExposure()),
        transport=RecordingTransport(),
    )
    reactions: list[ChannelReaction] = []

    async def receive(reaction: ChannelReaction) -> None:
        reactions.append(reaction)

    channel.set_reaction_handler(receive)
    message = ChannelMessage(
        conversation_id="chat",
        sender_id="paired-account",
        message_id="prompt",
        text="\U0001f44d",
        metadata={"from_self": from_self, "self_chat": self_chat},
    )
    assert await channel._dispatch_reaction(message) is expected
    assert len(reactions) == int(expected)


@pytest.mark.parametrize(
    "reason",
    [
        "sender_not_operator",
        "self_outside_self_chat",
        "missing_message_id",
        "missing_emoji",
        "missing_handler",
        None,
        "dispatch_failed",
    ],
)
async def test_reaction_diagnostics_are_private(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, reason: str | None
) -> None:
    caplog.set_level(logging.DEBUG, logger="deepagents_talon.channels.whatsapp")
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(
            session_dir=tmp_path,
            exposure=ChannelExposure(operator_ids=frozenset({"private-sender"})),
        ),
        transport=RecordingTransport(),
    )

    async def receive(_reaction: ChannelReaction) -> None:
        if reason == "dispatch_failed":
            msg = "private-exception"
            raise ValueError(msg)

    if reason != "missing_handler":
        channel.set_reaction_handler(receive)
    message = ChannelMessage(
        conversation_id="private-chat",
        sender_id="private-stranger" if reason == "sender_not_operator" else "private-sender",
        message_id=None if reason == "missing_message_id" else "private-message",
        text="" if reason == "missing_emoji" else "private-emoji",
        metadata={"from_self": reason == "self_outside_self_chat", "self_chat": False},
    )
    if reason == "dispatch_failed":
        with pytest.raises(ValueError, match="private-exception"):
            await channel._dispatch_reaction(message)
    else:
        assert await channel._dispatch_reaction(message) is (reason is None)
    events = [
        json.loads(record.message.split("talon_event ", 1)[1])
        for record in caplog.records
        if "talon_event " in record.message
    ]
    assert events[0]["event"] == "whatsapp.inbound.reaction.received"
    if reason is None or reason == "dispatch_failed":
        outcome = "dispatched" if reason is None else "dispatch_failed"
        assert events[-1]["event"] == f"whatsapp.inbound.reaction.{outcome}"
    else:
        assert events[-1]["reason"] == reason
        assert events[-1]["event"] == "whatsapp.inbound.reaction.rejected"
    assert "private-" not in caplog.text


class ApprovalTransport(RecordingTransport):
    def __init__(self) -> None:
        super().__init__()
        self.sent: asyncio.Queue[str] = asyncio.Queue()
        self.send_count = 0

    async def post(self, path: str, payload: dict[str, object]) -> object:
        result = await super().post(path, payload)
        if path == "/send":
            self.send_count += 1
            self.sent.put_nowait(str(payload["text"]))
            return {"success": True, "message_id": f"sent-{self.send_count}"}
        return result


@pytest.mark.parametrize(
    ("emoji", "decision"), [("\U0001f44d\U0001f3fd", "approve"), ("\U0001f44e", "reject")]
)
@pytest.mark.parametrize("self_chat", [False, True])
async def test_whatsapp_reaction_resolves_host_approval(
    tmp_path: Path,
    emoji: str,
    decision: str,
    *,
    self_chat: bool,
) -> None:
    transport = ApprovalTransport()
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(
            session_dir=tmp_path / "whatsapp",
            exposure=ChannelExposure(
                operator_ids=frozenset() if self_chat else frozenset({"operator"})
            ),
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
        for text in ("how is it going?", "one more thing"):
            transport.messages.append(
                {
                    "chat_id": "chat",
                    "user_id": "operator",
                    "message_id": "follow-up",
                    "text": text,
                    "from_self": self_chat,
                    "self_chat": self_chat,
                }
            )
            reminder = await asyncio.wait_for(transport.sent.get(), timeout=2)
            assert reminder == prompt
            assert len(agent.requests) == 1
            assert agent.recoveries == []
        transport.messages.append(
            {
                "event_type": "reaction",
                "chat_id": "chat",
                "user_id": "operator",
                "message_id": "sent-3",
                "text": emoji,
                "from_self": self_chat,
                "self_chat": self_chat,
            }
        )
        result = await asyncio.wait_for(transport.sent.get(), timeout=2)
        assert f"decision:{decision}" in result
        assert len(agent.requests) == 1
    finally:
        await host.stop()
