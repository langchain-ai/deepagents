"""Tests for the shared chat command registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from deepagents_talon.commands import (
    CHAT_COMMANDS,
    build_help_message,
    visible_commands,
)
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import ChannelMessage
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent, _config

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_talon.commands import ChatCommand


def test_help_message_lists_every_visible_command():
    message = build_help_message()

    for command in visible_commands():
        assert f"{command.text} — {command.summary}" in message


def test_help_message_omits_hidden_commands():
    message = build_help_message()
    hidden = [command for command in CHAT_COMMANDS if command.hidden]

    assert hidden, "expected at least one hidden command to make this meaningful"
    for command in hidden:
        assert f"{command.text} —" not in message


@pytest.mark.parametrize("command", CHAT_COMMANDS, ids=lambda command: command.name)
async def test_registered_commands_reply_without_invoking_agent(
    tmp_path: Path, command: ChatCommand
) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(
            channel, ChannelMessage(conversation_id="chat", text=command.text)
        )
        assert len(channel.sent) == 1
        chat, reply = channel.sent[0]
        assert chat == "chat"
        assert reply
        assert agent.requests == []
    finally:
        await host.stop()
