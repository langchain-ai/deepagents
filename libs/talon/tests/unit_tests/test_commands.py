"""Tests for the shared chat command registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from discord.app_commands.commands import validate_name

from deepagents_talon.commands import (
    CHAT_COMMANDS,
    COMMANDS_BY_NAME,
    MAX_SUMMARY_CHARS,
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


def test_registry_is_not_empty_and_has_unique_names():
    names = [command.name for command in CHAT_COMMANDS]

    assert names
    assert len(names) == len(set(names))


def test_command_text_carries_a_leading_slash():
    assert COMMANDS_BY_NAME["mcp-reload"].text == "/mcp-reload"


@pytest.mark.parametrize("command", CHAT_COMMANDS, ids=lambda command: command.name)
def test_command_names_are_valid_discord_command_names(command):
    """Discord rejects a name that is uppercase or has unsupported characters."""
    assert validate_name(command.name) == command.name


@pytest.mark.parametrize("command", CHAT_COMMANDS, ids=lambda command: command.name)
def test_command_summaries_fit_the_discord_description_limit(command):
    assert 0 < len(command.summary) <= MAX_SUMMARY_CHARS


@pytest.mark.parametrize("command", CHAT_COMMANDS, ids=lambda command: command.name)
def test_command_summaries_are_one_line(command):
    assert "\n" not in command.summary


def test_visible_commands_omit_hidden_entries():
    assert all(not command.hidden for command in visible_commands())
    assert "reset-all-history" not in {command.name for command in visible_commands()}


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
