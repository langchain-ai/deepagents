"""Tests for the shared chat command registry."""

from __future__ import annotations

import pytest
from discord.app_commands.commands import validate_name

from deepagents_talon import commands as chat_commands, host
from deepagents_talon.commands import (
    CHAT_COMMANDS,
    COMMANDS_BY_NAME,
    MAX_SUMMARY_CHARS,
    build_help_message,
    visible_commands,
)


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


def test_host_dispatch_constants_match_the_registry():
    """The host dispatches on literals; drift here would silently break a command."""
    assert host._HELP_COMMAND == chat_commands.HELP
    assert host._NEW_COMMAND == chat_commands.NEW
    assert host._STOP_COMMAND == chat_commands.STOP
    assert host._MCP_RELOAD_COMMAND == chat_commands.MCP_RELOAD
    assert host._RESET_ALL_HISTORY_COMMAND == chat_commands.RESET_ALL_HISTORY


def test_every_registry_command_is_dispatched_by_the_host():
    """A registered command with no host branch would silently reach the agent."""
    dispatched = {
        host._HELP_COMMAND,
        host._NEW_COMMAND,
        host._STOP_COMMAND,
        host._MCP_RELOAD_COMMAND,
        host._RESET_ALL_HISTORY_COMMAND,
    }

    assert {command.text for command in CHAT_COMMANDS} == dispatched


def test_host_help_message_comes_from_the_registry():
    assert build_help_message() == host._HELP_MESSAGE
