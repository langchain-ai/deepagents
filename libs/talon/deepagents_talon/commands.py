"""Chat command registry shared by the host and channel adapters.

Talon is an experimental runtime and is subject to change or removal at any time.

The host parses commands out of inbound message text, so this module is the one
place that knows which commands exist and how to describe them. Channel adapters
that can register commands with their platform -- Discord application commands,
for example -- build their registration from the same entries, so a new command
cannot appear in chat without also appearing wherever it is advertised.

This module deliberately imports nothing from the rest of Talon: both
`deepagents_talon.host` and the channel adapters depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

MAX_SUMMARY_CHARS = 100
"""Longest command summary Discord accepts as an application command description."""

_HELP_PREAMBLE = (
    "Talon is your personal agent in chat. Send a message to ask for help or get work done; "
    "ask for reminders or recurring tasks to schedule them. "
    "Each conversation keeps its context."
)
_HELP_TRAILER = (
    "MCP: Ask to view, add, update, or remove a server (Linux/macOS), "
    "then approve the change when prompted. Updated tools are available next turn.\n"
    "OAuth: Ask to authenticate a configured MCP server. Open the sign-in link, "
    "follow the prompts, and paste the full callback URL into the same chat when asked. "
    "Send /stop to cancel."
)


@dataclass(frozen=True, slots=True)
class ChatCommand:
    """One command a user can send in a chat.

    Args:
        name: Command name without its leading slash, for example `mcp-reload`.
            Lowercase, because platform command names are case-insensitive and
            Discord rejects anything else.
        summary: One-line description, reused verbatim for the `/help` listing and
            for platform command registration. At most `MAX_SUMMARY_CHARS`.
        hidden: Whether to leave the command out of `/help` and out of platform
            registration. A hidden command still runs when typed.
    """

    name: str
    summary: str
    hidden: bool = False

    @property
    def text(self) -> str:
        """Command as it appears in message text, including the leading slash."""
        return f"/{self.name}"


CHAT_COMMANDS: tuple[ChatCommand, ...] = (
    ChatCommand("help", "Show this guide."),
    ChatCommand("new", "Stop current work and start a fresh conversation."),
    ChatCommand("stop", "Stop current work."),
    ChatCommand("mcp-reload", "Reload MCP configuration after manual edits."),
    ChatCommand(
        "reset-all-history",
        "Delete this chat's stored history and start fresh. Cannot be undone.",
        # Deliberately unadvertised. Registering this with a platform would put an
        # irreversible deletion one keystroke into an autocomplete menu, and Talon
        # has no confirmation step to catch a mistaken invocation.
        hidden=True,
    ),
)
"""Every command the host dispatches, in the order `/help` lists them."""

HELP = CHAT_COMMANDS[0].text
NEW = CHAT_COMMANDS[1].text
STOP = CHAT_COMMANDS[2].text
MCP_RELOAD = CHAT_COMMANDS[3].text
RESET_ALL_HISTORY = CHAT_COMMANDS[4].text

COMMANDS_BY_NAME: Mapping[str, ChatCommand] = MappingProxyType(
    {command.name: command for command in CHAT_COMMANDS},
)
"""Commands keyed by bare name, for resolving a platform command invocation."""


def visible_commands() -> tuple[ChatCommand, ...]:
    """Return the commands to advertise in `/help` and to platform registries.

    Returns:
        Every command that is not hidden, in registry order.
    """
    return tuple(command for command in CHAT_COMMANDS if not command.hidden)


def build_help_message() -> str:
    """Build the `/help` response.

    Returns:
        Guide text listing every visible command with its summary.
    """
    listing = "\n".join(f"{command.text} — {command.summary}" for command in visible_commands())
    return f"{_HELP_PREAMBLE}\n\n{listing}\n\n{_HELP_TRAILER}"
