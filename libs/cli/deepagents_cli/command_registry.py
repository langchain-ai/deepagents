"""Unified slash-command registry.

Every slash command is declared once as a `SlashCommand` entry in `COMMANDS`.
Bypass-tier frozensets and autocomplete tuples are derived automatically — no
other file should hard-code command metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents_cli.skills.load import ExtendedSkillMetadata


class BypassTier(StrEnum):
    """Classification that controls whether a command can skip the message queue."""

    ALWAYS = "always"
    """Execute regardless of any busy state, including mid-thread-switch."""

    CONNECTING = "connecting"
    """Bypass only during initial server connection, not during agent/shell."""

    IMMEDIATE_UI = "immediate_ui"
    """Open modal UI immediately; real work deferred via `_defer_action` callback."""

    SIDE_EFFECT_FREE = "side_effect_free"
    """Execute the side effect immediately; defer chat output until idle."""

    QUEUED = "queued"
    """Must wait in the queue when the app is busy."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SlashCommand:
    """A single slash-command definition."""

    name: str
    """Canonical command name (e.g. `/quit`)."""

    description: str
    """Short user-facing description."""

    bypass_tier: BypassTier
    """Queue-bypass classification."""

    hidden_keywords: str = ""
    """Space-separated terms for fuzzy matching (never displayed)."""

    aliases: tuple[str, ...] = ()
    """Alternative names (e.g. `("/q",)` for `/quit`)."""


COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand(
        name="/clear",
        description="Clear chat and start new thread",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="reset",
    ),
    SlashCommand(
        name="/editor",
        description="Open prompt in external editor ($EDITOR)",
        bypass_tier=BypassTier.QUEUED,
    ),
    SlashCommand(
        name="/mcp",
        description="Show active MCP servers and tools",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
        hidden_keywords="servers",
    ),
    SlashCommand(
        name="/model",
        description="Switch or configure model (--model-params, --default)",
        bypass_tier=BypassTier.IMMEDIATE_UI,
    ),
    SlashCommand(
        name="/offload",
        description="Free up context window space by offloading older messages",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="compact",
        aliases=("/compact",),
    ),
    SlashCommand(  # Static alias; not auto-generated from skill discovery
        name="/remember",
        description="Update memory and skills from conversation",
        bypass_tier=BypassTier.QUEUED,
    ),
    SlashCommand(  # Static alias; not auto-generated from skill discovery
        name="/skill-creator",
        description="Guide for creating effective agent skills",
        bypass_tier=BypassTier.QUEUED,
    ),
    SlashCommand(
        name="/threads",
        description="Browse and resume previous threads",
        bypass_tier=BypassTier.IMMEDIATE_UI,
        hidden_keywords="continue history sessions",
    ),
    SlashCommand(
        name="/trace",
        description="Open current thread in LangSmith",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
    ),
    SlashCommand(
        name="/tokens",
        description="Token usage",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="cost",
    ),
    SlashCommand(
        name="/reload",
        description="Reload config from environment variables and .env",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="refresh",
    ),
    SlashCommand(
        name="/theme",
        description="Switch color theme",
        bypass_tier=BypassTier.IMMEDIATE_UI,
        hidden_keywords="dark light color appearance",
    ),
    SlashCommand(
        name="/update",
        description="Check for and install updates",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="upgrade",
    ),
    SlashCommand(
        name="/auto-update",
        description="Toggle automatic updates on or off",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
    ),
    SlashCommand(
        name="/changelog",
        description="Open changelog in browser",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
    ),
    SlashCommand(
        name="/version",
        description="Show version",
        bypass_tier=BypassTier.CONNECTING,
    ),
    SlashCommand(
        name="/feedback",
        description="Submit a bug report or feature request",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
    ),
    SlashCommand(
        name="/docs",
        description="Open documentation in browser",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
    ),
    SlashCommand(
        name="/help",
        description="Show help",
        bypass_tier=BypassTier.QUEUED,
    ),
    SlashCommand(
        name="/quit",
        description="Exit app",
        bypass_tier=BypassTier.ALWAYS,
        hidden_keywords="close leave",
        aliases=("/q",),
    ),
)
"""All slash commands."""


# ---------------------------------------------------------------------------
# Derived bypass-tier frozensets
# ---------------------------------------------------------------------------


def _build_bypass_set(tier: BypassTier) -> frozenset[str]:
    """Build a frozenset of command names (including aliases) for a tier.

    Args:
        tier: The bypass tier to collect.

    Returns:
        Frozenset of all names and aliases that belong to `tier`.
    """
    names: set[str] = set()
    for cmd in COMMANDS:
        if cmd.bypass_tier == tier:
            names.add(cmd.name)
            names.update(cmd.aliases)
    return frozenset(names)


ALWAYS_IMMEDIATE: frozenset[str] = _build_bypass_set(BypassTier.ALWAYS)
"""Commands that execute regardless of any busy state."""

BYPASS_WHEN_CONNECTING: frozenset[str] = _build_bypass_set(BypassTier.CONNECTING)
"""Commands that bypass only during initial server connection."""

IMMEDIATE_UI: frozenset[str] = _build_bypass_set(BypassTier.IMMEDIATE_UI)
"""Commands that open modal UI immediately, deferring real work."""

SIDE_EFFECT_FREE: frozenset[str] = _build_bypass_set(BypassTier.SIDE_EFFECT_FREE)
"""Commands whose side effect fires immediately; chat output deferred until idle."""

QUEUE_BOUND: frozenset[str] = _build_bypass_set(BypassTier.QUEUED)
"""Commands that must wait in the queue when the app is busy."""

HIDDEN_DEBUG: frozenset[str] = frozenset({"/debug-error"})
"""Hidden debug commands not exposed in autocomplete or help."""

ALL_CLASSIFIED: frozenset[str] = (
    ALWAYS_IMMEDIATE
    | BYPASS_WHEN_CONNECTING
    | IMMEDIATE_UI
    | SIDE_EFFECT_FREE
    | QUEUE_BOUND
    | HIDDEN_DEBUG
)
"""Union of all tiers plus hidden debug commands — used by drift tests."""


# ---------------------------------------------------------------------------
# Autocomplete tuples
# ---------------------------------------------------------------------------

SLASH_COMMANDS: list[tuple[str, str, str]] = [
    (cmd.name, cmd.description, cmd.hidden_keywords) for cmd in COMMANDS
]
"""`(name, description, hidden_keywords)` tuples for `SlashCommandController`."""


def parse_skill_command(command: str) -> tuple[str, str]:
    """Extract skill name and args from a `/skill:<name>` command.

    Args:
        command: The full command string (e.g., `/skill:web-research find X`).

    Returns:
        Tuple of `(skill_name, args)`.

            The skill name is normalized to lowercase. Both are empty strings
            when the command has no skill name after the prefix.
    """
    after_prefix = command[len("/skill:") :].strip()
    parts = after_prefix.split(maxsplit=1)
    if not parts or not parts[0]:
        return "", ""
    skill_name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    return skill_name, args


_STATIC_SKILL_ALIASES: frozenset[str] = frozenset({"remember", "skill-creator"})
"""Built-in skill names that have a dedicated top-level slash command.

Only list skills whose `/skill:<name>` form is redundant because a `/<name>`
convenience alias exists in `COMMANDS`.  Do **not** add every command name
here — that would silently suppress unrelated user skills that happen to share a
name with a slash command (e.g., a user skill called `model` should still
appear as `/skill:model`).
"""


def build_skill_commands(
    skills: list[ExtendedSkillMetadata],
) -> list[tuple[str, str, str]]:
    """Build autocomplete tuples for discovered skills.

    Each skill becomes a `/skill:<name>` entry with its description
    and the skill name as a hidden keyword for fuzzy matching.

    Skills that already have a dedicated slash command in `COMMANDS`
    (e.g., `remember` → `/remember`) are excluded to avoid duplicate
    autocomplete entries.

    Args:
        skills: List of discovered skill metadata.

    Returns:
        List of `(name, description, hidden_keywords)` tuples.
    """
    return [
        (f"/skill:{skill['name']}", skill["description"], skill["name"])
        for skill in skills
        if skill["name"] not in _STATIC_SKILL_ALIASES
    ]
