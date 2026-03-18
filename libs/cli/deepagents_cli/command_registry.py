"""Unified slash-command registry.

Every slash command is declared once as a `SlashCommand` entry in `COMMANDS`.
Bypass-tier frozensets and autocomplete tuples are derived automatically — no
other file should hard-code command metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BypassTier(StrEnum):
    """Classification that controls whether a command can skip the message queue."""

    ALWAYS = "always"
    """Execute regardless of any busy state, including mid-thread-switch."""

    CONNECTING = "connecting"
    """Bypass only during initial server connection, not during agent/shell."""

    IMMEDIATE_UI = "immediate_ui"
    """Open modal UI immediately; real work deferred via `_defer_action` callback."""

    SIDE_EFFECT_FREE = "side_effect_free"
    """Fire-and-forget actions (e.g. open browser) that don't touch agent state."""

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
        name="/changelog",
        description="Open changelog in browser",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
    ),
    SlashCommand(
        name="/clear",
        description="Clear chat and start new thread",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="reset",
    ),
    SlashCommand(
        name="/docs",
        description="Open documentation in browser",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
    ),
    SlashCommand(
        name="/editor",
        description="Open prompt in external editor ($EDITOR)",
        bypass_tier=BypassTier.QUEUED,
    ),
    SlashCommand(
        name="/feedback",
        description="Submit a bug report or feature request",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
    ),
    SlashCommand(
        name="/help",
        description="Show help",
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
    SlashCommand(
        name="/quit",
        description="Exit app",
        bypass_tier=BypassTier.ALWAYS,
        hidden_keywords="close leave",
        aliases=("/q",),
    ),
    SlashCommand(
        name="/reload",
        description="Reload config from environment variables and .env",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="refresh",
    ),
    SlashCommand(
        name="/remember",
        description="Update memory and skills from conversation",
        bypass_tier=BypassTier.QUEUED,
    ),
    SlashCommand(
        name="/threads",
        description="Browse and resume previous threads",
        bypass_tier=BypassTier.IMMEDIATE_UI,
        hidden_keywords="continue history sessions",
    ),
    SlashCommand(
        name="/tokens",
        description="Token usage",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="cost",
    ),
    SlashCommand(
        name="/trace",
        description="Open current thread in LangSmith",
        bypass_tier=BypassTier.QUEUED,
    ),
    SlashCommand(
        name="/version",
        description="Show version",
        bypass_tier=BypassTier.CONNECTING,
    ),
)
"""All slash commands, alphabetically sorted by name."""


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
"""Fire-and-forget commands that don't touch agent state."""

QUEUE_BOUND: frozenset[str] = _build_bypass_set(BypassTier.QUEUED)
"""Commands that must wait in the queue when the app is busy."""

ALL_CLASSIFIED: frozenset[str] = (
    ALWAYS_IMMEDIATE
    | BYPASS_WHEN_CONNECTING
    | IMMEDIATE_UI
    | SIDE_EFFECT_FREE
    | QUEUE_BOUND
)
"""Union of all five tiers — used by drift tests."""


# ---------------------------------------------------------------------------
# Autocomplete tuples
# ---------------------------------------------------------------------------

SLASH_COMMANDS: list[tuple[str, str, str]] = [
    (cmd.name, cmd.description, cmd.hidden_keywords) for cmd in COMMANDS
]
"""`(name, description, hidden_keywords)` tuples for `SlashCommandController`."""
