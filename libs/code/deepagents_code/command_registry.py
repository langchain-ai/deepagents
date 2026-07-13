"""Unified slash-command registry.

Every slash command is declared once as a `SlashCommand` entry in `COMMANDS`.
Bypass-tier frozensets and autocomplete entries are derived automatically — no
other file should hard-code command metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from deepagents_code.skills.load import ExtendedSkillMetadata


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

    argument_hint: str = ""
    """Placeholder text for autocomplete when the command accepts args."""

    aliases: tuple[str, ...] = ()
    """Alternative names (e.g. `("/q",)` for `/quit`)."""

    def to_entry(self) -> CommandEntry:
        """Project this command into a `CommandEntry` for autocomplete.

        Returns:
            A `CommandEntry` carrying only the fields the autocomplete
                layer needs.
        """
        return CommandEntry(
            name=self.name,
            description=self.description,
            hidden_keywords=self.hidden_keywords,
            argument_hint=self.argument_hint,
        )


COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand(
        name="/agents",
        description="Browse and switch between available agents",
        bypass_tier=BypassTier.IMMEDIATE_UI,
        hidden_keywords="switch profile persona",
    ),
    SlashCommand(
        name="/auth",
        description="Connect and manage provider and service credentials",
        bypass_tier=BypassTier.IMMEDIATE_UI,
        hidden_keywords=(
            "key keys credential credentials login token api tracing langsmith"
        ),
        aliases=("/connect",),
    ),
    SlashCommand(
        name="/clear",
        description="Clear the chat and start a new thread",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="reset",
    ),
    SlashCommand(
        name="/copy",
        description="Copy the latest assistant message to clipboard",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
    ),
    SlashCommand(
        name="/force-clear",
        description="Stop active work, clear the chat, and start a new thread",
        bypass_tier=BypassTier.ALWAYS,
        hidden_keywords="reset interrupt",
    ),
    SlashCommand(
        name="/goal",
        description="Turn an outcome into acceptance criteria and pursue it",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords=(
            "objective criteria acceptance rubric grader grading model iterations"
        ),
        argument_hint="[<objective>|show|clear|model|max-iterations]",
    ),
    SlashCommand(
        name="/editor",
        description="Open prompt in an external editor ($EDITOR)",
        bypass_tier=BypassTier.QUEUED,
    ),
    SlashCommand(
        name="/effort",
        description="Set reasoning effort for the current model",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="reasoning thinking level",
        argument_hint="[none|low|medium|high|xhigh|max|clear]",
    ),
    SlashCommand(
        name="/mcp",
        description="Manage MCP servers and authentication",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
        hidden_keywords="servers oauth authenticate reconnect disable enable",
        argument_hint="[login <server> | reconnect]",
    ),
    SlashCommand(
        name="/model",
        description="Switch models or edit model settings",
        bypass_tier=BypassTier.IMMEDIATE_UI,
    ),
    SlashCommand(
        name="/notifications",
        description="Configure startup warnings",
        bypass_tier=BypassTier.IMMEDIATE_UI,
        hidden_keywords="warnings alerts suppress",
    ),
    SlashCommand(
        name="/offload",
        description="Summarize and offload older messages to free context",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="compact",
        aliases=("/compact",),
    ),
    SlashCommand(  # Static alias; not auto-generated from skill discovery
        name="/remember",
        description="Save useful context to memory or skills",
        bypass_tier=BypassTier.QUEUED,
        argument_hint="[context]",
    ),
    SlashCommand(  # Static alias; not auto-generated from skill discovery
        name="/skill-creator",
        description="Create or refine agent skills",
        bypass_tier=BypassTier.QUEUED,
        argument_hint="[task]",
    ),
    SlashCommand(
        name="/threads",
        description="Browse and resume past threads",
        bypass_tier=BypassTier.IMMEDIATE_UI,
        hidden_keywords="continue history sessions resume back previous",
        argument_hint="[-r [ID]]",
    ),
    SlashCommand(
        name="/trace",
        description="Open this thread in LangSmith",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
    ),
    SlashCommand(
        name="/tokens",
        description="Show token usage",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="cost",
    ),
    SlashCommand(
        name="/tools",
        description="List the tools available to the agent",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="mcp functions capabilities builtin built-in",
    ),
    SlashCommand(
        name="/reload",
        description="Reload environment and config",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="refresh",
    ),
    SlashCommand(
        name="/rubric",
        description="Use acceptance criteria you already have",
        bypass_tier=BypassTier.IMMEDIATE_UI,
        hidden_keywords="criteria acceptance grader grading evaluation iterations",
        argument_hint="[set|next|file|show|clear|model|max-iterations]",
        aliases=("/criteria",),
    ),
    SlashCommand(
        name="/restart",
        description="Restart the agent server",
        bypass_tier=BypassTier.ALWAYS,
        hidden_keywords="respawn server",
    ),
    SlashCommand(
        name="/theme",
        description="Change color theme",
        bypass_tier=BypassTier.IMMEDIATE_UI,
        hidden_keywords="dark light color appearance",
    ),
    SlashCommand(
        name="/scrollbar",
        description="Show or hide the chat scrollbar",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
        hidden_keywords="scroll scroller bar vertical",
    ),
    SlashCommand(
        name="/timestamps",
        description="Show or hide message timestamps",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
        hidden_keywords="time footer footers date dates",
    ),
    SlashCommand(
        name="/update",
        description="Check for and install updates",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="upgrade dependencies deps refresh",
        argument_hint="[--deps] [--prerelease]",
    ),
    SlashCommand(
        name="/install",
        description="Install an optional integration",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="extra extras add provider sandbox dependency",
        argument_hint="<extra> [--force]",
    ),
    SlashCommand(
        name="/auto-update",
        description="Turn automatic updates on or off",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
    ),
    SlashCommand(
        name="/changelog",
        description="Open the changelog in a browser",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
    ),
    SlashCommand(
        name="/version",
        description="Show version information",
        bypass_tier=BypassTier.CONNECTING,
        aliases=("/about",),
    ),
    SlashCommand(
        name="/feedback",
        description="Send feedback or report an issue",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
    ),
    SlashCommand(
        name="/docs",
        description="Open the docs",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
    ),
    SlashCommand(
        name="/help",
        description="Show help and available commands",
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

HIDDEN_COMMANDS: frozenset[str] = frozenset({"/debug", "/debug-error"})
"""Power-user commands kept out of autocomplete and help."""

STARTUP_RECOVERY_COMMANDS: frozenset[str] = frozenset(
    {"/install", "/reload", "/update"}
)
"""`QUEUED`-tier commands that must still run when startup has failed.

When the configured model can't be built (e.g. its provider package is
missing) the server never starts and the app holds a `_server_startup_error`
state that parks queued messages. These are the recovery escape hatches for
that state — install the missing package, reload config/env, or upgrade the
tool — so they must bypass the queue rather than sit behind the very failure
they repair. `/model` and `/auth` already escape via `IMMEDIATE_UI` (which
opens a modal and defers the real work); the commands here instead perform
their repair work directly, so they stay `QUEUED` and rely on this exemption.
The bypass itself is gated in `_can_bypass_queue`. Every entry is also
`QUEUE_BOUND` — the recovery exemption is orthogonal to the normal queue.
"""

ALL_CLASSIFIED: frozenset[str] = (
    ALWAYS_IMMEDIATE
    | BYPASS_WHEN_CONNECTING
    | IMMEDIATE_UI
    | SIDE_EFFECT_FREE
    | QUEUE_BOUND
    | HIDDEN_COMMANDS
)
"""Union of all tiers plus hidden commands — used by drift tests."""


# ---------------------------------------------------------------------------
# Autocomplete entries
# ---------------------------------------------------------------------------


class CommandEntry(NamedTuple):
    """A single autocomplete entry for the slash-command controller."""

    name: str
    """Canonical command name (e.g. `/quit`)."""

    description: str
    """Short user-facing description."""

    hidden_keywords: str
    """Space-separated terms for fuzzy matching (never displayed)."""

    argument_hint: str
    """Placeholder text shown when the command accepts arguments (e.g. `[context]`)."""


SLASH_COMMANDS: list[CommandEntry] = [cmd.to_entry() for cmd in COMMANDS]
"""Autocomplete entries derived from `COMMANDS` for `SlashCommandController`."""


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
) -> list[CommandEntry]:
    """Build autocomplete entries for discovered skills.

    Each skill becomes a `/skill:<name>` entry with its description
    and the skill name as a hidden keyword for fuzzy matching.

    Skills that already have a dedicated slash command in `COMMANDS`
    (e.g., `remember` → `/remember`) are excluded to avoid duplicate
    autocomplete entries.

    Args:
        skills: List of discovered skill metadata.

    Returns:
        List of `CommandEntry` instances.
    """
    return [
        CommandEntry(
            name=f"/skill:{skill['name']}",
            description=skill["description"],
            hidden_keywords=skill["name"],
            argument_hint="",
        )
        for skill in skills
        if skill["name"] not in _STATIC_SKILL_ALIASES
    ]
