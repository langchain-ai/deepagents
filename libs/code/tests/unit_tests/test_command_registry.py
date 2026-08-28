"""Unit tests for the unified slash-command registry."""

from __future__ import annotations

import re
from pathlib import Path

from deepagents_code.command_registry import (
    ALL_CLASSIFIED,
    ALWAYS_IMMEDIATE,
    BYPASS_WHEN_CONNECTING,
    COMMANDS,
    HIDDEN_COMMANDS,
    IMMEDIATE_UI,
    IMMEDIATE_UI_ARG_FORMS,
    QUEUE_BOUND,
    SIDE_EFFECT_FREE,
    STARTUP_RECOVERY_COMMANDS,
    CommandEntry,
    get_slash_commands,
)


class TestCommandIntegrity:
    """Validate structural invariants of the COMMANDS registry."""


class TestBypassTiers:
    """Validate derived bypass-tier frozensets."""

    def test_startup_recovery_commands_are_queue_bound(self) -> None:
        # The recovery exemption is orthogonal to the normal tier: every
        # recovery command keeps its QUEUED tier and only gains an extra
        # failed-startup bypass. If one drifts to another tier, the comment
        # in STARTUP_RECOVERY_COMMANDS (and the bypass rationale) goes stale.
        assert STARTUP_RECOVERY_COMMANDS <= QUEUE_BOUND

    def test_startup_recovery_commands_are_known(self) -> None:
        names = {cmd.name for cmd in COMMANDS}
        assert names >= STARTUP_RECOVERY_COMMANDS

    def test_immediate_ui_arg_forms_extend_immediate_ui_commands(self) -> None:
        """Every whitelisted argument form must name an IMMEDIATE_UI command.

        `_can_bypass_queue` checks these forms only after the base command
        matches `IMMEDIATE_UI`; an entry under a command in any other tier
        would be dead config. The whitelist must also stay narrow — every
        entry is an exact no-further-arguments form whose handler defers all
        mutation to the modal's dismiss callback.
        """
        for form in IMMEDIATE_UI_ARG_FORMS:
            base = form.split(maxsplit=1)[0]
            assert base in IMMEDIATE_UI, (
                f"{form!r} is whitelisted but {base!r} is not IMMEDIATE_UI"
            )


class TestSlashCommands:
    """Validate the get_slash_commands() autocomplete list."""


class TestHiddenCommands:
    """`HIDDEN_COMMANDS` membership and autocomplete absence."""

    def test_hidden_not_in_autocomplete(self) -> None:
        names = {entry.name for entry in get_slash_commands()}
        for hidden in HIDDEN_COMMANDS:
            assert hidden not in names, (
                f"Hidden command {hidden!r} leaked into get_slash_commands()"
            )


class TestRestartCommand:
    """Validate the `/restart` entry specifically."""

    def test_restart_registered_for_autocomplete(self) -> None:
        restart_entry = next(
            entry for entry in get_slash_commands() if entry.name == "/restart"
        )

        # The generated catalog check pins exact wording; here we only assert
        # the entry is registered with a non-empty description.
        assert restart_entry.description

    def test_restart_classified_as_always_immediate(self) -> None:
        assert "/restart" in ALWAYS_IMMEDIATE
        assert "/restart" not in HIDDEN_COMMANDS


class TestAgentsCommand:
    """Validate the `/agents` entry specifically.

    The `/agents` command is reachable via fuzzy hidden-keyword matches
    (`switch`, `profile`, `persona`). Dropping any of those would silently
    regress discoverability.
    """


class TestMCPCommand:
    """Validate the `/mcp` entry specifically.

    `/mcp` now accepts an optional `login <server>` subcommand, so the
    entry must expose an argument hint that surfaces this in autocomplete
    without breaking the bare-form viewer invocation.
    """

    def test_mcp_hidden_keywords_cover_oauth(self) -> None:
        mcp_cmd = next(cmd for cmd in COMMANDS if cmd.name == "/mcp")
        keywords = mcp_cmd.hidden_keywords.split()
        assert "oauth" in keywords or "authenticate" in keywords


class TestToolsCommand:
    """Validate the `/tools` entry specifically."""


class TestCostCommand:
    """Validate `/cost` registration and discoverability metadata."""


class TestGoalCommand:
    """Validate the `/goal` entry specifically.

    `/goal` aliases the shared rubric grader controls (`model`,
    `max-iterations`), so the entry must advertise them in the argument hint
    and surface them via keyword search so goal-first users can discover
    grader tuning without knowing about `/rubric`.
    """


class TestCopyCommand:
    """Validate the `/copy` entry specifically."""

    def test_copy_classified_as_queue_bound(self) -> None:
        assert "/copy" in QUEUE_BOUND
