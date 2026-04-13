"""Unit tests for the unified slash-command registry."""

from __future__ import annotations

import re
from pathlib import Path

from deepagents_cli.command_registry import (
    ALL_CLASSIFIED,
    ALWAYS_IMMEDIATE,
    BYPASS_WHEN_CONNECTING,
    COMMANDS,
    HIDDEN_DEBUG,
    IMMEDIATE_UI,
    QUEUE_BOUND,
    SIDE_EFFECT_FREE,
    SLASH_COMMANDS,
    CommandEntry,
)


class TestCommandIntegrity:
    """Validate structural invariants of the COMMANDS registry."""

    def test_names_start_with_slash(self) -> None:
        for cmd in COMMANDS:
            assert cmd.name.startswith("/"), f"{cmd.name} missing leading slash"

    def test_aliases_start_with_slash(self) -> None:
        for cmd in COMMANDS:
            for alias in cmd.aliases:
                assert alias.startswith("/"), (
                    f"Alias {alias!r} of {cmd.name} missing leading slash"
                )

    def test_no_duplicate_names(self) -> None:
        names = [cmd.name for cmd in COMMANDS]
        assert len(names) == len(set(names)), "Duplicate command names found"

    def test_no_duplicate_aliases(self) -> None:
        all_names: list[str] = []
        for cmd in COMMANDS:
            all_names.append(cmd.name)
            all_names.extend(cmd.aliases)
        assert len(all_names) == len(set(all_names)), (
            "Duplicate name or alias across entries"
        )


class TestBypassTiers:
    """Validate derived bypass-tier frozensets."""

    def test_tiers_mutually_exclusive(self) -> None:
        tiers = [
            ALWAYS_IMMEDIATE,
            BYPASS_WHEN_CONNECTING,
            IMMEDIATE_UI,
            SIDE_EFFECT_FREE,
            QUEUE_BOUND,
        ]
        for i, a in enumerate(tiers):
            for b in tiers[i + 1 :]:
                assert not (a & b), f"Overlap between tiers: {a & b}"

    def test_all_classified_is_union(self) -> None:
        assert ALL_CLASSIFIED == (
            ALWAYS_IMMEDIATE
            | BYPASS_WHEN_CONNECTING
            | IMMEDIATE_UI
            | SIDE_EFFECT_FREE
            | QUEUE_BOUND
            | HIDDEN_DEBUG
        )

    def test_aliases_in_correct_tier(self) -> None:
        assert "/q" in ALWAYS_IMMEDIATE
        assert "/compact" in QUEUE_BOUND

    def test_every_command_classified(self) -> None:
        for cmd in COMMANDS:
            assert cmd.name in ALL_CLASSIFIED, f"{cmd.name} not in any tier"
            for alias in cmd.aliases:
                assert alias in ALL_CLASSIFIED, (
                    f"Alias {alias!r} of {cmd.name} not in any tier"
                )


class TestSlashCommands:
    """Validate the SLASH_COMMANDS autocomplete list."""

    def test_length_matches_commands(self) -> None:
        assert len(SLASH_COMMANDS) == len(COMMANDS)

    def test_entry_format(self) -> None:
        for entry in SLASH_COMMANDS:
            assert isinstance(entry, CommandEntry)
            assert isinstance(entry.name, str)
            assert entry.name.startswith("/")
            assert isinstance(entry.description, str)
            assert isinstance(entry.hidden_keywords, str)
            assert isinstance(entry.argument_hint, str)

    def test_excludes_aliases(self) -> None:
        names = {entry.name for entry in SLASH_COMMANDS}
        for cmd in COMMANDS:
            for alias in cmd.aliases:
                assert alias not in names, (
                    f"Alias {alias!r} should not appear in autocomplete"
                )


class TestHelpBodyDrift:
    """Ensure the /help body in app.py stays in sync with COMMANDS.

    The "Commands: ..." line in the `/help` handler is hand-maintained
    separately from the `COMMANDS` tuple in `command_registry.py`.  This
    test catches drift — e.g. a new command added to the registry but
    forgotten in the help output.
    """

    def test_help_body_lists_all_commands(self) -> None:
        """Every command in COMMANDS must appear in the /help body."""
        app_src = (
            Path(__file__).resolve().parents[2] / "deepagents_cli" / "app.py"
        ).read_text()

        # Isolate the "Commands: ..." section (before "Interactive Features")
        match = re.search(
            r'"Commands:\s*(.*?)(?=Interactive Features)',
            app_src,
            re.DOTALL,
        )
        assert match, "Could not locate Commands section in help_body"
        commands_section = match.group(1)

        help_cmds = set(re.findall(r"/[a-z][-a-z]*", commands_section))
        registry_cmds = {cmd.name for cmd in COMMANDS}

        # Commands intentionally omitted from the help body
        excluded = {"/version"}

        # /skill:<name> is dynamic, not a registry entry; regex extracts "/skill"
        help_cmds.discard("/skill")

        missing = registry_cmds - help_cmds - excluded
        extra = help_cmds - registry_cmds

        assert not missing, (
            f"Commands in COMMANDS but missing from /help body: {missing}\n"
            "Add them to help_body in app.py _handle_command()."
        )
        assert not extra, (
            f"Commands in /help body but missing from COMMANDS: {extra}\n"
            "Remove them from help_body or add to COMMANDS in command_registry.py."
        )
