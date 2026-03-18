"""Unit tests for the unified slash-command registry."""

from __future__ import annotations

from deepagents_cli.command_registry import (
    ALL_CLASSIFIED,
    ALWAYS_IMMEDIATE,
    BYPASS_WHEN_CONNECTING,
    COMMANDS,
    IMMEDIATE_UI,
    QUEUE_BOUND,
    SLASH_COMMANDS,
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

    def test_alphabetically_sorted(self) -> None:
        names = [cmd.name for cmd in COMMANDS]
        assert names == sorted(names), "COMMANDS must be sorted alphabetically by name"

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
            QUEUE_BOUND,
        ]
        for i, a in enumerate(tiers):
            for b in tiers[i + 1 :]:
                assert not (a & b), f"Overlap between tiers: {a & b}"

    def test_all_classified_is_union(self) -> None:
        assert ALL_CLASSIFIED == (
            ALWAYS_IMMEDIATE | BYPASS_WHEN_CONNECTING | IMMEDIATE_UI | QUEUE_BOUND
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

    def test_tuple_format(self) -> None:
        for entry in SLASH_COMMANDS:
            assert isinstance(entry, tuple)
            assert len(entry) == 3
            name, desc, keywords = entry
            assert isinstance(name, str)
            assert name.startswith("/")
            assert isinstance(desc, str)
            assert isinstance(keywords, str)

    def test_excludes_aliases(self) -> None:
        names = {entry[0] for entry in SLASH_COMMANDS}
        for cmd in COMMANDS:
            for alias in cmd.aliases:
                assert alias not in names, (
                    f"Alias {alias!r} should not appear in autocomplete"
                )
