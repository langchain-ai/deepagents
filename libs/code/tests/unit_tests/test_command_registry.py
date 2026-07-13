"""Unit tests for the unified slash-command registry."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from deepagents_code.command_registry import (
    ALL_CLASSIFIED,
    ALWAYS_IMMEDIATE,
    BYPASS_WHEN_CONNECTING,
    COMMANDS,
    HIDDEN_COMMANDS,
    IMMEDIATE_UI,
    QUEUE_BOUND,
    SIDE_EFFECT_FREE,
    STARTUP_RECOVERY_COMMANDS,
    CommandEntry,
    get_slash_commands,
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
            | HIDDEN_COMMANDS
        )

    def test_aliases_in_correct_tier(self) -> None:
        assert "/q" in ALWAYS_IMMEDIATE
        assert "/about" in BYPASS_WHEN_CONNECTING
        assert "/compact" in QUEUE_BOUND
        assert "/connect" in IMMEDIATE_UI

    def test_every_command_classified(self) -> None:
        for cmd in COMMANDS:
            assert cmd.name in ALL_CLASSIFIED, f"{cmd.name} not in any tier"
            for alias in cmd.aliases:
                assert alias in ALL_CLASSIFIED, (
                    f"Alias {alias!r} of {cmd.name} not in any tier"
                )

    def test_startup_recovery_commands_are_queue_bound(self) -> None:
        # The recovery exemption is orthogonal to the normal tier: every
        # recovery command keeps its QUEUED tier and only gains an extra
        # failed-startup bypass. If one drifts to another tier, the comment
        # in STARTUP_RECOVERY_COMMANDS (and the bypass rationale) goes stale.
        assert STARTUP_RECOVERY_COMMANDS <= QUEUE_BOUND

    def test_startup_recovery_commands_are_known(self) -> None:
        names = {cmd.name for cmd in COMMANDS}
        assert names >= STARTUP_RECOVERY_COMMANDS


class TestSlashCommands:
    """Validate the get_slash_commands() autocomplete list."""

    def test_length_matches_public_commands(self) -> None:
        entries = get_slash_commands()
        assert len(entries) <= len(COMMANDS)

    def test_entry_format(self) -> None:
        from deepagents_code.command_registry import get_slash_commands

        for entry in get_slash_commands():
            assert isinstance(entry, CommandEntry)
            assert isinstance(entry.name, str)
            assert entry.name.startswith("/")
            assert isinstance(entry.description, str)
            assert isinstance(entry.hidden_keywords, str)
            assert isinstance(entry.argument_hint, str)

    def test_excludes_aliases(self) -> None:
        from deepagents_code.command_registry import get_slash_commands

        names = {entry.name for entry in get_slash_commands()}
        for cmd in COMMANDS:
            for alias in cmd.aliases:
                assert alias not in names, (
                    f"Alias {alias!r} should not appear in autocomplete"
                )

    def test_entries_come_from_commands(self) -> None:
        """Every public entry is derived from the command registry."""
        command_entries = {command.to_entry() for command in COMMANDS}
        assert set(get_slash_commands()) <= command_entries

    def test_experimental_plugin_commands_hidden_by_default(self) -> None:
        from deepagents_code.command_registry import get_slash_commands

        names = {entry.name for entry in get_slash_commands()}
        assert "/plugins" not in names
        assert "/reload-plugins" not in names

    def test_experimental_plugin_commands_visible_when_enabled(
        self, monkeypatch
    ) -> None:
        from deepagents_code._env_vars import EXPERIMENTAL
        from deepagents_code.command_registry import get_slash_commands

        monkeypatch.setenv(EXPERIMENTAL, "1")
        names = {entry.name for entry in get_slash_commands()}
        assert "/plugins" in names
        assert "/reload-plugins" in names


class TestHiddenCommands:
    """`HIDDEN_COMMANDS` membership and autocomplete absence."""

    def test_debug_error_is_hidden(self) -> None:
        assert "/debug-error" in HIDDEN_COMMANDS

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

        # Exact wording is pinned by TestCommandsCatalogDrift; here we only
        # assert the entry is registered with a non-empty description.
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

    def test_agents_registered(self) -> None:
        names = {cmd.name for cmd in COMMANDS}
        assert "/agents" in names

    def test_agents_hidden_keywords(self) -> None:
        agents_cmd = next(cmd for cmd in COMMANDS if cmd.name == "/agents")
        keywords = agents_cmd.hidden_keywords.split()
        assert set(keywords) >= {"switch", "profile", "persona"}

    def test_agents_classified_as_immediate_ui(self) -> None:
        assert "/agents" in IMMEDIATE_UI


class TestMCPCommand:
    """Validate the `/mcp` entry specifically.

    `/mcp` now accepts an optional `login <server>` subcommand, so the
    entry must expose an argument hint that surfaces this in autocomplete
    without breaking the bare-form viewer invocation.
    """

    def test_mcp_registered(self) -> None:
        names = {cmd.name for cmd in COMMANDS}
        assert "/mcp" in names

    def test_mcp_argument_hint_advertises_login(self) -> None:
        mcp_cmd = next(cmd for cmd in COMMANDS if cmd.name == "/mcp")
        assert "login" in mcp_cmd.argument_hint

    def test_mcp_hidden_keywords_cover_oauth(self) -> None:
        mcp_cmd = next(cmd for cmd in COMMANDS if cmd.name == "/mcp")
        keywords = mcp_cmd.hidden_keywords.split()
        assert "oauth" in keywords or "authenticate" in keywords

    def test_mcp_argument_hint_advertises_reconnect(self) -> None:
        mcp_cmd = next(cmd for cmd in COMMANDS if cmd.name == "/mcp")
        assert "reconnect" in mcp_cmd.argument_hint

    def test_mcp_hidden_keywords_cover_reconnect(self) -> None:
        mcp_cmd = next(cmd for cmd in COMMANDS if cmd.name == "/mcp")
        keywords = mcp_cmd.hidden_keywords.split()
        assert "reconnect" in keywords


class TestToolsCommand:
    """Validate the `/tools` entry specifically."""

    def test_tools_registered(self) -> None:
        names = {cmd.name for cmd in COMMANDS}
        assert "/tools" in names

    def test_tools_classified_as_queue_bound(self) -> None:
        assert "/tools" in QUEUE_BOUND
        assert "/tools" not in HIDDEN_COMMANDS

    def test_tools_hidden_keywords_cover_mcp(self) -> None:
        tools_cmd = next(cmd for cmd in COMMANDS if cmd.name == "/tools")
        assert "mcp" in tools_cmd.hidden_keywords.split()


class TestGoalCommand:
    """Validate the `/goal` entry specifically.

    `/goal` aliases the shared rubric grader controls (`model`,
    `max-iterations`), so the entry must advertise them in the argument hint
    and surface them via keyword search so goal-first users can discover
    grader tuning without knowing about `/rubric`.
    """

    def test_goal_argument_hint_advertises_grader_aliases(self) -> None:
        goal_cmd = next(cmd for cmd in COMMANDS if cmd.name == "/goal")
        assert "model" in goal_cmd.argument_hint
        assert "max-iterations" in goal_cmd.argument_hint

    def test_goal_hidden_keywords_cover_grader_search(self) -> None:
        goal_cmd = next(cmd for cmd in COMMANDS if cmd.name == "/goal")
        keywords = goal_cmd.hidden_keywords.split()
        assert {"grader", "grading", "model", "iterations"} <= set(keywords)

    def test_goal_hidden_keywords_retain_acceptance(self) -> None:
        goal_cmd = next(cmd for cmd in COMMANDS if cmd.name == "/goal")
        assert "acceptance" in goal_cmd.hidden_keywords.split()


class TestCopyCommand:
    """Validate the `/copy` entry specifically."""

    def test_copy_registered_for_autocomplete(self) -> None:
        copy_entry = next(
            entry for entry in get_slash_commands() if entry.name == "/copy"
        )

        # Exact wording is pinned by TestCommandsCatalogDrift; here we only
        # assert the entry is registered with a non-empty description.
        assert copy_entry.description

    def test_copy_classified_as_side_effect_free(self) -> None:
        assert "/copy" in SIDE_EFFECT_FREE


class TestCommandsCatalogDrift:
    """Ensure `COMMANDS.md` stays in sync with the registry.

    `COMMANDS.md` is generated by `scripts/generate_commands_catalog.py` from
    `COMMANDS`/`HIDDEN_COMMANDS`. The `commands-catalog-check` make target is
    not wired into CI, so without this test a reworded description (or a
    forgotten `make commands-catalog`) would silently leave the catalog stale.
    This is also the canonical home for exact-wording coverage, which is why
    the per-command tests above assert only that descriptions are non-empty.
    """

    def test_catalog_matches_registry(self) -> None:
        code_dir = Path(__file__).resolve().parents[2]
        script = code_dir / "scripts" / "generate_commands_catalog.py"

        spec = importlib.util.spec_from_file_location(
            "generate_commands_catalog", script
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        catalog = (code_dir / "COMMANDS.md").read_text(encoding="utf-8")
        assert catalog == module.generate(), (
            "COMMANDS.md is stale. Run `make commands-catalog` from libs/code/."
        )


class TestHelpBodyDrift:
    """Ensure the /help body in app.py stays in sync with COMMANDS.

    The "Commands: ..." line in the `/help` handler is hand-maintained
    separately from the `COMMANDS` tuple in `command_registry.py`.  This
    test catches drift — e.g. a new command added to the registry but
    forgotten in the help output.
    """

    def test_help_body_uses_command_registry(self) -> None:
        """The help display composes command names from the public registry API."""
        app_src = (
            Path(__file__).resolve().parents[2] / "deepagents_code" / "app.py"
        ).read_text()

        assert (
            "from deepagents_code.command_registry import get_slash_commands" in app_src
        )
        assert "for entry in get_slash_commands()" in app_src

    def test_help_body_describes_incognito_shell_prefix(self) -> None:
        """The `/help` body should document local-only incognito shell mode."""
        app_src = (
            Path(__file__).resolve().parents[2] / "deepagents_code" / "app.py"
        ).read_text()

        # Locate the Interactive Features block where the `!!` row lives.
        match = re.search(
            r'"Interactive Features:\\n"(.*?)"\s*Docs:',
            app_src,
            re.DOTALL,
        )
        assert match, "Could not locate Interactive Features section in help_body"
        section = match.group(1)

        assert "!!command" in section, "Help body must show `!!command` literal"
        # Concept-level checks rather than exact wording — independent of
        # whether the sentence reads "command/output to model context" or
        # "output and command to model context".
        assert "model context" in section
        assert "command" in section
        assert "output" in section
