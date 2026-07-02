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
    SLASH_COMMANDS,
    STARTUP_RECOVERY_COMMANDS,
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

    def test_to_entry_matches_slash_commands(self) -> None:
        """SlashCommand.to_entry() produces the same entries as SLASH_COMMANDS."""
        for cmd, entry in zip(COMMANDS, SLASH_COMMANDS, strict=True):
            assert cmd.to_entry() == entry


class TestHiddenCommands:
    """`HIDDEN_COMMANDS` membership and autocomplete absence."""

    def test_debug_error_is_hidden(self) -> None:
        assert "/debug-error" in HIDDEN_COMMANDS

    def test_hidden_not_in_autocomplete(self) -> None:
        names = {entry.name for entry in SLASH_COMMANDS}
        for hidden in HIDDEN_COMMANDS:
            assert hidden not in names, (
                f"Hidden command {hidden!r} leaked into SLASH_COMMANDS"
            )


class TestRestartCommand:
    """Validate the `/restart` entry specifically."""

    def test_restart_registered_for_autocomplete(self) -> None:
        restart_entry = next(
            entry for entry in SLASH_COMMANDS if entry.name == "/restart"
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
        copy_entry = next(entry for entry in SLASH_COMMANDS if entry.name == "/copy")

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

    def test_help_body_lists_all_commands(self) -> None:
        """Every command in COMMANDS must appear in the /help body."""
        app_src = (
            Path(__file__).resolve().parents[2] / "deepagents_code" / "app.py"
        ).read_text()

        # Anchor on the `help_body = (` assignment so an unrelated "Commands:"
        # literal elsewhere in app.py (e.g. the /goal status display) can never
        # hijack the match. The assignment is the single source of the /help
        # body, so assert it is unique — if a second one appears, fail loudly
        # here rather than silently scraping the wrong block.
        anchors = re.findall(r"help_body = \(", app_src)
        assert len(anchors) == 1, (
            f"Expected exactly one `help_body = (` assignment in app.py, found "
            f"{len(anchors)}. Update this test's anchor if the /help body moved."
        )

        # Isolate the /help "Commands: ..." section (before "Interactive Features").
        match = re.search(
            r'help_body = \(\s*"Commands:\s*(.*?)(?=Interactive Features)',
            app_src,
            re.DOTALL,
        )
        assert match, "Could not locate Commands section in help_body"
        commands_section = match.group(1)

        # Sentinel check: the captured section must contain known-present
        # canonical commands. If the lazy `.*?` ever mis-captures (matching the
        # wrong region or sweeping unrelated source), this fails with a clear
        # message instead of surfacing a garbage token like `/non-` from a
        # comment further down the file.
        for sentinel in ("/quit", "/help"):
            assert sentinel in commands_section, (
                f"Expected {sentinel} in the captured /help Commands section; "
                "the help_body anchor likely matched the wrong region."
            )

        help_cmds = set(re.findall(r"/[a-z][-a-z]*", commands_section))
        registry_names = {cmd.name for cmd in COMMANDS}
        registry_aliases = {alias for cmd in COMMANDS for alias in cmd.aliases}

        # Commands intentionally omitted from the help body
        excluded = {"/version"}

        # /skill:<name> is dynamic, not a registry entry; regex extracts "/skill"
        help_cmds.discard("/skill")

        # Canonical names must appear in help; aliases (e.g. `/criteria`, `/q`)
        # may also be advertised but are never required.
        missing = registry_names - help_cmds - excluded
        extra = help_cmds - registry_names - registry_aliases

        assert not missing, (
            f"Commands in COMMANDS but missing from /help body: {missing}\n"
            "Add them to help_body in app.py _handle_command()."
        )
        assert not extra, (
            f"Commands in /help body but missing from COMMANDS: {extra}\n"
            "Remove them from help_body or add to COMMANDS in command_registry.py."
        )

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
