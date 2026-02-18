"""Unit tests for skills CLI commands."""

import argparse
import io
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from deepagents.middleware.skills import SkillMetadata, _parse_skill_metadata
from rich.console import Console

from deepagents_cli.main import parse_args
from deepagents_cli.skills.commands import (
    _delete,
    _format_info_fields,
    _generate_template,
    _info,
    _list,
    _validate_name,
    _validate_skill_path,
    execute_skills_command,
)


class TestValidateSkillName:
    """Test skill name validation per Agent Skills spec (https://agentskills.io/specification)."""

    def test_valid_skill_names(self):
        """Test that spec-compliant skill names are accepted.

        Per spec: lowercase alphanumeric, hyphens only, no start/end hyphen,
        no consecutive hyphens, max 64 chars.
        """
        valid_names = [
            "web-research",
            "langgraph-docs",
            "skill123",
            "skill-with-many-parts",
            "a",
            "a1",
            "code-review",
            "data-analysis",
        ]
        for name in valid_names:
            is_valid, error = _validate_name(name)
            assert is_valid, f"Valid name '{name}' was rejected: {error}"
            assert error == ""

    def test_invalid_names_per_spec(self):
        """Test that non-spec-compliant names are rejected."""
        invalid_names = [
            ("MySkill", "uppercase not allowed"),
            ("my_skill", "underscores not allowed"),
            ("skill_with_underscores", "underscores not allowed"),
            ("-skill", "cannot start with hyphen"),
            ("skill-", "cannot end with hyphen"),
            ("skill--name", "consecutive hyphens not allowed"),
        ]
        for name, reason in invalid_names:
            is_valid, error = _validate_name(name)
            assert not is_valid, f"Invalid name '{name}' ({reason}) was accepted"
            assert error != ""

    def test_path_traversal_attacks(self):
        """Test that path traversal attempts are blocked."""
        malicious_names = [
            "../../../etc/passwd",
            "../../.ssh/authorized_keys",
            "../.bashrc",
            "..\\..\\windows\\system32",
            "skill/../../../etc",
            "../../tmp/exploit",
            "../..",
            "..",
        ]
        for name in malicious_names:
            is_valid, error = _validate_name(name)
            assert not is_valid, f"Malicious name '{name}' was accepted"
            assert error != ""
            assert "path" in error.lower() or ".." in error

    def test_absolute_paths(self):
        """Test that absolute paths are blocked."""
        malicious_names = [
            "/etc/passwd",
            "/home/user/.ssh",
            "\\Windows\\System32",
            "/tmp/exploit",
        ]
        for name in malicious_names:
            is_valid, error = _validate_name(name)
            assert not is_valid, f"Absolute path '{name}' was accepted"
            assert error != ""

    def test_path_separators(self):
        """Test that path separators are blocked."""
        malicious_names = [
            "skill/name",
            "skill\\name",
            "path/to/skill",
            "parent\\child",
        ]
        for name in malicious_names:
            is_valid, error = _validate_name(name)
            assert not is_valid, f"Path with separator '{name}' was accepted"
            assert error != ""

    def test_invalid_characters(self):
        """Test that invalid characters are blocked."""
        malicious_names = [
            "skill name",  # space
            "skill;rm -rf /",  # command injection
            "skill`whoami`",  # command substitution
            "skill$(whoami)",  # command substitution
            "skill&ls",  # command chaining
            "skill|cat",  # pipe
            "skill>file",  # redirect
            "skill<file",  # redirect
            "skill*",  # wildcard
            "skill?",  # wildcard
            "skill[a]",  # pattern
            "skill{a,b}",  # brace expansion
            "skill$VAR",  # variable expansion
            "skill@host",  # at sign
            "skill#comment",  # hash
            "skill!event",  # exclamation
            "skill'quote",  # single quote
            'skill"quote',  # double quote
        ]
        for name in malicious_names:
            is_valid, error = _validate_name(name)
            assert not is_valid, f"Invalid character in '{name}' was accepted"
            assert error != ""

    def test_unicode_lowercase_accepted(self) -> None:
        """Unicode lowercase names should be accepted (matching SDK behavior).

        The SDK's `_validate_skill_name` accepts any character where
        ``c.isalpha() and c.islower()`` or ``c.isdigit()`` is True.
        """
        valid_unicode_names = [
            "caf\u00e9",  # cafe with accent
            "\u00fcber-tool",  # uber with umlaut
            "resum\u00e9",  # resume with accent
            "na\u00efve",  # naive with diaeresis
        ]
        for name in valid_unicode_names:
            is_valid, error = _validate_name(name)
            assert is_valid, f"Unicode lowercase name '{name}' was rejected: {error}"
            assert error == ""

    def test_unicode_uppercase_rejected(self) -> None:
        """Unicode uppercase characters should be rejected."""
        invalid_unicode_names = [
            "Caf\u00e9",  # leading uppercase
            "\u00dcber-tool",  # uppercase U-umlaut
        ]
        for name in invalid_unicode_names:
            is_valid, error = _validate_name(name)
            assert not is_valid, f"Unicode uppercase name '{name}' was accepted"
            assert error != ""

    def test_cjk_rejected(self) -> None:
        """CJK characters should be rejected (not lowercase alpha)."""
        cjk_names = [
            "\u6280\u80fd",  # Chinese characters
            "\u30b9\u30ad\u30eb",  # Japanese katakana
        ]
        for name in cjk_names:
            is_valid, error = _validate_name(name)
            assert not is_valid, f"CJK name '{name}' was accepted"
            assert error != ""

    def test_emoji_rejected(self) -> None:
        """Emoji characters should be rejected."""
        emoji_names = [
            "skill-\U0001f680",
            "\U0001f4dd-notes",
        ]
        for name in emoji_names:
            is_valid, error = _validate_name(name)
            assert not is_valid, f"Emoji name '{name}' was accepted"
            assert error != ""

    def test_empty_names(self):
        """Test that empty or whitespace names are blocked."""
        malicious_names = [
            "",
            "   ",
            "\t",
            "\n",
        ]
        for name in malicious_names:
            is_valid, error = _validate_name(name)
            assert not is_valid, f"Empty/whitespace name '{name}' was accepted"
            assert error != ""


class TestValidateSkillPath:
    """Test skill path validation to ensure paths stay within bounds."""

    def test_valid_path_within_base(self, tmp_path: Path) -> None:
        """Test that valid paths within base directory are accepted."""
        base_dir = tmp_path / "skills"
        base_dir.mkdir()

        skill_dir = base_dir / "my-skill"
        is_valid, error = _validate_skill_path(skill_dir, base_dir)
        assert is_valid, f"Valid path was rejected: {error}"
        assert error == ""

    def test_path_traversal_outside_base(self, tmp_path: Path) -> None:
        """Test that paths outside base directory are blocked."""
        base_dir = tmp_path / "skills"
        base_dir.mkdir()

        # Try to escape to parent directory
        malicious_dir = tmp_path / "malicious"
        is_valid, error = _validate_skill_path(malicious_dir, base_dir)
        assert not is_valid, "Path outside base directory was accepted"
        assert error != ""

    def test_symlink_path_traversal(self, tmp_path: Path) -> None:
        """Test that symlinks pointing outside base are detected."""
        base_dir = tmp_path / "skills"
        base_dir.mkdir()

        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()

        symlink_path = base_dir / "evil-link"
        try:
            symlink_path.symlink_to(outside_dir)

            is_valid, error = _validate_skill_path(symlink_path, base_dir)
            # The symlink resolves to outside the base, so it should be blocked
            assert not is_valid, "Symlink to outside directory was accepted"
            assert error != ""
        except OSError:
            # Symlink creation might fail on some systems
            pytest.skip("Symlink creation not supported")

    def test_nonexistent_path_validation(self, tmp_path: Path) -> None:
        """Test validation of paths that don't exist yet."""
        base_dir = tmp_path / "skills"
        base_dir.mkdir()

        # Path doesn't exist yet, but should be valid
        skill_dir = base_dir / "new-skill"
        is_valid, error = _validate_skill_path(skill_dir, base_dir)
        assert is_valid, f"Valid non-existent path was rejected: {error}"
        assert error == ""


class TestIntegrationSecurity:
    """Integration tests for security across the command flow."""

    def test_combined_validation(self, tmp_path: Path) -> None:
        """Test that both name and path validation work together."""
        base_dir = tmp_path / "skills"
        base_dir.mkdir()

        # Test various attack scenarios
        attack_vectors = [
            ("../../../etc/passwd", "path traversal"),
            ("/etc/passwd", "absolute path"),
            ("skill/../../../tmp", "hidden traversal"),
            ("skill;rm -rf", "command injection"),
        ]

        for skill_name, attack_type in attack_vectors:
            # First, name validation should catch it
            is_valid_name, name_error = _validate_name(skill_name)

            if is_valid_name:
                # If name validation doesn't catch it, path validation must
                skill_dir = base_dir / skill_name
                is_valid_path, _path_error = _validate_skill_path(skill_dir, base_dir)
                assert not is_valid_path, (
                    f"{attack_type} bypassed both validations: {skill_name}"
                )
            else:
                # Name validation caught it - this is good
                assert name_error != "", f"No error message for {attack_type}"


class TestGenerateTemplate:
    """Test the template generated by `_generate_template()`.

    These tests verify that the template conforms to the skill-creator
    `SKILL.md` guidance and the Agent Skills spec.
    """

    def test_template_parseable_by_middleware(self):
        """The generated template should be parseable by `_parse_skill_metadata`.

        This ensures the CLI template produces valid SKILL.md files that
        the middleware can load without errors.
        """
        template = _generate_template("my-test-skill")
        result = _parse_skill_metadata(
            content=template,
            skill_path="/tmp/my-test-skill/SKILL.md",
            directory_name="my-test-skill",
        )
        assert result is not None, "Middleware failed to parse generated template"
        assert result["name"] == "my-test-skill"

    def test_template_body_has_no_when_to_use_section(self):
        """`'When to Use'` should NOT appear in the body (below the `---` closer)."""
        template = _generate_template("my-skill")
        # Split on the closing --- to get the body
        parts = re.split(r"\n---\s*\n", template, maxsplit=1)
        assert len(parts) == 2, "Template should have frontmatter and body"
        body = parts[1]
        assert "## When to Use" not in body, (
            "Template body contains '## When to Use' section — "
            "this belongs in the description, not the body"
        )

    def test_template_description_includes_trigger_guidance(self):
        """The description placeholder should guide users to include triggers."""
        template = _generate_template("my-skill")
        # Extract the description line from frontmatter
        match = re.search(r"^description:\s*(.+)$", template, re.MULTILINE)
        assert match is not None, "No description field found in template"
        description = match.group(1).lower()
        assert "when" in description, (
            "Description placeholder should guide users to include 'when to use' info"
        )


def _make_skill(
    *,
    name: str = "test-skill",
    description: str = "A test skill",
    path: str = "/tmp/test-skill/SKILL.md",
    skill_license: str | None = None,
    compatibility: str | None = None,
    metadata: dict[str, str] | None = None,
    allowed_tools: list[str] | None = None,
) -> SkillMetadata:
    """Build a minimal `SkillMetadata` dict with overrides.

    Args:
        name: Skill identifier.
        description: What the skill does.
        path: Path to the SKILL.md file.
        skill_license: License name or `None`.
        compatibility: Environment requirements or `None`.
        metadata: Arbitrary key-value pairs.
        allowed_tools: Recommended tool names.

    Returns:
        A `SkillMetadata` TypedDict with the given values.
    """
    return SkillMetadata(
        name=name,
        description=description,
        path=path,
        license=skill_license,
        compatibility=compatibility,
        metadata=metadata if metadata is not None else {},
        allowed_tools=allowed_tools if allowed_tools is not None else [],
    )


class TestFormatInfoFields:
    """Tests for `_format_info_fields` optional metadata extraction."""

    def test_all_fields_present(self) -> None:
        """All four optional fields populated should produce four entries."""
        skill = _make_skill(
            skill_license="MIT",
            compatibility="Python 3.10+",
            allowed_tools=["Bash(git:*)", "Read"],
            metadata={"author": "acme", "version": "1.0"},
        )
        result = _format_info_fields(skill)
        labels = [label for label, _ in result]
        assert labels == [
            "License",
            "Compatibility",
            "Allowed Tools",
            "Metadata",
        ]
        assert result[0] == ("License", "MIT")
        assert result[1] == ("Compatibility", "Python 3.10+")
        assert result[2] == ("Allowed Tools", "Bash(git:*), Read")
        assert "author=acme" in result[3][1]
        assert "version=1.0" in result[3][1]

    def test_no_optional_fields(self) -> None:
        """When all optional fields are None/empty, return empty list."""
        skill = _make_skill()
        result = _format_info_fields(skill)
        assert result == []

    def test_license_only(self) -> None:
        """Only license set should return a single License entry."""
        skill = _make_skill(skill_license="Apache-2.0")
        result = _format_info_fields(skill)
        assert len(result) == 1
        assert result[0] == ("License", "Apache-2.0")

    def test_compatibility_only(self) -> None:
        """Only compatibility set should return a single Compatibility entry."""
        skill = _make_skill(compatibility="Requires poppler")
        result = _format_info_fields(skill)
        assert len(result) == 1
        assert result[0] == ("Compatibility", "Requires poppler")

    def test_allowed_tools_only(self) -> None:
        """Only allowed_tools populated should return entry."""
        skill = _make_skill(allowed_tools=["Bash", "Read"])
        result = _format_info_fields(skill)
        assert len(result) == 1
        assert result[0] == ("Allowed Tools", "Bash, Read")

    def test_metadata_only(self) -> None:
        """Only metadata populated should return a Metadata entry."""
        skill = _make_skill(metadata={"author": "test-org"})
        result = _format_info_fields(skill)
        assert len(result) == 1
        assert result[0] == ("Metadata", "author=test-org")

    def test_field_order(self) -> None:
        """Fields appear in order: License, Compatibility, Allowed Tools, Metadata."""
        skill = _make_skill(
            metadata={"k": "v"},
            skill_license="GPL-3.0",
            allowed_tools=["Write"],
            compatibility="macOS only",
        )
        result = _format_info_fields(skill)
        labels = [label for label, _ in result]
        assert labels == [
            "License",
            "Compatibility",
            "Allowed Tools",
            "Metadata",
        ]


class TestSkillsHelpFlag:
    """Test that `deepagents skills -h` shows skills-specific help."""

    def test_skills_help_shows_subcommands(self) -> None:
        """Running `deepagents skills -h` should show skills subcommands.

        Regression: -h on the skills subcommand was falling through to the
        global help screen, showing top-level options (--sandbox, --model, etc.)
        instead of skills-specific commands (list, create, info).
        """
        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=120)

        with (
            patch("sys.argv", ["deepagents", "skills", "-h"]),
            patch("deepagents_cli.ui.console", test_console),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()

        assert exc_info.value.code in (0, None)
        output = buf.getvalue()

        # Should contain skills-specific content
        assert "list" in output.lower()
        assert "create" in output.lower()
        assert "info" in output.lower()
        assert "delete" in output.lower()

        # Should NOT contain global-only content
        assert "Start interactive thread" not in output
        assert "--sandbox" not in output
        assert "--model" not in output

    def test_skills_list_help_shows_list_options(self) -> None:
        """Running `deepagents skills list -h` should show list-specific options."""
        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=120)

        with (
            patch("sys.argv", ["deepagents", "skills", "list", "-h"]),
            patch("deepagents_cli.ui.console", test_console),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()

        assert exc_info.value.code in (0, None)
        output = buf.getvalue()

        # Should contain list-specific content
        assert "--agent" in output
        assert "--project" in output

        # Should NOT contain global-only content
        assert "Start interactive thread" not in output
        assert "--sandbox" not in output


class TestThreadsHelpFlag:
    """Test that `deepagents threads -h` shows threads-specific help."""

    def test_threads_help_shows_threads_content(self) -> None:
        """Running `deepagents threads -h` should show threads subcommands.

        Regression: same pattern as skills -- -h on the threads subcommand
        should show threads-specific help, not the global help screen.
        """
        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=120)

        with (
            patch("sys.argv", ["deepagents", "threads", "-h"]),
            patch("deepagents_cli.ui.console", test_console),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()

        assert exc_info.value.code in (0, None)
        output = buf.getvalue()

        # Should contain threads-specific content
        assert "list" in output.lower()
        assert "delete" in output.lower()

        # Should NOT contain global-only content
        assert "Start interactive thread" not in output
        assert "--sandbox" not in output
        assert "--model" not in output


class TestThreadsListAlias:
    """Test that `deepagents threads ls` is parsed as a `list` alias."""

    def test_threads_ls_alias_parsed(self) -> None:
        """Verify `threads ls` sets threads_command to 'ls'."""
        with patch("sys.argv", ["deepagents", "threads", "ls"]):
            args = parse_args()
        assert args.command == "threads"
        assert args.threads_command == "ls"

    def test_threads_list_still_works(self) -> None:
        """Verify `threads list` still works after alias addition."""
        with patch("sys.argv", ["deepagents", "threads", "list"]):
            args = parse_args()
        assert args.command == "threads"
        assert args.threads_command == "list"


class TestSkillsListAlias:
    """Test that `deepagents skills ls` is parsed as a `list` alias."""

    def test_skills_ls_alias_parsed(self) -> None:
        """Verify `skills ls` sets skills_command to 'ls'."""
        with patch("sys.argv", ["deepagents", "skills", "ls"]):
            args = parse_args()
        assert args.command == "skills"
        assert args.skills_command == "ls"

    def test_skills_list_still_works(self) -> None:
        """Verify `skills list` still works after alias addition."""
        with patch("sys.argv", ["deepagents", "skills", "list"]):
            args = parse_args()
        assert args.command == "skills"
        assert args.skills_command == "list"


class TestInfoShadowWarning:
    """Test that `skills info` warns when a project skill shadows a user skill."""

    def _make_skill_dir(self, parent: Path, name: str, description: str) -> None:
        """Create a minimal skill directory with a valid SKILL.md.

        Args:
            parent: Parent skills directory.
            name: Skill name (used as directory name and frontmatter name).
            description: Skill description for frontmatter.
        """
        skill_dir = parent / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\nContent\n"
        )

    def test_shadow_note_shown_when_project_overrides_user(
        self, tmp_path: Path
    ) -> None:
        """When a project skill shadows a user skill, info should note it."""
        user_dir = tmp_path / "user_skills"
        project_dir = tmp_path / "project_skills"
        self._make_skill_dir(user_dir, "web-research", "User version")
        self._make_skill_dir(project_dir, "web-research", "Project version")

        mock_settings = patch(
            "deepagents_cli.skills.commands.Settings.from_environment",
            return_value=type(
                "FakeSettings",
                (),
                {
                    "get_built_in_skills_dir": staticmethod(lambda: None),
                    "get_user_skills_dir": lambda _, _a: user_dir,
                    "get_project_skills_dir": lambda _: project_dir,
                    "get_user_agent_skills_dir": lambda _: None,
                    "get_project_agent_skills_dir": lambda _: None,
                },
            )(),
        )

        output: list[str] = []

        def capture_print(*args: str, **_: str) -> None:
            output.append(" ".join(str(a) for a in args))

        with (
            mock_settings,
            patch("deepagents_cli.skills.commands.console") as mock_console,
        ):
            mock_console.print = capture_print
            _info("web-research", agent="agent")

        joined = "\n".join(output)
        assert "overrides" in joined.lower() or "shadows" in joined.lower()

    def test_no_shadow_note_when_no_conflict(self, tmp_path: Path) -> None:
        """When there is no name conflict, no shadow note should appear."""
        user_dir = tmp_path / "user_skills"
        project_dir = tmp_path / "project_skills"
        self._make_skill_dir(user_dir, "web-research", "User only skill")

        mock_settings = patch(
            "deepagents_cli.skills.commands.Settings.from_environment",
            return_value=type(
                "FakeSettings",
                (),
                {
                    "get_built_in_skills_dir": staticmethod(lambda: None),
                    "get_user_skills_dir": lambda _, _a: user_dir,
                    "get_project_skills_dir": lambda _: project_dir,
                    "get_user_agent_skills_dir": lambda _: None,
                    "get_project_agent_skills_dir": lambda _: None,
                },
            )(),
        )

        output: list[str] = []

        def capture_print(*args: str, **_: str) -> None:
            output.append(" ".join(str(a) for a in args))

        with (
            mock_settings,
            patch("deepagents_cli.skills.commands.console") as mock_console,
        ):
            mock_console.print = capture_print
            _info("web-research", agent="agent")

        joined = "\n".join(output)
        assert "overrides" not in joined.lower()
        assert "shadows" not in joined.lower()


class TestInfoBuiltInSkill:
    """Test that `skills info` displays built-in skills correctly."""

    def _make_skill_dir(self, parent: Path, name: str, description: str) -> None:
        """Create a minimal skill directory with a valid SKILL.md.

        Args:
            parent: Parent skills directory.
            name: Skill name (used as directory name and frontmatter name).
            description: Skill description for frontmatter.
        """
        skill_dir = parent / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\nContent\n"
        )

    def test_built_in_skill_shows_correct_label(self, tmp_path: Path) -> None:
        """Built-in skills should display '(Built-in Skill)' in magenta."""
        built_in_dir = tmp_path / "built_in_skills"
        self._make_skill_dir(built_in_dir, "test-builtin", "A built-in skill")

        mock_settings = patch(
            "deepagents_cli.skills.commands.Settings.from_environment",
            return_value=type(
                "FakeSettings",
                (),
                {
                    "get_built_in_skills_dir": staticmethod(lambda: built_in_dir),
                    "get_user_skills_dir": lambda _, _a: None,
                    "get_project_skills_dir": lambda _: None,
                    "get_user_agent_skills_dir": lambda _: None,
                    "get_project_agent_skills_dir": lambda _: None,
                },
            )(),
        )

        output: list[str] = []

        def capture_print(*args: str, **_: str) -> None:
            output.append(" ".join(str(a) for a in args))

        with (
            mock_settings,
            patch("deepagents_cli.skills.commands.console") as mock_console,
        ):
            mock_console.print = capture_print
            _info("test-builtin", agent="agent")

        joined = "\n".join(output)
        assert "Built-in Skill" in joined
        assert "User Skill" not in joined

    def test_built_in_skill_no_shadow_warning(self, tmp_path: Path) -> None:
        """Built-in skills should never trigger a shadow warning."""
        built_in_dir = tmp_path / "built_in_skills"
        user_dir = tmp_path / "user_skills"
        self._make_skill_dir(built_in_dir, "shared-skill", "Built-in version")
        self._make_skill_dir(user_dir, "shared-skill", "User version")

        mock_settings = patch(
            "deepagents_cli.skills.commands.Settings.from_environment",
            return_value=type(
                "FakeSettings",
                (),
                {
                    "get_built_in_skills_dir": staticmethod(lambda: built_in_dir),
                    "get_user_skills_dir": lambda _, _a: user_dir,
                    "get_project_skills_dir": lambda _: None,
                    "get_user_agent_skills_dir": lambda _: None,
                    "get_project_agent_skills_dir": lambda _: None,
                },
            )(),
        )

        output: list[str] = []

        def capture_print(*args: str, **_: str) -> None:
            output.append(" ".join(str(a) for a in args))

        with (
            mock_settings,
            patch("deepagents_cli.skills.commands.console") as mock_console,
        ):
            mock_console.print = capture_print
            # User overrides built-in; info shows user version, no shadow note
            _info("shared-skill", agent="agent")

        joined = "\n".join(output)
        assert "overrides" not in joined.lower()
        assert "shadows" not in joined.lower()


class TestListBuiltInSkillsDisplay:
    """Test that `skills list` renders built-in skills correctly."""

    def _make_skill_dir(self, parent: Path, name: str, description: str) -> None:
        """Create a minimal skill directory with a valid SKILL.md."""
        skill_dir = parent / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\nContent\n"
        )

    def test_built_in_section_rendered(self, tmp_path: Path) -> None:
        """Built-in skills should appear under 'Built-in Skills:' heading."""
        built_in_dir = tmp_path / "built_in_skills"
        self._make_skill_dir(built_in_dir, "test-builtin", "A built-in skill")

        mock_settings = patch(
            "deepagents_cli.skills.commands.Settings.from_environment",
            return_value=type(
                "FakeSettings",
                (),
                {
                    "get_built_in_skills_dir": staticmethod(lambda: built_in_dir),
                    "get_user_skills_dir": lambda _, _a: None,
                    "get_project_skills_dir": lambda _: None,
                    "get_user_agent_skills_dir": lambda _: None,
                    "get_project_agent_skills_dir": lambda _: None,
                },
            )(),
        )

        output: list[str] = []

        def capture_print(*args: str, **_: str) -> None:
            output.append(" ".join(str(a) for a in args))

        with (
            mock_settings,
            patch("deepagents_cli.skills.commands.console") as mock_console,
        ):
            mock_console.print = capture_print
            _list(agent="agent")

        joined = "\n".join(output)
        assert "Built-in Skills:" in joined
        assert "test-builtin" in joined

    def test_built_in_section_omits_path(self, tmp_path: Path) -> None:
        """Built-in skills should not display a filesystem path."""
        built_in_dir = tmp_path / "built_in_skills"
        self._make_skill_dir(built_in_dir, "test-builtin", "A built-in skill")

        mock_settings = patch(
            "deepagents_cli.skills.commands.Settings.from_environment",
            return_value=type(
                "FakeSettings",
                (),
                {
                    "get_built_in_skills_dir": staticmethod(lambda: built_in_dir),
                    "get_user_skills_dir": lambda _, _a: None,
                    "get_project_skills_dir": lambda _: None,
                    "get_user_agent_skills_dir": lambda _: None,
                    "get_project_agent_skills_dir": lambda _: None,
                },
            )(),
        )

        output: list[str] = []

        def capture_print(*args: str, **_: str) -> None:
            output.append(" ".join(str(a) for a in args))

        with (
            mock_settings,
            patch("deepagents_cli.skills.commands.console") as mock_console,
        ):
            mock_console.print = capture_print
            _list(agent="agent")

        joined = "\n".join(output)
        # Built-in section should NOT contain the tmp_path directory
        assert str(built_in_dir) not in joined


class TestSkillsLsDispatch:
    """Test that `execute_skills_command` dispatches 'ls' to `_list`."""

    def test_ls_dispatches_to_list(self, tmp_path: Path) -> None:
        """Verify `execute_skills_command` routes 'ls' to `_list()`."""
        built_in_dir = tmp_path / "built_in_skills"
        built_in_dir.mkdir()

        mock_settings = patch(
            "deepagents_cli.skills.commands.Settings.from_environment",
            return_value=type(
                "FakeSettings",
                (),
                {
                    "get_built_in_skills_dir": staticmethod(lambda: built_in_dir),
                    "get_user_skills_dir": lambda _, _a: None,
                    "get_project_skills_dir": lambda _: None,
                    "get_user_agent_skills_dir": lambda _: None,
                    "get_project_agent_skills_dir": lambda _: None,
                },
            )(),
        )

        args = argparse.Namespace(skills_command="ls", agent="agent", project=False)

        output: list[str] = []

        def capture_print(*args_p: str, **_: str) -> None:
            output.append(" ".join(str(a) for a in args_p))

        with (
            mock_settings,
            patch("deepagents_cli.skills.commands.console") as mock_console,
        ):
            mock_console.print = capture_print
            execute_skills_command(args)

        # Should have produced output (even if "No skills found")
        # rather than falling through to show_skills_help()
        joined = "\n".join(output)
        assert "No skills found" in joined or "Available Skills" in joined


class TestDeleteSkill:
    """Test cases for the _delete command."""

    @staticmethod
    def _create_test_skill(skills_dir: Path, skill_name: str) -> Path:
        """Create a test skill directory with a minimal SKILL.md.

        Args:
            skills_dir: Parent skills directory.
            skill_name: Name of the skill to create.

        Returns:
            Path to the created skill directory.
        """
        skill_dir = skills_dir / skill_name
        skill_dir.mkdir(parents=True)
        content = (
            "---\n"
            f"name: {skill_name}\n"
            "description: Test skill for unit tests\n"
            "---\n"
            "\n"
            f"# {skill_name} Skill\n"
            "\n"
            "Test content.\n"
        )
        (skill_dir / "SKILL.md").write_text(content)
        return skill_dir

    def test_delete_existing_skill_with_force(self, tmp_path: Path) -> None:
        """Test deleting an existing skill with --force flag."""
        user_skills_dir = tmp_path / ".deepagents" / "agent" / "skills"
        skill_dir = self._create_test_skill(user_skills_dir, "test-skill")
        assert skill_dir.exists()

        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_user_agent_skills_dir.return_value = None
        mock_settings.get_project_agent_skills_dir.return_value = None

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            _delete("test-skill", agent="agent", project=False, force=True)

        assert not skill_dir.exists()

    def test_delete_nonexistent_skill(self, tmp_path: Path) -> None:
        """Test deleting a skill that doesn't exist shows error."""
        user_skills_dir = tmp_path / ".deepagents" / "agent" / "skills"
        user_skills_dir.mkdir(parents=True)

        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_user_agent_skills_dir.return_value = None
        mock_settings.get_project_agent_skills_dir.return_value = None

        output: list[str] = []

        def capture_print(*args: str, **_: str) -> None:
            output.append(" ".join(str(a) for a in args))

        with (
            patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls,
            patch("deepagents_cli.skills.commands.console") as mock_console,
        ):
            mock_settings_cls.from_environment.return_value = mock_settings
            mock_console.print = capture_print
            _delete("nonexistent-skill", agent="agent", project=False, force=True)

        joined = "\n".join(output)
        assert "not found" in joined.lower()

    @pytest.mark.parametrize("response", ["y", "yes"])
    def test_delete_with_confirmation_accepted(
        self, tmp_path: Path, response: str
    ) -> None:
        """Test deleting a skill with user confirmation (y/yes)."""
        user_skills_dir = tmp_path / ".deepagents" / "agent" / "skills"
        skill_dir = self._create_test_skill(user_skills_dir, "test-skill")
        assert skill_dir.exists()

        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_user_agent_skills_dir.return_value = None
        mock_settings.get_project_agent_skills_dir.return_value = None

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            with patch("builtins.input", return_value=response):
                _delete("test-skill", agent="agent", project=False, force=False)

        assert not skill_dir.exists()

    def test_delete_with_confirmation_no(self, tmp_path: Path) -> None:
        """Test canceling skill deletion with user confirmation (no)."""
        user_skills_dir = tmp_path / ".deepagents" / "agent" / "skills"
        skill_dir = self._create_test_skill(user_skills_dir, "test-skill")
        assert skill_dir.exists()

        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_user_agent_skills_dir.return_value = None
        mock_settings.get_project_agent_skills_dir.return_value = None

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            with patch("builtins.input", return_value="n"):
                _delete("test-skill", agent="agent", project=False, force=False)

        assert skill_dir.exists()

    def test_delete_with_confirmation_empty_input(self, tmp_path: Path) -> None:
        """Test canceling skill deletion with empty input (default: no)."""
        user_skills_dir = tmp_path / ".deepagents" / "agent" / "skills"
        skill_dir = self._create_test_skill(user_skills_dir, "test-skill")
        assert skill_dir.exists()

        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_user_agent_skills_dir.return_value = None
        mock_settings.get_project_agent_skills_dir.return_value = None

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            with patch("builtins.input", return_value=""):
                _delete("test-skill", agent="agent", project=False, force=False)

        assert skill_dir.exists()

    def test_delete_with_keyboard_interrupt(self, tmp_path: Path) -> None:
        """Test canceling skill deletion with Ctrl+C."""
        user_skills_dir = tmp_path / ".deepagents" / "agent" / "skills"
        skill_dir = self._create_test_skill(user_skills_dir, "test-skill")
        assert skill_dir.exists()

        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_user_agent_skills_dir.return_value = None
        mock_settings.get_project_agent_skills_dir.return_value = None

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            with patch("builtins.input", side_effect=KeyboardInterrupt):
                _delete("test-skill", agent="agent", project=False, force=False)

        assert skill_dir.exists()

    def test_delete_with_eof_error(self, tmp_path: Path) -> None:
        """Test canceling skill deletion with EOF (piped stdin)."""
        user_skills_dir = tmp_path / ".deepagents" / "agent" / "skills"
        skill_dir = self._create_test_skill(user_skills_dir, "test-skill")
        assert skill_dir.exists()

        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_user_agent_skills_dir.return_value = None
        mock_settings.get_project_agent_skills_dir.return_value = None

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            with patch("builtins.input", side_effect=EOFError):
                _delete("test-skill", agent="agent", project=False, force=False)

        assert skill_dir.exists()

    def test_delete_invalid_skill_name(self, tmp_path: Path) -> None:
        """Test deleting with an invalid skill name shows error."""
        user_skills_dir = tmp_path / ".deepagents" / "agent" / "skills"
        user_skills_dir.mkdir(parents=True)

        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None

        invalid_names = [
            "../../../etc/passwd",
            "skill;rm -rf /",
            "",
            "skill name",  # space
        ]

        output: list[str] = []

        def capture_print(*args: str, **_: str) -> None:
            output.append(" ".join(str(a) for a in args))

        for invalid_name in invalid_names:
            output.clear()

            with (
                patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls,
                patch("deepagents_cli.skills.commands.console") as mock_console,
            ):
                mock_settings_cls.from_environment.return_value = mock_settings
                mock_console.print = capture_print
                _delete(invalid_name, agent="agent", project=False, force=True)

            joined = "\n".join(output)
            assert "invalid skill name" in joined.lower(), (
                f"Expected error for '{invalid_name}', got: {joined}"
            )

    def test_delete_project_skill(self, tmp_path: Path) -> None:
        """Test deleting a project-level skill."""
        project_skills_dir = tmp_path / "project" / ".deepagents" / "skills"
        skill_dir = self._create_test_skill(project_skills_dir, "project-skill")
        assert skill_dir.exists()

        mock_settings = MagicMock()
        user_dir = tmp_path / ".deepagents" / "agent" / "skills"
        mock_settings.get_user_skills_dir.return_value = user_dir
        mock_settings.get_project_skills_dir.return_value = project_skills_dir
        mock_settings.get_user_agent_skills_dir.return_value = None
        mock_settings.get_project_agent_skills_dir.return_value = None

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            _delete("project-skill", agent="agent", project=True, force=True)

        assert not skill_dir.exists()

    def test_delete_project_skill_not_in_project(self, tmp_path: Path) -> None:
        """Test deleting a project skill when not in a project directory."""
        mock_settings = MagicMock()
        user_dir = tmp_path / ".deepagents" / "agent" / "skills"
        mock_settings.get_user_skills_dir.return_value = user_dir
        mock_settings.get_project_skills_dir.return_value = None

        output: list[str] = []

        def capture_print(*args: str, **_: str) -> None:
            output.append(" ".join(str(a) for a in args))

        with (
            patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls,
            patch("deepagents_cli.skills.commands.console") as mock_console,
        ):
            mock_settings_cls.from_environment.return_value = mock_settings
            mock_console.print = capture_print
            _delete("any-skill", agent="agent", project=True, force=True)

        joined = "\n".join(output)
        assert "not in a project directory" in joined.lower()

    def test_delete_skill_with_supporting_files(self, tmp_path: Path) -> None:
        """Test deleting a skill that contains multiple supporting files."""
        user_skills_dir = tmp_path / ".deepagents" / "agent" / "skills"
        skill_dir = self._create_test_skill(user_skills_dir, "complex-skill")

        (skill_dir / "helper.py").write_text("# Helper script")
        (skill_dir / "config.json").write_text("{}")
        (skill_dir / "subdir").mkdir()
        (skill_dir / "subdir" / "nested.txt").write_text("nested file")

        assert skill_dir.exists()
        assert (skill_dir / "helper.py").exists()
        assert (skill_dir / "subdir" / "nested.txt").exists()

        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_user_agent_skills_dir.return_value = None
        mock_settings.get_project_agent_skills_dir.return_value = None

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            _delete("complex-skill", agent="agent", project=False, force=True)

        assert not skill_dir.exists()

    def test_delete_skill_for_specific_agent(self, tmp_path: Path) -> None:
        """Test deleting a skill for a specific agent."""
        agent1_skills_dir = tmp_path / ".deepagents" / "agent1" / "skills"
        agent2_skills_dir = tmp_path / ".deepagents" / "agent2" / "skills"

        skill_dir_agent1 = self._create_test_skill(agent1_skills_dir, "shared-skill")
        skill_dir_agent2 = self._create_test_skill(agent2_skills_dir, "shared-skill")

        assert skill_dir_agent1.exists()
        assert skill_dir_agent2.exists()

        mock_settings = MagicMock()
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_user_skills_dir.return_value = agent1_skills_dir
        mock_settings.get_user_agent_skills_dir.return_value = None
        mock_settings.get_project_agent_skills_dir.return_value = None

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            _delete("shared-skill", agent="agent1", project=False, force=True)

        assert not skill_dir_agent1.exists()
        assert skill_dir_agent2.exists()

    def test_delete_rmtree_os_error(self, tmp_path: Path) -> None:
        """Test that OSError during shutil.rmtree exits with code 1."""
        user_skills_dir = tmp_path / ".deepagents" / "agent" / "skills"
        skill_dir = self._create_test_skill(user_skills_dir, "test-skill")
        assert skill_dir.exists()

        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_user_agent_skills_dir.return_value = None
        mock_settings.get_project_agent_skills_dir.return_value = None

        output: list[str] = []

        def capture_print(*args: str, **_: str) -> None:
            output.append(" ".join(str(a) for a in args))

        with (
            patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls,
            patch("deepagents_cli.skills.commands.console") as mock_console,
            patch("shutil.rmtree", side_effect=OSError("Permission denied")),
        ):
            mock_settings_cls.from_environment.return_value = mock_settings
            mock_console.print = capture_print
            with pytest.raises(SystemExit) as exc_info:
                _delete("test-skill", agent="agent", project=False, force=True)

        assert exc_info.value.code == 1
        joined = "\n".join(output)
        assert "failed to fully delete skill" in joined.lower()
        assert "partially removed" in joined.lower()
        # Skill directory should still exist since rmtree was mocked to fail
        assert skill_dir.exists()

    def test_delete_refuses_when_base_dir_none(self, tmp_path: Path) -> None:
        """Deletion should be refused when base skills directory is None."""
        agent_skills_dir = tmp_path / ".agents" / "skills"
        self._create_test_skill(agent_skills_dir, "orphan-skill")

        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = None
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_user_agent_skills_dir.return_value = agent_skills_dir
        mock_settings.get_project_agent_skills_dir.return_value = None

        output: list[str] = []

        def capture_print(*args: str, **_: str) -> None:
            output.append(" ".join(str(a) for a in args))

        with (
            patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls,
            patch("deepagents_cli.skills.commands.console") as mock_console,
        ):
            mock_settings_cls.from_environment.return_value = mock_settings
            mock_console.print = capture_print
            _delete("orphan-skill", agent="agent", project=False, force=True)

        joined = "\n".join(output)
        assert "cannot determine" in joined.lower() or "refusing" in joined.lower()
        # Must NOT have been deleted
        assert (agent_skills_dir / "orphan-skill").exists()


class TestDeleteArgparsing:
    """Test argparse wiring for `deepagents skills delete`."""

    def test_delete_args_parsed(self) -> None:
        """Verify `skills delete my-skill --force --project` parses correctly."""
        with patch(
            "sys.argv",
            ["deepagents", "skills", "delete", "my-skill", "--force", "--project"],
        ):
            args = parse_args()
        assert args.command == "skills"
        assert args.skills_command == "delete"
        assert args.name == "my-skill"
        assert args.force is True
        assert args.project is True

    def test_delete_args_defaults(self) -> None:
        """Verify default values for optional delete arguments."""
        with patch("sys.argv", ["deepagents", "skills", "delete", "my-skill"]):
            args = parse_args()
        assert args.force is False
        assert args.project is False
        assert args.agent == "agent"

    def test_delete_help_shows_delete_options(self) -> None:
        """Running `deepagents skills delete -h` should show delete options."""
        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=120)

        with (
            patch("sys.argv", ["deepagents", "skills", "delete", "-h"]),
            patch("deepagents_cli.ui.console", test_console),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()

        assert exc_info.value.code in (0, None)
        output = buf.getvalue()
        assert "--force" in output or "-f" in output
        assert "--project" in output

    def test_execute_skills_command_dispatches_delete(self, tmp_path: Path) -> None:
        """Verify `execute_skills_command` routes 'delete' to `_delete()`."""
        user_skills_dir = tmp_path / ".deepagents" / "agent" / "skills"
        user_skills_dir.mkdir(parents=True)

        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_user_agent_skills_dir.return_value = None
        mock_settings.get_project_agent_skills_dir.return_value = None

        args = argparse.Namespace(
            skills_command="delete",
            name="nonexistent-skill",
            agent="agent",
            project=False,
            force=True,
        )

        output: list[str] = []

        def capture_print(*args_p: str, **_: str) -> None:
            output.append(" ".join(str(a) for a in args_p))

        with (
            patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls,
            patch("deepagents_cli.skills.commands.console") as mock_console,
        ):
            mock_settings_cls.from_environment.return_value = mock_settings
            mock_console.print = capture_print
            execute_skills_command(args)

        # Should have dispatched to _delete and shown "not found"
        # rather than falling through to show_skills_help()
        joined = "\n".join(output)
        assert "not found" in joined.lower()
