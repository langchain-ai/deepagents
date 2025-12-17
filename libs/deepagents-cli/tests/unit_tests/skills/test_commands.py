"""Unit tests for skills command sanitization and validation."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from deepagents_cli.skills.commands import _validate_name, _validate_skill_path, _delete


class TestValidateSkillName:
    """Test skill name validation against path traversal and injection attacks."""

    def test_valid_skill_names(self):
        """Test that valid skill names are accepted."""
        valid_names = [
            "web-research",
            "langgraph-docs",
            "my_skill",
            "skill123",
            "MySkill",
            "skill-with-many-parts",
            "skill_with_underscores",
        ]
        for name in valid_names:
            is_valid, error = _validate_name(name)
            assert is_valid, f"Valid name '{name}' was rejected: {error}"
            assert error == ""

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
            assert ".." in error or "traversal" in error.lower()

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
                assert not is_valid_path, f"{attack_type} bypassed both validations: {skill_name}"
            else:
                # Name validation caught it - this is good
                assert name_error != "", f"No error message for {attack_type}"


def _create_test_skill(skills_dir: Path, skill_name: str) -> Path:
    """Helper function to create a test skill."""
    skill_dir = skills_dir / skill_name
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(f"""---
name: {skill_name}
description: Test skill for unit tests
---

# {skill_name} Skill

Test content.
""")
    return skill_dir


class TestDeleteSkill:
    """Test cases for the _delete command."""

    def test_delete_existing_skill_with_force(self, tmp_path: Path) -> None:
        """Test deleting an existing skill with --force flag."""
        # Setup: Create a skill
        user_skills_dir = tmp_path / ".deepagents" / "agent" / "skills"
        skill_dir = _create_test_skill(user_skills_dir, "test-skill")
        assert skill_dir.exists()

        # Mock Settings to use our tmp_path
        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            _delete("test-skill", agent="agent", project=False, force=True)

        # Verify: Skill directory should be deleted
        assert not skill_dir.exists()

    def test_delete_nonexistent_skill(self, tmp_path: Path, capsys) -> None:
        """Test deleting a skill that doesn't exist."""
        user_skills_dir = tmp_path / ".deepagents" / "agent" / "skills"
        user_skills_dir.mkdir(parents=True)

        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            _delete("nonexistent-skill", agent="agent", project=False, force=True)

        # Should print error message (via rich console)
        # The function returns early without deleting anything

    def test_delete_with_confirmation_yes(self, tmp_path: Path) -> None:
        """Test deleting a skill with user confirmation (yes)."""
        user_skills_dir = tmp_path / ".deepagents" / "agent" / "skills"
        skill_dir = _create_test_skill(user_skills_dir, "test-skill")
        assert skill_dir.exists()

        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            # Mock user input to confirm deletion
            with patch("builtins.input", return_value="y"):
                _delete("test-skill", agent="agent", project=False, force=False)

        # Verify: Skill directory should be deleted
        assert not skill_dir.exists()

    def test_delete_with_confirmation_no(self, tmp_path: Path) -> None:
        """Test canceling skill deletion with user confirmation (no)."""
        user_skills_dir = tmp_path / ".deepagents" / "agent" / "skills"
        skill_dir = _create_test_skill(user_skills_dir, "test-skill")
        assert skill_dir.exists()

        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            # Mock user input to cancel deletion
            with patch("builtins.input", return_value="n"):
                _delete("test-skill", agent="agent", project=False, force=False)

        # Verify: Skill directory should NOT be deleted
        assert skill_dir.exists()

    def test_delete_with_confirmation_empty_input(self, tmp_path: Path) -> None:
        """Test canceling skill deletion with empty input (default: no)."""
        user_skills_dir = tmp_path / ".deepagents" / "agent" / "skills"
        skill_dir = _create_test_skill(user_skills_dir, "test-skill")
        assert skill_dir.exists()

        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            # Mock user pressing Enter without input
            with patch("builtins.input", return_value=""):
                _delete("test-skill", agent="agent", project=False, force=False)

        # Verify: Skill directory should NOT be deleted (default is No)
        assert skill_dir.exists()

    def test_delete_with_keyboard_interrupt(self, tmp_path: Path) -> None:
        """Test canceling skill deletion with Ctrl+C."""
        user_skills_dir = tmp_path / ".deepagents" / "agent" / "skills"
        skill_dir = _create_test_skill(user_skills_dir, "test-skill")
        assert skill_dir.exists()

        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            # Mock Ctrl+C
            with patch("builtins.input", side_effect=KeyboardInterrupt):
                _delete("test-skill", agent="agent", project=False, force=False)

        # Verify: Skill directory should NOT be deleted
        assert skill_dir.exists()

    def test_delete_invalid_skill_name(self, tmp_path: Path) -> None:
        """Test deleting with an invalid skill name."""
        user_skills_dir = tmp_path / ".deepagents" / "agent" / "skills"
        user_skills_dir.mkdir(parents=True)

        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None

        # These should be rejected by name validation
        invalid_names = [
            "../../../etc/passwd",
            "skill;rm -rf /",
            "",
            "skill name",  # space
        ]

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            for invalid_name in invalid_names:
                # Should return early without attempting deletion
                _delete(invalid_name, agent="agent", project=False, force=True)

    def test_delete_project_skill(self, tmp_path: Path) -> None:
        """Test deleting a project-level skill."""
        project_skills_dir = tmp_path / "project" / ".deepagents" / "skills"
        skill_dir = _create_test_skill(project_skills_dir, "project-skill")
        assert skill_dir.exists()

        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = tmp_path / ".deepagents" / "agent" / "skills"
        mock_settings.get_project_skills_dir.return_value = project_skills_dir

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            _delete("project-skill", agent="agent", project=True, force=True)

        # Verify: Project skill directory should be deleted
        assert not skill_dir.exists()

    def test_delete_project_skill_not_in_project(self, tmp_path: Path) -> None:
        """Test deleting a project skill when not in a project directory."""
        mock_settings = MagicMock()
        mock_settings.get_user_skills_dir.return_value = tmp_path / ".deepagents" / "agent" / "skills"
        mock_settings.get_project_skills_dir.return_value = None  # Not in a project

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            # Should print error and return early
            _delete("any-skill", agent="agent", project=True, force=True)

    def test_delete_skill_with_supporting_files(self, tmp_path: Path) -> None:
        """Test deleting a skill that contains multiple supporting files."""
        user_skills_dir = tmp_path / ".deepagents" / "agent" / "skills"
        skill_dir = _create_test_skill(user_skills_dir, "complex-skill")

        # Add supporting files
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

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            _delete("complex-skill", agent="agent", project=False, force=True)

        # Verify: Entire skill directory including all files should be deleted
        assert not skill_dir.exists()

    def test_delete_skill_for_specific_agent(self, tmp_path: Path) -> None:
        """Test deleting a skill for a specific agent."""
        # Create skills for different agents
        agent1_skills_dir = tmp_path / ".deepagents" / "agent1" / "skills"
        agent2_skills_dir = tmp_path / ".deepagents" / "agent2" / "skills"

        skill_dir_agent1 = _create_test_skill(agent1_skills_dir, "shared-skill")
        skill_dir_agent2 = _create_test_skill(agent2_skills_dir, "shared-skill")

        assert skill_dir_agent1.exists()
        assert skill_dir_agent2.exists()

        mock_settings = MagicMock()
        mock_settings.get_project_skills_dir.return_value = None

        # Delete only for agent1
        mock_settings.get_user_skills_dir.return_value = agent1_skills_dir

        with patch("deepagents_cli.skills.commands.Settings") as mock_settings_cls:
            mock_settings_cls.from_environment.return_value = mock_settings
            _delete("shared-skill", agent="agent1", project=False, force=True)

        # Verify: Only agent1's skill should be deleted
        assert not skill_dir_agent1.exists()
        assert skill_dir_agent2.exists()  # agent2's skill should remain
