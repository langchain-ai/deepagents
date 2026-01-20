"""Unit tests for skills command sanitization and validation."""

from pathlib import Path

import pytest

from deepagents_cli.skills.commands import _validate_name, _validate_skill_path


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
