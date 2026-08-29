"""Unit tests for skills loading functionality."""

from pathlib import Path
from unittest.mock import patch

from deepagents_code.skills.load import list_skills


def _create_skill(skill_dir: Path, name: str, description: str) -> None:
    """Create a minimal skill directory with a valid `SKILL.md`.

    Args:
        skill_dir: Directory to create the skill in (will be created if needed).
        name: Skill name for frontmatter.
        description: Skill description for frontmatter.
    """
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"""---
name: {name}
description: {description}
---
Content
""")


class TestListSkillsSingleDirectory:
    """Test list_skills function for loading skills from a single directory."""


class TestListSkillsMultipleDirectories:
    """Test list_skills function for loading from multiple directories."""


class TestListSkillsAliasDirectories:
    """Test `list_skills` with `.agents` alias directories."""

    def test_nonexistent_alias_directories(self, tmp_path: Path) -> None:
        """Test that nonexistent alias directories are handled gracefully."""
        nonexistent_user = tmp_path / "nonexistent_user"
        nonexistent_project = tmp_path / "nonexistent_project"

        skills = list_skills(
            user_skills_dir=None,
            project_skills_dir=None,
            user_agent_skills_dir=nonexistent_user,
            project_agent_skills_dir=nonexistent_project,
        )

        assert skills == []


class TestListSkillsBuiltIn:
    """Test list_skills with built-in skills directory."""

    def test_nonexistent_built_in_dir(self, tmp_path: Path) -> None:
        """Test that a nonexistent built-in directory is handled gracefully."""
        nonexistent = tmp_path / "nonexistent"

        skills = list_skills(
            built_in_skills_dir=nonexistent,
            user_skills_dir=None,
            project_skills_dir=None,
        )
        assert skills == []

    def test_oserror_in_one_source_does_not_break_others(self, tmp_path: Path) -> None:
        """An OSError in one source should not prevent other sources from loading.

        This verifies the per-source error isolation in `list_skills`.
        """
        # Create a healthy user skills directory
        user_dir = tmp_path / "user_skills"
        _create_skill(user_dir / "user-skill", "user-skill", "A user skill")

        # Use a built-in dir that exists but will fail when FilesystemBackend
        # tries to read it — we simulate this by patching list_skills_from_backend
        # to raise OSError only for the built-in source
        built_in_dir = tmp_path / "built_in_skills"
        built_in_dir.mkdir()

        original_list = __import__(
            "deepagents.middleware.skills", fromlist=["_list_skills"]
        )._list_skills

        call_count = 0

        def patched_list(backend: object, source_path: str) -> list[object]:
            nonlocal call_count
            call_count += 1
            # First call is the built-in source — make it fail
            if call_count == 1:
                msg = "simulated permission error"
                raise OSError(msg)
            return original_list(backend=backend, source_path=source_path)

        with patch(
            "deepagents_code.skills.load.list_skills_from_backend", patched_list
        ):
            skills = list_skills(
                built_in_skills_dir=built_in_dir,
                user_skills_dir=user_dir,
                project_skills_dir=None,
            )

        # User skills should still load despite built-in source failing
        assert len(skills) == 1
        assert skills[0]["name"] == "user-skill"


class TestListSkillsClaudeDirectories:
    """Test `list_skills` with experimental Claude skills directories."""

    def test_nonexistent_claude_dirs_handled(self, tmp_path: Path) -> None:
        """Nonexistent Claude dirs are handled gracefully."""
        skills = list_skills(
            user_skills_dir=None,
            project_skills_dir=None,
            user_claude_skills_dir=tmp_path / "nonexistent_user",
            project_claude_skills_dir=tmp_path / "nonexistent_project",
        )
        assert skills == []


class TestListSkillsPluginNamespacing:
    """Plugin sources namespace names, including nested subfolders."""
