"""Unit tests for skills loading functionality."""

from pathlib import Path
from unittest.mock import patch

from deepagents_cli._version import __version__ as _cli_version
from deepagents_cli.config import Settings
from deepagents_cli.skills.load import list_skills


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

    def test_list_skills_empty_directory(self, tmp_path: Path) -> None:
        """Test listing skills from an empty directory."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        skills = list_skills(user_skills_dir=skills_dir, project_skills_dir=None)
        assert skills == []

    def test_list_skills_with_valid_skill(self, tmp_path: Path) -> None:
        """Test listing a valid skill with proper YAML frontmatter."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: test-skill
description: A test skill
---

# Test Skill

This is a test skill.
""")

        skills = list_skills(user_skills_dir=skills_dir, project_skills_dir=None)
        assert len(skills) == 1
        assert skills[0]["name"] == "test-skill"
        assert skills[0]["description"] == "A test skill"
        assert skills[0]["source"] == "user"
        assert Path(skills[0]["path"]) == skill_md

    def test_list_skills_source_parameter(self, tmp_path: Path) -> None:
        """Test that source parameter is correctly set for project skills."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        skill_dir = skills_dir / "project-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: project-skill
description: A project skill
---

# Project Skill
""")

        # Test with project source
        skills = list_skills(user_skills_dir=None, project_skills_dir=skills_dir)
        assert len(skills) == 1
        assert skills[0]["source"] == "project"

    def test_list_skills_missing_frontmatter(self, tmp_path: Path) -> None:
        """Test that skills without YAML frontmatter are skipped."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        skill_dir = skills_dir / "invalid-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("# Invalid Skill\n\nNo frontmatter here.")

        skills = list_skills(user_skills_dir=skills_dir, project_skills_dir=None)
        assert skills == []

    def test_list_skills_missing_required_fields(self, tmp_path: Path) -> None:
        """Test that skills with incomplete frontmatter are skipped."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        # Missing description
        skill_dir_1 = skills_dir / "incomplete-1"
        skill_dir_1.mkdir()
        (skill_dir_1 / "SKILL.md").write_text("""---
name: incomplete-1
---
Content
""")

        # Missing name
        skill_dir_2 = skills_dir / "incomplete-2"
        skill_dir_2.mkdir()
        (skill_dir_2 / "SKILL.md").write_text("""---
description: Missing name
---
Content
""")

        skills = list_skills(user_skills_dir=skills_dir, project_skills_dir=None)
        assert skills == []

    def test_list_skills_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test listing skills from a non-existent directory."""
        skills_dir = tmp_path / "nonexistent"
        skills = list_skills(user_skills_dir=skills_dir, project_skills_dir=None)
        assert skills == []


class TestListSkillsMultipleDirectories:
    """Test list_skills function for loading from multiple directories."""

    def test_list_skills_user_only(self, tmp_path: Path) -> None:
        """Test loading skills from user directory only."""
        user_dir = tmp_path / "user_skills"
        user_dir.mkdir()

        skill_dir = user_dir / "user-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: user-skill
description: A user skill
---
Content
""")

        skills = list_skills(user_skills_dir=user_dir, project_skills_dir=None)
        assert len(skills) == 1
        assert skills[0]["name"] == "user-skill"
        assert skills[0]["source"] == "user"

    def test_list_skills_project_only(self, tmp_path: Path) -> None:
        """Test loading skills from project directory only."""
        project_dir = tmp_path / "project_skills"
        project_dir.mkdir()

        skill_dir = project_dir / "project-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: project-skill
description: A project skill
---
Content
""")

        skills = list_skills(user_skills_dir=None, project_skills_dir=project_dir)
        assert len(skills) == 1
        assert skills[0]["name"] == "project-skill"
        assert skills[0]["source"] == "project"

    def test_list_skills_both_sources(self, tmp_path: Path) -> None:
        """Test loading skills from both user and project directories."""
        user_dir = tmp_path / "user_skills"
        user_dir.mkdir()
        project_dir = tmp_path / "project_skills"
        project_dir.mkdir()

        # User skill
        user_skill_dir = user_dir / "user-skill"
        user_skill_dir.mkdir()
        (user_skill_dir / "SKILL.md").write_text("""---
name: user-skill
description: A user skill
---
Content
""")

        # Project skill
        project_skill_dir = project_dir / "project-skill"
        project_skill_dir.mkdir()
        (project_skill_dir / "SKILL.md").write_text("""---
name: project-skill
description: A project skill
---
Content
""")

        skills = list_skills(user_skills_dir=user_dir, project_skills_dir=project_dir)
        assert len(skills) == 2

        skill_names = {s["name"] for s in skills}
        assert "user-skill" in skill_names
        assert "project-skill" in skill_names

        # Verify sources
        user_skill = next(s for s in skills if s["name"] == "user-skill")
        project_skill = next(s for s in skills if s["name"] == "project-skill")
        assert user_skill["source"] == "user"
        assert project_skill["source"] == "project"

    def test_list_skills_project_overrides_user(self, tmp_path: Path) -> None:
        """Test that project skills override user skills with the same name."""
        user_dir = tmp_path / "user_skills"
        user_dir.mkdir()
        project_dir = tmp_path / "project_skills"
        project_dir.mkdir()

        # User skill
        user_skill_dir = user_dir / "shared-skill"
        user_skill_dir.mkdir()
        (user_skill_dir / "SKILL.md").write_text("""---
name: shared-skill
description: User version
---
Content
""")

        # Project skill with same name
        project_skill_dir = project_dir / "shared-skill"
        project_skill_dir.mkdir()
        (project_skill_dir / "SKILL.md").write_text("""---
name: shared-skill
description: Project version
---
Content
""")

        skills = list_skills(user_skills_dir=user_dir, project_skills_dir=project_dir)
        assert len(skills) == 1  # Only one skill with this name

        skill = skills[0]
        assert skill["name"] == "shared-skill"
        assert skill["description"] == "Project version"
        assert skill["source"] == "project"

    def test_list_skills_empty_directories(self, tmp_path: Path) -> None:
        """Test loading from empty directories."""
        user_dir = tmp_path / "user_skills"
        user_dir.mkdir()
        project_dir = tmp_path / "project_skills"
        project_dir.mkdir()

        skills = list_skills(user_skills_dir=user_dir, project_skills_dir=project_dir)
        assert skills == []

    def test_list_skills_no_directories(self):
        """Test loading with no directories specified."""
        skills = list_skills(user_skills_dir=None, project_skills_dir=None)
        assert skills == []

    def test_list_skills_multiple_user_skills(self, tmp_path: Path) -> None:
        """Test loading multiple skills from user directory."""
        user_dir = tmp_path / "user_skills"
        user_dir.mkdir()

        # Create multiple skills
        for i in range(3):
            skill_dir = user_dir / f"skill-{i}"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(f"""---
name: skill-{i}
description: Skill number {i}
---
Content
""")

        skills = list_skills(user_skills_dir=user_dir, project_skills_dir=None)
        assert len(skills) == 3
        skill_names = {s["name"] for s in skills}
        assert skill_names == {"skill-0", "skill-1", "skill-2"}

    def test_list_skills_mixed_valid_invalid(self, tmp_path: Path) -> None:
        """Test loading with a mix of valid and invalid skills."""
        user_dir = tmp_path / "user_skills"
        user_dir.mkdir()

        # Valid skill
        valid_skill_dir = user_dir / "valid-skill"
        valid_skill_dir.mkdir()
        (valid_skill_dir / "SKILL.md").write_text("""---
name: valid-skill
description: A valid skill
---
Content
""")

        # Invalid skill (missing description)
        invalid_skill_dir = user_dir / "invalid-skill"
        invalid_skill_dir.mkdir()
        (invalid_skill_dir / "SKILL.md").write_text("""---
name: invalid-skill
---
Content
""")

        skills = list_skills(user_skills_dir=user_dir, project_skills_dir=None)
        assert len(skills) == 1
        assert skills[0]["name"] == "valid-skill"


class TestListSkillsAliasDirectories:
    """Test `list_skills` with `.agents` alias directories."""

    def test_user_agent_skills_dir_precedence(self, tmp_path: Path) -> None:
        """Test that `~/.agents/skills` overrides `~/.deepagents/agent/skills`."""
        user_deepagents_dir = tmp_path / "user_deepagents_skills"
        user_agent_dir = tmp_path / "user_agent_skills"

        # Create same skill in both directories
        _create_skill(
            user_deepagents_dir / "shared-skill",
            "shared-skill",
            "From deepagents user dir",
        )
        _create_skill(
            user_agent_dir / "shared-skill",
            "shared-skill",
            "From agents user dir",
        )

        skills = list_skills(
            user_skills_dir=user_deepagents_dir,
            project_skills_dir=None,
            user_agent_skills_dir=user_agent_dir,
            project_agent_skills_dir=None,
        )

        assert len(skills) == 1
        assert skills[0]["name"] == "shared-skill"
        assert skills[0]["description"] == "From agents user dir"
        assert skills[0]["source"] == "user"

    def test_project_agent_skills_dir_precedence(self, tmp_path: Path) -> None:
        """Test that `.agents/skills` overrides `.deepagents/skills`."""
        project_deepagents_dir = tmp_path / "project_deepagents_skills"
        project_agent_dir = tmp_path / "project_agent_skills"

        # Create same skill in both directories
        _create_skill(
            project_deepagents_dir / "shared-skill",
            "shared-skill",
            "From deepagents project dir",
        )
        _create_skill(
            project_agent_dir / "shared-skill",
            "shared-skill",
            "From agents project dir",
        )

        skills = list_skills(
            user_skills_dir=None,
            project_skills_dir=project_deepagents_dir,
            user_agent_skills_dir=None,
            project_agent_skills_dir=project_agent_dir,
        )

        assert len(skills) == 1
        assert skills[0]["name"] == "shared-skill"
        assert skills[0]["description"] == "From agents project dir"
        assert skills[0]["source"] == "project"

    def test_full_precedence_chain(self, tmp_path: Path) -> None:
        """Test full precedence: `.agents/skills` (project) wins over all."""
        user_deepagents_dir = tmp_path / "user_deepagents_skills"
        user_agent_dir = tmp_path / "user_agent_skills"
        project_deepagents_dir = tmp_path / "project_deepagents_skills"
        project_agent_dir = tmp_path / "project_agent_skills"

        # Create same skill in all 4 directories
        _create_skill(
            user_deepagents_dir / "shared-skill",
            "shared-skill",
            "From deepagents user dir (lowest)",
        )
        _create_skill(
            user_agent_dir / "shared-skill",
            "shared-skill",
            "From agents user dir",
        )
        _create_skill(
            project_deepagents_dir / "shared-skill",
            "shared-skill",
            "From deepagents project dir",
        )
        _create_skill(
            project_agent_dir / "shared-skill",
            "shared-skill",
            "From agents project dir (highest)",
        )

        skills = list_skills(
            user_skills_dir=user_deepagents_dir,
            project_skills_dir=project_deepagents_dir,
            user_agent_skills_dir=user_agent_dir,
            project_agent_skills_dir=project_agent_dir,
        )

        assert len(skills) == 1
        assert skills[0]["name"] == "shared-skill"
        assert skills[0]["description"] == "From agents project dir (highest)"
        assert skills[0]["source"] == "project"

    def test_mixed_sources_with_aliases(self, tmp_path: Path) -> None:
        """Test different skills from different directories are all discovered."""
        user_deepagents_dir = tmp_path / "user_deepagents_skills"
        user_agent_dir = tmp_path / "user_agent_skills"
        project_deepagents_dir = tmp_path / "project_deepagents_skills"
        project_agent_dir = tmp_path / "project_agent_skills"

        # Create different skills in each directory
        _create_skill(
            user_deepagents_dir / "skill-a",
            "skill-a",
            "Skill A from deepagents user",
        )
        _create_skill(
            user_agent_dir / "skill-b",
            "skill-b",
            "Skill B from agents user",
        )
        _create_skill(
            project_deepagents_dir / "skill-c",
            "skill-c",
            "Skill C from deepagents project",
        )
        _create_skill(
            project_agent_dir / "skill-d",
            "skill-d",
            "Skill D from agents project",
        )

        skills = list_skills(
            user_skills_dir=user_deepagents_dir,
            project_skills_dir=project_deepagents_dir,
            user_agent_skills_dir=user_agent_dir,
            project_agent_skills_dir=project_agent_dir,
        )

        assert len(skills) == 4
        skill_names = {s["name"] for s in skills}
        assert skill_names == {"skill-a", "skill-b", "skill-c", "skill-d"}

        # Verify sources
        skill_a = next(s for s in skills if s["name"] == "skill-a")
        skill_b = next(s for s in skills if s["name"] == "skill-b")
        skill_c = next(s for s in skills if s["name"] == "skill-c")
        skill_d = next(s for s in skills if s["name"] == "skill-d")

        assert skill_a["source"] == "user"
        assert skill_b["source"] == "user"
        assert skill_c["source"] == "project"
        assert skill_d["source"] == "project"

    def test_alias_directories_only(self, tmp_path: Path) -> None:
        """Test loading skills from only the alias directories."""
        user_agent_dir = tmp_path / "user_agent_skills"
        project_agent_dir = tmp_path / "project_agent_skills"

        _create_skill(
            user_agent_dir / "user-skill",
            "user-skill",
            "From agents user dir",
        )
        _create_skill(
            project_agent_dir / "project-skill",
            "project-skill",
            "From agents project dir",
        )

        skills = list_skills(
            user_skills_dir=None,
            project_skills_dir=None,
            user_agent_skills_dir=user_agent_dir,
            project_agent_skills_dir=project_agent_dir,
        )

        assert len(skills) == 2
        skill_names = {s["name"] for s in skills}
        assert skill_names == {"user-skill", "project-skill"}

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

    def test_built_in_skills_discovered(self, tmp_path: Path) -> None:
        """Test that built-in skills are discovered with source 'built-in'."""
        built_in_dir = tmp_path / "built_in_skills"
        _create_skill(
            built_in_dir / "test-builtin",
            "test-builtin",
            "A built-in skill",
        )

        skills = list_skills(
            built_in_skills_dir=built_in_dir,
            user_skills_dir=None,
            project_skills_dir=None,
        )
        assert len(skills) == 1
        assert skills[0]["name"] == "test-builtin"
        assert skills[0]["source"] == "built-in"

    def test_built_in_lowest_precedence(self, tmp_path: Path) -> None:
        """Test that user skills override built-in skills with the same name."""
        built_in_dir = tmp_path / "built_in_skills"
        user_dir = tmp_path / "user_skills"

        _create_skill(
            built_in_dir / "shared-skill",
            "shared-skill",
            "Built-in version",
        )
        _create_skill(
            user_dir / "shared-skill",
            "shared-skill",
            "User version",
        )

        skills = list_skills(
            built_in_skills_dir=built_in_dir,
            user_skills_dir=user_dir,
            project_skills_dir=None,
        )
        assert len(skills) == 1
        assert skills[0]["name"] == "shared-skill"
        assert skills[0]["description"] == "User version"
        assert skills[0]["source"] == "user"

    def test_project_overrides_built_in(self, tmp_path: Path) -> None:
        """Test that project skills override built-in skills with the same name."""
        built_in_dir = tmp_path / "built_in_skills"
        project_dir = tmp_path / "project_skills"

        _create_skill(
            built_in_dir / "shared-skill",
            "shared-skill",
            "Built-in version",
        )
        _create_skill(
            project_dir / "shared-skill",
            "shared-skill",
            "Project version",
        )

        skills = list_skills(
            built_in_skills_dir=built_in_dir,
            user_skills_dir=None,
            project_skills_dir=project_dir,
        )
        assert len(skills) == 1
        assert skills[0]["name"] == "shared-skill"
        assert skills[0]["description"] == "Project version"
        assert skills[0]["source"] == "project"

    def test_built_in_coexists_with_other_skills(self, tmp_path: Path) -> None:
        """Test that built-in skills with different names appear alongside others."""
        built_in_dir = tmp_path / "built_in_skills"
        user_dir = tmp_path / "user_skills"
        project_dir = tmp_path / "project_skills"

        _create_skill(
            built_in_dir / "builtin-skill",
            "builtin-skill",
            "A built-in skill",
        )
        _create_skill(
            user_dir / "user-skill",
            "user-skill",
            "A user skill",
        )
        _create_skill(
            project_dir / "project-skill",
            "project-skill",
            "A project skill",
        )

        skills = list_skills(
            built_in_skills_dir=built_in_dir,
            user_skills_dir=user_dir,
            project_skills_dir=project_dir,
        )
        assert len(skills) == 3
        skill_names = {s["name"] for s in skills}
        assert skill_names == {"builtin-skill", "user-skill", "project-skill"}

        # Verify sources
        builtin = next(s for s in skills if s["name"] == "builtin-skill")
        user = next(s for s in skills if s["name"] == "user-skill")
        proj = next(s for s in skills if s["name"] == "project-skill")
        assert builtin["source"] == "built-in"
        assert user["source"] == "user"
        assert proj["source"] == "project"

    def test_nonexistent_built_in_dir(self, tmp_path: Path) -> None:
        """Test that a nonexistent built-in directory is handled gracefully."""
        nonexistent = tmp_path / "nonexistent"

        skills = list_skills(
            built_in_skills_dir=nonexistent,
            user_skills_dir=None,
            project_skills_dir=None,
        )
        assert skills == []

    def test_real_skill_creator_ships(self) -> None:
        """Verify the actual built-in skill-creator SKILL.md exists and loads.

        Unlike other tests in this file, this uses the real package directory
        (not `tmp_path`) to ensure the built-in skill ships correctly.
        """
        built_in_dir = Settings.get_built_in_skills_dir()
        skill_md = built_in_dir / "skill-creator" / "SKILL.md"
        assert skill_md.exists(), f"Expected {skill_md} to exist"

        skills = list_skills(
            built_in_skills_dir=built_in_dir,
            user_skills_dir=None,
            project_skills_dir=None,
        )
        skill_names = {s["name"] for s in skills}
        assert "skill-creator" in skill_names

        creator = next(s for s in skills if s["name"] == "skill-creator")
        assert creator["source"] == "built-in"
        assert len(creator["description"]) > 0
        assert creator["license"] == "MIT"
        assert creator["compatibility"] == "designed for deepagents-cli"
        assert "deepagents-cli-version" in creator["metadata"]
        assert creator["metadata"]["deepagents-cli-version"] == _cli_version

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

        with patch("deepagents_cli.skills.load.list_skills_from_backend", patched_list):
            skills = list_skills(
                built_in_skills_dir=built_in_dir,
                user_skills_dir=user_dir,
                project_skills_dir=None,
            )

        # User skills should still load despite built-in source failing
        assert len(skills) == 1
        assert skills[0]["name"] == "user-skill"
