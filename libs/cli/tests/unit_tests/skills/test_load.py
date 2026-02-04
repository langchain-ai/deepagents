"""Unit tests for skills loading functionality."""

from pathlib import Path

from deepagents_cli.skills.load import list_skills


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

    def _create_skill(self, skill_dir: Path, name: str, description: str) -> None:
        """Helper to create a skill directory with `SKILL.md`."""
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(f"""---
name: {name}
description: {description}
---
Content
""")

    def test_user_agent_skills_dir_precedence(self, tmp_path: Path) -> None:
        """Test that `~/.agents/skills` overrides `~/.deepagents/agent/skills`."""
        user_deepagents_dir = tmp_path / "user_deepagents_skills"
        user_agent_dir = tmp_path / "user_agent_skills"

        # Create same skill in both directories
        self._create_skill(
            user_deepagents_dir / "shared-skill",
            "shared-skill",
            "From deepagents user dir",
        )
        self._create_skill(
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
        self._create_skill(
            project_deepagents_dir / "shared-skill",
            "shared-skill",
            "From deepagents project dir",
        )
        self._create_skill(
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
        self._create_skill(
            user_deepagents_dir / "shared-skill",
            "shared-skill",
            "From deepagents user dir (lowest)",
        )
        self._create_skill(
            user_agent_dir / "shared-skill",
            "shared-skill",
            "From agents user dir",
        )
        self._create_skill(
            project_deepagents_dir / "shared-skill",
            "shared-skill",
            "From deepagents project dir",
        )
        self._create_skill(
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
        self._create_skill(
            user_deepagents_dir / "skill-a",
            "skill-a",
            "Skill A from deepagents user",
        )
        self._create_skill(
            user_agent_dir / "skill-b",
            "skill-b",
            "Skill B from agents user",
        )
        self._create_skill(
            project_deepagents_dir / "skill-c",
            "skill-c",
            "Skill C from deepagents project",
        )
        self._create_skill(
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

        self._create_skill(
            user_agent_dir / "user-skill",
            "user-skill",
            "From agents user dir",
        )
        self._create_skill(
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
