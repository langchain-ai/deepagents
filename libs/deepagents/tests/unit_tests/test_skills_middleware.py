"""Tests for SkillsMiddleware."""

import tempfile
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage

from deepagents.middleware.skills import (
    Skill,
    SkillsMiddleware,
    SkillsState,
    _discover_skills,
    _parse_skill_md,
    _validate_skill_name,
)


class TestValidateSkillName:
    """Tests for _validate_skill_name function."""

    def test_valid_names(self):
        """Test that valid skill names pass validation."""
        valid_names = [
            "test-skill",
            "python-expert",
            "code-reviewer",
            "skill123",
            "a",
            "test-skill-with-many-parts",
        ]
        for name in valid_names:
            _validate_skill_name(name)  # Should not raise

    def test_empty_name(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_skill_name("")

    def test_path_traversal_attempts(self):
        """Test that path traversal attempts are blocked."""
        invalid_names = [
            "../etc/passwd",
            "skill/../etc",
            "~/skill",
            "skill/subdir",
            "skill\\windows",
        ]
        for name in invalid_names:
            with pytest.raises(ValueError, match="invalid characters"):
                _validate_skill_name(name)

    def test_uppercase_rejected(self):
        """Test that uppercase letters are rejected."""
        with pytest.raises(ValueError, match="lowercase"):
            _validate_skill_name("MySkill")

    def test_special_chars_rejected(self):
        """Test that special characters are rejected."""
        invalid_names = ["skill@name", "skill_name", "skill.name", "skill name"]
        for name in invalid_names:
            with pytest.raises(ValueError, match="lowercase"):
                _validate_skill_name(name)


class TestParseSkillMd:
    """Tests for _parse_skill_md function."""

    def test_parse_valid_skill(self, tmp_path: Path):
        """Test parsing a valid SKILL.md file."""
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            """---
name: test-skill
description: A test skill for unit testing
author: Test Author
---

# Test Skill

This is a test skill with instructions.

## Examples
- Example 1
- Example 2
"""
        )

        skill = _parse_skill_md(skill_md)

        assert skill.name == "test-skill"
        assert skill.description == "A test skill for unit testing"
        assert "# Test Skill" in skill.instructions
        assert "Examples" in skill.instructions
        assert skill.metadata["author"] == "Test Author"
        assert str(tmp_path) in skill.source_path

    def test_parse_missing_file(self, tmp_path: Path):
        """Test parsing when SKILL.md doesn't exist."""
        skill_md = tmp_path / "nonexistent.md"

        with pytest.raises(ValueError, match="SKILL.md not found"):
            _parse_skill_md(skill_md)

    def test_parse_missing_frontmatter(self, tmp_path: Path):
        """Test parsing when YAML frontmatter is missing."""
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("Just some markdown without frontmatter")

        with pytest.raises(ValueError, match="missing YAML frontmatter"):
            _parse_skill_md(skill_md)

    def test_parse_invalid_yaml(self, tmp_path: Path):
        """Test parsing when YAML frontmatter is malformed."""
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            """---
name: test
invalid yaml: [unclosed bracket
---

Content
"""
        )

        with pytest.raises(ValueError, match="Failed to parse YAML"):
            _parse_skill_md(skill_md)

    def test_parse_missing_name(self, tmp_path: Path):
        """Test parsing when 'name' field is missing."""
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            """---
description: Missing name field
---

Content
"""
        )

        with pytest.raises(ValueError, match="missing required field 'name'"):
            _parse_skill_md(skill_md)

    def test_parse_missing_description(self, tmp_path: Path):
        """Test parsing when 'description' field is missing."""
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            """---
name: test-skill
---

Content
"""
        )

        with pytest.raises(ValueError, match="missing required field 'description'"):
            _parse_skill_md(skill_md)

    def test_parse_invalid_name_type(self, tmp_path: Path):
        """Test parsing when name is not a string."""
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            """---
name: 123
description: Test skill
---

Content
"""
        )

        with pytest.raises(ValueError, match="must be a string"):
            _parse_skill_md(skill_md)

    def test_parse_invalid_description_type(self, tmp_path: Path):
        """Test parsing when description is not a string."""
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            """---
name: test-skill
description: [list, instead, of, string]
---

Content
"""
        )

        with pytest.raises(ValueError, match="must be a string"):
            _parse_skill_md(skill_md)

    def test_parse_frontmatter_not_dict(self, tmp_path: Path):
        """Test parsing when frontmatter is not a dictionary."""
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            """---
- list
- instead
- of
- dict
---

Content
"""
        )

        with pytest.raises(ValueError, match="must be a dictionary"):
            _parse_skill_md(skill_md)

    def test_parse_invalid_skill_name_format(self, tmp_path: Path):
        """Test parsing when skill name has invalid format."""
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            """---
name: Invalid_Name
description: Test skill
---

Content
"""
        )

        with pytest.raises(ValueError, match="lowercase"):
            _parse_skill_md(skill_md)

    def test_parse_encoding_error(self, tmp_path: Path):
        """Test parsing when file has encoding issues."""
        skill_md = tmp_path / "SKILL.md"
        # Write invalid UTF-8
        skill_md.write_bytes(b"\x80\x81\x82")

        with pytest.raises(ValueError, match="Failed to read"):
            _parse_skill_md(skill_md)


class TestDiscoverSkills:
    """Tests for _discover_skills function."""

    def test_discover_no_directory(self, tmp_path: Path):
        """Test discovering skills when directory doesn't exist."""
        nonexistent = tmp_path / "nonexistent"
        skills = _discover_skills(nonexistent)
        assert skills == {}

    def test_discover_empty_directory(self, tmp_path: Path):
        """Test discovering skills in empty directory."""
        skills = _discover_skills(tmp_path)
        assert skills == {}

    def test_discover_single_skill(self, tmp_path: Path):
        """Test discovering a single valid skill."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            """---
name: test-skill
description: A test skill
---

Instructions here
"""
        )

        skills = _discover_skills(tmp_path)

        assert len(skills) == 1
        assert "test-skill" in skills
        assert skills["test-skill"].name == "test-skill"
        assert skills["test-skill"].description == "A test skill"

    def test_discover_multiple_skills(self, tmp_path: Path):
        """Test discovering multiple skills."""
        # Create first skill
        skill1_dir = tmp_path / "skill-one"
        skill1_dir.mkdir()
        (skill1_dir / "SKILL.md").write_text(
            """---
name: skill-one
description: First skill
---

Content 1
"""
        )

        # Create second skill
        skill2_dir = tmp_path / "skill-two"
        skill2_dir.mkdir()
        (skill2_dir / "SKILL.md").write_text(
            """---
name: skill-two
description: Second skill
---

Content 2
"""
        )

        skills = _discover_skills(tmp_path)

        assert len(skills) == 2
        assert "skill-one" in skills
        assert "skill-two" in skills

    def test_discover_ignores_files(self, tmp_path: Path):
        """Test that discovery ignores files (only processes directories)."""
        # Create a SKILL.md file directly in the skills directory
        (tmp_path / "SKILL.md").write_text(
            """---
name: ignored-skill
description: Should be ignored
---

Content
"""
        )

        skills = _discover_skills(tmp_path)
        assert len(skills) == 0

    def test_discover_ignores_directories_without_skill_md(self, tmp_path: Path):
        """Test that directories without SKILL.md are ignored."""
        (tmp_path / "empty-dir").mkdir()
        (tmp_path / "dir-with-readme").mkdir()
        (tmp_path / "dir-with-readme" / "README.md").write_text("readme")

        skills = _discover_skills(tmp_path)
        assert len(skills) == 0

    def test_discover_duplicate_skill_names(self, tmp_path: Path, capsys):
        """Test that duplicate skill names raise an error."""
        # Create first skill
        skill1_dir = tmp_path / "skill-dir-1"
        skill1_dir.mkdir()
        (skill1_dir / "SKILL.md").write_text(
            """---
name: duplicate-name
description: First skill
---

Content 1
"""
        )

        # Create second skill with same name
        skill2_dir = tmp_path / "skill-dir-2"
        skill2_dir.mkdir()
        (skill2_dir / "SKILL.md").write_text(
            """---
name: duplicate-name
description: Second skill
---

Content 2
"""
        )

        with pytest.raises(ValueError, match="Duplicate skill name 'duplicate-name'"):
            _discover_skills(tmp_path)

    def test_discover_skips_invalid_skills(self, tmp_path: Path, capsys):
        """Test that invalid skills are skipped with a warning."""
        # Create valid skill
        valid_dir = tmp_path / "valid-skill"
        valid_dir.mkdir()
        (valid_dir / "SKILL.md").write_text(
            """---
name: valid-skill
description: A valid skill
---

Content
"""
        )

        # Create invalid skill (missing description)
        invalid_dir = tmp_path / "invalid-skill"
        invalid_dir.mkdir()
        (invalid_dir / "SKILL.md").write_text(
            """---
name: invalid-skill
---

Content
"""
        )

        skills = _discover_skills(tmp_path)

        # Should only discover the valid skill
        assert len(skills) == 1
        assert "valid-skill" in skills

        # Check that a warning was printed
        captured = capsys.readouterr()
        assert "Warning: Failed to load skill" in captured.out
        assert "invalid-skill" in captured.out


class TestSkillsMiddleware:
    """Tests for SkillsMiddleware class."""

    def test_init_default(self, tmp_path: Path):
        """Test initialization with default settings."""
        # Create a simple skill
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: test-skill
description: Test skill
---

Instructions
"""
        )

        middleware = SkillsMiddleware(skills_dir=tmp_path)

        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "use_skill"
        assert middleware.system_prompt is not None

    def test_init_with_auto_activate(self, tmp_path: Path):
        """Test initialization with auto_activate."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: test-skill
description: Test skill
---

Instructions
"""
        )

        middleware = SkillsMiddleware(skills_dir=tmp_path, auto_activate=["test-skill"])

        assert middleware._auto_activate == ["test-skill"]

    def test_init_auto_activate_nonexistent_skill(self, tmp_path: Path):
        """Test that auto_activate with nonexistent skill raises error."""
        with pytest.raises(ValueError, match="Auto-activate skill 'nonexistent' not found"):
            SkillsMiddleware(skills_dir=tmp_path, auto_activate=["nonexistent"])

    def test_init_with_custom_system_prompt(self, tmp_path: Path):
        """Test initialization with custom system prompt."""
        custom_prompt = "Custom skills prompt"
        middleware = SkillsMiddleware(skills_dir=tmp_path, system_prompt=custom_prompt)

        assert middleware.system_prompt == custom_prompt

    def test_init_with_preloaded_skills(self):
        """Test initialization with pre-loaded skills dictionary."""
        skills = {
            "test-skill": Skill(
                name="test-skill",
                description="Pre-loaded test skill",
                instructions="Instructions",
            )
        }

        middleware = SkillsMiddleware(skills=skills)

        assert len(middleware._skills) == 1
        assert "test-skill" in middleware._skills

    def test_use_skill_tool_activates_skill(self, tmp_path: Path):
        """Test that use_skill tool activates a skill."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: test-skill
description: Test skill
---

Test instructions
"""
        )

        middleware = SkillsMiddleware(skills_dir=tmp_path)
        use_skill_tool = middleware.tools[0]

        # Create mock runtime
        state = SkillsState(active_skills=[])
        updates = []

        def stream_writer(update):
            updates.append(update)

        runtime = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_id",
            store=None,
            stream_writer=stream_writer,
            config={},
        )

        result = use_skill_tool.invoke({"runtime": runtime, "name": "test-skill"})

        assert "Activated skill: test-skill" in result
        assert "Test skill" in result
        assert "Test instructions" in result
        assert {"active_skills": ["test-skill"]} in updates

    def test_use_skill_tool_nonexistent_skill(self, tmp_path: Path):
        """Test that use_skill tool returns error for nonexistent skill."""
        middleware = SkillsMiddleware(skills_dir=tmp_path)
        use_skill_tool = middleware.tools[0]

        state = SkillsState(active_skills=[])
        runtime = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_id",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )

        result = use_skill_tool.invoke({"runtime": runtime, "name": "nonexistent"})

        assert "Error: Skill 'nonexistent' not found" in result

    def test_use_skill_tool_already_active(self, tmp_path: Path):
        """Test that use_skill tool handles already active skills."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: test-skill
description: Test skill
---

Instructions
"""
        )

        middleware = SkillsMiddleware(skills_dir=tmp_path)
        use_skill_tool = middleware.tools[0]

        state = SkillsState(active_skills=["test-skill"])
        runtime = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_id",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )

        result = use_skill_tool.invoke({"runtime": runtime, "name": "test-skill"})

        assert "already active" in result

    def test_middleware_adds_to_agent(self, tmp_path: Path):
        """Test that middleware correctly adds tools and state to agent."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: test-skill
description: Test skill
---

Instructions
"""
        )

        middleware = [SkillsMiddleware(skills_dir=tmp_path)]
        agent = create_agent(model="claude-sonnet-4-20250514", middleware=middleware, tools=[])

        # Check that state channel is added
        assert "active_skills" in agent.stream_channels

        # Check that use_skill tool is added
        agent_tools = agent.nodes["tools"].bound._tools_by_name.keys()
        assert "use_skill" in agent_tools


class TestSkillsState:
    """Tests for SkillsState schema."""

    def test_state_schema_reducer(self):
        """Test that active_skills uses set union reducer."""
        # The reducer should merge lists and remove duplicates
        left = ["skill1", "skill2"]
        right = ["skill2", "skill3"]

        # Get the reducer from the annotation
        from typing import get_args

        reducer = get_args(SkillsState.__annotations__["active_skills"])[1]

        result = reducer(left, right)

        assert set(result) == {"skill1", "skill2", "skill3"}
