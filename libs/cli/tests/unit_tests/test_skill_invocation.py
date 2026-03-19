"""Unit tests for /skill:<name> command parsing and skill content loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_cli.command_registry import build_skill_commands, parse_skill_command
from deepagents_cli.skills.load import load_skill_content

if TYPE_CHECKING:
    from pathlib import Path


class TestLoadSkillContent:
    """Test load_skill_content() reads SKILL.md files correctly."""

    def test_valid_skill_file(self, tmp_path: Path) -> None:
        skill_md = tmp_path / "SKILL.md"
        content = "---\nname: test\ndescription: A test\n---\n\n# Test Skill\n"
        skill_md.write_text(content, encoding="utf-8")

        result = load_skill_content(str(skill_md))
        assert result == content

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        result = load_skill_content(str(tmp_path / "nonexistent" / "SKILL.md"))
        assert result is None

    def test_encoding_error_returns_none(self, tmp_path: Path) -> None:
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_bytes(b"\x80\x81\x82\xff\xfe")

        result = load_skill_content(str(skill_md))
        assert result is None

    def test_empty_file_returns_empty_string(self, tmp_path: Path) -> None:
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("", encoding="utf-8")

        result = load_skill_content(str(skill_md))
        assert result == ""


class TestBuildSkillCommands:
    """Test build_skill_commands() produces correct autocomplete tuples."""

    def test_empty_list(self) -> None:
        assert build_skill_commands([]) == []

    def test_single_skill(self) -> None:
        skills = [
            {
                "name": "web-research",
                "description": "Research topics on the web",
                "path": "/some/path/SKILL.md",
                "license": None,
                "compatibility": None,
                "metadata": {},
                "allowed_tools": [],
                "source": "user",
            }
        ]
        result = build_skill_commands(skills)  # type: ignore[arg-type]
        assert len(result) == 1
        name, desc, keywords = result[0]
        assert name == "/skill:web-research"
        assert desc == "Research topics on the web"
        assert keywords == "web-research"

    def test_multiple_skills(self) -> None:
        skills = [
            {
                "name": "skill-a",
                "description": "Skill A",
                "path": "/a/SKILL.md",
                "license": None,
                "compatibility": None,
                "metadata": {},
                "allowed_tools": [],
                "source": "user",
            },
            {
                "name": "skill-b",
                "description": "Skill B",
                "path": "/b/SKILL.md",
                "license": None,
                "compatibility": None,
                "metadata": {},
                "allowed_tools": [],
                "source": "project",
            },
        ]
        result = build_skill_commands(skills)  # type: ignore[arg-type]
        assert len(result) == 2
        assert result[0][0] == "/skill:skill-a"
        assert result[1][0] == "/skill:skill-b"

    def test_tuple_format(self) -> None:
        """Each entry is a 3-tuple of strings."""
        skills = [
            {
                "name": "test",
                "description": "Test skill",
                "path": "/test/SKILL.md",
                "license": None,
                "compatibility": None,
                "metadata": {},
                "allowed_tools": [],
                "source": "built-in",
            }
        ]
        result = build_skill_commands(skills)  # type: ignore[arg-type]
        for entry in result:
            assert isinstance(entry, tuple)
            assert len(entry) == 3
            assert all(isinstance(s, str) for s in entry)


class TestSkillCommandParsing:
    """Test parse_skill_command() from command_registry."""

    def test_name_only(self) -> None:
        name, args = parse_skill_command("/skill:web-research")
        assert name == "web-research"
        assert args == ""

    def test_name_with_args(self) -> None:
        name, args = parse_skill_command("/skill:web-research find quantum computing")
        assert name == "web-research"
        assert args == "find quantum computing"

    def test_empty_skill_prefix(self) -> None:
        name, args = parse_skill_command("/skill:")
        assert name == ""
        assert args == ""

    def test_name_with_spaces(self) -> None:
        name, args = parse_skill_command("/skill:  web-research  some args ")
        assert name == "web-research"
        assert args == "some args"

    def test_case_normalization(self) -> None:
        name, args = parse_skill_command("/skill:Web-Research")
        assert name == "web-research"
        assert args == ""
