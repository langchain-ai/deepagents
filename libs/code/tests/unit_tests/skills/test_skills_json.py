"""Tests for skills commands JSON output."""

import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from deepagents_code.skills.commands import _create, _delete, _info, _list


class TestSkillsListJson:
    """Tests for _list JSON output."""

    def test_json_output_with_skills(self, tmp_path: Path) -> None:
        """JSON mode returns skill metadata array."""
        fake_skills = [
            {
                "name": "web-research",
                "description": "Search the web",
                "source": "user",
                "path": str(tmp_path / "web-research" / "SKILL.md"),
            }
        ]
        buf = StringIO()
        with (
            patch("deepagents_code.config.Credentials") as mock_settings_cls,
            patch(
                "deepagents_code.skills.commands.get_user_skills_dir",
                return_value=tmp_path / "skills",
            ),
            patch(
                "deepagents_code.skills.commands.get_project_skills_dir",
                return_value=None,
            ),
            patch(
                "deepagents_code.skills.commands.get_user_agent_skills_dir",
                return_value=tmp_path / "agent-skills",
            ),
            patch(
                "deepagents_code.skills.commands.get_project_agent_skills_dir",
                return_value=None,
            ),
            patch("deepagents_code.skills.load.list_skills", return_value=fake_skills),
            patch("sys.stdout", buf),
        ):
            settings = mock_settings_cls.from_environment.return_value
            settings.project_root = None
            _list(agent="agent", output_format="json")

        result = json.loads(buf.getvalue())
        assert result["command"] == "skills list"
        assert len(result["data"]) == 1
        assert result["data"][0]["name"] == "web-research"

    def test_json_output_empty(self, tmp_path: Path) -> None:
        """JSON mode returns empty array when no skills found."""
        buf = StringIO()
        with (
            patch("deepagents_code.config.Credentials") as mock_settings_cls,
            patch(
                "deepagents_code.skills.commands.get_user_skills_dir",
                return_value=tmp_path / "skills",
            ),
            patch(
                "deepagents_code.skills.commands.get_project_skills_dir",
                return_value=None,
            ),
            patch(
                "deepagents_code.skills.commands.get_user_agent_skills_dir",
                return_value=tmp_path / "agent-skills",
            ),
            patch(
                "deepagents_code.skills.commands.get_project_agent_skills_dir",
                return_value=None,
            ),
            patch("deepagents_code.skills.load.list_skills", return_value=[]),
            patch("sys.stdout", buf),
        ):
            settings = mock_settings_cls.from_environment.return_value
            settings.project_root = None
            _list(agent="agent", output_format="json")

        result = json.loads(buf.getvalue())
        assert result["data"] == []


class TestSkillsInfoJson:
    """Tests for _info JSON output."""

    def test_json_output(self, tmp_path: Path) -> None:
        """JSON mode returns skill metadata dict."""
        fake_skills = [
            {
                "name": "my-skill",
                "description": "Test skill",
                "source": "user",
                "path": str(tmp_path / "my-skill" / "SKILL.md"),
            }
        ]
        buf = StringIO()
        with (
            patch("deepagents_code.config.Credentials") as mock_settings_cls,
            patch(
                "deepagents_code.skills.commands.get_user_skills_dir",
                return_value=tmp_path / "skills",
            ),
            patch(
                "deepagents_code.skills.commands.get_project_skills_dir",
                return_value=None,
            ),
            patch(
                "deepagents_code.skills.commands.get_user_agent_skills_dir",
                return_value=tmp_path / "agent-skills",
            ),
            patch(
                "deepagents_code.skills.commands.get_project_agent_skills_dir",
                return_value=None,
            ),
            patch("deepagents_code.skills.load.list_skills", return_value=fake_skills),
            patch("sys.stdout", buf),
        ):
            settings = mock_settings_cls.from_environment.return_value
            settings.project_root = None
            _info("my-skill", agent="agent", output_format="json")

        result = json.loads(buf.getvalue())
        assert result["command"] == "skills info"
        assert result["data"]["name"] == "my-skill"


class TestSkillsCreateJson:
    """Tests for _create JSON output."""

    def test_json_output(self, tmp_path: Path) -> None:
        """JSON mode returns created skill metadata."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        buf = StringIO()
        with (
            patch("deepagents_code.config.Credentials") as mock_settings_cls,
            patch(
                "deepagents_code.skills.commands.ensure_user_skills_dir",
                return_value=skills_dir,
            ),
            patch("sys.stdout", buf),
        ):
            settings = mock_settings_cls.from_environment.return_value
            settings.ensure_user_skills_dir.return_value = skills_dir
            settings.project_root = None
            _create("test-skill", agent="agent", output_format="json")

        result = json.loads(buf.getvalue())
        assert result["command"] == "skills create"
        assert result["data"]["name"] == "test-skill"
        assert result["data"]["project"] is False


class TestSkillsDeleteJson:
    """Tests for _delete JSON output."""

    def test_json_output(self, tmp_path: Path) -> None:
        """JSON mode returns deletion confirmation."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_dir = skills_dir / "old-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: old-skill\n---\n")

        fake_skills = [
            {
                "name": "old-skill",
                "description": "Old skill",
                "source": "user",
                "path": str(skill_dir / "SKILL.md"),
            }
        ]
        buf = StringIO()
        with (
            patch("deepagents_code.config.Credentials") as mock_settings_cls,
            patch(
                "deepagents_code.skills.commands.get_user_skills_dir",
                return_value=skills_dir,
            ),
            patch(
                "deepagents_code.skills.commands.get_project_skills_dir",
                return_value=None,
            ),
            patch(
                "deepagents_code.skills.commands.get_user_agent_skills_dir",
                return_value=tmp_path / "agent-skills",
            ),
            patch(
                "deepagents_code.skills.commands.get_project_agent_skills_dir",
                return_value=None,
            ),
            patch("deepagents_code.skills.load.list_skills", return_value=fake_skills),
            patch("sys.stdout", buf),
        ):
            settings = mock_settings_cls.from_environment.return_value
            settings.project_root = None
            _delete("old-skill", agent="agent", force=True, output_format="json")

        result = json.loads(buf.getvalue())
        assert result["command"] == "skills delete"
        assert result["data"]["name"] == "old-skill"
        assert result["data"]["deleted"] is True
