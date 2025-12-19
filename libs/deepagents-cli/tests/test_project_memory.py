"""Tests for project-specific memory and dual agent.md loading."""

import os
from pathlib import Path

import pytest

from deepagents_cli.agent_memory import AgentMemoryMiddleware
from deepagents_cli.config import Settings
from deepagents_cli.skills import SkillsMiddleware


class TestAgentMemoryMiddleware:
    """Test dual memory loading in AgentMemoryMiddleware."""

    def test_load_user_memory_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading user agent.md when no project memory exists."""
        # Mock Path.home() to return tmp_path
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        # Create user agent directory
        agent_dir = tmp_path / ".deepagents" / "test_agent"
        agent_dir.mkdir(parents=True)
        user_md = agent_dir / "agent.md"
        user_md.write_text("User instructions")

        # Create a directory without .git to avoid project detection
        non_project_dir = tmp_path / "not-a-project"
        non_project_dir.mkdir()

        # Change to non-project directory for test
        original_cwd = Path.cwd()
        try:
            os.chdir(non_project_dir)

            # Create settings (no project detected from non_project_dir)
            test_settings = Settings.from_environment(start_path=non_project_dir)

            # Create middleware
            middleware = AgentMemoryMiddleware(settings=test_settings, assistant_id="test_agent")

            # Simulate before_agent call with no project root
            state = {}
            result = middleware.before_agent(state, None)

            assert result["user_memory"] == "User instructions"
            assert "project_memory" not in result
        finally:
            os.chdir(original_cwd)

    def test_load_both_memories(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading both user and project agent.md."""
        # Mock Path.home() to return tmp_path
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        # Create user agent directory
        agent_dir = tmp_path / ".deepagents" / "test_agent"
        agent_dir.mkdir(parents=True)
        user_md = agent_dir / "agent.md"
        user_md.write_text("User instructions")

        # Create project with .git and agent.md in .deepagents/
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        (project_root / ".deepagents").mkdir()
        project_md = project_root / ".deepagents" / "agent.md"
        project_md.write_text("Project instructions")

        original_cwd = Path.cwd()
        try:
            os.chdir(project_root)

            # Create settings (project detected from project_root)
            test_settings = Settings.from_environment(start_path=project_root)

            # Create middleware
            middleware = AgentMemoryMiddleware(settings=test_settings, assistant_id="test_agent")

            # Simulate before_agent call
            state = {}
            result = middleware.before_agent(state, None)

            assert result["user_memory"] == "User instructions"
            assert result["project_memory"] == "Project instructions"
        finally:
            os.chdir(original_cwd)

    def test_memory_not_reloaded_if_already_in_state(self, tmp_path: Path) -> None:
        """Test that memory is not reloaded if already in state."""
        agent_dir = tmp_path / ".deepagents" / "test_agent"
        agent_dir.mkdir(parents=True)

        # Create settings
        test_settings = Settings.from_environment(start_path=tmp_path)

        middleware = AgentMemoryMiddleware(settings=test_settings, assistant_id="test_agent")

        # State already has memory
        state = {"user_memory": "Existing memory", "project_memory": "Existing project"}
        result = middleware.before_agent(state, None)

        # Should return empty dict (no updates)
        assert result == {}


class TestSkillsPathResolution:
    """Test skills path resolution with per-agent structure."""

    def test_skills_middleware_paths(self, tmp_path: Path) -> None:
        """Test that skills middleware uses correct per-agent paths."""
        agent_dir = tmp_path / ".deepagents" / "test_agent"
        skills_dir = agent_dir / "skills"
        skills_dir.mkdir(parents=True)

        middleware = SkillsMiddleware(skills_dir=skills_dir, assistant_id="test_agent")

        # Check paths are correctly set
        assert middleware.skills_dir == skills_dir
        assert middleware.user_skills_display == "~/.deepagents/test_agent/skills"

    def test_skills_dir_per_agent(self, tmp_path: Path) -> None:
        """Test that different agents have separate skills directories."""
        from deepagents_cli.skills import SkillsMiddleware

        # Agent 1
        agent1_skills = tmp_path / ".deepagents" / "agent1" / "skills"
        agent1_skills.mkdir(parents=True)
        middleware1 = SkillsMiddleware(skills_dir=agent1_skills, assistant_id="agent1")

        # Agent 2
        agent2_skills = tmp_path / ".deepagents" / "agent2" / "skills"
        agent2_skills.mkdir(parents=True)
        middleware2 = SkillsMiddleware(skills_dir=agent2_skills, assistant_id="agent2")

        # Should have different paths
        assert middleware1.skills_dir != middleware2.skills_dir
        assert "agent1" in middleware1.user_skills_display
        assert "agent2" in middleware2.user_skills_display
