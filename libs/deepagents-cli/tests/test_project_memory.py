"""Tests for project-specific memory and dual agent.md loading."""

from pathlib import Path

import pytest

from deepagents_cli.agent_memory import AgentMemoryMiddleware
from deepagents_cli.config import _find_project_agent_md, _find_project_root


class TestAgentMemoryMiddleware:
    """Test dual memory loading in AgentMemoryMiddleware."""

    def test_load_user_memory_only(self, tmp_path):
        """Test loading user agent.md when no project memory exists."""
        import os

        # Create user agent directory
        agent_dir = tmp_path / ".deepagents" / "test-agent"
        agent_dir.mkdir(parents=True)
        user_md = agent_dir / "agent.md"
        user_md.write_text("User instructions")

        # Create a directory without .git to avoid project detection
        non_project_dir = tmp_path / "not-a-project"
        non_project_dir.mkdir()

        # Change to non-project directory for test
        original_cwd = os.getcwd()
        try:
            os.chdir(non_project_dir)

            # Create middleware
            middleware = AgentMemoryMiddleware(agent_dir=agent_dir, assistant_id="test-agent")

            # Simulate before_agent call with no project root
            state = {}
            result = middleware.before_agent(state, None)

            assert result["user_memory"] == "User instructions"
            assert result["project_memory"] == ""
        finally:
            os.chdir(original_cwd)

    def test_load_both_memories(self, tmp_path):
        """Test loading both user and project agent.md."""
        # Create user agent directory
        agent_dir = tmp_path / ".deepagents" / "test-agent"
        agent_dir.mkdir(parents=True)
        user_md = agent_dir / "agent.md"
        user_md.write_text("User instructions")

        # Create project with .git and agent.md
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        project_md = project_root / "agent.md"
        project_md.write_text("Project instructions")

        # Change to project directory for detection
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(project_root)

            # Create middleware
            middleware = AgentMemoryMiddleware(agent_dir=agent_dir, assistant_id="test-agent")

            # Simulate before_agent call
            state = {}
            result = middleware.before_agent(state, None)

            assert result["user_memory"] == "User instructions"
            assert result["project_memory"] == "Project instructions"
        finally:
            os.chdir(original_cwd)

    def test_memory_not_reloaded_if_already_in_state(self, tmp_path):
        """Test that memory is not reloaded if already in state."""
        agent_dir = tmp_path / ".deepagents" / "test-agent"
        agent_dir.mkdir(parents=True)

        middleware = AgentMemoryMiddleware(agent_dir=agent_dir, assistant_id="test-agent")

        # State already has memory
        state = {"user_memory": "Existing memory", "project_memory": "Existing project"}
        result = middleware.before_agent(state, None)

        # Should return empty dict (no updates)
        assert result == {}


class TestRealAgentMemory:
    """Test with real agent.md files (may not exist on all systems)."""

    def test_load_real_user_memory(self):
        """Test loading actual user agent.md if it exists."""
        agent_dir = Path.home() / ".deepagents" / "agent"
        agent_md = agent_dir / "agent.md"

        if not agent_md.exists():
            pytest.skip(
                "⚠️  User agent.md not found. "
                f"Create {agent_md} to enable this test. "
                "This is expected if you haven't set up your user agent yet."
            )

        middleware = AgentMemoryMiddleware(agent_dir=agent_dir, assistant_id="agent")
        state = {}
        result = middleware.before_agent(state, None)

        assert "user_memory" in result
        assert len(result["user_memory"]) > 0
        print(f"\n✓ Loaded user agent.md ({len(result['user_memory'])} chars)")

    def test_detect_current_project(self):
        """Test detecting project root from current directory."""
        project_root = _find_project_root()

        if project_root is None:
            pytest.skip(
                "⚠️  Not in a git project. "
                "Run this test from inside a git repository to test project detection."
            )

        assert project_root.exists()
        assert (project_root / ".git").exists()
        print(f"\n✓ Detected project root: {project_root}")

        # Check for project agent.md
        project_md_paths = _find_project_agent_md(project_root)
        if project_md_paths:
            for path in project_md_paths:
                print(f"✓ Found project agent.md: {path}")
        else:
            print(
                f"ℹ️  No project agent.md found. "
                f"Create {project_root}/.deepagents/agent.md or {project_root}/agent.md "
                f"to enable project-specific configuration."
            )


class TestSkillsPathResolution:
    """Test skills path resolution with per-agent structure."""

    def test_skills_middleware_paths(self, tmp_path):
        """Test that skills middleware uses correct per-agent paths."""
        from deepagents_cli.skills import SkillsMiddleware

        agent_dir = tmp_path / ".deepagents" / "test-agent"
        skills_dir = agent_dir / "skills"
        skills_dir.mkdir(parents=True)

        middleware = SkillsMiddleware(skills_dir=skills_dir, assistant_id="test-agent")

        # Check paths are correctly set
        assert middleware.skills_dir == skills_dir
        assert middleware.skills_dir_display == "~/.deepagents/test-agent/skills"
        assert middleware.skills_dir_absolute == str(skills_dir)

    def test_skills_dir_per_agent(self, tmp_path):
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
        assert "agent1" in middleware1.skills_dir_display
        assert "agent2" in middleware2.skills_dir_display
