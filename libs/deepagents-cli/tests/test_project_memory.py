"""Tests for project-specific memory and dual agent.md loading."""

import os
from pathlib import Path

import pytest

from deepagents import MemoryMiddleware, SkillsMiddleware
from deepagents.backends.filesystem import FilesystemBackend
from deepagents_cli.config import Settings


class TestAgentMemoryMiddleware:
    """Test dual memory loading in MemoryMiddleware (SDK)."""

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

            # Create backend and middleware using SDK
            backend = FilesystemBackend(root_dir="/", virtual_mode=False)
            middleware = MemoryMiddleware(
                backend=backend,
                sources=[
                    {"path": str(test_settings.get_user_agent_md_path("test_agent")), "name": "user"}
                ],
            )

            # Simulate before_agent call with no project root
            state = {}
            result = middleware.before_agent(state, None)

            assert result["memory_contents"]["user"] == "User instructions"
            assert "project" not in result["memory_contents"]
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

            # Create backend and middleware using SDK
            backend = FilesystemBackend(root_dir="/", virtual_mode=False)
            sources = [
                {"path": str(test_settings.get_user_agent_md_path("test_agent")), "name": "user"}
            ]
            project_path = test_settings.get_project_agent_md_path()
            if project_path:
                sources.append({"path": str(project_path), "name": "project"})
            
            middleware = MemoryMiddleware(backend=backend, sources=sources)

            # Simulate before_agent call
            state = {}
            result = middleware.before_agent(state, None)

            assert result["memory_contents"]["user"] == "User instructions"
            assert result["memory_contents"]["project"] == "Project instructions"
        finally:
            os.chdir(original_cwd)

    def test_memory_not_reloaded_if_already_in_state(self, tmp_path: Path) -> None:
        """Test that memory is not reloaded if already in state."""
        agent_dir = tmp_path / ".deepagents" / "test_agent"
        agent_dir.mkdir(parents=True)

        # Create settings
        test_settings = Settings.from_environment(start_path=tmp_path)

        # Create backend and middleware using SDK
        backend = FilesystemBackend(root_dir="/", virtual_mode=False)
        middleware = MemoryMiddleware(
            backend=backend,
            sources=[
                {"path": str(test_settings.get_user_agent_md_path("test_agent")), "name": "user"}
            ],
        )

        # State already has memory
        state = {"memory_contents": {"user": "Existing memory", "project": "Existing project"}}
        result = middleware.before_agent(state, None)

        # Should return None (no updates)
        assert result is None


class TestSkillsPathResolution:
    """Test skills path resolution with per-agent structure."""

    def test_skills_middleware_paths(self, tmp_path: Path) -> None:
        """Test that skills middleware uses correct per-agent paths."""
        agent_dir = tmp_path / ".deepagents" / "test_agent"
        skills_dir = agent_dir / "skills"
        skills_dir.mkdir(parents=True)

        # Create backend and middleware using SDK
        backend = FilesystemBackend(root_dir="/", virtual_mode=False)
        middleware = SkillsMiddleware(
            backend=backend,
            registries=[{"path": str(skills_dir), "name": "user"}],
        )

        # Check that registries are correctly set
        assert len(middleware.registries) == 1
        assert middleware.registries[0]["path"] == str(skills_dir)
        assert middleware.registries[0]["name"] == "user"

    def test_skills_dir_per_agent(self, tmp_path: Path) -> None:
        """Test that different agents have separate skills directories."""
        # Agent 1
        agent1_skills = tmp_path / ".deepagents" / "agent1" / "skills"
        agent1_skills.mkdir(parents=True)
        backend1 = FilesystemBackend(root_dir="/", virtual_mode=False)
        middleware1 = SkillsMiddleware(
            backend=backend1,
            registries=[{"path": str(agent1_skills), "name": "user"}],
        )

        # Agent 2
        agent2_skills = tmp_path / ".deepagents" / "agent2" / "skills"
        agent2_skills.mkdir(parents=True)
        backend2 = FilesystemBackend(root_dir="/", virtual_mode=False)
        middleware2 = SkillsMiddleware(
            backend=backend2,
            registries=[{"path": str(agent2_skills), "name": "user"}],
        )

        # Should have different paths
        assert middleware1.registries[0]["path"] != middleware2.registries[0]["path"]
        assert "agent1" in middleware1.registries[0]["path"]
        assert "agent2" in middleware2.registries[0]["path"]
