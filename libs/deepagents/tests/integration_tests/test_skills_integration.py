"""Integration tests for skills middleware with the deep agent.

These tests verify that the SkillsMiddleware properly integrates with
the agent system, including system prompt injection and state management.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import SkillsMiddleware, SkillsState


def make_skill_content(name: str, description: str, instructions: str = "") -> str:
    """Create SKILL.md content with YAML frontmatter."""
    return f"""---
name: {name}
description: {description}
---

# {name.replace("-", " ").title()} Skill

{instructions or "Follow the skill instructions."}
"""


class TestSkillsIntegration:
    """Integration tests for skills middleware."""

    def test_skills_middleware_modifies_system_prompt(self, tmp_path: Path) -> None:
        """Test that skills are injected into system prompt via wrap_model_call."""
        # Setup skills
        skills_dir = tmp_path / "skills" / "user"
        skills_dir.mkdir(parents=True)

        skill_path = skills_dir / "test-skill"
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text(make_skill_content("test-skill", "A test skill for verification"))

        # Create middleware
        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        middleware = SkillsMiddleware(
            backend=backend,
            registries=[{"path": str(skills_dir), "name": "user"}],
        )

        # Load skills via before_agent
        state_update = middleware.before_agent({}, None)  # type: ignore
        assert state_update is not None
        assert len(state_update["skills_metadata"]) == 1

        # Create a mock request with the skills in state
        mock_request = MagicMock()
        mock_request.state = {"skills_metadata": state_update["skills_metadata"]}
        mock_request.system_prompt = "You are a helpful assistant."

        # Track what system prompt is passed to handler
        captured_request = None

        def mock_handler(req):
            nonlocal captured_request
            captured_request = req
            return MagicMock()

        mock_request.override = lambda **kwargs: type("MockRequest", (), {**vars(mock_request), **kwargs, "state": mock_request.state})()

        # Call wrap_model_call
        middleware.wrap_model_call(mock_request, mock_handler)

        # Verify system prompt was modified
        assert captured_request is not None
        assert "test-skill" in captured_request.system_prompt
        assert "A test skill for verification" in captured_request.system_prompt
        assert "Skills System" in captured_request.system_prompt

    def test_skills_state_schema_is_correct(self) -> None:
        """Test that SkillsState has the expected structure."""
        # Verify the state schema includes skills_metadata
        from typing import get_type_hints

        hints = get_type_hints(SkillsState, include_extras=True)
        assert "skills_metadata" in hints

    def test_multiple_registries_with_override(self, tmp_path: Path) -> None:
        """Test that later registries override earlier ones for same skill name."""
        # Create base and user registries
        base_dir = tmp_path / "skills" / "base"
        user_dir = tmp_path / "skills" / "user"
        base_dir.mkdir(parents=True)
        user_dir.mkdir(parents=True)

        # Same skill name in both
        for dir_path, desc in [(base_dir, "Base version"), (user_dir, "User version")]:
            skill_path = dir_path / "shared-skill"
            skill_path.mkdir()
            (skill_path / "SKILL.md").write_text(make_skill_content("shared-skill", desc))

        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        middleware = SkillsMiddleware(
            backend=backend,
            registries=[
                {"path": str(base_dir), "name": "base"},
                {"path": str(user_dir), "name": "user"},  # Higher priority
            ],
        )

        state_update = middleware.before_agent({}, None)  # type: ignore
        assert state_update is not None

        # Should only have one skill (user version overrides base)
        skills = state_update["skills_metadata"]
        assert len(skills) == 1
        assert skills[0]["description"] == "User version"
        assert skills[0]["registry"] == "user"

    def test_empty_skills_still_injects_prompt(self, tmp_path: Path) -> None:
        """Test that system prompt is injected even with no skills."""
        skills_dir = tmp_path / "skills" / "user"
        skills_dir.mkdir(parents=True)

        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        middleware = SkillsMiddleware(
            backend=backend,
            registries=[{"path": str(skills_dir), "name": "user"}],
        )

        state_update = middleware.before_agent({}, None)  # type: ignore
        assert state_update is not None
        assert state_update["skills_metadata"] == []

        # Check that format_skills_list handles empty case
        skills_list = middleware._format_skills_list([])
        assert "No skills available" in skills_list

    def test_skill_with_allowed_tools(self, tmp_path: Path) -> None:
        """Test that allowed_tools are properly parsed."""
        skills_dir = tmp_path / "skills" / "user"
        skills_dir.mkdir(parents=True)

        skill_path = skills_dir / "tool-skill"
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text("""---
name: tool-skill
description: A skill with allowed tools
allowed-tools: read_file write_file execute
---

# Tool Skill

Use these tools wisely.
""")

        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        middleware = SkillsMiddleware(
            backend=backend,
            registries=[{"path": str(skills_dir), "name": "user"}],
        )

        state_update = middleware.before_agent({}, None)  # type: ignore
        assert state_update is not None

        skill = state_update["skills_metadata"][0]
        assert skill["allowed_tools"] == ["read_file", "write_file", "execute"]


class TestSkillsWithAgent:
    """Tests that require full agent setup (marked for manual/CI runs)."""

    @pytest.mark.skip(reason="Requires API key - run manually")
    def test_agent_sees_skills_in_prompt(self, tmp_path: Path) -> None:
        """Test that an actual agent can see and respond to skills."""
        from deepagents import create_deep_agent

        # Setup skills
        skills_dir = tmp_path / "skills" / "user"
        skills_dir.mkdir(parents=True)

        skill_path = skills_dir / "greeting-skill"
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text(
            make_skill_content("greeting-skill", "Skill for greeting users warmly", "When greeting users, always say 'Hello friend!' first.")
        )

        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        skills_middleware = SkillsMiddleware(
            backend=backend,
            registries=[{"path": str(skills_dir), "name": "user"}],
        )

        agent = create_deep_agent(
            middleware=[skills_middleware],
            system_prompt="You are a helpful assistant.",
        )

        # This would actually invoke the agent - requires API key
        # result = agent.invoke({"messages": [{"role": "user", "content": "What skills do you have?"}]})
        # assert "greeting-skill" in str(result)
