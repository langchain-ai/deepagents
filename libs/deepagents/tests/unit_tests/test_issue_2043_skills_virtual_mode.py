"""Unit test for issue 2043: Skills not recognized with virtual_mode=True.

https://github.com/langchain-ai/deepagents/issues/2043

When using create_deep_agent with skills=["./skills/"] and FilesystemBackend
with virtual_mode=True, the skills don't show up in the system prompt.

Root cause: SkillsMiddleware calls backend.ls(source_path) where source_path is a
relative path like "./skills/" that, when resolved relative to the backend's cwd,
creates a nested path (cwd/./skills/ = cwd/skills) that doesn't exist.
"""

import os
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.graph import create_deep_agent
from tests.unit_tests.chat_model import GenericFakeChatModel


def _extract_text_from_content(content: list[dict]) -> str:
    """Extract text from message content blocks.

    Message content can be a list of content blocks with 'type' and 'text' fields.
    """
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts)


def test_issue_2043_skills_with_virtual_mode() -> None:
    """Test that skills work correctly with virtual_mode=True.

    This test reproduces issue 2043 where skills specified as a relative path
    with trailing slash (e.g., "./skills/") don't get recognized when using
    FilesystemBackend with virtual_mode=True.

    The issue is that SkillsMiddleware calls backend.ls(source_path) where
    source_path is "./skills/" and backend.cwd is the resolved "./skills/" path.
    In virtual_mode, this resolves to cwd/skills which doesn't exist.
    The fix detects when resolved_source.parent == cwd and uses "/" instead.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir) / "skills"
        skills_dir.mkdir()
        my_skill = skills_dir / "my-skill"
        my_skill.mkdir()
        (my_skill / "SKILL.md").write_text("---\nname: my-skill\ndescription: A test skill\n---\n\n# My Skill\n\nThis is a test skill.\n")

        original_cwd = Path.cwd()
        os.chdir(tmpdir)
        try:
            backend = FilesystemBackend(root_dir="./skills/", virtual_mode=True)
        finally:
            os.chdir(str(original_cwd))

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Done")]))

        agent = create_deep_agent(
            model=model,
            tools=[],
            backend=backend,
            skills=["./skills/"],
        )

        agent.invoke({"messages": [{"role": "user", "content": "Hi"}]})

        assert hasattr(model, "call_history"), "Model should have call_history"
        assert len(model.call_history) > 0, "Model should have been called"

        first_call = model.call_history[0]
        messages = first_call.get("messages", [])
        assert len(messages) > 0, "Should have messages"

        system_message = messages[0]
        content = system_message.content if hasattr(system_message, "content") else []
        content_text = _extract_text_from_content(content) if isinstance(content, list) else str(content)

        assert "my-skill" in content_text, f"Skills should be recognized. System prompt should contain 'my-skill' but got: {content_text[:500]}..."
        assert "(No skills available" not in content_text, f"Skills section should not say 'No skills available'. Content: {content_text[:500]}..."
