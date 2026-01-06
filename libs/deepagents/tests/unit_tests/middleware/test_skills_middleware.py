"""Unit tests for skills middleware with FilesystemBackend.

This module tests the skills middleware and helper functions using temporary
directories and the FilesystemBackend in normal (non-virtual) mode.
"""

from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import (
    MAX_SKILL_DESCRIPTION_LENGTH,
    MAX_SKILL_FILE_SIZE,
    SkillMetadata,
    SkillsMiddleware,
    _list_skills_from_backend,
    _parse_skill_metadata,
    _validate_skill_name,
)
from tests.unit_tests.chat_model import GenericFakeChatModel


def make_skill_content(name: str, description: str) -> str:
    """Create SKILL.md content with YAML frontmatter.

    Args:
        name: Skill name for frontmatter
        description: Skill description for frontmatter

    Returns:
        Complete SKILL.md content as string
    """
    return f"""---
name: {name}
description: {description}
---

# {name.title()} Skill

Instructions go here.
"""


# ============================================================================
# Tests for _validate_skill_name
# ============================================================================


def test_validate_skill_name_valid() -> None:
    """Test _validate_skill_name with valid skill names."""
    # Valid simple name
    is_valid, error = _validate_skill_name("web-research", "web-research")
    assert is_valid
    assert error == ""

    # Valid name with multiple segments
    is_valid, error = _validate_skill_name("my-cool-skill", "my-cool-skill")
    assert is_valid
    assert error == ""

    # Valid name with numbers
    is_valid, error = _validate_skill_name("skill-v2", "skill-v2")
    assert is_valid
    assert error == ""


def test_validate_skill_name_invalid() -> None:
    """Test _validate_skill_name with invalid skill names."""
    # Empty name
    is_valid, error = _validate_skill_name("", "test")
    assert not is_valid
    assert "required" in error

    # Name too long (> 64 chars)
    long_name = "a" * 65
    is_valid, error = _validate_skill_name(long_name, long_name)
    assert not is_valid
    assert "64 characters" in error

    # Name with uppercase
    is_valid, error = _validate_skill_name("My-Skill", "My-Skill")
    assert not is_valid
    assert "lowercase" in error

    # Name starting with hyphen
    is_valid, error = _validate_skill_name("-skill", "-skill")
    assert not is_valid
    assert "lowercase" in error

    # Name ending with hyphen
    is_valid, error = _validate_skill_name("skill-", "skill-")
    assert not is_valid
    assert "lowercase" in error

    # Name with consecutive hyphens
    is_valid, error = _validate_skill_name("my--skill", "my--skill")
    assert not is_valid
    assert "lowercase" in error

    # Name with special characters
    is_valid, error = _validate_skill_name("my_skill", "my_skill")
    assert not is_valid
    assert "lowercase" in error

    # Name doesn't match directory
    is_valid, error = _validate_skill_name("skill-a", "skill-b")
    assert not is_valid
    assert "must match directory" in error


# ============================================================================
# Tests for _parse_skill_metadata
# ============================================================================


def test_parse_skill_metadata_valid() -> None:
    """Test _parse_skill_metadata with valid YAML frontmatter."""
    content = """---
name: test-skill
description: A test skill
license: MIT
compatibility: Python 3.8+
metadata:
  author: Test Author
  version: 1.0.0
allowed-tools: read_file write_file
---

# Test Skill

Instructions here.
"""

    result = _parse_skill_metadata(content, "/skills/test-skill/SKILL.md", "test-skill", "user")

    assert result is not None
    assert result["name"] == "test-skill"
    assert result["description"] == "A test skill"
    assert result["license"] == "MIT"
    assert result["compatibility"] == "Python 3.8+"
    assert result["metadata"] == {"author": "Test Author", "version": "1.0.0"}
    assert result["allowed_tools"] == ["read_file", "write_file"]
    assert result["path"] == "/skills/test-skill/SKILL.md"
    assert result["registry"] == "user"


def test_parse_skill_metadata_minimal() -> None:
    """Test _parse_skill_metadata with minimal required fields."""
    content = """---
name: minimal-skill
description: Minimal skill
---

# Minimal Skill
"""

    result = _parse_skill_metadata(content, "/skills/minimal-skill/SKILL.md", "minimal-skill", "project")

    assert result is not None
    assert result["name"] == "minimal-skill"
    assert result["description"] == "Minimal skill"
    assert result["license"] is None
    assert result["compatibility"] is None
    assert result["metadata"] == {}
    assert result["allowed_tools"] == []
    assert result["registry"] == "project"


def test_parse_skill_metadata_no_frontmatter() -> None:
    """Test _parse_skill_metadata with missing frontmatter."""
    content = """# Test Skill

No YAML frontmatter here.
"""

    result = _parse_skill_metadata(content, "/skills/test/SKILL.md", "test", "user")
    assert result is None


def test_parse_skill_metadata_invalid_yaml() -> None:
    """Test _parse_skill_metadata with invalid YAML."""
    content = """---
name: test
description: [unclosed list
---

Content
"""

    result = _parse_skill_metadata(content, "/skills/test/SKILL.md", "test", "user")
    assert result is None


def test_parse_skill_metadata_missing_required_fields() -> None:
    """Test _parse_skill_metadata with missing required fields."""
    # Missing description
    content = """---
name: test-skill
---

Content
"""
    result = _parse_skill_metadata(content, "/skills/test/SKILL.md", "test", "user")
    assert result is None

    # Missing name
    content = """---
description: Test skill
---

Content
"""
    result = _parse_skill_metadata(content, "/skills/test/SKILL.md", "test", "user")
    assert result is None


def test_parse_skill_metadata_description_truncation() -> None:
    """Test _parse_skill_metadata truncates long descriptions."""
    long_description = "A" * (MAX_SKILL_DESCRIPTION_LENGTH + 100)
    content = f"""---
name: test-skill
description: {long_description}
---

Content
"""

    result = _parse_skill_metadata(content, "/skills/test/SKILL.md", "test-skill", "user")
    assert result is not None
    assert len(result["description"]) == MAX_SKILL_DESCRIPTION_LENGTH


def test_parse_skill_metadata_too_large() -> None:
    """Test _parse_skill_metadata rejects oversized files."""
    # Create content larger than max size
    large_content = """---
name: test-skill
description: Test
---

""" + ("X" * MAX_SKILL_FILE_SIZE)

    result = _parse_skill_metadata(large_content, "/skills/test/SKILL.md", "test-skill", "user")
    assert result is None


def test_parse_skill_metadata_empty_optional_fields() -> None:
    """Test _parse_skill_metadata handles empty optional fields correctly."""
    content = """---
name: test-skill
description: Test skill
license: ""
compatibility: ""
---

Content
"""

    result = _parse_skill_metadata(content, "/skills/test/SKILL.md", "test-skill", "user")
    assert result is not None
    assert result["license"] is None  # Empty string should become None
    assert result["compatibility"] is None  # Empty string should become None


# ============================================================================
# Tests for _list_skills_from_backend
# ============================================================================


def test_list_skills_from_backend_single_skill(tmp_path: Path) -> None:
    """Test listing a single skill from filesystem backend."""
    # Create backend with actual filesystem (no virtual mode)
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create skill using backend's upload_files interface
    skills_dir = tmp_path / "skills"
    skill_path = str(skills_dir / "my-skill" / "SKILL.md")
    skill_content = make_skill_content("my-skill", "My test skill")

    responses = backend.upload_files([(skill_path, skill_content.encode("utf-8"))])
    assert responses[0].error is None

    # List skills using the full absolute path
    skills = _list_skills_from_backend(backend, str(skills_dir), "user")

    assert skills == [
        {
            "name": "my-skill",
            "description": "My test skill",
            "path": skill_path,
            "registry": "user",
            "metadata": {},
            "license": None,
            "compatibility": None,
            "allowed_tools": [],
        }
    ]


def test_list_skills_from_backend_multiple_skills(tmp_path: Path) -> None:
    """Test listing multiple skills from filesystem backend."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create multiple skills using backend's upload_files interface
    skills_dir = tmp_path / "skills"
    skill1_path = str(skills_dir / "skill-one" / "SKILL.md")
    skill2_path = str(skills_dir / "skill-two" / "SKILL.md")
    skill3_path = str(skills_dir / "skill-three" / "SKILL.md")

    skill1_content = make_skill_content("skill-one", "First skill")
    skill2_content = make_skill_content("skill-two", "Second skill")
    skill3_content = make_skill_content("skill-three", "Third skill")

    responses = backend.upload_files(
        [
            (skill1_path, skill1_content.encode("utf-8")),
            (skill2_path, skill2_content.encode("utf-8")),
            (skill3_path, skill3_content.encode("utf-8")),
        ]
    )

    assert all(r.error is None for r in responses)

    # List skills
    skills = _list_skills_from_backend(backend, str(skills_dir), "user")

    # Should return all three skills (order may vary)
    assert len(skills) == 3
    skill_names = {s["name"] for s in skills}
    assert skill_names == {"skill-one", "skill-two", "skill-three"}


def test_list_skills_from_backend_empty_directory(tmp_path: Path) -> None:
    """Test listing skills from an empty directory."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create empty skills directory
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Should return empty list
    skills = _list_skills_from_backend(backend, str(skills_dir), "user")
    assert skills == []


def test_list_skills_from_backend_nonexistent_path(tmp_path: Path) -> None:
    """Test listing skills from a path that doesn't exist."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Try to list from non-existent directory
    skills = _list_skills_from_backend(backend, str(tmp_path / "nonexistent"), "user")
    assert skills == []


def test_list_skills_from_backend_missing_skill_md(tmp_path: Path) -> None:
    """Test that directories without SKILL.md are skipped."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create a valid skill and an invalid one (missing SKILL.md)
    skills_dir = tmp_path / "skills"
    valid_skill_path = str(skills_dir / "valid-skill" / "SKILL.md")
    invalid_dir_file = str(skills_dir / "invalid-skill" / "readme.txt")

    valid_content = make_skill_content("valid-skill", "Valid skill")

    responses = backend.upload_files(
        [
            (valid_skill_path, valid_content.encode("utf-8")),
            (invalid_dir_file, b"Not a skill file"),
        ]
    )

    # List skills - should only get the valid one
    skills = _list_skills_from_backend(backend, str(skills_dir), "user")

    assert len(skills) == 1
    assert skills[0]["name"] == "valid-skill"


def test_list_skills_from_backend_invalid_frontmatter(tmp_path: Path) -> None:
    """Test that skills with invalid YAML frontmatter are skipped."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    skills_dir = tmp_path / "skills"
    valid_skill_path = str(skills_dir / "valid-skill" / "SKILL.md")
    invalid_skill_path = str(skills_dir / "invalid-skill" / "SKILL.md")

    valid_content = make_skill_content("valid-skill", "Valid skill")
    invalid_content = """---
name: invalid-skill
description: [unclosed yaml
---

Content
"""

    responses = backend.upload_files(
        [
            (valid_skill_path, valid_content.encode("utf-8")),
            (invalid_skill_path, invalid_content.encode("utf-8")),
        ]
    )

    # Should only get the valid skill
    skills = _list_skills_from_backend(backend, str(skills_dir), "user")

    assert len(skills) == 1
    assert skills[0]["name"] == "valid-skill"


def test_list_skills_from_backend_with_helper_files(tmp_path: Path) -> None:
    """Test that skills can have additional helper files."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create a skill with helper files
    skills_dir = tmp_path / "skills"
    skill_path = str(skills_dir / "my-skill" / "SKILL.md")
    helper_path = str(skills_dir / "my-skill" / "helper.py")

    skill_content = make_skill_content("my-skill", "My test skill")
    helper_content = "def helper(): pass"

    responses = backend.upload_files(
        [
            (skill_path, skill_content.encode("utf-8")),
            (helper_path, helper_content.encode("utf-8")),
        ]
    )

    # List skills - should find the skill and not be confused by helper files
    skills = _list_skills_from_backend(backend, str(skills_dir), "user")

    assert len(skills) == 1
    assert skills[0]["name"] == "my-skill"


# ============================================================================
# Tests for SkillsMiddleware
# ============================================================================


def test_format_skills_locations_single_registry() -> None:
    """Test _format_skills_locations with a single registry."""
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore
        registries=[{"path": "/skills/user/", "name": "user"}],
    )

    result = middleware._format_skills_locations()
    assert "User Skills" in result
    assert "/skills/user/" in result
    assert "(higher priority)" in result


def test_format_skills_locations_multiple_registries() -> None:
    """Test _format_skills_locations with multiple registries."""
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore
        registries=[
            {"path": "/skills/base/", "name": "base"},
            {"path": "/skills/user/", "name": "user"},
            {"path": "/skills/project/", "name": "project"},
        ],
    )

    result = middleware._format_skills_locations()
    assert "Base Skills" in result
    assert "User Skills" in result
    assert "Project Skills" in result
    assert result.count("(higher priority)") == 1
    assert "Project Skills" in result.split("(higher priority)")[0]


def test_format_skills_list_empty() -> None:
    """Test _format_skills_list with no skills."""
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore
        registries=[
            {"path": "/skills/user/", "name": "user"},
            {"path": "/skills/project/", "name": "project"},
        ],
    )

    result = middleware._format_skills_list([])
    assert "No skills available" in result
    assert "/skills/user/" in result
    assert "/skills/project/" in result


def test_format_skills_list_single_skill() -> None:
    """Test _format_skills_list with a single skill."""
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore
        registries=[{"path": "/skills/user/", "name": "user"}],
    )

    skills: list[SkillMetadata] = [
        {
            "name": "web-research",
            "description": "Research topics on the web",
            "path": "/skills/user/web-research/SKILL.md",
            "registry": "user",
            "license": None,
            "compatibility": None,
            "metadata": {},
            "allowed_tools": [],
        }
    ]

    result = middleware._format_skills_list(skills)
    assert "web-research" in result
    assert "Research topics on the web" in result
    assert "/skills/user/web-research/SKILL.md" in result
    assert "User Skills" in result


def test_format_skills_list_multiple_skills_multiple_registries() -> None:
    """Test _format_skills_list with skills from multiple registries."""
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore
        registries=[
            {"path": "/skills/user/", "name": "user"},
            {"path": "/skills/project/", "name": "project"},
        ],
    )

    skills: list[SkillMetadata] = [
        {
            "name": "skill-a",
            "description": "User skill A",
            "path": "/skills/user/skill-a/SKILL.md",
            "registry": "user",
            "license": None,
            "compatibility": None,
            "metadata": {},
            "allowed_tools": [],
        },
        {
            "name": "skill-b",
            "description": "Project skill B",
            "path": "/skills/project/skill-b/SKILL.md",
            "registry": "project",
            "license": None,
            "compatibility": None,
            "metadata": {},
            "allowed_tools": [],
        },
        {
            "name": "skill-c",
            "description": "User skill C",
            "path": "/skills/user/skill-c/SKILL.md",
            "registry": "user",
            "license": None,
            "compatibility": None,
            "metadata": {},
            "allowed_tools": [],
        },
    ]

    result = middleware._format_skills_list(skills)

    # Check that all skills are present
    assert "skill-a" in result
    assert "skill-b" in result
    assert "skill-c" in result

    # Check registry headings
    assert "User Skills" in result
    assert "Project Skills" in result

    # Check descriptions
    assert "User skill A" in result
    assert "Project skill B" in result
    assert "User skill C" in result


def test_before_agent_loads_skills(tmp_path: Path) -> None:
    """Test that before_agent loads skills from backend."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create some skills
    skills_dir = tmp_path / "skills" / "user"
    skill1_path = str(skills_dir / "skill-one" / "SKILL.md")
    skill2_path = str(skills_dir / "skill-two" / "SKILL.md")

    skill1_content = make_skill_content("skill-one", "First skill")
    skill2_content = make_skill_content("skill-two", "Second skill")

    backend.upload_files(
        [
            (skill1_path, skill1_content.encode("utf-8")),
            (skill2_path, skill2_content.encode("utf-8")),
        ]
    )

    middleware = SkillsMiddleware(
        backend=backend,
        registries=[{"path": str(skills_dir), "name": "user"}],
    )

    # Call before_agent
    result = middleware.before_agent({}, None)  # type: ignore

    assert result is not None
    assert "skills_metadata" in result
    assert len(result["skills_metadata"]) == 2

    skill_names = {s["name"] for s in result["skills_metadata"]}
    assert skill_names == {"skill-one", "skill-two"}


def test_before_agent_skill_override(tmp_path: Path) -> None:
    """Test that skills from later registries override earlier ones."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create same skill name in two registries
    base_dir = tmp_path / "skills" / "base"
    user_dir = tmp_path / "skills" / "user"

    base_skill_path = str(base_dir / "shared-skill" / "SKILL.md")
    user_skill_path = str(user_dir / "shared-skill" / "SKILL.md")

    base_content = make_skill_content("shared-skill", "Base description")
    user_content = make_skill_content("shared-skill", "User description")

    backend.upload_files(
        [
            (base_skill_path, base_content.encode("utf-8")),
            (user_skill_path, user_content.encode("utf-8")),
        ]
    )

    middleware = SkillsMiddleware(
        backend=backend,
        registries=[
            {"path": str(base_dir), "name": "base"},
            {"path": str(user_dir), "name": "user"},
        ],
    )

    # Call before_agent
    result = middleware.before_agent({}, None)  # type: ignore

    assert result is not None
    assert len(result["skills_metadata"]) == 1

    # Should have the user version (later registry wins)
    skill = result["skills_metadata"][0]
    assert skill["name"] == "shared-skill"
    assert skill["description"] == "User description"
    assert skill["registry"] == "user"


def test_before_agent_empty_registries(tmp_path: Path) -> None:
    """Test before_agent with empty registries."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create empty directories
    (tmp_path / "skills" / "user").mkdir(parents=True)

    middleware = SkillsMiddleware(
        backend=backend,
        registries=[{"path": str(tmp_path / "skills" / "user"), "name": "user"}],
    )

    result = middleware.before_agent({}, None)  # type: ignore

    assert result is not None
    assert result["skills_metadata"] == []


def test_agent_with_skills_middleware_system_prompt(tmp_path: Path) -> None:
    """Test that skills middleware injects skills into the system prompt."""
    # Create backend and add a skill
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skills_dir = tmp_path / "skills" / "user"
    skill_path = str(skills_dir / "test-skill" / "SKILL.md")
    skill_content = make_skill_content("test-skill", "A test skill for demonstration")

    responses = backend.upload_files([(skill_path, skill_content.encode("utf-8"))])
    assert responses[0].error is None

    # Create a fake chat model that we can inspect
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="I have processed your request using the test-skill."),
            ]
        )
    )

    # Create middleware
    middleware = SkillsMiddleware(
        backend=backend,
        registries=[{"path": str(skills_dir), "name": "user"}],
    )

    # Create agent with middleware
    agent = create_agent(
        model=fake_model,
        middleware=[middleware],
    )

    # Invoke the agent
    result = agent.invoke({"messages": [HumanMessage(content="Hello, please help me.")]})

    # Verify the agent was invoked
    assert "messages" in result
    assert len(result["messages"]) > 0

    # Inspect the call history to verify system prompt was injected
    assert len(fake_model.call_history) > 0, "Model should have been called at least once"

    # Get the first call
    first_call = fake_model.call_history[0]
    messages = first_call["messages"]

    # Find the system message (should be the first message)
    system_messages = [msg for msg in messages if msg.type == "system"]
    assert len(system_messages) > 0, "Should have at least one system message"

    system_message = system_messages[0]
    system_content = system_message.content

    # Verify skills documentation is in the system prompt
    assert "Skills System" in system_content, "System prompt should contain 'Skills System' section"
    assert "test-skill" in system_content, "System prompt should mention the test-skill"
    assert "A test skill for demonstration" in system_content, "System prompt should include skill description"
    assert skill_path in system_content, "System prompt should include the skill path for reading"
