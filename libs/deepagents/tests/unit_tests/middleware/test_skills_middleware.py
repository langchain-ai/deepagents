"""Unit tests for skills middleware with FilesystemBackend.

This module tests the skills middleware and helper functions using temporary
directories and the FilesystemBackend in normal (non-virtual) mode.
"""

from datetime import UTC, datetime
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from deepagents.graph import create_deep_agent
from deepagents.middleware.skills import (
    MAX_SKILL_DESCRIPTION_LENGTH,
    MAX_SKILL_FILE_SIZE,
    SkillMetadata,
    SkillsMiddleware,
    SkillsSource,
    _parse_skill_metadata,
    _validate_skill_name,
    list_skills,
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

    assert result == {
        "name": "test-skill",
        "description": "A test skill",
        "license": "MIT",
        "compatibility": "Python 3.8+",
        "metadata": {"author": "Test Author", "version": "1.0.0"},
        "allowed_tools": ["read_file", "write_file"],
        "path": "/skills/test-skill/SKILL.md",
        "source": "user",
    }


def test_parse_skill_metadata_minimal() -> None:
    """Test _parse_skill_metadata with minimal required fields."""
    content = """---
name: minimal-skill
description: Minimal skill
---

# Minimal Skill
"""

    result = _parse_skill_metadata(content, "/skills/minimal-skill/SKILL.md", "minimal-skill", "project")

    assert result == {
        "name": "minimal-skill",
        "description": "Minimal skill",
        "license": None,
        "compatibility": None,
        "metadata": {},
        "allowed_tools": [],
        "path": "/skills/minimal-skill/SKILL.md",
        "source": "project",
    }


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
    source = {"path": str(skills_dir), "name": "user"}
    skills = list_skills(backend, source)

    assert skills == [
        {
            "name": "my-skill",
            "description": "My test skill",
            "path": skill_path,
            "source": "user",
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
    source = {"path": str(skills_dir), "name": "user"}
    skills = list_skills(backend, source)

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
    source = {"path": str(skills_dir), "name": "user"}
    skills = list_skills(backend, source)
    assert skills == []


def test_list_skills_from_backend_nonexistent_path(tmp_path: Path) -> None:
    """Test listing skills from a path that doesn't exist."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Try to list from non-existent directory
    source = {"path": str(tmp_path / "nonexistent"), "name": "user"}
    skills = list_skills(backend, source)
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
    source = {"path": str(skills_dir), "name": "user"}
    skills = list_skills(backend, source)

    assert skills == [
        {
            "name": "valid-skill",
            "description": "Valid skill",
            "path": valid_skill_path,
            "source": "user",
            "metadata": {},
            "license": None,
            "compatibility": None,
            "allowed_tools": [],
        }
    ]


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
    source = {"path": str(skills_dir), "name": "user"}
    skills = list_skills(backend, source)

    assert skills == [
        {
            "name": "valid-skill",
            "description": "Valid skill",
            "path": valid_skill_path,
            "source": "user",
            "metadata": {},
            "license": None,
            "compatibility": None,
            "allowed_tools": [],
        }
    ]


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
    source = {"path": str(skills_dir), "name": "user"}
    skills = list_skills(backend, source)

    assert skills == [
        {
            "name": "my-skill",
            "description": "My test skill",
            "path": skill_path,
            "source": "user",
            "metadata": {},
            "license": None,
            "compatibility": None,
            "allowed_tools": [],
        }
    ]


def test_format_skills_locations_single_registry() -> None:
    """Test _format_skills_locations with a single source."""
    sources: list[SkillsSource] = [{"path": "/skills/user/", "name": "user"}]
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore
        sources=sources,
    )

    result = middleware._format_skills_locations()
    assert "User Skills" in result
    assert "/skills/user/" in result
    assert "(higher priority)" in result


def test_format_skills_locations_multiple_registries() -> None:
    """Test _format_skills_locations with multiple sources."""
    sources: list[SkillsSource] = [
        {"path": "/skills/base/", "name": "base"},
        {"path": "/skills/user/", "name": "user"},
        {"path": "/skills/project/", "name": "project"},
    ]
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore
        sources=sources,
    )

    result = middleware._format_skills_locations()
    assert "Base Skills" in result
    assert "User Skills" in result
    assert "Project Skills" in result
    assert result.count("(higher priority)") == 1
    assert "Project Skills" in result.split("(higher priority)")[0]


def test_format_skills_list_empty() -> None:
    """Test _format_skills_list with no skills."""
    sources: list[SkillsSource] = [
        {"path": "/skills/user/", "name": "user"},
        {"path": "/skills/project/", "name": "project"},
    ]
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore
        sources=sources,
    )

    result = middleware._format_skills_list([])
    assert "No skills available" in result
    assert "/skills/user/" in result
    assert "/skills/project/" in result


def test_format_skills_list_single_skill() -> None:
    """Test _format_skills_list with a single skill."""
    sources: list[SkillsSource] = [{"path": "/skills/user/", "name": "user"}]
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore
        sources=sources,
    )

    skills: list[SkillMetadata] = [
        {
            "name": "web-research",
            "description": "Research topics on the web",
            "path": "/skills/user/web-research/SKILL.md",
            "source": "user",
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
    """Test _format_skills_list with skills from multiple sources."""
    sources: list[SkillsSource] = [
        {"path": "/skills/user/", "name": "user"},
        {"path": "/skills/project/", "name": "project"},
    ]
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore
        sources=sources,
    )

    skills: list[SkillMetadata] = [
        {
            "name": "skill-a",
            "description": "User skill A",
            "path": "/skills/user/skill-a/SKILL.md",
            "source": "user",
            "license": None,
            "compatibility": None,
            "metadata": {},
            "allowed_tools": [],
        },
        {
            "name": "skill-b",
            "description": "Project skill B",
            "path": "/skills/project/skill-b/SKILL.md",
            "source": "project",
            "license": None,
            "compatibility": None,
            "metadata": {},
            "allowed_tools": [],
        },
        {
            "name": "skill-c",
            "description": "User skill C",
            "path": "/skills/user/skill-c/SKILL.md",
            "source": "user",
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

    sources: list[SkillsSource] = [{"path": str(skills_dir), "name": "user"}]
    middleware = SkillsMiddleware(
        backend=backend,
        sources=sources,
    )

    # Call before_agent
    result = middleware.before_agent({}, None)  # type: ignore

    assert result is not None
    assert "skills_metadata" in result
    assert len(result["skills_metadata"]) == 2

    skill_names = {s["name"] for s in result["skills_metadata"]}
    assert skill_names == {"skill-one", "skill-two"}


def test_before_agent_skill_override(tmp_path: Path) -> None:
    """Test that skills from later sources override earlier ones."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create same skill name in two sources
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

    sources: list[SkillsSource] = [
        {"path": str(base_dir), "name": "base"},
        {"path": str(user_dir), "name": "user"},
    ]
    middleware = SkillsMiddleware(
        backend=backend,
        sources=sources,
    )

    # Call before_agent
    result = middleware.before_agent({}, None)  # type: ignore

    assert result is not None
    assert len(result["skills_metadata"]) == 1

    # Should have the user version (later source wins)
    skill = result["skills_metadata"][0]
    assert skill == {
        "name": "shared-skill",
        "description": "User description",
        "path": user_skill_path,
        "source": "user",
        "metadata": {},
        "license": None,
        "compatibility": None,
        "allowed_tools": [],
    }


def test_before_agent_empty_registries(tmp_path: Path) -> None:
    """Test before_agent with empty sources."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create empty directories
    (tmp_path / "skills" / "user").mkdir(parents=True)

    sources: list[SkillsSource] = [{"path": str(tmp_path / "skills" / "user"), "name": "user"}]
    middleware = SkillsMiddleware(
        backend=backend,
        sources=sources,
    )

    result = middleware.before_agent({}, None)  # type: ignore

    assert result is not None
    assert result["skills_metadata"] == []


def test_agent_with_skills_middleware_system_prompt(tmp_path: Path) -> None:
    """Test that skills middleware injects skills into the system prompt."""
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
    sources: list[SkillsSource] = [{"path": str(skills_dir), "name": "user"}]
    middleware = SkillsMiddleware(
        backend=backend,
        sources=sources,
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

    system_message = messages[0]
    assert system_message.type == "system", "First message should be system prompt"
    content = system_message.text
    assert "Skills System" in content, "System prompt should contain 'Skills System' section"
    assert "test-skill" in content, "System prompt should mention the skill name"


def test_skills_middleware_with_state_backend_factory() -> None:
    """Test that SkillsMiddleware can be initialized with StateBackend factory."""
    # Test that the middleware accepts StateBackend as a factory function
    # This is the recommended pattern for StateBackend since it needs runtime context
    sources: list[SkillsSource] = [{"path": "/skills/user", "name": "user"}]
    middleware = SkillsMiddleware(
        backend=lambda rt: StateBackend(rt),
        sources=sources,
    )

    # Verify the middleware was created successfully
    assert middleware is not None
    assert callable(middleware._backend)
    assert len(middleware.sources) == 1
    assert middleware.sources[0]["name"] == "user"

    runtime = ToolRuntime(
        state={"messages": [], "files": {}},
        context=None,
        tool_call_id="test",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )

    backend = middleware._get_backend(runtime)
    assert isinstance(backend, StateBackend)
    assert backend.runtime == runtime


def test_skills_middleware_with_store_backend_factory() -> None:
    """Test that SkillsMiddleware can be initialized with StoreBackend factory."""
    # Test that the middleware accepts StoreBackend as a factory function
    # This is the recommended pattern for StoreBackend since it needs runtime context with store
    sources: list[SkillsSource] = [{"path": "/skills/user", "name": "user"}]
    middleware = SkillsMiddleware(
        backend=lambda rt: StoreBackend(rt),
        sources=sources,
    )

    # Verify the middleware was created successfully
    assert middleware is not None
    assert callable(middleware._backend)
    assert len(middleware.sources) == 1
    assert middleware.sources[0]["name"] == "user"

    # Test that we can create a runtime with store and get a backend from the factory
    store = InMemoryStore()
    runtime = ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="test",
        store=store,
        stream_writer=lambda _: None,
        config={},
    )

    backend = middleware._get_backend(runtime)
    assert isinstance(backend, StoreBackend)
    assert backend.runtime == runtime


async def test_agent_with_skills_middleware_async(tmp_path: Path) -> None:
    """Test that skills middleware works with async agent invocation."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skills_dir = tmp_path / "skills" / "user"
    skill_path = str(skills_dir / "async-skill" / "SKILL.md")
    skill_content = make_skill_content("async-skill", "A test skill for async testing")

    responses = backend.upload_files([(skill_path, skill_content.encode("utf-8"))])
    assert responses[0].error is None

    # Create a fake chat model
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="I have processed your async request using the async-skill."),
            ]
        )
    )

    # Create middleware
    sources: list[SkillsSource] = [{"path": str(skills_dir), "name": "user"}]
    middleware = SkillsMiddleware(
        backend=backend,
        sources=sources,
    )

    # Create agent with middleware
    agent = create_agent(
        model=fake_model,
        middleware=[middleware],
    )

    # Invoke the agent asynchronously
    result = await agent.ainvoke({"messages": [HumanMessage(content="Hello, please help me.")]})

    # Verify the agent was invoked
    assert "messages" in result
    assert len(result["messages"]) > 0

    # Verify skills_metadata is NOT in final state (it's a PrivateStateAttr)
    assert "skills_metadata" not in result, "skills_metadata should be private and not in final state"

    # Inspect the call history to verify system prompt was injected
    assert len(fake_model.call_history) > 0, "Model should have been called at least once"

    # Get the first call
    first_call = fake_model.call_history[0]
    messages = first_call["messages"]

    system_message = messages[0]
    assert system_message.type == "system", "First message should be system prompt"
    content = system_message.text
    assert "Skills System" in content, "System prompt should contain 'Skills System' section"
    assert "async-skill" in content, "System prompt should mention the skill name"


def test_agent_with_skills_middleware_multiple_registries_override(tmp_path: Path) -> None:
    """Test skills middleware with multiple sources where later sources override earlier ones."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create same-named skill in two sources with different descriptions
    base_dir = tmp_path / "skills" / "base"
    user_dir = tmp_path / "skills" / "user"

    base_skill_path = str(base_dir / "shared-skill" / "SKILL.md")
    user_skill_path = str(user_dir / "shared-skill" / "SKILL.md")

    base_content = make_skill_content("shared-skill", "Base registry description")
    user_content = make_skill_content("shared-skill", "User registry description - should win")

    responses = backend.upload_files(
        [
            (base_skill_path, base_content.encode("utf-8")),
            (user_skill_path, user_content.encode("utf-8")),
        ]
    )
    assert all(r.error is None for r in responses)

    # Create a fake chat model
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="I have processed your request."),
            ]
        )
    )

    # Create middleware with multiple sources - user should override base
    sources: list[SkillsSource] = [
        {"path": str(base_dir), "name": "base"},
        {"path": str(user_dir), "name": "user"},
    ]
    middleware = SkillsMiddleware(
        backend=backend,
        sources=sources,
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

    # Verify skills_metadata is NOT in final state (it's a PrivateStateAttr)
    assert "skills_metadata" not in result, "skills_metadata should be private and not in final state"

    # Inspect the call history to verify system prompt was injected with USER version
    assert len(fake_model.call_history) > 0, "Model should have been called at least once"

    # Get the first call
    first_call = fake_model.call_history[0]
    messages = first_call["messages"]

    system_message = messages[0]
    assert system_message.type == "system", "First message should be system prompt"
    content = system_message.text
    assert "Skills System" in content, "System prompt should contain 'Skills System' section"
    assert "shared-skill" in content, "System prompt should mention the skill name"
    assert "User registry description - should win" in content, "Should use user source description"
    assert "Base registry description" not in content, "Should not contain base source description"


def test_before_agent_skips_loading_if_metadata_present(tmp_path: Path) -> None:
    """Test that before_agent skips loading if skills_metadata is already in state."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create a skill in the backend
    skills_dir = tmp_path / "skills" / "user"
    skill_path = str(skills_dir / "test-skill" / "SKILL.md")
    skill_content = make_skill_content("test-skill", "A test skill")

    backend.upload_files([(skill_path, skill_content.encode("utf-8"))])

    sources: list[SkillsSource] = [{"path": str(skills_dir), "name": "user"}]
    middleware = SkillsMiddleware(
        backend=backend,
        sources=sources,
    )

    # Case 1: State has skills_metadata with some skills
    existing_metadata: list[SkillMetadata] = [
        {
            "name": "existing-skill",
            "description": "An existing skill",
            "path": "/some/path/SKILL.md",
            "source": "user",
            "metadata": {},
            "license": None,
            "compatibility": None,
            "allowed_tools": [],
        }
    ]
    state_with_metadata = {"skills_metadata": existing_metadata}
    result = middleware.before_agent(state_with_metadata, None)  # type: ignore

    # Should return None, not load new skills
    assert result is None

    # Case 2: State has empty list for skills_metadata
    state_with_empty_list = {"skills_metadata": []}
    result = middleware.before_agent(state_with_empty_list, None)  # type: ignore

    # Should still return None and not reload
    assert result is None

    # Case 3: State does NOT have skills_metadata key
    state_without_metadata = {}
    result = middleware.before_agent(state_without_metadata, None)  # type: ignore

    # Should load skills and return update
    assert result is not None
    assert "skills_metadata" in result
    assert len(result["skills_metadata"]) == 1
    assert result["skills_metadata"][0]["name"] == "test-skill"


def test_create_deep_agent_with_skills_and_filesystem_backend(tmp_path: Path) -> None:
    """Test end-to-end: create_deep_agent with skills parameter and FilesystemBackend."""
    # Create skill on filesystem
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skills_dir = tmp_path / "skills" / "user"
    skill_path = str(skills_dir / "test-skill" / "SKILL.md")
    skill_content = make_skill_content("test-skill", "A test skill for deep agents")

    backend.upload_files([(skill_path, skill_content.encode("utf-8"))])

    # Create agent with skills parameter and FilesystemBackend
    agent = create_deep_agent(
        backend=backend,
        skills=[{"path": str(skills_dir), "name": "user"}],
        model=GenericFakeChatModel(messages=iter([AIMessage(content="I see the test-skill in the system prompt.")])),
    )

    # Invoke agent
    result = agent.invoke({"messages": [HumanMessage(content="What skills are available?")]})

    # Verify invocation succeeded
    assert "messages" in result
    assert len(result["messages"]) > 0


def test_create_deep_agent_with_skills_empty_directory(tmp_path: Path) -> None:
    """Test that skills work gracefully when no skills are found (empty directory)."""
    # Create empty skills directory
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skills_dir = tmp_path / "skills" / "user"
    skills_dir.mkdir(parents=True)

    # Create agent with skills parameter but empty directory
    agent = create_deep_agent(
        backend=backend,
        skills=[{"path": str(skills_dir), "name": "user"}],
        model=GenericFakeChatModel(messages=iter([AIMessage(content="No skills found, but that's okay.")])),
    )

    # Invoke agent
    result = agent.invoke({"messages": [HumanMessage(content="What skills are available?")]})

    # Verify invocation succeeded even without skills
    assert "messages" in result
    assert len(result["messages"]) > 0


def test_create_deep_agent_with_skills_default_backend() -> None:
    """Test create_deep_agent with skills parameter using default backend (no backend specified).

    When no backend is specified, StateBackend is used by tools. Since SkillsMiddleware
    receives None for backend (no explicit backend provided), it logs a warning and
    returns empty skills. However, if we pass files via invoke(), tools can still
    access those files via StateBackend.
    """
    checkpointer = InMemorySaver()
    agent = create_deep_agent(
        skills=[{"path": "/skills/user", "name": "user"}],
        model=GenericFakeChatModel(messages=iter([AIMessage(content="Working with default backend.")])),
        checkpointer=checkpointer,
    )

    # Create skill content with proper
    skill_content = make_skill_content("test-skill", "A test skill for default backend")
    timestamp = datetime.now(UTC).isoformat()

    # Prepare files dict with FileData format (for StateBackend tools)
    # Note: SkillsMiddleware will still get backend=None, so it won't load these
    # But this demonstrates the proper format for StateBackend
    skill_files = {
        "/skills/user/test-skill/SKILL.md": {
            "content": skill_content.split("\n"),
            "created_at": timestamp,
            "modified_at": timestamp,
        }
    }

    config: RunnableConfig = {"configurable": {"thread_id": "123"}}

    # Invoke agent with files parameter
    # Skills won't be loaded (backend=None for SkillsMiddleware), but tools can access files
    result = agent.invoke(
        {
            "messages": [HumanMessage(content="What skills are available?")],
            "files": skill_files,
        },
        config,
    )

    assert len(result["messages"]) > 0

    checkpoint = agent.checkpointer.get(config)
    assert "/skills/user/test-skill/SKILL.md" in checkpoint["channel_values"]["files"]
    assert checkpoint["channel_values"]["skills_metadata"] == [
        {
            "allowed_tools": [],
            "compatibility": None,
            "description": "A test skill for default backend",
            "license": None,
            "metadata": {},
            "name": "test-skill",
            "path": "/skills/user/test-skill/SKILL.md",
            "source": "user",
        },
    ]
