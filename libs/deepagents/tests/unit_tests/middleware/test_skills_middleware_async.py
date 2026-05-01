"""Async unit tests for skills middleware with FilesystemBackend.

This module contains async versions of skills middleware tests.
"""

import logging
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import FileDownloadResponse, FileInfo, LsResult
from deepagents.middleware.skills import SkillsMiddleware, _alist_skills
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


async def test_alist_skills_from_backend_single_skill(tmp_path: Path) -> None:
    """Test listing a single skill from filesystem backend (async)."""
    # Create backend with actual filesystem (no virtual mode)
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create skill using backend's upload_files interface
    skills_dir = tmp_path / "skills"
    skill_path = (skills_dir / "my-skill" / "SKILL.md").as_posix()
    skill_content = make_skill_content("my-skill", "My test skill")

    responses = backend.upload_files([(skill_path, skill_content.encode("utf-8"))])
    assert responses[0].error is None

    # List skills using the full absolute path
    skills = await _alist_skills(backend, skills_dir.as_posix())

    assert skills == [
        {
            "name": "my-skill",
            "description": "My test skill",
            "path": skill_path,
            "metadata": {},
            "license": None,
            "compatibility": None,
            "allowed_tools": [],
        }
    ]


async def test_alist_skills_from_backend_multiple_skills(tmp_path: Path) -> None:
    """Test listing multiple skills from filesystem backend (async)."""
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
    skills = await _alist_skills(backend, str(skills_dir))

    # Should return all three skills (order may vary)
    assert len(skills) == 3
    skill_names = {s["name"] for s in skills}
    assert skill_names == {"skill-one", "skill-two", "skill-three"}


async def test_alist_skills_from_backend_empty_directory(tmp_path: Path) -> None:
    """Test listing skills from an empty directory (async)."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create empty skills directory
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Should return empty list
    skills = await _alist_skills(backend, str(skills_dir))
    assert skills == []


async def test_alist_skills_from_backend_nonexistent_path(tmp_path: Path) -> None:
    """Test listing skills from a path that doesn't exist (async)."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Try to list from non-existent directory
    skills = await _alist_skills(backend, str(tmp_path / "nonexistent"))
    assert skills == []


async def test_alist_skills_logs_ls_error(caplog: pytest.LogCaptureFixture) -> None:
    """A backend listing error should not look like a normal empty source."""
    backend = MagicMock()
    backend.als = AsyncMock(return_value=LsResult(error="Cannot list '/bad': denied", entries=[]))

    with caplog.at_level(logging.WARNING):
        skills = await _alist_skills(backend, "/bad")

    assert skills == []
    assert "Cannot load skills from '/bad'" in caplog.text
    backend.adownload_files.assert_not_called()


async def test_alist_skills_loads_partial_results_with_ls_error(caplog: pytest.LogCaptureFixture) -> None:
    """A partial listing warning should not hide valid sibling skills."""
    skill_content = make_skill_content("valid-skill", "Valid skill")
    skill_dir_path = "/skills/valid-skill/"
    skill_md_path = "/skills/valid-skill/SKILL.md"

    backend = MagicMock()
    backend.als = AsyncMock(
        return_value=LsResult(error="Cannot list '/skills/loop': symlink loop", entries=[FileInfo(path=skill_dir_path, is_dir=True)])
    )
    backend.adownload_files = AsyncMock(return_value=[FileDownloadResponse(path=skill_md_path, content=skill_content.encode("utf-8"), error=None)])

    with caplog.at_level(logging.WARNING):
        skills = await _alist_skills(backend, "/skills")

    assert len(skills) == 1
    assert skills[0]["name"] == "valid-skill"
    assert "Cannot load skills from '/skills'" in caplog.text
    backend.adownload_files.assert_called_once_with([skill_md_path])


async def test_alist_skills_from_backend_missing_skill_md(tmp_path: Path) -> None:
    """Test that directories without SKILL.md are skipped (async)."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create a valid skill and an invalid one (missing SKILL.md)
    skills_dir = tmp_path / "skills"
    valid_skill_path = (skills_dir / "valid-skill" / "SKILL.md").as_posix()
    invalid_dir_file = (skills_dir / "invalid-skill" / "readme.txt").as_posix()

    valid_content = make_skill_content("valid-skill", "Valid skill")

    backend.upload_files(
        [
            (valid_skill_path, valid_content.encode("utf-8")),
            (invalid_dir_file, b"Not a skill file"),
        ]
    )

    # List skills - should only get the valid one
    skills = await _alist_skills(backend, skills_dir.as_posix())

    assert skills == [
        {
            "name": "valid-skill",
            "description": "Valid skill",
            "path": valid_skill_path,
            "metadata": {},
            "license": None,
            "compatibility": None,
            "allowed_tools": [],
        }
    ]


async def test_alist_skills_from_backend_invalid_frontmatter(tmp_path: Path) -> None:
    """Test that skills with invalid YAML frontmatter are skipped (async)."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    skills_dir = tmp_path / "skills"
    valid_skill_path = (skills_dir / "valid-skill" / "SKILL.md").as_posix()
    invalid_skill_path = (skills_dir / "invalid-skill" / "SKILL.md").as_posix()

    valid_content = make_skill_content("valid-skill", "Valid skill")
    invalid_content = """---
name: invalid-skill
description: [unclosed yaml
---

Content
"""

    backend.upload_files(
        [
            (valid_skill_path, valid_content.encode("utf-8")),
            (invalid_skill_path, invalid_content.encode("utf-8")),
        ]
    )

    # Should only get the valid skill
    skills = await _alist_skills(backend, skills_dir.as_posix())

    assert skills == [
        {
            "name": "valid-skill",
            "description": "Valid skill",
            "path": valid_skill_path,
            "metadata": {},
            "license": None,
            "compatibility": None,
            "allowed_tools": [],
        }
    ]


async def test_abefore_agent_loads_skills(tmp_path: Path) -> None:
    """Test that abefore_agent loads skills from backend."""
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

    sources = [str(skills_dir)]
    middleware = SkillsMiddleware(
        backend=backend,
        sources=sources,
    )

    # Call abefore_agent
    result = await middleware.abefore_agent({}, None, {})  # type: ignore[arg-type]

    assert result is not None
    assert "skills_metadata" in result
    assert len(result["skills_metadata"]) == 2

    skill_names = {s["name"] for s in result["skills_metadata"]}
    assert skill_names == {"skill-one", "skill-two"}


async def test_abefore_agent_skill_override(tmp_path: Path) -> None:
    """Test that skills from later sources override earlier ones (async)."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create same skill name in two sources
    base_dir = tmp_path / "skills" / "base"
    user_dir = tmp_path / "skills" / "user"

    base_skill_path = (base_dir / "shared-skill" / "SKILL.md").as_posix()
    user_skill_path = (user_dir / "shared-skill" / "SKILL.md").as_posix()

    base_content = make_skill_content("shared-skill", "Base description")
    user_content = make_skill_content("shared-skill", "User description")

    backend.upload_files(
        [
            (base_skill_path, base_content.encode("utf-8")),
            (user_skill_path, user_content.encode("utf-8")),
        ]
    )

    sources = [
        base_dir.as_posix(),
        user_dir.as_posix(),
    ]
    middleware = SkillsMiddleware(
        backend=backend,
        sources=sources,
    )

    # Call abefore_agent
    result = await middleware.abefore_agent({}, None, {})  # type: ignore[arg-type]

    assert result is not None
    assert len(result["skills_metadata"]) == 1

    # Should have the user version (later source wins)
    skill = result["skills_metadata"][0]
    assert skill == {
        "name": "shared-skill",
        "description": "User description",
        "path": user_skill_path,
        "metadata": {},
        "license": None,
        "compatibility": None,
        "allowed_tools": [],
    }


async def test_abefore_agent_empty_sources(tmp_path: Path) -> None:
    """Test abefore_agent with empty sources (async)."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create empty directories
    (tmp_path / "skills" / "user").mkdir(parents=True)

    sources = [str(tmp_path / "skills" / "user")]
    middleware = SkillsMiddleware(
        backend=backend,
        sources=sources,
    )

    result = await middleware.abefore_agent({}, None, {})  # type: ignore[arg-type]

    assert result is not None
    assert result["skills_metadata"] == []


async def test_abefore_agent_records_skill_load_errors() -> None:
    """Source load errors should be available in private middleware state."""
    backend = SimpleNamespace(
        als=AsyncMock(return_value=LsResult(error="Cannot list '/bad': denied", entries=[])),
        adownload_files=AsyncMock(),
    )
    middleware = SkillsMiddleware(backend=backend, sources=["/bad"])

    result = await middleware.abefore_agent({}, None, {})  # type: ignore[arg-type]

    assert result is not None
    assert result["skills_metadata"] == []
    assert result["skills_load_errors"] == ["Cannot load skills from '/bad': Cannot list '/bad': denied"]


async def test_abefore_agent_partial_load_across_sources() -> None:
    """A failing source must not hide skills loaded from a sibling source (async)."""
    skill_content = make_skill_content("good-skill", "Skill from the working source")
    skill_dir_path = "/good/good-skill/"
    skill_md_path = "/good/good-skill/SKILL.md"

    async def als_side_effect(path: str) -> LsResult:
        if path == "/good":
            return LsResult(entries=[FileInfo(path=skill_dir_path, is_dir=True)])
        return LsResult(error="Cannot list '/bad': denied", entries=[])

    backend = SimpleNamespace(
        als=AsyncMock(side_effect=als_side_effect),
        adownload_files=AsyncMock(return_value=[FileDownloadResponse(path=skill_md_path, content=skill_content.encode("utf-8"), error=None)]),
    )
    middleware = SkillsMiddleware(backend=backend, sources=["/good", "/bad"])

    result = await middleware.abefore_agent({}, None, {})  # type: ignore[arg-type]

    assert result is not None
    assert [skill["name"] for skill in result["skills_metadata"]] == ["good-skill"]
    assert result["skills_load_errors"] == ["Cannot load skills from '/bad': Cannot list '/bad': denied"]


async def test_abefore_agent_skips_loading_if_metadata_present(tmp_path: Path) -> None:
    """Test that abefore_agent skips loading if skills_metadata is already in state."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create a skill in the backend
    skills_dir = tmp_path / "skills" / "user"
    skill_path = str(skills_dir / "test-skill" / "SKILL.md")
    skill_content = make_skill_content("test-skill", "A test skill")

    backend.upload_files([(skill_path, skill_content.encode("utf-8"))])

    sources = [str(skills_dir)]
    middleware = SkillsMiddleware(
        backend=backend,
        sources=sources,
    )

    # State has skills_metadata already
    state_with_metadata = {"skills_metadata": []}
    result = await middleware.abefore_agent(state_with_metadata, None, {})  # type: ignore[arg-type]

    # Should return None, not load new skills
    assert result is None


async def test_agent_with_skills_middleware_multiple_sources_async(tmp_path: Path) -> None:
    """Test agent with skills from multiple sources (async)."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create skills in multiple sources
    base_dir = tmp_path / "skills" / "base"
    user_dir = tmp_path / "skills" / "user"

    base_skill_path = str(base_dir / "base-skill" / "SKILL.md")
    user_skill_path = str(user_dir / "user-skill" / "SKILL.md")

    base_content = make_skill_content("base-skill", "Base skill description")
    user_content = make_skill_content("user-skill", "User skill description")

    responses = backend.upload_files(
        [
            (base_skill_path, base_content.encode("utf-8")),
            (user_skill_path, user_content.encode("utf-8")),
        ]
    )
    assert all(r.error is None for r in responses)

    # Create fake model
    fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="I see both skills.")]))

    # Create middleware with multiple sources
    sources = [
        str(base_dir),
        str(user_dir),
    ]
    middleware = SkillsMiddleware(
        backend=backend,
        sources=sources,
    )

    # Create agent
    agent = create_agent(model=fake_model, middleware=[middleware])

    # Invoke asynchronously
    result = await agent.ainvoke({"messages": [HumanMessage(content="Help me")]})

    assert "messages" in result
    assert len(result["messages"]) > 0

    # Verify both skills are in system prompt
    first_call = fake_model.call_history[0]
    system_message = first_call["messages"][0]
    content = system_message.text

    assert "base-skill" in content
    assert "user-skill" in content


async def test_agent_with_skills_middleware_empty_sources_async(tmp_path: Path) -> None:
    """Test that agent works with empty skills sources (async)."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create empty skills directory
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Create fake model
    fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="Working without skills.")]))

    # Create middleware with empty directory
    middleware = SkillsMiddleware(backend=backend, sources=[str(skills_dir)])

    # Create agent
    agent = create_agent(model=fake_model, middleware=[middleware])

    # Invoke asynchronously
    result = await agent.ainvoke({"messages": [HumanMessage(content="Hello")]})

    assert "messages" in result
    assert len(result["messages"]) > 0

    # Verify system prompt still contains Skills System section
    first_call = fake_model.call_history[0]
    system_message = first_call["messages"][0]
    content = system_message.text

    assert "Skills System" in content
    assert "No skills available" in content


@pytest.mark.parametrize(
    ("skill_dir_path", "source_path"),
    [
        ("C:\\Users\\project\\skills\\my-skill\\", "C:\\Users\\project\\skills\\"),
        ("C:\\Users\\project\\skills\\my-skill", "C:\\Users\\project\\skills"),
        ("C:\\Users\\project\\skills\\my-skill/", "C:\\Users\\project\\skills/"),
        ("\\\\server\\share\\skills\\my-skill\\", "\\\\server\\share\\skills\\"),
    ],
    ids=["trailing-backslash", "no-trailing-sep", "mixed-separators", "unc-path"],
)
async def test_alist_skills_with_windows_style_paths(skill_dir_path: str, source_path: str) -> None:
    """Async counterpart of `test_list_skills_with_windows_style_paths`."""
    skill_content = make_skill_content("my-skill", "My test skill")
    expected_skill_md_path = str(PurePosixPath(skill_dir_path.replace("\\", "/")) / "SKILL.md")

    backend = MagicMock()
    backend.als = AsyncMock(return_value=LsResult(entries=[FileInfo(path=skill_dir_path, is_dir=True)]))
    backend.adownload_files = AsyncMock(
        return_value=[
            FileDownloadResponse(
                path=expected_skill_md_path,
                content=skill_content.encode("utf-8"),
                error=None,
            )
        ]
    )

    skills = await _alist_skills(backend, source_path)

    backend.adownload_files.assert_awaited_once_with([expected_skill_md_path])
    assert len(skills) == 1
    assert skills[0]["name"] == "my-skill"
    assert skills[0]["description"] == "My test skill"
    assert skills[0]["path"] == expected_skill_md_path
