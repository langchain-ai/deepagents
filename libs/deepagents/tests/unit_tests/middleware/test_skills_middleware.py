"""Unit tests for skills middleware with FilesystemBackend.

This module tests the skills middleware and helper functions using temporary
directories and the FilesystemBackend in normal (non-virtual) mode.
"""

import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import CONF
from langgraph.runtime import CONFIG_KEY_RUNTIME, Runtime, ServerInfo

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
from langgraph.store.memory import InMemoryStore

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import FileDownloadResponse, FileInfo, LsResult
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from deepagents.graph import create_deep_agent
from deepagents.middleware.skills import (
    MAX_SKILL_COMPATIBILITY_LENGTH,
    MAX_SKILL_DESCRIPTION_LENGTH,
    MAX_SKILL_FILE_SIZE,
    SkillMetadata,
    SkillsMiddleware,
    _format_skill_annotations,
    _list_skills,
    _parse_skill_metadata,
    _skill_metadata_from_response,
    _validate_metadata,
    _validate_module_path,
    _validate_skill_name,
)
from tests.unit_tests.chat_model import GenericFakeChatModel


def _assistant_id_namespace(rt: Runtime) -> tuple[str, ...]:
    """Namespace factory: scope by assistant_id when running under LangGraph server."""
    assistant_id = rt.server_info.assistant_id if rt.server_info else None
    if assistant_id:
        return (assistant_id, "filesystem")
    return ("filesystem",)


@contextmanager
def _runtime_context(assistant_id: str | None = None):
    """Set a LangGraph Runtime in the current context so ``get_runtime()`` resolves."""
    server_info = ServerInfo(assistant_id=assistant_id, graph_id="test") if assistant_id is not None else None
    runtime = Runtime(server_info=server_info)
    token = var_child_runnable_config.set({CONF: {CONFIG_KEY_RUNTIME: runtime}})
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


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

    result = _parse_skill_metadata(content, "/skills/test-skill/SKILL.md", "test-skill")

    assert result == {
        "name": "test-skill",
        "description": "A test skill",
        "license": "MIT",
        "compatibility": "Python 3.8+",
        "metadata": {"author": "Test Author", "version": "1.0.0"},
        "allowed_tools": ["read_file", "write_file"],
        "path": "/skills/test-skill/SKILL.md",
    }


def test_parse_skill_metadata_minimal() -> None:
    """Test _parse_skill_metadata with minimal required fields."""
    content = """---
name: minimal-skill
description: Minimal skill
---

# Minimal Skill
"""

    result = _parse_skill_metadata(content, "/skills/minimal-skill/SKILL.md", "minimal-skill")

    assert result == {
        "name": "minimal-skill",
        "description": "Minimal skill",
        "license": None,
        "compatibility": None,
        "metadata": {},
        "allowed_tools": [],
        "path": "/skills/minimal-skill/SKILL.md",
    }


def test_parse_skill_metadata_no_frontmatter() -> None:
    """Test _parse_skill_metadata with missing frontmatter."""
    content = """# Test Skill

No YAML frontmatter here.
"""

    result = _parse_skill_metadata(content, "/skills/test/SKILL.md", "test")
    assert result is None


def test_validate_module_path_absent() -> None:
    """Missing key returns None — the vast majority of skills have no module."""
    assert _validate_module_path(None, "/skills/x/SKILL.md") is None


def test_validate_module_path_valid_bare() -> None:
    """A bare filename with a supported extension passes through unchanged."""
    assert _validate_module_path("index.ts", "/skills/x/SKILL.md") == "index.ts"


def test_validate_module_path_strips_dot_slash() -> None:
    """./index.ts → index.ts so the stored path matches how the loader keys files."""
    assert _validate_module_path("./index.ts", "/skills/x/SKILL.md") == "index.ts"


def test_validate_module_path_nested() -> None:
    """Subdirectory paths are fine as long as they stay inside the skill dir."""
    assert _validate_module_path("lib/entry.js", "/skills/x/SKILL.md") == "lib/entry.js"


def test_validate_module_path_all_supported_extensions() -> None:
    """Every quickjs-rs-accepted extension is accepted here too."""
    for ext in ("js", "mjs", "cjs", "ts", "mts", "cts", "jsx", "tsx"):
        path = f"index.{ext}"
        assert _validate_module_path(path, "/skills/x/SKILL.md") == path


def test_validate_module_path_rejects_non_string(caplog: pytest.LogCaptureFixture) -> None:
    """Non-string values log a warning and return None — don't crash the parse."""
    caplog.set_level(logging.WARNING)
    assert _validate_module_path(42, "/skills/x/SKILL.md") is None
    assert "non-string 'module'" in caplog.text


def test_validate_module_path_rejects_empty_string() -> None:
    """An empty / whitespace-only value is equivalent to absent."""
    assert _validate_module_path("", "/skills/x/SKILL.md") is None
    assert _validate_module_path("   ", "/skills/x/SKILL.md") is None


def test_validate_module_path_rejects_absolute(caplog: pytest.LogCaptureFixture) -> None:
    """Absolute paths could reach outside the skill dir — reject with a warning."""
    caplog.set_level(logging.WARNING)
    assert _validate_module_path("/etc/passwd", "/skills/x/SKILL.md") is None
    assert "absolute" in caplog.text


def test_validate_module_path_rejects_parent_traversal(caplog: pytest.LogCaptureFixture) -> None:
    """Any form of `..` traversal is rejected so skills can't read each other's code."""
    caplog.set_level(logging.WARNING)
    for bad in ("../other/index.js", "./../other/index.js", "lib/../../outside.js", ".."):
        caplog.clear()
        assert _validate_module_path(bad, "/skills/x/SKILL.md") is None
        assert "escapes" in caplog.text


def test_validate_module_path_rejects_unknown_extension(caplog: pytest.LogCaptureFixture) -> None:
    """Only JS/TS extensions quickjs-rs understands are valid entrypoints."""
    caplog.set_level(logging.WARNING)
    assert _validate_module_path("index.py", "/skills/x/SKILL.md") is None
    assert "extension" in caplog.text


def test_parse_skill_metadata_with_module() -> None:
    """End-to-end: a `module` frontmatter key lands on the returned metadata."""
    content = """---
name: pdf-extract
description: Parse PDFs
module: ./index.ts
---

# PDF extract
"""
    result = _parse_skill_metadata(content, "/skills/user/pdf-extract/SKILL.md", "pdf-extract")
    assert result is not None
    assert result["module"] == "index.ts"


def test_parse_skill_metadata_with_invalid_module_degrades_gracefully() -> None:
    """An invalid `module` value must not drop the skill.

    The prose is still useful; we just drop the module surface.
    """
    content = """---
name: bad-module
description: has a bad module path
module: /etc/passwd
---

# Bad module
"""
    result = _parse_skill_metadata(content, "/skills/user/bad-module/SKILL.md", "bad-module")
    assert result is not None
    assert "module" not in result
    assert result["name"] == "bad-module"


def test_parse_skill_metadata_invalid_yaml() -> None:
    """Test _parse_skill_metadata with invalid YAML."""
    content = """---
name: test
description: [unclosed list
---

Content
"""

    result = _parse_skill_metadata(content, "/skills/test/SKILL.md", "test")
    assert result is None


def test_parse_skill_metadata_missing_required_fields() -> None:
    """Test _parse_skill_metadata with missing required fields."""
    # Missing description
    content = """---
name: test-skill
---

Content
"""
    result = _parse_skill_metadata(content, "/skills/test/SKILL.md", "test")
    assert result is None

    # Missing name
    content = """---
description: Test skill
---

Content
"""
    result = _parse_skill_metadata(content, "/skills/test/SKILL.md", "test")
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

    result = _parse_skill_metadata(content, "/skills/test/SKILL.md", "test-skill")
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

    result = _parse_skill_metadata(large_content, "/skills/test/SKILL.md", "test-skill")
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

    result = _parse_skill_metadata(content, "/skills/test/SKILL.md", "test-skill")
    assert result is not None
    assert result["license"] is None  # Empty string should become None
    assert result["compatibility"] is None  # Empty string should become None


def test_parse_skill_metadata_compatibility_max_length() -> None:
    """Test _parse_skill_metadata truncates compatibility exceeding 500 chars.

    Per Agent Skills spec, compatibility field must be max 500 characters.
    """
    long_compat = "x" * 600
    content = f"""---
name: test-skill
description: A test skill
compatibility: {long_compat}
---

Content
"""

    result = _parse_skill_metadata(content, "/skills/test-skill/SKILL.md", "test-skill")
    assert result is not None
    assert result["compatibility"] is not None
    assert len(result["compatibility"]) == MAX_SKILL_COMPATIBILITY_LENGTH


def test_parse_skill_metadata_whitespace_only_description() -> None:
    """Test _parse_skill_metadata rejects whitespace-only description.

    A description of just spaces becomes empty after `str(...).strip()` and is
    then rejected by the `if not description` check.
    """
    content = """---
name: test-skill
description: "   "
---

Content
"""

    result = _parse_skill_metadata(content, "/skills/test-skill/SKILL.md", "test-skill")
    assert result is None


def test_parse_skill_metadata_allowed_tools_multiple_spaces() -> None:
    """Test _parse_skill_metadata handles multiple consecutive spaces in allowed-tools."""
    content = """---
name: test-skill
description: A test skill
allowed-tools: Bash  Read   Write
---

Content
"""

    result = _parse_skill_metadata(content, "/skills/test-skill/SKILL.md", "test-skill")
    assert result is not None
    assert result["allowed_tools"] == ["Bash", "Read", "Write"]


def test_validate_skill_name_unicode_lowercase() -> None:
    """Test _validate_skill_name accepts unicode lowercase alphanumeric characters."""
    # Unicode lowercase letters (e.g., accented characters)
    is_valid, _ = _validate_skill_name("café", "café")
    assert is_valid

    is_valid, _ = _validate_skill_name("über-tool", "über-tool")
    assert is_valid


def test_validate_skill_name_rejects_unicode_uppercase() -> None:
    """Test _validate_skill_name rejects unicode uppercase characters."""
    is_valid, error = _validate_skill_name("Café", "Café")
    assert not is_valid
    assert "lowercase" in error


def test_validate_skill_name_rejects_cjk_characters() -> None:
    """Test _validate_skill_name rejects CJK characters."""
    is_valid, error = _validate_skill_name("中文", "中文")
    assert not is_valid
    assert "lowercase" in error


def test_validate_skill_name_rejects_emoji() -> None:
    """Test _validate_skill_name rejects emoji characters."""
    is_valid, error = _validate_skill_name("tool-😀", "tool-😀")
    assert not is_valid
    assert "lowercase" in error


def test_format_skill_annotations_both_fields() -> None:
    """Test _format_skill_annotations with both license and compatibility."""
    skill = SkillMetadata(
        name="s",
        description="d",
        path="/p",
        license="MIT",
        compatibility="Python 3.10+",
        metadata={},
        allowed_tools=[],
    )
    assert _format_skill_annotations(skill) == "License: MIT, Compatibility: Python 3.10+"


def test_format_skill_annotations_license_only() -> None:
    """Test _format_skill_annotations with only license set."""
    skill = SkillMetadata(
        name="s",
        description="d",
        path="/p",
        license="Apache-2.0",
        compatibility=None,
        metadata={},
        allowed_tools=[],
    )
    assert _format_skill_annotations(skill) == "License: Apache-2.0"


def test_format_skill_annotations_compatibility_only() -> None:
    """Test _format_skill_annotations with only compatibility set."""
    skill = SkillMetadata(
        name="s",
        description="d",
        path="/p",
        license=None,
        compatibility="Requires poppler",
        metadata={},
        allowed_tools=[],
    )
    assert _format_skill_annotations(skill) == "Compatibility: Requires poppler"


def test_format_skill_annotations_neither_field() -> None:
    """Test _format_skill_annotations returns empty string when no fields set."""
    skill = SkillMetadata(
        name="s",
        description="d",
        path="/p",
        license=None,
        compatibility=None,
        metadata={},
        allowed_tools=[],
    )
    assert _format_skill_annotations(skill) == ""


def test_validate_metadata_non_dict_returns_empty() -> None:
    """Test _validate_metadata returns empty dict for non-dict input."""
    result = _validate_metadata("not a dict", "/skills/s/SKILL.md")
    assert result == {}


def test_validate_metadata_list_returns_empty() -> None:
    """Test _validate_metadata returns empty dict for list input."""
    result = _validate_metadata(["a", "b"], "/skills/s/SKILL.md")
    assert result == {}


def test_validate_metadata_coerces_values_to_str() -> None:
    """Test _validate_metadata coerces non-string values to strings."""
    result = _validate_metadata({"count": 42, "active": True}, "/skills/s/SKILL.md")
    assert result == {"count": "42", "active": "True"}


def test_validate_metadata_valid_dict_passthrough() -> None:
    """Test _validate_metadata passes through valid dict[str, str]."""
    result = _validate_metadata({"author": "acme"}, "/skills/s/SKILL.md")
    assert result == {"author": "acme"}


def test_parse_skill_metadata_allowed_tools_yaml_list_ignored() -> None:
    content = """---
name: test-skill
description: A test skill
allowed-tools:
  - Bash
  - Read
  - Write
---

Content
"""

    result = _parse_skill_metadata(content, "/skills/test-skill/SKILL.md", "test-skill")
    assert result is not None
    assert result["allowed_tools"] == []


def test_parse_skill_metadata_allowed_tools_yaml_list_non_strings_ignored() -> None:
    content = """---
name: test-skill
description: A test skill
allowed-tools:
  - Read
  - 123
  - true
  -
  - "  "
  - Write
---

Content
"""

    result = _parse_skill_metadata(content, "/skills/test-skill/SKILL.md", "test-skill")
    assert result is not None
    assert result["allowed_tools"] == []


def test_parse_skill_metadata_license_boolean_coerced() -> None:
    """Test _parse_skill_metadata coerces non-string license to string.

    YAML parses `license: true` as Python `True`. The parser should coerce it to
    a string rather than crashing.
    """
    content = """---
name: test-skill
description: A test skill
license: true
---

Content
"""

    result = _parse_skill_metadata(content, "/skills/test-skill/SKILL.md", "test-skill")
    assert result is not None
    assert result["license"] == "True"


def test_parse_skill_metadata_non_dict_metadata_ignored() -> None:
    """Test _parse_skill_metadata handles non-dict metadata gracefully.

    YAML parses `metadata: some-text` as a string. The parser should coerce it
    to an empty dict rather than crashing.
    """
    content = """---
name: test-skill
description: A test skill
metadata: some-text
---

Content
"""

    result = _parse_skill_metadata(content, "/skills/test-skill/SKILL.md", "test-skill")
    assert result is not None
    assert result["metadata"] == {}


def test_list_skills_from_backend_single_skill(tmp_path: Path) -> None:
    """Test listing a single skill from filesystem backend."""
    # Create backend with actual filesystem (no virtual mode)
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create skill using backend's upload_files interface
    skills_dir = tmp_path / "skills"
    skill_path = (skills_dir / "my-skill" / "SKILL.md").as_posix()
    skill_content = make_skill_content("my-skill", "My test skill")

    responses = backend.upload_files([(skill_path, skill_content.encode("utf-8"))])
    assert responses[0].error is None

    # List skills using the full absolute path
    skills = _list_skills(backend, skills_dir.as_posix())

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
    skills = _list_skills(backend, str(skills_dir))

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
    skills = _list_skills(backend, str(skills_dir))
    assert skills == []


def test_list_skills_from_backend_nonexistent_path(tmp_path: Path) -> None:
    """Test listing skills from a path that doesn't exist."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Try to list from non-existent directory
    skills = _list_skills(backend, str(tmp_path / "nonexistent"))
    assert skills == []


def test_list_skills_logs_ls_error(caplog: pytest.LogCaptureFixture) -> None:
    """A backend listing error should not look like a normal empty source."""
    backend = MagicMock()
    backend.ls.return_value = LsResult(error="Cannot list '/bad': denied", entries=[])

    with caplog.at_level(logging.WARNING):
        skills = _list_skills(backend, "/bad")

    assert skills == []
    assert "Cannot load skills from '/bad'" in caplog.text
    backend.download_files.assert_not_called()


def test_list_skills_loads_partial_results_with_ls_error(caplog: pytest.LogCaptureFixture) -> None:
    """A partial listing warning should not hide valid sibling skills."""
    skill_content = make_skill_content("valid-skill", "Valid skill")
    skill_dir_path = "/skills/valid-skill/"
    skill_md_path = "/skills/valid-skill/SKILL.md"

    backend = MagicMock()
    backend.ls.return_value = LsResult(error="Cannot list '/skills/loop': symlink loop", entries=[FileInfo(path=skill_dir_path, is_dir=True)])
    backend.download_files.return_value = [FileDownloadResponse(path=skill_md_path, content=skill_content.encode("utf-8"), error=None)]

    with caplog.at_level(logging.WARNING):
        skills = _list_skills(backend, "/skills")

    assert len(skills) == 1
    assert skills[0]["name"] == "valid-skill"
    assert "Cannot load skills from '/skills'" in caplog.text
    backend.download_files.assert_called_once_with([skill_md_path])


def test_list_skills_from_backend_missing_skill_md(tmp_path: Path) -> None:
    """Test that directories without SKILL.md are skipped."""
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
    skills = _list_skills(backend, skills_dir.as_posix())

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


def test_list_skills_from_backend_invalid_frontmatter(tmp_path: Path) -> None:
    """Test that skills with invalid YAML frontmatter are skipped."""
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
    skills = _list_skills(backend, skills_dir.as_posix())

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


def test_list_skills_from_backend_with_helper_files(tmp_path: Path) -> None:
    """Test that skills can have additional helper files."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create a skill with helper files
    skills_dir = tmp_path / "skills"
    skill_path = (skills_dir / "my-skill" / "SKILL.md").as_posix()
    helper_path = (skills_dir / "my-skill" / "helper.py").as_posix()

    skill_content = make_skill_content("my-skill", "My test skill")
    helper_content = "def helper(): pass"

    backend.upload_files(
        [
            (skill_path, skill_content.encode("utf-8")),
            (helper_path, helper_content.encode("utf-8")),
        ]
    )

    # List skills - should find the skill and not be confused by helper files
    skills = _list_skills(backend, skills_dir.as_posix())

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
def test_list_skills_with_windows_style_paths(skill_dir_path: str, source_path: str) -> None:
    r"""Skills load correctly when the backend returns Windows-style paths.

    Regression: `PurePosixPath` treats `\` as a literal filename char, so
    `_list_skills` must normalize before extracting the directory name and
    appending `SKILL.md`. Without normalization, the requested download
    path would be e.g. `C:\...\my-skill\SKILL.md\SKILL.md` (or would fail
    name validation entirely).
    """
    skill_content = make_skill_content("my-skill", "My test skill")
    expected_skill_md_path = str(PurePosixPath(skill_dir_path.replace("\\", "/")) / "SKILL.md")

    backend = MagicMock()
    backend.ls = MagicMock(return_value=LsResult(entries=[FileInfo(path=skill_dir_path, is_dir=True)]))
    backend.download_files = MagicMock(
        return_value=[
            FileDownloadResponse(
                path=expected_skill_md_path,
                content=skill_content.encode("utf-8"),
                error=None,
            )
        ]
    )

    skills = _list_skills(backend, source_path)

    # Pins the whole fix: the normalized POSIX path must be what gets requested.
    backend.download_files.assert_called_once_with([expected_skill_md_path])
    assert len(skills) == 1
    assert skills[0]["name"] == "my-skill"
    assert skills[0]["description"] == "My test skill"
    assert skills[0]["path"] == expected_skill_md_path


class TestSkillMetadataFromResponseLogging:
    """`_skill_metadata_from_response` must warn on non-`file_not_found` errors.

    `file_not_found` is the expected miss when a subdirectory isn't a skill,
    so it stays silent. Every other error (most importantly `is_directory`
    from `FilesystemBackend.download_files` for a path that happens to be a
    directory) must surface in logs so operators can debug missing skills
    without resorting to backend introspection.
    """

    def test_is_directory_error_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        response = FileDownloadResponse(
            path="/skills/my-skill/SKILL.md",
            content=None,
            error="is_directory",
        )
        with caplog.at_level("WARNING", logger="deepagents.middleware.skills"):
            result = _skill_metadata_from_response(
                response,
                skill_dir_path="/skills/my-skill",
                skill_md_path="/skills/my-skill/SKILL.md",
            )
        assert result is None
        assert any(
            record.levelname == "WARNING" and "is_directory" in record.getMessage() and "/skills/my-skill/SKILL.md" in record.getMessage()
            for record in caplog.records
        ), f"Expected is_directory warning, got records: {[r.getMessage() for r in caplog.records]}"

    def test_file_not_found_error_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        response = FileDownloadResponse(
            path="/skills/not-a-skill/SKILL.md",
            content=None,
            error="file_not_found",
        )
        with caplog.at_level("WARNING", logger="deepagents.middleware.skills"):
            result = _skill_metadata_from_response(
                response,
                skill_dir_path="/skills/not-a-skill",
                skill_md_path="/skills/not-a-skill/SKILL.md",
            )
        assert result is None
        assert caplog.records == []

    def test_permission_denied_error_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        response = FileDownloadResponse(
            path="/skills/locked/SKILL.md",
            content=None,
            error="permission_denied",
        )
        with caplog.at_level("WARNING", logger="deepagents.middleware.skills"):
            result = _skill_metadata_from_response(
                response,
                skill_dir_path="/skills/locked",
                skill_md_path="/skills/locked/SKILL.md",
            )
        assert result is None
        assert any("permission_denied" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    ("source_path", "expected_label"),
    [
        ("C:\\Users\\project\\skills\\", "Project"),
        ("C:\\Users\\project\\skills", "Project"),
        ("\\\\server\\share\\skills\\", "Share"),
    ],
    ids=["trailing-backslash", "no-trailing-sep", "unc-path"],
)
def test_format_skills_locations_with_windows_path(source_path: str, expected_label: str) -> None:
    r"""Derive a sensible label from Windows-style source paths.

    Without backslash normalization, `.name` on the resulting `PurePosixPath`
    returns the raw backslashed string (or the empty string for UNC paths
    where `\\\\server\\share\\skills` has no POSIX-delimited final component),
    not the intended directory label. A leaf of literal `skills` climbs one
    level so the label reflects the scope (`project`, `share`) rather than
    collapsing to the generic "Skills Skills" duplicate.
    """
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore[arg-type]
        sources=[source_path],
    )

    result = middleware._format_skills_locations()
    assert f"**{expected_label} Skills**:" in result
    assert "**Skills Skills**:" not in result
    assert source_path in result


def test_format_skills_locations_builtin_leaf() -> None:
    """`built_in_skills` collapses to `Built-in Skills` rather than the raw leaf."""
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore[arg-type]
        sources=["/pkg/deepagents_cli/built_in_skills"],
    )

    result = middleware._format_skills_locations()
    assert "**Built-in Skills**:" in result


def test_format_skills_locations_skills_leaf_climbs_to_parent() -> None:
    """A leaf of literal `skills` derives its label from the parent dir."""
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore[arg-type]
        sources=[
            "/home/me/.claude/skills",
            "/home/me/.agents/skills/",
            "/home/me/.deepagents/skills",
        ],
    )

    result = middleware._format_skills_locations()
    assert "**Claude Skills**:" in result
    assert "**Agents Skills**:" in result
    assert "**Deepagents Skills**:" in result
    assert "**Skills Skills**:" not in result


def test_format_skills_locations_explicit_label_tuples() -> None:
    """`(path, label)` tuples use the supplied label verbatim."""
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore[arg-type]
        sources=[
            ("/home/me/.claude/skills", "User Claude"),
            ("/repo/.claude/skills", "Project Claude"),
        ],
    )

    result = middleware._format_skills_locations()
    assert "**User Claude Skills**: `/home/me/.claude/skills`" in result
    assert "**Project Claude Skills**: `/repo/.claude/skills`" in result
    # Higher-priority marker belongs to the last entry.
    assert result.rstrip().endswith("(higher priority)")
    assert "Project Claude" in result.split("(higher priority)")[0]


@pytest.mark.parametrize(
    ("source_path", "expected_label"),
    [
        ("/skills", "Skills"),
        ("/skills/", "Skills"),
        ("skills", "Skills"),
        ("/foo/my_custom", "My_custom"),
        ("/foo/my-skills", "My-skills"),
    ],
    ids=[
        "root-anchored-skills",
        "root-anchored-skills-trailing",
        "bare-skills",
        "underscore-leaf-capitalize-fallback",
        "hyphen-leaf-capitalize-fallback",
    ],
)
def test_format_skills_locations_fallback_capitalize(source_path: str, expected_label: str) -> None:
    """Bare paths without a special leaf fall back to `.capitalize()`.

    Preserves the historical labelling for existing callers: a leaf like
    `my_custom` renders as `My_custom` (single-capital) rather than the
    more aggressive `My Custom` title-casing. Root-anchored and bare
    `skills` inputs cannot climb and fall back to `Skills`.
    """
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore[arg-type]
        sources=[source_path],
    )

    result = middleware._format_skills_locations()
    assert f"**{expected_label} Skills**:" in result


@pytest.mark.parametrize(
    "empty_path",
    ["", "/"],
    ids=["empty-string", "root-only"],
)
def test_format_skills_locations_empty_path_fallback(empty_path: str) -> None:
    """Empty/`/` inputs fall back to `Unnamed` without crashing render."""
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore[arg-type]
        sources=[empty_path],
    )

    result = middleware._format_skills_locations()
    assert "**Unnamed Skills**:" in result


@pytest.mark.parametrize(
    "bad_source",
    [
        ("/only-one-element",),
        ("/one", "two", "three"),
        ("/path", 42),
        (None, "label"),
    ],
    ids=["one-tuple", "three-tuple", "non-string-label", "non-string-path"],
)
def test_malformed_tuple_source_raises_type_error(bad_source: object) -> None:
    """Malformed tuple sources raise `TypeError` at construction time.

    Fails close to the caller rather than surfacing later as an
    `IndexError` in the middleware or a silently-coerced non-string path
    downstream.
    """
    with pytest.raises(TypeError, match=r"expected str or \(str, str\) tuple"):
        SkillsMiddleware(
            backend=None,  # type: ignore[arg-type]
            sources=[bad_source],  # type: ignore[list-item]
        )


def test_sources_attribute_is_paths_only() -> None:
    """`middleware.sources` exposes paths only; labels live on `source_labels`.

    Backwards-compat: callers that inspected `middleware.sources` before
    the tuple-form API was added continue to see a plain `list[str]`.
    """
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore[arg-type]
        sources=[
            "/skills/user/",
            ("/home/me/.claude/skills", "User Claude"),
        ],
    )

    assert middleware.sources == ["/skills/user/", "/home/me/.claude/skills"]
    assert middleware.source_labels == ["User", "User Claude"]


def test_format_skills_locations_single_registry() -> None:
    """Test _format_skills_locations with a single source."""
    sources = ["/skills/user/"]
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore[arg-type]
        sources=sources,
    )

    result = middleware._format_skills_locations()
    assert "User Skills" in result
    assert "/skills/user/" in result
    assert "(higher priority)" in result


def test_format_skills_locations_multiple_registries() -> None:
    """Test _format_skills_locations with multiple sources."""
    sources = [
        "/skills/base/",
        "/skills/user/",
        "/skills/project/",
    ]
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore[arg-type]
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
    sources = [
        "/skills/user/",
        "/skills/project/",
    ]
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore[arg-type]
        sources=sources,
    )

    result = middleware._format_skills_list([])
    assert "No skills available" in result
    assert "/skills/user/" in result
    assert "/skills/project/" in result


def test_format_skills_load_warnings() -> None:
    """Test _format_skills_load_warnings with source errors."""
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore[arg-type]
        sources=["/skills/user/"],
    )

    result = middleware._format_skills_load_warnings(["Cannot load skills from '/bad': denied"])

    assert "Skill Loading Warnings" in result
    assert "Cannot load skills from '/bad': denied" in result


def test_format_skills_list_single_skill() -> None:
    """Test _format_skills_list with a single skill."""
    sources = ["/skills/user/"]
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore[arg-type]
        sources=sources,
    )

    skills: list[SkillMetadata] = [
        {
            "name": "web-research",
            "description": "Research topics on the web",
            "path": "/skills/user/web-research/SKILL.md",
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


def test_format_skills_list_multiple_skills_multiple_registries() -> None:
    """Test _format_skills_list with skills from multiple sources."""
    sources = [
        "/skills/user/",
        "/skills/project/",
    ]
    middleware = SkillsMiddleware(
        backend=None,  # type: ignore[arg-type]
        sources=sources,
    )

    skills: list[SkillMetadata] = [
        {
            "name": "skill-a",
            "description": "User skill A",
            "path": "/skills/user/skill-a/SKILL.md",
            "license": None,
            "compatibility": None,
            "metadata": {},
            "allowed_tools": [],
        },
        {
            "name": "skill-b",
            "description": "Project skill B",
            "path": "/skills/project/skill-b/SKILL.md",
            "license": None,
            "compatibility": None,
            "metadata": {},
            "allowed_tools": [],
        },
        {
            "name": "skill-c",
            "description": "User skill C",
            "path": "/skills/user/skill-c/SKILL.md",
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

    # Check descriptions
    assert "User skill A" in result
    assert "Project skill B" in result
    assert "User skill C" in result


def test_format_skills_list_with_license_and_compatibility() -> None:
    """Test that both license and compatibility are shown in annotations."""
    middleware = SkillsMiddleware(backend=None, sources=["/skills/"])  # type: ignore[arg-type]

    skills: list[SkillMetadata] = [
        {
            "name": "my-skill",
            "description": "Does things",
            "path": "/skills/my-skill/SKILL.md",
            "license": "Apache-2.0",
            "compatibility": "Requires poppler",
            "metadata": {},
            "allowed_tools": [],
        }
    ]

    result = middleware._format_skills_list(skills)
    assert "(License: Apache-2.0, Compatibility: Requires poppler)" in result


def test_format_skills_list_license_only() -> None:
    """Test annotation with only license present."""
    middleware = SkillsMiddleware(backend=None, sources=["/skills/"])  # type: ignore[arg-type]

    skills: list[SkillMetadata] = [
        {
            "name": "licensed-skill",
            "description": "A licensed skill",
            "path": "/skills/licensed-skill/SKILL.md",
            "license": "MIT",
            "compatibility": None,
            "metadata": {},
            "allowed_tools": [],
        }
    ]

    result = middleware._format_skills_list(skills)
    assert "(License: MIT)" in result
    assert "Compatibility" not in result


def test_format_skills_list_compatibility_only() -> None:
    """Test annotation with only compatibility present."""
    middleware = SkillsMiddleware(backend=None, sources=["/skills/"])  # type: ignore[arg-type]

    skills: list[SkillMetadata] = [
        {
            "name": "compat-skill",
            "description": "A compatible skill",
            "path": "/skills/compat-skill/SKILL.md",
            "license": None,
            "compatibility": "Python 3.10+",
            "metadata": {},
            "allowed_tools": [],
        }
    ]

    result = middleware._format_skills_list(skills)
    assert "(Compatibility: Python 3.10+)" in result
    assert "License" not in result


def test_format_skills_list_no_optional_fields() -> None:
    """Test that no annotations appear when license/compatibility are empty."""
    middleware = SkillsMiddleware(backend=None, sources=["/skills/"])  # type: ignore[arg-type]

    skills: list[SkillMetadata] = [
        {
            "name": "plain-skill",
            "description": "A plain skill",
            "path": "/skills/plain-skill/SKILL.md",
            "license": None,
            "compatibility": None,
            "metadata": {},
            "allowed_tools": [],
        }
    ]

    result = middleware._format_skills_list(skills)
    # Description line should NOT have any parenthetical annotation
    assert "- **plain-skill**: A plain skill\n" in result
    assert "License" not in result
    assert "Compatibility" not in result
    assert "(advisory)" not in result


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

    sources = [str(skills_dir)]
    middleware = SkillsMiddleware(
        backend=backend,
        sources=sources,
    )

    # Call before_agent
    result = middleware.before_agent({}, None, {})  # type: ignore[arg-type]

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

    # Call before_agent
    result = middleware.before_agent({}, None, {})  # type: ignore[arg-type]

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


def test_before_agent_empty_registries(tmp_path: Path) -> None:
    """Test before_agent with empty sources."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create empty directories
    (tmp_path / "skills" / "user").mkdir(parents=True)

    sources = [str(tmp_path / "skills" / "user")]
    middleware = SkillsMiddleware(
        backend=backend,
        sources=sources,
    )

    result = middleware.before_agent({}, None, {})  # type: ignore[arg-type]

    assert result is not None
    assert result["skills_metadata"] == []


def test_before_agent_records_skill_load_errors() -> None:
    """Source load errors should be available in private middleware state."""
    backend = SimpleNamespace(
        ls=MagicMock(return_value=LsResult(error="Cannot list '/bad': denied", entries=[])),
        download_files=MagicMock(),
    )
    middleware = SkillsMiddleware(backend=backend, sources=["/bad"])

    result = middleware.before_agent({}, None, {})  # type: ignore[arg-type]

    assert result is not None
    assert result["skills_metadata"] == []
    assert result["skills_load_errors"] == ["Cannot load skills from '/bad': Cannot list '/bad': denied"]


def test_before_agent_partial_load_across_sources() -> None:
    """A failing source must not hide skills loaded from a sibling source."""
    skill_content = make_skill_content("good-skill", "Skill from the working source")
    skill_dir_path = "/good/good-skill/"
    skill_md_path = "/good/good-skill/SKILL.md"

    def ls_side_effect(path: str) -> LsResult:
        if path == "/good":
            return LsResult(entries=[FileInfo(path=skill_dir_path, is_dir=True)])
        return LsResult(error="Cannot list '/bad': denied", entries=[])

    backend = SimpleNamespace(
        ls=MagicMock(side_effect=ls_side_effect),
        download_files=MagicMock(return_value=[FileDownloadResponse(path=skill_md_path, content=skill_content.encode("utf-8"), error=None)]),
    )
    middleware = SkillsMiddleware(backend=backend, sources=["/good", "/bad"])

    result = middleware.before_agent({}, None, {})  # type: ignore[arg-type]

    assert result is not None
    assert [skill["name"] for skill in result["skills_metadata"]] == ["good-skill"]
    assert result["skills_load_errors"] == ["Cannot load skills from '/bad': Cannot list '/bad': denied"]


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
    sources = [str(skills_dir)]
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


def test_skills_middleware_with_state_backend() -> None:
    """Test that SkillsMiddleware can be initialized with StateBackend instance."""
    sources = ["/skills/user"]
    middleware = SkillsMiddleware(
        backend=StateBackend(),
        sources=sources,
    )

    # Verify the middleware was created successfully
    assert middleware is not None
    assert isinstance(middleware._backend, StateBackend)
    assert len(middleware.sources) == 1
    assert middleware.sources[0] == "/skills/user"

    runtime = SimpleNamespace(
        context=None,
        store=None,
        stream_writer=lambda _: None,
    )

    backend = middleware._get_backend({"messages": [], "files": {}}, runtime, {})
    assert isinstance(backend, StateBackend)


def test_skills_middleware_with_store_backend_instance() -> None:
    """Test that SkillsMiddleware can be initialized with StoreBackend instance."""
    store = InMemoryStore()
    sources = ["/skills/user"]
    middleware = SkillsMiddleware(
        backend=StoreBackend(store=store, namespace=_assistant_id_namespace),
        sources=sources,
    )

    # Verify the middleware was created successfully
    assert middleware is not None
    assert isinstance(middleware._backend, StoreBackend)
    assert len(middleware.sources) == 1
    assert middleware.sources[0] == "/skills/user"


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
    sources = [str(skills_dir)]
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
    sources = [
        str(base_dir),
        str(user_dir),
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

    sources = [str(skills_dir)]
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
            "metadata": {},
            "license": None,
            "compatibility": None,
            "allowed_tools": [],
        }
    ]
    state_with_metadata = {"skills_metadata": existing_metadata}
    result = middleware.before_agent(state_with_metadata, None, {})  # type: ignore[arg-type]

    # Should return None, not load new skills
    assert result is None

    # Case 2: State has empty list for skills_metadata
    state_with_empty_list = {"skills_metadata": []}
    result = middleware.before_agent(state_with_empty_list, None, {})  # type: ignore[arg-type]

    # Should still return None and not reload
    assert result is None

    # Case 3: State does NOT have skills_metadata key
    state_without_metadata = {}
    result = middleware.before_agent(state_without_metadata, None, {})  # type: ignore[arg-type]

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
        skills=[str(skills_dir)],
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
        skills=[str(skills_dir)],
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
        skills=["/skills/user"],
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
            "content": skill_content,
            "encoding": "utf-8",
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
        },
    ]


def create_store_skill_item(content: str) -> dict:
    """Create a skill item in StoreBackend FileData format.

    Args:
        content: Skill content string

    Returns:
        Dict with content as str, encoding, created_at, and modified_at
    """
    timestamp = datetime.now(UTC).isoformat()
    return {
        "content": content,
        "encoding": "utf-8",
        "created_at": timestamp,
        "modified_at": timestamp,
    }


def test_skills_middleware_with_store_backend_assistant_id() -> None:
    """Test namespace isolation: each assistant_id gets its own skills namespace."""
    store = InMemoryStore()
    middleware = SkillsMiddleware(
        backend=StoreBackend(store=store, namespace=_assistant_id_namespace),
        sources=["/skills/user"],
    )
    runtime = SimpleNamespace(context=None, store=store, stream_writer=lambda _: None)

    # Add skill for assistant-123 with namespace (assistant-123, filesystem)
    assistant_1_skill = make_skill_content("skill-one", "Skill for assistant 1")
    store.put(
        ("assistant-123", "filesystem"),
        "/skills/user/skill-one/SKILL.md",
        create_store_skill_item(assistant_1_skill),
    )

    # Test: assistant-123 can read its own skill
    with _runtime_context("assistant-123"):
        result_1 = middleware.before_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_1 is not None
    assert len(result_1["skills_metadata"]) == 1
    assert result_1["skills_metadata"][0]["name"] == "skill-one"
    assert result_1["skills_metadata"][0]["description"] == "Skill for assistant 1"

    # Test: assistant-456 cannot see assistant-123's skill (different namespace)
    with _runtime_context("assistant-456"):
        result_2 = middleware.before_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_2 is not None
    assert len(result_2["skills_metadata"]) == 0  # No skills in assistant-456's namespace yet

    # Add skill for assistant-456 with namespace (assistant-456, filesystem)
    assistant_2_skill = make_skill_content("skill-two", "Skill for assistant 2")
    store.put(
        ("assistant-456", "filesystem"),
        "/skills/user/skill-two/SKILL.md",
        create_store_skill_item(assistant_2_skill),
    )

    # Test: assistant-456 can read its own skill
    with _runtime_context("assistant-456"):
        result_3 = middleware.before_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_3 is not None
    assert len(result_3["skills_metadata"]) == 1
    assert result_3["skills_metadata"][0]["name"] == "skill-two"
    assert result_3["skills_metadata"][0]["description"] == "Skill for assistant 2"

    # Test: assistant-123 still only sees its own skill (no cross-contamination)
    with _runtime_context("assistant-123"):
        result_4 = middleware.before_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_4 is not None
    assert len(result_4["skills_metadata"]) == 1
    assert result_4["skills_metadata"][0]["name"] == "skill-one"
    assert result_4["skills_metadata"][0]["description"] == "Skill for assistant 1"


def test_skills_middleware_with_store_backend_no_assistant_id() -> None:
    """Test default namespace: when no assistant_id is provided, uses (filesystem,) namespace."""
    store = InMemoryStore()
    middleware = SkillsMiddleware(
        backend=StoreBackend(store=store, namespace=_assistant_id_namespace),
        sources=["/skills/user"],
    )
    runtime = SimpleNamespace(context=None, store=store, stream_writer=lambda _: None)

    # Add skill to default namespace (filesystem,) - no assistant_id
    shared_skill = make_skill_content("shared-skill", "Shared namespace skill")
    store.put(
        ("filesystem",),
        "/skills/user/shared-skill/SKILL.md",
        create_store_skill_item(shared_skill),
    )

    # Test: runtime without server_info accesses default namespace
    with _runtime_context(None):
        result_1 = middleware.before_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_1 is not None
    assert len(result_1["skills_metadata"]) == 1
    assert result_1["skills_metadata"][0]["name"] == "shared-skill"
    assert result_1["skills_metadata"][0]["description"] == "Shared namespace skill"

    # Test: runtime with server_info but empty assistant_id also uses default namespace
    with _runtime_context(""):
        result_2 = middleware.before_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_2 is not None
    assert len(result_2["skills_metadata"]) == 1
    assert result_2["skills_metadata"][0]["name"] == "shared-skill"
    assert result_2["skills_metadata"][0]["description"] == "Shared namespace skill"


async def test_skills_middleware_with_store_backend_assistant_id_async() -> None:
    """Test namespace isolation with async: each assistant_id gets its own skills namespace."""
    store = InMemoryStore()
    middleware = SkillsMiddleware(
        backend=StoreBackend(store=store, namespace=_assistant_id_namespace),
        sources=["/skills/user"],
    )
    runtime = SimpleNamespace(context=None, store=store, stream_writer=lambda _: None)

    # Add skill for assistant-123 with namespace (assistant-123, filesystem)
    assistant_1_skill = make_skill_content("async-skill-one", "Async skill for assistant 1")
    store.put(
        ("assistant-123", "filesystem"),
        "/skills/user/async-skill-one/SKILL.md",
        create_store_skill_item(assistant_1_skill),
    )

    # Test: assistant-123 can read its own skill
    with _runtime_context("assistant-123"):
        result_1 = await middleware.abefore_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_1 is not None
    assert len(result_1["skills_metadata"]) == 1
    assert result_1["skills_metadata"][0]["name"] == "async-skill-one"
    assert result_1["skills_metadata"][0]["description"] == "Async skill for assistant 1"

    # Test: assistant-456 cannot see assistant-123's skill (different namespace)
    with _runtime_context("assistant-456"):
        result_2 = await middleware.abefore_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_2 is not None
    assert len(result_2["skills_metadata"]) == 0  # No skills in assistant-456's namespace yet

    # Add skill for assistant-456 with namespace (assistant-456, filesystem)
    assistant_2_skill = make_skill_content("async-skill-two", "Async skill for assistant 2")
    store.put(
        ("assistant-456", "filesystem"),
        "/skills/user/async-skill-two/SKILL.md",
        create_store_skill_item(assistant_2_skill),
    )

    # Test: assistant-456 can read its own skill
    with _runtime_context("assistant-456"):
        result_3 = await middleware.abefore_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_3 is not None
    assert len(result_3["skills_metadata"]) == 1
    assert result_3["skills_metadata"][0]["name"] == "async-skill-two"
    assert result_3["skills_metadata"][0]["description"] == "Async skill for assistant 2"

    # Test: assistant-123 still only sees its own skill (no cross-contamination)
    with _runtime_context("assistant-123"):
        result_4 = await middleware.abefore_agent({}, runtime, {})  # type: ignore[arg-type]

    assert result_4 is not None
    assert len(result_4["skills_metadata"]) == 1
    assert result_4["skills_metadata"][0]["name"] == "async-skill-one"
    assert result_4["skills_metadata"][0]["description"] == "Async skill for assistant 1"
