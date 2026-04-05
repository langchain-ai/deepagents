"""Skills middleware for loading and exposing agent skills to the system prompt.

This module implements Anthropic's agent skills pattern with progressive disclosure,
loading skills from backend storage via configurable sources.

## Architecture

Skills are loaded from one or more **sources** - paths in a backend where skills are
organized. Sources are loaded in order, with later sources overriding earlier ones
when skills have the same name (last one wins). This enables layering: base -> user
-> project -> team skills.

The middleware uses backend APIs exclusively (no direct filesystem access), making it
portable across different storage backends (filesystem, state, remote storage, etc.).

For StateBackend (ephemeral/in-memory):
```python
SkillsMiddleware(backend=StateBackend(), ...)
```

## Skill Structure

Each skill is a directory containing a SKILL.md file with YAML frontmatter:

```
/skills/user/web-research/
├── SKILL.md          # Required: YAML frontmatter + markdown instructions
└── helper.py         # Optional: supporting files
```

SKILL.md format:
```markdown
---
name: web-research
description: Structured approach to conducting thorough web research
license: MIT
---

# Web Research Skill

## When to Use
- User asks you to research a topic
...
```

## Skill Metadata (SkillMetadata)

Parsed from YAML frontmatter per Agent Skills specification:
- `name`: Skill identifier (max 64 chars, lowercase alphanumeric and hyphens)
- `description`: What the skill does (max 1024 chars)
- `path`: Backend path to the SKILL.md file
- Optional: `license`, `compatibility`, `metadata`, `allowed_tools`

## Sources

Sources are simply paths to skill directories in the backend. The source name is
derived from the last component of the path (e.g., "/skills/user/" -> "user").

Example sources:
```python
[
    "/skills/user/",
    "/skills/project/"
]
```

## Path Conventions

All paths use POSIX conventions (forward slashes) via `PurePosixPath`:
- Backend paths: "/skills/user/web-research/SKILL.md"
- Virtual, platform-independent
- Backends handle platform-specific conversions as needed

## Usage

```python
from deepagents.backends.state import StateBackend
from deepagents.middleware.skills import SkillsMiddleware

middleware = SkillsMiddleware(
    backend=my_backend,
    sources=[
        "/skills/base/",
        "/skills/user/",
        "/skills/project/",
    ],
)
```
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any

import yaml
from langchain.agents.middleware.types import PrivateStateAttr
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.tools.tool_node import ToolCallRequest
    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime
    from langgraph.types import Command

    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol

from typing import NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool

from deepagents.backends.protocol import LsResult
from deepagents.middleware._utils import append_to_system_message

logger = logging.getLogger(__name__)

# Security: Maximum size for SKILL.md files to prevent DoS attacks (10MB)
MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024
MAX_INLINE_SKILL_BODY_BYTES = 64 * 1024

# Agent Skills specification constraints (https://agentskills.io/specification)
MAX_SKILL_NAME_LENGTH = 64
MAX_SKILL_DESCRIPTION_LENGTH = 1024
MAX_SKILL_COMPATIBILITY_LENGTH = 500


class SkillMetadata(TypedDict):
    """Metadata for a skill per Agent Skills specification (https://agentskills.io/specification)."""

    path: str
    """Path to the SKILL.md file."""

    name: str
    """Skill identifier.

    Constraints per Agent Skills specification:

    - 1-64 characters
    - Unicode lowercase alphanumeric and hyphens only (`a-z` and `-`).
    - Must not start or end with `-`
    - Must not contain consecutive `--`
    - Must match the parent directory name containing the `SKILL.md` file
    """

    description: str
    """What the skill does.

    Constraints per Agent Skills specification:

    - 1-1024 characters
    - Should describe both what the skill does and when to use it
    - Should include specific keywords that help agents identify relevant tasks
    """

    license: str | None
    """License name or reference to bundled license file."""

    compatibility: str | None
    """Environment requirements.

    Constraints per Agent Skills specification:

    - 1-500 characters if provided
    - Should only be included if there are specific compatibility requirements
    - Can indicate intended product, required packages, etc.
    """

    metadata: dict[str, str]
    """Arbitrary key-value mapping for additional metadata.

    Clients can use this to store additional properties not defined by the spec.

    It is recommended to keep key names unique to avoid conflicts.
    """

    allowed_tools: list[str]
    """Tool names the skill recommends using.

    Warning: this is experimental.

    Constraints per Agent Skills specification:

    - Space-delimited list of tool names
    """


class SkillsState(AgentState):
    """State for the skills middleware."""

    skills_metadata: NotRequired[Annotated[list[SkillMetadata], PrivateStateAttr]]
    """List of loaded skill metadata from configured sources. Not propagated to parent agents."""


class SkillsStateUpdate(TypedDict):
    """State update for the skills middleware."""

    skills_metadata: list[SkillMetadata]
    """List of loaded skill metadata to merge into state."""


class SkillSectionManifestEntry(TypedDict):
    """Metadata describing a markdown section in a skill body."""

    section_id: str
    title: str
    level: int
    start_line: int
    end_line: int
    required: bool


class LoadSkillPayload(TypedDict):
    """Structured response returned by the `load_skill` tool."""

    skill_name: str
    source_path: str
    root_path: str
    is_complete: bool
    truncated: bool
    byte_count: int
    line_count: int
    content_sha256: str
    frontmatter: SkillMetadata
    content: str
    section_manifest: list[SkillSectionManifestEntry]
    supporting_files_manifest: list[dict[str, Any]]
    requires_sectional_loading: bool
    missing_section_ids: list[str]


class LoadedSkillSection(TypedDict):
    """A fully loaded section returned by `get_skill_sections`."""

    section_id: str
    title: str
    level: int
    start_line: int
    end_line: int
    content: str
    content_sha256: str


class GetSkillSectionsPayload(TypedDict):
    """Structured response returned by the `get_skill_sections` tool."""

    skill_name: str
    source_path: str
    root_path: str
    loaded_sections: list[LoadedSkillSection]
    loaded_section_ids: list[str]
    missing_section_ids: list[str]
    is_complete: bool
    content_sha256: str


class LoadSkillSchema(BaseModel):
    """Input schema for the `load_skill` tool."""

    skill_name: str = Field(
        description="Canonical skill name from the available skills list.",
    )
    purpose: str | None = Field(
        default=None,
        description="Optional short explanation of why this skill is being activated.",
    )


class GetSkillSectionsSchema(BaseModel):
    """Input schema for the `get_skill_sections` tool."""

    skill_name: str = Field(
        description="Canonical skill name from the available skills list.",
    )
    section_ids: list[str] = Field(
        description="Section identifiers from `load_skill(...).section_manifest` to load in full.",
        min_length=1,
    )


def _validate_skill_name(name: str, directory_name: str) -> tuple[bool, str]:
    """Validate skill name per Agent Skills specification.

    Constraints per Agent Skills specification:

    - 1-64 characters
    - Unicode lowercase alphanumeric and hyphens only (`a-z` and `-`).
    - Must not start or end with `-`
    - Must not contain consecutive `--`
    - Must match the parent directory name containing the `SKILL.md` file

    Unicode lowercase alphanumeric means any character where `c.isalpha() and
    c.islower()` or `c.isdigit()` returns `True`, which covers accented Latin
    characters (e.g., `'café'`, `'über-tool'`) and other scripts.

    Args:
        name: Skill name from YAML frontmatter
        directory_name: Parent directory name

    Returns:
        `(is_valid, error_message)` tuple.

            Error message is empty if valid.
    """
    if not name:
        return False, "name is required"
    if len(name) > MAX_SKILL_NAME_LENGTH:
        return False, "name exceeds 64 characters"
    if name.startswith("-") or name.endswith("-") or "--" in name:
        return False, "name must be lowercase alphanumeric with single hyphens only"
    for c in name:
        if c == "-":
            continue
        if (c.isalpha() and c.islower()) or c.isdigit():
            continue
        return False, "name must be lowercase alphanumeric with single hyphens only"
    if name != directory_name:
        return False, f"name '{name}' must match directory name '{directory_name}'"
    return True, ""


def _parse_skill_metadata(  # noqa: C901
    content: str,
    skill_path: str,
    directory_name: str,
) -> SkillMetadata | None:
    """Parse YAML frontmatter from `SKILL.md` content.

    Extracts metadata per Agent Skills specification from YAML frontmatter
    delimited by `---` markers at the start of the content.

    Args:
        content: Content of the `SKILL.md` file
        skill_path: Path to the `SKILL.md` file (for error messages and metadata)
        directory_name: Name of the parent directory containing the skill

    Returns:
        `SkillMetadata` if parsing succeeds, `None` if parsing fails or
            validation errors occur
    """
    if len(content) > MAX_SKILL_FILE_SIZE:
        logger.warning("Skipping %s: content too large (%d bytes)", skill_path, len(content))
        return None

    # Match YAML frontmatter between --- delimiters
    frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(frontmatter_pattern, content, re.DOTALL)

    if not match:
        logger.warning("Skipping %s: no valid YAML frontmatter found", skill_path)
        return None

    frontmatter_str = match.group(1)

    # Parse YAML using safe_load for proper nested structure support
    try:
        frontmatter_data = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as e:
        logger.warning("Invalid YAML in %s: %s", skill_path, e)
        return None

    if not isinstance(frontmatter_data, dict):
        logger.warning("Skipping %s: frontmatter is not a mapping", skill_path)
        return None

    name = str(frontmatter_data.get("name", "")).strip()
    description = str(frontmatter_data.get("description", "")).strip()
    if not name or not description:
        logger.warning("Skipping %s: missing required 'name' or 'description'", skill_path)
        return None

    # Validate name format per spec (warn but continue loading for backwards compatibility)
    is_valid, error = _validate_skill_name(str(name), directory_name)
    if not is_valid:
        logger.warning(
            "Skill '%s' in %s does not follow Agent Skills specification: %s. Consider renaming for spec compliance.",
            name,
            skill_path,
            error,
        )

    description_str = description
    if len(description_str) > MAX_SKILL_DESCRIPTION_LENGTH:
        logger.warning(
            "Description exceeds %d characters in %s, truncating",
            MAX_SKILL_DESCRIPTION_LENGTH,
            skill_path,
        )
        description_str = description_str[:MAX_SKILL_DESCRIPTION_LENGTH]

    raw_tools = frontmatter_data.get("allowed-tools")
    if isinstance(raw_tools, str):
        allowed_tools = [
            t.strip(",")  # Support commas for compatibility with skills created for Claude Code.
            for t in raw_tools.split()
            if t.strip(",")
        ]
    else:
        if raw_tools is not None:
            logger.warning(
                "Ignoring non-string 'allowed-tools' in %s (got %s)",
                skill_path,
                type(raw_tools).__name__,
            )
        allowed_tools = []

    compatibility_str = str(frontmatter_data.get("compatibility", "")).strip() or None
    if compatibility_str and len(compatibility_str) > MAX_SKILL_COMPATIBILITY_LENGTH:
        logger.warning(
            "Compatibility exceeds %d characters in %s, truncating",
            MAX_SKILL_COMPATIBILITY_LENGTH,
            skill_path,
        )
        compatibility_str = compatibility_str[:MAX_SKILL_COMPATIBILITY_LENGTH]

    return SkillMetadata(
        name=str(name),
        description=description_str,
        path=skill_path,
        metadata=_validate_metadata(frontmatter_data.get("metadata", {}), skill_path),
        license=str(frontmatter_data.get("license", "")).strip() or None,
        compatibility=compatibility_str,
        allowed_tools=allowed_tools,
    )


def _validate_metadata(
    raw: object,
    skill_path: str,
) -> dict[str, str]:
    """Validate and normalize the metadata field from YAML frontmatter.

    YAML `safe_load` can return any type for the `metadata` key. This
    ensures the values in `SkillMetadata` are always a `dict[str, str]` by
    coercing via `str()` and rejecting non-dict inputs.

    Args:
        raw: Raw value from `frontmatter_data.get("metadata", {})`.
        skill_path: Path to the `SKILL.md` file (for warning messages).

    Returns:
        A validated `dict[str, str]`.
    """
    if not isinstance(raw, dict):
        if raw:
            logger.warning(
                "Ignoring non-dict metadata in %s (got %s)",
                skill_path,
                type(raw).__name__,
            )
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def _format_skill_annotations(skill: SkillMetadata) -> str:
    """Build a parenthetical annotation string from optional skill fields.

    Combines license and compatibility into a comma-separated string for
    display in the system prompt skill listing.

    Args:
        skill: Skill metadata to extract annotations from.

    Returns:
        Annotation string like `'License: MIT, Compatibility: Python 3.10+'`,
            or empty string if neither field is set.
    """
    parts: list[str] = []
    if skill.get("license"):
        parts.append(f"License: {skill['license']}")
    if skill.get("compatibility"):
        parts.append(f"Compatibility: {skill['compatibility']}")
    return ", ".join(parts)


def _list_skills(backend: BackendProtocol, source_path: str) -> list[SkillMetadata]:
    """List all skills from a backend source.

    Scans backend for subdirectories containing `SKILL.md` files, downloads
    their content, parses YAML frontmatter, and returns skill metadata.

    Expected structure:

    ```txt
    source_path/
    └── skill-name/
        ├── SKILL.md   # Required
        └── helper.py  # Optional
    ```

    Args:
        backend: Backend instance to use for file operations
        source_path: Path to the skills directory in the backend

    Returns:
        List of skill metadata from successfully parsed `SKILL.md` files
    """
    skills: list[SkillMetadata] = []
    ls_result = backend.ls(source_path)
    items = ls_result.entries if isinstance(ls_result, LsResult) else ls_result

    # Find all skill directories (directories containing SKILL.md)
    skill_dirs = []
    for item in items or []:
        if not item.get("is_dir"):
            continue
        skill_dirs.append(item["path"])

    if not skill_dirs:
        return []

    # For each skill directory, check if SKILL.md exists and download it
    skill_md_paths = []
    for skill_dir_path in skill_dirs:
        # Construct SKILL.md path using PurePosixPath for safe, standardized path operations
        skill_dir = PurePosixPath(skill_dir_path)
        skill_md_path = str(skill_dir / "SKILL.md")
        skill_md_paths.append((skill_dir_path, skill_md_path))

    paths_to_download = [skill_md_path for _, skill_md_path in skill_md_paths]
    responses = backend.download_files(paths_to_download)

    # Parse each downloaded SKILL.md
    for (skill_dir_path, skill_md_path), response in zip(skill_md_paths, responses, strict=True):
        if response.error:
            # Skill doesn't have a SKILL.md, skip it
            continue

        if response.content is None:
            logger.warning("Downloaded skill file %s has no content", skill_md_path)
            continue

        try:
            content = response.content.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.warning("Error decoding %s: %s", skill_md_path, e)
            continue

        # Extract directory name from path using PurePosixPath
        directory_name = PurePosixPath(skill_dir_path).name

        # Parse metadata
        skill_metadata = _parse_skill_metadata(
            content=content,
            skill_path=skill_md_path,
            directory_name=directory_name,
        )
        if skill_metadata:
            skills.append(skill_metadata)

    return skills


async def _alist_skills(backend: BackendProtocol, source_path: str) -> list[SkillMetadata]:
    """List all skills from a backend source (async version).

    Scans backend for subdirectories containing `SKILL.md` files, downloads
    their content, parses YAML frontmatter, and returns skill metadata.

    Expected structure:

    ```txt
    source_path/
    └── skill-name/
        ├── SKILL.md   # Required
        └── helper.py  # Optional
    ```

    Args:
        backend: Backend instance to use for file operations
        source_path: Path to the skills directory in the backend

    Returns:
        List of skill metadata from successfully parsed `SKILL.md` files
    """
    skills: list[SkillMetadata] = []
    ls_result = await backend.als(source_path)
    items = ls_result.entries if isinstance(ls_result, LsResult) else ls_result

    # Find all skill directories (directories containing SKILL.md)
    skill_dirs = []
    for item in items or []:
        if not item.get("is_dir"):
            continue
        skill_dirs.append(item["path"])

    if not skill_dirs:
        return []

    # For each skill directory, check if SKILL.md exists and download it
    skill_md_paths = []
    for skill_dir_path in skill_dirs:
        # Construct SKILL.md path using PurePosixPath for safe, standardized path operations
        skill_dir = PurePosixPath(skill_dir_path)
        skill_md_path = str(skill_dir / "SKILL.md")
        skill_md_paths.append((skill_dir_path, skill_md_path))

    paths_to_download = [skill_md_path for _, skill_md_path in skill_md_paths]
    responses = await backend.adownload_files(paths_to_download)

    # Parse each downloaded SKILL.md
    for (skill_dir_path, skill_md_path), response in zip(skill_md_paths, responses, strict=True):
        if response.error:
            # Skill doesn't have a SKILL.md, skip it
            continue

        if response.content is None:
            logger.warning("Downloaded skill file %s has no content", skill_md_path)
            continue

        try:
            content = response.content.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.warning("Error decoding %s: %s", skill_md_path, e)
            continue

        # Extract directory name from path using PurePosixPath
        directory_name = PurePosixPath(skill_dir_path).name

        # Parse metadata
        skill_metadata = _parse_skill_metadata(
            content=content,
            skill_path=skill_md_path,
            directory_name=directory_name,
        )
        if skill_metadata:
            skills.append(skill_metadata)

    return skills


def _split_frontmatter_and_body(content: str) -> tuple[str, str]:
    """Split raw `SKILL.md` content into YAML frontmatter and markdown body."""
    frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n?"
    match = re.match(frontmatter_pattern, content, re.DOTALL)
    if not match:
        return "", content
    return match.group(1), content[match.end() :]


def _slugify_heading(title: str) -> str:
    """Convert a markdown heading into a stable section identifier."""
    slug = title.strip().lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "section"


def _build_section_manifest(body: str) -> list[SkillSectionManifestEntry]:
    """Build a section manifest from markdown headings in a skill body."""
    lines = body.splitlines()
    headings: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines, start=1):
        match = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if match:
            headings.append((idx, len(match.group(1)), match.group(2).strip()))

    if not headings:
        line_count = len(lines)
        return [
            SkillSectionManifestEntry(
                section_id="full-document",
                title="Full Document",
                level=1,
                start_line=1,
                end_line=max(1, line_count),
                required=True,
            )
        ]

    manifest: list[SkillSectionManifestEntry] = []
    for i, (start_line, level, title) in enumerate(headings):
        end_line = len(lines)
        if i + 1 < len(headings):
            end_line = headings[i + 1][0] - 1
        manifest.append(
            SkillSectionManifestEntry(
                section_id=_slugify_heading(title),
                title=title,
                level=level,
                start_line=start_line,
                end_line=max(start_line, end_line),
                required=True,
            )
        )
    return manifest


def _extract_section_contents(
    body: str,
    section_manifest: list[SkillSectionManifestEntry],
    section_ids: list[str],
) -> tuple[list[LoadedSkillSection], list[str]]:
    """Extract the requested sections from a skill body."""
    lines = body.splitlines()
    sections_by_id = {section["section_id"]: section for section in section_manifest}

    loaded_sections: list[LoadedSkillSection] = []
    missing_section_ids: list[str] = []
    for section_id in section_ids:
        section = sections_by_id.get(section_id)
        if section is None:
            missing_section_ids.append(section_id)
            continue

        start_idx = max(0, section["start_line"] - 1)
        end_idx = max(start_idx, section["end_line"])
        content = "\n".join(lines[start_idx:end_idx])
        loaded_sections.append(
            LoadedSkillSection(
                section_id=section["section_id"],
                title=section["title"],
                level=section["level"],
                start_line=section["start_line"],
                end_line=section["end_line"],
                content=content,
                content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
        )

    return loaded_sections, missing_section_ids


def _build_supporting_files_manifest(
    backend: BackendProtocol,
    skill_root: str,
) -> list[dict[str, Any]]:
    """List supporting files that live alongside `SKILL.md`."""
    ls_result = backend.ls(skill_root)
    entries = ls_result.entries if isinstance(ls_result, LsResult) else ls_result
    manifest: list[dict[str, Any]] = []
    for entry in entries or []:
        path = str(entry.get("path", ""))
        if not path or PurePosixPath(path).name == "SKILL.md":
            continue
        manifest.append(
            {
                "path": path,
                "is_dir": bool(entry.get("is_dir", False)),
                "size": entry.get("size"),
                "modified_at": entry.get("modified_at"),
            }
        )
    return manifest


async def _abuild_supporting_files_manifest(
    backend: BackendProtocol,
    skill_root: str,
) -> list[dict[str, Any]]:
    """Async version of `_build_supporting_files_manifest`."""
    ls_result = await backend.als(skill_root)
    entries = ls_result.entries if isinstance(ls_result, LsResult) else ls_result
    manifest: list[dict[str, Any]] = []
    for entry in entries or []:
        path = str(entry.get("path", ""))
        if not path or PurePosixPath(path).name == "SKILL.md":
            continue
        manifest.append(
            {
                "path": path,
                "is_dir": bool(entry.get("is_dir", False)),
                "size": entry.get("size"),
                "modified_at": entry.get("modified_at"),
            }
        )
    return manifest


def _build_load_skill_payload(
    *,
    skill: SkillMetadata,
    raw_content: str,
    supporting_files_manifest: list[dict[str, Any]],
) -> LoadSkillPayload:
    """Build the structured payload returned by the `load_skill` tool."""
    _frontmatter, body = _split_frontmatter_and_body(raw_content)
    root_path = str(PurePosixPath(skill["path"]).parent)
    body_bytes = body.encode("utf-8")
    section_manifest = _build_section_manifest(body)
    requires_sectional_loading = len(body_bytes) > MAX_INLINE_SKILL_BODY_BYTES
    return LoadSkillPayload(
        skill_name=skill["name"],
        source_path=skill["path"],
        root_path=root_path,
        is_complete=not requires_sectional_loading,
        truncated=requires_sectional_loading,
        byte_count=len(body_bytes),
        line_count=max(1, len(body.splitlines())),
        content_sha256=hashlib.sha256(body_bytes).hexdigest(),
        frontmatter=skill,
        content="" if requires_sectional_loading else body,
        section_manifest=section_manifest,
        supporting_files_manifest=supporting_files_manifest,
        requires_sectional_loading=requires_sectional_loading,
        missing_section_ids=[section["section_id"] for section in section_manifest] if requires_sectional_loading else [],
    )


SKILLS_SYSTEM_PROMPT = """

## Skills System

You have access to a skills library that provides specialized capabilities and domain knowledge.

{skills_locations}

**Available Skills:**

{skills_list}

**How to Use Skills (Progressive Disclosure):**

Skills follow a **progressive disclosure** pattern - you see their name and description above, but only load full instructions when needed:

1. **Recognize when a skill applies**: Check if the user's task matches a skill's description
2. **Load the skill's full instructions**: Call `load_skill(skill_name=...)` before acting on that skill
3. **Only treat a skill as fully loaded when the tool returns `is_complete: true`**
4. **If `load_skill` returns `requires_sectional_loading: true`**: Call `get_skill_sections(...)` with the section IDs from `missing_section_ids`
5. **Follow the skill's instructions**: The loaded skill content contains step-by-step workflows, constraints, and examples
6. **Access supporting files**: Skills may include helper scripts, configs,
   or reference docs - use the returned `root_path` and
   `supporting_files_manifest`

**When to Use Skills:**
- User's request matches a skill's domain (e.g., "research X" -> web-research skill)
- You need specialized knowledge or structured workflows
- A skill provides proven patterns for complex tasks

**Important Rules:**
- Do NOT use `read_file` to load a skill's `SKILL.md` for execution. Use `load_skill`.
- For very large skills, complete the loading flow with `get_skill_sections` before acting.
- Use normal file tools for supporting files or direct skill edits only when the task is about modifying files, not loading skill instructions.
- If a skill is relevant, load it first and then act.

**Executing Skill Scripts:**
Skills may contain Python scripts or other executable files. Always use absolute paths returned by `load_skill`.

**Example Workflow:**

User: "Can you research the latest developments in quantum computing?"

1. Check available skills -> See "web-research" skill with its path
2. Call `load_skill(skill_name="web-research")`
3. Follow the skill's research workflow (search -> organize -> synthesize)
4. Use any helper scripts with absolute paths

Remember: Skills make you more capable and consistent. When in doubt, check if a skill exists for the task!
"""


class SkillsMiddleware(AgentMiddleware[SkillsState, ContextT, ResponseT]):
    """Middleware for loading and exposing agent skills to the system prompt.

    Loads skills from backend sources and injects them into the system prompt
    using progressive disclosure (metadata first, full content on demand).

    Skills are loaded in source order with later sources overriding
    earlier ones.

    Example:
        ```python
        from deepagents.backends.filesystem import FilesystemBackend

        backend = FilesystemBackend(root_dir="/path/to/skills")
        middleware = SkillsMiddleware(
            backend=backend,
            sources=[
                "/path/to/skills/user/",
                "/path/to/skills/project/",
            ],
        )
        ```

    Args:
        backend: Backend instance for file operations
        sources: List of skill source paths.

            Source names are derived from the last path component.
    """

    state_schema = SkillsState

    def __init__(self, *, backend: BACKEND_TYPES, sources: list[str]) -> None:
        """Initialize the skills middleware.

        Args:
            backend: Backend instance (e.g. ``StateBackend()``).
            sources: List of skill source paths (e.g.,
                `['/skills/user/', '/skills/project/']`).
        """
        self._backend = backend
        self.sources = sources
        self.system_prompt_template = SKILLS_SYSTEM_PROMPT
        self._skill_content_cache: dict[tuple[str, str], LoadSkillPayload] = {}
        self._cached_backend: BackendProtocol | None = None
        self._cached_skills_metadata: list[SkillMetadata] = []
        self.tools = [
            self._create_load_skill_tool(),
            self._create_get_skill_sections_tool(),
        ]

    def _get_backend(self, state: SkillsState, runtime: Runtime, config: RunnableConfig) -> BackendProtocol:
        """Resolve backend from instance or factory.

        Args:
            state: Current agent state.
            runtime: Runtime context for factory functions.
            config: Runnable config to pass to backend factory.

        Returns:
            Resolved backend instance
        """
        if callable(self._backend):
            # Construct an artificial tool runtime to resolve backend factory
            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            backend = self._backend(tool_runtime)  # ty: ignore[call-top-callable, invalid-argument-type]
            if backend is None:
                msg = "SkillsMiddleware requires a valid backend instance"
                raise AssertionError(msg)
            return backend

        return self._backend

    def _format_skills_locations(self) -> str:
        """Format skills locations for display in system prompt."""
        locations = []

        for i, source_path in enumerate(self.sources):
            name = PurePosixPath(source_path.rstrip("/")).name.capitalize()
            suffix = " (higher priority)" if i == len(self.sources) - 1 else ""
            locations.append(f"**{name} Skills**: `{source_path}`{suffix}")

        return "\n".join(locations)

    def _format_skills_list(self, skills: list[SkillMetadata]) -> str:
        """Format skills metadata for display in system prompt."""
        if not skills:
            paths = [f"{source_path}" for source_path in self.sources]
            return f"(No skills available yet. You can create skills in {' or '.join(paths)})"

        lines = []
        for skill in skills:
            annotations = _format_skill_annotations(skill)
            desc_line = f"- **{skill['name']}**: {skill['description']}"
            if annotations:
                desc_line += f" ({annotations})"
            lines.append(desc_line)
            if skill["allowed_tools"]:
                lines.append(f"  -> Allowed tools: {', '.join(skill['allowed_tools'])}")
            lines.append(f"  -> Call `load_skill(skill_name=\"{skill['name']}\")` to load full instructions")
            lines.append(f"  -> Managed file: `{skill['path']}`")

        return "\n".join(lines)

    def _resolve_tool_backend(self, runtime: ToolRuntime | None) -> BackendProtocol:
        """Resolve a backend for tool execution, even if runtime injection fails."""
        if runtime is not None:
            backend = self._get_backend(runtime.state, runtime, runtime.config)
            self._cached_backend = backend
            return backend

        if self._cached_backend is not None:
            return self._cached_backend

        if callable(self._backend):
            msg = "SkillsMiddleware requires ToolRuntime for backend factories"
            raise ValueError(msg)

        self._cached_backend = self._backend
        return self._backend

    def _resolve_tool_skills_metadata(self, runtime: ToolRuntime | None) -> list[SkillMetadata]:
        """Resolve skill metadata for tool execution, with a middleware-level fallback."""
        if runtime is not None:
            skills_metadata = runtime.state.get("skills_metadata", [])
            if skills_metadata:
                self._cached_skills_metadata = skills_metadata
            return skills_metadata

        return self._cached_skills_metadata

    @staticmethod
    def _find_skill_by_name(skills_metadata: list[SkillMetadata], skill_name: str) -> SkillMetadata | None:
        """Return the metadata for a named skill."""
        return next((skill for skill in skills_metadata if skill["name"] == skill_name), None)

    def _cache_key(self, skill: SkillMetadata, raw_content: str) -> tuple[str, str]:
        """Build a cache key for a fully loaded skill."""
        return (skill["path"], hashlib.sha256(raw_content.encode("utf-8")).hexdigest())

    def _download_skill_body(
        self,
        backend: BackendProtocol,
        skill: SkillMetadata,
    ) -> tuple[str, str] | str:
        """Download and decode a skill body."""
        response = backend.download_files([skill["path"]])[0]
        if response.error or response.content is None:
            return f"Error: could not load skill '{skill['name']}' from {skill['path']}: {response.error or 'no content'}"

        try:
            raw_content = response.content.decode("utf-8")
        except UnicodeDecodeError as exc:
            return f"Error: could not decode skill '{skill['name']}' as UTF-8: {exc}"

        _frontmatter, body = _split_frontmatter_and_body(raw_content)
        return raw_content, body

    async def _adownload_skill_body(
        self,
        backend: BackendProtocol,
        skill: SkillMetadata,
    ) -> tuple[str, str] | str:
        """Async version of `_download_skill_body`."""
        response = (await backend.adownload_files([skill["path"]]))[0]
        if response.error or response.content is None:
            return f"Error: could not load skill '{skill['name']}' from {skill['path']}: {response.error or 'no content'}"

        try:
            raw_content = response.content.decode("utf-8")
        except UnicodeDecodeError as exc:
            return f"Error: could not decode skill '{skill['name']}' as UTF-8: {exc}"

        _frontmatter, body = _split_frontmatter_and_body(raw_content)
        return raw_content, body

    def _load_skill_payload(
        self,
        backend: BackendProtocol,
        skills_metadata: list[SkillMetadata],
        skill_name: str,
    ) -> LoadSkillPayload | str:
        """Synchronously load a full skill payload."""
        skill = self._find_skill_by_name(skills_metadata, skill_name)
        if skill is None:
            return f"Error: unknown skill '{skill_name}'. Choose one of the advertised skill names."

        downloaded = self._download_skill_body(backend, skill)
        if isinstance(downloaded, str):
            return downloaded
        raw_content, _body = downloaded

        cache_key = self._cache_key(skill, raw_content)
        cached = self._skill_content_cache.get(cache_key)
        if cached is not None:
            return cached

        payload = _build_load_skill_payload(
            skill=skill,
            raw_content=raw_content,
            supporting_files_manifest=_build_supporting_files_manifest(
                backend,
                str(PurePosixPath(skill["path"]).parent),
            ),
        )
        self._skill_content_cache[cache_key] = payload
        return payload

    async def _aload_skill_payload(
        self,
        backend: BackendProtocol,
        skills_metadata: list[SkillMetadata],
        skill_name: str,
    ) -> LoadSkillPayload | str:
        """Asynchronously load a full skill payload."""
        skill = self._find_skill_by_name(skills_metadata, skill_name)
        if skill is None:
            return f"Error: unknown skill '{skill_name}'. Choose one of the advertised skill names."

        downloaded = await self._adownload_skill_body(backend, skill)
        if isinstance(downloaded, str):
            return downloaded
        raw_content, _body = downloaded

        cache_key = self._cache_key(skill, raw_content)
        cached = self._skill_content_cache.get(cache_key)
        if cached is not None:
            return cached

        payload = _build_load_skill_payload(
            skill=skill,
            raw_content=raw_content,
            supporting_files_manifest=await _abuild_supporting_files_manifest(
                backend,
                str(PurePosixPath(skill["path"]).parent),
            ),
        )
        self._skill_content_cache[cache_key] = payload
        return payload

    def _create_load_skill_tool(self) -> BaseTool:
        """Create the `load_skill` tool."""

        def sync_load_skill(
            skill_name: str,
            runtime: ToolRuntime = None,  # type: ignore[assignment]
            purpose: str | None = None,  # noqa: ARG001  # intentional context for the model
        ) -> LoadSkillPayload | str:
            """Synchronously load a skill's full verified instructions."""
            backend = self._resolve_tool_backend(runtime)
            skills_metadata = self._resolve_tool_skills_metadata(runtime)
            return self._load_skill_payload(backend, skills_metadata, skill_name)

        async def async_load_skill(
            skill_name: str,
            runtime: ToolRuntime = None,  # type: ignore[assignment]
            purpose: str | None = None,  # noqa: ARG001  # intentional context for the model
        ) -> LoadSkillPayload | str:
            """Asynchronously load a skill's full verified instructions."""
            backend = self._resolve_tool_backend(runtime)
            skills_metadata = self._resolve_tool_skills_metadata(runtime)
            return await self._aload_skill_payload(backend, skills_metadata, skill_name)

        return StructuredTool.from_function(
            name="load_skill",
            description="Load the full verified instructions for a skill. Use this tool instead of read_file when a skill applies.",
            func=sync_load_skill,
            coroutine=async_load_skill,
            infer_schema=False,
            args_schema=LoadSkillSchema,
        )

    def _get_skill_sections_payload(
        self,
        backend: BackendProtocol,
        skills_metadata: list[SkillMetadata],
        skill_name: str,
        section_ids: list[str],
    ) -> GetSkillSectionsPayload | str:
        """Synchronously load the requested skill sections."""
        skill = self._find_skill_by_name(skills_metadata, skill_name)
        if skill is None:
            return f"Error: unknown skill '{skill_name}'. Choose one of the advertised skill names."

        downloaded = self._download_skill_body(backend, skill)
        if isinstance(downloaded, str):
            return downloaded
        raw_content, body = downloaded
        section_manifest = _build_section_manifest(body)
        loaded_sections, missing_section_ids = _extract_section_contents(body, section_manifest, section_ids)
        return GetSkillSectionsPayload(
            skill_name=skill["name"],
            source_path=skill["path"],
            root_path=str(PurePosixPath(skill["path"]).parent),
            loaded_sections=loaded_sections,
            loaded_section_ids=[section["section_id"] for section in loaded_sections],
            missing_section_ids=missing_section_ids,
            is_complete=not missing_section_ids,
            content_sha256=hashlib.sha256(raw_content.encode("utf-8")).hexdigest(),
        )

    async def _aget_skill_sections_payload(
        self,
        backend: BackendProtocol,
        skills_metadata: list[SkillMetadata],
        skill_name: str,
        section_ids: list[str],
    ) -> GetSkillSectionsPayload | str:
        """Async version of `_get_skill_sections_payload`."""
        skill = self._find_skill_by_name(skills_metadata, skill_name)
        if skill is None:
            return f"Error: unknown skill '{skill_name}'. Choose one of the advertised skill names."

        downloaded = await self._adownload_skill_body(backend, skill)
        if isinstance(downloaded, str):
            return downloaded
        raw_content, body = downloaded
        section_manifest = _build_section_manifest(body)
        loaded_sections, missing_section_ids = _extract_section_contents(body, section_manifest, section_ids)
        return GetSkillSectionsPayload(
            skill_name=skill["name"],
            source_path=skill["path"],
            root_path=str(PurePosixPath(skill["path"]).parent),
            loaded_sections=loaded_sections,
            loaded_section_ids=[section["section_id"] for section in loaded_sections],
            missing_section_ids=missing_section_ids,
            is_complete=not missing_section_ids,
            content_sha256=hashlib.sha256(raw_content.encode("utf-8")).hexdigest(),
        )

    def _create_get_skill_sections_tool(self) -> BaseTool:
        """Create the `get_skill_sections` tool."""

        def sync_get_skill_sections(
            skill_name: str,
            section_ids: list[str],
            runtime: ToolRuntime = None,  # type: ignore[assignment]
        ) -> GetSkillSectionsPayload | str:
            """Synchronously load specific sections from a skill."""
            backend = self._resolve_tool_backend(runtime)
            skills_metadata = self._resolve_tool_skills_metadata(runtime)
            return self._get_skill_sections_payload(
                backend,
                skills_metadata,
                skill_name,
                section_ids,
            )

        async def async_get_skill_sections(
            skill_name: str,
            section_ids: list[str],
            runtime: ToolRuntime = None,  # type: ignore[assignment]
        ) -> GetSkillSectionsPayload | str:
            """Asynchronously load specific sections from a skill."""
            backend = self._resolve_tool_backend(runtime)
            skills_metadata = self._resolve_tool_skills_metadata(runtime)
            return await self._aget_skill_sections_payload(
                backend,
                skills_metadata,
                skill_name,
                section_ids,
            )

        return StructuredTool.from_function(
            name="get_skill_sections",
            description="Load specific sections from a large skill after `load_skill` indicates sectional loading is required.",
            func=sync_get_skill_sections,
            coroutine=async_get_skill_sections,
            infer_schema=False,
            args_schema=GetSkillSectionsSchema,
        )

    def modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Inject skills documentation into a model request's system message.

        Args:
            request: Model request to modify

        Returns:
            New model request with skills documentation injected into system message
        """
        skills_metadata = request.state.get("skills_metadata", [])
        skills_locations = self._format_skills_locations()
        skills_list = self._format_skills_list(skills_metadata)

        skills_section = self.system_prompt_template.format(
            skills_locations=skills_locations,
            skills_list=skills_list,
        )

        new_system_message = append_to_system_message(request.system_message, skills_section)

        return request.override(system_message=new_system_message)

    def before_agent(self, state: SkillsState, runtime: Runtime, config: RunnableConfig) -> SkillsStateUpdate | None:  # ty: ignore[invalid-method-override]
        """Load skills metadata before agent execution (synchronous).

        Loads skills once per session from all configured sources. If
        `skills_metadata` is already present in state (from a prior turn or
        checkpointed session), the load is skipped and `None` is returned.

        Skills are loaded in source order with later sources overriding
        earlier ones if they contain skills with the same name (last one wins).

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with `skills_metadata` populated, or `None` if already present.
        """
        # Skip if skills_metadata is already present in state (even if empty)
        if "skills_metadata" in state:
            return None

        # Resolve backend (supports both direct instances and factory functions)
        backend = self._get_backend(state, runtime, config)
        self._cached_backend = backend
        all_skills: dict[str, SkillMetadata] = {}

        # Load skills from each source in order
        # Later sources override earlier ones (last one wins)
        for source_path in self.sources:
            source_skills = _list_skills(backend, source_path)
            for skill in source_skills:
                all_skills[skill["name"]] = skill

        skills = list(all_skills.values())
        self._cached_skills_metadata = skills
        return SkillsStateUpdate(skills_metadata=skills)

    async def abefore_agent(self, state: SkillsState, runtime: Runtime, config: RunnableConfig) -> SkillsStateUpdate | None:  # ty: ignore[invalid-method-override]
        """Load skills metadata before agent execution (async).

        Loads skills once per session from all configured sources. If
        `skills_metadata` is already present in state (from a prior turn or
        checkpointed session), the load is skipped and `None` is returned.

        Skills are loaded in source order with later sources overriding
        earlier ones if they contain skills with the same name (last one wins).

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with `skills_metadata` populated, or `None` if already present.
        """
        # Skip if skills_metadata is already present in state (even if empty)
        if "skills_metadata" in state:
            return None

        # Resolve backend (supports both direct instances and factory functions)
        backend = self._get_backend(state, runtime, config)
        self._cached_backend = backend
        all_skills: dict[str, SkillMetadata] = {}

        # Load skills from each source in order
        # Later sources override earlier ones (last one wins)
        for source_path in self.sources:
            source_skills = await _alist_skills(backend, source_path)
            for skill in source_skills:
                all_skills[skill["name"]] = skill

        skills = list(all_skills.values())
        self._cached_skills_metadata = skills
        return SkillsStateUpdate(skills_metadata=skills)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Inject skills documentation into the system prompt.

        Args:
            request: Model request being processed
            handler: Handler function to call with modified request

        Returns:
            Model response from handler
        """
        modified_request = self.modify_request(request)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Inject skills documentation into the system prompt (async version).

        Args:
            request: Model request being processed
            handler: Async handler function to call with modified request

        Returns:
            Model response from handler
        """
        modified_request = self.modify_request(request)
        return await handler(modified_request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Redirect direct `read_file` access for skill bodies to `load_skill`."""
        if request.tool_call["name"] != "read_file":
            return handler(request)

        args = request.tool_call.get("args", {})
        file_path = args.get("file_path") if isinstance(args, dict) else None
        if not isinstance(file_path, str):
            return handler(request)

        skills_metadata = request.runtime.state.get("skills_metadata", [])
        skill = next((item for item in skills_metadata if item["path"] == file_path), None)
        if skill is None:
            return handler(request)

        return ToolMessage(
            tool_call_id=request.tool_call["id"],
            content=(
                f"Error: `{file_path}` is a managed skill file. "
                f"Use load_skill(skill_name='{skill['name']}') instead of read_file "
                "when you need the skill's execution instructions."
            ),
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Async version of `wrap_tool_call`."""
        if request.tool_call["name"] != "read_file":
            return await handler(request)

        args = request.tool_call.get("args", {})
        file_path = args.get("file_path") if isinstance(args, dict) else None
        if not isinstance(file_path, str):
            return await handler(request)

        skills_metadata = request.runtime.state.get("skills_metadata", [])
        skill = next((item for item in skills_metadata if item["path"] == file_path), None)
        if skill is None:
            return await handler(request)

        return ToolMessage(
            tool_call_id=request.tool_call["id"],
            content=(
                f"Error: `{file_path}` is a managed skill file. "
                f"Use load_skill(skill_name='{skill['name']}') instead of read_file "
                "when you need the skill's execution instructions."
            ),
        )


__all__ = ["SkillMetadata", "SkillsMiddleware"]
