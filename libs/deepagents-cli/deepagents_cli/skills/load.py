"""Skill loader for parsing and loading agent skills from SKILL.md files.

This module implements Anthropic's agent skills pattern with YAML frontmatter parsing.
Each skill is a directory containing a SKILL.md file with:
- YAML frontmatter (name, description required)
- Markdown instructions for the agent
- Optional supporting files (scripts, configs, etc.)

Example SKILL.md structure:
```markdown
---
name: web-research
description: Structured approach to conducting thorough web research
---

# Web Research Skill

## When to Use
- User asks you to research a topic
...
```
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, NotRequired, TypedDict

import yaml

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum size for SKILL.md files (10MB)
MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024

# Agent Skills spec constraints (https://agentskills.io/specification)
MAX_SKILL_NAME_LENGTH = 64
MAX_SKILL_DESCRIPTION_LENGTH = 1024


class SkillMetadata(TypedDict):
    """Metadata for a skill per Agent Skills spec (https://agentskills.io/specification)."""

    name: str
    """Name of the skill (max 64 chars, lowercase alphanumeric and hyphens)."""

    description: str
    """Description of what the skill does (max 1024 chars)."""

    path: str
    """Path to the SKILL.md file."""

    source: str
    """Source of the skill ('user' or 'project')."""

    # Optional fields per Agent Skills spec
    license: NotRequired[str | None]
    """License name or reference to bundled license file."""

    compatibility: NotRequired[str | None]
    """Environment requirements (max 500 chars)."""

    metadata: NotRequired[dict[str, str] | None]
    """Arbitrary key-value mapping for additional metadata."""

    allowed_tools: NotRequired[str | None]
    """Space-delimited list of pre-approved tools."""


def _is_safe_path(path: Path, base_dir: Path) -> bool:
    """Check if a path is safely contained within base_dir.

    This prevents directory traversal attacks via symlinks or path manipulation.
    The function resolves both paths to their canonical form (following symlinks)
    and verifies that the target path is within the base directory.

    Args:
        path: The path to validate
        base_dir: The base directory that should contain the path

    Returns:
        True if the path is safely within base_dir, False otherwise

    Example:
        >>> base = Path("/home/user/.deepagents/skills")
        >>> safe = Path("/home/user/.deepagents/skills/web-research/SKILL.md")
        >>> unsafe = Path("/home/user/.deepagents/skills/../../.ssh/id_rsa")
        >>> _is_safe_path(safe, base)
        True
        >>> _is_safe_path(unsafe, base)
        False
    """
    try:
        # Resolve both paths to their canonical form (follows symlinks)
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()

        # Check if the resolved path is within the base directory
        # This catches symlinks that point outside the base directory
        resolved_path.relative_to(resolved_base)
        return True
    except ValueError:
        # Path is not relative to base_dir (outside the directory)
        return False
    except (OSError, RuntimeError):
        # Error resolving paths (e.g., circular symlinks, too many levels)
        return False


def _validate_skill_name(name: str, directory_name: str) -> tuple[bool, str]:
    """Validate skill name per Agent Skills spec.

    Requirements:
    - Max 64 characters
    - Lowercase alphanumeric and hyphens only (a-z, 0-9, -)
    - Cannot start or end with hyphen
    - No consecutive hyphens
    - Must match parent directory name

    Args:
        name: The skill name from YAML frontmatter.
        directory_name: The parent directory name.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    if not name:
        return False, "name is required"
    if len(name) > MAX_SKILL_NAME_LENGTH:
        return False, "name exceeds 64 characters"
    # Pattern: lowercase alphanumeric, single hyphens between segments, no start/end hyphen
    if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", name):
        return False, "name must be lowercase alphanumeric with single hyphens only"
    if name != directory_name:
        return False, f"name '{name}' must match directory name '{directory_name}'"
    return True, ""


def _parse_skill_metadata(skill_md_path: Path, source: str) -> SkillMetadata | None:
    """Parse YAML frontmatter from a SKILL.md file per Agent Skills spec.

    Args:
        skill_md_path: Path to the SKILL.md file.
        source: Source of the skill ('user' or 'project').

    Returns:
        SkillMetadata with all fields, or None if parsing fails.
    """
    try:
        # Security: Check file size to prevent DoS attacks
        file_size = skill_md_path.stat().st_size
        if file_size > MAX_SKILL_FILE_SIZE:
            logger.warning("Skipping %s: file too large (%d bytes)", skill_md_path, file_size)
            return None

        content = skill_md_path.read_text(encoding="utf-8")

        # Match YAML frontmatter between --- delimiters
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if not match:
            logger.warning("Skipping %s: no valid YAML frontmatter found", skill_md_path)
            return None

        frontmatter_str = match.group(1)

        # Parse YAML using safe_load for proper nested structure support
        try:
            frontmatter_data = yaml.safe_load(frontmatter_str)
        except yaml.YAMLError as e:
            logger.warning("Invalid YAML in %s: %s", skill_md_path, e)
            return None

        if not isinstance(frontmatter_data, dict):
            logger.warning("Skipping %s: frontmatter is not a mapping", skill_md_path)
            return None

        # Validate required fields
        name = frontmatter_data.get("name")
        description = frontmatter_data.get("description")

        if not name or not description:
            logger.warning("Skipping %s: missing required 'name' or 'description'", skill_md_path)
            return None

        # Validate name format per spec (warn but still load for backwards compatibility)
        directory_name = skill_md_path.parent.name
        is_valid, error = _validate_skill_name(str(name), directory_name)
        if not is_valid:
            logger.warning(
                "Skill '%s' in %s does not follow Agent Skills spec: %s. "
                "Consider renaming to be spec-compliant.",
                name,
                skill_md_path,
                error,
            )

        # Validate description length (spec: max 1024 chars)
        description_str = str(description)
        if len(description_str) > MAX_SKILL_DESCRIPTION_LENGTH:
            logger.warning(
                "Description exceeds %d chars in %s, truncating",
                MAX_SKILL_DESCRIPTION_LENGTH,
                skill_md_path,
            )
            description_str = description_str[:MAX_SKILL_DESCRIPTION_LENGTH]

        return SkillMetadata(
            name=str(name),
            description=description_str,
            path=str(skill_md_path),
            source=source,
            license=frontmatter_data.get("license"),
            compatibility=frontmatter_data.get("compatibility"),
            metadata=frontmatter_data.get("metadata"),
            allowed_tools=frontmatter_data.get("allowed-tools"),
        )

    except (OSError, UnicodeDecodeError) as e:
        logger.warning("Error reading %s: %s", skill_md_path, e)
        return None


def _list_skills(skills_dir: Path, source: str) -> list[SkillMetadata]:
    """List all skills from a single skills directory (internal helper).

    Scans the skills directory for subdirectories containing SKILL.md files,
    parses YAML frontmatter, and returns skill metadata.

    Skills are organized as:
    skills/
    ├── skill-name/
    │   ├── SKILL.md        # Required: instructions with YAML frontmatter
    │   ├── script.py       # Optional: supporting files
    │   └── config.json     # Optional: supporting files

    Args:
        skills_dir: Path to the skills directory.
        source: Source of the skills ('user' or 'project').

    Returns:
        List of skill metadata dictionaries with name, description, path, and source.
    """
    # Check if skills directory exists
    skills_dir = skills_dir.expanduser()
    if not skills_dir.exists():
        return []

    # Resolve base directory to canonical path for security checks
    try:
        resolved_base = skills_dir.resolve()
    except (OSError, RuntimeError):
        # Can't resolve base directory, fail safe
        return []

    skills: list[SkillMetadata] = []

    # Iterate through subdirectories
    for skill_dir in skills_dir.iterdir():
        # Security: Catch symlinks pointing outside the skills directory
        if not _is_safe_path(skill_dir, resolved_base):
            continue

        if not skill_dir.is_dir():
            continue

        # Look for SKILL.md file
        skill_md_path = skill_dir / "SKILL.md"
        if not skill_md_path.exists():
            continue

        # Security: Validate SKILL.md path is safe before reading
        # This catches SKILL.md files that are symlinks pointing outside
        if not _is_safe_path(skill_md_path, resolved_base):
            continue

        # Parse metadata
        metadata = _parse_skill_metadata(skill_md_path, source=source)
        if metadata:
            skills.append(metadata)

    return skills


def list_skills(
    *, user_skills_dir: Path | None = None, project_skills_dir: Path | None = None
) -> list[SkillMetadata]:
    """List skills from user and/or project directories.

    When both directories are provided, project skills with the same name as
    user skills will override them.

    Args:
        user_skills_dir: Path to the user-level skills directory.
        project_skills_dir: Path to the project-level skills directory.

    Returns:
        Merged list of skill metadata from both sources, with project skills
        taking precedence over user skills when names conflict.
    """
    all_skills: dict[str, SkillMetadata] = {}

    # Load user skills first (foundation)
    if user_skills_dir:
        user_skills = _list_skills(user_skills_dir, source="user")
        for skill in user_skills:
            all_skills[skill["name"]] = skill

    # Load project skills second (override/augment)
    if project_skills_dir:
        project_skills = _list_skills(project_skills_dir, source="project")
        for skill in project_skills:
            # Project skills override user skills with the same name
            all_skills[skill["name"]] = skill

    return list(all_skills.values())
