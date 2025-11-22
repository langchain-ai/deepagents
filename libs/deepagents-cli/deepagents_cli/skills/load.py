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

import re
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

# Maximum size for SKILL.md files (10MB)
MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024


class SkillMetadata(TypedDict):
    """Metadata for a skill."""

    name: str
    """Name of the skill."""

    description: str
    """Description of what the skill does."""

    path: str
    """Path to the SKILL.md file."""

    source: str
    """Source of the skill ('user' or 'project')."""


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


def _parse_skill_metadata(skill_md_path: Path, source: str) -> SkillMetadata | None:
    """Parse YAML frontmatter from a SKILL.md file.

    Args:
        skill_md_path: Path to the SKILL.md file.
        source: Source of the skill ('user' or 'project').

    Returns:
        SkillMetadata with name, description, path, and source, or None if parsing fails.
    """
    try:
        # Security: Check file size to prevent DoS attacks
        file_size = skill_md_path.stat().st_size
        if file_size > MAX_SKILL_FILE_SIZE:
            # Silently skip files that are too large
            return None

        content = skill_md_path.read_text(encoding="utf-8")

        # Match YAML frontmatter between --- delimiters
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if not match:
            return None

        frontmatter = match.group(1)

        # Parse key-value pairs from YAML (simple parsing, no nested structures)
        metadata: dict[str, str] = {}
        for line in frontmatter.split("\n"):
            # Match "key: value" pattern
            kv_match = re.match(r"^(\w+):\s*(.+)$", line.strip())
            if kv_match:
                key, value = kv_match.groups()
                metadata[key] = value.strip()

        # Validate required fields
        if "name" not in metadata or "description" not in metadata:
            return None

        return SkillMetadata(
            name=metadata["name"],
            description=metadata["description"],
            path=str(skill_md_path),
            source=source,
        )

    except (OSError, UnicodeDecodeError):
        # Silently skip malformed or inaccessible files
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
