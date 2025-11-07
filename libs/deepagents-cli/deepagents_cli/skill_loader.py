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

import re
from pathlib import Path
from typing import TypedDict


class SkillMetadata(TypedDict):
    """Metadata for a skill."""

    name: str
    """Name of the skill."""

    description: str
    """Description of what the skill does."""

    path: str
    """Path to the SKILL.md file."""


class SkillLoader:
    """Loader for agent skills with YAML frontmatter parsing.

    Skills are organized as:
    skills/
    ├── skill-name/
    │   ├── SKILL.md        # Required: instructions with YAML frontmatter
    │   ├── script.py       # Optional: supporting files
    │   └── config.json     # Optional: supporting files

    Example:
        ```python
        from pathlib import Path

        skills_dir = Path.home() / ".deepagents" / "skills"
        loader = SkillLoader(skills_dir=skills_dir)
        skills = loader.list()
        for skill in skills:
            print(f"{skill['name']}: {skill['description']}")
        ```
    """

    def __init__(self, skills_dir: Path) -> None:
        """Initialize the skill loader.

        Args:
            skills_dir: Path to the skills directory.
        """
        self.skills_dir = skills_dir.expanduser()

    def _parse_skill_metadata(self, skill_md_path: Path) -> SkillMetadata | None:
        """Parse YAML frontmatter from a SKILL.md file.

        Args:
            skill_md_path: Path to the SKILL.md file.

        Returns:
            SkillMetadata with name, description, and path, or None if parsing fails.
        """
        try:
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
            )

        except (OSError, UnicodeDecodeError):
            # Silently skip malformed or inaccessible files
            return None

    def list(self) -> list[SkillMetadata]:
        """List all skills from the skills directory.

        Scans the skills directory for subdirectories containing SKILL.md files,
        parses YAML frontmatter, and returns skill metadata.

        Returns:
            List of skill metadata dictionaries with name, description, and path.
        """
        skills: list[SkillMetadata] = []

        # Check if skills directory exists
        if not self.skills_dir.exists():
            return skills

        # Iterate through subdirectories
        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            # Look for SKILL.md file
            skill_md_path = skill_dir / "SKILL.md"
            if not skill_md_path.exists():
                continue

            # Parse metadata
            metadata = self._parse_skill_metadata(skill_md_path)
            if metadata:
                skills.append(metadata)

        return skills

    def format_skills_for_system_message(self, skills: list[SkillMetadata]) -> str:
        """Format skills metadata for injection into system prompt.

        Creates a formatted list of skills with their descriptions and paths,
        following Anthropic's progressive disclosure pattern.

        Args:
            skills: List of skill metadata to format.

        Returns:
            Formatted string for system prompt.
        """
        if not skills:
            return "No skills available."

        lines = ["Available skills:"]
        for skill in skills:
            lines.append(f"- **{skill['name']}**: {skill['description']}")
            lines.append(f"  Read `/skills/{Path(skill['path']).parent.name}/SKILL.md` for details")

        return "\n".join(lines)


def load_skills(skills_dir: Path | None = None) -> tuple[list[SkillMetadata], str]:
    """Convenience function to load skills and format for system prompt.

    Args:
        skills_dir: Path to the skills directory. Defaults to ~/.deepagents/skills

    Returns:
        Tuple of (skills metadata list, formatted string for system prompt).
    """
    if skills_dir is None:
        skills_dir = Path.home() / ".deepagents" / "skills"
    loader = SkillLoader(skills_dir)
    skills = loader.list()
    formatted = loader.format_skills_for_system_message(skills)
    return skills, formatted
