"""Claude Skills discovery and parsing for deepagents CLI."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Skill:
    """Represents a Claude Skill with its metadata."""

    name: str
    description: str
    path: Path
    license: Optional[str] = None
    allowed_tools: Optional[list[str]] = None
    metadata: Optional[dict] = None


def parse_yaml_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown content.

    Args:
        content: Markdown content with optional YAML frontmatter

    Returns:
        Tuple of (frontmatter_dict, remaining_content)
    """
    # Check if content starts with ---
    if not content.startswith("---"):
        return {}, content

    # Find the closing ---
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if not match:
        return {}, content

    yaml_content = match.group(1)
    remaining_content = match.group(2)

    # Simple YAML parser for our needs (name, description, license, allowed-tools, metadata)
    frontmatter = {}
    current_key = None
    list_items = []

    for line in yaml_content.split("\n"):
        # Check for key: value
        if ":" in line and not line.strip().startswith("-"):
            if current_key and list_items:
                frontmatter[current_key] = list_items
                list_items = []

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if value:
                frontmatter[key] = value
                current_key = None
            else:
                # This might be a list or multi-line value
                current_key = key
        # Check for list item
        elif line.strip().startswith("-") and current_key:
            item = line.strip()[1:].strip()
            list_items.append(item)

    # Handle final list
    if current_key and list_items:
        frontmatter[current_key] = list_items

    return frontmatter, remaining_content


def find_all_skill_mds(skill_path: Path) -> list[Path]:
    """Find all SKILL.md files in the skill directory and its subdirectories.

    Args:
        skill_path: Path to the skill directory

    Returns:
        List of paths to all SKILL.md files found
    """
    skill_mds = []
    
    # First check if SKILL.md exists in the root
    direct_path = skill_path / "SKILL.md"
    if direct_path.exists():
        skill_mds.append(direct_path)
        return skill_mds  # If root has SKILL.md, don't search subdirectories
    
    # Recursively search subdirectories for SKILL.md
    try:
        for skill_md in skill_path.rglob("SKILL.md"):
            if skill_md.is_file():
                skill_mds.append(skill_md)
    except Exception:
        pass

    return skill_mds


def parse_skill_md(skill_md_path: Path) -> Optional[Skill]:
    """Parse a SKILL.md file and extract metadata.

    Args:
        skill_md_path: Path to the SKILL.md file

    Returns:
        Skill object if valid, None if invalid or missing required fields
    """
    if not skill_md_path.exists():
        return None

    try:
        content = skill_md_path.read_text(encoding="utf-8")
    except Exception:
        return None

    frontmatter, _ = parse_yaml_frontmatter(content)

    # Required fields
    name = frontmatter.get("name")
    description = frontmatter.get("description")

    if not name or not description:
        return None

    # Optional fields
    license_val = frontmatter.get("license")
    allowed_tools = frontmatter.get("allowed-tools")
    metadata = frontmatter.get("metadata")

    # Convert metadata string to dict if needed (simplified)
    if isinstance(metadata, str):
        metadata = {"raw": metadata}

    skill_path = skill_md_path.parent

    return Skill(
        name=name,
        description=description,
        path=skill_path,
        license=license_val,
        allowed_tools=allowed_tools if isinstance(allowed_tools, list) else None,
        metadata=metadata if isinstance(metadata, dict) else None,
    )


def discover_skills(skills_dir: Optional[Path] = None) -> list[Skill]:
    """Discover all Claude Skills in the skills directory.

    Args:
        skills_dir: Path to skills directory (defaults to ~/.claude/skills)

    Returns:
        List of discovered Skill objects
    """
    if skills_dir is None:
        skills_dir = Path.home() / ".claude" / "skills"

    if not skills_dir.exists() or not skills_dir.is_dir():
        return []

    skills = []

    try:
        for item in skills_dir.iterdir():
            if not item.is_dir():
                continue

            skill_md_paths = find_all_skill_mds(item)
            
            for skill_md_path in skill_md_paths:
                skill = parse_skill_md(skill_md_path)
                if skill:
                    skills.append(skill)
    except Exception:
        # Silently fail if we can't read the directory
        return []

    return sorted(skills, key=lambda s: s.name)


def format_skills_section(skills: list[Skill]) -> str:
    """Format the skills section for the system prompt.

    Args:
        skills: List of discovered skills

    Returns:
        Formatted markdown section describing available skills
    """
    if not skills:
        return ""

    lines = [
        "### Available Skills",
        "",
        "You have access to Claude Skills located in ~/.claude/skills/. When a task matches a skill's description, read the skill's SKILL.md file to get full instructions and follow them carefully.",
        "",
    ]

    for skill in skills:
        lines.append(f"**{skill.name}**: {skill.description}")
        lines.append(f"  - Location: `{skill.path}`")
        lines.append(f"  - Instructions: `{skill.path / 'SKILL.md'}`")
        lines.append("")

    return "\n".join(lines)
