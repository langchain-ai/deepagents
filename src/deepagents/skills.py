"""Helper utilities for working with skills."""

import re
from typing import TypedDict


class SkillFiles(TypedDict):
    """Dictionary mapping relative paths to content."""
    pass  # Key: path (str), Value: content (str | bytes)


class SkillDefinition(TypedDict):
    """Single skill for virtual filesystem."""
    name: str
    files: SkillFiles


class SkillMetadata(TypedDict):
    """Metadata extracted from a skill's YAML frontmatter."""
    name: str
    description: str
    version: str | None
    path: str
    source: str


def parse_skill_frontmatter(content: str) -> dict[str, str]:
    """Parse YAML frontmatter from SKILL.md content.
    
    Args:
        content: The content of a SKILL.md file.
        
    Returns:
        Dictionary of parsed YAML frontmatter fields.
        
    Example:
        >>> content = '''---
        ... name: slack-gif-creator
        ... description: Create animated GIFs
        ... version: 1.0.0
        ... ---
        ... # Rest of content'''
        >>> parse_skill_frontmatter(content)
        {'name': 'slack-gif-creator', 'description': 'Create animated GIFs', 'version': '1.0.0'}
    """
    # Match YAML frontmatter between --- markers
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not match:
        return {}
    
    frontmatter_text = match.group(1)
    result = {}
    
    # Parse simple key: value pairs
    for line in frontmatter_text.split('\n'):
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip()
    
    return result


def load_skills(skills_dict: dict[str, dict[str, str | bytes]]) -> list[SkillDefinition]:
    """Convert a dictionary of skills to list[SkillDefinition] format.
    
    Args:
        skills_dict: Dictionary where keys are skill names and values are
            dictionaries mapping file paths to content.
            
    Returns:
        List of SkillDefinition objects.
        
    Example:
        >>> skills = load_skills({
        ...     "api-wrapper": {
        ...         "SKILL.md": "---\\nname: api-wrapper\\n---\\n...",
        ...         "scripts/client.py": "import requests...",
        ...     }
        ... })
    """
    return [
        {"name": name, "files": files}
        for name, files in skills_dict.items()
    ]
