"""Skill loader - re-exports from core package.

This module is maintained for backward compatibility.
New code should import from deepagents.middleware.skills directly.

Example:
    # Preferred (new code)
    from deepagents.middleware.skills import list_skills, SkillMetadata

    # Deprecated (backward compatible)
    from deepagents_cli.skills.load import list_skills, SkillMetadata
"""

# Re-export from core package
from deepagents.middleware.skills import (
    MAX_SKILL_FILE_SIZE,
    SkillMetadata,
    list_skills,
)

# Keep internal helpers available for backward compatibility
from deepagents.middleware.skills import (
    _is_safe_path,
    _list_skills_from_dir,
    _parse_skill_metadata,
)

__all__ = [
    "MAX_SKILL_FILE_SIZE",
    "SkillMetadata",
    "_is_safe_path",
    "_list_skills_from_dir",
    "_parse_skill_metadata",
    "list_skills",
]
