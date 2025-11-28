"""Skills middleware - re-exports from core package.

This module is maintained for backward compatibility.
New code should import from deepagents.middleware.skills directly.

Example:
    # Preferred (new code)
    from deepagents.middleware.skills import SkillsMiddleware

    # Deprecated (backward compatible)
    from deepagents_cli.skills.middleware import SkillsMiddleware
"""

# Re-export from core package
from deepagents.middleware.skills import (
    SKILLS_SYSTEM_PROMPT,
    SkillsMiddleware,
    SkillsState,
    SkillsStateUpdate,
)

__all__ = [
    "SKILLS_SYSTEM_PROMPT",
    "SkillsMiddleware",
    "SkillsState",
    "SkillsStateUpdate",
]
