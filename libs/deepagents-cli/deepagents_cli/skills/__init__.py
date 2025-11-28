"""Skills module for deepagents CLI.

Public API:
- SkillsMiddleware: Middleware for integrating skills into agent execution
- SkillMetadata: TypedDict for skill metadata
- list_skills: Function to list skills from directories
- execute_skills_command: Execute skills subcommands (list/create/info)
- setup_skills_parser: Setup argparse configuration for skills commands

Note: SkillsMiddleware, SkillMetadata, and list_skills are re-exported from the
core deepagents package for backward compatibility. New code should import
directly from deepagents.middleware.skills.
"""

# Re-export from core package for backward compatibility
from deepagents.middleware.skills import SkillMetadata, SkillsMiddleware, list_skills

# CLI-specific commands
from deepagents_cli.skills.commands import (
    execute_skills_command,
    setup_skills_parser,
)

__all__ = [
    "SkillMetadata",
    "SkillsMiddleware",
    "execute_skills_command",
    "list_skills",
    "setup_skills_parser",
]
