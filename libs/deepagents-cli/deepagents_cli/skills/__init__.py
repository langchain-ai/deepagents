"""Skills module for deepagents CLI.

Public API:
- SkillsMiddleware: Re-exported from deepagents.middleware.skills
- execute_skills_command: Execute skills subcommands (list/create/info)
- setup_skills_parser: Setup argparse configuration for skills commands

All other components are internal implementation details.
"""

from deepagents.middleware.skills import SkillsMiddleware

from deepagents_cli.skills.commands import (
    execute_skills_command,
    setup_skills_parser,
)

__all__ = [
    "SkillsMiddleware",
    "execute_skills_command",
    "setup_skills_parser",
]
