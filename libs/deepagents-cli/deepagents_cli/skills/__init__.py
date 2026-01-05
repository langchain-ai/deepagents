"""Skills module for deepagents CLI.

Public API:
- execute_skills_command: Execute skills subcommands (list/create/info)
- setup_skills_parser: Setup argparse configuration for skills commands

Note: SkillsMiddleware is now provided by the SDK (deepagents.SkillsMiddleware).
The CLI uses the SDK version directly.
"""

from deepagents_cli.skills.commands import (
    execute_skills_command,
    setup_skills_parser,
)

__all__ = [
    "execute_skills_command",
    "setup_skills_parser",
]
