"""Skills module for deepagents CLI.

Public API:
- execute_skills_command: Execute skills subcommands (list/create/info)
- setup_skills_parser: Setup argparse configuration for skills commands

All other components are internal implementation details.
"""

from deepagents_cli.skills.commands import (
    execute_skills_command,
    setup_skills_parser,
)

__all__ = [
    "execute_skills_command",
    "setup_skills_parser",
]
