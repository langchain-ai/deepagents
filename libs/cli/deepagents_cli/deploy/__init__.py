"""Deploy module for deepagents CLI.

Public API:
- execute_deploy_command: Run the deploy workflow
- setup_deploy_parser: Setup argparse configuration for the deploy subcommand

All other components are internal implementation details.
"""

from deepagents_cli.deploy.commands import (
    execute_deploy_command,
    setup_deploy_parser,
)

__all__ = [
    "execute_deploy_command",
    "setup_deploy_parser",
]
