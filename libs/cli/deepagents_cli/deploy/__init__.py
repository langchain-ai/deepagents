"""Deploy commands for the Managed Deep Agents (`/v1/deepagents/*`) surface."""

from deepagents_cli.deploy.commands import (
    execute_agents_command,
    execute_deploy_command,
    execute_init_command,
    execute_mcp_servers_command,
    setup_deploy_parsers,
)

__all__ = [
    "execute_agents_command",
    "execute_deploy_command",
    "execute_init_command",
    "execute_mcp_servers_command",
    "setup_deploy_parsers",
]
