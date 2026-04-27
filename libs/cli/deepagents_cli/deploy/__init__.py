"""Deploy commands for bundling and shipping deep agents.

Imports are intentionally limited to lightweight parser/config helpers so
`from deepagents_cli.deploy import setup_deploy_parsers` (used by `--help`)
does not pull in the heavy `deepagents` backend stack. Heavyweight symbols
such as `ContextHubBackend` are deliberately not re-exported here; import
them directly from their submodule when needed:

    from deepagents_cli.deploy.context_hub import ContextHubBackend
"""

from deepagents_cli.deploy.commands import (
    execute_deploy_command,
    execute_dev_command,
    execute_init_command,
    setup_deploy_parsers,
)
from deepagents_cli.deploy.config import SandboxProvider, SandboxScope

__all__ = [
    "SandboxProvider",
    "SandboxScope",
    "execute_deploy_command",
    "execute_dev_command",
    "execute_init_command",
    "setup_deploy_parsers",
]
