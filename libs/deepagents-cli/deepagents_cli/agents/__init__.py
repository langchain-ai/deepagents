"""Agent management module for deepagents CLI.

This module provides commands for managing agents:
- push: Upload an agent from ~/.deepagents/<name>/ to the remote filesystem
- pull: Download an agent to ~/.deepagents/<name>/
- agents: List available agents
"""

from .commands import (
    execute_agents_command,
    execute_pull_command,
    execute_push_command,
    setup_agents_parser,
    setup_pull_parser,
    setup_push_parser,
)

__all__ = [
    "execute_agents_command",
    "execute_pull_command",
    "execute_push_command",
    "setup_agents_parser",
    "setup_pull_parser",
    "setup_push_parser",
]
