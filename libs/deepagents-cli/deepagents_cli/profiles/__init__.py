"""Profile management module for deepagents CLI.

This module provides commands for managing agent profiles:
- push: Upload a local directory as a profile
- pull: Download a profile to local storage
- profiles: List available profiles
"""

from .commands import (
    execute_profiles_command,
    execute_pull_command,
    execute_push_command,
    setup_profiles_parser,
    setup_pull_parser,
    setup_push_parser,
)

__all__ = [
    "execute_profiles_command",
    "execute_pull_command",
    "execute_push_command",
    "setup_profiles_parser",
    "setup_pull_parser",
    "setup_push_parser",
]
