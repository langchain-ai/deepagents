"""Provider-specific MCP OAuth dispatch.

`resolve_provider(url)` returns the registered policy whose `matches`
predicate fires for `url`, with `GenericProvider` as the fallback.
"""

from deepagents_cli.mcp_providers._registry import resolve_provider
from deepagents_cli.mcp_providers.base import (
    GenericProvider,
    LoginResult,
    OAuthProvider,
)
from deepagents_cli.mcp_providers.github import GitHubProvider
from deepagents_cli.mcp_providers.slack import SlackProvider

__all__ = [
    "GenericProvider",
    "GitHubProvider",
    "LoginResult",
    "OAuthProvider",
    "SlackProvider",
    "resolve_provider",
]
