"""Ordered provider registry for MCP OAuth dispatch.

`resolve_provider` walks `_REGISTRY` in order and returns the first
provider whose `matches(url)` is `True`. `GenericProvider` sits last so
spec-compliant servers always resolve to a usable policy.
"""

from __future__ import annotations

from deepagents_cli.mcp_providers.base import GenericProvider, OAuthProvider
from deepagents_cli.mcp_providers.github import GitHubProvider
from deepagents_cli.mcp_providers.slack import SlackProvider

_REGISTRY: tuple[OAuthProvider, ...] = (
    SlackProvider(),
    GitHubProvider(),
    GenericProvider(),
)
"""Ordered provider list; `GenericProvider` must stay last as the fallback."""


def resolve_provider(server_url: str) -> OAuthProvider:
    """Return the provider policy that owns `server_url`.

    Args:
        server_url: Remote MCP endpoint URL.

    Returns:
        The first matching `OAuthProvider`; falls back to `GenericProvider`.

    Raises:
        RuntimeError: If no provider matches (unreachable in practice since
            `GenericProvider.matches` always returns `True`).
    """
    for provider in _REGISTRY:
        if provider.matches(server_url):
            return provider
    msg = f"No MCP OAuth provider matched {server_url!r}"
    raise RuntimeError(msg)
