"""GitHub-hosted MCP OAuth provider.

GitHub's remote MCP at `api.githubcopilot.com` authenticates via RFC
8628 Device Authorization Grant — the CLI runs the device flow,
persists the resulting token plus a stub client-info record, and skips
the standard Authorization Code handshake entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

from mcp.shared.auth import AnyUrl, OAuthClientInformationFull

from deepagents_cli.mcp_auth import _run_device_flow
from deepagents_cli.mcp_providers.base import LoginResult, OAuthProvider

if TYPE_CHECKING:
    from deepagents_cli.mcp_auth import FileTokenStorage


_GITHUB_MCP_CLIENT_ID = "Iv23libxz8qOApH0WQL3"
"""Public OAuth client ID for the GitHub App backing GitHub's remote MCP."""

_GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
"""GitHub Device Authorization Grant endpoint that issues the user/device code pair."""

_GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"  # noqa: S105
"""GitHub OAuth token endpoint polled while the user completes the device flow."""


def _is_github_mcp_url(url: str) -> bool:
    """Return `True` when `url` points at GitHub's remote MCP endpoint."""
    return (urlparse(url).hostname or "") == "api.githubcopilot.com"


async def _preseed_github_auth(storage: FileTokenStorage) -> None:
    """Run GitHub Device Flow and persist the token and stub client info."""
    token = await _run_device_flow(
        device_code_url=_GITHUB_DEVICE_CODE_URL,
        token_url=_GITHUB_TOKEN_URL,
        client_id=_GITHUB_MCP_CLIENT_ID,
    )
    await storage.set_tokens_and_client_info(
        token,
        OAuthClientInformationFull(
            client_id=_GITHUB_MCP_CLIENT_ID,
            redirect_uris=[AnyUrl("http://localhost/callback")],
            grant_types=["urn:ietf:params:oauth:grant-type:device_code"],
            response_types=["code"],
            token_endpoint_auth_method="none",  # noqa: S106
        ),
    )


class GitHubProvider(OAuthProvider):
    """GitHub-hosted MCP: RFC 8628 Device Authorization Grant."""

    def matches(self, server_url: str) -> bool:  # noqa: PLR6301  # subclass hook
        """Match `api.githubcopilot.com`.

        Args:
            server_url: Remote MCP endpoint URL.

        Returns:
            `True` when `server_url`'s host is GitHub's MCP endpoint.
        """
        return _is_github_mcp_url(server_url)

    async def run_login(  # noqa: PLR6301  # subclass hook
        self,
        *,
        server_name: str,
        server_url: str,
        storage: FileTokenStorage,
    ) -> LoginResult:
        """Run the device flow and short-circuit the Authorization Code handshake.

        Args:
            server_name: MCP server name (unused).
            server_url: Remote MCP endpoint URL (unused).
            storage: File-backed token storage for this server identity.

        Returns:
            `LoginResult(completed=True)` — tokens are already persisted so
            the caller must skip the handshake step.
        """
        del server_name, server_url
        await _preseed_github_auth(storage)
        return LoginResult(completed=True)
