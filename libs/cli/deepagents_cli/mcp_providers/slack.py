"""Slack-hosted MCP OAuth provider.

Slack's hosted MCP endpoint uses the Authorization Code flow with a
hardcoded public client ID and the loopback redirect URI. The user
copy-pastes the redirected URL back into the CLI rather than running a
local server, and an optional `team` query parameter selects the
workspace to install the app into.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

from mcp.shared.auth import (
    AnyUrl,
    OAuthClientInformationFull,
    OAuthClientMetadata,
)

from deepagents_cli.mcp_providers.base import LoginResult, OAuthProvider

if TYPE_CHECKING:
    from deepagents_cli.mcp_auth import FileTokenStorage


# Public OAuth client ID — safe to check in. No secret is associated;
# Slack treats this as a browser-style public client where the security
# boundary is the redirect URI rather than client secrecy.
_SLACK_MCP_CLIENT_ID = "4518649543379.10944517634130"
"""Public OAuth client ID registered with Slack for the hosted MCP endpoint."""

_SLACK_REDIRECT_URI = "https://localhost"
"""Loopback redirect URI Slack hands the authorization code back to; the user
copy-pastes the resulting URL into the CLI rather than running a local server."""


def _is_slack_mcp_url(url: str) -> bool:
    """Return `True` when `url` points at a Slack-hosted MCP endpoint."""
    host = urlparse(url).hostname or ""
    return host == "slack.com" or host.endswith(".slack.com")


async def _prompt_slack_team() -> str | None:
    """Interactively ask the user which Slack workspace to install into.

    Runs the blocking `input()` in a worker thread so `login()` stays safe
    to await from an already-running event loop (Textual worker, IPython).

    Returns:
        The entered Slack team ID, or `None` if the prompt was left blank.
    """
    import asyncio

    raw = await asyncio.to_thread(
        input,
        "Slack team ID to install the app into "
        "(e.g. T01234567 — leave blank to pick on Slack's page): ",
    )
    stripped = raw.strip()
    return stripped or None


async def _preseed_slack_client_info(storage: FileTokenStorage) -> None:
    """Write the hardcoded Slack `client_info` to `storage` if not already set."""
    existing = await storage.get_client_info()
    if existing is not None and existing.client_id == _SLACK_MCP_CLIENT_ID:
        return
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id=_SLACK_MCP_CLIENT_ID,
            redirect_uris=[AnyUrl(_SLACK_REDIRECT_URI)],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",  # noqa: S106
        )
    )


class SlackProvider(OAuthProvider):
    """Slack-hosted MCP: paste-back Authorization Code with a public client."""

    def matches(self, server_url: str) -> bool:  # noqa: PLR6301  # subclass hook
        """Match `slack.com` and any `*.slack.com` subdomain.

        Args:
            server_url: Remote MCP endpoint URL.

        Returns:
            `True` when `server_url`'s host is Slack.
        """
        return _is_slack_mcp_url(server_url)

    def client_metadata(self) -> OAuthClientMetadata:  # noqa: PLR6301  # subclass hook
        """Return public-client metadata with the Slack loopback redirect URI.

        Returns:
            Metadata configured for Slack's public OAuth client (no token secret).
        """
        return OAuthClientMetadata(
            client_name="deepagents-cli",
            redirect_uris=[AnyUrl(_SLACK_REDIRECT_URI)],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",  # noqa: S106
        )

    async def run_login(  # noqa: PLR6301  # subclass hook
        self,
        *,
        server_name: str,
        server_url: str,
        storage: FileTokenStorage,
    ) -> LoginResult:
        """Preseed client info and optionally thread the team ID into auth URL.

        Args:
            server_name: MCP server name (unused).
            server_url: Remote MCP endpoint URL (unused).
            storage: File-backed token storage for this server identity.

        Returns:
            A `LoginResult` carrying the optional `team=<id>` extra param
            so the Slack authorize URL installs into the chosen workspace.
        """
        del server_name, server_url
        await _preseed_slack_client_info(storage)
        team_id = await _prompt_slack_team()
        extras = {"team": team_id} if team_id else {}
        return LoginResult(extra_auth_params=extras)
