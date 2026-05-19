"""Slack-hosted MCP OAuth provider.

Slack's hosted MCP endpoint uses the Authorization Code flow with a
hardcoded public client ID and a static pre-registered redirect URI. The
user copy-pastes the redirected URL back into the app rather than running
a local server, and an optional `team` query parameter selects the
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

from deepagents_code.mcp_providers.base import LoginResult, OAuthProvider

if TYPE_CHECKING:
    from deepagents_code.mcp_auth import FileTokenStorage
    from deepagents_code.mcp_oauth_ui import OAuthInteraction


# Public OAuth client ID — safe to check in. No secret is associated;
# Slack treats this as a browser-style public client where the security
# boundary is the redirect URI rather than client secrecy.
_SLACK_MCP_CLIENT_ID = "4518649543379.10944517634130"
"""Public OAuth client ID registered with Slack for the hosted MCP endpoint."""

_SLACK_REDIRECT_URI = "https://localhost"
"""Static pre-registered redirect URI Slack hands the authorization code back
to; the user copy-pastes the resulting URL into the app rather than running a
local server."""


def _is_slack_mcp_url(url: str) -> bool:
    """Return `True` when `url` points at a Slack-hosted MCP endpoint."""
    host = urlparse(url).hostname or ""
    return host == "slack.com" or host.endswith(".slack.com")


async def _prompt_slack_team(ui: OAuthInteraction) -> str | None:
    """Interactively ask the user which Slack workspace to install into.

    Delegates to the supplied `OAuthInteraction` so the same flow drives
    either the CLI stdin/stdout prompt or a TUI input widget.

    Args:
        ui: Interaction surface to use.

    Returns:
        The entered Slack team ID, or `None` if the prompt was left blank.
    """
    return await ui.prompt_slack_team_id()


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

    def supports_loopback_callback(self) -> bool:  # noqa: PLR6301  # subclass hook
        """Return `False` because Slack uses a fixed pre-registered redirect URI.

        Returns:
            Always `False`.
        """
        return False

    def client_metadata(  # noqa: PLR6301  # subclass hook
        self, *, redirect_uri: str | None = None
    ) -> OAuthClientMetadata:
        """Return public-client metadata with Slack's static pre-registered URI.

        Args:
            redirect_uri: Ignored; Slack requires its static pre-registered URI.

        Returns:
            Metadata configured for Slack's public OAuth client (no token secret).
        """
        del redirect_uri
        return OAuthClientMetadata(
            client_name="deepagents-code",
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
        ui: OAuthInteraction,
    ) -> LoginResult:
        """Preseed client info and optionally thread the team ID into auth URL.

        Args:
            server_name: MCP server name (unused).
            server_url: Remote MCP endpoint URL (unused).
            storage: File-backed token storage for this server identity.
            ui: Interaction surface used to prompt for the Slack team ID.

        Returns:
            A `LoginResult` carrying the optional `team=<id>` extra param
            so the Slack authorize URL installs into the chosen workspace.
        """
        del server_name, server_url
        await _preseed_slack_client_info(storage)
        team_id = await _prompt_slack_team(ui)
        extras = {"team": team_id} if team_id else {}
        return LoginResult(extra_auth_params=extras)
