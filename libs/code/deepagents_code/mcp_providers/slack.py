"""Slack-hosted MCP OAuth provider.

Slack's hosted MCP endpoint uses the Authorization Code flow with a
hardcoded public client ID and a fixed pre-registered loopback redirect
URI (`http://localhost:3118/callback`). The local callback server listens
on that port so the browser redirect completes automatically. An optional
`team` query parameter selects the workspace to install the app into.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable
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


@runtime_checkable
class _SupportsSlackTeamPrompt(Protocol):
    """Optional interaction-surface capability: prompt for a Slack team ID.

    `OAuthInteraction` deliberately omits this method — only the CLI
    surface implements it. The TUI lets Slack's browser page handle
    workspace selection. Marked `runtime_checkable` so `isinstance`
    can replace `getattr`-style structural probes at the call site.
    """

    async def prompt_slack_team_id(self) -> str | None: ...


# Public OAuth client ID — safe to check in. No secret is associated;
# Slack treats this as a browser-style public client where the security
# boundary is the redirect URI rather than client secrecy.
_SLACK_MCP_CLIENT_ID = "4518649543379.10944517634130"
"""Public OAuth client ID registered with Slack for the hosted MCP endpoint."""

_SLACK_LOOPBACK_PORT = 3118
"""Fixed TCP port the local callback server binds to for Slack OAuth.

Slack validates the redirect URI against the app's registered allowlist, so
the port must be pre-registered in the Slack app dashboard. Only one Slack
login can proceed at a time per machine (port conflicts are surfaced as an
OSError from the loopback server's `start()` call).
"""

_SLACK_REDIRECT_URI = f"http://localhost:{_SLACK_LOOPBACK_PORT}/callback"
"""Pre-registered loopback redirect URI for the Slack MCP OAuth app."""


def _is_slack_mcp_url(url: str) -> bool:
    """Return `True` when `url` points at a Slack-hosted MCP endpoint."""
    host = urlparse(url).hostname or ""
    return host == "slack.com" or host.endswith(".slack.com")


async def _prompt_slack_team(ui: OAuthInteraction) -> str | None:
    """Return a Slack team ID when the interaction surface supports prompting.

    `prompt_slack_team_id` is **not** a member of the `OAuthInteraction`
    Protocol — only the CLI surface implements it. The TUI omits it so
    Slack's browser page handles workspace selection instead. Detection
    uses `isinstance` against the `runtime_checkable`
    `_SupportsSlackTeamPrompt` capability protocol.

    Args:
        ui: Interaction surface to use.

    Returns:
        The entered Slack team ID, or `None` when the surface lacks the
            optional prompt method or the user declined to specify one.
    """
    if not isinstance(ui, _SupportsSlackTeamPrompt):
        return None
    return await ui.prompt_slack_team_id()


async def _preseed_slack_client_info(storage: FileTokenStorage) -> None:
    """Write the hardcoded Slack `client_info` to `storage` if not current."""
    existing = await storage.get_client_info()
    redirect_uris = existing.redirect_uris if existing is not None else None
    current_redirect = str(redirect_uris[0]) if redirect_uris else None
    if (
        existing is not None
        and existing.client_id == _SLACK_MCP_CLIENT_ID
        and current_redirect == _SLACK_REDIRECT_URI
    ):
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
    """Slack-hosted MCP: loopback Authorization Code with a public client."""

    def matches(self, server_url: str) -> bool:  # noqa: PLR6301  # subclass hook
        """Match `slack.com` and any `*.slack.com` subdomain.

        Args:
            server_url: Remote MCP endpoint URL.

        Returns:
            `True` when `server_url`'s host is Slack.
        """
        return _is_slack_mcp_url(server_url)

    def loopback_port(self) -> int:  # noqa: PLR6301  # subclass hook
        """Return the fixed loopback port registered in the Slack OAuth app.

        Returns:
            `_SLACK_LOOPBACK_PORT` (3118).
        """
        return _SLACK_LOOPBACK_PORT

    def client_metadata(  # noqa: PLR6301  # subclass hook
        self, *, redirect_uri: str | None = None
    ) -> OAuthClientMetadata:
        """Return public-client metadata with Slack's pre-registered loopback URI.

        Args:
            redirect_uri: Ignored; Slack requires its pre-registered loopback URI.

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
