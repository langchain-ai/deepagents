"""Policy interface for provider-specific MCP OAuth quirks.

Each concrete provider module (e.g. `slack`, `github`) subclasses
`OAuthProvider` to encode its own URL match rule, client metadata, and
any pre-handshake login steps (preseeding client info, running a
device flow, prompting for workspace IDs). `mcp_auth` dispatches to
the first matching provider via `resolve_provider`, so adding a new
provider is one new module plus one registry entry — no edits to
`build_oauth_provider` or `login`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mcp.shared.auth import AnyUrl, OAuthClientMetadata

if TYPE_CHECKING:
    from deepagents_code.mcp_auth import FileTokenStorage
    from deepagents_code.mcp_oauth_ui import OAuthInteraction


@dataclass(frozen=True)
class LoginResult:
    """Outcome of a provider's pre-handshake `run_login` step."""

    completed: bool = False
    """`True` means tokens are persisted and `login` should skip the handshake."""

    extra_auth_params: dict[str, str] = field(default_factory=dict)
    """Extra query params to thread into the interactive auth URL."""


class OAuthProvider(ABC):
    """Base class for provider-specific OAuth dispatch.

    Subclasses override `matches` plus whichever of `client_metadata`
    and `run_login` they customize. The default implementations cover
    the spec-compliant Authorization Code + PKCE + Dynamic Client
    Registration path.
    """

    @abstractmethod
    def matches(self, server_url: str) -> bool:
        """Return `True` when this provider owns `server_url`."""

    def supports_loopback_callback(self) -> bool:  # noqa: PLR6301  # subclass hook
        """Return whether this provider can use a runtime loopback redirect URI.

        When `False`, `client_metadata()` ignores the `redirect_uri` argument
        and uses the provider's own pre-registered static URI instead.

        Returns:
            `True` when the provider accepts dynamically registered redirect URIs.
        """
        return True

    def loopback_port(self) -> int | None:  # noqa: PLR6301  # subclass hook
        """Return a fixed loopback port, or `None` for a random ephemeral port.

        Override when the provider's app registration requires a specific
        pre-registered port (e.g. Slack registers `http://localhost:3118/callback`).
        Ignored when `supports_loopback_callback()` is `False`.

        Returns:
            A fixed TCP port, or `None` to pick a random ephemeral port.
        """
        return None

    def client_metadata(  # noqa: PLR6301  # subclass hook
        self,
        *,
        redirect_uri: str | None = None,
    ) -> OAuthClientMetadata:
        """Return the `OAuthClientMetadata` used to build the auth provider.

        Args:
            redirect_uri: Optional runtime redirect URI for CLI loopback auth.

        Returns:
            Metadata for the spec-compliant Authorization Code + PKCE +
            Dynamic Client Registration flow.
        """
        return OAuthClientMetadata(
            client_name="deepagents-code",
            redirect_uris=[AnyUrl(redirect_uri or "http://localhost/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
        )

    async def run_login(  # noqa: PLR6301  # subclass hook
        self,
        *,
        server_name: str,
        server_url: str,
        storage: FileTokenStorage,
        ui: OAuthInteraction,
    ) -> LoginResult:
        """Perform any provider-specific pre-handshake work.

        Args:
            server_name: MCP server name from `mcpServers`.
            server_url: Remote MCP endpoint URL.
            storage: File-backed token storage for this server identity.
            ui: Interaction surface used for any provider-specific
                prompts (e.g. device-code instructions, workspace IDs).

        Returns:
            `LoginResult.completed=True` if the provider finished the
            login itself (e.g. device flow). Otherwise the caller drives
            the standard Authorization Code handshake and passes any
            returned `extra_auth_params` to the redirect URL.
        """
        del server_name, server_url, storage, ui
        return LoginResult()


class GenericProvider(OAuthProvider):
    """Fallback provider for spec-compliant MCP servers with no quirks."""

    def matches(self, server_url: str) -> bool:  # noqa: PLR6301  # subclass hook
        """Match any URL — the registry places this provider last.

        Args:
            server_url: Remote MCP endpoint URL (unused).

        Returns:
            Always `True`.
        """
        del server_url
        return True
