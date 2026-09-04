"""OAuth support for Talon MCP servers."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import ValidationError

from deepagents_talon.authorization import (
    AuthorizationBinding,
    AuthorizationURL,
    CallbackURLRequested,
    current_authorization_attempt,
    current_authorization_handler,
    current_authorization_invocation,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_REDIRECT_URI = "http://localhost:3000/callback"
_TOKEN_DIR = Path(".deepagents/mcp-tokens")
_AUTHORIZATION_TIMEOUT_SECONDS = 10 * 60


class MCPAuthorizationError(RuntimeError):
    """An MCP authorization flow could not be completed safely."""


class FileTokenStorage:
    """Store MCP OAuth credentials in the user's Deep Agents state directory."""

    def __init__(
        self,
        server_name: str,
        *,
        server_url: str,
        force_authorization: bool = False,
    ) -> None:
        """Bind storage to a server name and URL.

        Args:
            server_name: Configured MCP server name.
            server_url: Remote MCP endpoint used to isolate credentials.
            force_authorization: Whether the first token read should require a
                fresh OAuth flow without deleting the stored credential.
        """
        digest = hashlib.sha256(server_url.encode()).hexdigest()[:12]
        self.path = Path.home() / _TOKEN_DIR / f"{server_name}-{digest}.json"
        self._force_authorization = force_authorization

    async def get_tokens(self) -> OAuthToken | None:
        """Return stored OAuth tokens."""
        if self._force_authorization:
            self._force_authorization = False
            return None
        data = await asyncio.to_thread(self._read)
        raw = data.get("tokens") if data is not None else None
        return OAuthToken.model_validate(raw) if raw is not None else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Persist OAuth tokens."""
        await asyncio.to_thread(self._update, "tokens", json.loads(tokens.model_dump_json()))
        attempt = current_authorization_attempt()
        if attempt is not None and attempt.binding is not None:
            attempt.completed = True

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """Return stored OAuth client registration."""
        data = await asyncio.to_thread(self._read)
        raw = data.get("client_info") if data is not None else None
        return OAuthClientInformationFull.model_validate(raw) if raw is not None else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Persist OAuth client registration."""
        value = json.loads(client_info.model_dump_json(exclude_none=True))
        await asyncio.to_thread(self._update, "client_info", value)

    def _read(self) -> dict[str, object] | None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if not isinstance(raw, dict):
            msg = f"Invalid MCP credential file: {self.path}"
            raise TypeError(msg)
        return raw

    def _update(self, key: str, value: object) -> None:
        directory = self.path.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)
        data = self._read() or {}
        data[key] = value
        descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".tokens-", text=True)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(data, file)
            Path(temporary).replace(self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(descriptor)
            Path(temporary).unlink(missing_ok=True)
            raise


def build_oauth_provider(
    *, server_name: str, server_url: str, storage: FileTokenStorage, interactive: bool
) -> OAuthClientProvider:
    """Build an MCP SDK OAuth provider for Talon."""
    redirect, callback = _interactive_handlers() if interactive else _channel_handlers(server_name)
    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=OAuthClientMetadata(
            redirect_uris=[_REDIRECT_URI],
            client_name="Deep Agents Talon",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",  # noqa: S106
        ),
        storage=storage,
        redirect_handler=redirect,
        callback_handler=callback,
    )


def _interactive_handlers() -> tuple[
    Callable[[str], Awaitable[None]], Callable[[], Awaitable[tuple[str, str | None]]]
]:
    async def redirect(url: str) -> None:
        print("Open this URL in a browser and approve access:\n")  # noqa: T201
        print(f"  {url}\n")  # noqa: T201

    async def callback() -> tuple[str, str | None]:
        try:
            raw = await asyncio.to_thread(input, "Paste the full callback URL: ")
        except EOFError as exc:
            msg = "No callback URL received; re-run the login command."
            raise RuntimeError(msg) from exc
        return _parse_callback_url(raw)

    return redirect, callback


def _channel_handlers(
    server_name: str,
) -> tuple[Callable[[str], Awaitable[None]], Callable[[], Awaitable[tuple[str, str | None]]]]:
    async def redirect(url: str) -> None:
        handler = current_authorization_handler()
        invocation_id = current_authorization_invocation()
        attempt = current_authorization_attempt()
        if handler is None or invocation_id is None or attempt is None:
            msg = "MCP authorization requires an interactive Talon channel"
            raise MCPAuthorizationError(msg)
        binding = AuthorizationBinding(
            server_name=server_name,
            invocation_id=invocation_id,
            expires_at=asyncio.get_running_loop().time() + _AUTHORIZATION_TIMEOUT_SECONDS,
        )
        attempt.binding = binding
        await handler(AuthorizationURL(binding=binding, url=url))

    async def callback() -> tuple[str, str | None]:
        handler = current_authorization_handler()
        attempt = current_authorization_attempt()
        binding = None if attempt is None else attempt.binding
        if handler is None or binding is None:
            msg = "MCP authorization callback is unavailable"
            raise MCPAuthorizationError(msg)
        raw = await handler(CallbackURLRequested(binding=binding))
        if not isinstance(raw, str):
            msg = "MCP authorization callback was not received"
            raise MCPAuthorizationError(msg)
        return _parse_callback_url(raw)

    return redirect, callback


def _parse_callback_url(raw: str) -> tuple[str, str | None]:
    parsed = urlparse(raw.strip())
    expected = urlparse(_REDIRECT_URI)
    if (parsed.scheme, parsed.netloc, parsed.path) != (
        expected.scheme,
        expected.netloc,
        expected.path,
    ):
        msg = "MCP authorization callback is invalid"
        raise MCPAuthorizationError(msg)
    query = parse_qs(parsed.query)
    if query.get("error"):
        msg = "MCP authorization was denied"
        raise MCPAuthorizationError(msg)
    code = query.get("code", [None])[0]
    state = query.get("state", [None])[0]
    if not code or not state:
        msg = "MCP authorization callback is invalid"
        raise MCPAuthorizationError(msg)
    return code, state


def format_login_error(exc: BaseException) -> str:
    """Return a credential-safe OAuth failure message."""
    if isinstance(exc, (OSError, ValidationError, TypeError, ValueError)):
        return str(exc).splitlines()[0]
    return type(exc).__name__


__all__ = [
    "FileTokenStorage",
    "MCPAuthorizationError",
    "build_oauth_provider",
    "format_login_error",
]
