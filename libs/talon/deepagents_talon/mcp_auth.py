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

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_REDIRECT_URI = "http://localhost:3000/callback"
_TOKEN_DIR = Path(".deepagents/mcp-tokens")


class FileTokenStorage:
    """Store MCP OAuth credentials in the user's Deep Agents state directory."""

    def __init__(self, server_name: str, *, server_url: str) -> None:
        """Bind storage to a server name and URL."""
        digest = hashlib.sha256(server_url.encode()).hexdigest()[:12]
        self.path = Path.home() / _TOKEN_DIR / f"{server_name}-{digest}.json"

    async def get_tokens(self) -> OAuthToken | None:
        """Return stored OAuth tokens."""
        data = await asyncio.to_thread(self._read)
        raw = data.get("tokens") if data is not None else None
        return OAuthToken.model_validate(raw) if raw is not None else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Persist OAuth tokens."""
        await asyncio.to_thread(self._update, "tokens", json.loads(tokens.model_dump_json()))

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
    redirect, callback = (
        _interactive_handlers() if interactive else _noninteractive_handlers(server_name)
    )
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
        query = parse_qs(urlparse(raw.strip()).query)
        if error := query.get("error"):
            msg = f"Authorization failed: {error[0]}"
            raise RuntimeError(msg)
        code = query.get("code", [None])[0]
        state = query.get("state", [None])[0]
        if not code:
            msg = "The callback URL did not contain an authorization code."
            raise RuntimeError(msg)
        return code, state

    return redirect, callback


def _noninteractive_handlers(
    server_name: str,
) -> tuple[Callable[[str], Awaitable[None]], Callable[[], Awaitable[tuple[str, str | None]]]]:
    async def redirect(_url: str) -> None:
        msg = f"MCP server {server_name!r} needs authentication"
        raise RuntimeError(msg)

    async def callback() -> tuple[str, str | None]:
        msg = f"MCP server {server_name!r} needs authentication"
        raise RuntimeError(msg)

    return redirect, callback


def format_login_error(exc: BaseException) -> str:
    """Return a credential-safe OAuth failure message."""
    if isinstance(exc, (OSError, ValidationError, TypeError, ValueError)):
        return str(exc).splitlines()[0]
    return type(exc).__name__


__all__ = ["FileTokenStorage", "build_oauth_provider", "format_login_error"]
