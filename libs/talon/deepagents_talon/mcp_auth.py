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

import httpx
from langchain_core._security._ssrf_protection import validate_safe_url
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import AnyUrl, OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError

from deepagents_talon.authorization import (
    AuthorizationBinding,
    AuthorizationURL,
    CallbackURLRequested,
    DeviceCode,
    current_authorization_attempt,
    current_authorization_handler,
    current_authorization_invocation,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_REDIRECT_URI = "http://localhost:3000/callback"
_TOKEN_DIR = Path(".deepagents/mcp-tokens")
_AUTHORIZATION_TIMEOUT_SECONDS = 10 * 60
_GITHUB_MCP_CLIENT_ID = "Iv23libxz8qOApH0WQL3"
_GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
_GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"  # noqa: S105
_DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"


class _DeviceCodeResponse(BaseModel):
    """Validated RFC 8628 device-authorization response."""

    model_config = ConfigDict(extra="ignore")

    device_code: SecretStr
    user_code: SecretStr
    verification_uri: str
    expires_in: int
    interval: int = 5


class _DeviceTokenResponse(BaseModel):
    """Validated device token response with redacted credential fields."""

    model_config = ConfigDict(extra="ignore")

    access_token: SecretStr
    token_type: str = "Bearer"  # noqa: S105  # OAuth token type, not a credential.
    expires_in: int | None = None
    scope: str | None = None
    refresh_token: SecretStr | None = None


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
        _mark_authorization_complete()

    async def set_tokens_and_client_info(
        self,
        tokens: OAuthToken,
        client_info: OAuthClientInformationFull,
    ) -> None:
        """Atomically persist OAuth tokens and their client registration."""
        values = {
            "tokens": json.loads(tokens.model_dump_json()),
            "client_info": json.loads(client_info.model_dump_json(exclude_none=True)),
        }
        await asyncio.to_thread(self._update_values, values)
        _mark_authorization_complete()

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
        self._update_values({key: value})

    def _update_values(self, values: dict[str, object]) -> None:
        directory = self.path.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)
        data = self._read() or {}
        data.update(values)
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


def _mark_authorization_complete() -> None:
    attempt = current_authorization_attempt()
    if attempt is not None and attempt.binding is not None:
        attempt.completed = True


def is_github_mcp_url(url: str) -> bool:
    """Return whether a URL points to GitHub's hosted MCP server."""
    return (urlparse(url).hostname or "").lower() == "api.githubcopilot.com"


async def authorize_github_mcp(
    server_name: str,
    storage: FileTokenStorage,
    *,
    interactive: bool,
) -> None:
    """Run GitHub's device grant and persist the resulting MCP credential."""
    token = await _run_github_device_flow(server_name, interactive=interactive)
    await storage.set_tokens_and_client_info(
        token,
        OAuthClientInformationFull(
            client_id=_GITHUB_MCP_CLIENT_ID,
            redirect_uris=[AnyUrl("http://localhost/callback")],
            grant_types=[_DEVICE_GRANT_TYPE],
            response_types=["code"],
            token_endpoint_auth_method="none",  # noqa: S106
        ),
    )


async def _run_github_device_flow(server_name: str, *, interactive: bool) -> OAuthToken:
    device_url = validate_safe_url(_GITHUB_DEVICE_CODE_URL, allow_http=False)
    token_url = validate_safe_url(_GITHUB_TOKEN_URL, allow_http=False)
    async with httpx.AsyncClient(timeout=30.0) as client:
        device = await _request_device_code(client, device_url)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + device.expires_in
        await _present_device_code(
            server_name,
            device,
            deadline=deadline,
            interactive=interactive,
        )
        return await _poll_for_device_token(client, token_url, device, deadline=deadline)


async def _request_device_code(
    client: httpx.AsyncClient,
    device_url: str,
) -> _DeviceCodeResponse:
    response = await client.post(
        device_url,
        data={"client_id": _GITHUB_MCP_CLIENT_ID},
        headers={"Accept": "application/json"},
    )
    if response.is_error:
        msg = f"GitHub device code request failed with HTTP {response.status_code}."
        raise MCPAuthorizationError(msg)
    try:
        return _DeviceCodeResponse.model_validate(response.json())
    except (ValueError, ValidationError) as exc:
        msg = "GitHub returned an invalid device code response."
        raise MCPAuthorizationError(msg) from exc


async def _present_device_code(
    server_name: str,
    device: _DeviceCodeResponse,
    *,
    deadline: float,
    interactive: bool,
) -> None:
    verification_uri = device.verification_uri
    user_code = device.user_code.get_secret_value()
    if interactive:
        print("Open this URL in a browser and enter the code shown below:\n")  # noqa: T201
        print(f"  {verification_uri}")  # noqa: T201
        print(f"  Code: {user_code}\n")  # noqa: T201
        return
    handler = current_authorization_handler()
    invocation_id = current_authorization_invocation()
    attempt = current_authorization_attempt()
    if handler is None or invocation_id is None or attempt is None:
        msg = "MCP authorization requires an interactive Talon channel"
        raise MCPAuthorizationError(msg)
    binding = AuthorizationBinding(
        server_name=server_name,
        invocation_id=invocation_id,
        expires_at=deadline,
    )
    attempt.binding = binding
    await handler(
        DeviceCode(
            binding=binding,
            verification_uri=verification_uri,
            user_code=user_code,
        )
    )


async def _poll_for_device_token(
    client: httpx.AsyncClient,
    token_url: str,
    device: _DeviceCodeResponse,
    *,
    deadline: float,
) -> OAuthToken:
    interval = max(device.interval, 1)
    loop = asyncio.get_running_loop()
    while (remaining := deadline - loop.time()) > 0:
        await asyncio.sleep(min(interval, remaining))
        if loop.time() >= deadline:
            break
        response = await client.post(
            token_url,
            data={
                "client_id": _GITHUB_MCP_CLIENT_ID,
                "device_code": device.device_code.get_secret_value(),
                "grant_type": _DEVICE_GRANT_TYPE,
            },
            headers={"Accept": "application/json"},
        )
        try:
            body = response.json()
        except ValueError as exc:
            msg = "GitHub returned an invalid device token response."
            raise MCPAuthorizationError(msg) from exc
        if not isinstance(body, dict):
            msg = "GitHub returned an invalid device token response."
            raise MCPAuthorizationError(msg)
        error = body.get("error")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
            continue
        if error is not None:
            msg = "GitHub device authorization failed."
            raise MCPAuthorizationError(msg)
        if response.is_error:
            msg = f"GitHub device token request failed with HTTP {response.status_code}."
            raise MCPAuthorizationError(msg)
        return _parse_device_token(body)
    msg = "GitHub device authorization expired; try logging in again."
    raise TimeoutError(msg)


def _parse_device_token(body: dict[str, object]) -> OAuthToken:
    try:
        response = _DeviceTokenResponse.model_validate(body)
    except ValidationError as exc:
        msg = "GitHub returned an invalid device token response."
        raise MCPAuthorizationError(msg) from exc
    if response.token_type.lower() != "bearer":
        msg = "GitHub returned an unsupported device token type."
        raise MCPAuthorizationError(msg)
    return OAuthToken(
        access_token=response.access_token.get_secret_value(),
        token_type="Bearer",  # noqa: S106  # OAuth token type, not a credential.
        expires_in=response.expires_in,
        scope=response.scope,
        refresh_token=(
            response.refresh_token.get_secret_value()
            if response.refresh_token is not None
            else None
        ),
    )


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
    "authorize_github_mcp",
    "build_oauth_provider",
    "format_login_error",
    "is_github_mcp_url",
]
