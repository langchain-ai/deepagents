"""OAuth support for Talon MCP servers."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urljoin, urlparse

import httpx
from langchain_core._security._ssrf_protection import validate_safe_url
from mcp.client.auth import OAuthClientProvider
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
)
from mcp.client.streamable_http import MCP_PROTOCOL_VERSION
from mcp.shared.auth import (
    AnyUrl,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)
from mcp.types import LATEST_PROTOCOL_VERSION
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, SecretStr, ValidationError

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

    from mcp.client.auth.oauth2 import OAuthContext

_REDIRECT_URI = "http://localhost:3000/callback"
_TOKEN_DIR = Path(".deepagents/mcp-tokens")
_AUTHORIZATION_TIMEOUT_SECONDS = 10 * 60
_GITHUB_MCP_CLIENT_ID = "Iv23libxz8qOApH0WQL3"
_GITHUB_MCP_HOST = "api.githubcopilot.com"
_DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
_HTTP_TIMEOUT_SECONDS = 5.0
_DISCOVERY_TIMEOUT_SECONDS = 10.0
_MAX_OAUTH_RESPONSE_BYTES = 64 * 1024
_MIN_RESPONSE_STATUS = 200
_REDIRECT_STATUS = 300
_BAD_REQUEST_STATUS = 400
_SERVER_ERROR_STATUS = 500


class _AuthorizationServerMetadata(BaseModel):
    """OAuth metadata fields required for device authorization."""

    model_config = ConfigDict(extra="ignore")

    issuer: AnyHttpUrl
    token_endpoint: AnyHttpUrl
    registration_endpoint: AnyHttpUrl | None = None
    grant_types_supported: list[str] | None = None
    device_authorization_endpoint: AnyHttpUrl | None = None


@dataclass(frozen=True, slots=True)
class _DeviceAuthorization:
    """Discovered endpoints and public client for one device flow."""

    device_endpoint: str
    token_endpoint: str
    resource: str
    client_info: OAuthClientInformationFull


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


class DeviceAuthorizationCompletedError(RuntimeError):
    """Device authorization persisted credentials and ended the code flow."""


async def prepare_device_client(server_url: str, storage: FileTokenStorage) -> None:
    """Seed the public client required by GitHub's non-registering OAuth server."""
    if _origin(server_url) != ("https", _GITHUB_MCP_HOST, 443):
        return
    existing = await storage.get_client_info()
    if not _is_public_device_client(existing):
        await storage.set_client_info(_public_client_info(_GITHUB_MCP_CLIENT_ID))


async def _authorize_discovered_device(
    server_name: str,
    storage: FileTokenStorage,
    context: OAuthContext,
    *,
    interactive: bool,
) -> bool:
    if not interactive and current_authorization_handler() is None:
        return False
    if context.auth_server_url is None or context.client_info is None:
        return False
    metadata = await _discover_device_metadata(context.auth_server_url)
    if metadata is None:
        return False
    client_info = context.client_info
    if not _is_public_device_client(client_info):
        client_info = await _register_device_client(metadata)
    if client_info is None:
        return False
    authorization = _DeviceAuthorization(
        device_endpoint=_issuer_endpoint(metadata.device_authorization_endpoint, metadata),
        token_endpoint=_issuer_endpoint(metadata.token_endpoint, metadata),
        resource=context.get_resource_url(),
        client_info=client_info,
    )
    token = await _run_device_flow(server_name, authorization, interactive=interactive)
    await storage.set_tokens_and_client_info(token, client_info)
    return True


async def _discover_device_metadata(
    auth_server_url: str,
) -> _AuthorizationServerMetadata | None:
    try:
        issuer = _safe_https_url(auth_server_url)
        async with asyncio.timeout(_DISCOVERY_TIMEOUT_SECONDS):
            async with httpx.AsyncClient(
                timeout=_HTTP_TIMEOUT_SECONDS,
                follow_redirects=False,
            ) as client:
                for candidate in build_oauth_authorization_server_metadata_discovery_urls(
                    issuer, issuer
                ):
                    metadata = await _read_device_metadata(client, issuer, candidate)
                    if metadata is not None:
                        return metadata
    except (httpx.HTTPError, MCPAuthorizationError, OSError, ValidationError, ValueError):
        return None
    return None


async def _read_device_metadata(
    client: httpx.AsyncClient,
    issuer: str,
    candidate: str,
) -> _AuthorizationServerMetadata | None:
    if _origin(candidate) != _origin(issuer):
        return None
    body = await _get_json(client, _safe_https_url(candidate))
    if body is None:
        return None
    try:
        metadata = _AuthorizationServerMetadata.model_validate(body)
    except ValidationError:
        return None
    if _normalized_url(str(metadata.issuer)) != _normalized_url(issuer):
        return None
    if metadata.device_authorization_endpoint is None or _DEVICE_GRANT_TYPE not in (
        metadata.grant_types_supported or ()
    ):
        return None
    _issuer_endpoint(metadata.device_authorization_endpoint, metadata)
    _issuer_endpoint(metadata.token_endpoint, metadata)
    return metadata


async def _register_device_client(
    metadata: _AuthorizationServerMetadata,
) -> OAuthClientInformationFull | None:
    registration_endpoint = metadata.registration_endpoint or AnyHttpUrl(
        urljoin(str(metadata.issuer), "/register")
    )
    endpoint = _issuer_endpoint(registration_endpoint, metadata)
    async with httpx.AsyncClient(
        timeout=_HTTP_TIMEOUT_SECONDS,
        follow_redirects=False,
    ) as client:
        body = await _post_json(
            client,
            endpoint,
            json_data={
                "client_name": "Deep Agents Talon",
                "grant_types": [_DEVICE_GRANT_TYPE],
                "redirect_uris": [_REDIRECT_URI],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
    if body is None:
        return None
    try:
        client_info = OAuthClientInformationFull.model_validate(body)
    except ValidationError:
        return None
    return client_info if _is_public_device_client(client_info) else None


def _is_public_device_client(client_info: OAuthClientInformationFull | None) -> bool:
    return bool(
        client_info is not None
        and client_info.client_id
        and client_info.client_secret is None
        and client_info.token_endpoint_auth_method in {None, "none"}
        and _DEVICE_GRANT_TYPE in client_info.grant_types
    )


def _public_client_info(client_id: str) -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id=client_id,
        redirect_uris=[AnyUrl("http://localhost/callback")],
        grant_types=[_DEVICE_GRANT_TYPE],
        response_types=["code"],
        token_endpoint_auth_method="none",  # noqa: S106
    )


async def _run_device_flow(
    server_name: str,
    authorization: _DeviceAuthorization,
    *,
    interactive: bool,
) -> OAuthToken:
    async with httpx.AsyncClient(
        timeout=_HTTP_TIMEOUT_SECONDS,
        follow_redirects=False,
    ) as client:
        device = await _request_device_code(client, authorization)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + device.expires_in
        await _present_device_code(
            server_name,
            device,
            deadline=deadline,
            interactive=interactive,
        )
        return await _poll_for_device_token(
            client,
            authorization,
            device,
            deadline=deadline,
        )


async def _request_device_code(
    client: httpx.AsyncClient,
    authorization: _DeviceAuthorization,
) -> _DeviceCodeResponse:
    body = await _post_json(
        client,
        authorization.device_endpoint,
        data={
            "client_id": authorization.client_info.client_id,
            "resource": authorization.resource,
        },
    )
    if body is None:
        msg = "Device code request failed."
        raise MCPAuthorizationError(msg)
    try:
        device = _DeviceCodeResponse.model_validate(body)
        device.verification_uri = _safe_https_url(device.verification_uri)
    except (ValidationError, ValueError) as exc:
        msg = "Authorization server returned an invalid device code response."
        raise MCPAuthorizationError(msg) from exc
    else:
        return device


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
    authorization: _DeviceAuthorization,
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
        body = await _post_json(
            client,
            authorization.token_endpoint,
            data={
                "client_id": authorization.client_info.client_id,
                "device_code": device.device_code.get_secret_value(),
                "grant_type": _DEVICE_GRANT_TYPE,
                "resource": authorization.resource,
            },
        )
        if body is None:
            msg = "Authorization server returned an invalid device token response."
            raise MCPAuthorizationError(msg)
        error = body.get("error")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
            continue
        if error is not None:
            msg = "Device authorization failed."
            raise MCPAuthorizationError(msg)
        return _parse_device_token(body)
    msg = "Device authorization expired; try logging in again."
    raise TimeoutError(msg)


def _parse_device_token(body: dict[str, object]) -> OAuthToken:
    try:
        response = _DeviceTokenResponse.model_validate(body)
    except ValidationError as exc:
        msg = "Authorization server returned an invalid device token response."
        raise MCPAuthorizationError(msg) from exc
    if response.token_type.lower() != "bearer":
        msg = "Authorization server returned an unsupported device token type."
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


async def _get_json(client: httpx.AsyncClient, url: str) -> dict[str, object] | None:
    return await _request_json(client, "GET", url)


async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    data: dict[str, object] | None = None,
    json_data: dict[str, object] | None = None,
) -> dict[str, object] | None:
    return await _request_json(client, "POST", url, data=data, json_data=json_data)


async def _request_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    data: dict[str, object] | None = None,
    json_data: dict[str, object] | None = None,
) -> dict[str, object] | None:
    headers = {
        "Accept": "application/json",
        MCP_PROTOCOL_VERSION: LATEST_PROTOCOL_VERSION,
    }
    async with client.stream(method, url, data=data, json=json_data, headers=headers) as response:
        if (
            response.status_code >= _SERVER_ERROR_STATUS
            or response.status_code < _MIN_RESPONSE_STATUS
            or (
                response.status_code >= _REDIRECT_STATUS
                and not (method == "POST" and response.status_code == _BAD_REQUEST_STATUS)
            )
        ):
            return None
        content = bytearray()
        async for chunk in response.aiter_bytes():
            if len(content) + len(chunk) > _MAX_OAUTH_RESPONSE_BYTES:
                msg = "OAuth response exceeded the size limit."
                raise MCPAuthorizationError(msg)
            content.extend(chunk)
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _safe_https_url(url: str) -> str:
    return validate_safe_url(url, allow_http=False)


def _issuer_endpoint(
    endpoint: AnyHttpUrl | None,
    metadata: _AuthorizationServerMetadata,
) -> str:
    if endpoint is None:
        msg = "OAuth endpoint is missing."
        raise MCPAuthorizationError(msg)
    validated = _safe_https_url(str(endpoint))
    if _origin(validated) != _origin(str(metadata.issuer)):
        msg = "OAuth endpoint does not match the authorization server."
        raise MCPAuthorizationError(msg)
    return validated


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    port = parsed.port or (443 if scheme == "https" else 80 if scheme == "http" else None)
    return scheme, (parsed.hostname or "").lower(), port


def _normalized_url(url: str) -> str:
    return url.rstrip("/")


def build_oauth_provider(
    *, server_name: str, server_url: str, storage: FileTokenStorage, interactive: bool
) -> OAuthClientProvider:
    """Build an MCP SDK OAuth provider for Talon."""
    fallback, callback = _interactive_handlers() if interactive else _channel_handlers(server_name)
    provider: OAuthClientProvider | None = None

    async def redirect(url: str) -> None:
        if provider is not None and await _authorize_discovered_device(
            server_name,
            storage,
            provider.context,
            interactive=interactive,
        ):
            raise DeviceAuthorizationCompletedError
        await fallback(url)

    provider = OAuthClientProvider(
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
    return provider


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
    "DeviceAuthorizationCompletedError",
    "FileTokenStorage",
    "MCPAuthorizationError",
    "build_oauth_provider",
    "format_login_error",
    "prepare_device_client",
]
