"""OAuth support for Talon MCP servers."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import parse_qs, urljoin, urlparse

import httpx
from langchain_core._security._exceptions import SSRFBlockedError
from langchain_core._security._policy import SSRFPolicy
from langchain_core._security._ssrf_protection import validate_safe_url
from langchain_core._security._transport import SSRFSafeTransport
from mcp.client.auth import OAuthClientProvider
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
    create_oauth_metadata_request,
    extract_resource_metadata_from_www_auth,
    handle_auth_metadata_response,
    handle_protected_resource_response,
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
from deepagents_talon.mcp_config import locked_path, warn_agent_workspace_path

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from mcp.client.auth.oauth2 import OAuthContext

_REDIRECT_URI = "http://localhost:3000/callback"
_SLACK_MCP_CLIENT_ID = "4518649543379.10944517634130"
_SLACK_REDIRECT_URI = "http://localhost:3118/callback"
_OAUTH_CALLBACK_ENDPOINTS = frozenset(
    {
        ("http", "localhost:3000", "/callback"),
        ("http", "localhost:3118", "/callback"),
    }
)
_TOKEN_DIR = Path(".deepagents/mcp-tokens")
_AUTHORIZATION_TIMEOUT_SECONDS = 10 * 60
_GITHUB_MCP_CLIENT_ID = "Iv23libxz8qOApH0WQL3"
_GITHUB_MCP_HOST = "api.githubcopilot.com"
_DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
_HTTP_TIMEOUT_SECONDS = 5.0
_DISCOVERY_TIMEOUT_SECONDS = 10.0
_RESOLVE_TIMEOUT_SECONDS = 5.0
_MAX_OAUTH_RESPONSE_BYTES = 64 * 1024
_MIN_RESPONSE_STATUS = 200
_REDIRECT_STATUS = 300
_BAD_REQUEST_STATUS = 400
_SERVER_ERROR_STATUS = 500
_UNSAFE_ENDPOINT_MESSAGE = "OAuth endpoint is not a safe public HTTPS URL."


def _no_authorization_channel_error(server_name: str) -> MCPAuthorizationError:
    """Return the failure for a run with nowhere to send an authorization prompt.

    The conversation may well be interactive; what is missing is a handler for
    this run. Scheduled jobs invoke the agent without one, and a background
    subagent does not inherit it, so name the remedy instead of the channel.

    Args:
        server_name: Configured MCP server name, for the login command.

    Returns:
        The error to raise.
    """
    msg = (
        "MCP authorization has no conversation to prompt in: this run was not "
        "started by a channel message (a scheduled job or a background subagent). "
        f"Authorize from a chat, or run: deepagents-talon mcp login {server_name}"
    )
    return MCPAuthorizationError(msg)


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


_ASCII_CONTROL_LIMIT = 32
_ASCII_DELETE = 127


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
        agent_root: Path | None = None,
    ) -> None:
        """Bind storage to a server name and URL.

        These files hold live bearer and refresh tokens in cleartext, so the
        directory is restricted to the owner and a location inside the agent
        workspace is warned about. Against Talon's default shell backend that is
        hardening, not a boundary, and it cannot become one here: the agent can
        read any absolute path it is given, and filesystem deny rules cannot be
        applied to a backend that executes commands. See
        `mcp_config.warn_agent_workspace_path`.

        Args:
            server_name: Configured MCP server name.
            server_url: Remote MCP endpoint used to isolate credentials.
            force_authorization: Whether the first token read should require a
                fresh OAuth flow without deleting the stored credential.
            agent_root: Agent workspace root. Defaults to the process workspace.
        """
        digest = hashlib.sha256(server_url.encode()).hexdigest()[:12]
        directories: list[Path] = []
        current = Path.home()
        for part in _TOKEN_DIR.parts:
            current = current / part
            directories.append(current)
        self._directories = tuple(directories)
        path = current / f"{server_name}-{digest}.json"
        warn_agent_workspace_path(path, agent_root, subject="MCP credential storage")
        self.path = path
        self._force_authorization = force_authorization

    async def get_tokens(self) -> OAuthToken | None:
        """Return stored OAuth tokens."""
        if self._force_authorization:
            self._force_authorization = False
            return None
        data = await asyncio.to_thread(self._read)
        raw = data.get("tokens") if data is not None else None
        return OAuthToken.model_validate(raw) if raw is not None else None

    async def get_token_expiry(self) -> float | None:
        """Return the stored absolute token expiry time."""
        data = await asyncio.to_thread(self._read)
        raw = data.get("expires_at") if data is not None else None
        if raw is None:
            return None
        if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(raw):
            msg = f"Invalid MCP credential file: {self.path}"
            raise TypeError(msg)
        return float(raw)

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Persist OAuth tokens and their absolute expiry time.

        RFC 6749 section 6 lets a refresh response omit `refresh_token`, meaning
        "keep the one you have", and many servers do. Replacing the whole blob
        would destroy the long-lived credential on the first such refresh and
        force a full interactive login, so the stored value is carried forward.
        """
        values = {
            "tokens": json.loads(tokens.model_dump_json()),
            "expires_at": _token_expiry(tokens),
        }
        await asyncio.to_thread(self._update_values, values, keep_refresh_token=True)
        _mark_authorization_complete()

    async def set_tokens_and_client_info(
        self,
        tokens: OAuthToken,
        client_info: OAuthClientInformationFull,
    ) -> None:
        """Atomically persist OAuth tokens and their client registration.

        This records a *fresh* grant, so unlike `set_tokens` it does not carry a
        stored `refresh_token` forward. RFC 6749 section 6's "keep the one you
        have" applies to a refresh response; stitching the previous grant's
        refresh token onto a new access token would leave a credential the
        authorization server may already have revoked -- which is exactly what an
        explicit reauthentication invalidates -- instead of correctly recording
        that this grant is not refreshable.
        """
        values = {
            "tokens": json.loads(tokens.model_dump_json()),
            "expires_at": _token_expiry(tokens),
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

    def _update_values(
        self, values: dict[str, object], *, keep_refresh_token: bool = False
    ) -> None:
        directory = self.path.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        for parent in self._directories[:-1]:
            # mkdir(mode=...) applies the mode to the leaf only, so an
            # intermediate it created keeps the process umask.
            with contextlib.suppress(OSError):
                parent.chmod(0o700)
        directory.chmod(0o700)
        # Two concurrent refreshes read-modify-write the same file, so without
        # the lock one side's tokens and client_info are silently dropped.
        with locked_path(self.path):
            data = self._read() or {}
            if keep_refresh_token:
                values = _with_stored_refresh_token(values, data)
            data.update(values)
            self._write(data, directory)

    def _write(self, data: dict[str, object], directory: Path) -> None:
        descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".tokens-", text=True)
        try:
            os.fchmod(descriptor, 0o600)
            stream = os.fdopen(descriptor, "w", encoding="utf-8")
            # fdopen owns the descriptor from here, and this runs on a worker
            # thread: closing it twice could close an unrelated reused fd.
            descriptor = -1
            with stream as file:
                json.dump(data, file)
                file.flush()
                os.fsync(file.fileno())
            Path(temporary).replace(self.path)
        except BaseException:
            if descriptor != -1:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
            Path(temporary).unlink(missing_ok=True)
            raise


def _token_expiry(tokens: OAuthToken) -> float | None:
    return time.time() + tokens.expires_in if tokens.expires_in is not None else None


def _with_stored_refresh_token(
    values: dict[str, object], data: dict[str, object]
) -> dict[str, object]:
    incoming = values.get("tokens")
    stored = data.get("tokens")
    if not isinstance(incoming, dict) or not isinstance(stored, dict):
        return values
    current = cast("dict[str, object]", incoming)
    previous = cast("dict[str, object]", stored)
    if current.get("refresh_token") is not None or previous.get("refresh_token") is None:
        return values
    return {**values, "tokens": {**current, "refresh_token": previous["refresh_token"]}}


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
        device_endpoint=await _issuer_endpoint(metadata.device_authorization_endpoint, metadata),
        token_endpoint=await _issuer_endpoint(metadata.token_endpoint, metadata),
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
        # Every await inside the timeout must yield, so name resolution runs on a
        # worker thread; a blocking getaddrinfo here cannot be interrupted.
        async with asyncio.timeout(_DISCOVERY_TIMEOUT_SECONDS):
            issuer = await _resolved_https_url(auth_server_url)
            async with _oauth_http_client() as client:
                for candidate in build_oauth_authorization_server_metadata_discovery_urls(
                    issuer, issuer
                ):
                    metadata = await _read_device_metadata(client, issuer, candidate)
                    if metadata is not None:
                        return metadata
    except (
        httpx.HTTPError,
        MCPAuthorizationError,
        OSError,
        SSRFBlockedError,
        ValidationError,
        ValueError,
    ):
        return None
    return None


async def _read_device_metadata(
    client: httpx.AsyncClient,
    issuer: str,
    candidate: str,
) -> _AuthorizationServerMetadata | None:
    if _origin(candidate) != _origin(issuer):
        return None
    body = await _get_json(client, await _resolved_https_url(candidate))
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
    await _issuer_endpoint(metadata.device_authorization_endpoint, metadata)
    await _issuer_endpoint(metadata.token_endpoint, metadata)
    return metadata


async def _register_device_client(
    metadata: _AuthorizationServerMetadata,
) -> OAuthClientInformationFull | None:
    registration_endpoint = metadata.registration_endpoint or AnyHttpUrl(
        urljoin(str(metadata.issuer), "/register")
    )
    endpoint = await _issuer_endpoint(registration_endpoint, metadata)
    try:
        async with _oauth_http_client() as client:
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
    except SSRFBlockedError:
        return None
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
    # The device code and the issued tokens travel on this client, so it must be
    # the pinned, proxy-ignoring one rather than an ambient-environment client.
    try:
        async with _oauth_http_client() as client:
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
    except SSRFBlockedError:
        raise MCPAuthorizationError(_UNSAFE_ENDPOINT_MESSAGE) from None


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
        device.verification_uri = await _resolved_https_url(device.verification_uri)
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
        raise _no_authorization_channel_error(server_name)
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


async def _resolved_https_url(url: str) -> str:
    """Validate and resolve `url` off the event loop, under its own bound.

    `asyncio.to_thread` keeps a blocking `getaddrinfo` from stalling the host,
    but it also puts the resolution outside HTTPX's timeouts, which bound only
    the request a client makes afterwards. Without a bound here, a slow or
    hostile resolver stalls whichever flow is waiting on it: the device login or
    the token refresh never returns, even though the loop stays responsive.

    The bound is deliberately separate from the request timeout rather than
    shared with it. Resolution happens before there is a client or a request to
    charge it to, and letting it consume the request budget would make a healthy
    authorization server look unreachable. It also does not use the discovery
    bound, which covers a loop of several requests rather than one name lookup.

    A thread cannot be cancelled, so a timed-out resolution keeps running until
    the C call returns; only the waiter gives up. Every caller goes through here
    rather than calling `asyncio.to_thread` directly, so boundedness does not
    depend on which callers happen to sit inside a timeout -- that dependence is
    how the device-code and registration paths ended up unbounded.

    Args:
        url: Endpoint to validate and resolve.

    Returns:
        The validated URL.

    Raises:
        MCPAuthorizationError: The resolution did not finish in time.
        ValueError: The URL is not a safe public HTTPS URL.
    """
    try:
        async with asyncio.timeout(_RESOLVE_TIMEOUT_SECONDS):
            return await asyncio.to_thread(_safe_https_url, url)
    except TimeoutError:
        msg = "Timed out resolving an OAuth endpoint."
        raise MCPAuthorizationError(msg) from None


async def _issuer_endpoint(
    endpoint: AnyHttpUrl | None,
    metadata: _AuthorizationServerMetadata,
) -> str:
    if endpoint is None:
        msg = "OAuth endpoint is missing."
        raise MCPAuthorizationError(msg)
    validated = await _resolved_https_url(str(endpoint))
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


class _OAuthHTTPTransport(httpx.AsyncHTTPTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        hostname = request.extensions.get("sni_hostname")
        if isinstance(hostname, bytes):
            request.extensions["sni_hostname"] = hostname.decode("ascii")
        return await super().handle_async_request(request)


class _OAuthSafeTransport(SSRFSafeTransport):
    def __init__(self) -> None:
        self._policy = SSRFPolicy(allowed_schemes=frozenset({"https"}))
        self._inner = _OAuthHTTPTransport(trust_env=False)


def _oauth_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=_OAuthSafeTransport(),
        timeout=_HTTP_TIMEOUT_SECONDS,
        follow_redirects=False,
        trust_env=False,
    )


def _reject_oauth_redirect(response: httpx.Response) -> None:
    if _REDIRECT_STATUS <= response.status_code < _BAD_REQUEST_STATUS:
        msg = "OAuth redirects are not allowed."
        raise MCPAuthorizationError(msg)


async def _validate_oauth_url(url: str) -> None:
    msg = _UNSAFE_ENDPOINT_MESSAGE
    try:
        parsed = urlparse(url)
        await _resolved_https_url(url)
    except ValueError:
        raise MCPAuthorizationError(msg) from None
    if parsed.username is not None or parsed.password is not None:
        raise MCPAuthorizationError(msg)


class _PersistedExpiryOAuthProvider(OAuthClientProvider):
    async def _initialize(self) -> None:
        storage = self.context.storage
        if not isinstance(storage, FileTokenStorage):
            msg = "Talon OAuth requires file token storage"
            raise TypeError(msg)
        self.context.current_tokens = await storage.get_tokens()
        self.context.client_info = await storage.get_client_info()
        self.context.token_expiry_time = await storage.get_token_expiry()
        self._initialized = True

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        """Keep discovered OAuth requests outside the resource client's redirect policy."""
        async with contextlib.aclosing(super().async_auth_flow(request)) as flow:
            outbound = await anext(flow)
            while True:
                response = (
                    (yield outbound)
                    if outbound is request
                    else await self._send_oauth_request(outbound)
                )
                try:
                    outbound = await flow.asend(response)
                except StopAsyncIteration:
                    return

    async def _send_oauth_request(self, request: httpx.Request) -> httpx.Response:
        await self._validate_metadata()
        await _validate_oauth_url(str(request.url))
        try:
            async with asyncio.timeout(_HTTP_TIMEOUT_SECONDS), _oauth_http_client() as client:
                response = await client.send(request, follow_redirects=False)
                _reject_oauth_redirect(response)
                return response
        except SSRFBlockedError:
            raise MCPAuthorizationError(_UNSAFE_ENDPOINT_MESSAGE) from None

    async def _validate_metadata(self) -> None:
        metadata = self.context.oauth_metadata
        if metadata is None:
            return
        issuer = self.context.auth_server_url or self.context.get_authorization_base_url(
            self.context.server_url
        )
        # RFC 8414 section 3: the issuer must match the URL the document was
        # discovered from, which this is -- `issuer` derives from the same value
        # the SDK builds its discovery candidates from, in both paths. Endpoints
        # are deliberately not pinned to the issuer's origin: split-origin
        # authorization servers conform (Google issues from accounts.google.com
        # with its token endpoint on oauth2.googleapis.com).
        if _normalized_url(str(metadata.issuer)) != _normalized_url(issuer):
            msg = "OAuth metadata issuer does not match the authorization server."
            raise MCPAuthorizationError(msg)
        for endpoint in (
            metadata.issuer,
            metadata.authorization_endpoint,
            metadata.token_endpoint,
            metadata.registration_endpoint,
        ):
            if endpoint is not None:
                await _validate_oauth_url(str(endpoint))

    async def _perform_authorization(self) -> httpx.Request:
        await self._validate_metadata()
        if self.context.oauth_metadata is None:
            base = self.context.get_authorization_base_url(self.context.server_url)
            await _validate_oauth_url(urljoin(base, "/authorize"))
        return await super()._perform_authorization()

    async def _refresh_token(self) -> httpx.Request:
        if self.context.oauth_metadata is None:
            async with asyncio.timeout(_DISCOVERY_TIMEOUT_SECONDS):
                await self._discover_refresh_metadata()
        await self._validate_metadata()
        return await super()._refresh_token()

    async def _discover_refresh_metadata(self) -> None:
        await self._discover_refresh_resource()
        candidates = build_oauth_authorization_server_metadata_discovery_urls(
            self.context.auth_server_url, self.context.server_url
        )
        for url in candidates:
            response = await self._send_oauth_request(create_oauth_metadata_request(url))
            ok, metadata = await handle_auth_metadata_response(response)
            if not ok:
                break
            if metadata is not None:
                self.context.oauth_metadata = metadata
                await self._validate_metadata()
                return
        msg = "Could not discover a safe OAuth token endpoint for refresh."
        raise MCPAuthorizationError(msg)

    async def _discover_refresh_resource(self) -> None:
        try:
            async with (
                _oauth_http_client() as client,
                client.stream("GET", self.context.server_url) as response,
            ):
                _reject_oauth_redirect(response)
                challenge = extract_resource_metadata_from_www_auth(response)
        except SSRFBlockedError:
            raise MCPAuthorizationError(_UNSAFE_ENDPOINT_MESSAGE) from None
        for url in build_protected_resource_metadata_discovery_urls(
            challenge, self.context.server_url
        ):
            response = await self._send_oauth_request(create_oauth_metadata_request(url))
            metadata = await handle_protected_resource_response(response)
            if metadata is not None:
                await self._validate_resource_match(metadata)
                self.context.protected_resource_metadata = metadata
                self.context.auth_server_url = str(metadata.authorization_servers[0])
                return


def build_oauth_provider(
    *,
    server_name: str,
    server_url: str,
    storage: FileTokenStorage,
    interactive: bool,
) -> OAuthClientProvider:
    """Build an MCP SDK OAuth provider for Talon.

    Args:
        server_name: Configured MCP server name.
        server_url: Remote MCP endpoint URL.
        storage: Credential storage bound to the server identity.
        interactive: Whether to use terminal instead of channel authorization.

    Returns:
        A configured MCP SDK OAuth provider.
    """
    redirect_uri = _SLACK_REDIRECT_URI if _is_slack_mcp_url(server_url) else _REDIRECT_URI
    if interactive:
        fallback, callback = _interactive_handlers(redirect_uri)
    else:
        fallback, callback = _channel_handlers(server_name, redirect_uri)
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

    provider = _PersistedExpiryOAuthProvider(
        server_url=server_url,
        client_metadata=_client_metadata(redirect_uri),
        storage=storage,
        redirect_handler=redirect,
        callback_handler=callback,
    )
    return provider


async def prepare_oauth_login(*, server_url: str, storage: FileTokenStorage) -> None:
    """Preseed provider-specific OAuth client information when required.

    Args:
        server_url: Remote MCP endpoint URL.
        storage: Credential storage bound to the server identity.
    """
    if _is_slack_mcp_url(server_url):
        await _preseed_slack_client_info(storage)
    else:
        await prepare_device_client(server_url, storage)


def _interactive_handlers(
    redirect_uri: str,
) -> tuple[Callable[[str], Awaitable[None]], Callable[[], Awaitable[tuple[str, str | None]]]]:
    async def redirect(url: str) -> None:
        print("Open this URL in a browser and approve access:\n")  # noqa: T201
        print(f"  {url}\n")  # noqa: T201

    async def callback() -> tuple[str, str | None]:
        try:
            raw = await asyncio.to_thread(input, "Paste the full callback URL: ")
        except EOFError as exc:
            msg = "No callback URL received; re-run the login command."
            raise RuntimeError(msg) from exc
        return _parse_callback_url(raw, redirect_uri)

    return redirect, callback


def _channel_handlers(
    server_name: str, redirect_uri: str
) -> tuple[Callable[[str], Awaitable[None]], Callable[[], Awaitable[tuple[str, str | None]]]]:
    async def redirect(url: str) -> None:
        handler = current_authorization_handler()
        invocation_id = current_authorization_invocation()
        attempt = current_authorization_attempt()
        if handler is None or invocation_id is None or attempt is None:
            raise _no_authorization_channel_error(server_name)
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
        return _parse_callback_url(raw, redirect_uri)

    return redirect, callback


def _parse_callback_url(raw: str, redirect_uri: str = _REDIRECT_URI) -> tuple[str, str | None]:
    parsed = urlparse(raw.strip())
    expected = urlparse(redirect_uri)
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


def extract_oauth_callback_url(text: str) -> str | None:
    """Return a recognized Talon OAuth callback URL from a channel message.

    Args:
        text: Raw channel message text.

    Returns:
        The normalized callback URL, or `None` when the message is not a
        recognized callback.
    """
    candidate = text.strip()
    if candidate.startswith("<") and candidate.endswith(">"):
        candidate = candidate[1:-1].strip()
    if any(
        ord(character) < _ASCII_CONTROL_LIMIT or ord(character) == _ASCII_DELETE
        for character in candidate
    ):
        return None
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return None
    if (parsed.scheme, parsed.netloc, parsed.path) not in _OAUTH_CALLBACK_ENDPOINTS:
        return None
    query = parse_qs(parsed.query)
    if not query.get("state") or not (query.get("code") or query.get("error")):
        return None
    return candidate


def _is_slack_mcp_url(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return host == "slack.com" or host.endswith(".slack.com")


def _client_metadata(redirect_uri: str) -> OAuthClientMetadata:
    return OAuthClientMetadata(
        redirect_uris=[AnyUrl(redirect_uri)],
        client_name="Deep Agents Talon",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",  # noqa: S106
    )


async def _preseed_slack_client_info(storage: FileTokenStorage) -> None:
    existing = await storage.get_client_info()
    redirect_uris = existing.redirect_uris if existing is not None else None
    current_redirect = str(redirect_uris[0]) if redirect_uris else None
    if (
        existing is not None
        and existing.client_id == _SLACK_MCP_CLIENT_ID
        and current_redirect == _SLACK_REDIRECT_URI
    ):
        return
    client_info = OAuthClientInformationFull(
        client_id=_SLACK_MCP_CLIENT_ID,
        redirect_uris=[AnyUrl(_SLACK_REDIRECT_URI)],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",  # noqa: S106
    )
    await storage.set_client_info(client_info)


def format_login_error(exc: BaseException) -> str:
    """Return a credential-safe OAuth failure message.

    Args:
        exc: Failure to describe.

    Returns:
        The first line of the message for exception types whose text is known
        not to embed credentials, and the type name otherwise -- including when
        the message is empty, as `str(OSError())` is. Both call sites are
        themselves error handlers, so this must not raise.
    """
    if isinstance(exc, (OSError, ValidationError, TypeError, ValueError)):
        first = str(exc).splitlines()
        if first and first[0]:
            return first[0]
    return type(exc).__name__


__all__ = [
    "DeviceAuthorizationCompletedError",
    "FileTokenStorage",
    "MCPAuthorizationError",
    "build_oauth_provider",
    "extract_oauth_callback_url",
    "format_login_error",
    "prepare_device_client",
    "prepare_oauth_login",
]
