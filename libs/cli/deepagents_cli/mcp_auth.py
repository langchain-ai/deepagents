"""OAuth login flow and token storage for MCP servers.

Note: `mcp.shared.auth.OAuthToken` is a pydantic model whose default
`repr` includes the access and refresh token strings verbatim. Never
log one via `%r`, `str()`, f-string interpolation, or
`logger.exception`/`exc_info` on an exception that wraps one — the
tokens will land in stdout, log files, and error-reporting
pipelines. Pass only structural facts ("refreshed token for
server X") rather than the token itself.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import stat
from collections.abc import Awaitable, Callable
from enum import Enum
from pathlib import Path
from typing import Literal, TypedDict
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import (
    AnyUrl,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)
from pydantic import BaseModel, ConfigDict, ValidationError


class _DeviceCodeResponse(BaseModel):
    """RFC 8628 §3.2 device-authorization response payload."""

    model_config = ConfigDict(extra="ignore")

    device_code: str
    """Opaque device code the client polls with at the token endpoint."""

    user_code: str
    """Short code the user enters in the browser to approve the device."""

    verification_uri: str
    """Provider URL the user visits to complete device authorization."""

    expires_in: int
    """Lifetime of the device code in seconds."""

    interval: int = 5
    """Recommended polling interval in seconds when the provider omits one."""


class McpServerSpec(TypedDict, total=False):
    """Parsed MCP server config entry.

    All keys are optional at the type level because `mcpServers` entries
    are validated shape-first by `_validate_server_config` rather than by
    the type system. This TypedDict documents the accepted shape for
    readers and static checkers — validate the fields at use sites before
    relying on them.
    """

    auth: Literal["oauth"]
    """Authentication mode for remote MCP servers that require OAuth login."""

    type: Literal["stdio", "http", "sse"]
    """Transport type when the config uses the `type` key."""

    transport: Literal["stdio", "http", "sse"]
    """Transport type when the config uses the `transport` key."""

    url: str
    """Remote endpoint URL for HTTP or SSE MCP servers."""

    headers: dict[str, str]
    """Optional request headers sent when connecting to the remote server."""

    command: str
    """Executable for stdio MCP servers."""

    args: list[str]
    """Command-line arguments passed to the stdio server executable."""

    env: dict[str, str]
    """Environment overrides for launching a stdio MCP server."""


logger = logging.getLogger(__name__)

_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
"""Matches `${VAR}` placeholders inside config strings for env-var substitution."""

_STORAGE_VERSION = 1
"""Schema version stamped into persisted credential files; bump on incompatible
shape changes so `_load_*` can reject or migrate older payloads."""

# Public OAuth client IDs — safe to check in. No secret is associated, and
# Slack/GitHub treat these as browser-style public clients where the security
# boundary is the redirect URI / device flow rather than client secrecy.
_SLACK_MCP_CLIENT_ID = "4518649543379.10944517634130"
"""Public OAuth client ID registered with Slack for the hosted MCP endpoint."""

_SLACK_REDIRECT_URI = "https://localhost"
"""Loopback redirect URI Slack hands the authorization code back to; the user
copy-pastes the resulting URL into the CLI rather than running a local server."""

_GITHUB_MCP_CLIENT_ID = "Iv23libxz8qOApH0WQL3"
"""Public OAuth client ID for the GitHub App backing GitHub's remote MCP."""

_GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
"""GitHub Device Authorization Grant endpoint that issues the user/device code pair."""

_GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"  # noqa: S105
"""GitHub OAuth token endpoint polled while the user completes the device flow."""


class _OAuthFlavor(Enum):
    """Provider-specific OAuth dispatch label.

    Each flavor encodes a set of provider facts — whether login uses the
    Authorization Code paste-back dance vs. the Device Authorization
    Grant, the hardcoded client ID (if any), redirect URI requirements,
    and whether extra auth-URL params are accepted. The login flow
    branches on this enum instead of re-running URL predicates at every
    site, and new flavors are added by extending this one enum plus
    `_detect_flavor`.
    """

    SLACK = "slack"
    """Slack-hosted MCP: Authorization Code flow with a hardcoded public
    client ID and the loopback redirect URI; the user pastes the
    redirected URL back into the CLI."""

    GITHUB_DEVICE = "github_device"
    """GitHub-hosted MCP: Device Authorization Grant against
    `github.com/login/device/code`; the CLI polls the token endpoint
    while the user approves the device code in a browser."""

    GENERIC = "generic"
    """Spec-compliant MCP server: Dynamic Client Registration plus the
    standard Authorization Code + PKCE flow — used when no provider
    special-case fires."""


def _is_slack_mcp_url(url: str) -> bool:
    """Return `True` when `url` points at a Slack-hosted MCP endpoint."""
    host = urlparse(url).hostname or ""
    return host == "slack.com" or host.endswith(".slack.com")


def _is_github_mcp_url(url: str) -> bool:
    """Return `True` when `url` points at GitHub's remote MCP endpoint."""
    return (urlparse(url).hostname or "") == "api.githubcopilot.com"


def _detect_flavor(server_url: str) -> _OAuthFlavor:
    """Classify `server_url` into a provider-specific OAuth dispatch label.

    Args:
        server_url: Remote MCP endpoint URL.

    Returns:
        The matching `_OAuthFlavor` (`GENERIC` when no special-case fires).
    """
    if _is_slack_mcp_url(server_url):
        return _OAuthFlavor.SLACK
    if _is_github_mcp_url(server_url):
        return _OAuthFlavor.GITHUB_DEVICE
    return _OAuthFlavor.GENERIC


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


def resolve_headers(
    headers: dict[str, str],
    *,
    server_name: str | None = None,
) -> dict[str, str]:
    """Resolve `${VAR}` env-var references in header values.

    Args:
        headers: Raw header mapping from MCP config.
        server_name: Optional server name for error messages.

    Returns:
        A new dict with env-var references resolved to current values.

    Raises:
        TypeError: If a header value is not a string.
        RuntimeError: If a `${VAR}` reference points to an unset env var.
    """  # noqa: DOC502 - RuntimeError is raised via `_interpolate`
    resolved: dict[str, str] = {}
    for name, value in headers.items():
        if not isinstance(value, str):
            where = f"mcpServers.{server_name}.headers.{name}" if server_name else name
            msg = f"{where} must be a string, got {type(value).__name__}"
            raise TypeError(msg)
        resolved[name] = _interpolate(value, header=name, server_name=server_name)
    return resolved


def _interpolate(s: str, *, header: str, server_name: str | None) -> str:
    """Expand `${VAR}` references in `s` against the current environment.

    Args:
        s: Raw header value.
        header: Header name, used in error messages.
        server_name: Owning server name for error messages.

    Returns:
        Interpolated string.

    Raises:
        RuntimeError: If a referenced env var is unset.
    """  # noqa: DOC502 - raised inside the inner `replace` substitution callback

    def replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        val = os.environ.get(var_name)
        if val is None:
            where = (
                f"mcpServers.{server_name}.headers.{header}" if server_name else header
            )
            msg = (
                f"{where} references unset env var {var_name}. "
                f"Set {var_name} in the environment or remove the reference."
            )
            raise RuntimeError(msg)
        return val

    return _REF_RE.sub(replace, s)


def _tokens_dir() -> Path:
    """Return `~/.deepagents/mcp-tokens/`."""
    return Path.home() / ".deepagents" / "mcp-tokens"


def _token_file_stem(server_name: str, server_url: str | None) -> str:
    """Return a path-safe storage stem for this server identity.

    Safety of the stem depends on `server_name` already having passed
    `_SERVER_NAME_RE` in `_validate_server_config` — the URL is hashed
    to a hex digest, so only the server name can carry path separators.
    """
    if server_url is None:
        return server_name
    digest = hashlib.sha256(server_url.encode("utf-8")).hexdigest()[:16]
    return f"{server_name}-{digest}"


class FileTokenStorage(TokenStorage):
    """File-backed `TokenStorage` under `~/.deepagents/mcp-tokens/`."""

    def __init__(self, server_name: str, *, server_url: str | None = None) -> None:
        """Bind this storage to a configured MCP server identity."""
        self._server_name = server_name
        self._server_url = server_url

    @property
    def path(self) -> Path:
        """Return the on-disk token file path for this server."""
        stem = _token_file_stem(self._server_name, self._server_url)
        return _tokens_dir() / f"{stem}.json"

    async def get_tokens(self) -> OAuthToken | None:
        """Return the stored `OAuthToken`, or `None` if none is persisted."""
        data = self._read()
        if data is None:
            return None
        raw = data.get("tokens")
        if raw is None:
            return None
        return OAuthToken.model_validate(raw)

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Persist `tokens` to disk, preserving any stored client info."""
        data = self._read() or {}
        data["version"] = _STORAGE_VERSION
        data["tokens"] = json.loads(tokens.model_dump_json(exclude_none=True))
        self._write(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """Return the stored client registration, or `None` if none is persisted."""
        data = self._read()
        if data is None:
            return None
        raw = data.get("client_info")
        if raw is None:
            return None
        return OAuthClientInformationFull.model_validate(raw)

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Persist `client_info` to disk, preserving any stored tokens."""
        data = self._read() or {}
        data["version"] = _STORAGE_VERSION
        data["client_info"] = json.loads(client_info.model_dump_json(exclude_none=True))
        self._write(data)

    async def set_tokens_and_client_info(
        self,
        tokens: OAuthToken,
        client_info: OAuthClientInformationFull,
    ) -> None:
        """Persist tokens and client info in a single atomic write.

        Prevents the state where one call succeeds and the other fails,
        leaving an orphan on disk.
        """
        data = self._read() or {}
        data["version"] = _STORAGE_VERSION
        data["tokens"] = json.loads(tokens.model_dump_json(exclude_none=True))
        data["client_info"] = json.loads(client_info.model_dump_json(exclude_none=True))
        self._write(data)

    def _read(self) -> dict | None:
        path = self.path
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            msg = (
                f"Failed to read MCP token file {path}: {exc}. "
                "Delete the file and re-run `deepagents mcp login "
                f"{self._server_name}` if it is corrupt."
            )
            raise RuntimeError(msg) from exc
        if data.get("version") != _STORAGE_VERSION:
            msg = (
                f"MCP token file {path} has unsupported version "
                f"{data.get('version')!r} (expected {_STORAGE_VERSION}). "
                "Delete it and re-run `deepagents mcp login "
                f"{self._server_name}`."
            )
            raise RuntimeError(msg)
        return data

    def _write(self, data: dict) -> None:
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "chmod"):
            try:
                path.parent.chmod(stat.S_IRWXU)
            except OSError as exc:
                # A failing chmod on the parent dir leaves the tokens
                # directory at the default umask. Warn so operators on
                # shared hosts notice.
                logger.warning(
                    "Could not lock down MCP tokens dir %s (mode 0700): %s. "
                    "Tokens may be readable by other local users.",
                    path.parent,
                    exc,
                )
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = json.dumps(data, separators=(",", ":")).encode("utf-8")
        # O_EXCL + mode 0600 means the token file is never visible at the
        # default umask between open() and chmod(). On Windows, os.open()
        # ignores the mode bits, so the explicit chmod below is the
        # cross-platform guarantee.
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        fd = os.open(str(tmp), flags, 0o600)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
        except Exception:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        try:
            tmp.replace(path)
        except Exception:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        if hasattr(os, "chmod"):
            # Already 0600 from os.open on POSIX; a second chmod covers
            # filesystems that ignore the create-mode argument.
            try:
                path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            except OSError as exc:
                logger.warning(
                    "Could not set mode 0600 on MCP token file %s: %s. "
                    "Stored refresh/access tokens may be world-readable.",
                    path,
                    exc,
                )


RedirectHandler = Callable[[str], Awaitable[None]]
CallbackHandler = Callable[[], Awaitable[tuple[str, str | None]]]


class MCPReauthRequiredError(RuntimeError):
    """Raised when an MCP server needs interactive re-authentication."""

    def __init__(self, server_name: str) -> None:
        """Build with `server_name` so the message tells the user what to fix."""
        self.server_name = server_name
        super().__init__(
            f"MCP server {server_name!r} needs re-authentication. "
            f"Run: deepagents mcp login {server_name}"
        )


def _make_reauth_required_handlers(
    server_name: str,
) -> tuple[RedirectHandler, CallbackHandler]:
    """Return OAuth handlers that refuse to prompt and raise instead.

    Used in non-interactive server mode so that a missing or expired token
    surfaces as `MCPReauthRequiredError` rather than hanging on `input()`.
    """

    async def redirect(_auth_url: str) -> None:  # noqa: RUF029
        raise MCPReauthRequiredError(server_name)

    async def callback() -> tuple[str, str | None]:  # noqa: RUF029
        raise MCPReauthRequiredError(server_name)

    return redirect, callback


def _make_paste_back_handlers(
    *, extra_auth_params: dict[str, str] | None = None
) -> tuple[RedirectHandler, CallbackHandler]:
    """Create paste-back redirect and callback handlers for OAuth.

    Args:
        extra_auth_params: Extra query params to append to the auth URL.

    Returns:
        A tuple of `(redirect_handler, callback_handler)`.
    """
    extras = dict(extra_auth_params or {})

    async def redirect(auth_url: str) -> None:  # noqa: RUF029
        final_url = _append_query_params(auth_url, extras) if extras else auth_url
        print(  # noqa: T201 - intentional user-facing prompt
            "\nOpen this URL in a browser, approve access, then paste the full "
            "callback URL back here:\n"
            f"\n  {final_url}\n"
        )

    async def callback() -> tuple[str, str | None]:
        import asyncio

        try:
            raw = await asyncio.to_thread(input, "Callback URL: ")
        except EOFError as exc:
            msg = (
                "No callback URL received (stdin closed). "
                "Re-run `deepagents mcp login <server>` and paste the URL."
            )
            raise RuntimeError(msg) from exc
        url = raw.strip()
        params = parse_qs(urlparse(url).query)
        if "error" in params:
            err_code = params["error"][0]
            err_desc = (params.get("error_description") or [""])[0]
            detail = f": {err_desc}" if err_desc else ""
            msg = f"Authorization denied by provider: {err_code}{detail}"
            raise RuntimeError(msg)
        if "code" not in params or not params["code"]:
            msg = "Callback URL is missing the 'code' parameter."
            raise RuntimeError(msg)
        return params["code"][0], (params.get("state") or [None])[0]

    return redirect, callback


def _append_query_params(url: str, params: dict[str, str]) -> str:
    """Return `url` with `params` replacing any same-named query keys."""
    from urllib.parse import urlencode, urlunparse

    parsed = urlparse(url)
    existing = dict(parse_qs(parsed.query, keep_blank_values=True))
    for key, value in params.items():
        existing[key] = [value]
    return urlunparse(parsed._replace(query=urlencode(existing, doseq=True)))


def build_oauth_provider(
    *,
    server_name: str,
    server_url: str,
    storage: TokenStorage,
    extra_auth_params: dict[str, str] | None = None,
    interactive: bool = True,
) -> OAuthClientProvider:
    """Construct an `OAuthClientProvider` for an MCP server.

    Args:
        server_name: MCP server name used in re-auth messages.
        server_url: Remote MCP server URL.
        storage: Token storage implementation for this server.
        extra_auth_params: Optional query params for the interactive auth URL.
        interactive: Whether the provider may prompt on stdin.

    Returns:
        A configured `OAuthClientProvider`.
    """
    if interactive:
        redirect, callback = _make_paste_back_handlers(
            extra_auth_params=extra_auth_params
        )
    else:
        redirect, callback = _make_reauth_required_handlers(server_name=server_name)

    if _detect_flavor(server_url) is _OAuthFlavor.SLACK:
        metadata = OAuthClientMetadata(
            client_name="deepagents-cli",
            redirect_uris=[AnyUrl(_SLACK_REDIRECT_URI)],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",  # noqa: S106
        )
    else:
        metadata = OAuthClientMetadata(
            client_name="deepagents-cli",
            redirect_uris=[AnyUrl("http://localhost/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
        )

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=metadata,
        storage=storage,
        redirect_handler=redirect,
        callback_handler=callback,
    )


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


async def _run_device_flow(
    *,
    device_code_url: str,
    token_url: str,
    client_id: str,
    scope: str | None = None,
) -> OAuthToken:
    """Run OAuth 2.0 Device Authorization Grant and return the token.

    Args:
        device_code_url: Provider endpoint that issues a device + user code.
        token_url: Provider endpoint to poll for the access token.
        client_id: Registered OAuth client ID.
        scope: Optional space-delimited scope string.

    Returns:
        The issued OAuth access token payload.

    Raises:
        RuntimeError: If the device flow fails, times out, or the provider
            returns an unexpected HTTP status on the device-code request.
    """
    import asyncio

    import httpx

    init_data = {"client_id": client_id}
    if scope is not None:
        init_data["scope"] = scope

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            device_code_url,
            data=init_data,
            headers={"Accept": "application/json"},
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            msg = (
                f"Device code request failed: HTTP {response.status_code} "
                f"from {device_code_url}."
            )
            raise RuntimeError(msg) from exc
        try:
            device = _DeviceCodeResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            msg = (
                f"Device code response from {device_code_url} is missing "
                f"required fields: {exc}"
            )
            raise RuntimeError(msg) from exc

        print(  # noqa: T201
            f"\nVisit {device.verification_uri} and enter code: "
            f"{device.user_code}\n(code expires in {device.expires_in}s)\n"
        )

        interval = max(device.interval, 1)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + device.expires_in
        while loop.time() < deadline:
            await asyncio.sleep(interval)
            token_response = await client.post(
                token_url,
                data={
                    "client_id": client_id,
                    "device_code": device.device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                headers={"Accept": "application/json"},
            )
            # RFC 8628 §3.5 lets providers return `authorization_pending` /
            # `slow_down` with either a 200 or 400 response. Check the body
            # before raise_for_status so 400-returning providers work.
            try:
                body = token_response.json()
            except ValueError:
                body = {}
            err = body.get("error")
            if err == "authorization_pending":
                continue
            if err == "slow_down":
                interval += 5
                continue
            if err:
                msg = f"Device flow failed: {err}: {body.get('error_description', '')}"
                raise RuntimeError(msg)
            try:
                token_response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                msg = (
                    f"Token request failed: HTTP {token_response.status_code} "
                    f"from {token_url}."
                )
                raise RuntimeError(msg) from exc
            try:
                return OAuthToken.model_validate(body)
            except ValidationError as exc:
                msg = (
                    f"Token response from {token_url} is not a valid "
                    f"OAuth token payload: {exc}"
                )
                raise RuntimeError(msg) from exc

    msg = "Device flow timed out; re-run `deepagents mcp login <server>`."
    raise RuntimeError(msg)


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


def find_reauth_required(exc: BaseException) -> MCPReauthRequiredError | None:
    """Find an `MCPReauthRequiredError` anywhere inside `exc`'s tree.

    Walks `exceptions` (for `ExceptionGroup`), then `__cause__` and
    `__context__`, tracking visited nodes to terminate on cyclic chains.

    Args:
        exc: Root exception to inspect.

    Returns:
        The nested `MCPReauthRequiredError`, or `None` if not present.
    """
    visited: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, MCPReauthRequiredError):
            return current
        sub_exceptions = getattr(current, "exceptions", None)
        if sub_exceptions:
            stack.extend(sub_exceptions)
        cause = current.__cause__ or current.__context__
        if cause is not None:
            stack.append(cause)
    return None


async def _drive_handshake(connections: dict) -> None:
    """Open a one-shot MCP session for `connections` to trigger OAuth handshake."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(connections=connections)
    server_name = next(iter(connections))
    async with client.session(server_name):
        pass


async def login(
    *,
    server_name: str,
    server_config: McpServerSpec,
) -> None:
    """Drive OAuth login for `server_name`, persisting tokens on success.

    Args:
        server_name: Name of the configured MCP server.
        server_config: Parsed server config for that entry.

    Raises:
        ValueError: If `server_config` isn't an OAuth http/sse server.
        RuntimeError: If header env-var interpolation fails, the device
            flow fails or times out, or the OAuth handshake aborts.
    """  # noqa: DOC502 - `RuntimeError` surfaces via `resolve_headers` / `_run_device_flow`
    from langchain_mcp_adapters.sessions import (
        SSEConnection,
        StreamableHttpConnection,
    )

    if server_config.get("auth") != "oauth":
        msg = (
            f"Server '{server_name}' does not use OAuth "
            '(set "auth": "oauth" in mcpServers).'
        )
        raise ValueError(msg)

    transport = server_config.get("type") or server_config.get("transport", "stdio")
    if transport not in {"http", "sse"}:
        msg = (
            f"Server '{server_name}' uses {transport!r} transport; "
            "OAuth login is only valid for http/sse."
        )
        raise ValueError(msg)

    storage = FileTokenStorage(server_name, server_url=server_config["url"])
    flavor = _detect_flavor(server_config["url"])

    if flavor is _OAuthFlavor.GITHUB_DEVICE:
        await _preseed_github_auth(storage)
        print(  # noqa: T201
            f"Logged in to MCP server '{server_name}'. Tokens saved to {storage.path}."
        )
        return

    extra_auth_params: dict[str, str] = {}
    if flavor is _OAuthFlavor.SLACK:
        await _preseed_slack_client_info(storage)
        team_id = await _prompt_slack_team()
        if team_id:
            extra_auth_params["team"] = team_id

    provider = build_oauth_provider(
        server_name=server_name,
        server_url=server_config["url"],
        storage=storage,
        extra_auth_params=extra_auth_params or None,
    )
    conn: StreamableHttpConnection | SSEConnection
    if transport == "http":
        conn = StreamableHttpConnection(
            transport="streamable_http",
            url=server_config["url"],
            auth=provider,
        )
    else:
        conn = SSEConnection(
            transport="sse",
            url=server_config["url"],
            auth=provider,
        )

    if "headers" in server_config:
        conn["headers"] = resolve_headers(
            server_config["headers"],
            server_name=server_name,
        )

    await _drive_handshake({server_name: conn})
    print(  # noqa: T201
        f"Logged in to MCP server '{server_name}'. Tokens saved to {storage.path}."
    )
