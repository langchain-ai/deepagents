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

import asyncio
import concurrent.futures
import contextlib
import contextvars
import hashlib
import html
import json
import logging
import os
import re
import secrets
import stat
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, TypedDict
from urllib.parse import parse_qs, urlparse

import httpx
from filelock import FileLock, Timeout
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
    create_oauth_metadata_request,
    handle_auth_metadata_response,
    handle_protected_resource_response,
)
from mcp.client.streamable_http import MCP_PROTOCOL_VERSION
from mcp.shared.auth import (
    AnyUrl,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthMetadata,
    OAuthToken,
)
from pydantic import BaseModel, ConfigDict, ValidationError
from typing_extensions import override

if TYPE_CHECKING:
    from pathlib import Path

    from mcp.client.auth.oauth2 import OAuthContext

    from deepagents_code.mcp_oauth_ui import OAuthInteraction


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

    oauth: OAuthClientConfig
    """Optional configured OAuth client credentials for a remote server."""

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


class OAuthClientConfig(TypedDict):
    """Configured OAuth client credentials for a remote MCP server."""

    clientId: str
    """OAuth client ID, optionally with `${VAR}` references."""

    clientSecret: str
    """OAuth client secret, optionally with `${VAR}` references."""

    redirectUri: str
    """Registered loopback callback URI, optionally with `${VAR}` references."""


logger = logging.getLogger(__name__)
_SUPPRESS_EXPECTED_REAUTH_LOGS: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "suppress_expected_mcp_reauth_logs",
    default=False,
)


_TOKEN_REFRESH_FAILED_PREFIX = "Token refresh failed: "  # noqa: S105  # log-message prefix, not a credential
"""Prefix of the SDK's `Token refresh failed: <status>` warning (`oauth2.py`)."""

_EXPECTED_REAUTH_REFRESH_STATUSES = frozenset({"400", "401", "403"})
"""Refresh-endpoint statuses that mean the grant was rejected (token stale).

The SDK logs `Token refresh failed: <status>` and clears tokens for *any*
non-200 on the refresh endpoint. Only these statuses indicate the refresh
token itself is expired/revoked — i.e. the expected re-auth cases our hint
replaces. Transient failures (`429`, `5xx`, gateway timeouts) must stay
visible so a provider outage isn't silently relabeled as "go re-login".
"""
_EXPECTED_REAUTH_REFRESH_STATUS_CODES = frozenset(
    int(status) for status in _EXPECTED_REAUTH_REFRESH_STATUSES
)


class _ExpectedReauthLogFilter(logging.Filter):
    """Drop SDK OAuth log records that are replaced by our reauth hint."""

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        """Return whether the SDK OAuth log record should be emitted."""
        if not _SUPPRESS_EXPECTED_REAUTH_LOGS.get():
            return True
        message = record.getMessage()
        if message.startswith(_TOKEN_REFRESH_FAILED_PREFIX):
            status = message.removeprefix(_TOKEN_REFRESH_FAILED_PREFIX)
            if status in _EXPECTED_REAUTH_REFRESH_STATUSES:
                return False
        if message == "OAuth flow error" and record.exc_info is not None:
            exc = record.exc_info[1]
            if exc is not None and find_reauth_required(exc) is not None:
                return False
        return True


logging.getLogger("mcp.client.auth.oauth2").addFilter(_ExpectedReauthLogFilter())

_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
"""Matches `${VAR}` placeholders inside config strings for env-var substitution."""

_SAFE_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
"""Matches server names that are safe to embed in token-file basenames.

Mirrors `_SERVER_NAME_RE` in `mcp_tools` — duplicated here because
`mcp_auth` cannot import from `mcp_tools` at module top-level without
risking a circular import. Keep both regexes in sync."""

_STORAGE_VERSION = 1
"""Schema version stamped into persisted credential files; bump on incompatible
shape changes so `_load_*` can reject or migrate older payloads."""

_REFRESH_SAFETY_MARGIN_SECONDS = 30.0
"""Refresh access tokens this many seconds before their advertised expiry.

Absorbs clock skew and request latency so a token deemed valid locally isn't
rejected as expired by the server — without the margin, a 401 sends the SDK
into the full re-auth (browser) flow instead of the cheaper refresh grant.
"""

_REFRESH_LOCK_TIMEOUT_SECONDS = 60.0
"""Longest a provider waits for the cross-process token-refresh lock.

Bounds the wait so a live-but-stuck peer (e.g. one whose refresh network call
hangs) can't block tool calls indefinitely. A peer that outright crashes is not
the concern: the OS releases an `fcntl` lock when the holding process exits. On
timeout the provider reloads tokens from disk and avoids using any still-stale
refresh token (see `_acquire_refresh_lock`).
"""


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
        resolved[name] = _interpolate(
            value,
            field=f"headers.{name}",
            server_name=server_name,
        )
    return resolved


def resolve_oauth_client_config(
    config: OAuthClientConfig,
    *,
    server_name: str,
) -> OAuthClientInformationFull:
    """Resolve a configured OAuth client only when its server is activated.

    Args:
        config: Validated configured OAuth client values from MCP configuration.
        server_name: Owning MCP server name for interpolation errors.

    Returns:
        Static client information for the OAuth authorization-code flow.

    Raises:
        ValueError: If interpolation produces a non-local callback URI.
    """
    redirect_uri = _interpolate(
        config["redirectUri"],
        field="oauth.redirectUri",
        server_name=server_name,
    )
    if not _is_loopback_callback_uri(redirect_uri):
        msg = (
            f"mcpServers.{server_name}.oauth.redirectUri must be an exact "
            "http://localhost:<port>/callback URI."
        )
        raise ValueError(msg)
    return OAuthClientInformationFull(
        client_id=_interpolate(
            config["clientId"], field="oauth.clientId", server_name=server_name
        ),
        client_secret=_interpolate(
            config["clientSecret"],
            field="oauth.clientSecret",
            server_name=server_name,
        ),
        redirect_uris=[AnyUrl(redirect_uri)],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="client_secret_post",  # noqa: S106  # OAuth method, not a secret
    )


def _is_loopback_callback_uri(uri: str) -> bool:
    """Return whether `uri` is a canonical local OAuth callback URI."""
    try:
        parsed = urlparse(uri)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "http"
        and parsed.hostname == _LOOPBACK_URI_HOST
        and port is not None
        and uri == f"http://{_LOOPBACK_URI_HOST}:{port}{_LOOPBACK_CALLBACK_PATH}"
    )


def _interpolate(s: str, *, field: str, server_name: str | None) -> str:
    """Expand `${VAR}` references in `s` against the current environment.

    Args:
        s: Raw header value.
        field: Configuration field name, used in error messages.
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
                f"mcpServers.{server_name}.{field}" if server_name else field
            )
            msg = (
                f"{where} references unset env var {var_name}. "
                f"Set {var_name} in the environment or remove the reference."
            )
            raise RuntimeError(msg)
        return val

    return _REF_RE.sub(replace, s)


def _tokens_dir() -> Path:
    """Return `~/.deepagents/.state/mcp-tokens/`.

    The deferred import lets tests redirect token storage into a temp
    directory by patching `deepagents_code.model_config.DEFAULT_STATE_DIR`.
    """
    from deepagents_code.model_config import DEFAULT_STATE_DIR

    return DEFAULT_STATE_DIR / "mcp-tokens"


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
    """File-backed `TokenStorage` under `~/.deepagents/.state/mcp-tokens/`."""

    def __init__(self, server_name: str, *, server_url: str | None = None) -> None:
        """Bind this storage to a configured MCP server identity.

        Raises:
            ValueError: If `server_name` contains characters that would let
                it escape the `~/.deepagents/.state/mcp-tokens/` directory
                when used as the token-file basename.
        """
        if not _SAFE_SERVER_NAME_RE.fullmatch(server_name):
            msg = (
                f"Invalid MCP server name {server_name!r}: token storage "
                "names must match [A-Za-z0-9_-]+ to keep the on-disk path "
                "inside ~/.deepagents/.state/mcp-tokens/."
            )
            raise ValueError(msg)
        self._server_name = server_name
        self._server_url = server_url

    @property
    def path(self) -> Path:
        """On-disk token file path for this server."""
        stem = _token_file_stem(self._server_name, self._server_url)
        return _tokens_dir() / f"{stem}.json"

    @property
    def refresh_lock_path(self) -> Path:
        """Sibling lock file that serializes token refreshes across processes.

        A dedicated `.lock` file (never the token file itself) lets `filelock`
        coordinate refreshes between dcode processes and provider instances
        without ever holding an exclusive lock on the credential file. It holds
        no token material.
        """
        path = self.path
        return path.with_name(f"{path.name}.lock")

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
        """Persist `tokens` to disk, preserving any stored client info.

        Also records the absolute Unix-epoch expiry as a sidecar so a
        cold-started provider can detect a stale access token and trigger
        the SDK's `refresh_token` grant instead of a full browser re-auth.
        Cleared when `expires_in` is absent so the sidecar can't go stale.
        """
        data = self._read() or {}
        data["version"] = _STORAGE_VERSION
        data["tokens"] = json.loads(tokens.model_dump_json(exclude_none=True))
        if tokens.expires_in is not None:
            data["expires_at"] = time.time() + tokens.expires_in
        else:
            data.pop("expires_at", None)
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

    async def get_oauth_metadata(self) -> OAuthMetadata | None:
        """Return stored public OAuth authorization metadata, if available."""
        data = self._read()
        if data is None:
            return None
        raw = data.get("oauth_metadata")
        if raw is None:
            return None
        return OAuthMetadata.model_validate(raw)

    async def set_oauth_metadata(self, metadata: OAuthMetadata) -> None:
        """Persist public OAuth authorization metadata beside the token state."""
        data = self._read() or {}
        data["version"] = _STORAGE_VERSION
        data["oauth_metadata"] = json.loads(metadata.model_dump_json(exclude_none=True))
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
        if tokens.expires_in is not None:
            data["expires_at"] = time.time() + tokens.expires_in
        else:
            data.pop("expires_at", None)
        self._write(data)

    async def get_expires_at(self) -> float | None:
        """Return the stored absolute token expiry (Unix epoch), or `None`.

        Returns `None` for token files written before this field existed,
        for tokens whose provider omitted `expires_in`, or when the sidecar
        value fails to coerce to `float`. Callers should treat `None` as
        "expiry unknown" and decide policy (skip, assume-expired, etc.).
        """
        data = self._read()
        if data is None:
            return None
        raw = data.get("expires_at")
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            # Log the value's type, never the value itself — the sidecar lives
            # next to the bearer token in the same JSON envelope, so a
            # misplaced token string would land here.
            logger.warning(
                "MCP token sidecar 'expires_at' for %s is not numeric (%s); "
                "treating as unknown. The next request will trigger a refresh "
                "or browser re-auth.",
                self._server_name,
                type(raw).__name__,
            )
            return None

    def stored_loopback_port(self) -> int | None:
        """Return the stored loopback redirect URI port, if one is reusable.

        DCR registers `client_id` against a specific `redirect_uri`. If the
        callback server binds a fresh random port on a later launch, the
        authorize request will carry a `redirect_uri` that no longer matches
        the one registered with the persisted `client_id`, and the
        authorization server will reject it ("invalid or missing redirect_uri").
        Reusing the persisted port keeps the registration valid across runs.

        Returns:
            The integer port parsed from a stored
                `http://localhost:<port>/callback` redirect URI, or `None` if
                no usable port is on disk.
        """
        try:
            data = self._read()
        except RuntimeError as exc:
            logger.warning(
                "MCP token file for %s is unreadable during loopback port "
                "lookup; falling back to a fresh random port. Delete the file "
                "and log in again if OAuth authorization fails: %s",
                self.path,
                exc,
            )
            return None
        if data is None:
            return None
        client_info = data.get("client_info") or {}
        redirect_uris = client_info.get("redirect_uris") or []
        if not redirect_uris:
            return None
        uri = str(redirect_uris[0])
        port = self._loopback_callback_port(uri)
        if port is None:
            logger.warning(
                "Stored MCP OAuth redirect URI for %s is not a reusable "
                "loopback callback URI; falling back to a fresh random port. "
                "OAuth authorization may fail if the server requires the "
                "persisted client registration redirect URI: %s",
                self.path,
                uri,
            )
            return None
        return port

    @staticmethod
    def _loopback_callback_port(uri: str) -> int | None:
        """Return `uri`'s port if it is a reusable loopback callback URI.

        A reusable URI is `http://localhost:<port>/callback` with an explicit
        port. Anything else (a different scheme/host/path, or a portless
        `http://localhost/callback`) returns `None` because it cannot be paired
        with the loopback callback server a CLI login binds.
        """
        parsed = urlparse(uri)
        try:
            port = parsed.port
        except ValueError:
            return None
        if (
            parsed.scheme != "http"
            or parsed.hostname != _LOOPBACK_URI_HOST
            or parsed.path != _LOOPBACK_CALLBACK_PATH
            or port is None
        ):
            return None
        return port

    def discard_client_info_if_loopback_unusable(self) -> bool:
        """Drop a persisted client registration that can't serve loopback login.

        `stored_loopback_port` explains why the authorize request's
        `redirect_uri` must match the one registered against the persisted
        `client_id`. When that registered URI is not a reusable loopback
        callback (e.g. a portless `http://localhost/callback` left by an earlier
        non-loopback login), no port can be reused, so a CLI login binds a fresh
        random port and the server rejects the mismatched `redirect_uri`
        ("invalid or missing redirect_uri"). Removing the stale `client_info`
        makes the SDK perform a fresh DCR with the loopback redirect URI it will
        actually use.

        Only discards when no token is persisted at all, so a session with a
        usable (or refreshable) token is never downgraded to a full re-auth.
        The presence check is deliberately conservative: it errs toward keeping
        the registration, never toward deleting one that might still be needed.

        Returns:
            `True` if a stale client registration was removed. `False` covers
                both "nothing to discard" and "could not discard" (an
                unreadable or unwritable token file, logged where it happens).
        """
        try:
            data = self._read()
        except RuntimeError as exc:
            # Mirror `stored_loopback_port`: a corrupt or unsupported-version
            # token file carries actionable "delete the file" guidance in the
            # `RuntimeError` message, so surface it rather than dropping it.
            logger.warning(
                "MCP token file for %s is unreadable while checking for a "
                "stale client registration; skipping self-heal. Delete the "
                "file and log in again if OAuth authorization fails: %s",
                self.path,
                exc,
            )
            return False
        if data is None or "client_info" not in data:
            return False
        # A persisted access/refresh token can still authenticate (or refresh)
        # without re-running the authorization-code grant, so leave the
        # registration intact rather than forcing an avoidable re-auth.
        if data.get("tokens") is not None:
            return False
        redirect_uris = (data.get("client_info") or {}).get("redirect_uris") or []
        if (
            redirect_uris
            and self._loopback_callback_port(str(redirect_uris[0])) is not None
        ):
            return False
        del data["client_info"]
        try:
            self._write(data)
        except OSError as exc:
            # Surface the failure but don't crash login: the stale registration
            # simply remains, and the login attempt fails the same way it would
            # have without this self-heal.
            logger.warning(
                "Could not remove stale MCP client registration in %s: %s",
                self.path,
                exc,
            )
            return False
        return True

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
                f"Delete the file and run `/mcp login {self._server_name}` "
                f"in the TUI (or `dcode mcp login {self._server_name}`)."
            )
            raise RuntimeError(msg) from exc
        if data.get("version") != _STORAGE_VERSION:
            msg = (
                f"MCP token file {path} has unsupported version "
                f"{data.get('version')!r} (expected {_STORAGE_VERSION}). "
                f"Delete it and run `/mcp login {self._server_name}` in the "
                f"TUI (or `dcode mcp login {self._server_name}`)."
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
_LOOPBACK_BIND_HOST = "127.0.0.1"
_LOOPBACK_URI_HOST = "localhost"
_LOOPBACK_CALLBACK_PATH = "/callback"
_LOOPBACK_CALLBACK_TIMEOUT = 300.0


class _LoopbackCallbackTimeoutError(RuntimeError):
    """Raised when the browser never reaches the local callback server."""


class _LoopbackCallbackUnavailableError(RuntimeError):
    """Raised when the local callback server cannot be started."""


def _choose_loopback_port() -> int:
    """Return a high local TCP port candidate without opening a socket.

    The OAuth redirect URI must be known before the provider starts the
    handshake, but the actual callback server should not keep a socket open
    unless a browser redirect is needed.

    Returns:
        A port number from the dynamic/private port range.
    """
    return 49152 + secrets.randbelow(65535 - 49152 + 1)


class _LoopbackOAuthCallbackServer:
    """Single-use loopback HTTP server for CLI OAuth callbacks.

    Port selection and socket binding are intentionally separated: the
    redirect URI is fixed at construction time so it can be registered with
    the OAuth provider, while the socket is not opened until `start()` is
    called from the redirect handler.
    """

    def __init__(self, *, port: int) -> None:
        """Prepare a callback server for a previously selected loopback port.

        Args:
            port: TCP port to bind when `start()` is called.
        """
        self._port = port
        self.redirect_uri = (
            f"http://{_LOOPBACK_URI_HOST}:{port}{_LOOPBACK_CALLBACK_PATH}"
        )
        self._future: concurrent.futures.Future[tuple[str, str | None]] = (
            concurrent.futures.Future()
        )
        self._server: object | None = None
        self._started = False
        self._closed = False
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """Loopback TCP port this server will bind on `start()`."""
        return self._port

    def start(self) -> None:
        """Bind and start serving callback requests in a background thread."""
        if self._started:
            return
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parent._handle_get(self)

            def log_message(  # noqa: PLR6301  # stdlib override
                self,
                format: str,  # noqa: A002
                *args: object,
            ) -> None:
                del format, args

        self._server = ThreadingHTTPServer((_LOOPBACK_BIND_HOST, self._port), Handler)
        self._started = True
        self._thread = threading.Thread(
            target=self._serve_forever,
            name="deepagents-mcp-oauth-callback",
            daemon=True,
        )
        self._thread.start()

    def _serve_forever(self) -> None:
        from http.server import ThreadingHTTPServer

        if isinstance(self._server, ThreadingHTTPServer):
            self._server.serve_forever()

    async def wait(self) -> tuple[str, str | None]:
        """Wait for the authorization callback and return `(code, state)`.

        Returns:
            The OAuth authorization code and optional state.

        Raises:
            _LoopbackCallbackTimeoutError: If no callback arrives before the timeout.
            _LoopbackCallbackUnavailableError: If the server could not be started.
            RuntimeError: If the provider returned an OAuth error or the callback
                URL lacked a `code` parameter.
        """  # noqa: DOC502 - _LoopbackCallbackUnavailableError/RuntimeError set on future
        import asyncio

        try:
            return await asyncio.wait_for(
                asyncio.wrap_future(self._future),
                timeout=_LOOPBACK_CALLBACK_TIMEOUT,
            )
        except TimeoutError as exc:
            msg = "Browser callback was not received before the timeout."
            raise _LoopbackCallbackTimeoutError(msg) from exc

    def fail(self, exc: Exception) -> None:
        """Poison the future so `wait()` raises `exc` immediately.

        Args:
            exc: Exception to surface to the awaiting coroutine.
        """
        if not self._future.done():
            self._future.set_exception(exc)

    def close(self) -> None:
        """Stop the local callback server and release its socket."""
        if self._closed:
            return
        self._closed = True
        from http.server import ThreadingHTTPServer

        if isinstance(self._server, ThreadingHTTPServer):
            if self._started:
                self._server.shutdown()
            self._server.server_close()

    def _handle_get(self, request: object) -> None:
        from http.server import BaseHTTPRequestHandler

        handler = request
        if not isinstance(handler, BaseHTTPRequestHandler):
            return

        if self._future.done():
            # Duplicate browser request (retry, prefetch, favicon) after the
            # flow already completed. Respond and return without touching the
            # future — avoids InvalidStateError from a concurrent set_result.
            # Branch on the future's terminal state: a previous error must
            # not be papered over with a success page.
            if self._future.exception() is None:
                self._send_html(
                    handler,
                    200,
                    _oauth_success_html(
                        "MCP authorization complete. "
                        "You can close this tab and return to your terminal.",
                    ),
                )
            else:
                self._send_html(
                    handler,
                    400,
                    _oauth_error_html(
                        "Authorization did not complete. "
                        "Return to your terminal for details.",
                    ),
                )
            return

        parsed = urlparse(handler.path)
        if parsed.path != _LOOPBACK_CALLBACK_PATH:
            self._send_html(
                handler,
                404,
                _oauth_error_html("Callback route not found."),
            )
            return

        params = parse_qs(parsed.query)
        if "error" in params:
            err_code = params["error"][0]
            err_desc = (params.get("error_description") or [""])[0]
            detail = f": {err_desc}" if err_desc else ""
            msg = f"Authorization denied by provider: {err_code}{detail}"
            self._future.set_exception(RuntimeError(msg))
            self._send_html(handler, 400, _oauth_error_html(msg))
            return

        if "code" not in params or not params["code"]:
            msg = "Callback URL is missing the 'code' parameter."
            self._future.set_exception(RuntimeError(msg))
            self._send_html(handler, 400, _oauth_error_html(msg))
            return

        self._future.set_result((params["code"][0], (params.get("state") or [None])[0]))
        self._send_html(
            handler,
            200,
            _oauth_success_html(
                "MCP authorization complete. "
                "You can close this tab and return to your terminal.",
            ),
        )

    @staticmethod
    def _send_html(handler: object, status: int, body: str) -> None:
        from http.server import BaseHTTPRequestHandler

        if not isinstance(handler, BaseHTTPRequestHandler):
            return
        payload = body.encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)


def _oauth_success_html(message: str) -> str:
    return _oauth_result_html(
        title="Authorization complete",
        heading="You're signed in",
        message=message,
        status="success",
    )


def _oauth_error_html(message: str) -> str:
    return _oauth_result_html(
        title="Authorization failed",
        heading="Authorization failed",
        message=message,
        status="error",
    )


def _oauth_result_html(
    *,
    title: str,
    heading: str,
    message: str,
    status: Literal["success", "error"],
) -> str:
    accent = "#137333" if status == "success" else "#b3261e"
    background = "#eef7f0" if status == "success" else "#fceeee"
    mark = "✓" if status == "success" else "!"
    escaped_title = html.escape(title)
    escaped_heading = html.escape(heading)
    escaped = html.escape(message)
    # `window.close()` is only honored for tabs the script itself opened
    # (browser policy). The loopback flow launches the browser via
    # `webbrowser.open`, so the callback tab was usually opened by the OS and
    # the browser refuses to close it. Attempt the close for the rare
    # script-opened case; the static message already reads correctly whether
    # or not the tab closes, so nothing is rewritten and the text never shifts.
    auto_close = (
        "<script>setTimeout(function(){window.close();},1000);</script>"
        if status == "success"
        else ""
    )
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escaped_title}</title>"
        "<style>"
        "body{margin:0;min-height:100vh;display:grid;place-items:center;"
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "background:#f8faf9;color:#1f2328}"
        ".panel{width:min(480px,calc(100vw - 40px));box-sizing:border-box;"
        "padding:32px;border:1px solid #d8dee4;border-radius:8px;"
        "background:#fff;box-shadow:0 18px 45px rgba(31,35,40,.08)}"
        ".mark{width:44px;height:44px;border-radius:50%;display:grid;"
        "place-items:center;margin-bottom:20px;font-weight:700}"
        "h1{font-size:24px;line-height:1.2;margin:0 0 10px}"
        "p{font-size:15px;line-height:1.5;margin:0;color:#57606a}"
        "</style></head><body>"
        '<main class="panel">'
        f'<div class="mark" style="background:{background};color:{accent}">{mark}</div>'
        f"<h1>{escaped_heading}</h1><p>{escaped}</p>"
        "</main>"
        f"{auto_close}"
        "</body></html>"
    )


class MCPReauthRequiredError(RuntimeError):
    """Raised when an MCP server needs interactive re-authentication."""

    def __init__(self, server_name: str) -> None:
        """Build with `server_name` so the message tells the user what to fix."""
        self.server_name = server_name
        super().__init__(
            f"MCP server {server_name!r} needs re-authentication. "
            f"Run `/mcp login {server_name}` in the TUI, or "
            f"`dcode mcp login {server_name}` from the shell.",
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
    *,
    extra_auth_params: dict[str, str] | None = None,
    ui: OAuthInteraction | None = None,
) -> tuple[RedirectHandler, CallbackHandler]:
    """Create paste-back redirect and callback handlers for OAuth.

    Args:
        extra_auth_params: Extra query params to append to the auth URL.
        ui: Interaction surface for the auth URL display and the
            pasted-back callback URL prompt.

    Returns:
        A tuple of `(redirect_handler, callback_handler)`.
    """
    extras = dict(extra_auth_params or {})
    interaction = ui if ui is not None else _default_ui()

    async def redirect(auth_url: str) -> None:
        final_url = _append_query_params(auth_url, extras) if extras else auth_url
        await interaction.show_authorize_url(final_url, opened_in_browser=False)

    async def callback() -> tuple[str, str | None]:
        url = await interaction.request_callback_url()
        return _parse_callback_url(url)

    return redirect, callback


def _parse_callback_url(url: str) -> tuple[str, str | None]:
    """Parse a provider callback URL into `(code, state)`.

    Args:
        url: Raw callback URL pasted by the user.

    Returns:
        The `code` and optional `state` query parameters.

    Raises:
        RuntimeError: If the URL contains `error=` or lacks `code`.
    """
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


def _default_ui() -> OAuthInteraction:
    """Return the default `OAuthInteraction` implementation (CLI stdio)."""
    from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

    return CliOAuthInteraction()


def _make_loopback_handlers(
    *,
    callback_server: _LoopbackOAuthCallbackServer,
    extra_auth_params: dict[str, str] | None = None,
    ui: OAuthInteraction | None = None,
) -> tuple[RedirectHandler, CallbackHandler]:
    """Create browser loopback redirect and callback handlers for OAuth.

    Args:
        callback_server: Prepared local callback server for this login attempt.
            The socket is bound when the returned redirect handler is first called.
        extra_auth_params: Extra query params to append to the auth URL.
        ui: Interaction surface for the browser-opened or fallback prompts.

    Returns:
        A tuple of `(redirect_handler, callback_handler)`.
    """
    extras = dict(extra_auth_params or {})
    interaction = ui if ui is not None else _default_ui()
    last_authorize_url: str | None = None
    _paste_redirect, paste_callback = _make_paste_back_handlers(
        extra_auth_params=extra_auth_params,
        ui=interaction,
    )

    async def redirect(auth_url: str) -> None:
        import asyncio
        import webbrowser

        nonlocal last_authorize_url
        final_url = _append_query_params(auth_url, extras) if extras else auth_url
        last_authorize_url = final_url

        # Resolve a browser explicitly before opening so headless / SSH
        # environments fall through to paste-back without burning the
        # 300s loopback timeout. `webbrowser.open` can return `True` in
        # those environments even when nothing launches.
        try:
            await asyncio.to_thread(webbrowser.get)
            has_browser = True
        except webbrowser.Error:
            has_browser = False

        if has_browser:
            opened = await asyncio.to_thread(webbrowser.open, final_url)
        else:
            opened = False
        if not opened:
            callback_server.fail(
                _LoopbackCallbackUnavailableError(
                    "No browser is available to complete the OAuth flow.",
                ),
            )
            await interaction.show_authorize_url(final_url, opened_in_browser=False)
            return
        try:
            callback_server.start()
        except OSError as exc:
            logger.warning(
                "Could not start loopback OAuth callback server on port %s: %s",
                callback_server.port,
                exc,
            )
            msg = "Local OAuth callback server could not be started."
            callback_server.fail(_LoopbackCallbackUnavailableError(msg))
            await interaction.show_notice(
                "Could not start the local OAuth callback server.",
            )
            await interaction.show_authorize_url(final_url, opened_in_browser=False)
            return
        await interaction.show_authorize_url(final_url, opened_in_browser=True)

    async def callback() -> tuple[str, str | None]:
        try:
            return await callback_server.wait()
        except (
            _LoopbackCallbackTimeoutError,
            _LoopbackCallbackUnavailableError,
        ) as exc:
            if last_authorize_url is not None:
                await interaction.show_authorize_url(
                    last_authorize_url,
                    opened_in_browser=False,
                )
            await interaction.show_notice(
                f"{exc}\nPaste the full callback URL instead.",
            )
            return await paste_callback()
        finally:
            callback_server.close()

    return redirect, callback


def _append_query_params(url: str, params: dict[str, str]) -> str:
    """Return `url` with `params` replacing any same-named query keys."""
    from urllib.parse import urlencode, urlunparse

    parsed = urlparse(url)
    existing = dict(parse_qs(parsed.query, keep_blank_values=True))
    for key, value in params.items():
        existing[key] = [value]
    return urlunparse(parsed._replace(query=urlencode(existing, doseq=True)))


def _strip_duplicate_client_id_under_basic_auth(context: OAuthContext) -> None:
    """Drop the redundant body `client_id` when token auth uses HTTP Basic.

    The MCP SDK copies `client_id` into the token-request body (on both the
    authorization-code exchange and refresh paths) and, for
    `token_endpoint_auth_method == "client_secret_basic"`, *also* sends it in the
    `Authorization: Basic` header. RFC 6749 §2.3.1 carries the client identity in
    the header for Basic auth, so the body copy is redundant; some authorization
    servers (e.g. Pylon) reject the duplicate identity with an `OAuthTokenError`.
    Wrapping `prepare_token_auth` strips the body `client_id` only when a Basic
    header is present, leaving `client_secret_post`/`none` flows untouched.
    """
    original = context.prepare_token_auth

    def prepare_token_auth(
        data: dict[str, str],
        headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        data, headers = original(data, headers)
        # RFC 7617 makes the auth-scheme token case-insensitive, so match
        # `basic` regardless of casing rather than coupling to the SDK's exact
        # `Basic ` literal.
        if headers.get("Authorization", "").lower().startswith("basic "):
            data = {k: v for k, v in data.items() if k != "client_id"}
        return data, headers

    context.prepare_token_auth = prepare_token_auth  # ty: ignore[invalid-assignment]


class _ExpiryAwareOAuthClientProvider(OAuthClientProvider):
    """`OAuthClientProvider` that restores `token_expiry_time` from storage.

    Upstream `_initialize` loads stored tokens but leaves
    `context.token_expiry_time` at `None`, which makes `is_token_valid`
    report any stored access token — even one that expired hours ago —
    as valid. The SDK then sends a stale `Bearer`, gets a 401, and falls
    into a full re-auth (browser) instead of the `refresh_token` grant.

    Restoring the persisted absolute expiry to the context after load
    lets the SDK's refresh-when-invalid-and-refreshable branch fire on
    the first request after a cold start. When the sidecar is absent
    (older token files written before this field existed), assume the
    token is expired so the refresh path still gets a chance before
    falling back to 401.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._configured_client_info = kwargs.pop("configured_client_info", None)
        self._suppress_expected_reauth_logs = bool(
            kwargs.pop("suppress_expected_reauth_logs", False)
        )
        super().__init__(*args, **kwargs)
        _strip_duplicate_client_id_under_basic_auth(self.context)

    async def _initialize(self) -> None:
        # Overrides a leading-underscore SDK method; behavior depends on
        # `super()._initialize()` populating `context.current_tokens` from
        # storage. If an upstream rename or refactor breaks that contract,
        # the test suite's TestExpiryAwareOAuthClientProvider cases will
        # fail loudly rather than silently regress to the 401-on-restart
        # bug this class exists to prevent.
        await super()._initialize()
        if self._configured_client_info is not None:
            # Configured credentials are intentionally not written to token
            # storage. They also take precedence over an older DCR result for
            # the same server URL, which may otherwise have been restored above.
            self.context.client_info = self._configured_client_info
        await self._apply_stored_expiry()

    async def _apply_stored_expiry(self) -> None:
        """Seed `context.token_expiry_time` from the persisted sidecar.

        Upstream `_initialize` loads stored tokens but leaves the expiry unset,
        so a token whose access portion expired long ago still reports as valid
        and is sent stale. Restoring the absolute expiry recorded beside the
        token lets `is_token_valid` return `False` in time for the cheaper
        refresh grant to fire. Also caches persisted OAuth metadata so the
        refresh uses the advertised token endpoint. Safe to call repeatedly, so
        it doubles as the post-reload expiry refresh.
        """
        if self.context.oauth_metadata is None:
            get_oauth_metadata = getattr(
                self.context.storage,
                "get_oauth_metadata",
                None,
            )
            if get_oauth_metadata is not None:
                self.context.oauth_metadata = await get_oauth_metadata()
        get_expires_at = getattr(self.context.storage, "get_expires_at", None)
        if get_expires_at is None:
            return
        expires_at = await get_expires_at()
        tokens = self.context.current_tokens
        if expires_at is None:
            # Use 1.0 (one second after the Unix epoch) rather than 0.0 so the
            # SDK's `not self.token_expiry_time` falsy-zero check doesn't treat
            # the sentinel as "no expiry known" and mark the token valid again.
            if tokens is not None and tokens.refresh_token:
                self.context.token_expiry_time = 1.0
            elif tokens is not None and tokens.access_token:
                # Legacy file with no refresh_token: nothing we can do to
                # pre-empt expiry. Surface a structural breadcrumb so the
                # 401-then-browser-reauth flow isn't completely silent.
                logger.info(
                    "Legacy MCP token file for %s has no refresh_token; "
                    "cannot pre-empt expiry. The next 401 will trigger "
                    "browser re-auth.",
                    self.context.server_url,
                )
            return
        if expires_at - time.time() < _REFRESH_SAFETY_MARGIN_SECONDS:
            # Token already inside its safety margin (or past it) — likely a
            # cold start after a long pause, or a misconfigured server issuing
            # sub-margin lifetimes. Log only the duration, never any token
            # material.
            logger.debug(
                "MCP token for %s is within %.0fs of expiry on load; "
                "scheduling refresh on next request.",
                self.context.server_url,
                _REFRESH_SAFETY_MARGIN_SECONDS,
            )
        self.context.token_expiry_time = expires_at - _REFRESH_SAFETY_MARGIN_SECONDS

    async def _reload_tokens_from_storage(self) -> None:
        """Re-read persisted tokens so a peer's refresh is observed.

        Another dcode process (or a separate provider instance in this process)
        may have rotated the refresh token on disk while this provider held a
        now-stale copy in memory. Re-reading before deciding to refresh keeps
        this provider from replaying an already-rotated refresh token, which
        the LangSmith OAuth server treats as reuse and punishes by revoking the
        whole identity+client token family.
        """
        self.context.current_tokens = await self.context.storage.get_tokens()
        if self._configured_client_info is not None:
            self.context.client_info = self._configured_client_info
        else:
            client_info = await self.context.storage.get_client_info()
            if client_info is not None:
                self.context.client_info = client_info
        await self._apply_stored_expiry()

    async def _acquire_refresh_lock(self, lock: FileLock) -> bool:
        """Wait for the cross-process refresh lock off the event loop.

        `lock.acquire` blocks for up to `_REFRESH_LOCK_TIMEOUT_SECONDS` while a
        peer finishes its refresh, so it runs in a worker thread to avoid
        stalling the event loop for that long.

        Args:
            lock: The `filelock.FileLock` serializing refreshes for this server
                (backed by the sidecar `.lock` file, not the token file).

        Returns:
            `True` when the lock was acquired; `False` when the wait timed out
            or the lock could not be created, signalling the caller to avoid
            using the possibly in-flight refresh token after reloading.
        """
        try:
            await asyncio.to_thread(
                lock.acquire,
                timeout=_REFRESH_LOCK_TIMEOUT_SECONDS,
            )
        except Timeout:
            # A timeout means a peer may still be mid-refresh with this same
            # token. Do not refresh unlocked: rotating-token servers can treat
            # the second grant as reuse and revoke the whole token family.
            logger.warning(
                "Timed out after %.0fs waiting for the MCP token refresh lock "
                "for %s; skipping refresh to avoid refresh-token reuse.",
                _REFRESH_LOCK_TIMEOUT_SECONDS,
                self.context.server_url,
            )
            return False
        except OSError as exc:
            # Creating/locking the sidecar can fail (read-only or missing
            # tokens dir, permission denial on a hardened host). Avoid an
            # unlocked refresh so we do not replay a rotating refresh token if a
            # peer did manage to take the lock.
            logger.warning(
                "Could not acquire the MCP token refresh lock for %s (%s); "
                "skipping refresh to avoid refresh-token reuse.",
                self.context.server_url,
                type(exc).__name__,
            )
            return False
        return True

    @contextlib.asynccontextmanager
    async def _refresh_lock_guard(self, lock_path: Path) -> AsyncIterator[bool]:
        """Hold the cross-process refresh lock across the serialized refresh.

        Acquires the lock (waiting up to `_REFRESH_LOCK_TIMEOUT_SECONDS`; on
        timeout it yields `False` so the caller can avoid the refresh grant) and
        always releases it on exit. Release is gated on `lock.is_locked` rather
        than the acquire result, so a cancellation that lands *after* the worker
        thread took the lock still frees it instead of orphaning it until GC.

        Args:
            lock_path: Sibling `.lock` path from `FileTokenStorage`.

        Yields:
            Whether the refresh lock was acquired.
        """
        # `thread_local=False` because acquire and release run in different
        # `asyncio.to_thread` worker threads; the default would refuse the
        # cross-thread release and leak the OS lock until process exit.
        lock = FileLock(str(lock_path), thread_local=False)
        try:
            yield await self._acquire_refresh_lock(lock)
        finally:
            if lock.is_locked:
                await asyncio.to_thread(lock.release)

    async def _persist_oauth_metadata(self) -> None:
        """Persist discovered public OAuth metadata when storage supports it."""
        if self.context.oauth_metadata is None:
            return
        set_oauth_metadata = getattr(self.context.storage, "set_oauth_metadata", None)
        if set_oauth_metadata is not None:
            await set_oauth_metadata(self.context.oauth_metadata)

    async def _handle_token_response(self, response: httpx.Response) -> None:
        """Persist tokens and any metadata discovered during full OAuth login."""
        await super()._handle_token_response(response)
        await self._persist_oauth_metadata()

    async def _handle_locked_refresh_response(self, response: httpx.Response) -> bool:
        """Handle a serialized refresh without bypassing SDK re-auth fallback.

        Args:
            response: Refresh endpoint response returned through the auth generator.

        Returns:
            `True` when refresh succeeded, otherwise `False` so the caller can
            continue into the delegated SDK flow.
        """
        try:
            return bool(await self._handle_refresh_response(response))
        except Exception:
            if response.status_code not in _EXPECTED_REAUTH_REFRESH_STATUS_CODES:
                raise
            logger.debug(
                "Locked MCP token refresh for %s failed with %s; "
                "deferring to the SDK re-auth flow.",
                self.context.server_url,
                response.status_code,
            )
            self.context.clear_tokens()
            self._initialized = False
            return False

    async def async_auth_flow(
        self,
        request: httpx.Request,
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        """Discover and cache OAuth metadata before the SDK refresh branch.

        Yields:
            HTTP requests for OAuth metadata discovery and the delegated SDK auth flow.
        """
        async with self.context.lock:
            if not self._initialized:
                await self._initialize()
            self.context.protocol_version = request.headers.get(MCP_PROTOCOL_VERSION)
            if (
                not self.context.is_token_valid()
                and self.context.can_refresh_token()
                and self.context.oauth_metadata is None
            ):
                # Pre-empt the SDK's 401-path discovery so its refresh branch
                # finds populated `oauth_metadata` and uses the advertised token
                # endpoint instead of guessing `/token`. The resource-metadata
                # URL is `None`: no 401 yet, so no `WWW-Authenticate` to read.
                try:
                    prm_urls = build_protected_resource_metadata_discovery_urls(
                        None,
                        self.context.server_url,
                    )
                    for url in prm_urls:
                        # ASYNC119: yielding the request to receive its response is
                        # this auth generator's handshake protocol, not a value
                        # escaping a context manager.
                        response = yield create_oauth_metadata_request(url)  # noqa: ASYNC119
                        prm = await handle_protected_resource_response(response)
                        if prm is None:
                            logger.debug(
                                "Protected resource metadata discovery failed: %s",
                                url,
                            )
                            continue
                        self.context.protected_resource_metadata = prm
                        self.context.auth_server_url = str(prm.authorization_servers[0])
                        break

                    asm_urls = build_oauth_authorization_server_metadata_discovery_urls(
                        self.context.auth_server_url,
                        self.context.server_url,
                    )
                    for url in asm_urls:
                        # ASYNC119: yielding the request to receive its response is
                        # this auth generator's handshake protocol, not a value
                        # escaping a context manager.
                        response = yield create_oauth_metadata_request(url)  # noqa: ASYNC119
                        ok, metadata = await handle_auth_metadata_response(response)
                        if not ok:
                            break
                        if metadata is None:
                            logger.debug("OAuth metadata discovery failed: %s", url)
                            continue
                        self.context.oauth_metadata = metadata
                        await self._persist_oauth_metadata()
                        break
                except httpx.HTTPError as exc:
                    # Log only the exception type, never its payload — discovery
                    # responses travel the same channel as bearer tokens.
                    logger.debug(
                        "Pre-emptive OAuth metadata discovery for %s raised %s; "
                        "deferring to the SDK auth flow.",
                        self.context.server_url,
                        type(exc).__name__,
                    )

            if (
                not self.context.is_token_valid()
                and self.context.can_refresh_token()
                and isinstance(self.context.storage, FileTokenStorage)
            ):
                # Serialize the refresh across processes and provider instances.
                # Without this, two holders of the same token file can both
                # replay the same refresh token; the LangSmith OAuth server
                # rotates refresh tokens and revokes the entire token family on
                # reuse, which surfaces as requests hanging until a full
                # re-auth. `self.context.lock` only guards this one provider,
                # so a file lock is required for the cross-process case.
                async with self._refresh_lock_guard(
                    self.context.storage.refresh_lock_path
                ) as refresh_lock_acquired:
                    # A peer may have rotated the token while we waited for the
                    # lock; reload so a now-valid token skips the refresh.
                    await self._reload_tokens_from_storage()
                    if (
                        not self.context.is_token_valid()
                        and self.context.can_refresh_token()
                    ):
                        if refresh_lock_acquired:
                            # ASYNC119: the refresh lock must stay held across this
                            # yield — the request/response round-trip is the
                            # critical section being serialized. Release is safe
                            # because httpx deterministically drives and
                            # `aclose()`s this generator (see the delegation note
                            # below), so the guard's `finally` runs rather than
                            # deferring cleanup to GC.
                            refresh_response = yield await self._refresh_token()  # noqa: ASYNC119
                            await self._handle_locked_refresh_response(refresh_response)
                        else:
                            # The delegated SDK flow has its own refresh branch;
                            # clear only in-memory tokens so this request falls
                            # through to re-auth instead of replaying the refresh
                            # token while another process may still be using it.
                            self.context.clear_tokens()

        # Delegate to the SDK flow by manually pumping the inner generator so
        # the HTTP responses httpx feeds back via `auth_flow.asend(response)`
        # are forwarded into it. A plain `async for` would advance the inner
        # generator with `__anext__()` (i.e. `asend(None)`), discarding every
        # response — the SDK's `response = yield request` and refresh-path
        # `yield refresh_request` would then see `None` and raise
        # `AttributeError: 'NoneType' object has no attribute 'status_code'`,
        # surfacing as the `ExceptionGroup` users hit on MCP OAuth login.
        # httpx primes the flow with `__anext__()`, then drives it with
        # `asend`/`aclose` (never `athrow`), so forwarding sent values and
        # closing the inner generator on `GeneratorExit` is sufficient — no
        # `athrow` forwarding needed.
        token: contextvars.Token[bool] | None = None
        if self._suppress_expected_reauth_logs:
            token = _SUPPRESS_EXPECTED_REAUTH_LOGS.set(True)
        inner = super().async_auth_flow(request)
        try:
            # Prime with `anext()` (no response to send yet); thereafter every
            # resume carries httpx's response back in via `asend`.
            flow_request = await anext(inner)
            while True:
                response = yield flow_request
                flow_request = await inner.asend(response)
        except StopAsyncIteration:
            return
        finally:
            await inner.aclose()
            if token is not None:
                _SUPPRESS_EXPECTED_REAUTH_LOGS.reset(token)


def build_oauth_provider(
    *,
    server_name: str,
    server_url: str,
    storage: TokenStorage,
    extra_auth_params: dict[str, str] | None = None,
    oauth_config: OAuthClientConfig | None = None,
    interactive: bool = True,
    ui: OAuthInteraction | None = None,
) -> OAuthClientProvider:
    """Construct an `OAuthClientProvider` for an MCP server.

    Args:
        server_name: MCP server name used in re-auth messages.
        server_url: Remote MCP server URL.
        storage: Token storage implementation for this server.
        extra_auth_params: Optional query params for the interactive auth URL.
        oauth_config: Optional validated static OAuth client configuration.
        interactive: Whether the provider may prompt on stdin.
        ui: Interaction surface used for URL display and paste-back
            input in interactive mode.

    Returns:
        A configured `OAuthClientProvider`.

    Raises:
        ValueError: If configured callback validation fails.
    """
    from deepagents_code.mcp_providers import resolve_provider

    policy = resolve_provider(server_url)
    configured_client_info = (
        resolve_oauth_client_config(oauth_config, server_name=server_name)
        if oauth_config is not None
        else None
    )
    redirect_uri: str | None = None

    if interactive:
        if configured_client_info is not None:
            configured_redirect_uris = configured_client_info.redirect_uris
            if not configured_redirect_uris:  # Constructed above with one URI.
                msg = "Configured OAuth client has no redirect URI"
                raise ValueError(msg)
            configured_redirect_uri = str(configured_redirect_uris[0])
            configured_port = urlparse(configured_redirect_uri).port
            if configured_port is None:  # Validated config makes this unreachable.
                msg = "Configured OAuth redirect URI has no port"
                raise ValueError(msg)
            callback_server = _LoopbackOAuthCallbackServer(port=configured_port)
            redirect_uri = configured_redirect_uri
            redirect, callback = _make_loopback_handlers(
                callback_server=callback_server,
                extra_auth_params=extra_auth_params,
                ui=ui,
            )
        elif policy.supports_loopback_callback():
            fixed = policy.loopback_port()
            if fixed is not None:
                port = fixed
            else:
                # Reuse the port from a prior DCR registration when available,
                # so the authorize request's redirect_uri matches what was
                # registered against the persisted client_id. A fresh random
                # port on every launch would otherwise invalidate the URI on
                # the second run and force the server to reject the request.
                stored = (
                    storage.stored_loopback_port()
                    if isinstance(storage, FileTokenStorage)
                    else None
                )
                # No reusable port means any persisted registration can't be
                # paired with the random loopback port we're about to bind. Drop
                # a stale registration so the handshake re-runs DCR with a
                # matching redirect URI instead of failing with "invalid or
                # missing redirect_uri".
                if (
                    stored is None
                    and isinstance(storage, FileTokenStorage)
                    and storage.discard_client_info_if_loopback_unusable()
                ):
                    logger.info(
                        "Discarded a stale MCP client registration for %s "
                        "whose redirect URI can't serve loopback login; the "
                        "handshake will register a fresh client.",
                        server_name,
                    )
                port = stored if stored is not None else _choose_loopback_port()
            callback_server = _LoopbackOAuthCallbackServer(port=port)
            redirect_uri = callback_server.redirect_uri
            redirect, callback = _make_loopback_handlers(
                callback_server=callback_server,
                extra_auth_params=extra_auth_params,
                ui=ui,
            )
        else:
            redirect, callback = _make_paste_back_handlers(
                extra_auth_params=extra_auth_params,
                ui=ui,
            )
    else:
        redirect, callback = _make_reauth_required_handlers(server_name=server_name)

    if configured_client_info is not None:
        metadata = OAuthClientMetadata(
            client_name="deepagents-code",
            redirect_uris=configured_client_info.redirect_uris,
            grant_types=configured_client_info.grant_types,
            response_types=configured_client_info.response_types,
            token_endpoint_auth_method="client_secret_post",  # noqa: S106  # OAuth method, not a secret
        )
    else:
        metadata = (
            policy.client_metadata(redirect_uri=redirect_uri)
            if redirect_uri is not None
            else policy.client_metadata()
        )

    return _ExpiryAwareOAuthClientProvider(
        server_url=server_url,
        client_metadata=metadata,
        storage=storage,
        redirect_handler=redirect,
        callback_handler=callback,
        configured_client_info=configured_client_info,
        suppress_expected_reauth_logs=not interactive,
    )


async def _run_device_flow(
    *,
    device_code_url: str,
    token_url: str,
    client_id: str,
    scope: str | None = None,
    ui: OAuthInteraction | None = None,
) -> OAuthToken:
    """Run OAuth 2.0 Device Authorization Grant and return the token.

    Args:
        device_code_url: Provider endpoint that issues a device + user code.
        token_url: Provider endpoint to poll for the access token.
        client_id: Registered OAuth client ID.
        scope: Optional space-delimited scope string.
        ui: Interaction surface used to display the device code.

    Returns:
        The issued OAuth access token payload.

    Raises:
        RuntimeError: If the device flow fails, times out, or the provider
            returns an unexpected HTTP status on the device-code request.
    """
    import asyncio

    import httpx

    interaction = ui if ui is not None else _default_ui()

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

        await interaction.show_device_code(
            verification_uri=device.verification_uri,
            user_code=device.user_code,
            expires_in=device.expires_in,
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
            except ValueError as exc:
                # Malformed JSON would otherwise cascade into a confusing
                # OAuthToken.model_validate({}) error below; log the cause
                # explicitly so debugging is possible.
                logger.warning(
                    "Token endpoint %s returned non-JSON body: %s",
                    token_url,
                    exc,
                )
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

    msg = "Device flow timed out. Try logging in again."
    raise RuntimeError(msg)


def format_login_failure(exc: BaseException) -> str:
    """Return a token-safe single-line summary of an OAuth-login exception.

    OAuth handshakes commonly surface as `ExceptionGroup` (anyio task
    groups) or as MCP-SDK errors whose `args`/`repr` may include an
    `OAuthToken`. Never call `str()`/`repr()` on the raw exception for
    display or logging — instead, prefer a known-safe nested
    `MCPReauthRequiredError` message, fall back to the messages of our
    own loopback-related exception types, and degrade to a class-name
    chain for anything else.

    Args:
        exc: Root exception caught from the login worker.

    Returns:
        A user-displayable string that is safe to log and to render.
    """
    reauth = find_reauth_required(exc)
    if reauth is not None:
        return str(reauth)

    safe_types = (
        _LoopbackCallbackTimeoutError,
        _LoopbackCallbackUnavailableError,
    )
    if isinstance(exc, safe_types):
        return f"{type(exc).__name__}: {exc}"

    parts: list[str] = []
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        parts.append(type(current).__name__)
        if isinstance(current, BaseExceptionGroup):
            parts.append(
                "[" + ", ".join(type(e).__name__ for e in current.exceptions[:5]) + "]"
            )
            break
        current = current.__cause__ or current.__context__
    return " -> ".join(parts) if parts else type(exc).__name__


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
        if isinstance(current, BaseExceptionGroup):
            stack.extend(current.exceptions)
        cause = current.__cause__ or current.__context__
        if cause is not None:
            stack.append(cause)
    return None


_BEARER_SCHEME_RE = re.compile(r"(?:^|,)\s*bearer\b", re.IGNORECASE)
"""Match a `Bearer` auth scheme at the start of a challenge or after a comma.

A `WWW-Authenticate` line may list several schemes (RFC 7235); anchoring to
the start or a preceding comma finds `Bearer` even when it isn't listed first.
"""

_RESOURCE_METADATA_RE = re.compile(
    r'(?:^|[\s,])resource_metadata\s*=\s*"?([^",\s]+)',
    re.IGNORECASE,
)
"""Capture the RFC 9728 `resource_metadata` URL from a Bearer challenge."""


def _oauth_resource_challenge(headers: httpx.Headers) -> str | None:
    """Return the RFC 9728 `resource_metadata` URL from a Bearer challenge.

    A single `WWW-Authenticate` header line may carry several comma-separated
    challenges (RFC 7235), and a response may repeat the header. Scan every
    value for a `Bearer` scheme — anywhere in the line, not only first — that
    advertises a `resource_metadata` parameter.

    Args:
        headers: Response headers to inspect.

    Returns:
        The `resource_metadata` URL when a Bearer challenge carries one,
            else `None`.
    """
    for value in headers.get_list("www-authenticate"):
        if _BEARER_SCHEME_RE.search(value) is None:
            continue
        match = _RESOURCE_METADATA_RE.search(value)
        if match is not None:
            return match.group(1)
    return None


def find_oauth_challenge(exc: BaseException) -> str | None:
    """Return the `resource_metadata` URL of a 401 OAuth challenge in `exc`.

    Per the MCP authorization spec (RFC 9728), a server requiring OAuth
    answers an unauthenticated request with HTTP 401 plus a Bearer
    `WWW-Authenticate` challenge pointing at its protected-resource metadata.
    The MCP client surfaces that as an `httpx.HTTPStatusError`. Walks
    `exceptions` (for `ExceptionGroup`), then `__cause__`/`__context__`,
    tracking visited nodes to terminate on cyclic chains.

    Args:
        exc: Root exception to inspect.

    Returns:
        The `resource_metadata` URL when a 401 response carrying a Bearer
            challenge is found, else `None`.
    """
    visited: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, httpx.HTTPStatusError):
            response = current.response
            if (
                response is not None and response.status_code == 401  # noqa: PLR2004  # HTTP Unauthorized
            ):
                challenge = _oauth_resource_challenge(response.headers)
                if challenge is not None:
                    return challenge
        if isinstance(current, BaseExceptionGroup):
            stack.extend(current.exceptions)
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
    ui: OAuthInteraction,
) -> None:
    """Drive OAuth login for `server_name`, persisting tokens on success.

    Args:
        server_name: Name of the configured MCP server.
        server_config: Parsed server config for that entry.
        ui: Interaction surface for all user prompts and progress messages
            during the flow.

    Raises:
        ValueError: If `server_config` isn't an http/sse server.
        RuntimeError: If header env-var interpolation fails, the device
            flow fails or times out, or the OAuth handshake aborts.
    """  # noqa: DOC502 - `RuntimeError` surfaces via `resolve_headers` / `_run_device_flow`
    from langchain_mcp_adapters.sessions import (
        SSEConnection,
        StreamableHttpConnection,
    )

    from deepagents_code.mcp_tools import _resolve_server_type

    # OAuth login is discovery-based (RFC 9728), so it works for any remote
    # http/sse server — whether the config opted in with `auth: oauth` or the
    # server was auto-detected as needing auth via a 401 challenge. Only the
    # transport needs gating; stdio servers can't speak OAuth.
    transport = _resolve_server_type(server_config)
    if transport not in {"http", "sse"}:
        msg = (
            f"Server '{server_name}' uses {transport!r} transport; "
            "OAuth login is only valid for http/sse."
        )
        raise ValueError(msg)

    from deepagents_code.mcp_providers import resolve_provider

    storage = FileTokenStorage(server_name, server_url=server_config["url"])
    policy = resolve_provider(server_config["url"])
    result = await policy.run_login(
        server_name=server_name,
        server_url=server_config["url"],
        storage=storage,
        ui=ui,
    )

    success_message = (
        f"Logged in to MCP server '{server_name}'. Tokens saved to {storage.path}."
    )

    if result.completed:
        await ui.show_success(success_message)
        return

    provider = build_oauth_provider(
        server_name=server_name,
        server_url=server_config["url"],
        storage=storage,
        extra_auth_params=result.extra_auth_params or None,
        oauth_config=server_config.get("oauth"),
        ui=ui,
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
    await ui.show_success(success_message)
