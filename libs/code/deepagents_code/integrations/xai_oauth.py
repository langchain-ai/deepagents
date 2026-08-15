"""xAI (Grok) OAuth integration for the `xai_oauth` model provider.

xAI does not publish a `langchain_xai` OAuth module the way `langchain_openai`
ships `chatgpt_oauth`, so unlike `integrations.openai_codex` this module
cannot be a thin wrapper over upstream primitives — it implements the RFC
8628 Device Authorization Grant flow, token storage, and refresh itself,
reusing the pieces of `deepagents_code.mcp_auth` that are generic enough to
share:

- `mcp_auth._run_device_flow` — the actual RFC 8628 client (device-code
    request, poll loop, `authorization_pending` / `slow_down` handling). Not
    MCP-specific; reused verbatim.
- `mcp_auth.FileTokenStorage` — atomic, `0600`-permissioned on-disk token
    persistence at `~/.deepagents/.state/mcp-tokens/<server_name>.json`.
    Instantiated here with `server_name="xai"`; nothing about it is
    MCP-session-specific.

What is *not* reused is `mcp_auth._ExpiryAwareOAuthClientProvider`, which
wraps an active MCP/httpx transport for automatic refresh — there is no bare
"give me a valid bearer token, refreshing if needed" function to call
standalone. `_ensure_valid_access_token` below is that missing piece,
written fresh for this module.

!!! warning "Unofficial, undocumented-by-xAI OAuth surface"

    This integration talks to `auth.x.ai` using a public `client_id` that is
    shared with other independent, third-party CLI tools (not something xAI
    has published as a supported integration surface for third parties). It
    was reverse-engineered by observing those tools' network traffic, not
    from any xAI-published API reference. xAI could change the device-code
    or token endpoint response shape, rotate or revoke the client, or gate
    this surface entirely, at any time and without notice — this module is
    written to fail with a clear, specific error in that case rather than a
    raw traceback, but it cannot detect a break before it happens.

    Signing in this way also requires a SuperGrok / X Premium+ entitlement
    on the underlying xAI account; accounts without one will see requests
    rejected with `HTTP 403` (`XaiOAuthTierDeniedError`) even after a
    successful device-code login. Users without that entitlement should use
    the existing API-key path (`XAI_API_KEY`, provider `"xai"`) instead.

Token lifetime and rotation notes (load-bearing for the refresh logic below):

- Access tokens are JWTs with a real `exp` claim. This module decodes it
    (unverified — no signature check is needed client-side; the token is
    only ever used as a bearer value forwarded to xAI itself) to compute a
    remaining lifetime for proactive-refresh decisions, rather than trusting
    a separately-tracked `expires_in` alone, since observed token lifetimes
    vary a lot (device-code logins can be as short-lived as ~15 minutes).
- xAI rotates the refresh token on every use (single-use). The new
    `refresh_token` from every refresh response is persisted; if a response
    omits one, the prior refresh token is kept so the next refresh attempt
    still has something to send.
- Concurrent refreshes are serialized with a cross-process file lock
    (`FileTokenStorage.refresh_lock_path`) so two processes racing to use
    the same single-use refresh token cannot silently invalidate each
    other's session.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from filelock import FileLock, Timeout
from mcp.shared.auth import OAuthToken
from pydantic import ValidationError

if TYPE_CHECKING:
    from datetime import datetime

    from langchain_core.language_models import BaseChatModel
    from langchain_xai import ChatXAI  # ty: ignore[unresolved-import]

    from deepagents_code.mcp_auth import FileTokenStorage

logger = logging.getLogger(__name__)


XAI_OAUTH_SERVER_NAME = "xai"
"""`FileTokenStorage` identity for the xAI OAuth token file.

Yields `~/.deepagents/.state/mcp-tokens/xai.json` — a plain server name (no
URL) so the token file stem is just `xai`, matching the `mcp_auth._SAFE_SERVER_NAME_RE`
constraint.
"""

_XAI_DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
"""xAI's RFC 8628 device-authorization endpoint that issues the user/device code."""

_XAI_TOKEN_URL = "https://auth.x.ai/oauth2/token"  # noqa: S105
"""xAI OAuth token endpoint, polled during device-code login and used for refresh.

Hardcoded directly rather than resolved via OIDC discovery
(`https://auth.x.ai/.well-known/openid-configuration`) — both approaches were
verified to currently resolve to this same URL, and hardcoding is the
simpler, sufficient choice for a first implementation. A future change could
add discovery (with host-pinning against cache-poisoning) if xAI starts
rotating this endpoint.
"""

_XAI_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
"""Public OAuth client ID shared by independent third-party xAI CLI tools.

Not published by xAI as a supported third-party integration surface — see
the module docstring's warning.
"""

_XAI_SCOPE = "openid profile email offline_access grok-cli:access api:access"
"""OAuth scope requested during the device-code login."""

_DEFAULT_REFRESH_SKEW_SECONDS = 300.0
"""How far ahead of expiry to proactively refresh, for an ordinary-lifetime token."""

_SHORT_LIFETIME_THRESHOLD_SECONDS = 45 * 60
"""Token lifetimes at or below this are treated as "short-lived" (e.g. ~15min JWTs)."""

_SHORT_LIFETIME_REFRESH_SKEW_SECONDS = 120.0
"""Refresh skew used for short-lived tokens.

Device-code logins are sometimes issued short-lived (~15 minute) JWTs. Using
the ordinary 300s skew against a token that short would trigger a proactive
refresh almost immediately after login, and every additional caller racing
to be "first" to refresh increases the odds of one process's request
invalidating another's freshly-rotated (single-use) refresh token. Shrinking
the skew for short-lived tokens narrows that window.
"""

_REFRESH_LOCK_TIMEOUT_SECONDS = 30.0
"""Max time to wait for the cross-process refresh lock before giving up."""


class XaiOAuthError(RuntimeError):
    """Base class for xAI OAuth failures raised by this module."""


class XaiOAuthProtocolError(XaiOAuthError):
    """The device-code or token endpoint returned something unexpected.

    Covers network failures, non-JSON bodies, and response payloads that
    don't validate as an OAuth token — i.e. exactly the failure mode the
    module docstring's warning describes: xAI changing this undocumented
    surface without notice. Always carries a specific, user-safe message
    (no raw tracebacks) so a caller can render it directly.
    """


class XaiOAuthReauthRequiredError(XaiOAuthError):
    """A stored token cannot be refreshed and a fresh sign-in is required.

    Raised for `HTTP 400`/`401` on the refresh grant (dead/invalid refresh
    token), or when no refresh token is available at all. Distinct from a
    missing token (`FileNotFoundError`): a token file exists, but the
    session cannot be resumed automatically.
    """


class XaiOAuthTierDeniedError(XaiOAuthError):
    """The refresh/auth request was rejected with `HTTP 403`.

    xAI's OAuth surface gates on a SuperGrok / X Premium+ entitlement; a 403
    here means the account is signed in but not entitled, not that the
    session is dead. Callers should point the user at `XAI_API_KEY` as a
    fallback rather than prompting another sign-in.
    """


class XaiLoginCancelledError(RuntimeError):
    """Raised when the user cancels a device-code sign-in mid-flow."""


@dataclass(frozen=True, slots=True)
class XaiOAuthStatus:
    """Snapshot of the xAI OAuth login state.

    Attributes:
        logged_in: Whether a usable token bundle exists on disk.
        store_path: Path to the token store (present whether or not the
            token exists, so the UI can show "no token yet at <path>").
        expires_at: Access token expiry (UTC), decoded from the JWT `exp`
            claim at snapshot time. `None` if no token is stored or the
            token has no decodable expiry.
        is_expired: Whether the stored access token is past `expires_at`.
            Computed once at snapshot time; a live check would decode again.
            The stored refresh token (if any) may still be usable even when
            this is `True` — `build_chat_model` refreshes transparently.
        unreadable_reason: Set when the token file exists but cannot be
            parsed. Surfaces corruption to the UI without crashing
            credential listing.
    """

    logged_in: bool
    store_path: Any
    expires_at: datetime | None = None
    is_expired: bool = False
    unreadable_reason: str | None = None

    def __post_init__(self) -> None:
        """Reject incoherent snapshots the factory should never build.

        Raises:
            ValueError: An unreadable token is also marked `logged_in`, or a
                logged-out snapshot carries `expires_at`/`is_expired`.
        """
        if self.unreadable_reason is not None and self.logged_in:
            msg = (
                "`unreadable_reason` implies the token is not usable; "
                "`logged_in` must be False."
            )
            raise ValueError(msg)
        if not self.logged_in and self.expires_at is not None:
            msg = "`expires_at` is only meaningful when `logged_in` is True."
            raise ValueError(msg)
        if not self.logged_in and self.is_expired:
            msg = "`is_expired` is only meaningful when `logged_in` is True."
            raise ValueError(msg)


def default_storage() -> FileTokenStorage:
    """Return the `FileTokenStorage` bound to the xAI OAuth token file.

    Reused from `mcp_auth` even though this isn't an MCP session — see the
    module docstring. Resolves to
    `~/.deepagents/.state/mcp-tokens/xai.json`.
    """
    from deepagents_code.mcp_auth import FileTokenStorage

    return FileTokenStorage(XAI_OAUTH_SERVER_NAME)


def _decode_jwt_exp(access_token: str) -> float | None:
    """Return the unverified `exp` claim (Unix epoch seconds) from a JWT.

    No signature check is performed — the token is only ever forwarded to
    xAI itself as a bearer credential, so verifying it client-side buys
    nothing. This is purely to drive proactive-refresh timing.

    Args:
        access_token: The raw JWT access token string.

    Returns:
        The `exp` claim as a float, or `None` if `access_token` is not a
            well-formed JWT or carries no numeric `exp`.
    """
    parts = access_token.split(".")
    if len(parts) != 3:  # noqa: PLR2004  # header.payload.signature
        return None
    payload_segment = parts[1]
    padding = "=" * (-len(payload_segment) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload_segment + padding)
        payload = json.loads(raw)
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    exp = payload.get("exp") if isinstance(payload, dict) else None
    if isinstance(exp, (int, float)):
        return float(exp)
    return None


def get_status(*, storage: FileTokenStorage | None = None) -> XaiOAuthStatus:
    """Return the current xAI OAuth sign-in state.

    Reads the on-disk token *without* triggering a refresh (a passive
    inspect, suitable for switcher labels and the `/auth` manager). Uses
    `FileTokenStorage`'s sync, private read primitives directly (`_read`,
    `_tokens_from_data`) since this module needs a synchronous status check
    and those primitives are already the sync core the async public API
    wraps with `asyncio.to_thread`.

    Args:
        storage: Override the token storage. Defaults to `default_storage()`.

    Returns:
        A `XaiOAuthStatus` populated from the on-disk token, or one with
            `logged_in=False` when no token exists or the file is unreadable.
    """
    from datetime import UTC, datetime

    store = storage or default_storage()
    try:
        data = store._read()
    except RuntimeError as exc:
        return XaiOAuthStatus(
            logged_in=False,
            store_path=store.path,
            unreadable_reason=str(exc),
        )
    if data is None:
        return XaiOAuthStatus(logged_in=False, store_path=store.path)
    token = store._tokens_from_data(data)
    if token is None:
        return XaiOAuthStatus(logged_in=False, store_path=store.path)
    exp = _decode_jwt_exp(token.access_token)
    expires_at = datetime.fromtimestamp(exp, tz=UTC) if exp is not None else None
    is_expired = expires_at is not None and expires_at <= datetime.now(UTC)
    return XaiOAuthStatus(
        logged_in=True,
        store_path=store.path,
        expires_at=expires_at,
        is_expired=is_expired,
    )


def is_logged_in(*, storage: FileTokenStorage | None = None) -> bool:
    """Return whether an xAI OAuth token is stored on disk."""
    return get_status(storage=storage).logged_in


def logout(*, storage: FileTokenStorage | None = None) -> bool:
    """Delete the stored xAI OAuth token.

    Args:
        storage: Override the token storage. Defaults to `default_storage()`.

    Returns:
        `True` if a token file was removed, `False` if no file existed.
    """
    store = storage or default_storage()
    path = store.path
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


class XaiLoginInteraction:
    """UI hooks for the device-code sign-in flow.

    Implements `mcp_oauth_ui.OAuthInteraction`'s full Protocol (structurally
    — this class does not import or subclass it) so `mcp_auth._run_device_flow`,
    whose `ui` parameter is typed against that Protocol, accepts an instance
    directly. The device flow itself only ever calls `show_device_code`,
    `show_success`, and `show_error`; the remaining methods
    (`show_authorize_url`, `request_callback_url`, `show_notice`) exist only
    to satisfy the Protocol shape and are never invoked by a device-code
    flow. The default base class prints to stdout for headless / CLI
    callers; UI callers (e.g. the Textual sign-in modal) override these.
    """

    async def show_authorize_url(  # Protocol shape; unused by device flow
        self, url: str, *, opened_in_browser: bool
    ) -> None:
        """Unused by the device-code flow; present only to satisfy the Protocol."""

    async def request_callback_url(self) -> str:  # noqa: PLR6301
        """Unused by the device-code flow; present only to satisfy the Protocol.

        Raises:
            RuntimeError: Always — the device-code flow never calls this.
        """
        msg = "XaiLoginInteraction does not support a callback-URL paste-back."
        raise RuntimeError(msg)

    async def show_device_code(  # noqa: PLR6301  # override hook
        self,
        *,
        verification_uri: str,
        user_code: str,
        expires_in: int,
    ) -> None:
        """Show RFC 8628 device-code instructions to the user."""
        print(  # noqa: T201
            f"\nVisit {verification_uri} and enter code: "
            f"{user_code}\n(code expires in {expires_in}s)\n",
        )

    async def show_success(self, message: str) -> None:  # noqa: PLR6301
        """Report a successful login step. Never pass token material here."""
        print(message)  # noqa: T201

    async def show_notice(self, message: str) -> None:  # noqa: PLR6301
        """Unused by the device-code flow; present only to satisfy the Protocol."""
        print(message)  # noqa: T201

    async def show_error(self, message: str) -> None:  # noqa: PLR6301
        """Report a fatal (flow-ending) error."""
        print(message)  # noqa: T201


async def run_device_login(
    interaction: XaiLoginInteraction | None = None,
    *,
    storage: FileTokenStorage | None = None,
) -> XaiOAuthStatus:
    """Run the xAI OAuth Device Authorization Grant and persist the token.

    Delegates the actual RFC 8628 handshake to `mcp_auth._run_device_flow`
    (see the module docstring) and persists the result with this module's
    own token/expiry bookkeeping (`_persist_token`) so the refresh routine's
    lifetime-aware skew has the data it needs.

    Args:
        interaction: UI hooks for the device-code prompt. A default
            stdout-based implementation is used when `None`.
        storage: Override the token storage. Defaults to `default_storage()`.

    Returns:
        A fresh `XaiOAuthStatus` reflecting the just-saved token.

    Raises:
        XaiOAuthProtocolError: The device-code or token endpoint returned an
            unexpected response. Also raised (wrapping `filelock.Timeout`)
            if the cross-process refresh lock could not be acquired while
            persisting the new token via `_persist_token`.
    """
    from deepagents_code.mcp_auth import _run_device_flow

    ui = interaction if interaction is not None else XaiLoginInteraction()
    store = storage or default_storage()
    try:
        token = await _run_device_flow(
            device_code_url=_XAI_DEVICE_CODE_URL,
            token_url=_XAI_TOKEN_URL,
            client_id=_XAI_CLIENT_ID,
            scope=_XAI_SCOPE,
            ui=ui,
        )
    except RuntimeError as exc:
        # `_run_device_flow` raises plain `RuntimeError` for every failure
        # mode (HTTP errors, malformed responses, timeout, explicit
        # `error=` from the provider). Normalize to our own exception type
        # so callers can `except XaiOAuthProtocolError` without coupling to
        # `mcp_auth`'s internals.
        msg = str(exc)
        raise XaiOAuthProtocolError(msg) from exc
    _persist_token(store, token)
    await ui.show_success("Signed in to xAI.")
    return get_status(storage=store)


def _write_token_data(storage: FileTokenStorage, token: OAuthToken) -> None:
    """Write `token` to disk alongside its absolute expiry and lifetime.

    Does *no* locking itself — callers that already hold
    `storage.refresh_lock_path` (e.g. `_ensure_valid_access_token`'s refresh
    branch) must call this directly rather than `_persist_token`, or they
    would try to re-acquire the same lock file from a second `FileLock`
    instance on the same thread, which `filelock` treats as a deadlock and
    raises `RuntimeError` for rather than silently reentering.

    Uses `FileTokenStorage`'s sync private primitives directly (`_read`,
    `_write`) rather than its async public API — `build_chat_model` and the
    refresh routine both run synchronously (mirroring `openai_codex`, whose
    `build_chat_model` calls its token provider synchronously too), so a
    sync write path avoids needing an event loop here.

    Stores an extra `xai_lifetime_seconds` field beyond what
    `FileTokenStorage`'s own schema defines — this module owns that field
    and is the only reader of it (via `_needs_refresh`), so it can't
    conflict with the MCP token-refresh code path that also uses this class.

    Args:
        storage: The token storage to write to.
        token: The token to persist (rotates the refresh token per xAI's
            single-use policy — callers must pass the token with the
            correct refresh_token already resolved, see `_refresh_access_token`).
    """
    from deepagents_code.mcp_auth import _STORAGE_VERSION

    data = storage._read() or {}
    data["version"] = _STORAGE_VERSION
    data["tokens"] = json.loads(token.model_dump_json(exclude_none=True))
    exp = _decode_jwt_exp(token.access_token)
    now = time.time()
    if exp is not None:
        data["expires_at"] = exp
        data["xai_lifetime_seconds"] = max(exp - now, 0.0)
    elif token.expires_in is not None:
        data["expires_at"] = now + token.expires_in
        data["xai_lifetime_seconds"] = token.expires_in
    else:
        data.pop("expires_at", None)
        data.pop("xai_lifetime_seconds", None)
    storage._write(data)


def _persist_token(storage: FileTokenStorage, token: OAuthToken) -> None:
    """Acquire the cross-process refresh lock, then write `token` to disk.

    Use this from any caller that does *not* already hold
    `storage.refresh_lock_path` (e.g. `run_device_login`, which persists a
    brand-new token outside of the refresh path). Callers that already hold
    the lock (the refresh branch of `_ensure_valid_access_token`) must call
    `_write_token_data` directly instead — see its docstring.

    Args:
        storage: The token storage to write to.
        token: The token to persist.

    Raises:
        XaiOAuthProtocolError: The cross-process refresh lock could not be
            acquired within `_REFRESH_LOCK_TIMEOUT_SECONDS` (wraps
            `filelock.Timeout`).
    """
    lock = FileLock(str(storage.refresh_lock_path), thread_local=True)
    try:
        with lock.acquire(timeout=_REFRESH_LOCK_TIMEOUT_SECONDS):
            _write_token_data(storage, token)
    except Timeout as exc:
        msg = (
            "Could not acquire the xAI OAuth refresh lock in time; another "
            "process may be refreshing the same session. Try again."
        )
        raise XaiOAuthProtocolError(msg) from exc


def _needs_refresh(expires_at: float | None, lifetime_seconds: float | None) -> bool:
    """Return whether the stored access token should be proactively refreshed.

    Args:
        expires_at: Absolute Unix-epoch expiry, or `None` if unknown (in
            which case a refresh is always requested, fail-safe).
        lifetime_seconds: The token's total lifetime at issuance, used to
            pick the refresh skew (see `_SHORT_LIFETIME_REFRESH_SKEW_SECONDS`).
    """
    if expires_at is None:
        return True
    skew = _DEFAULT_REFRESH_SKEW_SECONDS
    if (
        lifetime_seconds is not None
        and lifetime_seconds <= _SHORT_LIFETIME_THRESHOLD_SECONDS
    ):
        skew = _SHORT_LIFETIME_REFRESH_SKEW_SECONDS
    return time.time() >= (expires_at - skew)


def _refresh_access_token(refresh_token: str) -> OAuthToken:
    """POST the `refresh_token` grant to xAI's token endpoint.

    Args:
        refresh_token: The current (single-use) refresh token.

    Returns:
        The new token. If xAI's response omits `refresh_token` (it usually
            won't, since xAI rotates it on every use, but the response
            contract isn't documented), the prior `refresh_token` is carried
            forward so the caller always has something to persist and use
            for the *next* refresh.

    Raises:
        XaiOAuthTierDeniedError: `HTTP 403` — SuperGrok/entitlement gate.
        XaiOAuthReauthRequiredError: `HTTP 400`/`401` — refresh token dead.
        XaiOAuthProtocolError: Any other network failure or unexpected
            response shape.
    """
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                _XAI_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": _XAI_CLIENT_ID,
                    "refresh_token": refresh_token,
                },
                headers={"Accept": "application/json"},
            )
    except httpx.HTTPError as exc:
        msg = f"xAI OAuth token refresh request failed: {type(exc).__name__}."
        raise XaiOAuthProtocolError(msg) from exc

    if response.status_code == httpx.codes.FORBIDDEN:
        msg = (
            "xAI rejected the OAuth refresh with HTTP 403. This usually means "
            "the signed-in account does not have the SuperGrok / X Premium+ "
            "entitlement this OAuth surface requires. Set XAI_API_KEY and use "
            "the standard xai provider instead."
        )
        raise XaiOAuthTierDeniedError(msg)
    if response.status_code in {httpx.codes.BAD_REQUEST, httpx.codes.UNAUTHORIZED}:
        msg = (
            f"xAI OAuth session could not be refreshed (HTTP {response.status_code}). "
            "Sign in again via /auth."
        )
        raise XaiOAuthReauthRequiredError(msg)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        msg = f"xAI OAuth token refresh failed: HTTP {response.status_code}."
        raise XaiOAuthProtocolError(msg) from exc

    try:
        body = response.json()
    except ValueError as exc:
        msg = "xAI OAuth token refresh returned a non-JSON response."
        raise XaiOAuthProtocolError(msg) from exc
    try:
        new_token = OAuthToken.model_validate(body)
    except ValidationError as exc:
        msg = f"xAI OAuth token refresh response is missing required fields: {exc}"
        raise XaiOAuthProtocolError(msg) from exc

    if new_token.refresh_token is None:
        # xAI rotates the refresh token on every use; if a response ever
        # omits one, fall back to the token we just spent so the next
        # refresh attempt still has *something* to send rather than
        # silently losing the session.
        new_token = new_token.model_copy(update={"refresh_token": refresh_token})
    return new_token


def _ensure_valid_access_token(storage: FileTokenStorage) -> str:
    """Return a valid bearer access token, refreshing on disk if needed.

    The one piece of net-new logic this module contains (everything else
    delegates to `mcp_auth`): checks the stored token's expiry, and if it is
    near/past expiry, refreshes it under a cross-process file lock so
    concurrent callers cannot race to spend the same single-use refresh
    token.

    Args:
        storage: The token storage to read from and refresh into.

    Returns:
        A currently-valid access token string.

    Raises:
        FileNotFoundError: No token is stored at all.
        XaiOAuthReauthRequiredError: The stored token has no refresh token
            (or one exists but was rejected as dead) and needs a fresh
            sign-in.
        XaiOAuthTierDeniedError: The refresh was rejected as an
            entitlement-tier gate (`HTTP 403`).
        XaiOAuthProtocolError: The refresh request or response was
            otherwise malformed. Also raised (wrapping `filelock.Timeout`)
            if the cross-process refresh lock could not be acquired.
    """  # noqa: DOC502  # `XaiOAuthTierDeniedError` is raised by `_refresh_access_token`, called below
    data = storage._read()
    token = storage._tokens_from_data(data)
    if token is None:
        msg = "No stored xAI OAuth token."
        raise FileNotFoundError(msg)

    expires_at = storage._expires_at_from_data(data)
    lifetime = (data or {}).get("xai_lifetime_seconds")
    if not _needs_refresh(expires_at, lifetime):
        return token.access_token

    if token.refresh_token is None:
        msg = "xAI OAuth access token has no refresh token; sign in again."
        raise XaiOAuthReauthRequiredError(msg)

    lock = FileLock(str(storage.refresh_lock_path), thread_local=True)
    try:
        with lock.acquire(timeout=_REFRESH_LOCK_TIMEOUT_SECONDS):
            # A peer may have refreshed while we waited for the lock; reload
            # so a now-valid token skips the refresh entirely.
            data = storage._read()
            token = storage._tokens_from_data(data)
            expires_at = storage._expires_at_from_data(data)
            lifetime = (data or {}).get("xai_lifetime_seconds")
            if token is not None and not _needs_refresh(expires_at, lifetime):
                return token.access_token
            if token is None or token.refresh_token is None:
                msg = "xAI OAuth access token has no refresh token; sign in again."
                raise XaiOAuthReauthRequiredError(msg)
            new_token = _refresh_access_token(token.refresh_token)
            # The refresh lock is already held here (we're inside
            # `lock.acquire()` above); write directly rather than going
            # through `_persist_token`, which would try to re-acquire the
            # same lock file and deadlock. See `_write_token_data`'s
            # docstring.
            _write_token_data(storage, new_token)
            return new_token.access_token
    except Timeout as exc:
        msg = (
            "Could not acquire the xAI OAuth refresh lock in time; another "
            "process may be refreshing the same session. Try again."
        )
        raise XaiOAuthProtocolError(msg) from exc


def build_chat_model(
    model_name: str, /, *, storage: FileTokenStorage | None = None, **kwargs: Any
) -> BaseChatModel:
    """Construct a `ChatXAI` model wired to the current OAuth access token.

    Args:
        model_name: xAI (Grok) model identifier (e.g., `grok-4`).
        storage: Override the token storage. Defaults to `default_storage()`
            so the model reads the same file that `get_status` /
            `run_device_login` write.
        **kwargs: Extra constructor kwargs forwarded to `ChatXAI`.

    Returns:
        A configured `ChatXAI` instance, narrowed to `BaseChatModel` so
            `create_model` can splice it into the standard return path. Its
            underlying HTTP clients fetch a fresh token from `storage` on
            every request (see `_wire_refreshing_clients`) rather than
            reusing the token captured here at construction time, so a
            single instance stays usable for the lifetime of a long-running
            session.

    Raises:
        FileNotFoundError: No token has been stored yet. Surfaces as a
            `MissingCredentialsError` upstream in `create_model`.
        XaiOAuthReauthRequiredError: A token exists but could not be
            refreshed and needs a fresh sign-in. Also surfaces as
            `MissingCredentialsError` upstream.
        XaiOAuthTierDeniedError: The account lacks the SuperGrok / X
            Premium+ entitlement this OAuth surface requires. Surfaces as a
            distinct `MissingCredentialsError` pointing at `XAI_API_KEY`.
    """  # noqa: DOC502  # all three are raised by `_ensure_valid_access_token`, not literally in this body
    from langchain_xai import ChatXAI  # ty: ignore[unresolved-import]
    from pydantic import SecretStr

    store = storage or default_storage()
    # A valid token is still required up front so a missing/dead session
    # surfaces as `FileNotFoundError` / `XaiOAuthReauthRequiredError` /
    # `XaiOAuthTierDeniedError` here (the same up-front-failure contract
    # `create_model` relies on), and so `ChatXAI`'s own credential check
    # (which rejects a `None`/empty `api_key`) has something to validate.
    # It is otherwise unused for authentication once `_wire_refreshing_clients`
    # replaces the clients below.
    access_token = _ensure_valid_access_token(store)
    model = ChatXAI(model=model_name, api_key=SecretStr(access_token), **kwargs)
    _wire_refreshing_clients(model, store)
    return model


def _wire_refreshing_clients(model: ChatXAI, storage: FileTokenStorage) -> None:
    """Rebuild `model`'s OpenAI clients to fetch a fresh token on every request.

    `ChatXAI.validate_environment` (a `langchain_xai` internal, not this
    module's code) bakes the access token passed to its constructor into a
    *static* `api_key` string on the `openai.OpenAI` / `openai.AsyncOpenAI`
    clients it builds. That defeats this module's proactive refresh
    (`_ensure_valid_access_token`, `_needs_refresh`): in a long-running
    session, the token captured at construction time would eventually
    expire and every subsequent request would fail with an auth error, even
    though the token stored on disk has since been refreshed.

    `openai.OpenAI`/`AsyncOpenAI` accept `api_key` as a callable (sync for
    `OpenAI`, a coroutine function for `AsyncOpenAI`) that is invoked fresh
    immediately before every request (`_refresh_api_key` in
    `openai._client`), rather than only once at client-construction time.
    This replaces `model`'s four client attributes (`client`, `async_client`,
    `root_client`, `root_async_client` — `ChatXAI` builds these as two
    independent client pairs, not derived from one another) with clients
    built the same way `ChatXAI.validate_environment` builds them, except
    with that callable wired to `_ensure_valid_access_token(storage)` in
    place of the static token string. Everything else about request
    construction (base URL, timeout, headers, retries, the caller-supplied
    `http_client`/`http_async_client`) is read back off the already-built
    `model` so it stays exactly what `ChatXAI` resolved from its own
    defaults and the caller's kwargs.

    Args:
        model: A freshly constructed `ChatXAI` instance (already validated,
            with its default clients in place).
        storage: The token storage `_ensure_valid_access_token` reads from
            and refreshes into on each call.
    """
    import asyncio

    import openai

    def _sync_token_provider() -> str:
        return _ensure_valid_access_token(storage)

    async def _async_token_provider() -> str:
        return await asyncio.to_thread(_ensure_valid_access_token, storage)

    client_params: dict[str, Any] = {
        "base_url": model.xai_api_base,
        "timeout": model.request_timeout,
        "default_headers": model.default_headers,
        "default_query": model.default_query,
    }
    if model.max_retries is not None:
        client_params["max_retries"] = model.max_retries

    model.client = openai.OpenAI(
        api_key=_sync_token_provider, http_client=model.http_client, **client_params
    ).chat.completions
    model.root_client = openai.OpenAI(
        api_key=_sync_token_provider, http_client=model.http_client, **client_params
    )
    model.async_client = openai.AsyncOpenAI(
        api_key=_async_token_provider,
        http_client=model.http_async_client,
        **client_params,
    ).chat.completions
    model.root_async_client = openai.AsyncOpenAI(
        api_key=_async_token_provider,
        http_client=model.http_async_client,
        **client_params,
    )


__all__ = [
    "XAI_OAUTH_SERVER_NAME",
    "XaiLoginCancelledError",
    "XaiLoginInteraction",
    "XaiOAuthError",
    "XaiOAuthProtocolError",
    "XaiOAuthReauthRequiredError",
    "XaiOAuthStatus",
    "XaiOAuthTierDeniedError",
    "build_chat_model",
    "default_storage",
    "get_status",
    "is_logged_in",
    "logout",
    "run_device_login",
]
