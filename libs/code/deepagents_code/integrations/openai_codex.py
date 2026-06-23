"""ChatGPT OAuth integration for the `openai_codex` model provider.

Thin orchestration layer over `langchain_openai.chatgpt_oauth`. Reuses the
upstream PKCE/token primitives directly (`_generate_pkce_pair`,
`_build_authorize_url`, `_CallbackHandler`, `_post_form`,
`_token_from_response`, `_FileChatGPTOAuthTokenProvider`) so this module only
adds:

- a UI-friendly entry point that surfaces the authorize URL to the caller
    *before* the callback server starts blocking (the upstream `login_chatgpt`
    uses `print()`, which is invisible inside a Textual app),
- a cancel-aware reimplementation of the loopback callback wait so a
    cancelled sign-in releases the bound port promptly instead of holding it
    until the upstream wall-clock timeout elapses (see
    `_wait_for_oauth_callback`), and
- helpers for `/auth` to read sign-in status, expiry, and the linked account
    without re-implementing token parsing.

The browser-loopback flow mirrors the MCP one in `mcp_auth` from the user's
POV — PKCE + `state` CSRF + browser launch with a manual-URL fallback — but
the callback server here is upstream's single-threaded polling
`http.server.HTTPServer` (reused via `_CallbackHandler`), not the
`ThreadingHTTPServer` that `mcp_auth` runs itself. The OAuth-specific parts
(token exchange, refresh, file storage with 0600 perms) are delegated to
upstream.

!!! note

    The OAuth primitives are reused from `langchain-openai>=1.3.1`, which is
    the first release to ship `langchain_openai.chatgpt_oauth`. This module
    deliberately reaches into upstream-internal (`_`-prefixed) helpers; their
    stability is an upstream concern, so pin the minimum `langchain-openai`
    version when bumping and re-check the imported names against the release.
"""

from __future__ import annotations

import logging
import secrets
import webbrowser
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import threading
    from datetime import datetime
    from pathlib import Path

    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CodexAuthStatus:
    """Snapshot of the ChatGPT OAuth login state.

    Attributes:
        logged_in: Whether a usable token bundle exists on disk.
        store_path: Path to the token store (present whether or not the
            token exists, so the UI can show "no token yet at <path>").
        account_id: ChatGPT account ID parsed from the ID token, when
            available.
        plan_type: ChatGPT plan tier (e.g. `plus`, `pro`), when available.
        expires_at: Token expiry (UTC). `None` if no token is stored.
        is_expired: Whether the stored token is past expiry *or within the
            refresh skew window of it* (upstream's `_ChatGPTToken.is_expired`
            uses a 5-minute skew). Computed once at snapshot time, so it does
            not track the clock — re-fetch for a live check. Cheap users of
            this struct (e.g. switcher labels) often only need `logged_in`;
            this field lets the manager surface a "token expired" warning
            explicitly.
        unreadable_reason: Set when the token file exists but cannot be
            parsed. Surfaces corruption to the UI without crashing
            credential listing.
    """

    logged_in: bool
    store_path: Path
    account_id: str | None = None
    plan_type: str | None = None
    expires_at: datetime | None = None
    is_expired: bool = False
    unreadable_reason: str | None = None

    def __post_init__(self) -> None:
        """Reject incoherent snapshots the factory should never build.

        The valid/expired/missing/unreadable states are encoded as a
        `bool` plus optionals rather than a tagged union, so this guards the
        cross-field rules the attribute docs promise — catching future
        construction drift the same way upstream's `_ChatGPTToken` guards
        its own invariants.

        Raises:
            ValueError: An unreadable token is also marked `logged_in`; a
                logged-out snapshot carries an `expires_at`, `is_expired`,
                `account_id`, or `plan_type`; or a logged-in snapshot is
                missing its `expires_at`.
        """
        if self.unreadable_reason is not None and self.logged_in:
            msg = (
                "`unreadable_reason` implies the token is not usable; "
                "`logged_in` must be False."
            )
            raise ValueError(msg)
        if self.logged_in and self.expires_at is None:
            msg = "`expires_at` must be set when `logged_in` is True."
            raise ValueError(msg)
        if not self.logged_in and self.expires_at is not None:
            msg = "`expires_at` is only meaningful when `logged_in` is True."
            raise ValueError(msg)
        if not self.logged_in and self.is_expired:
            msg = "`is_expired` is only meaningful when `logged_in` is True."
            raise ValueError(msg)
        if not self.logged_in and (self.account_id or self.plan_type):
            msg = (
                "`account_id`/`plan_type` are only meaningful when `logged_in` is True."
            )
            raise ValueError(msg)


def default_store_path() -> Path:
    """Return the ChatGPT OAuth token store path.

    Stored under Deep Agents' own state dir
    (`~/.deepagents/.state/chatgpt-auth.json`) so the credential lives
    alongside the rest of the app's state rather than in `langchain_openai`'s
    default `~/.langchain`.
    """
    from deepagents_code.model_config import DEFAULT_STATE_DIR

    return DEFAULT_STATE_DIR / "chatgpt-auth.json"


def get_status(*, store_path: Path | None = None) -> CodexAuthStatus:
    """Return the current ChatGPT OAuth sign-in state.

    Reads the on-disk token *without* triggering a refresh (a passive
    inspect, suitable for switcher labels and the `/auth` manager). If a
    refresh is needed for actual usage, callers should construct a
    `_FileChatGPTOAuthTokenProvider` and call `get_token()` instead.

    Args:
        store_path: Override the token store path. Defaults to
            `default_store_path()` (`~/.deepagents/.state/chatgpt-auth.json`).

    Returns:
        A `CodexAuthStatus` populated from the on-disk token, or one with
            `logged_in=False` when no token exists or the file is unreadable.
    """
    path = store_path or default_store_path()
    try:
        from langchain_openai.chatgpt_oauth import (
            _FileChatGPTOAuthTokenProvider,  # noqa: PLC2701
        )
    except ImportError as exc:
        # Defensive: `get_status` sits on the `/auth` manager and `/model`
        # switcher hot paths. A future `langchain-openai` that renamed these
        # upstream-internal symbols would otherwise crash credential listing;
        # degrade to an unreadable status so the UI still renders.
        return CodexAuthStatus(
            logged_in=False,
            store_path=path,
            unreadable_reason=f"langchain-openai ChatGPT OAuth API unavailable: {exc}",
        )

    provider = _FileChatGPTOAuthTokenProvider(path=path)
    try:
        # `_read_from_disk` is a passive inspect; the public `get_token`
        # would refresh on expiry and hit the network, which is wrong for
        # the switcher/`/auth` listing paths that call this on the hot
        # path. Project policy allows SLF001 access.
        token = provider._read_from_disk()
    except RuntimeError as exc:
        return CodexAuthStatus(
            logged_in=False,
            store_path=path,
            unreadable_reason=str(exc),
        )
    if token is None:
        return CodexAuthStatus(logged_in=False, store_path=path)
    return CodexAuthStatus(
        logged_in=True,
        store_path=path,
        account_id=token.account_id,
        plan_type=token.plan_type,
        expires_at=token.expires_at,
        is_expired=token.is_expired(),
    )


def is_logged_in(*, store_path: Path | None = None) -> bool:
    """Return whether a ChatGPT OAuth token is stored on disk."""
    return get_status(store_path=store_path).logged_in


def logout(*, store_path: Path | None = None) -> bool:
    """Delete the stored ChatGPT OAuth token.

    Args:
        store_path: Override the token store path.

    Returns:
        `True` if a token file was removed, `False` if no file existed.
    """
    path = store_path or default_store_path()
    if not path.exists():
        return False
    path.unlink()
    return True


class CodexLoginCancelledError(RuntimeError):
    """Raised when the user cancels a sign-in flow mid-callback wait."""


class CodexAuthExpiredError(RuntimeError):
    """Raised when a stored ChatGPT token cannot be refreshed.

    Distinct from a missing token (`FileNotFoundError`): a token exists on
    disk but its refresh token was revoked or expired, so an automatic
    refresh failed and the user must sign in again. `create_model` maps this
    to a `MissingCredentialsError` that points at `/auth`.
    """


class CodexLoginInteraction:
    """UI hooks for the browser loopback sign-in flow.

    Implementations decide how to surface the authorize URL and whether to
    auto-open a browser. The Textual screen subclasses this; CLI / headless
    callers can supply a minimal stdout-based implementation.

    The default base class implements the print-to-stdout fallback so it
    works for headless tests and the `-x` non-interactive path; UI callers
    override `show_authorize_url`.
    """

    async def show_authorize_url(  # noqa: PLR6301  # override hook; `self` is meaningful in subclasses
        self,
        url: str,
        *,
        opened_in_browser: bool,
    ) -> None:
        """Surface the authorize URL to the user.

        Called once, immediately after the URL is built and before the
        callback wait begins. `opened_in_browser` is whether we already
        invoked `webbrowser.open` successfully — false means the user has
        to copy the URL manually.
        """
        prefix = (
            "Browser opened to: "
            if opened_in_browser
            else "Open this URL in a browser: "
        )
        print(f"\n{prefix}{url}\n")  # noqa: T201


_LOOPBACK_TIMEOUT_SECONDS = 300.0
"""Total seconds to wait for the browser callback before giving up.

Matches the upstream `login_chatgpt(timeout=300.0)` default so behavior is
identical whether a user signs in via the TUI or the bare upstream helper.
"""


async def run_browser_login(
    interaction: CodexLoginInteraction | None = None,
    *,
    store_path: Path | None = None,
    open_browser: bool = True,
    cancel_event: threading.Event | None = None,
) -> CodexAuthStatus:
    """Run the ChatGPT OAuth Authorization Code Flow with PKCE.

    Reimplements the upstream `langchain_openai.chatgpt_oauth` browser
    sign-in flow over its lower-level helpers, but routes the authorize-URL
    display through `interaction` so a Textual screen can render it inline.
    The blocking callback wait and the synchronous token exchange both run
    inside `asyncio.to_thread` so the calling event loop stays responsive.

    Args:
        interaction: UI hooks for surfacing the URL / notices. A default
            stdout-based implementation is used when `None`.
        store_path: Override the token store path. Defaults to
            `default_store_path()` (`~/.deepagents/.state/chatgpt-auth.json`).
        open_browser: Whether to call `webbrowser.open`. Disable in
            headless environments or tests.
        cancel_event: Optional event the caller can set to abandon the
            wait. Polled between callback server requests by
            `_wait_for_oauth_callback`, so setting it stops the wait and
            closes the loopback server within one poll interval (~1s),
            freeing the port for an immediate retry.

    Returns:
        A fresh `CodexAuthStatus` reflecting the just-saved token.

    Raises:
        CodexLoginCancelledError: `cancel_event` was set before the callback
            arrived.
        RuntimeError: OAuth state mismatch, missing code, port bind failure,
            or upstream token-endpoint error.

    !!! note

        `_wait_for_oauth_callback` raises `TimeoutError` after 300s of
        inactivity; that exception propagates unchanged.
    """
    import asyncio

    # We deliberately reach into the underscored building blocks of
    # `chatgpt_oauth` rather than calling `login_chatgpt`. The top-level
    # helper uses `print()` to surface the authorize URL — invisible inside
    # a Textual app — and bundles the browser open into one blocking call,
    # so we cannot show the URL ahead of the wait. These private helpers are
    # the same primitives upstream's own `login_chatgpt` composes; their
    # stability is an upstream concern (see the module docstring), so re-check
    # these names whenever the `langchain-openai` pin is bumped.
    from langchain_openai.chatgpt_oauth import (
        CHATGPT_AUTHORIZE_URL,
        CHATGPT_CLIENT_ID,
        CHATGPT_TOKEN_URL,
        DEFAULT_REDIRECT_HOST,
        DEFAULT_REDIRECT_PATH,
        DEFAULT_REDIRECT_PORT,
        _build_authorize_url,  # noqa: PLC2701
        _FileChatGPTOAuthTokenProvider,  # noqa: PLC2701
        _generate_pkce_pair,  # noqa: PLC2701
        _post_form,  # noqa: PLC2701
        _token_from_response,  # noqa: PLC2701
    )

    ui = interaction if interaction is not None else CodexLoginInteraction()
    redirect_uri = (
        f"http://{DEFAULT_REDIRECT_HOST}:{DEFAULT_REDIRECT_PORT}{DEFAULT_REDIRECT_PATH}"
    )
    state = secrets.token_urlsafe(32)
    verifier, challenge = _generate_pkce_pair()
    authorize_url = _build_authorize_url(
        client_id=CHATGPT_CLIENT_ID,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=challenge,
    )
    logger.info("Starting ChatGPT OAuth sign-in flow at %s", CHATGPT_AUTHORIZE_URL)

    opened = False
    if open_browser:
        try:
            opened = await asyncio.to_thread(webbrowser.open, authorize_url)
        except (webbrowser.Error, OSError) as exc:
            # `webbrowser.open` can raise `OSError` (not just `webbrowser.Error`)
            # when a configured launcher's binary is missing. Fall through to
            # the manual-URL display instead of aborting sign-in — the
            # headless / SSH case is exactly where this fallback matters.
            logger.warning("Could not launch a browser for ChatGPT sign-in: %s", exc)
            opened = False
    await ui.show_authorize_url(authorize_url, opened_in_browser=opened)

    callback_result = await asyncio.to_thread(
        _wait_for_oauth_callback,
        host=DEFAULT_REDIRECT_HOST,
        port=DEFAULT_REDIRECT_PORT,
        callback_path=DEFAULT_REDIRECT_PATH,
        timeout=_LOOPBACK_TIMEOUT_SECONDS,
        cancel_event=cancel_event,
    )

    if cancel_event is not None and cancel_event.is_set():
        # Defense in depth: `_wait_for_oauth_callback` already raises on
        # cancel, but a callback landing in the same poll as the cancel
        # returns the result instead — honor the explicit cancel here so the
        # token exchange below never runs and no token is saved.
        msg = "Sign-in was cancelled."
        raise CodexLoginCancelledError(msg)

    if callback_result.get("state") != state:
        msg = "ChatGPT OAuth callback state mismatch."
        raise RuntimeError(msg)
    if "error" in callback_result:
        description = callback_result.get("error_description", "")
        msg = (
            f"ChatGPT OAuth callback returned error: "
            f"{callback_result['error']} {description}".rstrip()
        )
        raise RuntimeError(msg)
    code = callback_result.get("code")
    if not code:
        msg = "ChatGPT OAuth callback did not include an authorization code."
        raise RuntimeError(msg)

    response = await asyncio.to_thread(
        _post_form,
        CHATGPT_TOKEN_URL,
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": CHATGPT_CLIENT_ID,
            "code_verifier": verifier,
        },
    )
    if cancel_event is not None and cancel_event.is_set():
        # The user cancelled while the token exchange was in flight. The
        # exchange already completed in-memory, but honor the cancel so no
        # token lands on disk for a sign-in the user abandoned — the same
        # "no token saved on cancel" guarantee the post-callback check makes.
        msg = "Sign-in was cancelled."
        raise CodexLoginCancelledError(msg)

    token = _token_from_response(response)
    path = store_path or default_store_path()
    file_provider = _FileChatGPTOAuthTokenProvider(path=path)
    file_provider.save(token)
    return get_status(store_path=path)


def _wait_for_oauth_callback(
    *,
    host: str,
    port: int,
    callback_path: str,
    timeout: float,
    cancel_event: threading.Event | None = None,
    poll_interval: float = 1.0,
) -> dict[str, str]:
    """Block on the loopback OAuth callback, polling `cancel_event`.

    A cancel-aware reimplementation of upstream's `_wait_for_callback`,
    reusing its `_CallbackHandler` so the request parsing stays identical.
    Upstream's version loops on a wall-clock deadline only, so a cancelled
    sign-in leaves the worker thread holding the bound port until a real
    callback lands or the full `timeout` elapses (up to 5 minutes). This
    variant checks `cancel_event` between single-request polls and always
    closes the server in a `finally`, so a cancel frees the port within one
    `poll_interval` and an immediate retry can rebind.

    Args:
        host: Loopback callback host (e.g. `localhost`).
        port: Loopback callback port.
        callback_path: Path the OAuth provider redirects back to.
        timeout: Total seconds to wait before raising `TimeoutError`.
        cancel_event: When set, abandon the wait at the next poll boundary.
        poll_interval: Seconds each `handle_request` blocks before the loop
            re-checks `cancel_event` and the deadline.

    Returns:
        The parsed callback query (`code` / `state`, or `error` /
            `error_description`).

    Raises:
        CodexLoginCancelledError: `cancel_event` was set before a callback
            arrived.
        RuntimeError: The callback server could not bind to the port.
        TimeoutError: No callback arrived within `timeout` seconds.
    """
    import http.server
    import time

    from langchain_openai.chatgpt_oauth import (
        _CallbackHandler,  # noqa: PLC2701
    )

    # A per-call subclass so the class-level `server_result` dict never leaks
    # state across sign-in attempts (upstream's `_wait_for_callback` does the
    # same).
    class _BoundCallbackHandler(_CallbackHandler):
        server_result: dict[str, str] = {}  # noqa: RUF012  # mirrors upstream handler shape

    _BoundCallbackHandler.callback_path = callback_path
    try:
        server = http.server.HTTPServer((host, port), _BoundCallbackHandler)
    except OSError as exc:
        msg = (
            f"Could not bind ChatGPT OAuth callback server on "
            f"http://{host}:{port}: {exc}. Free the port and try again."
        )
        raise RuntimeError(msg) from exc
    server.timeout = poll_interval
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                msg = "Sign-in was cancelled."
                raise CodexLoginCancelledError(msg)
            # Blocks up to `poll_interval`; returns promptly once a request
            # lands so the loop can re-check the cancel flag and deadline.
            server.handle_request()
            result = _BoundCallbackHandler.server_result
            if result.get("code") or result.get("error"):
                return dict(result)
    finally:
        server.server_close()
    msg = f"Timed out waiting for ChatGPT OAuth callback on http://{host}:{port}"
    raise TimeoutError(msg)


def build_chat_model(
    model_name: str, /, *, store_path: Path | None = None, **kwargs: Any
) -> BaseChatModel:
    """Construct a `_ChatOpenAICodex` model wired to the on-disk token store.

    Args:
        model_name: Codex model identifier (e.g., `gpt-5.2-codex`).
        store_path: Override the token store path. Defaults to
            `default_store_path()` so the model reads the same file that
            `get_status` / `run_browser_login` write.
        **kwargs: Extra constructor kwargs forwarded to `_ChatOpenAICodex`.

    Returns:
        A configured `_ChatOpenAICodex` instance, narrowed to `BaseChatModel`
            so `create_model` can splice it into the standard return path.

    Raises:
        FileNotFoundError: If no token has been stored yet. Surfaces as a
            `MissingCredentialsError` upstream in `create_model`.
        CodexAuthExpiredError: If a token exists but its refresh token was
            revoked/expired so the eager refresh failed. Also surfaces as a
            `MissingCredentialsError` upstream, pointing the user at `/auth`.
    """  # noqa: DOC502  # `FileNotFoundError` is raised by `provider.get_token()` (`_load_existing`) when the on-disk token is missing
    from langchain_openai.chat_models.codex import (
        _ChatOpenAICodex,  # noqa: PLC2701
    )
    from langchain_openai.chatgpt_oauth import (
        _ChatGPTOAuthRefreshError,  # noqa: PLC2701
        _FileChatGPTOAuthTokenProvider,  # noqa: PLC2701
    )

    provider = _FileChatGPTOAuthTokenProvider(path=store_path or default_store_path())
    # Touch the provider's read path eagerly so a missing token surfaces as
    # `FileNotFoundError` here instead of on first invocation — the app's
    # `create_model` path expects credential failures up front. `get_token`
    # refreshes if the stored token is past `refresh_skew`, so the model
    # is guaranteed to receive a valid bearer at construction time.
    try:
        provider.get_token()
    except _ChatGPTOAuthRefreshError as exc:
        # A token exists but the refresh token is dead. This is a credentials
        # problem with a clear remedy (sign in again), not an unexpected
        # construction failure — translate it so `create_model` can route it
        # to the same `/auth` recovery path as a missing token rather than a
        # generic "failed to initialize" error.
        msg = "ChatGPT session expired and could not be refreshed automatically."
        raise CodexAuthExpiredError(msg) from exc
    return _ChatOpenAICodex(
        model=model_name,
        token_provider=provider,
        **kwargs,
    )
