"""ChatGPT OAuth integration for the `openai_codex` model provider.

Thin orchestration layer over `langchain_openai.chatgpt_oauth`. Reuses the
upstream PKCE/loopback primitives directly (`_generate_pkce_pair`,
`_build_authorize_url`, `_wait_for_callback`, `_post_form`,
`_token_from_response`, `FileChatGPTOAuthTokenProvider`) so this module only
adds:

- a UI-friendly entry point that surfaces the authorize URL to the caller
  *before* the callback server starts blocking (the upstream `login_chatgpt`
  uses `print()`, which is invisible inside a Textual app), and
- helpers for `/auth` to read sign-in status, expiry, and the linked account
  without re-implementing token parsing.

The browser-loopback flow mirrors the MCP one in `mcp_auth`
(PKCE + state CSRF + ThreadingHTTPServer + browser launch with manual-URL
fallback), with the OAuth-specific parts (token exchange, refresh, file
storage with 0600 perms) delegated to upstream.

!!! warning
    This module pins `langchain-openai` to an unreleased git ref (PR
    <https://github.com/langchain-ai/langchain/pull/37569>) while the
    upstream module is under review. Once that PR lands and is released,
    drop the `[tool.uv.sources]` override in `pyproject.toml`; no code in
    this module needs to change.
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
        is_expired: Whether the stored token is past expiry. Cheap users
            of this struct (e.g. switcher labels) often only need
            `logged_in`; this field lets the manager surface a "token
            expired" warning explicitly.
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


def default_store_path() -> Path:
    """Return the default ChatGPT OAuth token store path.

    Defers the upstream import so callers on the startup hot path don't
    pull in `langchain_openai` / `httpx` just to learn the path string.
    """
    from langchain_openai.chatgpt_oauth import DEFAULT_STORE_PATH

    return DEFAULT_STORE_PATH


def get_status(*, store_path: Path | None = None) -> CodexAuthStatus:
    """Return the current ChatGPT OAuth sign-in state.

    Reads the on-disk token *without* triggering a refresh (a passive
    inspect, suitable for switcher labels and the `/auth` manager). If a
    refresh is needed for actual usage, callers should construct a
    `FileChatGPTOAuthTokenProvider` and call `get_token()` instead.

    Args:
        store_path: Override the token store path. Defaults to the
            upstream default (`~/.langchain/chatgpt-auth.json`).

    Returns:
        A `CodexAuthStatus` populated from the on-disk token, or one with
            `logged_in=False` when no token exists or the file is unreadable.
    """
    from langchain_openai.chatgpt_oauth import (
        _FileChatGPTOAuthTokenProvider,  # noqa: PLC2701
    )

    path = store_path or default_store_path()
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

    async def notice(self, message: str) -> None:  # noqa: PLR6301  # override hook
        """Surface a one-line informational notice (e.g. fallback hints)."""
        print(f"\n{message}\n")  # noqa: T201


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
        store_path: Override the token store path. Defaults to upstream's
            `~/.langchain/chatgpt-auth.json`.
        open_browser: Whether to call `webbrowser.open`. Disable in
            headless environments or tests.
        cancel_event: Optional event the caller can set to abandon the
            wait. Not yet plumbed into the upstream callback server (which
            polls every 1s), so cancellation is best-effort and only takes
            effect on the next poll.

    Returns:
        A fresh `CodexAuthStatus` reflecting the just-saved token.

    Raises:
        CodexLoginCancelledError: The caller set `cancel_event` before the
            callback arrived.
        RuntimeError: OAuth state mismatch, missing code, port bind failure,
            or upstream token-endpoint error.

    !!! note

        Upstream's `_wait_for_callback` raises `TimeoutError` after 300s of
        inactivity; that exception propagates unchanged.
    """
    import asyncio

    # We deliberately reach into the underscored building blocks of
    # `chatgpt_oauth` rather than calling `login_chatgpt`. The top-level
    # helper uses `print()` to surface the authorize URL — invisible inside
    # a Textual app — and bundles the browser open into one blocking call,
    # so we cannot show the URL ahead of the wait. These private helpers
    # are the same primitives upstream's own `login_chatgpt` composes,
    # documented in PR 37569 as the supported reuse path for downstream
    # frameworks that need a custom UI surface. PR 37569 commits to keeping
    # them stable as an internal API; when it lands and is released we can
    # revisit whether a public helper has appeared.
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
        _wait_for_callback,  # noqa: PLC2701
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
        except webbrowser.Error as exc:
            logger.warning("Could not launch a browser for ChatGPT sign-in: %s", exc)
            opened = False
    await ui.show_authorize_url(authorize_url, opened_in_browser=opened)

    callback_result = await asyncio.to_thread(
        _wait_for_callback,
        host=DEFAULT_REDIRECT_HOST,
        port=DEFAULT_REDIRECT_PORT,
        callback_path=DEFAULT_REDIRECT_PATH,
        timeout=_LOOPBACK_TIMEOUT_SECONDS,
    )

    if cancel_event is not None and cancel_event.is_set():
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
    token = _token_from_response(response)
    path = store_path or default_store_path()
    file_provider = _FileChatGPTOAuthTokenProvider(path=path)
    file_provider.save(token)
    return get_status(store_path=path)


def build_chat_model(model_name: str, /, **kwargs: Any) -> BaseChatModel:
    """Construct a `_ChatOpenAICodex` model wired to the on-disk token store.

    Args:
        model_name: Codex model identifier (e.g., `gpt-5.2-codex`).
        **kwargs: Extra constructor kwargs forwarded to `_ChatOpenAICodex`.

    Returns:
        A configured `_ChatOpenAICodex` instance, narrowed to `BaseChatModel`
            so `create_model` can splice it into the standard return path.

    Raises:
        FileNotFoundError: If no token has been stored yet. Surfaces as a
            `MissingCredentialsError` upstream in `create_model`.
    """  # noqa: DOC502  # `FileNotFoundError` is raised by `provider.get_token()` (`_load_existing`) when the on-disk token is missing
    from langchain_openai.chat_models.codex import (
        _ChatOpenAICodex,  # noqa: PLC2701
    )
    from langchain_openai.chatgpt_oauth import (
        _FileChatGPTOAuthTokenProvider,  # noqa: PLC2701
    )

    provider = _FileChatGPTOAuthTokenProvider()
    # Touch the provider's read path eagerly so a missing token surfaces as
    # `FileNotFoundError` here instead of on first invocation — the app's
    # `create_model` path expects credential failures up front. `get_token`
    # refreshes if the stored token is past `refresh_skew`, so the model
    # is guaranteed to receive a valid bearer at construction time.
    provider.get_token()
    return _ChatOpenAICodex(
        model=model_name,
        token_provider=provider,
        **kwargs,
    )
